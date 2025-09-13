"""Exposed PatchTST model, taken from HuggingFace transformers"""

try:
    from flash_attn import flash_attn_func
    _flash_attn_available = True
    print("Flash Attention v2 is available and will be used.")
except ImportError:
    _flash_attn_available = False
    print("Flash Attention v2 is not available, falling back to standard attention.")

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from transformers import PatchTSTConfig, PatchTSTPreTrainedModel
from transformers.models.patchtst.modeling_patchtst import (
    ACT2CLS,
    BaseModelOutput,
    NegativeBinomialOutput,
    NormalOutput,
    PatchTSTForPredictionOutput,
    PatchTSTForPretrainingOutput,
    PatchTSTMasking,
    PatchTSTModelOutput,
    PatchTSTScaler,
    SamplePatchTSTOutput,
    StudentTOutput,
    nll,
    weighted_average,
)
from transformers.utils import ModelOutput

from .modules import (
    DyT,
    PatchTSTKernelEmbedding,
    PatchTSTPatchify,
    PatchTSTRMSNorm,
    apply_p_rope_to_qk,
)

import kymatio.torch as kymatio

class TCNResidualBlock(nn.Module):
    """
    时间卷积网络(TCN)的核心残差模块
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        
        # 1. 定义层 (保持不变)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.gelu2 = nn.GELU()
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        # 2. 【关键修正】: 删除下面两行代码和整个 _init_weights 方法
        # self._init_weights()  <-- 删除这一行

    # 【关键修正】: 删除整个 _init_weights 方法
    # def _init_weights(self):
    #     ...

    def forward(self, x):
        residual = self.residual_conv(x) if self.residual_conv is not None else x
        out = self.conv1(x)
        out = out[..., :-self.padding].contiguous()
        out = self.gelu1(out)
        out = self.conv2(out)
        out = out[..., :-self.padding].contiguous()
        out = self.gelu2(out)
        return out + residual
# =================================================================================
#  【围追堵截方案 - 步骤1】: 新增一个完全自定义、透明可控的Transformer层
# =================================================================================
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation=F.gelu, batch_first=True):
        super().__init__()
        # 确保 d_model 能被 nhead 整除
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        # 我们将使用你代码中已有的、经过验证的 PatchTSTRopeAttention
        # 注意：这里我们禁用 RoPE，因为变量之间没有序列位置关系
        self.self_attn = PatchTSTRopeAttention(
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout,
            use_rope=False,  # 关键：变量维度没有位置顺序，禁用RoPE
        )
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        # 确保 batch_first 语义被遵守 (虽然我们的 Attention 层默认就是)
        # 这个参数主要是为了接口兼容性
        self.batch_first = batch_first

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pytorch 的 TransformerEncoderLayer 接受的 mask 参数我们在这里用不上，但保留接口一致性
        x = src
        
        # --- Self-Attention Block ---
        # Pre-Normalization (更稳定)
        x_norm = self.norm1(x)
        # 我们的 Attention 层返回 (attn_output, attn_weights_reshaped, past_key_value)
        attn_output, _, _ = self.self_attn(x_norm)
        x = x + self.dropout1(attn_output)

        # --- Feedforward Block ---
        # Pre-Normalization
        x_norm = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(ff_output)
        
        return x

class GeometryVAE(nn.Module):
    def __init__(self, d_model, tcn_channels, num_tcn_layers, latent_dim,
                 transformer_nhead=4):
        super().__init__()
        self.d_model = d_model
        
        # --- 模块定义 ---
        tcn_layers = []
        for i in range(num_tcn_layers):
            dilation = 2**i
            in_channels = d_model if i == 0 else tcn_channels
            tcn_layers.append(
                TCNResidualBlock(in_channels, tcn_channels, kernel_size=7, dilation=dilation)
            )
        self.temporal_encoder_tcn = nn.Sequential(*tcn_layers)
        
        self.transformer_norm = nn.LayerNorm(tcn_channels, eps=1e-5)
        
        self.coupling_transformer = CustomTransformerEncoderLayer(
            d_model=tcn_channels, nhead=transformer_nhead, dim_feedforward=tcn_channels*2, 
            batch_first=True, activation=F.gelu
        )
        
        self.vae_head = nn.Linear(tcn_channels, latent_dim * 2)
        self.decoder_initial_mlp = nn.Linear(latent_dim, tcn_channels)
        
        self.decoder_gru = nn.GRU(input_size=tcn_channels, hidden_size=tcn_channels, num_layers=2, batch_first=True)
        self.decoder_output_layer = nn.Linear(tcn_channels, 1)

        # 【关键修复 - 步骤1】: 强制 GRU 模块使用 float32 进行计算
        # 这可以防止在混合精度(autocast)环境下出现数值不稳定的问题
        self.decoder_gru.to(torch.float32)

        self.apply(self._init_weights)
        print(">>> GeometryVAE: Applied clean and simple _init_weights at creation time.")

    # 在 GeometryVAE 类中...

    # 【回归简洁】: 一个干净、简单的初始化函数
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None:
                torch.nn.init.ones_(m.weight)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    # In GeometryVAE.encode
    def encode(self, x):
        # print_tensor_stats("VAE ENCODER INPUT (x_patched)", x)

        B, V, P, D = x.shape
        x_reshaped = x.reshape(B * V, P, D).permute(0, 2, 1)
        # print_tensor_stats("VAE reshaped", x_reshaped) # 检查1

        tcn_out = self.temporal_encoder_tcn(x_reshaped)
        # print_tensor_stats("VAE after TCN", tcn_out) # 检查2

        temporal_features = tcn_out.mean(dim=2)
        per_variable_features = temporal_features.view(B, V, -1)
        # print_tensor_stats("VAE per_variable_features", per_variable_features) # 检查3

        coupled_features = self.coupling_transformer(per_variable_features)
        # print_tensor_stats("VAE after Transformer", coupled_features) # 检查5

        global_features = coupled_features.mean(dim=1)
        # print_tensor_stats("VAE FEATURES BEFORE HEAD (global_features)", global_features)

        mu_logvar = self.vae_head(global_features)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=-1)

        # print_tensor_stats("VAE mu (from head)", mu)
        # print_tensor_stats("VAE logvar (from head, before clamp)", logvar)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 在 GeometryVAE 类中...

    def decode(self, z, V, T):
        # 导入 autocast
        from torch.cuda.amp import autocast

        # z: (B, latent_dim)
        B = z.shape[0]
        original_dtype = z.dtype # 记录输入的数据类型，以便最后恢复

        # 使用 autocast(enabled=False) 创建一个全精度计算的安全区
        with autocast(enabled=False):
            # 将输入转换为 float32
            z_fp32 = z.to(torch.float32)

            # 1. 初始状态生成
            initial_h = self.decoder_initial_mlp(z_fp32)
            initial_h = torch.tanh(initial_h)
            initial_h_expanded = initial_h.unsqueeze(0).repeat(self.decoder_gru.num_layers, 1, 1)
            h0 = initial_h_expanded.unsqueeze(2).repeat(1, 1, V, 1).view(self.decoder_gru.num_layers, B * V, -1)
            
            # 2. 自回归生成
            # 创建一个与 GRU 期望输入维度匹配的 float32 虚拟输入
            dummy_input = torch.zeros(B * V, T, self.decoder_initial_mlp.out_features, device=z.device, dtype=torch.float32)
            
            # 在这个安全区内，GRU 模块和它的输入都是 float32，计算是稳定的
            gru_out, _ = self.decoder_gru(dummy_input, h0)
            gru_out_stabilized = torch.tanh(gru_out) # 增加tanh稳定输出
            
            # 3. 最终输出映射
            output_flat = self.decoder_output_layer(gru_out_stabilized.reshape(-1, self.decoder_initial_mlp.out_features))
            reconstructed_trajectory = output_flat.view(B, V, T)

        # 离开安全区后，将输出转换回原始的 dtype，以匹配模型的其余部分
        return reconstructed_trajectory.to(original_dtype)

    def forward(self, x_patched, target_T=None):
        # x_patched: (B, V, P, D)
        # target_T: 目标重构的原始时间步长
        # print_tensor_stats("VAE FORWARD INPUT (x_patched)", x_patched)
        if target_T is None:
            # 如果不提供，可以做一个简单的假设，例如P*stride
            target_T = x_patched.shape[2] * 4 

        mu, logvar = self.encode(x_patched)
        # print_tensor_stats("VAE FORWARD mu (from encoder)", mu)
        # print_tensor_stats("VAE FORWARD logvar (from encoder)", logvar)
        
        z = self.reparameterize(mu, logvar)
        # print_tensor_stats("VAE FORWARD z (after reparameterize)", z)

        x_recon = self.decode(z, x_patched.shape[1], target_T)
        
        # decode 内部会打印最终的 x_recon，这里就不重复了
        return x_recon, mu, logvar, z

# LayerNorm 通常作用于最后一个维度，所以需要调整一下结构
# LayerNorm 通常作用于最后一个维度，所以需要调整一下结构
class CnnExtractorWithLayerNorm(nn.Module):
    def __init__(self, n_coeffs):
        super().__init__()
        self.conv1 = nn.Conv1d(n_coeffs, 64, kernel_size=7, padding=3)
        self.ln1 = nn.LayerNorm(64) 
        self.gelu1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.ln2 = nn.LayerNorm(128)
        self.gelu2 = nn.GELU()
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # print_tensor_stats("CnnExtractor Input", x) # <--- 新增
        
        # --- Block 1 ---
        x = self.conv1(x)
        # print_tensor_stats("After conv1", x) # <--- 新增
        
        x = x.permute(0, 2, 1) # (B, C, L) -> (B, L, C) for LayerNorm
        x = self.ln1(x)
        # print_tensor_stats("After ln1", x) # <--- 新增
        
        x = x.permute(0, 2, 1) # (B, L, C) -> (B, C, L) back to conv format
        x = self.gelu1(x)
        # print_tensor_stats("After gelu1", x) # <--- 新增
        
        # --- Block 2 ---
        x = self.conv2(x)
        # print_tensor_stats("After conv2", x) # <--- 新增
        
        x = x.permute(0, 2, 1) # (B, C, L) -> (B, L, C) for LayerNorm
        x = self.ln2(x)
        # print_tensor_stats("After ln2", x) # <--- 新增
        
        x = x.permute(0, 2, 1) # (B, L, C) -> (B, C, L) back to conv format
        x = self.gelu2(x)
        # print_tensor_stats("After gelu2", x) # <--- 新增
        
        # --- Final layers ---
        x = self.pool(x)
        # print_tensor_stats("After AdaptiveAvgPool1d", x) # <--- 新增
        
        x = self.flatten(x)
        # print_tensor_stats("CnnExtractor Output (after flatten)", x) # <--- 新增
        
        return x
class WaveletAnalyzer(nn.Module):
    def __init__(self, input_timesteps, feature_dim, J=8, Q=8):
        super().__init__()
        self.scattering = kymatio.Scattering1D(J=J, shape=(input_timesteps,), Q=Q)
        with torch.no_grad():
            dummy_input = torch.randn(1, input_timesteps)
            # Kymatio 的输出形状是 (B, C, L)，我们需要通道数 C
            n_coeffs = self.scattering(dummy_input).shape[1]
            
        self.cnn_extractor = CnnExtractorWithLayerNorm(n_coeffs)
        self.final_mlp = nn.Linear(128, feature_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print_tensor_stats("WaveletAnalyzer Input (x)", x) # <--- 新增
        B, V, T = x.shape
        
        x_reshaped = x.reshape(B * V, T)
        # print_tensor_stats("Reshaped input for Scattering", x_reshaped) # <--- 新增

        scattering_coeffs = self.scattering(x_reshaped.contiguous())
        # print_tensor_stats("After Scattering (Scattering Coeffs)", scattering_coeffs) # <--- 新增
        
        # cnn_extractor 内部会打印所有子层的输出
        cnn_features = self.cnn_extractor(scattering_coeffs)

        features = self.final_mlp(cnn_features)
        # print_tensor_stats("After final_mlp", features) # <--- 新增
        
        features_reshaped = features.view(B, V, -1)
        # print_tensor_stats("After reshaping back (view)", features_reshaped) # <--- 新增

        final_embedding = features_reshaped.mean(dim=1)
        # print_tensor_stats("After mean pooling (Before Stabilization)", final_embedding) # <--- 新增
        
        stabilized_embedding = torch.sign(final_embedding) * torch.log(torch.abs(final_embedding) + 1)
        # print_tensor_stats("Final Output (stabilized_embedding)", stabilized_embedding) # <--- 新增
        
        return stabilized_embedding
    
class Memory(nn.Module):
    """ Memory prompt """
    def __init__(self, num_memory, memory_dim):
        super().__init__()
        self.num_memory = num_memory
        self.memory_dim = memory_dim

        self.memMatrix = nn.Parameter(torch.empty(num_memory, memory_dim))  # M,C
        self.keyMatrix = nn.Parameter(torch.empty(num_memory, memory_dim))  # M,C
        self.x_proj = nn.Linear(memory_dim, memory_dim)

        self.initialize_weights()
        print("Initialized Memory (Prompt Network)")

    def initialize_weights(self):
        # 这里的初始化会在 from_pretrained 时被调用
        torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
        torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 让 Memory 内部的 nn.Linear 也使用 Kaiming 初始化
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_uniform_(m.weight) # <- 替换
            torch.nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu') # <- 使用
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # assert x.shape[-1] == self.memory_dim, "Dimension mismatch in Memory network"
        # print_tensor_stats("Memory Input (global_embedding)", x) # <--- 新增
        
        x_query = torch.tanh(self.x_proj(x))
        # print_tensor_stats("Memory Query (after tanh)", x_query) # <--- 新增
        
        att_logits = F.linear(input=x_query, weight=self.keyMatrix)
        # print_tensor_stats("Memory Attention Logits (before softmax)", att_logits) # <--- 新增
        
        att_weight = F.softmax(att_logits, dim=-1)
        # print_tensor_stats("Memory Attention Weights (after softmax)", att_weight) # <--- 新增
        
        out = F.linear(att_weight, self.memMatrix.permute(1, 0))
        # print_tensor_stats("Memory Output (final prompt)", out) # <--- 新增
        
        return out

# =================================================================================
#  新增模块: MMD Loss 计算函数
# =================================================================================

def rational_quadratic_kernel(x, y, sigma_list=[0.2, 0.5, 0.9, 1.3]):
    if x.dim() == 3:
        x = x.squeeze(1)  # [B, N]
    if y.dim() == 3:
        y = y.squeeze(1)  # [B, N]
    x = x.unsqueeze(1)  # [B, 1, N]
    y = y.unsqueeze(0)  # [1, B, N]
    
    squared_dist = torch.sum((x - y) ** 2, dim=-1)  # [B, B]
    
    # 将 sigma_list 转换为 tensor 并使用广播机制
    sigma = torch.tensor(sigma_list, device=x.device).view(-1, 1, 1)  # [len(sigma_list), 1, 1]
    sigma_squared = sigma ** 2  # [len(sigma_list), 1, 1]
    
    # 计算 kernel 值
    kernel_val = sigma_squared / (sigma_squared + squared_dist)  # [len(sigma_list), B, B]
    
    # 对所有的 sigma_list 求和
    kernel_val = kernel_val.sum(dim=0)  # [B, B]
    
    return kernel_val

def compute_mmd(x, y, mean_value, variance_value):
    if x.dim() == 3:
        x = x.squeeze(1)  # [B, N]
    if y.dim() == 3:
        y = y.squeeze(1)  # [B, N]
    if mean_value.dim() == 1:
        mean_value = mean_value.unsqueeze(0)  # [1, N]
    if variance_value.dim() == 1:
        variance_value = variance_value.unsqueeze(0)  # [1, N]
    
    # 使用广播机制进行归一化
    x = (x - mean_value) / torch.sqrt(variance_value + 1e-6)  # [B, N]
    y = (y - mean_value) / torch.sqrt(variance_value + 1e-6)  # [B, N]
    
    B = x.size(0)
    xx = rational_quadratic_kernel(x, x)
    yy = rational_quadratic_kernel(y, y)
    xy = rational_quadratic_kernel(x, y)
    
    # 按照MMD的 unbiased estimator U-statistic 计算
    # http://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf
    # 为了避免 B*(B-1) 可能导致的除零错误 (当B=1时)，使用更稳定的版本
    if B > 1:
        term1 = (xx.sum() - xx.diag().sum()) / (B * (B - 1))
        term2 = (yy.sum() - yy.diag().sum()) / (B * (B - 1))
        term3 = xy.sum() / (B * B)
    else:
        term1, term2, term3 = 0, 0, 0

    return (term1 + term2 - 2 * term3).clamp(min=0)

def conditional_mmd_multi_step(input_traj, true_traj, pred_traj, mean, variance, steps=None):
    """
    计算多步条件 MMD: 平均 D((S^t)_# mu*, (S_theta^t)_# mu*) for t in steps
    input_traj: 输入轨迹 [B, T, N] (在此函数中未使用)
    true_traj: 真实未来轨迹 [B, H, N]
    pred_traj: 模型预测轨迹 [B, H, N]
    mean: 均值 [N] or [1, N] or [B, N]
    variance: 方差 [N] or [1, N] or [B, N]
    steps: 使用的预测时间步列表，默认为所有步 [0, 1, ..., H-1]
    返回: 平均 MMD 值
    """
    H = pred_traj.shape[1]
    
    # 默认使用所有预测步
    if steps is None:
        steps = range(H)  # [0, 1, 2, ..., H-1]
    
    mmd_sum = 0.0
    for t in steps:
        true_evolved = true_traj[:, t, :]   # [B, N]
        model_evolved = pred_traj[:, t, :] # [B, N]
        mmd_sum += compute_mmd(true_evolved, model_evolved, mean, variance)
    
    return mmd_sum / len(steps) if len(steps) > 0 else 0.0

@dataclass
class CompletionsPatchTSTOutput(ModelOutput):
    completions: torch.FloatTensor
    patched_past_values: Optional[torch.FloatTensor] = None
    mask: Optional[torch.FloatTensor] = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None

# =================================================================================
#  新增模块: Expert (用于MoE)
# =================================================================================
class Expert(nn.Module):
    """
    一个简单的Expert模块，本质上就是一个前馈网络 (FFN)。
    """
    def __init__(self, d_model: int, ffn_dim: int, config: PatchTSTConfig):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=config.bias),
            ACT2CLS[config.activation_function](),
            nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, d_model, bias=config.bias),
        )

    def forward(self, x):
        return self.ff(x)

# =================================================================================
#  新增模块: NaiveMoE (核心MoE层)
# =================================================================================
class NaiveMoE(nn.Module):
    """
    一个简单的MoE层，包含门控网络和多个Experts。
    实现了Top-k路由和负载均衡损失。
    """
    def __init__(self, d_model: int, ffn_dim: int, num_experts: int, top_k: int, config: PatchTSTConfig):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 门控网络，为每个token决定experts的权重
        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        # Experts列表
        self.experts = nn.ModuleList([Expert(d_model, ffn_dim, config) for _ in range(self.num_experts)])

    def forward(self, x: torch.Tensor):
        # x: [..., d_model]
        original_shape = x.shape
        # 将输入reshape为 [num_tokens, d_model]
        x_reshaped = x.reshape(-1, self.d_model)
        num_tokens, _ = x_reshaped.shape

        # 1. 通过门控网络获取每个expert的logits
        gate_logits = self.gate(x_reshaped) # [num_tokens, num_experts]
        
        # 2. 计算负载均衡损失 (Load Balancing Loss)
        # 这是为了鼓励门控网络将负载均匀地分配给所有experts，防止少数experts过载而其他experts空闲
        router_probs = F.softmax(gate_logits, dim=-1)
        tokens_per_expert = router_probs.mean(dim=0) # 每个expert处理的token的平均比例
        # 计算每个expert被选中的概率的平方和，乘以expert数量，作为loss的一部分
        load_balance_loss = self.num_experts * torch.sum(tokens_per_expert * router_probs.mean(dim=0))

        # 3. 找到Top-k的experts并进行路由
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1) # [num_tokens, top_k]
        
        # 归一化top-k的权重
        top_k_weights = top_k_weights / torch.sum(top_k_weights, dim=-1, keepdim=True)
        
        # 4. 分发token到对应的expert并计算输出
        final_output = torch.zeros_like(x_reshaped)
        # one-hot编码，标记每个token被分配给了哪些experts
        flat_top_k_indices = top_k_indices.flatten()
        
        # 将token和权重组合起来，方便后续计算
        # 使用scatter_add_可以高效地将token路由到不同的expert
        # 这里为了代码清晰，我们使用一个循环，但在大规模训练中可以进一步优化
        # 创建一个路由掩码，(num_tokens, num_experts)
        routing_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1) # [num_tokens, num_experts]
        
        for i in range(self.num_experts):
            # 找到被分配给当前expert i 的token的索引
            token_indices_for_expert_i = torch.where(routing_mask[:, i] == 1)[0]
            
            if token_indices_for_expert_i.numel() > 0:
                # 获取这些token的输入
                inputs_for_expert_i = x_reshaped[token_indices_for_expert_i]
                
                # 计算expert的输出
                outputs_for_expert_i = self.experts[i](inputs_for_expert_i)
                
                # 获取这些token对应的权重
                # top_k_indices: [num_tokens, top_k]
                # top_k_weights: [num_tokens, top_k]
                # 找到当前expert在top_k列表中的位置，并获取相应权重
                weights_for_expert_i = (top_k_indices[token_indices_for_expert_i] == i).float() * top_k_weights[token_indices_for_expert_i]
                weights_for_expert_i = weights_for_expert_i.sum(dim=1, keepdim=True)

                # 将加权后的输出加到最终结果中
                final_output.index_add_(0, token_indices_for_expert_i, outputs_for_expert_i * weights_for_expert_i)

        # 恢复原始形状并返回
        return final_output.reshape(original_shape), load_balance_loss

# =================================================================================
#  新增模块: ConvNeXtBlock1D (用于跳跃连接)
# =================================================================================
class ConvNeXtBlock1D(nn.Module):
    """
    一维版本的ConvNeXt模块，用于处理U-Net中的跳跃连接。
    将scOT中的二维操作适配到PatchTST的一维序列数据上。
    """

    def __init__(self, dim, path_dropout=0.0, layer_scale_init_value=1e-6, norm_eps=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # 1D depthwise conv
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = nn.Dropout(path_dropout) if path_dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] (Batch, NumPatches, Dim)
        input = x
        x = x.permute(0, 2, 1)  # [B, D, N]
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # [B, N, D]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        
        x = input + self.drop_path(x)
        return x
    
# =================================================================================
#  新增模块: PatchMerging (用于Encoder下采样)
# =================================================================================
class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    将序列长度减半 (N -> N/2)，将特征维度加倍 (D -> 2*D)。
    """
    def __init__(self, dim: int, norm_eps=1e-6):
        super().__init__()
        self.dim = dim
        # 这里的reduction将2*dim映射到2*dim，scOT中是4*dim到2*dim，因为我们是一维合并
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(2 * dim, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        batch_size, num_patches, dim = x.shape
        
        # 确保序列长度是偶数，便于合并
        if num_patches % 2 != 0:
            # 补齐
            x = nn.functional.pad(x, (0, 0, 0, 1))
            num_patches += 1

        x = x.reshape(batch_size, num_patches // 2, 2, dim)
        x = x.flatten(2) # [B, N/2, 2*D]
        
        x = self.reduction(x)
        x = self.norm(x)

        return x

# =================================================================================
#  新增模块: PatchExpansion (用于Decoder上采样)
# =================================================================================
class PatchExpansion(nn.Module):
    """
    Patch Expansion Layer.
    将序列长度加倍 (N -> 2*N)，将特征维度减半 (D -> D/2)。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 这个线性层将维度从 D 扩展到 D，因为后续reshape会减半
        self.expand = nn.Linear(dim, dim, bias=False)
        # 注意：scOT中是C -> 2C，然后reshape后变成C/2。这里我们简化为 D -> D，reshape后D/2。
        # 如果要完全对齐scOT的逻辑，应该是 D -> 2D，然后reshape成(B, N*2, D)，再通过一个线性层 D -> D/2
        # 为了最小化修改，我们采取更直接的方式。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        batch_size, num_patches, dim = x.shape
        x = self.expand(x)
        x = x.view(batch_size, num_patches, 2, dim // 2)
        x = x.view(batch_size, -1, dim // 2) # [B, N*2, D/2]
        return x

class PatchTSTEmbedding(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.input_embedding = nn.Linear(config.patch_length, config.d_model)

    def forward(self, patch_input: torch.Tensor):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input for embedding
        return:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, d_model)`
        """
        embeddings = self.input_embedding(patch_input)
        return embeddings


class PatchTSTRopeAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper

    Implemented with p-rotary positional embeddings and integrated Flash Attention v2
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        use_rope: bool = True,
        max_wavelength: int = 10000,
        rope_percent: float = 0.5,
        config: Optional[PatchTSTConfig] = None,
        # =================================================================================
        # 2. 新增 use_flash_attention 参数
        # =================================================================================
        use_flash_attention: bool = True, 
    ):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        self.max_wavelength = max_wavelength
        self.rope_percent = rope_percent
        self.use_rope = use_rope
        self.config = config
        
        # =================================================================================
        # 3. 将 use_flash_attention 保存为类属性
        # =================================================================================
        self.use_flash_attention = use_flash_attention and _flash_attn_available

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return torch.arange(seq_len, device=device, dtype=dtype) + offset

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        linear_attn: bool = False, # `linear_attn` 与 flash attention 逻辑冲突, 在FA中忽略
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        if self.is_decoder:
            past_key_value = (key_states, value_states)
            
        # Reshape Q, K, V for attention calculation
        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        src_len = key_states.size(2)

        # RoPE is applied before attention
        if self.use_rope:
            # Flatten batch and head dims for RoPE
            # Shape: (bsz * num_heads, seq_len, head_dim)
            q_for_rope = query_states.reshape(-1, tgt_len, self.head_dim)
            k_for_rope = key_states.reshape(-1, src_len, self.head_dim)
            
            position_ids = self.get_seq_pos(
                src_len, key_states.device, key_states.dtype
            )
            k_for_rope, q_for_rope = apply_p_rope_to_qk(
                k_for_rope,
                q_for_rope,
                position_ids,
                self.head_dim,
                self.max_wavelength,
                self.rope_percent,
            )
            # Reshape back after RoPE
            query_states = q_for_rope.view(bsz, self.num_heads, tgt_len, self.head_dim)
            key_states = k_for_rope.view(bsz, self.num_heads, src_len, self.head_dim)

        # =================================================================================
        # 4. Flash Attention V2 核心替换逻辑
        # =================================================================================
        # 条件检查: 启用FA, 在CUDA上, attention_mask为空, 且数据类型为FP16/BF16
        can_use_flash_attn = (
            self.use_flash_attention
            and hidden_states.is_cuda
            and attention_mask is None
            and hidden_states.dtype in [torch.float16, torch.bfloat16]
        )

        if can_use_flash_attn:
            # Flash Attention 需要 (batch_size, seq_len, num_heads, head_dim)
            # 当前形状是 (batch_size, num_heads, seq_len, head_dim), 所以需要转置
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            # 调用 flash_attn_func
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.is_causal,
            )

            # Flash Attention 不返回注意力权重
            attn_weights_reshaped = None

            # 将输出形状从 (bsz, tgt_len, num_heads, head_dim) 变回 (bsz, tgt_len, embed_dim)
            attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        else: # 回退到原始的 PyTorch attention 实现
            query_states = query_states * self.scaling
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))

            if attn_weights.size() != (bsz, self.num_heads, tgt_len, src_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, tgt_len, src_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask.to(attn_weights.device)

            if not linear_attn:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            if layer_head_mask is not None:
                if layer_head_mask.size() != (self.num_heads,):
                    raise ValueError(
                        f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                        f" {layer_head_mask.size()}"
                    )
                attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

            if output_attentions:
                attn_weights_reshaped = attn_weights
            else:
                attn_weights_reshaped = None
            
            attn_probs = nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

            attn_output = torch.matmul(attn_probs, value_states)

            if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        # 最终的输出投影层, 对两种实现都适用
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class PatchTSTEncoderLayerWithRope(nn.Module):
    """
    PatchTST encoder layer with rope positional embeddings
    """

    def __init__(self, config: PatchTSTConfig, d_model: int, num_heads: int):
        super().__init__()
        
        # ========= 新增MoE超参数 (硬编码) =========
        self.use_moe = True # 开关，方便调试
        self.num_experts = 8 # 设置8个专家
        self.top_k = 2       # 每个token路由到最相关的2个专家
        # ========================================

        self.channel_attention = config.channel_attention
        # Multi-Head attention
        self.temporal_self_attn = PatchTSTRopeAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=config.attention_dropout,
            use_rope=True,
            max_wavelength=config.max_wavelength,
            rope_percent=config.rope_percent,
        )
        # self.temporal_mamba = Mamba2(
        #     d_model=config.d_model,
        #     d_state=1024,
        #     d_conv=4,
        #     expand=2,
        #     headdim=64
        # )
        if self.channel_attention:
            self.channel_self_attn = PatchTSTRopeAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=config.attention_dropout,
                use_rope=config.channel_rope,  # channels are not positional
                max_wavelength=config.max_wavelength,
                rope_percent=config.rope_percent,
            )

        # Add & Norm of the sublayer 1
        self.dropout_path1 = (
            nn.Dropout(config.path_dropout)
            if config.path_dropout > 0
            else nn.Identity()
        )
        if config.norm_type == "rmsnorm":
            self.norm_sublayer1 = PatchTSTRMSNorm(d_model, config.norm_eps)
        elif config.norm_type == "layernorm":
            self.norm_sublayer1 = nn.LayerNorm(d_model, eps=config.norm_eps)
        elif config.norm_type == "dyt":
            self.norm_sublayer1 = DyT(d_model)
        else:
            raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

        # Add & Norm of the sublayer 2
        if self.channel_attention:
            self.dropout_path2 = (
                nn.Dropout(config.path_dropout)
                if config.path_dropout > 0
                else nn.Identity()
            )
            if config.norm_type == "rmsnorm":
                self.norm_sublayer2 = PatchTSTRMSNorm(d_model, config.norm_eps)
            elif config.norm_type == "layernorm":
                self.norm_sublayer2 = nn.LayerNorm(d_model, eps=config.norm_eps)
            elif config.norm_type == "dyt":
                self.norm_sublayer2 = DyT(d_model)
            else:
                raise ValueError(
                    f"{config.norm_type} is not a supported norm layer type."
                )

        # ========= 【修改部分】: 使用MoE替换FFN =========
        ffn_dim = d_model * 4
        if self.use_moe:
            self.ff = NaiveMoE(
                d_model=d_model, 
                ffn_dim=ffn_dim, 
                num_experts=self.num_experts, 
                top_k=self.top_k, 
                config=config
            )
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, ffn_dim, bias=config.bias),
                ACT2CLS[config.activation_function](),
                nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
                nn.Linear(ffn_dim, d_model, bias=config.bias),
            )
        # =============================================


        # Add & Norm of sublayer 3
        self.dropout_path3 = (
            nn.Dropout(config.path_dropout)
            if config.path_dropout > 0
            else nn.Identity()
        )
        if config.norm_type == "rmsnorm":
            self.norm_sublayer3 = PatchTSTRMSNorm(d_model, config.norm_eps)
        elif config.norm_type == "layernorm":
            self.norm_sublayer3 = nn.LayerNorm(d_model, eps=config.norm_eps)
        elif config.norm_type == "dyt":
            self.norm_sublayer3 = DyT(d_model)
        else:
            raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

        self.pre_norm = config.pre_norm

    def forward(
        self,
        hidden_state: torch.Tensor,
        output_attentions: Optional[bool] = None,
        channel_attention_mask: Optional[torch.Tensor] = None,
        linear_attn: bool = False,
    ):
        """
        Parameters:
            hidden_state (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, d_model)`, *required*):
                Past values of the time series
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
        Return:
            `torch.Tensor` of shape `(batch_size, num_channels, sequence_length, d_model)`

        """
        batch_size, num_input_channels, sequence_length, d_model = hidden_state.shape

        # First sublayer: attention across time
        # hidden_states: [(bs*num_channels) x sequence_length x d_model]
        hidden_state = hidden_state.view(
            batch_size * num_input_channels, sequence_length, d_model
        )

        if self.pre_norm:
            ## Norm and Multi-Head attention and Add residual connection
            attn_output, attn_weights, _ = self.temporal_self_attn(
                hidden_states=self.norm_sublayer1(hidden_state),
                output_attentions=output_attentions,
            )
            # Add: residual connection with residual dropout
            hidden_state = hidden_state + self.dropout_path1(attn_output)
            # mamba_input = self.norm_sublayer1(hidden_state)
            # mamba_output = self.temporal_mamba(mamba_input)
            # hidden_state = hidden_state + self.dropout_path1(mamba_output)
        else:
            ## Multi-Head attention and Add residual connection and Norm - Standard Transformer from BERT
            attn_output, attn_weights, _ = self.temporal_self_attn(
                hidden_states=hidden_state,
                output_attentions=output_attentions,
                linear_attn=linear_attn,
            )
            # hidden_states: [(bs*num_channels) x sequence_length x d_model]
            hidden_state = self.norm_sublayer1(
                hidden_state + self.dropout_path1(attn_output)
            )
            # mamba_output = self.temporal_mamba(hidden_state)
            # hidden_state = self.norm_sublayer1(hidden_state + self.dropout_path1(mamba_output))
            
        # attn_weights = None

        # hidden_state: [bs x num_channels x sequence_length x d_model]
        hidden_state = hidden_state.reshape(
            batch_size, num_input_channels, sequence_length, d_model
        )

        # second sublayer: attention across variable at any given time
        if self.channel_attention:
            # hidden_state: [bs x sequence_length x num_channels x d_model]
            hidden_state = hidden_state.transpose(2, 1).contiguous()
            # hidden_state: [(bs*sequence_length) x num_channels x d_model]
            hidden_state = hidden_state.view(
                batch_size * sequence_length, num_input_channels, d_model
            )
            if self.pre_norm:
                ## Norm and Multi-Head attention and Add residual connection
                attn_output, channel_attn_weights, _ = self.channel_self_attn(
                    hidden_states=self.norm_sublayer2(hidden_state),
                    output_attentions=output_attentions,
                    attention_mask=channel_attention_mask,
                )
                # Add: residual connection with residual dropout
                hidden_state = hidden_state + self.dropout_path2(attn_output)
            else:
                ## Multi-Head attention and Add residual connection and Norm
                attn_output, channel_attn_weights, _ = self.channel_self_attn(
                    hidden_states=hidden_state,
                    output_attentions=output_attentions,
                    attention_mask=channel_attention_mask,
                    linear_attn=linear_attn,
                )
                # hidden_states: [(bs*sequence_length) x num_channels x d_model]
                hidden_state = self.norm_sublayer2(
                    hidden_state + self.dropout_path2(attn_output)
                )

            # Reshape hidden state
            # hidden_state: [bs x sequence_length x num_channels x d_model]
            hidden_state = hidden_state.reshape(
                batch_size, sequence_length, num_input_channels, d_model
            )
            # hidden_state: [bs x num_channels x sequence_length x d_model]
            hidden_state = hidden_state.transpose(1, 2).contiguous()

        # Third sublayer: mixing across hidden
        # hidden_state: [(batch_size*num_channels) x sequence_length x d_model]
        hidden_state = hidden_state.view(
            batch_size * num_input_channels, sequence_length, d_model
        )
        # ========= 【修改部分】: 处理MoE的输出和损失 =========
        moe_loss = torch.tensor(0.0, device=hidden_state.device) # 初始化损失
        if self.pre_norm:
            normalized_hidden_state = self.norm_sublayer3(hidden_state)
            if self.use_moe:
                ff_output, moe_loss = self.ff(normalized_hidden_state)
            else:
                ff_output = self.ff(normalized_hidden_state)
            hidden_state = hidden_state + self.dropout_path3(ff_output)
        else:
            if self.use_moe:
                ff_output, moe_loss = self.ff(hidden_state)
            else:
                ff_output = self.ff(hidden_state)
            hidden_state = self.norm_sublayer3(
                hidden_state + self.dropout_path3(ff_output)
            )
        # =================================================

        # [bs x num_channels x sequence_length x d_model]
        hidden_state = hidden_state.reshape(
            batch_size, num_input_channels, sequence_length, d_model
        )

        # ========= 【修改部分】: 在返回值中增加moe_loss =========
        outputs = (hidden_state,)
        if output_attentions:
            outputs += (
                (attn_weights, channel_attn_weights)
                if self.channel_attention
                else (attn_weights,)
            )
        outputs += (moe_loss,) # 将moe_loss添加到返回元组的末尾
        # =================================================

        return outputs

# =================================================================================
#  核心架构: PatchTSTUNetEncoder
# =================================================================================
class PatchTSTUNetEncoder(nn.Module):
    def __init__(self, config: PatchTSTConfig, depths: list, num_heads_list: list):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList()
        current_dim = config.d_model
        for i, depth in enumerate(depths):
            stage_layers = nn.ModuleList(
                [PatchTSTEncoderLayerWithRope(config, d_model=current_dim, num_heads=num_heads_list[i]) for _ in range(depth)]
            )
            
            downsample = PatchMerging(dim=current_dim, norm_eps=config.norm_eps) if i < len(depths) - 1 else None
            
            self.stages.append(nn.ModuleDict({
                "layers": stage_layers,
                "downsample": downsample
            }))
            
            if downsample:
                current_dim *= 2

    def forward(self, hidden_state, output_attentions=None, channel_attention_mask=None, linear_attn=False):
        skip_connections = []
        total_moe_loss = 0.0 # 初始化总moe损失
        
        for stage in self.stages:
            # 在下采样前保存skip connection
            skip_connections.append(hidden_state) 
            for layer in stage["layers"]:
                layer_outputs = layer(
                    hidden_state,
                    output_attentions=output_attentions,
                    channel_attention_mask=channel_attention_mask,
                    linear_attn=linear_attn,
                )
                hidden_state = layer_outputs[0]
                total_moe_loss += layer_outputs[-1]
            
            if stage["downsample"] is not None:
                batch_size, num_channels, num_patches, d_model = hidden_state.shape
                hidden_state_reshaped = hidden_state.view(batch_size * num_channels, num_patches, d_model)
                hidden_state_downsampled = stage["downsample"](hidden_state_reshaped)
                
                num_patches = hidden_state_downsampled.shape[1]
                d_model = hidden_state_downsampled.shape[2]
                hidden_state = hidden_state_downsampled.view(batch_size, num_channels, num_patches, d_model)

        return hidden_state, skip_connections, total_moe_loss

# =================================================================================
#  核心架构: PatchTSTUNetDecoder
# =================================================================================
class PatchTSTUNetDecoder(nn.Module):
    def __init__(self, config: PatchTSTConfig, depths: list, skip_connections_depths: list, num_heads_list: list):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList()
        
        reversed_depths = list(reversed(depths))
        reversed_num_heads = list(reversed(num_heads_list))
        
        # 获取 Encoder 输出的最高维度，例如 384
        encoder_bottleneck_dim = config.d_model

        # current_decoder_dim 用于追踪 decoder 中数据流的维度，初始值为 bottleneck 的维度
        current_decoder_dim = encoder_bottleneck_dim

        for i, depth in enumerate(reversed_depths):
            
            # 目标维度 (上采样后，或与 skip connection 融合后的维度)
            # 随着 i 增加，维度减半: 384 -> 192 -> 96 -> 48
            target_dim = encoder_bottleneck_dim // (2 ** i)
            current_num_heads = reversed_num_heads[i]
            
            # Upsample 模块需要知道它的输入维度 (current_decoder_dim)，并输出 target_dim
            # 我们的 PatchExpansion(D) 可以实现 D -> D/2，正好符合要求
            upsample = PatchExpansion(dim=current_decoder_dim) if i > 0 else None
            
            # Skip Processor 应该处理对应 Encoder 层的维度, 即 target_dim
            skip_processor_dim = target_dim
            skip_connection_processor = nn.ModuleList(
                [ConvNeXtBlock1D(skip_processor_dim, norm_eps=config.norm_eps) for _ in range(skip_connections_depths[len(depths)-1-i])]
            )

            # Transformer 层处理的维度是融合后的维度, 即 target_dim
            stage_layers = nn.ModuleList(
                [PatchTSTEncoderLayerWithRope(config, d_model=target_dim, num_heads=current_num_heads) for _ in range(depth)]
            )
            
            self.stages.append(nn.ModuleDict({
                "upsample": upsample,
                "skip_processor": skip_connection_processor,
                "layers": stage_layers,
            }))

            # 为下一次迭代更新维度
            current_decoder_dim = target_dim

    def forward(self, hidden_state, skip_connections, output_attentions=None, channel_attention_mask=None, linear_attn=False):
        reversed_skips = list(reversed(skip_connections))
        total_moe_loss = 0.0 # 初始化总moe损失
        
        # (关键修改): 创建一个列表来存储每个阶段的输出
        all_stage_outputs = []

        for i, stage in enumerate(self.stages):
            # Bottleneck 层 (i=0) 直接通过, 不进行上采样和融合
            if stage["upsample"] is not None:
                # 1. 上采样
                batch_size, num_channels, num_patches, d_model = hidden_state.shape
                hidden_state_reshaped = hidden_state.view(batch_size * num_channels, num_patches, d_model)
                hidden_state_upsampled = stage["upsample"](hidden_state_reshaped)
                
                # 2. 处理跳跃连接 (scOT 使用 ConvNeXt Blocks)
                skip = reversed_skips[i]
                bs_s, nc_s, np_s, nd_s = skip.shape
                skip_reshaped = skip.view(bs_s * nc_s, np_s, nd_s)
                for processor in stage["skip_processor"]:
                    skip_reshaped = processor(skip_reshaped)
                
                # 3. 融合 (scOT 使用 Addition)
                # 检查并处理由于padding导致的序列长度不匹配问题
                if hidden_state_upsampled.shape[1] != skip_reshaped.shape[1]:
                    diff = hidden_state_upsampled.shape[1] - skip_reshaped.shape[1]
                    skip_reshaped = nn.functional.pad(skip_reshaped, (0, 0, 0, diff))

                hidden_state = hidden_state_upsampled + skip_reshaped
                
                # 恢复形状
                num_patches, d_model = hidden_state.shape[1], hidden_state.shape[2]
                hidden_state = hidden_state.view(batch_size, num_channels, num_patches, d_model)
            
            # 4. 特征提取 (Transformer Layers)
            for layer in stage["layers"]:
                layer_outputs = layer(
                    hidden_state,
                    output_attentions=output_attentions,
                    channel_attention_mask=channel_attention_mask,
                    linear_attn=linear_attn,
                )
                hidden_state = layer_outputs[0]
                total_moe_loss += layer_outputs[-1]
            
            # (关键修改): 存储当前阶段的输出
            all_stage_outputs.append(hidden_state)

        # (关键修改): 将return语句移到循环外部，并返回所有阶段的输出列表
        return all_stage_outputs, total_moe_loss

class PatchTSTEncoder(PatchTSTPreTrainedModel):
    """
    PatchTST Encoder
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        if config.use_dynamics_embedding:
            # self.embedder = PatchTSTPolynomialEmbedding(config)
            self.embedder = PatchTSTKernelEmbedding(config)
        else:
            self.embedder = PatchTSTEmbedding(config)

        self.layers = nn.ModuleList(
            [
                PatchTSTEncoderLayerWithRope(config)
                for i in range(config.num_hidden_layers)
            ]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        patch_input: torch.Tensor,
        channel_attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        linear_attn: bool = False,
    ) -> BaseModelOutput:
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Past values of the time series
            output_hidden_states (bool, optional): Indicates if hidden states should be outputted.
            output_attentions (bool, optional): Indicates if attentions should be outputted.

        return:
            `BaseModelOutput`
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # Input embedding
        patch_input = self.embedder(patch_input)
        hidden_state = patch_input

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_state,)  # type: ignore

            layer_outputs = encoder_layer(
                hidden_state=hidden_state,
                output_attentions=output_attentions,
                channel_attention_mask=channel_attention_mask,
                linear_attn=linear_attn,
            )
            # get hidden state. hidden_state shape is [bs x num_channels x num_patches x d_model]
            # or [bs x num_channels x (num_patches+1) x d_model] if use cls_token
            hidden_state = layer_outputs[0]
            # append attention matrix at each layer
            if output_attentions:
                all_attentions = all_attentions + layer_outputs[1:]  # type: ignore
        # return past_values, hidden_states
        return BaseModelOutput(
            last_hidden_state=hidden_state,  # type: ignore
            hidden_states=encoder_states,  # type: ignore
            attentions=all_attentions,
        )


class PatchTSTModel(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.scaler = PatchTSTScaler(config)
        self.patchifier = PatchTSTPatchify(config)

        self.do_mask_input = config.do_mask_input

        if self.do_mask_input:
            self.masking = PatchTSTMasking(config)
        else:
            self.masking = nn.Identity()
        self.encoder = PatchTSTEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        channel_attention_mask: Optional[torch.Tensor] = None,
        linear_attn: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSTModelOutput]:
        r"""
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            future_values (`torch.BoolTensor` of shape `(batch_size, prediction_length, num_input_channels)`, *optional*):
                Future target values associated with the `past_values`
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTModelOutput` or tuple of `torch.Tensor` (if `return_dict`=False or `config.return_dict`=False)

        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
        patched_values = self.patchifier(scaled_past_values)

        if self.do_mask_input:
            masked_values, mask = self.masking(patched_values)
        else:
            masked_values, mask = self.masking(patched_values), None

        encoder_output = self.encoder(
            patch_input=masked_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            channel_attention_mask=channel_attention_mask,
            linear_attn=linear_attn,
        )

        if not return_dict:
            outputs = (
                encoder_output.last_hidden_state,
                encoder_output.hidden_states,
                encoder_output.attentions,
            )
            outputs = outputs + (mask, loc, scale, patched_values)
            return tuple(v for v in outputs if v is not None)

        return PatchTSTModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
            mask=mask,  # type: ignore
            loc=loc,
            scale=scale,
            patch_input=patched_values,
        )


class PatchTSTMaskPretrainHead(nn.Module):
    """
    Pretraining head for mask modelling
    """

    def __init__(
        self,
        d_model: int,
        patch_length: int,
        head_dropout: float = 0.0,
        use_cls_token: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        self.linear = nn.Linear(d_model, patch_length)
        self.use_cls_token = use_cls_token

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                    `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                            `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True

        """
        embedding = self.linear(
            self.dropout(embedding)
        )  # [bs x num_channels x num_patches x patch_length]
        if self.use_cls_token:
            embedding = embedding[:, :, 1:, :]  # remove the first cls token
        return embedding


class PatchTSTForPretraining(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        config.do_mask_input = True
        self.model = PatchTSTModel(config=config)
        self.head = PatchTSTMaskPretrainHead(
            d_model=config.d_model,
            patch_length=config.patch_length,
            head_dropout=config.head_dropout,
            use_cls_token=config.use_cls_token,
        )

        if config.loss == "mse":
            self.loss = nn.MSELoss(reduction="none")
        elif config.loss == "huber":
            self.loss = nn.HuberLoss(reduction="none", delta=config.huber_delta)
        else:
            raise ValueError(f"Unknown loss {config.loss}")
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        channel_attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        schedule_param: float = 0.0,
    ) -> Union[Tuple, PatchTSTForPretrainingOutput]:
        r"""
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*): Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForPretrainingOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # past_values: [bs x num_channels x num_patches x d_model] or
        # [bs x num_channels x (num_patches+1) x d_model] if use cls_token
        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            channel_attention_mask=channel_attention_mask,
            return_dict=True,
        )

        # last_hidden_state: [bs x num_channels x num_patches x d_model] or
        x_hat = model_output.last_hidden_state

        # [bs x num_channels x (num_patches+1) x d_model] if use cls_token
        # x_hat: [bs x num_channels x num_patches x patch_length]
        x_hat = self.head(x_hat)

        # reduce over the patch length dim first, then compute the masked loss over the tokens
        loss_val = self.loss(x_hat, model_output.patch_input)
        masked_loss = (loss_val.mean(dim=-1) * model_output.mask).sum() / (
            model_output.mask.sum() + 1e-10
        )

        encoder_states = model_output.hidden_states
        if not return_dict:
            outputs = (x_hat,) + model_output[1:-4]
            outputs = (masked_loss,) + outputs if masked_loss is not None else outputs
            return outputs
        return PatchTSTForPretrainingOutput(
            loss=masked_loss,
            prediction_output=x_hat,
            hidden_states=encoder_states,
            attentions=model_output.attentions,
        )

    @torch.no_grad()
    def generate_completions(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        channel_attention_mask: Optional[torch.Tensor] = None,
    ) -> CompletionsPatchTSTOutput:
        r"""
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Returns:
            `CompletionPatchTSTOutput`

        """

        # past_values: [bs x num_channels x num_patches x d_model] or
        # [bs x num_channels x (num_patches+1) x d_model] if use cls_token
        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            return_dict=True,
            channel_attention_mask=channel_attention_mask,
        )

        # last_hidden_state: [bs x num_channels x num_patches x d_model] or
        x_hat = model_output.last_hidden_state

        # [bs x num_channels x (num_patches+1) x d_model] if use cls_token
        # x_hat: [bs x num_channels x num_patches x patch_length]
        x_hat = self.head(x_hat)

        return CompletionsPatchTSTOutput(
            completions=x_hat,
            patched_past_values=model_output.patch_input,
            loc=model_output.loc,
            scale=model_output.scale,
            mask=model_output.mask,
        )


class PatchTSTPredictionHead(nn.Module):
    def __init__(
        self, config: PatchTSTConfig, num_patches: int = 1, distribution_output=None
    ):
        super().__init__()

        self.use_cls_token = config.use_cls_token
        self.pooling_type = config.pooling_type
        if self.pooling_type or self.use_cls_token:  # this should always be true
            head_dim = config.d_model
        else:  # included for completeness
            # num_patches is set to a dummy value,
            head_dim = config.d_model * num_patches

        # all the channels share the same head
        self.flatten = nn.Flatten(start_dim=2)
        if distribution_output is None:
            # use linear head with custom weight initialization
            self.projection = nn.Linear(head_dim, config.prediction_length, bias=False)
        else:
            # use distribution head
            self.projection = distribution_output.get_parameter_projection(head_dim)
        self.dropout = (
            nn.Dropout(config.head_dropout)
            if config.head_dropout > 0
            else nn.Identity()
        )

    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                     `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, forecast_len, num_channels)`

        """
        if self.use_cls_token:
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding[:, :, 0, :]
        else:
            if self.pooling_type == "mean":
                # pooled_embedding: [bs x num_channels x d_model]
                pooled_embedding = embedding.mean(dim=2)
            elif self.pooling_type == "max":
                # pooled_embedding: [bs x num_channels x d_model]
                pooled_embedding = embedding.max(dim=2).values
            else:
                # pooled_embedding: [bs x num_channels x num_patches x d_model]
                pooled_embedding = embedding

        # pooled_embedding: [bs x num_channels x (d_model * num_patches)] or [bs x num_channels x d_model)]
        pooled_embedding = self.flatten(pooled_embedding)
        pooled_embedding = self.dropout(pooled_embedding)

        # output: [bs x num_channels x forecast_len] or
        # tuple ([bs x num_channels x forecast_len], [bs x num_channels x forecast_len]) if using distribution head
        output = self.projection(pooled_embedding)

        if isinstance(output, tuple):
            # output: ([bs x forecast_len x num_channels], [bs x forecast_len x num_channels])
            output = tuple(z.transpose(2, 1) for z in output)
        else:
            output = output.transpose(2, 1)  # [bs x forecast_len x num_channels]
        return output

def print_tensor_stats(name: str, x: torch.Tensor):
    """打印张量的统计信息"""
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    
    print(
        f"--- Stats for: {name} ---\n"
        f"  Shape: {x.shape}\n"
        f"  dtype: {x.dtype}\n"
        f"  min: {x.min().item():.4f}, max: {x.max().item():.4f}\n"
        f"  mean: {x.mean().item():.4f}, std: {x.std().item():.4f}\n"
        f"  has_nan: {torch.isnan(x).any().item()}\n"
        f"-------------------------"
    )

class MultiStagePredictionHead(nn.Module):
    """
    一个预测头，用于处理来自U-Net Decoder多个阶段的输出。
    它会对每个阶段的输出进行 mean pooling，然后在特征维度上拼接，
    最后通过一个线性层进行预测。
    """
    def __init__(self, config: PatchTSTConfig, depths: list, distribution_output=None):
        super().__init__()
        self.config = config
        
        # 计算每个decoder阶段输出的特征维度
        encoder_bottleneck_dim = config.d_model * (2 ** (len(depths) - 1))
        decoder_dims = [encoder_bottleneck_dim // (2 ** i) for i in range(len(depths))]
        
        # 总的拼接后的维度
        total_concatenated_dim = sum(decoder_dims)
        head_dim = total_concatenated_dim

        self.flatten = nn.Flatten(start_dim=2)
        if distribution_output is None:
            self.projection = nn.Linear(head_dim, config.prediction_length, bias=False)
        else:
            self.projection = distribution_output.get_parameter_projection(head_dim)
        
        self.dropout = (
            nn.Dropout(config.head_dropout)
            if config.head_dropout > 0
            else nn.Identity()
        )

    def forward(self, decoder_outputs_list: list):
        """
        Parameters:
            decoder_outputs_list (`list` of `torch.Tensor`):
                每个元素的形状为 (bs, num_channels, num_patches_i, d_model_i)
        Returns:
            `torch.Tensor` of shape `(bs, forecast_len, num_channels)`
        """
        pooled_outputs = []
        for embedding in decoder_outputs_list:
            # 对每个阶段的输出在 patch 维度 (dim=2) 上进行 mean pooling
            pooled_embedding = embedding.mean(dim=2)
            pooled_outputs.append(pooled_embedding)
            
        # 在特征维度 (最后一个维度) 上拼接所有池化后的结果
        concatenated_embedding = torch.cat(pooled_outputs, dim=-1)
        
        # 后续处理与原版Head相同
        pooled_embedding = self.flatten(concatenated_embedding)
        pooled_embedding = self.dropout(pooled_embedding)
        
        output = self.projection(pooled_embedding)

        if isinstance(output, tuple):
            output = tuple(z.transpose(2, 1) for z in output)
        else:
            output = output.transpose(2, 1) # [bs x forecast_len x num_channels]
            
        return output

class PatchTSTForPrediction(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.config = config

        self.use_prompt_network = getattr(config, "use_prompt_network", False)

        # =========== 新增U-Net超参数 ===========
        self.depths = [1, 1, 1, 1]  # 每个阶段的Transformer层数
        self.skip_connections_depths = [2, 2, 2, 0]
        self.num_heads_list = [3, 6, 12, 24] 
        # =====================================
        # ========= 新增MoE损失系数 (硬编码) =========
        self.load_balance_coeff = 0.1
        # =========================================
        # ========= 【新增部分】: 新增MMD损失系数 (硬编码) =========
        self.mmd_loss_coeff = 0.5
        # =====================================================
        # ============= 新增 VAE 损失系数 =============
        self.kl_loss_coeff = 0.01  # VAE的KL散度损失权重
        self.recon_loss_coeff = 0.1 # VAE的重构损失权重
        # =========================================

        self.scaler = PatchTSTScaler(config)
        self.patchifier = PatchTSTPatchify(config)
        
        if config.use_dynamics_embedding:
            # self.embedder = PatchTSTPolynomialEmbedding(config)
            self.encoder_embedding = PatchTSTKernelEmbedding(config) # 假设我们使用Kernel Embedding
        else:
            self.encoder_embedding = PatchTSTEmbedding(config)
        
        # ============= 新增 Prompt Network 模块实例化 =============
        if self.use_prompt_network:
          # 1. 保留小波分析器，用于提取全局特征
          self.wavelet_analyzer = WaveletAnalyzer(
              input_timesteps=config.context_length,
              feature_dim=config.d_model
          )

          # 2. 【新增】实例化几何流形VAE
          # 定义VAE超参数 (可以移到config中)
          self.latent_dim = 32 # VAE隐空间维度
          self.geometry_vae = GeometryVAE(
              d_model=config.d_model,
              tcn_channels=64,
              num_tcn_layers=4,
              latent_dim=self.latent_dim
          )

          # 3. 【修改】调整 Memory 模块以接受融合后的特征
          # 融合特征维度 = 小波特征维度 + VAE隐空间维度
          fused_feature_dim = config.d_model + self.latent_dim
          self.prompt_network = Memory(
              num_memory=128,
              memory_dim=config.d_model # Prompt的最终维度保持不变
          )
          # 【关键】修改 Memory 内部的 x_proj 线性层以匹配新的输入维度
          self.prompt_network.x_proj = nn.Linear(fused_feature_dim, config.d_model)


          # 4. 融合MLP部分保持不变，因为它处理的是 prompt 和 bottleneck 的融合
          bottleneck_dim = config.d_model * (2 ** (len(self.depths) - 1))
          prompt_dim = config.d_model

          self.fusion_mlp = nn.Sequential(
              nn.Linear(bottleneck_dim + prompt_dim, bottleneck_dim),
              nn.GELU(),
              nn.Linear(bottleneck_dim, bottleneck_dim)
          )
        # ========================================================
        
        original_d_model = config.d_model
        self.encoder = PatchTSTUNetEncoder(config, depths=self.depths, num_heads_list=self.num_heads_list)
        config.d_model = original_d_model * (2 ** (len(self.depths) - 1))
        self.decoder = PatchTSTUNetDecoder(config, depths=self.depths, skip_connections_depths=self.skip_connections_depths, num_heads_list=self.num_heads_list)
        config.d_model = original_d_model

        # =========== 【修正部分】 完整继承 Distribution Output 逻辑 ===========
        if config.loss == "mse" or config.loss == "huber":
            self.distribution_output = None
        else:
            if config.distribution_output == "student_t":
                self.distribution_output = StudentTOutput(dim=config.prediction_length)
            elif config.distribution_output == "normal":
                self.distribution_output = NormalOutput(dim=config.prediction_length)
            elif config.distribution_output == "negative_binomial":
                self.distribution_output = NegativeBinomialOutput(dim=config.prediction_length)
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.head = MultiStagePredictionHead(config, depths=self.depths, distribution_output=self.distribution_output)

        if config.loss == "mse":
            self.loss = nn.MSELoss(reduction="mean")
        elif config.loss == "huber":
            self.loss = nn.HuberLoss(reduction="mean", delta=config.huber_delta)
        # 移除了对 else 的 raise ValueError，因为概率损失将在 forward 中处理
        # =================================================================

        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        channel_attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        linear_attn: bool = False,
    ) -> Union[Tuple, PatchTSTForPredictionOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)
            
        prompt = None
        vae_outputs = {} # 用于存储VAE的输出和损失
        if self.use_prompt_network:
            # 1. 【保留】使用小波分析器提取全局特征
            wavelet_input = past_values.permute(0, 2, 1)
            global_embedding = self.wavelet_analyzer(wavelet_input) # -> (B, d_model)

            # 2. Scaling and Patching (提前，因为VAE需要patched_values)
            scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
            patched_values = self.patchifier(scaled_past_values)

            # 3. Embedding (提前，因为VAE需要embedded_values)
            embedded_values = self.encoder_embedding(patched_values) # -> (B, V, P, D)

            # 4. 【新增】使用GeometryVAE提取流形特征
            # VAE的重构目标是归一化后的原始序列
            target_for_vae_recon = scaled_past_values.permute(0, 2, 1) # -> (B, V, T)
            x_recon, mu, logvar, z = self.geometry_vae(embedded_values, target_T=target_for_vae_recon.shape[2])

            # 存储VAE的输出用于计算损失
            vae_outputs['x_recon'] = x_recon
            vae_outputs['target'] = target_for_vae_recon
            vae_outputs['mu'] = mu
            vae_outputs['logvar'] = logvar

            # 5. 【新增】融合两种特征
            fused_features = torch.cat([global_embedding, z], dim=-1) # -> (B, d_model + latent_dim)

            # 6. 【修改】使用融合特征生成最终Prompt
            prompt = self.prompt_network(fused_features) # -> (B, d_model)

        # 如果不使用prompt network，则正常执行
        if not self.use_prompt_network:
            # 1. Scaling and Patching
            scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
            patched_values = self.patchifier(scaled_past_values)
            # 2. Embedding
            embedded_values = self.encoder_embedding(patched_values)
        
        # 3. UNet Encoder
        encoder_output, skip_connections, encoder_moe_loss = self.encoder(
            embedded_values,
            output_attentions=output_attentions,
            channel_attention_mask=channel_attention_mask,
            linear_attn=linear_attn,
        )
        
        decoder_input = encoder_output
        if self.use_prompt_network and prompt is not None:
            # encoder_output (bottleneck) 形状: (B, C, N_bottle, D_bottle)
            B, C, N_bottle, D_bottle = encoder_output.shape
            D_prompt = prompt.shape[-1]
            
            # 将 prompt (B, D_prompt) 扩展为 (B, C, N_bottle, D_prompt)
            prompt_expanded = prompt.view(B, 1, 1, D_prompt).expand(-1, C, N_bottle, -1)
            # print_tensor_stats("Expanded Prompt for Fusion", prompt_expanded) # <--- 新增

            # 在特征维度上拼接
            concatenated_features = torch.cat([encoder_output, prompt_expanded], dim=-1)
            # print_tensor_stats("Concatenated Features (before fusion_mlp)", concatenated_features) # <--- 新增

            # 通过 MLP 进行融合
            fused_features = self.fusion_mlp(concatenated_features.view(-1, D_bottle + D_prompt))
            # print_tensor_stats("Fused Features (after fusion_mlp)", fused_features) # <--- 新增

            # 恢复形状为 (B, C, N_bottle, D_bottle)
            decoder_input = fused_features.view(B, C, N_bottle, D_bottle)
            # print_tensor_stats("Final Decoder Input (after fusion)", decoder_input) # <--- 新增

        # 4. UNet Decoder
        decoder_outputs_list, decoder_moe_loss = self.decoder(
            decoder_input,
            skip_connections,
            output_attentions=output_attentions,
            channel_attention_mask=channel_attention_mask,
            linear_attn=linear_attn,
        )
        
        # 5. Prediction Head
        y_hat = self.head(decoder_outputs_list)
        
        loss_val = None
        if future_values is not None:
            # 初始化各个损失分量
            prediction_loss = torch.tensor(0.0, device=past_values.device)
            mmd_loss = torch.tensor(0.0, device=past_values.device)
            kl_loss = torch.tensor(0.0, device=past_values.device)
            recon_loss = torch.tensor(0.0, device=past_values.device)

            if self.distribution_output:
                # 概率预测损失
                y_hat_out = y_hat
                distribution = self.distribution_output.distribution(y_hat, loc=loc, scale=scale)
                prediction_loss = nll(distribution, future_values)
                prediction_loss = weighted_average(prediction_loss)
            else:
                # 点预测损失
                y_hat_out = y_hat * scale + loc
                prediction_loss = self.loss(y_hat_out, future_values)
                
                # 计算 Conditional MMD Loss
                if self.mmd_loss_coeff > 0:
                    batch_mean = loc.mean(dim=0)
                    batch_variance = (scale**2).mean(dim=0)
                    mmd_loss = conditional_mmd_multi_step(
                        input_traj=None,
                        true_traj=future_values,
                        pred_traj=y_hat_out,
                        mean=batch_mean,
                        variance=batch_variance,
                    )
                    
            # ======================= 【关键修正】 =======================
            # 1. 始终从模型输出中获取 MoE 损失，以保持计算图完整
            total_moe_loss = encoder_moe_loss + decoder_moe_loss

            # 2. 初始化 VAE 损失为零
            kl_loss = torch.tensor(0.0, device=past_values.device)
            recon_loss = torch.tensor(0.0, device=past_values.device)

            # 3. 根据是否微调(use_prompt_network)来决定损失的构成
            if self.use_prompt_network:
                # 在微调阶段，计算 VAE 损失
                # KL Divergence Loss
                kl_loss = -0.5 * torch.sum(1 + vae_outputs['logvar'] - vae_outputs['mu'].pow(2) - vae_outputs['logvar'].exp())
                kl_loss = kl_loss / past_values.shape[0]

                # Reconstruction Loss
                recon_loss = self.loss(vae_outputs['x_recon'], vae_outputs['target'])
                
                # 【核心】在微调阶段，我们不希望 MoE 损失有任何贡献
                effective_moe_coeff = 0.0
            else:
                # 在第一阶段预训练时，使用正常的 MoE 损失系数
                effective_moe_coeff = self.load_balance_coeff
            # ==========================================================

            # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            #     print("\n--- Loss Breakdown ---")
            #     print(f"  Prediction Loss: {prediction_loss.item():.6f}")
            #     
            #     weighted_mmd = self.mmd_loss_coeff * mmd_loss
            #     print(f"  MMD Loss (raw): {mmd_loss.item():.6f} | Weighted: {weighted_mmd.item():.6f} (coeff: {self.mmd_loss_coeff})")
            #     
            #     weighted_moe = effective_moe_coeff * total_moe_loss
            #     print(f"  Total MoE Loss (raw): {total_moe_loss.item():.6f} | Weighted: {weighted_moe.item():.6f} (coeff: {effective_moe_coeff:.4f})")
            #     
            #     weighted_kl = self.kl_loss_coeff * kl_loss
            #     print(f"  VAE KL Loss (raw): {kl_loss.item():.6f} | Weighted: {weighted_kl.item():.6f} (coeff: {self.kl_loss_coeff})")
# 
            #     weighted_recon = self.recon_loss_coeff * recon_loss
            #     print(f"  VAE Recon Loss (raw): {recon_loss.item():.6f} | Weighted: {weighted_recon.item():.6f} (coeff: {self.recon_loss_coeff})")
            #     
            #     # 手动计算总损失以验证
            #     manual_total_loss = prediction_loss + weighted_mmd + weighted_moe + weighted_kl + weighted_recon
            #     print(f"  --------------------")
            #     print(f"  Calculated Total Loss: {manual_total_loss.item():.6f}")
            #     print("----------------------\n")
            # # =================================================================
            
            # 组合所有损失分量，计算最终的总损失
            loss_val = (
                prediction_loss 
                + self.mmd_loss_coeff * mmd_loss 
                + effective_moe_coeff * total_moe_loss  # <--- 使用条件系数
                + self.kl_loss_coeff * kl_loss
                + self.recon_loss_coeff * recon_loss
            )

        # 根据是否有分布输出，确定最终的预测值
        if self.distribution_output:
            y_hat_out = y_hat
        else:
            y_hat_out = y_hat * scale + loc
        
        if not return_dict:
            outputs = (y_hat_out, loc, scale)
            return (loss_val,) + outputs if loss_val is not None else outputs
        
        return PatchTSTForPredictionOutput(
            loss=loss_val,
            prediction_outputs=y_hat_out,
            hidden_states=None,
            attentions=None,
            loc=loc,
            scale=scale,
        )
        
    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        channel_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> SamplePatchTSTOutput:
        """
        从带有概率分布头的模型中生成样本序列。
        """
        # get number of samples
        num_parallel_samples = self.config.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            future_values=None,
            past_observed_mask=past_observed_mask,
            output_hidden_states=False,
            channel_attention_mask=channel_attention_mask,
            output_attentions=output_attentions,
        )

        if self.distribution_output:
            # get distribution
            distribution = self.distribution_output.distribution(
                outputs.prediction_outputs, loc=outputs.loc, scale=outputs.scale
            )
            # get samples: list of [bs x forecast_len x num_channels]
            samples = [distribution.sample() for _ in range(num_parallel_samples)]
            # samples: [bs x num_samples x forecast_len x num_channels]
            samples = torch.stack(samples, dim=1)
        else:
            samples = outputs.prediction_outputs.unsqueeze(1)

        return SamplePatchTSTOutput(sequences=samples)
