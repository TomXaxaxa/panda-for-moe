"""Exposed PatchTST model, taken from HuggingFace transformers"""

try:
    from flash_attn import flash_attn_func
    _flash_attn_available = True
    print("Flash Attention is available and will be used.")
except ImportError:
    _flash_attn_available = False
    print("Flash Attention is not available, falling back to standard attention.")

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# from mamba_ssm import Mamba2
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

import numpy as np

def calculate_activated_params(model: nn.Module, top_k: int, num_experts: int):
    """
    计算并打印模型中的总参数量和激活参数量。

    Args:
        model (nn.Module): 您的 PyTorch 模型实例。
        top_k (int): MoE 层中每个 token 激活的 expert 数量。
        num_experts (int): MoE 层中总的 expert 数量。
    """
    total_params = 0
    shared_params = 0
    expert_params = 0
    
    # 存储各类参数的详细信息
    param_details = {
        "shared": {"count": 0, "size": []},
        "expert": {"count": 0, "size": []},
        "gate": {"count": 0, "size": []}
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        num_params = param.numel()
        total_params += num_params

        # MoE 的 experts 参数通常在其模块名中包含 "experts"
        if 'experts' in name:
            expert_params += num_params
            param_details["expert"]["count"] += num_params
            param_details["expert"]["size"].append((name, num_params))
        # MoE 的 gate 参数通常在其模块名中包含 "gate"
        elif 'gate' in name:
            shared_params += num_params # 门控网络是共享的
            param_details["gate"]["count"] += num_params
            param_details["gate"]["size"].append((name, num_params))
        else:
            shared_params += num_params
            param_details["shared"]["count"] += num_params
            param_details["shared"]["size"].append((name, num_params))

    # Expert 参数需要特殊处理
    if num_experts > 0 and expert_params > 0:
        # 1. 计算单个 expert 的参数量
        params_per_expert = expert_params / num_experts
        
        # 2. 计算激活的 expert 参数量
        activated_expert_params = params_per_expert * top_k
        
        # 3. 计算总的激活参数量
        total_activated_params = shared_params + activated_expert_params
    else:
        # 如果模型中没有 MoE 层，则激活参数量等于总参数量
        params_per_expert = 0
        activated_expert_params = 0
        total_activated_params = total_params

    # 格式化输出
    def format_M(num):
        return f"{num / 1e6:.2f}M"

    print("="*60)
    print("Model Parameter Analysis (PatchTST with MoE)")
    print("="*60)
    print(f"Total Parameters: {format_M(total_params)}")
    print(f"Activated Parameters: {format_M(total_activated_params)}")
    print("-"*60)
    print("Breakdown:")
    print(f"  - Shared Parameters: {format_M(shared_params)}")
    print(f"    (Includes embeddings, attention, norms, head, MoE gate, etc.)")
    print(f"  - Total Expert Parameters: {format_M(expert_params)} (across all {num_experts} experts)")
    print(f"  - Parameters per Expert: {format_M(params_per_expert)}")
    print(f"  - Activated Expert Params: {format_M(activated_expert_params)} (top_k={top_k})")
    print("="*60)
    
    # 返回详细数值以便程序化使用
    details = {
        "total_params": total_params,
        "activated_params": total_activated_params,
        "shared_params": shared_params,
        "total_expert_params": expert_params,
        "params_per_expert": params_per_expert,
        "activated_expert_params": activated_expert_params,
    }
    
    return total_activated_params, total_params, details

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
        x = self.conv1(x)
        
        x = x.permute(0, 2, 1) # (B, C, L) -> (B, L, C) for LayerNorm
        x = self.ln1(x)
        
        x = x.permute(0, 2, 1) # (B, L, C) -> (B, C, L) back to conv format
        x = self.gelu1(x)
        
        x = self.conv2(x)
        
        x = x.permute(0, 2, 1) # (B, C, L) -> (B, L, C) for LayerNorm
        x = self.ln2(x)
        
        x = x.permute(0, 2, 1) # (B, L, C) -> (B, C, L) back to conv format
        x = self.gelu2(x)
        
        x = self.pool(x)
        x = self.flatten(x)
        
        return x
class WaveletAnalyzer(nn.Module):
    def __init__(self, input_timesteps, feature_dim, J=6, Q=8):
        super().__init__()
        self.scattering = kymatio.Scattering1D(J=J, shape=(input_timesteps,), Q=Q)
        with torch.no_grad():
            dummy_input = torch.randn(1, input_timesteps)
            n_coeffs = self.scattering(dummy_input).shape[1]
            
        self.cnn_extractor = CnnExtractorWithLayerNorm(n_coeffs)
        self.final_mlp = nn.Linear(128, feature_dim)

    def forward(self, x):
        B, V, T = x.shape
        
        x_reshaped = x.reshape(B * V, T)
        scattering_coeffs = self.scattering(x_reshaped.contiguous())
        cnn_features = self.cnn_extractor(scattering_coeffs)
        features = self.final_mlp(cnn_features)
        features_reshaped = features.view(B, V, -1)
        final_embedding = features_reshaped.mean(dim=1)
        stabilized_embedding = torch.sign(final_embedding) * torch.log(torch.abs(final_embedding) + 1)
        
        return stabilized_embedding

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
    一个使用基于置换的路由策略进行优化的MoE层。
    它通过向量化操作消除了主要的for循环瓶颈，显著提升了计算效率。
    """
    def __init__(self, d_model: int, ffn_dim: int, num_experts: int, top_k: int, config: PatchTSTConfig):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 门控网络
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
        gate_logits = self.gate(x_reshaped)
        
        # 2. 计算负载均衡损失 (与原始版本相同)
        router_probs = F.softmax(gate_logits, dim=-1)
        tokens_per_expert_prob = router_probs.mean(dim=0)
        # 修正一下原始代码中的小错误，应该是 tokens_per_expert_prob 的平方
        load_balance_loss = self.num_experts * torch.sum(tokens_per_expert_prob * tokens_per_expert_prob)

        # 3. 找到Top-k的experts并进行路由
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1) # [num_tokens, top_k]
        
        # 归一化top-k的权重
        top_k_weights = top_k_weights / torch.sum(top_k_weights, dim=-1, keepdim=True)
        
        # 4. === 高效的向量化分发 (Vectorized Dispatch) ===
        # 将 top_k 信息展平
        flat_top_k_indices = top_k_indices.flatten()  # Shape: [num_tokens * top_k]
        
        # 创建一个置换索引，它能将 token-expert 对按照 expert_id 排序
        # 这是优化的核心：将随机访问模式转换为顺序访问模式
        perm = torch.argsort(flat_top_k_indices)
        
        # 应用置换
        perm_flat_top_k_indices = flat_top_k_indices[perm]

        # 找到每个专家处理的 token 在置换后序列中的边界
        # counts 告诉我们每个专家收到了多少 token
        counts = torch.bincount(perm_flat_top_k_indices, minlength=self.num_experts)
        # starts 是每个专家 token 块的起始索引
        starts = torch.cat((torch.tensor([0], device=x.device), counts.cumsum(0)[:-1]))

        # 根据置换索引，高效地收集需要处理的 token
        # 首先，我们需要知道原始 token 的索引
        token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)
        perm_token_indices = token_indices[perm]
        perm_inputs = x_reshaped[perm_token_indices]

        # 5. === 批量化专家计算 ===
        # 初始化置换后的输出张量
        perm_outputs = torch.zeros_like(perm_inputs)
        
        # 这个循环远快于原始版本，因为它操作的是连续的数据块 (memory-coalesced access)
        for i in range(self.num_experts):
            start, end = starts[i], starts[i] + counts[i]
            if start < end:  # 仅当专家接收到 token 时才计算
                expert_input = perm_inputs[start:end]
                expert_output = self.experts[i](expert_input)
                perm_outputs[start:end] = expert_output
        
        # 6. === 高效的向量化聚合 (Vectorized Scatter) ===
        # 创建逆置换索引，用于将结果恢复到原始顺序
        inv_perm = torch.argsort(perm)
        
        # 恢复顺序
        unperm_outputs = perm_outputs[inv_perm]
        
        # 应用路由权重
        unperm_outputs = unperm_outputs * top_k_weights.flatten().unsqueeze(-1)
        
        # 将 top_k 个专家的输出加权求和
        # unperm_outputs shape: [num_tokens * top_k, d_model]
        # 借助 zero_like 和 index_add_ 实现高效的 scatter-add
        final_output_reshaped = torch.zeros_like(x_reshaped).index_add_(0, token_indices, unperm_outputs)

        # 7. 恢复原始形状并返回
        return final_output_reshaped.reshape(original_shape), load_balance_loss

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
        self.weight = (
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
        if self.weight is not None:
            x = self.weight * x
        
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


# =================================================================================
#  新增模块: ChannelIndependentMasking (用于Pre-training)
# =================================================================================
class ChannelIndependentMasking(nn.Module):
    """
    实现通道独立的掩码策略 (Channel-Independent Masking)。
    - 对每个通道独立进行掩码。
    - 使用一个可学习的 `mask_token` 向量替换被掩码的Patch。
    """
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.mask_ratio = 0.5 # 从config读取或默认为50%
        # 定义一个可学习的 mask token embedding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config.d_model))
        # 对 mask_token 进行高斯初始化
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (`torch.Tensor` of shape `(B, V, N, D)`): 输入的embedded patches.
                B=batch_size, V=num_channels, N=num_patches, D=d_model

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - masked_x: 经过掩码处理的输入
            - mask: 二进制掩码张量 (1代表被掩码, 0代表可见)
        """
        B, V, N, D = x.shape

        # 为每个通道独立生成用于选择掩码位置的随机噪声
        noise = torch.rand(B, V, N, device=x.device)

        # 计算每个通道需要掩码的patch数量
        num_masked = int(N * self.mask_ratio)
        if num_masked == 0:
             # 如果不进行掩码，直接返回原输入和全0的mask
             return x, torch.zeros(B, V, N, device=x.device)


        # 对每个通道独立地找到噪声最小的 top-k 个索引，作为掩码位置
        _, masked_indices = torch.topk(noise, num_masked, dim=2, largest=False)

        # 创建二进制掩码张量 `mask` (True/1 表示该位置被掩码)
        mask = torch.zeros(B, V, N, device=x.device, dtype=torch.bool)
        mask.scatter_(2, masked_indices, True)

        # 将掩码张量扩展到与输入 x 相同的维度，以便进行替换操作
        # bool_mask shape: (B, V, N, 1)
        bool_mask = mask.unsqueeze(-1)

        # 使用 torch.where 高效地将 `mask_token` 替换到被掩码的位置
        masked_x = torch.where(bool_mask, self.mask_token, x)

        # 返回处理后的输入和用于计算损失的掩码 (转换为 float 类型)
        return masked_x, mask.float()

# =================================================================================
#  核心修改: 更新 PatchTSTForPretraining 以适配 U-Net 架构和新掩码策略
# =================================================================================
class PatchTSTForPretraining(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.config = config

        # =========== 引用与预测模型一致的U-Net超参数 ===========
        self.depths = [2, 2, 2, 2]
        self.skip_connections_depths = [2, 2, 2, 0]
        self.num_heads_list = [3, 6, 12, 24]
        # =======================================================
        # =========== 引用与预测模型一致的MoE损失系数 ===========
        self.load_balance_coeff = 0.1
        # =======================================================

        # 1. 核心组件
        self.scaler = PatchTSTScaler(config)
        self.patchifier = PatchTSTPatchify(config)
        
        # 2. 采用新的通道独立掩码模块
        self.masking = ChannelIndependentMasking(config)

        # 3. Embedding层
        if config.use_dynamics_embedding:
            self.encoder_embedding = PatchTSTKernelEmbedding(config)
        else:
            self.encoder_embedding = PatchTSTEmbedding(config)

        # 4. U-Net Encoder 和 Decoder
        # 注意: U-Net的各阶段维度会变化，初始化时需正确处理 config.d_model
        original_d_model = config.d_model
        self.encoder = PatchTSTUNetEncoder(config, depths=self.depths, num_heads_list=self.num_heads_list)
        
        # 临时更新config中的d_model以正确初始化Decoder
        config.d_model = original_d_model * (2 ** (len(self.depths) - 1))
        self.decoder = PatchTSTUNetDecoder(config, depths=self.depths, skip_connections_depths=self.skip_connections_depths, num_heads_list=self.num_heads_list)
        # 恢复config中的d_model
        config.d_model = original_d_model

        # 5. 专用的预训练头 (保持不变，其功能是 d_model -> patch_length)
        self.head = PatchTSTMaskPretrainHead(
            d_model=config.d_model, # Head在Decoder最后一层输出上操作，维度是原始d_model
            patch_length=config.patch_length,
            head_dropout=config.head_dropout,
            use_cls_token=config.use_cls_token,
        )

        # 6. 损失函数 (reduction='none'以便手动计算掩码损失)
        if config.loss == "mse":
            self.loss = nn.MSELoss(reduction="none")
        elif config.loss == "huber":
            self.loss = nn.HuberLoss(reduction="none", delta=config.huber_delta)
        else:
            raise ValueError(f"Unknown loss {config.loss}")
            
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        channel_attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSTForPretrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        # 步骤 1: Scaling, Patching, Embedding
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
        # `target_patches` 是我们需要重建的目标
        target_patches = self.patchifier(scaled_past_values)
        embedded_values = self.encoder_embedding(target_patches)

        # 步骤 2: 应用通道独立掩码
        # `masked_values` 是部分patch被替换为mask_token后的输入
        # `mask` 是二进制掩码 (1 for masked)
        masked_values, mask = self.masking(embedded_values)

        # 步骤 3: U-Net Encoder
        encoder_output, skip_connections, encoder_moe_loss = self.encoder(
            masked_values,
            output_attentions=output_attentions,
            channel_attention_mask=channel_attention_mask,
        )

        # 步骤 4: U-Net Decoder
        decoder_outputs_list, decoder_moe_loss = self.decoder(
            encoder_output,
            skip_connections,
            output_attentions=output_attentions,
            channel_attention_mask=channel_attention_mask,
        )
        # 预训练头作用在最后一层输出上，其分辨率与原始输入一致
        final_decoder_output = decoder_outputs_list[-1]

        # 步骤 5: 通过预训练头进行重建
        reconstructed_patches = self.head(final_decoder_output)

        # 步骤 6: 计算损失
        # 计算所有patch的重建损失 (element-wise)
        loss_per_patch = self.loss(reconstructed_patches, target_patches)

        # 只对被掩码的patch计算损失
        # loss_per_patch: [B, V, N, P], mask: [B, V, N]
        # 先在patch内部取均值，再用mask筛选
        reconstruction_loss = (loss_per_patch.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-10)
        
        # 合并MoE的负载均衡损失
        total_moe_loss = encoder_moe_loss + decoder_moe_loss
        total_loss = reconstruction_loss + self.load_balance_coeff * total_moe_loss

        if not return_dict:
            output = (reconstructed_patches,)
            return (total_loss,) + output

        return PatchTSTForPretrainingOutput(
            loss=total_loss,
            prediction_output=reconstructed_patches,
            hidden_states=None, # U-Net结构复杂，暂不传递所有中间状态
            attentions=None,
        )
        
    # (这是需要添加到 PatchTSTForPretraining 类内部的方法)

    @torch.no_grad()
    def generate_completions(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        channel_attention_mask: Optional[torch.Tensor] = None,
    ) -> CompletionsPatchTSTOutput:
        """
        使用预训练好的模型，对输入序列进行掩码并生成其重建/补全版本。
        这是一个推理函数，不计算梯度或损失。
        """
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        # 步骤 1: Scaling, Patching, Embedding (与forward方法一致)
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
        target_patches = self.patchifier(scaled_past_values)
        embedded_values = self.encoder_embedding(target_patches)

        # 步骤 2: 应用通道独立掩码 (与forward方法一致)
        # 即使在推理时，也应用相同的随机掩码策略来观察模型的重建能力
        masked_values, mask = self.masking(embedded_values)

        # 步骤 3: U-Net Encoder
        encoder_output, skip_connections, _ = self.encoder(
            masked_values,
            output_attentions=False, # 推理时通常不输出注意力图
            channel_attention_mask=channel_attention_mask,
        )

        # 步骤 4: U-Net Decoder
        decoder_outputs_list, _ = self.decoder(
            encoder_output,
            skip_connections,
            output_attentions=False,
            channel_attention_mask=channel_attention_mask,
        )
        final_decoder_output = decoder_outputs_list[-1]

        # 步骤 5: 通过预训练头进行重建
        reconstructed_patches = self.head(final_decoder_output)

        # 步骤 6: 封装并返回结果
        return CompletionsPatchTSTOutput(
            completions=reconstructed_patches,    # 模型重建的patch
            patched_past_values=target_patches, # 原始的patch (用于对比)
            loc=loc,                              # 反归一化参数
            scale=scale,                          # 反归一化参数
            mask=mask,                            # 本次推理中使用的掩码
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

class MultiStagePredictionHead(nn.Module):
    """
    方案二: 使用MLP对齐小波特征后再进行融合
    """
    def __init__(self, config: PatchTSTConfig, depths: list, distribution_output=None, wavelet_feature_dim: int = 0):
        super().__init__()
        self.config = config
        self.wavelet_feature_dim = wavelet_feature_dim
        
        # 【新增】定义一个MLP来处理小波特征
        if self.wavelet_feature_dim > 0:
            self.wavelet_mlp = nn.Sequential(
                nn.Linear(self.wavelet_feature_dim, self.wavelet_feature_dim * 2),
                nn.GELU(),
                nn.Linear(self.wavelet_feature_dim * 2, self.wavelet_feature_dim)
            )

        # 计算时域特征维度
        encoder_bottleneck_dim = config.d_model * (2 ** (len(depths) - 1))
        decoder_dims = [encoder_bottleneck_dim // (2 ** i) for i in range(len(depths))]
        
        total_time_domain_dim = sum(decoder_dims)
        # 最终线性层的输入维度不变，因为MLP的输出维度与输入相同
        head_dim = total_time_domain_dim + self.wavelet_feature_dim

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

    def forward(self, decoder_outputs_list: list, wavelet_embedding: Optional[torch.Tensor] = None):
        B, V, _, _ = decoder_outputs_list[0].shape
        
        pooled_outputs = []
        for embedding in decoder_outputs_list:
            pooled_embedding = embedding.mean(dim=2)
            pooled_outputs.append(pooled_embedding)
            
        time_domain_embedding = torch.cat(pooled_outputs, dim=-1)

        if wavelet_embedding is not None and self.wavelet_feature_dim > 0:
            # 【修改】让小波特征先通过MLP进行变换
            processed_wavelet_embedding = self.wavelet_mlp(wavelet_embedding)
            
            wavelet_embedding_expanded = processed_wavelet_embedding.unsqueeze(1).expand(-1, V, -1)
            final_embedding = torch.cat([time_domain_embedding, wavelet_embedding_expanded], dim=-1)
        else:
            final_embedding = time_domain_embedding

        flattened_embedding = self.flatten(final_embedding)
        dropped_embedding = self.dropout(flattened_embedding)
        
        output = self.projection(dropped_embedding)

        if isinstance(output, tuple):
            output = tuple(z.transpose(2, 1) for z in output)
        else:
            output = output.transpose(2, 1)
            
        return output

class PatchTSTForPrediction(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.config = config

        # ==================== 【新增代码段 1】 ====================
        # 定义变长训练和固定长度推理的超参数
        # 这里的数值可以根据您的数据集特性进行调整
        self.training_truncate_lengths = [128, 256, 384, 512] 
        self.inference_truncate_length = 512 # 推理时使用固定的长度以保证结果可复现

        # 确保推理长度也在训练列表中，这通常是一个好的实践
        if self.inference_truncate_length not in self.training_truncate_lengths:
            self.training_truncate_lengths.append(self.inference_truncate_length)
        
        # 确保所有指定的长度都不超过模型配置的最大上下文长度
        self.training_truncate_lengths = [
            min(l, config.context_length) for l in self.training_truncate_lengths
        ]
        self.inference_truncate_length = min(self.inference_truncate_length, config.context_length)
        
        # 引入 random 模块用于随机选择
        import random
        self.random = random
        # =========================================================

        # =========== 新增U-Net超参数 ===========
        self.depths = [1, 1, 1, 1]  # 每个阶段的Transformer层数
        self.skip_connections_depths = [2, 2, 2, 0]
        self.num_heads_list = [3, 6, 12, 24]  # 每个阶段的注意力头数
        # =====================================
        # ========= 新增MoE损失系数 (硬编码) =========
        self.load_balance_coeff = 0.1
        # =========================================
        # ========= 【新增部分】: 新增MMD损失系数 (硬编码) =========
        self.mmd_loss_coeff = 0.5
        # =====================================================
        
        # ========= 【新增部分】: 初始化小波分析器 =========
        self.wavelet_feature_dim = 48 # 定义频域特征维度
        self.wavelet_analyzer = WaveletAnalyzer(
            input_timesteps=config.context_length,
            feature_dim=self.wavelet_feature_dim
        )
        # =====================================================

        self.scaler = PatchTSTScaler(config)
        self.patchifier = PatchTSTPatchify(config)
        
        if config.use_dynamics_embedding:
            # self.embedder = PatchTSTPolynomialEmbedding(config)
            self.encoder_embedding = PatchTSTKernelEmbedding(config) # 假设我们使用Kernel Embedding
        else:
            self.encoder_embedding = PatchTSTEmbedding(config)
        
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

        self.head = MultiStagePredictionHead(config, depths=self.depths, distribution_output=self.distribution_output, wavelet_feature_dim=self.wavelet_feature_dim)

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

        # ==================== 【新增代码段 2】 ====================
        # 1. 根据模型状态（训练/推理）确定目标截断长度
        if self.training:
            # 训练阶段：从列表中随机选择一个长度
            target_len = self.random.choice(self.training_truncate_lengths)
        else:
            # 推理阶段：使用固定的长度
            target_len = self.inference_truncate_length
        
        # 确保截断长度不超过当前输入的实际长度
        current_seq_len = past_values.shape[1]
        target_len = min(target_len, current_seq_len)

        # 2. 对时域分支的输入进行截断
        past_values_truncated = past_values[:, :target_len, :]
        
        if past_observed_mask is not None:
            past_observed_mask_truncated = past_observed_mask[:, :target_len, :]
        else:
            past_observed_mask_truncated = torch.ones_like(past_values_truncated)
        
        # 3. 对频域分支的输入进行填充
        #    a. 先进行同样的截断
        wavelet_input_truncated = past_values[:, -target_len:, :].permute(0, 2, 1) # (B, V, T_truncated)
        
        #    b. 计算需要填充的长度
        #       注意：这里使用 config.context_length 来确保总是填充到小波模块初始化的固定长度
        padding_needed = self.config.context_length - wavelet_input_truncated.shape[2]
        
        #    c. 如果需要，执行右侧零填充
        if padding_needed > 0:
            # F.pad 的格式是 (pad_left, pad_right)，我们只在右边填充
            wavelet_input_padded = F.pad(wavelet_input_truncated, (0, padding_needed), "constant", 0)
        else:
            wavelet_input_padded = wavelet_input_truncated
        # =========================================================

        # ========= 【修改部分】: 使用新创建的变量 =========
        # 频域分支
        # 原代码: wavelet_input = past_values.permute(0, 2, 1)
        # 现在 wavelet_input_padded 已经准备好了
        wavelet_embedding = self.wavelet_analyzer(wavelet_input_padded)

        # 时域分支 (Scaling and Patching)
        # 原代码: scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
        # 修改为使用截断后的变量
        scaled_past_values, loc, scale = self.scaler(past_values_truncated, past_observed_mask_truncated)
        # ======================================================
        
        
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

        # 4. UNet Decoder
        decoder_outputs_list, decoder_moe_loss = self.decoder(
            encoder_output,
            skip_connections,
            output_attentions=output_attentions,
            channel_attention_mask=channel_attention_mask,
            linear_attn=linear_attn,
        )
        
        # 5. Prediction Head
        y_hat = self.head(decoder_outputs_list, wavelet_embedding=wavelet_embedding)
        
        loss_val = None
        if future_values is not None:
            # 初始化各个损失分量
            prediction_loss = torch.tensor(0.0, device=past_values.device)
            mmd_loss = torch.tensor(0.0, device=past_values.device)

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
            
            # 计算总的 MoE 负载均衡损失
            total_moe_loss = encoder_moe_loss + decoder_moe_loss

#             # ====================【新增功能】打印当前Batch的各项损失====================
#             # 使用 .item() 获取tensor的标量值，用于打印，避免占用GPU显存
#             print(
#                 f"\n--- Batch Losses --- \n"
#                 f"  - Prediction Loss : {prediction_loss.item():.6f} (weight: 1.0)\n"
#                 f"  - MMD Loss        : {mmd_loss.item():.6f} (weight: {self.mmd_loss_coeff})\n"
#                 f"  - MoE Loss        : {total_moe_loss.item():.6f} (weight: {self.load_balance_coeff})\n"
#                 f"--------------------"
#             )
#             # ========================================================================

            # 组合所有损失分量，计算最终的总损失
            loss_val = (
                prediction_loss 
                + self.mmd_loss_coeff * mmd_loss 
                + self.load_balance_coeff * total_moe_loss
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