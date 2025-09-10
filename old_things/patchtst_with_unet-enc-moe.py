"""Exposed PatchTST model, taken from HuggingFace transformers"""

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

    Implemented with p-rotary positional embeddings
    """

    def __init__(
        self,
        # embed_dim: int,
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
        linear_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]  # type: ignore
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)  # type: ignore
            value_states = torch.cat([past_key_value[1], value_states], dim=2)  # type: ignore
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)  # type: ignore

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)
        src_len = key_states.size(1)

        # apply rotary positional embeddings
        if self.use_rope:
            position_ids = self.get_seq_pos(
                src_len, key_states.device, key_states.dtype
            )
            key_states, query_states = apply_p_rope_to_qk(
                key_states,
                query_states,
                position_ids,
                self.head_dim,
                self.max_wavelength,
                self.rope_percent,
            )

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            ) + attention_mask.to(attn_weights.device)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not linear_attn:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

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
        
        # 获取 Encoder 输出的最高维度
        encoder_bottleneck_dim = config.d_model

        # current_decoder_dim 用于追踪 decoder 中数据流的维度，初始值为 bottleneck 的维度
        current_decoder_dim = encoder_bottleneck_dim

        for i, depth in enumerate(reversed_depths):
            
            # 目标维度 (上采样后，或与 skip connection 融合后的维度)
            target_dim = encoder_bottleneck_dim // (2 ** i)
            current_num_heads = reversed_num_heads[i]
            
            # Upsample 模块
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
        """
        这个 forward 方法经过修改，以支持多阶段输出和MoE损失收集。
        
        Args:
            hidden_state (torch.Tensor): 来自Encoder bottleneck的输出.
            skip_connections (list): 来自Encoder各个阶段的跳跃连接列表.
        
        Returns:
            tuple[list, torch.Tensor]:
                - all_stage_outputs (list): 一个列表，包含解码器每个阶段的输出张量。
                - total_moe_loss (torch.Tensor): 从所有解码器层收集到的总负载均衡损失。
        """
        reversed_skips = list(reversed(skip_connections))
        
        # 【修改点 1】: 初始化用于存储多阶段输出的列表和累加MoE损失的变量
        all_stage_outputs = []
        total_moe_loss = torch.tensor(0.0, device=hidden_state.device)

        for i, stage in enumerate(self.stages):
            # Bottleneck 层 (i=0) 直接通过, 不进行上采样和融合
            # 后续层 (i>0) 先进行上采样和融合
            if stage["upsample"] is not None:
                # 1. 上采样
                batch_size, num_channels, num_patches, d_model = hidden_state.shape
                hidden_state_reshaped = hidden_state.view(batch_size * num_channels, num_patches, d_model)
                hidden_state_upsampled = stage["upsample"](hidden_state_reshaped)
                
                # 2. 处理跳跃连接
                skip = reversed_skips[i]
                bs_s, nc_s, np_s, nd_s = skip.shape
                skip_reshaped = skip.view(bs_s * nc_s, np_s, nd_s)
                for processor in stage["skip_processor"]:
                    skip_reshaped = processor(skip_reshaped)
                
                # 3. 融合 (处理可能因padding导致的的序列长度不匹配问题)
                if hidden_state_upsampled.shape[1] != skip_reshaped.shape[1]:
                    diff = hidden_state_upsampled.shape[1] - skip_reshaped.shape[1]
                    skip_reshaped = F.pad(skip_reshaped, (0, 0, 0, diff))
                
                hidden_state = hidden_state_upsampled + skip_reshaped
                
                # 恢复形状
                num_patches, d_model = hidden_state.shape[1], hidden_state.shape[2]
                hidden_state = hidden_state.view(batch_size, num_channels, num_patches, d_model)
        
            # 4. 特征提取 (当前阶段的 Transformer Layers)
            for layer in stage["layers"]:
                layer_outputs = layer(
                    hidden_state,
                    output_attentions=output_attentions,
                    channel_attention_mask=channel_attention_mask,
                    linear_attn=linear_attn,
                )
                hidden_state = layer_outputs[0]
                
                # 【修改点 2】: 累加从每个层返回的MoE损失
                # layer_outputs[-1] 是我们约定好的moe_loss
                if isinstance(layer_outputs[-1], torch.Tensor):
                    total_moe_loss += layer_outputs[-1]

            # 【修改点 3】: 将当前阶段处理后的输出添加到列表中
            all_stage_outputs.append(hidden_state)
    
        # 【修改点 4】: 返回包含所有阶段输出的列表和总的MoE损失
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

class MultiStagePredictionHead(nn.Module):
    """
    一个新的预测头，用于处理来自U-Net Decoder多个阶段的输出。
    它会对每个阶段的输出进行 mean pooling，然后在特征维度上拼接，
    最后通过一个线性层进行预测。
    """
    def __init__(self, config: PatchTSTConfig, depths: list, distribution_output=None):
        super().__init__()
        self.config = config
        
        # 计算每个decoder阶段输出的特征维度
        # Encoder的输出维度是 d_model * 2^(len(depths)-1)
        encoder_bottleneck_dim = config.d_model * (2 ** (len(depths) - 1))
        # Decoder 阶段的维度是倒序的
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
            # pooled_embedding: [bs x num_channels x d_model_i]
            pooled_embedding = embedding.mean(dim=2)
            pooled_outputs.append(pooled_embedding)
            
        # 在特征维度 (最后一个维度) 上拼接所有池化后的结果
        # concatenated_embedding: [bs x num_channels x total_concatenated_dim]
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

        # =========== U-Net超参数 ===========
        # 你可以根据需求调整这些参数
        self.depths = [1, 1, 1, 1]  # 每个阶段的Transformer层数
        self.skip_connections_depths = [2, 2, 2, 0] # 每个跳跃连接上的ConvNeXt层数
        self.num_heads_list = [3, 6, 12, 24] 
        
        # 【修改点 1】: 新增MoE负载均衡损失的权重系数
        self.load_balance_coeff = 0.01

        # =========== 基础模块 ===========
        self.scaler = PatchTSTScaler(config)
        self.patchifier = PatchTSTPatchify(config)
        
        if config.use_dynamics_embedding:
            self.encoder_embedding = PatchTSTKernelEmbedding(config)
        else:
            self.encoder_embedding = PatchTSTEmbedding(config)
        
        # =========== U-Net 编码器和解码器 ===========
        # 编码器
        original_d_model = config.d_model
        self.encoder = PatchTSTUNetEncoder(config, depths=self.depths, num_heads_list=self.num_heads_list)
        
        # 为解码器动态设置d_model，以匹配编码器最深层的维度
        config.d_model = original_d_model * (2 ** (len(self.depths) - 1))
        self.decoder = PatchTSTUNetDecoder(config, depths=self.depths, skip_connections_depths=self.skip_connections_depths, num_heads_list=self.num_heads_list)
        
        # 恢复config中的d_model，以防影响其他模块
        config.d_model = original_d_model

        # =========== 概率预测头逻辑 (与原版保持一致) ===========
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

        # 【修改点 2】: 使用新的多阶段预测头 MultiStagePredictionHead
        self.head = MultiStagePredictionHead(config, depths=self.depths, distribution_output=self.distribution_output)

        # =========== 损失函数定义 (与原版保持一致) ===========
        if config.loss == "mse":
            self.loss = nn.MSELoss(reduction="mean")
        elif config.loss == "huber":
            self.loss = nn.HuberLoss(reduction="mean", delta=config.huber_delta)
        
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

        # 1. Scaling and Patching
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
        patched_values = self.patchifier(scaled_past_values)
        
        # 2. Embedding
        embedded_values = self.encoder_embedding(patched_values)
        
        # 【修改点 3】: 修改Encoder和Decoder的调用，以接收MoE损失
        # 3. UNet Encoder - 现在返回三个值
        encoder_output, skip_connections, encoder_moe_loss = self.encoder(
            embedded_values,
            output_attentions=output_attentions,
            channel_attention_mask=channel_attention_mask,
            linear_attn=linear_attn,
        )

        # 4. UNet Decoder - 现在返回输出列表和MoE损失
        decoder_outputs_list, decoder_moe_loss = self.decoder(
            encoder_output,
            skip_connections,
            output_attentions=output_attentions,
            channel_attention_mask=channel_attention_mask,
            linear_attn=linear_attn,
        )
        
        # 5. Prediction Head - 接收解码器输出列表作为输入
        y_hat = self.head(decoder_outputs_list)
        
        # 6. Loss Calculation
        loss_val = None
        if future_values is not None:
            # 6.1 计算主任务的预测损失 (与原版逻辑一致)
            if self.distribution_output:
                # 概率预测损失
                distribution = self.distribution_output.distribution(y_hat, loc=loc, scale=scale)
                loss_val = nll(distribution, future_values)
                loss_val = weighted_average(loss_val)
            else:
                # 点预测损失
                y_hat_out = y_hat * scale + loc
                loss_val = self.loss(y_hat_out, future_values)
            
            # 【修改点 4】: 将MoE的负载均衡损失加入到总损失中
            total_moe_loss = encoder_moe_loss + decoder_moe_loss
            loss_val = loss_val + self.load_balance_coeff * total_moe_loss

        # 根据是否有分布输出，确定最终的预测值
        if self.distribution_output:
            y_hat_out = y_hat
        else:
            # 点预测需要反归一化
            y_hat_out = y_hat * scale + loc
        
        if not return_dict:
            outputs = (y_hat_out, loc, scale)
            return (loss_val,) + outputs if loss_val is not None else outputs
        
        return PatchTSTForPredictionOutput(
            loss=loss_val,
            prediction_outputs=y_hat_out,
            hidden_states=None, # U-Net结构不直接返回所有层的hidden_states
            attentions=None,    # U-Net结构不直接返回所有层的attentions
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
        从带有概率分布头的模型中生成样本序列 (此方法无需修改)。
        """
        # get number of samples
        num_parallel_samples = self.config.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            future_values=None, # no ground truth available for generation
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
            # for point forecasts, we just add a sample dimension
            samples = outputs.prediction_outputs.unsqueeze(1)

        return SamplePatchTSTOutput(sequences=samples)