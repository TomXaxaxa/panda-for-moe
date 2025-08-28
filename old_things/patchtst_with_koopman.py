"""Exposed PatchTST model, taken from HuggingFace transformers"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import math
import torch
import torch.nn as nn
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

# ==========================================================================================
# ================================ 新增 Koopa 核心模块 =======================================
# ==========================================================================================

# 模块1: 动态傅里叶滤波器
class FourierFilter(nn.Module):
    """
    根据您的要求 [将模型分解为低频部分和高频部分]，实现动态傅里叶滤波器。
    它会根据 [能量占比的方式选取低频谱] [要求 1]，对每个批次的数据进行实时划分。
    """
    def __init__(self, energy_percent=0.8):
        super(FourierFilter, self).__init__()
        if not (0 < energy_percent < 1):
            raise ValueError("energy_percent must be between 0 and 1.")
        self.energy_percent = energy_percent

    def forward(self, x):
        # x shape: [Batch, Sequence Length, Channels]
        # 傅里叶变换和能量计算
        xf = torch.fft.rfft(x, dim=1)
        energies = torch.abs(xf) ** 2

        # 排序能量并计算累积能量，以确定低频分量的索引
        sorted_energies, sorted_indices = torch.sort(energies, dim=1, descending=True)
        total_energy_per_series = torch.sum(energies, dim=1, keepdim=True)
        cumulative_energies = torch.cumsum(sorted_energies, dim=1)
        
        # 找到达到能量阈值的频率数量 k
        k = (cumulative_energies > total_energy_per_series * self.energy_percent).float().argmax(dim=1) + 1

        # 创建动态掩码，Koopa论文的逻辑是掩盖掉低频部分来得到高频部分
        num_frequencies = sorted_indices.shape[1]
        # arange_f: [1, F, 1]
        arange_f = torch.arange(num_frequencies, device=x.device).view(1, -1, 1)

        # rank_tensor: [B, F, C] 表示每个频率的能量排名
        rank_tensor = torch.empty_like(sorted_indices)
        rank_tensor.scatter_(1, sorted_indices, arange_f.expand_as(sorted_indices))

        # is_low_freq: [B, F, C] 布尔张量，能量排在k之前的为True
        # k.unsqueeze(1): [B, 1, C]
        is_low_freq = rank_tensor < k.unsqueeze(1)

        # 创建掩码，将低频部分置为0
        mask = torch.ones_like(xf, device=x.device)
        mask[is_low_freq] = 0
        
        # 应用掩码分离高低频
        # x_var 是高频部分 (time-variant)
        x_var = torch.fft.irfft(xf * mask, n=x.shape[1], dim=1)
        # x_inv 是低频部分 (time-invariant)，通过从原始信号中减去高频部分得到
        x_inv = x - x_var
        
        return x_var, x_inv

# 模块2: MLP (多层感知机)
class MLP(nn.Module):
    """
    通用MLP模块，用于 [高频部分的Koopman网络实现] [要求 2] 中的编码器和解码器。
    """
    def __init__(self, f_in, f_out, hidden_dim=128, hidden_layers=2, dropout=0.05, activation='tanh'):
        super(MLP, self).__init__()
        self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        layers = [nn.Linear(f_in, hidden_dim), self.activation, nn.Dropout(dropout)]
        for _ in range(hidden_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), self.activation, nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# 模块3: Koopman预测层 (KPLayer)
class KPLayer(nn.Module):
    """
    Koopman预测层，使用Moore-Penrose伪逆计算Koopman矩阵，
    并实现了您要求的 [有界性约束] [要求 3]。
    """
    def __init__(self):
        super(KPLayer, self).__init__()
        self.K = None

    def forward(self, z, pred_len=1):
        B, input_len, E = z.shape
        x, y = z[:, :-1], z[:, 1:]
        
        # 使用伪逆求解Koopman矩阵 K
        self.K = torch.linalg.lstsq(x, y).solution
        
        # --- 核心修改：实现 [有界性约束] [要求 3] ---
        # 计算特征值并进行缩放，确保算子稳定
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(self.K)
            max_abs_eig = torch.abs(eigenvalues).max(dim=-1).values
            scaling_factor = torch.clamp(max_abs_eig, min=1.0).unsqueeze(-1).unsqueeze(-1)
        self.K = self.K / scaling_factor
        # --- 修改结束 ---

        # 使用约束后的 K 进行预测
        z_pred = torch.bmm(z[:, -1:], self.K)
        z_preds = [z_pred]
        for _ in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)
        
        return torch.cat(z_preds, dim=1)

# 模块4: 时间可变 Koopman 预测器 (TimeVarKP)
class TimeVarKP(nn.Module):
    """
    实现了您要求的 [高频部分的Koopman网络实现] [要求 2]，
    参考 "Koopa" 的 Time-Variant KP 实现。
    """
    def __init__(self, enc_in, input_len, pred_len, seg_len, dynamic_dim, encoder, decoder):
        super(TimeVarKP, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.encoder, self.decoder = encoder, decoder
        self.freq = math.ceil(self.input_len / self.seg_len)
        self.step = math.ceil(self.pred_len / self.seg_len)
        self.padding_len = self.seg_len * self.freq - self.input_len
        self.dynamics = KPLayer()

    def forward(self, x):
        B, L, C = x.shape
        res = torch.cat((x[:, L-self.padding_len:, :], x), dim=1)
        res = res.chunk(self.freq, dim=1)
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)
        
        res_encoded = self.encoder(res)
        x_pred_encoded = self.dynamics(res_encoded, self.step)
        
        x_pred = self.decoder(x_pred_encoded)
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]
        
        # Koopa原文返回了backcast和forecast，我们这里只关心forecast
        return x_pred

# ================================ 新增模块结束 =========================================


@dataclass
class CompletionsPatchTSTOutput(ModelOutput):
    completions: torch.FloatTensor
    patched_past_values: Optional[torch.FloatTensor] = None
    mask: Optional[torch.FloatTensor] = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None


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
        embed_dim: int,
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
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
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

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

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

    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.channel_attention = config.channel_attention
        # Multi-Head attention
        self.temporal_self_attn = PatchTSTRopeAttention(
            embed_dim=config.d_model,
            num_heads=config.num_attention_heads,
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
                embed_dim=config.d_model,
                num_heads=config.num_attention_heads,
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
            self.norm_sublayer1 = PatchTSTRMSNorm(config.d_model, config.norm_eps)
        elif config.norm_type == "layernorm":
            self.norm_sublayer1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        elif config.norm_type == "dyt":
            self.norm_sublayer1 = DyT(config.d_model)
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
                self.norm_sublayer2 = PatchTSTRMSNorm(config.d_model, config.norm_eps)
            elif config.norm_type == "layernorm":
                self.norm_sublayer2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            elif config.norm_type == "dyt":
                self.norm_sublayer2 = DyT(config.d_model)
            else:
                raise ValueError(
                    f"{config.norm_type} is not a supported norm layer type."
                )

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim, bias=config.bias),
            ACT2CLS[config.activation_function](),
            nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
            nn.Linear(config.ffn_dim, config.d_model, bias=config.bias),
        )

        # Add & Norm of sublayer 3
        self.dropout_path3 = (
            nn.Dropout(config.path_dropout)
            if config.path_dropout > 0
            else nn.Identity()
        )
        if config.norm_type == "rmsnorm":
            self.norm_sublayer3 = PatchTSTRMSNorm(config.d_model, config.norm_eps)
        elif config.norm_type == "layernorm":
            self.norm_sublayer3 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        elif config.norm_type == "dyt":
            self.norm_sublayer3 = DyT(config.d_model)
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
        if self.pre_norm:
            ## Norm and Position-wise Feed-Forward and Add residual connection
            # Add: residual connection with residual dropout
            hidden_state = hidden_state + self.dropout_path3(
                self.ff(self.norm_sublayer3(hidden_state))
            )
        else:
            ## Position-wise Feed-Forward and Add residual connection and Norm
            # Add: residual connection with residual dropout
            hidden_state = self.norm_sublayer3(
                hidden_state + self.dropout_path3(self.ff(hidden_state))
            )

        # [bs x num_channels x sequence_length x d_model]
        hidden_state = hidden_state.reshape(
            batch_size, num_input_channels, sequence_length, d_model
        )

        outputs = (hidden_state,)
        if output_attentions:
            outputs += (
                (attn_weights, channel_attn_weights)
                if self.channel_attention
                else (attn_weights,)
            )

        return outputs


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


class PatchTSTForPrediction(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        # --- 1. 硬编码新增的超参数 [要求 4] ---
        # 您可以在这里修改这些值进行实验
        config.energy_percent = 0.8  # 能量占比阈值
        config.dynamic_dim = 128     # Koopman嵌入维度
        config.seg_len = 24          # Koopman分段长度
        config.hidden_dim = 128      # MLP隐藏层维度
        config.hidden_layers = 2     # MLP隐藏层数量
        config.num_input_channels = 3
        
        self.fusion_weight_raw = nn.Parameter(torch.tensor([0.0]))
        
        # --- 2. 实例化混合模型所需的所有组件 ---
        
        # 实例化动态傅里叶滤波器 [要求 1]
        self.fourier_filter = FourierFilter(energy_percent=config.energy_percent)
        
        # 实例化高频路径的Koopman模块 [要求 2]
        # 现在，编码器和解码器只处理单通道数据 (通道维度为1)
        # f_in 和 f_out 不再乘以 config.num_input_channels
        koopman_encoder = MLP(
            f_in=config.seg_len * config.num_input_channels, 
            f_out=config.dynamic_dim, activation='tanh',
            hidden_dim=config.hidden_dim, hidden_layers=config.hidden_layers)
        koopman_decoder = MLP(
            f_in=config.dynamic_dim, 
            f_out=config.seg_len * config.num_input_channels, activation='tanh',
            hidden_dim=config.hidden_dim, hidden_layers=config.hidden_layers)
        self.time_var_kp = TimeVarKP(
            enc_in=config.num_input_channels, input_len=config.context_length,
            pred_len=config.prediction_length, seg_len=config.seg_len,
            dynamic_dim=config.dynamic_dim, encoder=koopman_encoder,
            decoder=koopman_decoder)
                                     
        # 实例化低频路径的PatchTST模型
        self.patchtst_model = PatchTSTModel(config)
        # 为低频路径创建一个简单的线性预测头
        self.low_freq_prediction_head = nn.Linear(config.d_model, config.prediction_length)

        # 定义损失函数
        if config.loss == "mse":
            self.loss = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Only MSE loss is supported for this hybrid model.")
            
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        # 其他参数可以保留，但在此实现中未使用
        **kwargs
    ) -> Union[Tuple, PatchTSTForPredictionOutput]:

        # --- 3. 实现双路并行预测的 forward 逻辑 ---

        # 步骤 A: 数据标准化
        loc = past_values.mean(dim=1, keepdim=True)
        scale = torch.sqrt(torch.var(past_values, dim=1, keepdim=True, unbiased=False) + 1e-5)
        scaled_past_values = (past_values - loc) / scale
        
        # 步骤 B: 傅里叶动态解耦 [要求 1]
        # x_var 是高频, x_inv 是低频
        x_var, x_inv = self.fourier_filter(scaled_past_values)
        
        # 步骤 C: 低频路径 (由原PatchTST处理) [要求 1]
        # 内部处理 patch 化和 embedding
        patched_x_inv = self.patchtst_model.patchifier(x_inv)
        encoder_output = self.patchtst_model.encoder(patch_input=patched_x_inv)
        low_freq_embedding = encoder_output.last_hidden_state
        # 使用 mean pooling 和线性头进行预测
        low_freq_pooled = low_freq_embedding.mean(dim=2)
        low_freq_pred_scaled = self.low_freq_prediction_head(low_freq_pooled).transpose(2, 1)
        
        # 步骤 D: 高频路径 (由Koopman网络处理) [要求 2]
        high_freq_pred_scaled = self.time_var_kp(x_var)
        
        # 步骤 E: 融合预测结果并逆标准化
        fusion_weight = torch.sigmoid(self.fusion_weight_raw)
        y_hat_scaled = (fusion_weight * low_freq_pred_scaled) + ((1 - fusion_weight) * high_freq_pred_scaled)
        y_hat = y_hat_scaled * scale + loc
        
        # 步骤 F: 计算损失
        loss_val = None
        if future_values is not None:
            loss_val = self.loss(y_hat, future_values)
        
        return PatchTSTForPredictionOutput(
            loss=loss_val,
            prediction_outputs=y_hat
        )

    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None, # 注意：此参数在新模型中未使用，但保留以兼容接口
        channel_attention_mask: Optional[torch.Tensor] = None # 注意：此参数在新模型中未使用，但保留以兼容接口
    ) -> SamplePatchTSTOutput:
        """
        为我们新的 Koopa-Transformer 混合模型生成确定性预测序列。
        此版本已根据修改后的 forward 方法进行了简化。
        """
        # 将模型设置为评估模式
        self.eval()
        
        # 在 no_grad 上下文中执行，因为我们只是在进行推理，不需要计算梯度
        with torch.no_grad():
            # 调用我们重写过的 forward 方法进行预测
            # future_values 设为 None，因为这是推理阶段
            outputs = self.forward(
                past_values=past_values,
                future_values=None,
            )

        # 从输出中提取预测结果
        # outputs.prediction_outputs 的形状是 [bs, forecast_len, num_channels]
        prediction = outputs.prediction_outputs
        
        # generate 函数期望的输出形状是 [bs, num_samples, forecast_len, num_channels]
        # 因为我们是确定性预测，所以 num_samples = 1
        samples = prediction.unsqueeze(1)

        return SamplePatchTSTOutput(sequences=samples)