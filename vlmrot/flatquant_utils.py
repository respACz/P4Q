import logging
import math
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vlmrot import model_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
    )
except Exception:  # pragma: no cover - depends on installed transformers
    llama_apply_rotary_pos_emb = None

try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        apply_multimodal_rotary_pos_emb as qwen2vl_apply_rotary_pos_emb,
    )
except Exception:  # pragma: no cover - depends on installed transformers
    qwen2vl_apply_rotary_pos_emb = None

try:
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
except Exception:  # pragma: no cover - depends on installed transformers
    repeat_kv = None


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x


def get_qmin_qmax(bits, sym):
    if sym:
        q_max = torch.tensor(2 ** (bits - 1) - 1)
        q_min = -q_max - 1
    else:
        q_max = torch.tensor(2**bits - 1)
        q_min = torch.tensor(0)
    return q_max, q_min


def sym_quant_dequant(x, scale, maxq):
    q = torch.clamp(round_ste(x / scale.to(x.device)), -(maxq + 1), maxq)
    return q * scale.to(x.device)


def asym_quant_dequant(x, scale, zero, maxq):
    q = torch.clamp(round_ste(x / scale.to(x.device)) + zero.to(x.device), 0, maxq)
    return scale.to(x.device) * (q - zero.to(x.device))


class ActivationQuantizer(nn.Module):
    def __init__(self, bits, sym=False, lac=False, groupsize=-1, clip_ratio=None):
        super().__init__()
        if groupsize > 0:
            raise NotImplementedError("FlatQuant activation groupsize is not supported yet")
        self.bits = bits
        self.q_max, self.q_min = get_qmin_qmax(bits, sym)
        self.sym = sym
        self.lac = lac
        self.groupsize = groupsize
        self._clip_ratio = clip_ratio
        self.enable = True
        if self.lac:
            init_value = 4.0
            self.sigmoid = nn.Sigmoid()
            self.clip_factor_a_max = nn.Parameter(torch.ones(1) * init_value)
            self.clip_factor_a_min = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x):
        if self.bits == 16 or not self.enable:
            return x
        x_dtype = x.dtype
        scale, zero = self.get_scale_zero(x)
        if self.sym:
            return sym_quant_dequant(x, scale, self.q_max.to(x.device)).to(x_dtype)
        return asym_quant_dequant(x, scale, zero, self.q_max.to(x.device)).to(x_dtype)

    def get_scale_zero(self, x):
        q_max = self.q_max.to(x.device)
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-1])
        xmax = reshaped_x.amax(1, keepdim=True)
        xmin = reshaped_x.amin(1, keepdim=True)
        zeros = torch.zeros_like(xmax)
        xmax = torch.maximum(xmax, zeros)
        xmin = torch.minimum(xmin, zeros)
        if self.lac:
            xmax = xmax * self.sigmoid(self.clip_factor_a_max)
            xmin = xmin * self.sigmoid(self.clip_factor_a_min)
        elif self._clip_ratio is not None:
            xmax = xmax * self._clip_ratio
            xmin = xmin * self._clip_ratio

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            empty = xmax == 0
            scale = xmax / q_max
            scale[empty] = 1
            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = torch.zeros_like(scale)
            return scale, zero

        empty = (xmin == 0) & (xmax == 0)
        xmin[empty] = -1
        xmax[empty] = 1
        scale = (xmax - xmin) / q_max
        zero = torch.round(-xmin / scale)
        scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
        zero = zero.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
        return scale, zero


class WeightQuantizer(nn.Module):
    def __init__(self, shape=1):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))
        self.enable = True

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.maxq = torch.tensor(2 ** (bits - 1) - 1 if sym else 2**bits - 1)

    def find_params(self, x):
        if self.bits == 16 or not self.enable:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)
        shape = x.shape
        work = x.flatten(1) if self.perchannel else x.flatten().unsqueeze(0)

        zeros = torch.zeros(work.shape[0], device=dev)
        xmin = torch.minimum(work.min(1)[0], zeros)
        xmax = torch.maximum(work.max(1)[0], zeros)
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            empty = (xmin == 0) & (xmax == 0)
            xmin[empty] = -1
            xmax[empty] = 1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if not self.perchannel:
            self.scale = self.scale.repeat(shape[0])
            self.zero = self.zero.repeat(shape[0])
        self.scale = self.scale.reshape([-1] + [1] * (len(shape) - 1))
        self.zero = self.zero.reshape([-1] + [1] * (len(shape) - 1))

    def quantize(self, x):
        x_dtype = x.dtype
        if not self.ready() or self.bits == 16 or not self.enable:
            return x
        if self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq.to(x.device)).to(x_dtype)
        return asym_quant_dequant(x, self.scale, self.zero, self.maxq.to(x.device)).to(x_dtype)

    def forward(self, x):
        return self.quantize(x)

    def ready(self):
        return torch.all(self.scale != 0)


def get_decompose_dim(n):
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            return a - b, a + b
        a += 1


def get_init_scale(w_smax, x_smax, alpha=0.5):
    return (w_smax.pow(1 - alpha) / x_smax.pow(alpha)).clamp(min=1e-5)


def get_init_weight(dim):
    matrix = torch.randn(dim, dim, dtype=torch.float32)
    q, r = torch.linalg.qr(matrix)
    return q @ torch.diag(torch.sign(torch.diag(r)))


def get_inverse(matrix):
    dtype = matrix.dtype
    return matrix.double().inverse().to(dtype)


def kronecker_matmul(x, had_l, had_r):
    init_shape = x.shape
    x = x.reshape(-1, had_l.shape[0], had_r.shape[0])
    x = torch.matmul(x, had_r)
    x = torch.matmul(had_l.T, x)
    return x.reshape(init_shape)


class SVDSingleTransMatrix(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear_u = nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear_u.weight.data = get_init_weight(size).to(self.linear_u.weight)
        self.linear_u = nn.utils.parametrizations.orthogonal(
            self.linear_u, orthogonal_map="cayley", use_trivialization=False
        )
        self.linear_v = nn.Linear(size, size, bias=False, dtype=torch.float32)
        self.linear_v.weight.data = get_init_weight(size).to(self.linear_v.weight)
        self.linear_v = nn.utils.parametrizations.orthogonal(
            self.linear_v, orthogonal_map="cayley", use_trivialization=False
        )
        self.linear_diag = nn.Parameter(torch.ones(size, dtype=torch.float32))
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        init_shape = inp.shape
        matrix = self.get_matrix(inv_t=inv_t).to(inp)
        return inp.reshape(-1, matrix.shape[0]).matmul(matrix).reshape(init_shape)

    def get_matrix(self, inv_t=False):
        if self._eval_mode:
            return self.matrix_inv_t if inv_t else self.matrix
        diag = 1 / self.linear_diag if inv_t else self.linear_diag
        return self.linear_u.weight @ torch.diag(diag) @ self.linear_v.weight.T

    def to_eval_mode(self):
        if self._eval_mode:
            return
        self.matrix = nn.Parameter(self.get_matrix(inv_t=False), requires_grad=False)
        self.matrix_inv_t = nn.Parameter(self.get_matrix(inv_t=True), requires_grad=False)
        del self.linear_u, self.linear_v, self.linear_diag
        self._eval_mode = True


class SVDDecomposeTransMatrix(nn.Module):
    def __init__(self, left_size, right_size, add_diag=False, diag_init_para=None):
        super().__init__()
        self.linear_u_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.linear_u_left.weight.data = get_init_weight(left_size).to(self.linear_u_left.weight)
        self.linear_u_left = nn.utils.parametrizations.orthogonal(
            self.linear_u_left, orthogonal_map="cayley", use_trivialization=False
        )
        self.linear_v_left = nn.Linear(left_size, left_size, bias=False, dtype=torch.float32)
        self.linear_v_left.weight.data = get_init_weight(left_size).to(self.linear_v_left.weight)
        self.linear_v_left = nn.utils.parametrizations.orthogonal(
            self.linear_v_left, orthogonal_map="cayley", use_trivialization=False
        )
        self.linear_diag_left = nn.Parameter(torch.ones(left_size, dtype=torch.float32))

        self.linear_u_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.linear_u_right.weight.data = get_init_weight(right_size).to(self.linear_u_right.weight)
        self.linear_u_right = nn.utils.parametrizations.orthogonal(
            self.linear_u_right, orthogonal_map="cayley", use_trivialization=False
        )
        self.linear_v_right = nn.Linear(right_size, right_size, bias=False, dtype=torch.float32)
        self.linear_v_right.weight.data = get_init_weight(right_size).to(self.linear_v_right.weight)
        self.linear_v_right = nn.utils.parametrizations.orthogonal(
            self.linear_v_right, orthogonal_map="cayley", use_trivialization=False
        )
        self.linear_diag_right = nn.Parameter(torch.ones(right_size, dtype=torch.float32))
        self.add_diag = add_diag
        self.use_diag = True
        if self.add_diag:
            diag = torch.ones(left_size * right_size, dtype=torch.float32)
            self.diag_scale = nn.Parameter(diag if diag_init_para is None else diag_init_para)
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        if self.add_diag and self.use_diag:
            inp = inp / self.diag_scale.to(inp) if inv_t else inp * self.diag_scale.to(inp)
        if self._eval_mode:
            left = self.matrix_left_inv if inv_t else self.matrix_left
            right = self.matrix_right_inv if inv_t else self.matrix_right
            return kronecker_matmul(inp, left.to(inp), right.to(inp))
        diag_left = 1 / self.linear_diag_left if inv_t else self.linear_diag_left
        diag_right = 1 / self.linear_diag_right if inv_t else self.linear_diag_right
        left = self.linear_u_left.weight @ torch.diag(diag_left) @ self.linear_v_left.weight.T
        right = self.linear_u_right.weight @ torch.diag(diag_right) @ self.linear_v_right.weight.T
        return kronecker_matmul(inp, left.to(inp), right.to(inp))

    def to_eval_mode(self):
        if self._eval_mode:
            return
        self.matrix_left = nn.Parameter(
            self.linear_u_left.weight @ torch.diag(self.linear_diag_left) @ self.linear_v_left.weight.T,
            requires_grad=False,
        )
        self.matrix_right = nn.Parameter(
            self.linear_u_right.weight @ torch.diag(self.linear_diag_right) @ self.linear_v_right.weight.T,
            requires_grad=False,
        )
        self.matrix_left_inv = nn.Parameter(
            self.linear_u_left.weight @ torch.diag(1 / self.linear_diag_left) @ self.linear_v_left.weight.T,
            requires_grad=False,
        )
        self.matrix_right_inv = nn.Parameter(
            self.linear_u_right.weight @ torch.diag(1 / self.linear_diag_right) @ self.linear_v_right.weight.T,
            requires_grad=False,
        )
        del self.linear_u_left, self.linear_v_left, self.linear_diag_left
        del self.linear_u_right, self.linear_v_right, self.linear_diag_right
        self._eval_mode = True


class InvSingleTransMatrix(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False)
        self.linear.weight.data = get_init_weight(size).to(self.linear.weight)
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        init_shape = inp.shape
        matrix = self.get_matrix(inv_t=inv_t).to(inp)
        return inp.reshape(-1, matrix.shape[0]).matmul(matrix).reshape(init_shape)

    def get_matrix(self, inv_t=False):
        if self._eval_mode:
            return self.matrix_inv_t if inv_t else self.matrix
        return get_inverse(self.linear.weight).T if inv_t else self.linear.weight

    def to_eval_mode(self):
        if self._eval_mode:
            return
        self.matrix = nn.Parameter(self.linear.weight, requires_grad=False)
        self.matrix_inv_t = nn.Parameter(get_inverse(self.linear.weight).T, requires_grad=False)
        self._eval_mode = True


class InvDecomposeTransMatrix(nn.Module):
    def __init__(self, left_size, right_size, add_diag=False, diag_init_para=None):
        super().__init__()
        self.linear_left = nn.Linear(left_size, left_size, bias=False)
        self.linear_left.weight.data = get_init_weight(left_size).to(self.linear_left.weight)
        self.linear_right = nn.Linear(right_size, right_size, bias=False)
        self.linear_right.weight.data = get_init_weight(right_size).to(self.linear_right.weight)
        self.add_diag = add_diag
        self.use_diag = True
        if self.add_diag:
            diag = torch.ones(left_size * right_size)
            self.diag_scale = nn.Parameter(diag if diag_init_para is None else diag_init_para)
        self._eval_mode = False

    def forward(self, inp, inv_t=False):
        if self.add_diag and self.use_diag:
            inp = inp / self.diag_scale.to(inp) if inv_t else inp * self.diag_scale.to(inp)
        if self._eval_mode:
            left = self.matrix_left_inv if inv_t else self.matrix_left
            right = self.matrix_right_inv if inv_t else self.matrix_right
        else:
            left = get_inverse(self.linear_left.weight).T if inv_t else self.linear_left.weight
            right = get_inverse(self.linear_right.weight).T if inv_t else self.linear_right.weight
        return kronecker_matmul(inp, left.to(inp), right.to(inp))

    def to_eval_mode(self):
        if self._eval_mode:
            return
        self.matrix_left = nn.Parameter(self.linear_left.weight, requires_grad=False)
        self.matrix_right = nn.Parameter(self.linear_right.weight, requires_grad=False)
        self.matrix_left_inv = nn.Parameter(get_inverse(self.linear_left.weight).T, requires_grad=False)
        self.matrix_right_inv = nn.Parameter(get_inverse(self.linear_right.weight).T, requires_grad=False)
        del self.linear_left, self.linear_right
        self._eval_mode = True


class FlatQuantizedLinear(nn.Module):
    def __init__(self, args, linear: nn.Linear):
        super().__init__()
        self.args = args
        self.linear = linear
        self.weight_quantizer = WeightQuantizer()
        self.weight_quantizer.configure(args.w_bits, perchannel=True, sym=not args.w_asym, mse=False)
        self.act_quantizer = ActivationQuantizer(
            bits=args.a_bits,
            sym=not args.a_asym,
            lac=args.lac,
            groupsize=args.a_groupsize,
        )
        self.lwc = args.lwc
        if self.lwc:
            init_value = 4.0
            self.clip_factor_w_max = nn.Parameter(torch.ones((linear.weight.shape[0], 1)) * init_value)
            self.clip_factor_w_min = nn.Parameter(torch.ones((linear.weight.shape[0], 1)) * init_value)
            self.sigmoid = nn.Sigmoid()
        self._eval_mode = False

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def apply_wclip(self, weight):
        wmin = weight.min(1, keepdim=True)[0]
        wmax = weight.max(1, keepdim=True)[0]
        clip_min = self.sigmoid(self.clip_factor_w_min).to(weight)
        clip_max = self.sigmoid(self.clip_factor_w_max).to(weight)
        return torch.clamp(
            weight,
            min=wmin * clip_min,
            max=wmax * clip_max,
        )

    def apply_trans(self, weight, qa_trans):
        if isinstance(qa_trans, list):
            return kronecker_matmul(weight, qa_trans[0].to(weight), qa_trans[1].to(weight))
        return qa_trans(weight, inv_t=True)

    def _ori_forward(self, hidden_states):
        return self.linear(hidden_states)

    def forward(self, hidden_states, qa_trans=None, out_trans=None):
        x_dtype = hidden_states.dtype
        if self._eval_mode:
            hidden_states = self.act_quantizer(hidden_states).to(self.linear.weight.dtype)
            return self.linear(hidden_states).to(x_dtype)

        weight = self.linear.weight.data
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self.apply_wclip(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
        self.weight_quantizer.find_params(weight)
        weight = self.weight_quantizer(weight)
        hidden_states = self.act_quantizer(hidden_states).to(weight.dtype)
        bias = out_trans(self.linear.bias.data) if out_trans is not None and self.linear.bias is not None else self.linear.bias
        if bias is not None:
            bias = bias.to(weight.dtype)
        return F.linear(hidden_states, weight, bias).to(x_dtype)

    def reparameterize(self, qa_trans=None, out_trans=None):
        weight = self.linear.weight.data
        ori_dtype = weight.dtype
        weight = weight.to(torch.float64)
        if qa_trans is not None:
            weight = self.apply_trans(weight, qa_trans)
        if self.lwc:
            weight = self.apply_wclip(weight)
        if out_trans is not None:
            weight = out_trans(weight.T).T
            if self.linear.bias is not None:
                self.linear.bias.data = out_trans(self.linear.bias.data)
        self.linear.weight.data = weight.to(ori_dtype)
        self._eval_mode = True


class FlatQuantMLP(nn.Module):
    def __init__(self, args, module):
        super().__init__()
        self.args = args
        self.act_fn = module.act_fn
        self.gate_proj = FlatQuantizedLinear(args, module.gate_proj)
        self.up_proj = FlatQuantizedLinear(args, module.up_proj)
        self.down_proj = FlatQuantizedLinear(args, module.down_proj)
        self._ori_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            self.up_smax = torch.ones_like(self.up_proj.linear.weight.abs().max(dim=0)[0]) * 1e-5
            self.down_smax = torch.ones_like(self.down_proj.linear.weight.abs().max(dim=0)[0]) * 1e-5
        self.add_fq_trans()

    def add_fq_trans(self):
        matrix_cls = InvDecomposeTransMatrix if self.args.direct_inv else SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            up_left, up_right = get_decompose_dim(self.up_proj.linear.weight.shape[1])
            down_left, down_right = get_decompose_dim(self.down_proj.linear.weight.shape[1])
            self.up_gate_trans = matrix_cls(up_left, up_right, add_diag=self.args.add_diag)
            self.down_trans = matrix_cls(down_left, down_right, add_diag=self.args.add_diag)
        else:
            self.up_gate_trans = None
            self.down_trans = None

    def forward(self, x):
        if self._ori_mode:
            if self.diag_init == "sq_style":
                self.up_smax = torch.maximum(
                    self.up_smax.to(x.device),
                    x.reshape(-1, x.shape[-1]).abs().max(0)[0].detach(),
                )
            x_act = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
            if self.diag_init == "sq_style":
                self.down_smax = torch.maximum(
                    self.down_smax.to(x_act.device),
                    x_act.reshape(-1, x_act.shape[-1]).abs().max(0)[0].detach(),
                )
            return self.down_proj._ori_forward(x_act)

        x_ts = self.up_gate_trans(x) if self.up_gate_trans is not None else x
        up_states = self.up_proj(x_ts, qa_trans=self.up_gate_trans)
        gate_states = self.gate_proj(x_ts, qa_trans=self.up_gate_trans)
        x_act = self.act_fn(gate_states) * up_states
        x_ts_2 = self.down_trans(x_act) if self.down_trans is not None else x_act
        return self.down_proj(x_ts_2, qa_trans=self.down_trans)

    def init_diag_scale(self, alpha=0.5):
        if not hasattr(self, "up_smax"):
            return
        upw_smax = torch.cat([self.up_proj.linear.weight, self.gate_proj.linear.weight], dim=0).abs().max(dim=0)[0]
        downw_smax = self.down_proj.linear.weight.abs().max(dim=0)[0]
        if self.up_gate_trans is not None:
            self.up_gate_trans.diag_scale.data = get_init_scale(upw_smax, self.up_smax.to(upw_smax.device), alpha)
        if self.down_trans is not None:
            self.down_trans.diag_scale.data = get_init_scale(downw_smax, self.down_smax.to(downw_smax.device), alpha)
        del self.up_smax, self.down_smax
        self.diag_init = None

    def rep_matrix_only(self):
        if self.up_gate_trans is not None:
            self.up_gate_trans.to_eval_mode()
            self.down_trans.to_eval_mode()

    def reparameterize(self):
        self.rep_matrix_only()
        self.gate_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.up_proj.reparameterize(qa_trans=self.up_gate_trans)
        self.down_proj.reparameterize(qa_trans=self.down_trans)
        if self.up_gate_trans is not None:
            self.up_gate_trans.use_diag = False
        if self.down_trans is not None and self.down_trans.add_diag:
            up_weight = self.up_proj.linear.weight
            ori_dtype = up_weight.dtype
            up_weight = up_weight.to(torch.float64).T.mul(self.down_trans.diag_scale.to(torch.float64)).T
            self.up_proj.linear.weight.data = up_weight.to(ori_dtype)
            self.down_trans.use_diag = False


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(q, k, cos, sin, position_ids, rotary_fn, mrope_section=None):
    if rotary_fn is not None:
        call_args = []
        if mrope_section is not None:
            # Qwen2-VL's fifth argument is mrope_section, not position_ids.
            # Do not probe it with the generic legacy signature because a
            # position tensor can be accepted and then fail inside split().
            call_args.append((q, k, cos, sin, mrope_section))
        elif position_ids is not None:
            call_args.append((q, k, cos, sin, position_ids))
        call_args.append((q, k, cos, sin))
        for args in call_args:
            try:
                return rotary_fn(*args)
            except TypeError:
                continue
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(hidden_states, num_key_value_groups):
    if repeat_kv is not None:
        return repeat_kv(hidden_states, num_key_value_groups)
    if num_key_value_groups == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, num_key_value_groups, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * num_key_value_groups, slen, head_dim)


def _update_cache(cache, key_states, value_states, layer_idx, cache_kwargs):
    try:
        # Transformers 4.52 accepts cache_kwargs as the fourth argument.  It
        # is required by StaticCache for cache_position and by RoPE-aware
        # caches for cos/sin.  Calling the three-argument form first can
        # silently succeed while dropping this information.
        return cache.update(
            key_states, value_states, layer_idx, cache_kwargs
        )
    except TypeError:
        return cache.update(key_states, value_states, layer_idx)


class FlatQuantAttention(nn.Module):
    def __init__(
        self,
        args,
        module,
        rotary_fn=None,
        return_past_key_value=False,
    ):
        super().__init__()
        self.args = args
        self.config = getattr(module, "config", None)
        self.layer_idx = getattr(module, "layer_idx", None)
        self.rotary_emb = getattr(module, "rotary_emb", None)
        self.hidden_size = getattr(module, "hidden_size", getattr(self.config, "hidden_size", None))
        self.num_heads = getattr(module, "num_heads", getattr(module, "num_attention_heads", None))
        if self.num_heads is None:
            self.num_heads = getattr(self.config, "num_attention_heads")
        self.num_key_value_heads = getattr(
            module,
            "num_key_value_heads",
            getattr(self.config, "num_key_value_heads", self.num_heads),
        )
        self.num_key_value_groups = getattr(
            module,
            "num_key_value_groups",
            self.num_heads // self.num_key_value_heads,
        )
        self.head_dim = getattr(module, "head_dim", self.hidden_size // self.num_heads)
        self.scaling = getattr(module, "scaling", self.head_dim**-0.5)
        self.attention_dropout = getattr(module, "attention_dropout", 0.0)
        self.is_causal = getattr(module, "is_causal", True)
        self.attn_implementation = getattr(
            self.config, "_attn_implementation", "sdpa"
        )
        self.rotary_fn = rotary_fn
        self.return_past_key_value = bool(return_past_key_value)
        rope_scaling = getattr(module, "rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = getattr(self.config, "rope_scaling", None)
        self.mrope_section = (
            rope_scaling.get("mrope_section")
            if isinstance(rope_scaling, dict)
            else None
        )
        self.q_proj = FlatQuantizedLinear(args, module.q_proj)
        self.k_proj = FlatQuantizedLinear(args, module.k_proj)
        self.v_proj = FlatQuantizedLinear(args, module.v_proj)
        self.o_proj = FlatQuantizedLinear(args, module.o_proj)
        self._ori_mode = False
        self._eval_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            self.ln_smax = torch.ones_like(self.q_proj.linear.weight.abs().max(dim=0)[0]) * 1e-5
        self.add_fq_trans()
        if args.q_bits < 16:
            self.q_cache_quantizer = ActivationQuantizer(args.q_bits, sym=not args.q_asym, lac=args.lac)
        if args.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(args.k_bits, sym=not args.k_asym, lac=args.lac)
        if args.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(args.v_bits, sym=not args.v_asym, lac=args.lac)

    def add_fq_trans(self):
        single_cls = InvSingleTransMatrix if self.args.direct_inv else SVDSingleTransMatrix
        decompose_cls = InvDecomposeTransMatrix if self.args.direct_inv else SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            left, right = get_decompose_dim(self.q_proj.linear.weight.shape[1])
            self.ln_trans = decompose_cls(left, right, add_diag=self.args.add_diag)
            self.o_trans = single_cls(self.num_heads)
        else:
            self.ln_trans = None
            self.o_trans = None
        self.kcache_trans = single_cls(self.head_dim) if self.args.k_bits < 16 or self.args.q_bits < 16 else None
        self.vcache_trans = (
            single_cls(self.head_dim)
            if self.args.v_bits < 16 or self.args.w_bits < 16 or self.args.a_bits < 16
            else None
        )

    def _project_qkv(self, hidden_states):
        if self._ori_mode:
            if self.diag_init == "sq_style" and hasattr(self, "ln_smax"):
                self.ln_smax = torch.maximum(
                    self.ln_smax.to(hidden_states.device),
                    hidden_states.reshape(-1, hidden_states.shape[-1]).abs().max(0)[0].detach(),
                )
            return (
                self.q_proj._ori_forward(hidden_states),
                self.k_proj._ori_forward(hidden_states),
                self.v_proj._ori_forward(hidden_states),
            )

        hidden_states = self.ln_trans(hidden_states) if self.ln_trans is not None else hidden_states
        query_states = self.q_proj(hidden_states, qa_trans=self.ln_trans)
        key_states = self.k_proj(hidden_states, qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans)
        else:
            value_states = self.v_proj(hidden_states, qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        return query_states, key_states, value_states

    def quant_kcache(self, q, k):
        if not (self.args.k_bits < 16 or self.args.q_bits < 16):
            return q, k
        if self.kcache_trans is not None:
            q = self.kcache_trans(q, inv_t=True)
            k = self.kcache_trans(k)
        if self.args.q_bits < 16:
            q = self.q_cache_quantizer(q).to(q.dtype)
        if self.args.k_bits < 16:
            k = self.k_cache_quantizer(k).to(k.dtype)
        return q, k

    def quant_vcache(self, value_states):
        if self.args.separate_vtrans and self.vcache_trans is not None:
            value_states = self.vcache_trans(value_states)
        if self.args.v_bits < 16:
            value_states = self.v_cache_quantizer(value_states).to(value_states.dtype)
        return value_states

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        cache = past_key_values if past_key_values is not None else past_key_value
        query_states, key_states, value_states = self._project_qkv(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if cache is not None and hasattr(cache, "get_usable_length"):
            kv_seq_len += cache.get_usable_length(kv_seq_len, self.layer_idx)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = _apply_rotary(
                query_states, key_states, cos, sin, position_ids, self.rotary_fn, self.mrope_section
            )
        elif self.rotary_emb is not None:
            try:
                cos, sin = self.rotary_emb(value_states, position_ids)
            except TypeError:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = _apply_rotary(
                query_states, key_states, cos, sin, position_ids, self.rotary_fn, self.mrope_section
            )
        else:
            raise ValueError(
                "FlatQuantAttention requires position_embeddings with "
                "Transformers 4.52.x. Ensure the wrapper is called from the "
                "supported multimodal language-model decoder."
            )

        if not self._ori_mode:
            query_states, key_states = self.quant_kcache(query_states, key_states)
            value_states = self.quant_vcache(value_states)

        if cache is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = _update_cache(
                cache, key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = (
            attention_mask[:, :, :, : key_states.shape[-2]]
            if attention_mask is not None
            else None
        )
        if self.attn_implementation == "eager" or output_attentions:
            # SDPA does not return attention probabilities.  Match the HF
            # fallback and use eager attention only when they are requested.
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) * self.scaling
            if causal_mask is not None:
                attn_weights = attn_weights + causal_mask
            if self.return_past_key_value and query_states.dtype == torch.float16:
                # Match Qwen2-VL's eager-attention overflow guard.
                attn_weights = torch.where(
                    torch.isinf(attn_weights),
                    torch.zeros_like(attn_weights),
                    attn_weights,
                )
            attn_weights = F.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights = F.dropout(
                attn_weights,
                p=self.attention_dropout if self.training else 0.0,
                training=self.training,
            )
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            # Transformers 4.52 may omit its explicit mask when SDPA can use
            # the causal kernel directly.  Never set is_causal together with
            # an explicit mask; for single-token decoding all cached keys are
            # already valid, so no causal mask is required.
            is_causal = causal_mask is None and q_len > 1
            attn_output = F.scaled_dot_product_attention(
                query_states.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scaling,
            )
            attn_weights = None
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            bsz, q_len, self.hidden_size
        )

        if self._ori_mode:
            attn_output = self.o_proj._ori_forward(attn_output)
        elif self.o_trans is None and self.vcache_trans is not None:
            init_shape = attn_output.shape
            attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim)
            attn_output = torch.matmul(attn_output, self.vcache_trans.get_matrix(inv_t=True).T.to(attn_output))
            attn_output = self.o_proj(attn_output.reshape(init_shape))
        else:
            init_shape = attn_output.shape
            attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim)
            attn_output = torch.matmul(self.o_trans.get_matrix().T.to(attn_output), attn_output).reshape(init_shape)
            if self._eval_mode:
                attn_output = self.o_proj(attn_output)
            else:
                attn_o_inv = self.o_trans.get_matrix(inv_t=True)
                attn_v_inv = self.vcache_trans.get_matrix(inv_t=True)
                attn_output = self.o_proj(attn_output, qa_trans=[attn_o_inv, attn_v_inv])

        if self.return_past_key_value:
            return attn_output, attn_weights, cache
        return attn_output, attn_weights

    def init_diag_scale(self, alpha=0.5):
        if not hasattr(self, "ln_smax"):
            return
        qkvw_smax = torch.cat(
            [self.q_proj.linear.weight, self.k_proj.linear.weight, self.v_proj.linear.weight],
            dim=0,
        ).abs().max(dim=0)[0]
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(qkvw_smax, self.ln_smax.to(qkvw_smax.device), alpha)
        del self.ln_smax
        self.diag_init = None

    def rep_matrix_only(self):
        for trans in (self.ln_trans, self.kcache_trans, self.vcache_trans, self.o_trans):
            if trans is not None:
                trans.to_eval_mode()

    def reparameterize(self):
        self.rep_matrix_only()
        self.q_proj.reparameterize(qa_trans=self.ln_trans)
        self.k_proj.reparameterize(qa_trans=self.ln_trans)
        if self.args.separate_vtrans:
            self.v_proj.reparameterize(qa_trans=self.ln_trans)
        else:
            self.v_proj.reparameterize(qa_trans=self.ln_trans, out_trans=self.vcache_trans)
        if self.o_trans is not None and self.vcache_trans is not None:
            self.o_proj.reparameterize(
                qa_trans=[self.o_trans.get_matrix(inv_t=True), self.vcache_trans.get_matrix(inv_t=True)]
            )
        self._eval_mode = True


def _rotary_fn_for_model(model):
    if type(model) == model_utils.QWEN2_VL:
        return qwen2vl_apply_rotary_pos_emb
    if type(model) in (model_utils.LLAVA_HF, model_utils.LLAVA_NEXT_HF):
        return llama_apply_rotary_pos_emb
    raise ValueError(
        f"FlatQuant supports Qwen2-VL, LLaVA-1.5, and LLaVA-Next; got {type(model)}"
    )


def apply_flatquant_to_model(args, model):
    if type(model) == getattr(model_utils, "QWEN2_5_VL", None):
        from vlmrot.qwen25_p4q import apply_flatquant
        return apply_flatquant(args, model)
    rotary_fn = _rotary_fn_for_model(model)
    is_qwen2vl = type(model) == model_utils.QWEN2_VL
    for idx, layer in enumerate(model_utils.get_layers(model)):
        layer.self_attn = FlatQuantAttention(
            args,
            layer.self_attn,
            rotary_fn=rotary_fn,
            return_past_key_value=is_qwen2vl,
        )
        layer.mlp = FlatQuantMLP(args, layer.mlp)
    logger.info(f'Applied FlatQuant adapters to {len(model_utils.get_layers(model))} layers.')
    return model


def set_flatquant_attention_backend(model, backend):
    if backend not in ("eager", "sdpa"):
        raise ValueError(f"Unsupported FlatQuant attention backend: {backend}")

    # The language model builds its causal mask according to this setting, so
    # switch the outer mask producer and every FlatQuant attention together.
    lm_config = model_utils.get_lm_config(model)
    lm_config._attn_implementation = backend
    model_utils.get_language_model(model).config._attn_implementation = backend
    for layer in model_utils.get_layers(model):
        layer.self_attn.attn_implementation = backend
        layer.self_attn.config._attn_implementation = backend
    logger.info("Set FlatQuant attention backend to %s", backend)
    return model


def reparameterize_ln(ln, trans):
    ln_weight = ln.weight.data
    ori_dtype = ln_weight.dtype
    ln.weight.data = (ln_weight.to(torch.float64) * trans.diag_scale.to(torch.float64)).to(ori_dtype)
    trans.use_diag = False


def reparameterize_model(model):
    for layer in model_utils.get_layers(model):
        layer.self_attn.reparameterize()
        layer.mlp.reparameterize()
        if layer.self_attn.ln_trans is not None and layer.self_attn.ln_trans.add_diag:
            reparameterize_ln(layer.input_layernorm, layer.self_attn.ln_trans)
        if layer.mlp.up_gate_trans is not None and layer.mlp.up_gate_trans.add_diag:
            reparameterize_ln(layer.post_attention_layernorm, layer.mlp.up_gate_trans)
    return model


def _matrix_checkpoint_path(path_or_dir):
    path = Path(path_or_dir)
    return path / "flat_matrices.pth" if path.suffix == "" else path


def _is_flat_param(name):
    prefixes = (
        "self_attn.ln_trans.",
        "self_attn.o_trans.",
        "self_attn.kcache_trans.",
        "self_attn.vcache_trans.",
        "mlp.up_gate_trans.",
        "mlp.down_trans.",
    )
    return name.startswith(prefixes) or "clip_factor_w" in name or "clip_factor_a" in name


def get_flat_matrix_state(layer):
    layer.self_attn.rep_matrix_only()
    layer.mlp.rep_matrix_only()
    return OrderedDict(
        (name, tensor.detach().cpu())
        for name, tensor in layer.state_dict().items()
        if _is_flat_param(name)
    )


def save_flat_matrices(args, model, output_dir=None):
    base_dir = Path(
        output_dir
        if output_dir is not None
        else get_flatquant_output_dir(args)
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "flat_matrices.pth"
    flat_matrices = {
        idx: get_flat_matrix_state(layer)
        for idx, layer in enumerate(model_utils.get_layers(model))
    }
    if type(model) == getattr(model_utils, "QWEN2_5_VL", None):
        from vlmrot.qwen25_flatquant_training import require_finite
        require_finite(flat_matrices, "checkpoint before save")
    torch.save(flat_matrices, path)
    logger.info("FlatQuant matrices saved at %s", path)
    return path


def load_flat_matrices(args, model, path=None):
    load_path = _matrix_checkpoint_path(path or args.flatquant_matrix_path)
    if not load_path.exists():
        raise FileNotFoundError(f"FlatQuant matrix checkpoint not found: {load_path}")
    flat_matrices = torch.load(load_path, map_location="cpu", weights_only=False)
    if type(model) == getattr(model_utils, "QWEN2_5_VL", None):
        from vlmrot.qwen25_flatquant_training import require_finite
        require_finite(flat_matrices, f"checkpoint {load_path}")
    layers = model_utils.get_layers(model)
    for idx, layer in enumerate(layers):
        layer.self_attn.rep_matrix_only()
        layer.mlp.rep_matrix_only()
        state = flat_matrices[idx] if idx in flat_matrices else flat_matrices[str(idx)]
        missing, unexpected = layer.load_state_dict(state, strict=False)
        flat_missing = [name for name in missing if _is_flat_param(name)]
        if flat_missing or unexpected:
            raise RuntimeError(
                f"Failed to load FlatQuant layer {idx}: missing={flat_missing}, unexpected={unexpected}"
            )
    logger.info("Loaded FlatQuant matrices from %s", load_path)
    return model


def get_flatquant_output_dir(args):
    model_name = args.model.split("/")[-1]
    calibration_name = str(getattr(args, "cal_dataset", "unknown")).strip()
    calibration_aliases = {
        "scienceqa_train": "SQA",
        "coco_sharegpt4v": "COCO",
    }
    calibration_tag = calibration_aliases.get(calibration_name.lower())
    if calibration_tag is None:
        calibration_tag = "-".join(
            part for part in calibration_name.replace("/", "-").split() if part
        ) or "unknown"
    experiment_tag = (
        f"w{args.w_bits}a{args.a_bits}"
        f"_cal-{calibration_tag}"
        f"-n{int(args.nsamples)}"
        f"-s{int(args.seed)}"
    )
    return os.path.join(args.save_flatquant_dir, model_name, experiment_tag)


def get_trainable_flat_params(layer, args):
    for param in layer.parameters():
        param.requires_grad = False
    trainable = []
    names = []
    if args.cali_trans:
        names.append("trans")
    if args.add_diag:
        names.append("diag_scale")
    if args.lwc:
        names.append("clip_factor_w")
    if args.lac:
        names.append("clip_factor_a")
    for name, param in layer.named_parameters():
        if any(token in name for token in names):
            param.requires_grad = True
            trainable.append(param)
    return trainable
