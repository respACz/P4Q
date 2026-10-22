from vlmrot import model_utils, utils, hadamard_utils
from vlmrot.model_utils import LLAVA_NEXT_HF, LLAVA_OV_HF, QWEN2_VL
import torch, transformers
import inspect, logging, typing
from tqdm import tqdm
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(message)s')

def fuse_layernorm(model, fuse_module):
    if fuse_module == 'vit':
        layers = model_utils.get_vit_layers(model)
        if type(model) == LLAVA_NEXT_HF or type(model) == LLAVA_OV_HF or type(model) == QWEN2_VL:
            vit_model = model.vision_tower.vision_model
            vit_model.pre_layrnorm = model_utils.LN_(vit_model.pre_layrnorm)
            for idx, layer in enumerate(layers):
                # print(inspect.getsource(layer.forward))
                fuse_ln_linear(layer.layer_norm1, [layer.self_attn.q_proj,
                                                   layer.self_attn.k_proj, layer.self_attn.v_proj])
                fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1])
                w, b = fold_mean_into_linear(layer.self_attn.out_proj)
                if idx < len(layers) - 1:
                    w, b = fold_mean_into_linear(layer.mlp.fc2)
                else:
                    logging.info(f"skip mlp mean in layer{idx}")

            model_utils.replace_modules(
                vit_model.encoder,
                torch.nn.LayerNorm,
                lambda _: model_utils.RMSN(model_utils.get_vit_config(model).hidden_size, eps=1e-5),
                replace_layers=False,
            )

    elif fuse_module == 'llm':
        layers = model_utils.get_transformer_layers(model)
        pre_head_norm = model_utils.get_pre_head_layernorm(model)
        lm_head = model_utils.get_lm_head(model)
        fuse_ln_linear(pre_head_norm, [lm_head])
        for layer in layers:
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj,
                                                    layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])

        norm_type = model_utils.get_lm_norm_type(model)
        model_utils.replace_modules(
            model,
            norm_type,
            lambda _: model_utils.RMSN(model.config.text_config.hidden_size, eps=1e-5),
            replace_layers=False,
        )
    else:
        raise ValueError

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()

        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)

def fold_mean_into_linear(linear: torch.nn.Linear):
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    W_mean = W_.mean(dim=-2, keepdim=True)
    linear.weight.data = W_ - W_mean
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        b_mean = b_.mean()
        linear.bias.data = b_ - b_mean
        linear.bias.data = linear.bias.data.to(linear_dtype)
    return (W_mean, b_mean)

def rotate_model(model, rotate_module, online_rotate, Q=None):
    if rotate_module == 'llm':
        config = model_utils.get_lm_config(model)
        if Q is None:
            Q = get_orthogonal_matrix(config.hidden_size, 'hadamard')
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads

        model_type = type(model)
        rotate_embeddings(model, Q) # +rotate_projector
        rotate_projectors(model, Q) # +rotate_projector
        rotate_head(model, Q) # is lm head
        utils.cleanup_memory()
        layers = model_utils.get_transformer_layers(model)
        for idx, layer in enumerate(tqdm(layers, unit="layer", desc="Rotating")):
            rotate_attention_inputs(layers[idx], Q)
            rotate_attention_output(layers[idx], Q, model_type)
            rotate_mlp_input(layers[idx], Q, model_type)
            rotate_mlp_output(layers[idx], Q, model_type, online_rotate) # no in vit
            if online_rotate:
                rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)
        return Q

def get_orthogonal_matrix(size, mode, device=utils.get_dev()):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return hadamard_utils.random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')

def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in model_utils.get_embeddings(model):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_projectors(model, Q: torch.Tensor) -> None:
    # Rotate the projectors.
    from vlmrot.quant_utils import ActQuantWrapper
    for proj in model_utils.get_projector(model):
        if type(proj) in [nn.Linear, ActQuantWrapper]:
            W = proj # for smolvlm
        elif type(model) == model_utils.LLAVA_NEXT_HF or type(model) == model_utils.LLAVA_OV_HF:
            W = proj.linear_2
        elif type(model) == QWEN2_VL:
            W = proj.mlp[-1]  # Qwen2-VL's visual_projection_layer is a Linear layer
        else:
            W = proj[-1]
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_inputs(layer, Q) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == model_utils.LLAVA_NEXT_HF or model_type == model_utils.LLAVA_OV_HF or model_type == QWEN2_VL:
        W = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAVA_NEXT_HF or model_type == model_utils.LLAVA_OV_HF or model_type == QWEN2_VL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_mlp_output(layer, Q, model_type, online_rotate, had_dim=-1):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAVA_NEXT_HF or model_type == model_utils.LLAVA_OV_HF or model_type == QWEN2_VL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if online_rotate:
        # apply exact (inverse) hadamard on the weights of mlp down proj input
        hadamard_utils.apply_exact_had_to_linear(W, had_dim=had_dim, output=False) 
    if W.bias is not None:
        b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_ov_proj(layer, model_type, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    if model_type == model_utils.LLAVA_NEXT_HF or model_type == model_utils.LLAVA_OV_HF or model_type == QWEN2_VL:
        o_proj = layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    hadamard_utils.apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    hadamard_utils.apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q
