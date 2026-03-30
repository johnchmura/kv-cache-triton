import torch

def quantize_and_pack_int4(x: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetrically quantizes a float16/float32 tensor `x` to INT4,
    packing two 4-bit values into one `uint8` along `dim`.
    
    Returns:
        packed (torch.Tensor of uint8): The quantized and packed tensor.
        scales (torch.Tensor): The FP16 scaling factors used, maintaining `keepdim=True`.
    """
    dim = dim if dim >= 0 else len(x.shape) + dim
    if x.shape[dim] % 2 != 0:
        raise ValueError(f"Dimension {dim} must be even to pack two 4-bit values into one byte.")
        
    scales = x.abs().amax(dim=dim, keepdim=True) / 7.0
    scales.clamp_(min=1e-5) 
    
    x_q = torch.round(x / scales).to(torch.int8) + 8
    
    shape = list(x_q.shape)
    shape[dim] = shape[dim] // 2
    shape.insert(dim + 1, 2)
    x_pairs = x_q.view(*shape)
    
    val1 = x_pairs.select(dim + 1, 0).to(torch.uint8)
    val2 = x_pairs.select(dim + 1, 1).to(torch.uint8)
    
    packed = (val1 & 0x0F) | ((val2 & 0x0F) << 4)
    
    return packed, scales.to(torch.float16)

def unpack_and_dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Unpacks a `uint8` tensor containing two 4-bit pairs per element
    along `dim` and dequantizes back to float16.
    
    Returns:
        unpacked (torch.Tensor of float16): The restored tensor.
    """
    dim = dim if dim >= 0 else len(packed.shape) + dim
    
    val1 = (packed & 0x0F).to(torch.int8) - 8
    val2 = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    
    shape = list(packed.shape)
    shape[dim] = shape[dim] * 2
    
    stacked = torch.stack([val1, val2], dim=dim + 1)
    x_q = stacked.view(*shape)
    
    return x_q.to(torch.float16) * scales
