import torch 
 
class Model(torch.nn.Module): 
    def __init__(self, ng): 
        super().__init__() 
        self.ng= ng
 
    def forward(self, x): 
        return torch.nn.GroupNorm(self.ng, num_channels=x.shape[1], eps=1e-05) 
 
from torch.onnx.symbolic_registry import register_op 
 
def group_norm(g, ng, nc, *, out=None): 
    return g.op("custom::group_norm", ng, nc) 
 
register_op('group_norm', group_norm, '', 9) 
 
model = Model(8) 
input = torch.rand(1, 8, 10, 10).cuda()
torch.onnx.export(model, input, 'group_norm.onnx') 
@parse_args("v", "i", "v", "v", "f", "i", "none") 
def symbolic(g,  
        input, 
        weight, 
        offset, 
        mask, 
        bias, 
        stride_h, stride_w, 
        pad_h, pad_w, 
        dil_h, dil_w, 
        n_weight_grps, 
        n_offset_grps, 
        use_mask): 
    return g.op("custom::deform_conv2d", input, offset) 

@_onnx_symbolic("aten::group_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "i", "v", "v", "f", "i")
@_beartype.beartype
def group_norm(g, input, num_groups, weight, bias, eps, cudnn_enabled):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at(
            "group_norm",
            input,
            weight,
            bias,
            num_groups_i=num_groups,
            eps_f=eps,
            cudnn_enabled_i=cudnn_enabled,
        )

    channel_size = symbolic_helper._get_tensor_dim_size(input, 1)
    if channel_size is not None:
        assert channel_size % num_groups == 0
    input_rank = symbolic_helper._get_tensor_rank(input)
    if input_rank is None:
        return symbolic_helper._unimplemented("group_norm", "unknown input rank", input)
    # 0 in the shape list keeps dimension value unchanged.
    shape = [0, num_groups, -1]
    input_reshaped = symbolic_helper._reshape_helper(
        g, input, g.op("Constant", value_t=torch.LongTensor(shape))
    )

    # C is always divisible by num_groups
    # Due to shape difference. we need to apply weight and bias after
    # instance norm computation and reshape
    weight_ = g.op(
        "Constant",
        value_t=torch.tensor(
            [1.0] * num_groups,
            dtype=_type_utils.JitScalarType.from_name(
                input.type().scalarType()
            ).dtype(),
        ),
    )
    bias_ = g.op(
        "Constant",
        value_t=torch.tensor(
            [0.0] * num_groups,
            dtype=_type_utils.JitScalarType.from_name(
                input.type().scalarType()
            ).dtype(),
        ),
    )

    norm_reshaped = g.op(
        "InstanceNormalization", input_reshaped, weight_, bias_, epsilon_f=eps
    )
    norm = symbolic_helper._reshape_helper(g, norm_reshaped, g.op("Shape", input))

    if weight is None or weight.node().mustBeNone():
        weight_value = torch.tensor(
            [1.0],
            dtype=_type_utils.JitScalarType.from_name(
                input.type().scalarType()
            ).dtype(),
        )
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or bias.node().mustBeNone():
        bias_value = torch.tensor(
            [0.0],
            dtype=_type_utils.JitScalarType.from_name(
                input.type().scalarType()
            ).dtype(),
        )
        bias = g.op("Constant", value_t=bias_value)

    # Norm has shape [N, C, *] so we reshape weight and bias to [C, *]
    axes = list(range(1, input_rank - 1))
    return add(
        g,
        mul(g, norm, symbolic_helper._unsqueeze_helper(g, weight, axes)),
        symbolic_helper._unsqueeze_helper(g, bias, axes),
    )