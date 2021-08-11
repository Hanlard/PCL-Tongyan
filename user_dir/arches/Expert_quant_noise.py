r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from fmoe.gates import NaiveGate
from fmoe.layers import FMoE, FMoELinear
from fairseq.modules.fairseq_dropout import FairseqDropout


import time
import signal


def mark_module_parallel_comm(module, comm):
    r"""
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, "dp_comm", comm)

def mark_module_parallel_comm_expert(module, comm):
    r"""
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, "dp_comm", comm)
        setattr(p, "expert", "I'm an expert.")

def quant_noise_FMoE(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:

            # gather weight and sizes
            weight = mod.weight            
            in_features = weight.size(-1)
            out_features = weight.size(-2)
            num_experts = weight.size(0)
            # split weight matrix into blocks and randomly drop selected blocks
            mask = torch.zeros(num_experts,
                in_features // block_size * out_features, device=weight.device
            )
            mask.bernoulli_(p)
            mask = mask.repeat_interleave(block_size, -1).view(num_experts,-1, in_features)


            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class _Expert_quant_noise(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, drop_model, quant_noise, quant_noise_block_size,  rank=0):
        super().__init__()
        self.htoh4 = quant_noise_FMoE(
            FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank),
            quant_noise,
            quant_noise_block_size
        )
        self.h4toh = quant_noise_FMoE(
            FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank),
            quant_noise,
            quant_noise_block_size
        )
        self.dropout_module = drop_model
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        if self.dropout_module:
            x = self.dropout_module(x)
        x = self.h4toh(x, fwd_expert_count)
        return x



class FMoETransformerMLP_quant_noise(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        world_size=1,
        mp_group=None,
        activation=torch.nn.GELU(),
        gate=NaiveGate,
        top_k=2,
        expert_dp_comm="none",
        gate_hook=None,
        mask=None,
        mask_dict=None,
        drop_model = None,
        quant_noise=None,
        quant_noise_block_size=None
    ):
        super().__init__(
            num_expert=num_expert,
            d_model=d_model,
            gate=gate,
            top_k=top_k,
            world_size=world_size,
            mp_group=mp_group,
            gate_hook=gate_hook,
            mask=mask,
            mask_dict=mask_dict
        )
        self.experts = _Expert_quant_noise(
            num_expert, d_model, d_hidden, activation, drop_model, quant_noise, quant_noise_block_size, rank=self.mp_rank
        )
        self.mark_parallel_comm(expert_dp_comm)
        
    def mark_parallel_comm(self, expert_dp_comm="none"):
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm_expert(e, comm)
            else:
                mark_module_parallel_comm_expert(self.experts, comm)
        mark_module_parallel_comm(self.gate, "world")
        
    
    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
#         try:
#             inp = inp.reshape(-1, self.d_model)
#             output = super().forward(inp)
#             return output.reshape(original_shape)
#         except:
#             print(f"[1]Expert_quant_noise.py line 178 <end> forward original_shape={original_shape} is_NaN-inp={torch.isnan(inp).any()} rank={torch.distributed.get_rank()}")
#             return None
        inp = inp.reshape(-1, self.d_model)
#         print(f"Expert_quant_noise.py line 190 rank={torch.distributed.get_rank()}")
        output = super().forward(inp)
#         print(f"Expert_quant_noise.py line 192 rank={torch.distributed.get_rank()}")
        return output.reshape(original_shape)
        




