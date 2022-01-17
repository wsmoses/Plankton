import sys
import time
import os
import click

import torch
from torch import nn
from torch.nn import functional as F

from daceml.pytorch import dace_module
import plankton
from plankton.util import benchmark

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

@click.command()
@click.option('--schedule', default='noop', help='Pre-AD schedule to use',
              type=click.Choice(['noop', 'fuse'], case_sensitive=False))
@click.option('--enzyme/--no-enzyme', default=True, help='Use Enzyme as auto-differentiator')
@click.option('--cuda', default=False, help='Use CUDA', is_flag=True)
@click.option('--reps', default=100, help='Number of repetitions in benchmarking')
def main(schedule, enzyme, cuda, reps):
    if enzyme:
        DaCeMish = plankton.EnzymeModule(dace_module(cuda=cuda, backward=False)(Mish))
    else:
        DaCeMish = dace_module(cuda=cuda, backward=True)(Mish)
              
    # Create test inputs (size taken from YOLOv4)
    with torch.no_grad():
        dace_input = torch.rand(8, 32, 224, 224)
        torch_input = torch.clone(dace_input)
        dace_dy = torch.rand(8, 32, 224, 224).cuda()
        torch_dy = torch.clone(dace_dy)
    
    dace_input.requires_grad = True
    torch_input.requires_grad = True
    
    torch_mish = Mish()
    dace_mish = DaCeMish()
    
    if cuda:
        torch_mish = torch_mish.cuda()
        dace_input = dace_input.cuda()
        dace_dy = dace_dy.cuda()

    if schedule == 'fuse':
        optimize(dace_mish)
        
    
    # Warmup & verification
    out = dace_mish(dace_input)
    dace_mish.backward(dace_dy)
    
    torch_output = torch_mish(torch_input)
    torch_output.backward(torch_dy)
    
    assert torch.allclose(out, torch_output)
    assert torch.allclose(dace_input.grad, torch_input.grad)

    def func():
        out = dace_mish(dace_input)
        dace_mish.backward(dace_dy)    
    
    # Benchmark
    benchmark(func, reps)


def optimize(mod, vectorize=False):   
    from daceml.transformation import TaskletFusion
    from dace.transformation.dataflow import Vectorization, TrivialMapRangeElimination
    from dace.transformation.subgraph import SubgraphFusion
    from daceml.util import utils
    from dace.library import change_default
    from daceml import onnx as donnx
    
    
    # expand the onnx nodes, and apply automatic transformations like inlining
    def expand_and_strict_transforms(module):
        # use the pure expansions of operators
        with change_default(donnx, "pure"):
            utils.auto_optimize(module.sdfg, cuda=True, apply_strict=True)
    
    
    dace_mish.append_post_onnx_hook("auto_optimize", expand_and_strict_transforms)
    
    
    # apply subgraph fusion
    def fuse_sg(module):
        sdfg = module.sdfg
        sdfg.apply_transformations_repeated(TrivialMapRangeElimination)
        SubgraphFusion.apply_to(sdfg, *sdfg.node(0).nodes())
    
    
    dace_mish.append_post_onnx_hook("subgraph_fusion", fuse_sg)
    
    if vectorize:
        # apply vectorization
        def vectorize(fwd, bwd):
            fwd.apply_transformations(Vectorization)
            bwd.apply_transformations(Vectorization)
        
        
        dace_mish.append_post_autodiff_hook("vectorize", vectorize)
    

#################################################################
# Benchmarking code

if __name__ == '__main__':
    main()
