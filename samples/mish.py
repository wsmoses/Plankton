import sys
import time
import os
from typing import Union
import click

import torch
from torch import nn
from torch.nn import functional as F

from daceml.pytorch import DaceModule
from daceml import onnx as donnx
from plankton.module import EnzymeModule
from plankton.util import benchmark


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


@click.command()
@click.option('--schedule',
              default='noop',
              help='Pre-AD schedule to use',
              type=click.Choice(['noop', 'fuse', 'tfuse'],
                                case_sensitive=False))
@click.option('--enzyme/--no-enzyme',
              default=True,
              help='Use Enzyme as auto-differentiator')
@click.option('--cuda', default=False, help='Use CUDA', is_flag=True)
@click.option('--reps',
              default=100,
              help='Number of repetitions in benchmarking')
def main(schedule, enzyme, cuda, reps):
    # Never use ONNXRuntime
    donnx.default_implementation = 'pure'

    dace_mish = Mish()

    if enzyme:
        dace_mish = EnzymeModule(dace_mish, cuda=cuda)
    else:
        dace_mish = DaceModule(dace_mish, cuda=cuda, backward=True)

    # Create test inputs (size taken from YOLOv4)
    with torch.no_grad():
        dace_input = torch.rand(8, 32, 224, 224)
        torch_input = torch.clone(dace_input)
        dace_dy = torch.rand(8, 32, 224, 224)
        torch_dy = torch.clone(dace_dy)

    dace_input.requires_grad = True
    torch_input.requires_grad = True

    torch_mish = Mish()

    if cuda:
        torch_mish = torch_mish.cuda()
        dace_input = dace_input.cuda()
        dace_dy = dace_dy.cuda()

    if schedule == 'fuse':
        optimize(dace_mish)
    elif schedule == 'tfuse':
        optimize(dace_mish, fuse_tasklets=True)

    # Warmup & verification
    out = dace_mish(dace_input)
    out.backward(dace_dy)

    torch_output = torch_mish(torch_input)
    torch_output.backward(torch_dy)

    assert torch.allclose(out, torch_output)
    assert torch.allclose(dace_input.grad, torch_input.grad)

    def func():
        out = dace_mish(dace_input)
        out.backward(dace_dy)

    print('Verified. Benchmarking...')

    # Benchmark
    benchmark(func, reps)


def optimize(mod: Union[EnzymeModule, DaceModule],
             vectorize=False,
             fuse_tasklets=False):
    import dace
    from dace import data as dt
    from dace.transformation.dataflow import Vectorization, TrivialMapRangeElimination
    from dace.transformation.subgraph import SubgraphFusion
    from daceml.util import utils
    from plankton.dml_flatten_elemwise import flatten_elementwise_onnx

    if isinstance(mod, EnzymeModule):
        mod: DaceModule = mod.module

    # expand the onnx nodes, and apply automatic transformations like inlining
    def expand_and_strict_transforms(module):
        utils.auto_optimize(module.sdfg, cuda=True, apply_strict=True)

    mod.append_post_onnx_hook("auto_optimize", expand_and_strict_transforms)

    # apply subgraph fusion
    def fuse_sg(module):
        sdfg = module.sdfg
        sdfg.apply_transformations_repeated(TrivialMapRangeElimination)
        SubgraphFusion.apply_to(sdfg, *sdfg.node(0).nodes())

    mod.append_post_onnx_hook("subgraph_fusion", fuse_sg)

    # apply tasklet fusion
    if fuse_tasklets:

        def _fuse(sdfg):
            from daceml.transformation import TaskletFusion
            sdfg.apply_transformations_repeated(TaskletFusion)

        mod.append_post_onnx_hook("fuse_tasklets", lambda x: _fuse(x.sdfg))

    # flatten elementwise maps
    flatten_elementwise_onnx(mod)

    if vectorize:
        # apply vectorization
        def vectorize(fwd, bwd):
            fwd.apply_transformations(Vectorization)
            bwd.apply_transformations(Vectorization)

        mod.append_post_autodiff_hook("vectorize", vectorize)


#################################################################
# Benchmarking code

if __name__ == '__main__':
    main()
