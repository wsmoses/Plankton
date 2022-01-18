from typing import Any, Tuple, Union
import torch
from torch import nn
from dace.codegen import compiler, targets
from dace.codegen.codeobject import CodeObject
from dace.codegen.compiled_sdfg import CompiledSDFG
from daceml.onnx.onnx_importer import create_output_array
from daceml.pytorch.environments.pytorch_env import PyTorch
from daceml.pytorch.module import DaceModule
from daceml.pytorch.dispatchers.common import DaCeMLTorchFunction
from daceml.pytorch.dispatchers.cpp_torch_extension import (
    get_env_for_sdfg, code_for_backward_function, code_for_module, indent_code)

import operator
import os

from daceml.util.utils import platform_library_name


class EnzymeModule:
    """ Wrapper class for a DaCeML/PyTorch module that uses Enzyme AD """
    def __init__(self, module: nn.Module, cuda: bool = False):
        # Override DaCeML to not compute AD
        self.module = DaceModule(module, cuda=cuda, backward=False)

    def __call__(self, *args, **kwargs):
        if self.module.function is None:
            self.module.function = self.module._initialize_sdfg(args, compile=False)
            required_grads = self.module.required_gradients()
            # This creates the torch extension and overrides the normal dace codegen
            self.module.compiled_function = compile_enzyme_torch_extension(self.module, args)

        # Passthrough to dace module
        return self.module(*args, **kwargs).requires_grad_()

def compile_enzyme_torch_extension(module: DaceModule,
                                   dummy_inputs: Any) -> DaCeMLTorchFunction:
    """
    Get a torch callable for the module. This will compile the sdfg with Enzyme calls injected, 
    compile a PyTorch C++ operator, register it with PyTorch and return the function that calls it.

    This function handles code generation for both the forward and backward pass.
    :param module: the module.
    :param dummy_inputs: dummy inputs to initialize the model with.
    :return: the callable function for the SDFG.
    """
    raise NotImplementedError

    # module.sdfg.global_code += ...
