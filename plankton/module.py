class EnzymeModule:
    """ Wrapper class for a DaCeML/PyTorch module that uses Enzyme AD """
    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        # TODO: Override codegen

        # Passthrough to dace module
        return self.module(*args, **kwargs)
