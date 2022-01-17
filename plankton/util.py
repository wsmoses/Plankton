import torch
import numpy as np
import time

# Code to help with profiling CUDA kernels (since PyTorch's API is nonblocking)
class CudaTimer:
    """Time CUDA kernels with CUDA events."""
    def __init__(self, n):
        """Support timing n regions."""
        self.events = []
        for _ in range(n + 1):
            self.events.append(torch.cuda.Event(enable_timing=True))
        self.cur = 0

    def start(self):
        """Start timing the first region."""
        self.events[0].record()
        self.cur = 1

    def next(self):
        """Start timing for the next region."""
        if self.cur >= len(self.events):
            raise RuntimeError('Trying to time too many regions')
        self.events[self.cur].record()
        self.cur += 1

    def end(self):
        """Finish timing."""
        if self.cur != len(self.events):
            raise RuntimeError('Called end too early')
        self.events[-1].synchronize()

    def get_times(self):
        """Return measured time for each region."""
        times = []
        for i in range(1, len(self.events)):
            times.append(self.events[i-1].elapsed_time(self.events[i]))
        return times

def benchmark(func, reps=10):
    timer = CudaTimer(reps)

    # Timing
    torch.cuda.profiler.start()
    
    start = time.time()
    timer.start()
    for i in range(reps):
        func()
        timer.next()
    timer.end()
    end = time.time()
    
    torch.cuda.profiler.stop()

    print('Average overall time:', (end - start) / reps, 's')
    print('Net kernel time (ms):', ' '.join(map(str, timer.get_times())))
