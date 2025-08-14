import torch
from typing import Callable, Any

def profile_forward(module: torch.nn.Module, name: str, timings: dict, *args, **kwargs) -> Any:
    """Helper function to profile a module's forward pass."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    output = module(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    
    timings[name] = timings.get(name, 0) + start_event.elapsed_time(end_event)
    return output

def profile_function(func: Callable, name: str, timings: dict, *args, **kwargs) -> Any:
    """Helper function to profile a standalone function call."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    output = func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    
    timings[name] = timings.get(name, 0) + start_event.elapsed_time(end_event)
    return output
