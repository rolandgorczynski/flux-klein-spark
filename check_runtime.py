"""Check hardware, pipeline import and model authorization without large downloads."""
import torch
from diffusers import Flux2KleinPipeline
from runtime import model_access_status
assert torch.cuda.is_available()
x = torch.ones((32, 32), device='cuda')
assert (x @ x).sum().item() == 32768
print(f'CUDA PASS: {torch.cuda.get_device_name(0)}, torch {torch.__version__}')
import app
assert app.pipe.pipeline is None, 'UI import must not load model weights'
print(f'UI PASS: {len(app.demo.config["components"])} components')
print(model_access_status())
