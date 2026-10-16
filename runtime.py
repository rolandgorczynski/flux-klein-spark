"""Local GPU execution and lazy model loading for the standalone Spark service."""
from functools import wraps
from threading import RLock
import gradio as gr
import torch
from config import MODEL_REPO, MODEL_VARIANT

GPU_LOCK = RLock()

def GPU(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        with GPU_LOCK:
            with torch.inference_mode():
                return fn(*args, **kwargs)
    return run

class LazyPipeline:
    def __init__(self):
        self.pipeline = None

    def load(self):
        with GPU_LOCK:
            if self.pipeline is None:
                if not torch.cuda.is_available():
                    raise gr.Error('CUDA unavailable. Start the container with GPU support.')
                try:
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(MODEL_REPO, 'model_index.json')
                except Exception as exc:
                    from huggingface_hub.errors import GatedRepoError
                    if isinstance(exc, GatedRepoError):
                        raise gr.Error('FLUX.2 Klein 9B access denied. Accept model access on Hugging Face and allow gated-model access for your local token, then retry.') from exc
                    raise
                from diffusers import Flux2KleinPipeline
                cls = Flux2KleinPipeline
                if MODEL_VARIANT == '9B-KV':
                    from diffusers import Flux2KleinKVPipeline
                    cls = Flux2KleinKVPipeline
                print(f'Loading {MODEL_REPO} on {torch.cuda.get_device_name(0)}', flush=True)
                candidate = cls.from_pretrained(MODEL_REPO, torch_dtype=torch.bfloat16).to('cuda')
                candidate.vae.enable_tiling()
                self.pipeline = candidate
                print('Model ready', flush=True)
            return self.pipeline

    def __getattr__(self, name):
        return getattr(self.load(), name)

    def __call__(self, *args, **kwargs):
        return self.load()(*args, **kwargs)


def model_access_status():
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    try:
        hf_hub_download(MODEL_REPO, 'model_index.json')
        return 'Model access granted. First generation will download and load the model.'
    except GatedRepoError:
        return 'Model access blocked: accept the 9B model conditions on Hugging Face and enable gated-model access for your local token. Then click Check access.'
    except Exception as exc:
        return f'Model access check failed ({type(exc).__name__}). See container logs.'
