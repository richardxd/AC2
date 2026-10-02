"""E1 acceptance plus real flash-attn forward/backward on physical GPUs 1–7."""
import importlib.metadata as md
import json
import os

import flash_attn
import torch
import verl
import vllm

assert torch.__version__ == "2.11.0+cu129"
assert vllm.__version__ == "0.23.0"
assert flash_attn.__version__ == "2.8.1"
assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2,3,4,5,6,7"
assert torch.cuda.device_count() == 7
receipts = []
for i in range(7):
    torch.manual_seed(101)
    q = torch.randn(2, 64, 4, 128, device=f"cuda:{i}", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out = flash_attn.flash_attn_func(q, k, v, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
    error = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, atol=.03, rtol=.03)
    out.float().square().mean().backward()
    assert all(torch.isfinite(x.grad).all().item() for x in [q, k, v])
    receipts.append({"logical_gpu": i, "physical_gpu": i+1, "name": torch.cuda.get_device_name(i),
                     "capability": torch.cuda.get_device_capability(i), "max_abs_error": error,
                     "finite_gradients": True})
print(json.dumps({"torch": torch.__version__, "vllm": md.version("vllm"),
                  "flash_attn": flash_attn.__version__, "verl": verl.__file__,
                  "device_count": torch.cuda.device_count(), "kernel_checks": receipts}, indent=2))
