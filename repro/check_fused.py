"""Validate the shipped chunked PPO output kernel against eager math on GPU1."""
import json
import os
import subprocess
from pathlib import Path

import torch
from verl.utils.experimental.torch_functional import FusedLinearForPPOFunction


def main():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1" and torch.cuda.device_count() == 1
    uuid = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid",
                                    "--format=csv,noheader"], text=True).strip().removeprefix("GPU-")
    assert str(torch.cuda.get_device_properties(0).uuid) == uuid
    torch.manual_seed(192)
    results = []
    for dtype, tolerance in [(torch.float32, 1e-4), (torch.bfloat16, .02)]:
        h = (torch.randn(65, 64, device="cuda", dtype=dtype) * .1).requires_grad_()
        w = (torch.randn(1024, 64, device="cuda", dtype=dtype) * .1).requires_grad_()
        labels = torch.randint(1024, (65,), device="cuda")
        hf, wf = h.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
        lp, entropy = FusedLinearForPPOFunction.apply(hf, wf, labels, .8, 32)
        logits = (h @ w.T).float() / .8
        logp = logits.log_softmax(-1)
        reference_lp = logp.gather(-1, labels[:, None]).squeeze(-1)
        reference_entropy = -(logp.exp() * logp).sum(-1)
        weights = torch.linspace(-1, 1, 65, device="cuda")
        (lp * weights + .03 * entropy).mean().backward()
        (reference_lp * weights + .03 * reference_entropy).mean().backward()
        errors, relative_gradient_errors = {}, {}
        for name, actual, expected in [("log_probs", lp, reference_lp), ("entropy", entropy, reference_entropy),
                                       ("hidden_gradient", hf.grad, h.grad), ("weight_gradient", wf.grad, w.grad)]:
            absolute_tolerance = (2e-5 if dtype == torch.bfloat16 else 1e-6) if "gradient" in name else tolerance
            torch.testing.assert_close(actual.float(), expected.float(), atol=absolute_tolerance, rtol=tolerance)
            assert torch.isfinite(actual).all()
            errors[name] = float((actual.float() - expected.float()).abs().max().detach())
            if "gradient" in name:
                relative_gradient_errors[name] = float((actual.float() - expected.float()).norm() / expected.float().norm())
                assert relative_gradient_errors[name] < tolerance
        assert hf.grad.norm() > 0 and wf.grad.norm() > 0
        results.append({"dtype": str(dtype), "forward_atol_rtol_and_gradient_rtol": tolerance,
                        "max_absolute_errors": errors, "gradient_relative_l2_errors": relative_gradient_errors})
    receipt = {"physical_gpu": 1, "uuid": uuid, "shape": [65, 64, 1024], "chunk": 32,
               "temperature": .8, "objective": "signed weighted log-prob +0.03 entropy", "checks": results}
    out = Path("runs/e6/fused-kernel-check-02.json")
    with out.open("x") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
