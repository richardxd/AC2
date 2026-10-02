"""Expose installed cuRAND headers and the private toolkit's lib64 alias."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    scratch = Path(os.environ["AC2_SCRATCH"]).resolve()
    toolkit = Path(os.environ["CUDA_HOME"]).resolve()
    assert toolkit.is_relative_to(scratch)
    package = importlib.metadata.distribution("nvidia-curand-cu12")
    assert package.version == "10.3.10.19"
    include = Path(package.locate_file("nvidia/curand/include")).resolve()
    assert include.is_relative_to(scratch)
    headers = sorted(include.glob("*.h"))
    assert (include / "curand.h") in headers and (include / "curand_kernel.h") in headers
    receipt = {"package": "nvidia-curand-cu12", "version": package.version, "headers": []}
    for source in headers:
        target = toolkit / "include" / source.name
        if target.is_symlink() or target.exists():
            assert target.is_symlink() and target.resolve() == source
        else:
            target.symlink_to(source)
        receipt["headers"].append({"source": str(source), "target": str(target),
                                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    runtime = toolkit / "lib" / "libcudart.so"
    assert runtime.is_file() and runtime.resolve().is_relative_to(scratch)
    alias = toolkit / "lib64"
    if alias.is_symlink() or alias.exists():
        assert alias.is_symlink() and alias.resolve() == toolkit / "lib"
    else:
        alias.symlink_to("lib", target_is_directory=True)
    receipt["lib64"] = {"alias": str(alias), "target": str(alias.resolve()),
                        "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest()}
    with (root / "runs/e9/cuda-sampling-repair.json").open("x") as f:
        json.dump(receipt, f, indent=2)
    print(f"Linked and hashed {len(headers)} existing pinned headers; no download or package change")


if __name__ == "__main__":
    main()
