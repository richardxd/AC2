"""Pinned, loopback-only surrogate judge configuration (Richard approved 2026-10-02)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
MODEL_PATH = ROOT / "models/gpt-oss-20b" / REVISION
URL = "http://127.0.0.1:18801/v1"
UPSTREAM = "http://127.0.0.1:18800/v1/chat/completions"
MAX_TOKENS = 65536
EFFORT = "high"
PINS = {"finegrained_noref_judge.txt": "474fc5f44191ebcee4bb9e3bd1d3cd3f407d059dfda316439dc4fd930ee2fbe6",
        "imo_proofautograder.txt": "3b910a4e1084eb668c31aeb9cb8926fa903bb343f109583a090ecbfaafd40a31"}


def templates():
    result = {}
    for name, sha in PINS.items():
        data = (ROOT / "src/ac2/rewards/templates" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha, f"surrogate template drift: {name}"
        result["train" if name.startswith("finegrained") else "val"] = data.decode()
    return result


def route(prompt):
    matches = [kind for kind, template in templates().items() if prompt.startswith(template.split("{problem}")[0])]
    assert len(matches) == 1, "unrecognized or ambiguous original template"
    return matches[0]


def accounting():
    import sqlite3
    db = sqlite3.connect(f"file:{ROOT / 'runs/surrogate/calls.sqlite'}?mode=ro", uri=True)
    rows = db.execute("SELECT id,started,finished,status,receipt FROM calls ORDER BY id").fetchall()
    db.close()
    return [dict(zip(("id", "started", "finished", "status", "receipt"), row)) for row in rows]


def profile():
    templates()
    return {"model": MODEL, "revision": REVISION, "model_path": str(MODEL_PATH),
            "url": URL, "upstream": UPSTREAM, "payload_style": "gptoss",
            "reasoning_effort": EFFORT, "max_tokens": MAX_TOKENS, "template_sha256": PINS}
