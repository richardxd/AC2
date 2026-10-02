"""Local, single-accounting-point DeepSeek API adapter for the unmodified reward.

Run one instance. Every outgoing attempt reserves worst-case cost durably before
I/O. Uncertain attempts keep their reservation. Peak prices bound off-peak spend.
No key enters command arguments, receipts, requests saved to disk, or logs.
"""
import argparse
import asyncio
import fcntl
import json
import sqlite3
import time
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web

MODEL = "deepseek-flash"
KEY_FILE = Path("/home-nfs/richard1xur/.codex/.deepseek_key")
# USD per million, peak rates; verified 2026-10-01 from official pricing page.
INPUT_MISS, INPUT_HIT, OUTPUT = 0.30, 0.006, 1.20
CAP_USD = 5.0


def usage_cost(usage):
    prompt = int(usage["prompt_tokens"])
    completion = int(usage["completion_tokens"])
    hit = int(usage.get("prompt_cache_hit_tokens", 0))
    assert 0 <= hit <= prompt and completion >= 0
    return ((prompt - hit) * INPUT_MISS + hit * INPUT_HIT + completion * OUTPUT) / 1e6


class Gateway:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock_file = (root / "gateway.lock").open("a")
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.db = sqlite3.connect(root / "spend.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, state TEXT, charged REAL, receipt TEXT)")
        self.db.commit()
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(4)
        self.session = None

    def total(self):
        return self.db.execute("SELECT COALESCE(SUM(charged),0) FROM calls").fetchone()[0]

    async def lifecycle(self, app):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240, sock_connect=20)) as session:
            self.session = session
            yield
        self.db.close()

    async def health(self, request):
        return web.json_response({"model": MODEL, "cap_usd": CAP_USD,
                                  "charged_upper_usd": self.total(),
                                  "calls": self.db.execute("SELECT state,COUNT(*) FROM calls GROUP BY state").fetchall()})

    async def chat(self, request):
        incoming = await request.json()
        assert len(incoming["messages"]) == 1
        assert incoming["messages"][0]["role"] == "user"
        content = incoming["messages"][0]["content"]
        assert isinstance(content, str)
        max_tokens = int(incoming["max_tokens"])
        if not 1 <= max_tokens <= 40000:
            raise web.HTTPBadRequest(text="judge output budget outside 1..40000")
        payload = {"model": MODEL, "messages": incoming["messages"],
                   "max_tokens": max_tokens, "stream": False,
                   "thinking": {"type": "enabled"},
                   "reasoning_effort": incoming.get("chat_template_kwargs", {}).get("reasoning_effort", "high")}
        # A text token cannot require less than one UTF-8 byte; allow 1000 extra
        # tokens for message framing. All input is conservatively cache-miss.
        reservation = ((len(content.encode()) + 1000) * INPUT_MISS + max_tokens * OUTPUT) / 1e6
        async with self.semaphore:
            async with self.lock:
                if self.total() + reservation > CAP_USD:
                    raise web.HTTPPaymentRequired(text="reproduction judge cap: request not sent")
                call_id = uuid.uuid4().hex
                self.db.execute("INSERT INTO calls VALUES (?,?,?,?)", (call_id, "reserved", reservation, ""))
                self.db.commit()
            started = time.time()
            headers = {"Authorization": "Bearer " + KEY_FILE.read_text().strip()}
            # No automatic retries here. The existing reward HTTP layer can retry;
            # every repeated request obtains its own reservation first.
            async with self.session.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers) as response:
                data = await response.json()
                status = response.status
            elapsed = time.time() - started
            receipt = {"id": call_id, "started_unix": started, "latency_s": elapsed,
                       "request": payload, "response": data, "http_status": status,
                       "reserved_usd": reservation, "prices_per_million":
                       {"input_miss": INPUT_MISS, "input_hit": INPUT_HIT, "output": OUTPUT}}
            charge = reservation
            state = "uncertain"
            if status == 200:
                charge = usage_cost(data["usage"])
                assert charge <= reservation, "usage exceeded reserved upper bound"
                state = "complete"
                receipt["cost_upper_usd"] = charge
            receipt_path = self.root / (call_id + ".json")
            with receipt_path.open("x") as f:
                json.dump(receipt, f, ensure_ascii=False, indent=2)
            async with self.lock:
                self.db.execute("UPDATE calls SET state=?,charged=?,receipt=? WHERE id=?", (state, charge, str(receipt_path), call_id))
                self.db.commit()
            print(json.dumps({"id": call_id, "status": status, "latency_s": elapsed,
                              "cost_upper_usd": charge, "total_upper_usd": self.total()}), flush=True)
            return web.json_response(data, status=status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18791)
    args = parser.parse_args()
    gateway = Gateway(Path(__file__).resolve().parents[1] / "runs/judge")
    app = web.Application(client_max_size=8 * 1024**2)
    app.cleanup_ctx.append(gateway.lifecycle)
    app.router.add_get("/health", gateway.health)
    app.router.add_post("/v1/chat/completions", gateway.chat)
    web.run_app(app, host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
