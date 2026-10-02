"""Record native gptoss requests; the only upstream is the local GPU0 vLLM server."""
import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path

import aiohttp
from aiohttp import web
from surrogate_common import ROOT, MODEL, UPSTREAM, MAX_TOKENS, EFFORT, CONCURRENCY, HTTP_TIMEOUT, profile, route


async def main():
    root = ROOT / "runs/surrogate"
    receipts = root / "requests"
    receipts.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "calls.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, started REAL, finished REAL, status INTEGER, receipt TEXT)")
    db.commit()
    limit = asyncio.Semaphore(CONCURRENCY)

    async def lifecycle(app):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT-10)) as session:
            app["session"] = session
            yield
        db.close()

    async def health(request):
        async with request.app["session"].get("http://127.0.0.1:18800/health") as response:
            assert response.status == 200
        return web.json_response(profile())

    async def chat(request):
        payload = await request.json()
        assert "chat_template_kwargs" not in payload and payload["include_reasoning"] is True
        assert payload["reasoning_effort"] == EFFORT and payload["max_tokens"] == MAX_TOKENS
        assert len(payload["messages"]) == 1 and payload["messages"][0]["role"] == "user"
        kind = route(payload["messages"][0]["content"])
        payload["model"] = MODEL
        call_id, started = uuid.uuid4().hex, time.time()
        path = receipts / f"{call_id}.json"
        db.execute("INSERT INTO calls VALUES (?,?,NULL,NULL,?)", (call_id, started, str(path)))
        db.commit()
        async with limit:
            dispatched = time.time()
            async with request.app["session"].post(UPSTREAM, json=payload) as response:
                data, status = await response.json(), response.status
        finished = time.time()
        with path.open("x") as f:
            json.dump({"id": call_id, "route": kind, "started": started, "dispatched": dispatched,
                       "finished": finished, "latency_s": finished-started, "status": status,
                       "profile": profile(), "request": payload, "response": data}, f)
        db.execute("UPDATE calls SET finished=?,status=? WHERE id=?", (finished, status, call_id))
        db.commit()
        return web.json_response(data, status=status)

    app = web.Application(client_max_size=4*1024**2)
    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18801).start()
    print(json.dumps(profile()), flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
