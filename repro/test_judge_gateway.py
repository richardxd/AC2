"""Offline checks for financial guardrails; no credential reads or paid calls."""
import asyncio
import importlib.util
from pathlib import Path

import pytest
from aiohttp import web

spec = importlib.util.spec_from_file_location("gateway", Path(__file__).with_name("judge_gateway.py"))
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


def test_usage_counts_cache_hits_and_output():
    assert gateway.usage_cost({"prompt_tokens": 1000000, "completion_tokens": 1000000,
                               "prompt_cache_hit_tokens": 250000}) == pytest.approx(1.4265)
    assert gateway.usage_cost({"prompt_tokens": 0, "completion_tokens": 0}) == 0
    with pytest.raises(AssertionError):
        gateway.usage_cost({"prompt_tokens": 1, "completion_tokens": 0, "prompt_cache_hit_tokens": 2})


def test_reservations_survive_restart_and_cap_blocks_before_io(tmp_path):
    g = gateway.Gateway(tmp_path)
    g.db.execute("INSERT INTO calls VALUES ('uncertain','reserved',4.999,'')")
    g.db.commit()
    g.db.close()
    g.lock_file.close()
    g = gateway.Gateway(tmp_path)
    class Request:
        async def json(self):
            return {"messages": [{"role": "user", "content": "proof"}], "max_tokens": 40000}
    with pytest.raises(web.HTTPPaymentRequired):
        asyncio.run(g.chat(Request()))
    assert g.total() == 4.999
    assert g.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 1
    g.db.close()
    g.lock_file.close()


def test_second_gateway_cannot_share_accounting_file(tmp_path):
    g = gateway.Gateway(tmp_path)
    with pytest.raises(BlockingIOError):
        gateway.Gateway(tmp_path)
    g.db.close()
    g.lock_file.close()
