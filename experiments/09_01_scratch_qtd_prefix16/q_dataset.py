"""Training dataset class SPQReadinessDataset (``data.custom_cls``, loaded by file path).

Produces the training batch as (step, slot) rows: n_orig fresh problems for the refill lane and
n_replay prefixes cut from trajectories in the replay buffer (sp_replay). It installs the
sp_q_readiness harness (critic-buffer seed, reference bank, readiness table, routing) right
after sp_replay.install, so the critic state lives in the same driver process as the replay
state, and stamps per-row routing: ``extra_info.sp_q_route`` in {"full", "short", "audit"} from
the readiness table and the pinned per-step audit draw, plus a top-level
``sp_q_max_new_tokens`` cap (min(b, C - a_i) for "short" rows, where C is SP_MAX_RESPONSE_LEN and
a_i the prefix length; 0 otherwise) consumed by sp_prefix_agent. Refill rows are always
"full"/0. With SP_SCRATCH_INFLOW_ONLY=1 every row also carries ``sp_rollout_n``: 1 for refill
(statement) rows, 0 (= config rollout.n) for replay rows, consumed by the trainer's per-row
repeat sites.

No-group TD (this run): when the harness has sp_q_td_enable=1 and a replay slot is routed
"short" (ready), the slot is materialized as td_siblings DISTINCT cuts drawn by prefix_cuts_td
(own RNG purpose, finer sp_q_td_cut_grain), stamped as sp_td_cuts/sp_td_caps/sp_td_traj_ids
(uniform schema: empty on every other row) plus extra_info sp_q_td_group/sp_td_cuts/sp_td_len
for the trainer's driver-side rewrite. With td_enable=0 none of these fields carry content.

Validation rows delegate to the plain RLHFDataset. Requires SP_REPLAY_ENABLE=1 and
SP_Q_ENABLE=1 (asserted).
"""
from __future__ import annotations

import os

from verl.trainer.ppo import difficulty as _difficulty
from verl.trainer.ppo import sp_q_readiness, sp_replay
from verl.utils.dataset.rl_dataset import RLHFDataset


def _normalize_files(files) -> list[str]:
    if files is None:
        return []
    if isinstance(files, str):
        return [files]
    return [str(f) for f in files]


class SPQReadinessDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )
        train_files = _normalize_files(config.get("train_files"))
        self.sp_train_mode = _normalize_files(data_files) == train_files and bool(train_files)
        print(
            f"[sp_q] SPQReadinessDataset mode: "
            f"{'TRAIN (step,slot)+Q routing' if self.sp_train_mode else 'VAL (plain RLHFDataset delegation)'} "
            f"for data_files={_normalize_files(data_files)}",
            flush=True,
        )
        if not self.sp_train_mode:
            return

        assert sp_replay.enabled(), (
            "SPQReadinessDataset requires SP_REPLAY_ENABLE=1 (the Q mechanism rides on the "
            "replay dataset's (step,slot) stream)."
        )
        assert sp_q_readiness.enabled(), (
            "SPQReadinessDataset is the train dataset but SP_Q_ENABLE is not set — the Q "
            "trainer hooks would silently no-op; refuse to run half-wired."
        )
        assert not _difficulty.enabled(), (
            "SP_DIFF_SAMPLING is on: incompatible with the deterministic (step,slot) dataset."
        )
        assert not config.get("shuffle", True), "replay dataset requires data.shuffle=False"
        assert int(config.get("dataloader_num_workers", 8)) == 0, (
            "replay dataset requires data.dataloader_num_workers=0 (lazy driver-process rows)"
        )

        n_rows = super().__len__()
        row_qids = [
            _difficulty.qid_from_messages(self.dataframe[i][self.prompt_key]) for i in range(n_rows)
        ]
        self.harness = sp_replay.install(config, num_statements=n_rows, row_qids=row_qids)
        self._row_qids = row_qids

        def _raw_prompt_of_qid(qid: str):
            idx = self.harness.row_of_qid.get(qid)
            assert idx is not None, f"sp_q: qid {qid[:12]}... has no train-parquet row"
            return self.dataframe[idx][self.prompt_key]

        self.q_harness = sp_q_readiness.install(
            config, tokenizer=tokenizer, replay_harness=self.harness,
            raw_prompt_of_qid=_raw_prompt_of_qid,
        )
        print(
            f"[sp_q] train mode: {n_rows} statements, n_base={self.harness.n_base} "
            f"(orig {self.harness.n_orig} + replay {self.harness.n_replay}), "
            f"g={self.q_harness.budget_g}, ready_thresh={self.q_harness.ready_thresh}, "
            f"fifo_cap={self.q_harness.fifo_cap}, train_n={self.q_harness.train_n}, "
            f"ctx_limit={self.q_harness.ctx_limit}, q_rng_seed={self.q_harness.rng_seed}",
            flush=True,
        )

    # -- length / rows -------------------------------------------------------

    def __len__(self):
        if not getattr(self, "sp_train_mode", False):
            return super().__len__()
        return self.harness.local_steps * self.harness.n_base

    def __getitem__(self, item):
        if not getattr(self, "sp_train_mode", False):
            return super().__getitem__(item)

        h = self.harness
        assert h._finalized, (
            "sp_replay row build before on_checkpoint_load finalized the cursor — "
            "_load_checkpoint must run before the first dataloader iteration"
        )
        local, slot = divmod(int(item), h.n_base)
        step = h.resume_base + local

        if slot < h.n_orig:
            parquet_idx = h.statement_indices(step)[slot]
            return self._finalize_row(
                super().__getitem__(parquet_idx),
                step=step,
                qid=self._row_qids[parquet_idx],
                source_type="statement",
                prefix_ids=[],
                cut_fraction=0.0,
                entry_id="",
                fill=False,
                q_route="full",
                q_cap=0,
            )

        rslot = slot - h.n_orig
        qids, fill_indices = h.replay_plan_for_step(step)
        if rslot >= len(qids):  # coverage shortfall -> uniform statement fill row
            parquet_idx = fill_indices[rslot - len(qids)]
            # COLD BOOTSTRAP: an EMPTY buffer (no seed, before the first admission)
            # yields zero replay qids, so every replay slot becomes a fill row. Those rows
            # must be TRAINED -- tagged "statement" they would be inflow-only, dropped
            # before the loss, and the step would have nothing left to train on at all.
            # "cold_scratch" = a fresh problem generated at the full rollout count and
            # trained like a replay row, but with no prefix and no parent trajectory.
            _cold = bool(getattr(h, "cold_bootstrap", False)) and not qids
            return self._finalize_row(
                super().__getitem__(parquet_idx),
                step=step,
                qid=self._row_qids[parquet_idx],
                source_type="cold_scratch" if _cold else "statement",
                prefix_ids=[],
                cut_fraction=0.0,
                entry_id="",
                fill=True,
                q_route="full",
                q_cap=0,
            )

        qid = qids[rslot]
        entry = h.entry_for_slot(qid, step, rslot)
        parquet_idx = h.row_of_qid[qid]
        # Critic routing: readiness + the pinned per-step audit draw, per replay slot.
        route = self.q_harness.route_plan_for_step(step)[rslot]

        # ---- no-group TD (this run's mechanism): a READY slot does not
        # run a sibling group. The dataset stamps td_siblings distinct cuts of this
        # entry's trajectory (plus the trajectory itself and the per-cut short caps); the
        # trainer's per-copy rewrite then turns the slot's rollout.n copies into
        # single-rollout requests at those cuts, each its own uid. The row's OWN
        # prefix/cap are cuts[0]'s (rewritten anyway; kept coherent so any consumer that
        # reads the un-rewritten row sees a real configuration, not a placeholder).
        # The "cut" rng stream is NOT consumed for a TD slot -- rng_for is purpose-keyed
        # and prefix_cuts_td draws under its own "td_cut" purpose.
        if getattr(self.q_harness, "td_enable", False) and route == "short":
            n_sib = int(self.q_harness.td_siblings)
            grain = int(getattr(self.q_harness, "td_cut_grain", 0)) or None
            td_cuts = h.prefix_cuts_td(entry, step, rslot, n_sib, grain=grain)
            traj = [int(x) for x in entry["response_token_ids"]]
            td_caps = [int(self.q_harness.budget_cap_for("short", c)) for c in td_cuts]
            prefix_ids = traj[: td_cuts[0]]
            cut_fraction = (td_cuts[0] / len(traj)) if traj else 0.0
            q_cap = td_caps[0]
            return self._finalize_row(
                super().__getitem__(parquet_idx),
                step=step,
                qid=qid,
                source_type="replay",
                prefix_ids=prefix_ids,
                cut_fraction=cut_fraction,
                entry_id=entry["entry_id"],
                fill=False,
                q_route=route,
                q_cap=q_cap,
                td_cuts=td_cuts,
                td_caps=td_caps,
                td_traj_ids=traj,
                td_slot=rslot,
            )

        prefix_ids, cut_fraction = h.prefix_cut(entry, step, rslot)
        q_cap = self.q_harness.budget_cap_for(route, len(prefix_ids)) if route == "short" else 0
        return self._finalize_row(
            super().__getitem__(parquet_idx),
            step=step,
            qid=qid,
            source_type="replay",
            prefix_ids=prefix_ids,
            cut_fraction=cut_fraction,
            entry_id=entry["entry_id"],
            fill=False,
            q_route=route,
            q_cap=q_cap,
        )

    _INFLOW_ONLY = os.environ.get("SP_SCRATCH_INFLOW_ONLY", "0") in ("1", "true", "True")

    @staticmethod
    def _finalize_row(row: dict, *, step, qid, source_type, prefix_ids, cut_fraction,
                      entry_id, fill, q_route, q_cap,
                      td_cuts=None, td_caps=None, td_traj_ids=None, td_slot=None) -> dict:
        row["sp_prefix_token_ids"] = [int(t) for t in prefix_ids]
        row["sp_q_max_new_tokens"] = int(q_cap)   # uniform schema; 0 = no cap
        # no-group TD fields: UNIFORM SCHEMA like sp_q_max_new_tokens above -- the default
        # collate stacks per key, so a key present on some rows only would crash the
        # dataloader. Empty lists mean "not a TD slot"; the trainer's rewrite keys on
        # non-emptiness (and, driver-side, on extra_info's sp_q_td_group).
        row["sp_td_cuts"] = [int(c) for c in (td_cuts or [])]
        row["sp_td_caps"] = [int(c) for c in (td_caps or [])]
        row["sp_td_traj_ids"] = [int(t) for t in (td_traj_ids or [])]
        row["agent_name"] = "sp_prefix_agent"
        if SPQReadinessDataset._INFLOW_ONLY:
            # Refill (statement) rows are inflow-only -> 1 rollout; 0 = config default n.
            # Stamped BOTH top-level and in extra_info: the top-level key has been observed
            # to vanish from the live dataloader batch while extra_info survives to the
            # trainer, so extra_info is authoritative.
            row["sp_rollout_n"] = 1 if source_type == "statement" else 0
        ei = dict(row.get("extra_info") or {})
        ei.update(
            {
                "sp_dataset_step": int(step),
                "sp_qid": qid,
                "sp_source_type": source_type,
                "sp_prefix_len": len(prefix_ids),
                "sp_cut_fraction": float(cut_fraction),
                "sp_entry_id": entry_id,
                "sp_fill": bool(fill),
                "sp_q_route": q_route,
                # driver-side rewrite bookkeeping: cuts + trajectory length ride in
                # extra_info because the top-level sp_td_* fields are popped into the GEN
                # batch and never reach the driver batch's repeat site. sp_q_td_group
                # marks the SLOT; the rewrite marks each copy sp_q_td_row=1.
                "sp_q_td_group": 1 if td_cuts else 0,
                "sp_td_cuts": [int(c) for c in (td_cuts or [])],
                "sp_td_len": len(td_traj_ids or []),
                # slot id for the Bellman-backup admission cap (sp_q_td_admit_per_slot):
                # the same qid can occupy two slots via the draw's wraparound, and
                # (qid, entry_id) could collide across them, so the cap keys on the slot.
                "sp_td_slot": int(td_slot) if td_slot is not None else -1,
            }
        )
        if SPQReadinessDataset._INFLOW_ONLY:
            ei["sp_rollout_n"] = 1 if source_type == "statement" else 0
        row["extra_info"] = ei
        return row
