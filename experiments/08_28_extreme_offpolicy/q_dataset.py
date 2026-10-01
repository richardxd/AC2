"""SPQReadinessDataset: the training dataset (``data.custom_cls``) of the AC2 runs.

Self-contained copy: ``data.custom_cls`` loads by FILE path, so cross-experiment imports are
fragile. In TRAIN mode each step's batch is built by the replay harness (sp_replay):

  * statement rows are refill rows: fresh problems with an empty prefix whose rollouts
    refill the replay buffer (dropped before the loss when SP_SCRATCH_INFLOW_ONLY=1);
  * replay rows are stored trajectories cut at a prefix; replay slots the buffer cannot cover
    become fill rows (``cold_scratch`` while the buffer is still empty);
  * the sp_q_readiness harness (critic buffer seed, reference bank, readiness table, routing)
    is installed right after sp_replay.install -- the critic state lives in the SAME driver
    process as the replay state;
  * per-row critic routing is stamped: ``extra_info.sp_q_route`` in {"full","short","audit"}
    from the readiness table + the pinned per-step audit draw, and a top-level
    ``sp_q_max_new_tokens`` cap (min(b, C - a_i) for "short" rows, where b is the action-chunk
    length SP_Q_BUDGET_G, C is SP_MAX_RESPONSE_LEN and a_i the prefix length; 0 otherwise)
    consumed by sp_prefix_agent. Statement rows are always "full"/0;
  * when SP_SCRATCH_INFLOW_ONLY=1 every row carries ``sp_rollout_n`` -- the refill stamp
    (see below) for statement rows, 0 (= config rollout.n) for replay rows -- consumed by
    the trainer's per-row repeat sites.

In VAL mode it delegates to plain RLHFDataset. Requires SP_REPLAY_ENABLE=1 AND SP_Q_ENABLE=1
(asserted).

Burst refill schedule (08_28_extreme_offpolicy): when ``sp_inflow_burst_period`` (config,
pinned by the runner from SP_INFLOW_BURST_PERIOD) is > 0, the statement-row stamp follows a
schedule instead of being the constant 1:

  * on a BURST dataset step (step % period == phase) every statement row is stamped
    ``sp_rollout_n = burst_n`` -- 192 rows x 10 = 1,920 refill trajectories generated from
    the single policy snapshot of that step, all offered to admission at its reward site;
  * on every OTHER step statement rows are stamped ``sp_rollout_n = -1`` -- the trainer's
    per-row repeat resolver turns a NEGATIVE stamp into repeat-count 0, which removes the
    row from both the gen batch and the driver batch: never generated, never judged,
    never admitted.

Replay / cold_scratch rows keep the 0 stamp (= uniform config rollout.n) on every step, so
the trained rows are untouched by the schedule. With the burst config absent the stamp is
the constant 1, as in the main run.
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
        # ---- burst refill schedule -----------------------------------------------------
        # Read from the COMPOSED config (+data.sp_inflow_burst_*), not the environment: the
        # dataset is constructed inside the TaskRunner Ray actor, where an unforwarded env
        # var silently no-ops. 0 / absent = no schedule -> the constant-1 stamp of the main
        # run.
        self._burst_period = int(config.get("sp_inflow_burst_period", 0) or 0)
        self._burst_n = int(config.get("sp_inflow_burst_n", 1) or 1)
        self._burst_phase = int(config.get("sp_inflow_burst_phase", 0) or 0)
        if self._burst_period:
            assert self._burst_n >= 1, self._burst_n
            assert 0 <= self._burst_phase < self._burst_period, (
                self._burst_phase, self._burst_period)
            assert SPQReadinessDataset._INFLOW_ONLY, (
                "sp_inflow_burst_period is set but SP_SCRATCH_INFLOW_ONLY is off: without "
                "the per-row repeat stamp the schedule cannot reach the trainer, and every "
                "statement row would be generated at the uniform config n AND trained."
            )
            print(
                f"[sp_inflow] BURST schedule: {self.harness.n_orig} statement rows x "
                f"{self._burst_n} rollouts on dataset steps == {self._burst_phase} "
                f"(mod {self._burst_period}); on all other steps statement rows are stamped "
                f"-1 and DROPPED pre-generation (zero inflow)",
                flush=True,
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

    def _inflow_stamp(self, step: int) -> int:
        """Per-row ``sp_rollout_n`` for a statement (inflow) row at this dataset step.

        No burst schedule -> 1 (one refill rollout per row on every step). With one:
        burst_n on burst steps; -1 (drop the row entirely) on every other step. A pure
        function of the dataset step, so a resume rebuilds the same schedule."""
        if not getattr(self, "_burst_period", 0):
            return 1
        return self._burst_n if (step % self._burst_period) == self._burst_phase else -1

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
                inflow_n=self._inflow_stamp(step),
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
                # a statement-tagged fill row is inflow like any other statement row, so it
                # follows the burst schedule; cold_scratch ignores inflow_n (stamped 0).
                inflow_n=self._inflow_stamp(step),
            )

        qid = qids[rslot]
        entry = h.entry_for_slot(qid, step, rslot)
        prefix_ids, cut_fraction = h.prefix_cut(entry, step, rslot)
        parquet_idx = h.row_of_qid[qid]
        # Critic routing: readiness + pinned audit draw, per replay slot.
        route = self.q_harness.route_plan_for_step(step)[rslot]
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
                      entry_id, fill, q_route, q_cap, inflow_n: int = 1) -> dict:
        row["sp_prefix_token_ids"] = [int(t) for t in prefix_ids]
        row["sp_q_max_new_tokens"] = int(q_cap)   # uniform schema; 0 = no cap
        row["agent_name"] = "sp_prefix_agent"
        if SPQReadinessDataset._INFLOW_ONLY:
            # statement (refill) rows carry the schedule's stamp (burst_n on burst steps,
            # -1 = drop otherwise; constant 1 with no schedule); 0 = config default n for
            # trained rows.
            # Stamped BOTH top-level and in extra_info: the top-level key was observed to
            # vanish from the live dataloader batch (cause unresolved) while extra_info
            # provably survives to the trainer, so extra_info is authoritative.
            row["sp_rollout_n"] = int(inflow_n) if source_type == "statement" else 0
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
            }
        )
        if SPQReadinessDataset._INFLOW_ONLY:
            ei["sp_rollout_n"] = int(inflow_n) if source_type == "statement" else 0
        row["extra_info"] = ei
        return row
