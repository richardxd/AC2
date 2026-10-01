"""SPReplayNoQDataset -- AC2's training dataset with the critic layer REMOVED (Prefix GRPO).

Derived from AC2's q_dataset.py (08_13_tiedq_seed192) by deleting the generative-Q layer and
nothing else (``data.custom_cls`` loads by FILE path, so this is a self-contained copy rather
than an import). The replay side -- (step,slot) stream, global-FIFO buffer, grain-quantized
prefix cuts, cold bootstrap, the inflow-only rollout-n stamp -- is identical to AC2's, so the
two configurations differ in ONE factor.

WHAT IS GONE
------------
AC2 installs sp_q_readiness beside sp_replay and stamps each replay slot with a route in
{"full","short","audit"} plus an ``sp_q_max_new_tokens`` cap. Only "short" rows are capped (at
SP_Q_BUDGET_G new tokens, the action chunk) and only they take their value from the critic;
"audit" and "full" rows run to termination and get a judge score.

Here there is no critic harness and no routing draw: EVERY row is "full" with cap 0, so every
rollout runs to termination and every reward is a judge score.

THE ROW SCHEMA IS UNCHANGED ON PURPOSE. ``sp_q_max_new_tokens`` (always 0) and
``extra_info.sp_q_route`` (always "full") are still stamped, because the trainer and
sp_prefix_agent read both keys unconditionally -- dropping them would KeyError rather than
degrade, and keeping them means the metrics/dashboard parsers see the same shape as AC2 with
the route histogram pinned to 100% full.

Requires SP_REPLAY_ENABLE=1 and refuses to run with SP_Q_ENABLE set (asserted both ways): a
half-wired configuration -- critic env on but no critic harness installed, or this dataset
paired with the critic trainer hooks -- would silently produce a run that is neither AC2 nor
Prefix GRPO.
"""
from __future__ import annotations

import os

from verl.trainer.ppo import difficulty as _difficulty
from verl.trainer.ppo import sp_replay
from verl.utils.dataset.rl_dataset import RLHFDataset


def _normalize_files(files) -> list[str]:
    if files is None:
        return []
    if isinstance(files, str):
        return [files]
    return [str(f) for f in files]


def _q_env_on() -> bool:
    return os.environ.get("SP_Q_ENABLE", "0") in ("1", "true", "True", "yes")


class SPReplayNoQDataset(RLHFDataset):
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
            f"[sp_replay-noq] SPReplayNoQDataset mode: "
            f"{'TRAIN (step,slot), every row full (p=1)' if self.sp_train_mode else 'VAL (plain RLHFDataset delegation)'} "
            f"for data_files={_normalize_files(data_files)}",
            flush=True,
        )
        if not self.sp_train_mode:
            return

        assert sp_replay.enabled(), (
            "SPReplayNoQDataset requires SP_REPLAY_ENABLE=1 -- the ablation keeps the replay "
            "buffer and removes only Q."
        )
        # The inverse of AC2's assert. There, Q env off with that dataset selected
        # meant the Q hooks would no-op; here, Q env ON would mean the trainer
        # installs Q hooks that this dataset never routes for -- a run that is
        # neither configuration.
        assert not _q_env_on(), (
            "SP_Q_ENABLE is set, but this is the NO-Q ablation and no Q harness is installed. "
            "Q trainer hooks would run against rows that are all routed 'full', producing a "
            "run that is neither the main arm nor the ablation. Unset it, or use the main "
            "run's q_dataset.py instead."
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
        print(
            f"[sp_replay-noq] train mode: {n_rows} statements, n_base={self.harness.n_base} "
            f"(orig {self.harness.n_orig} + replay {self.harness.n_replay}); "
            f"NO Q harness -- every row routes 'full' (p=1)",
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
            )

        rslot = slot - h.n_orig
        qids, fill_indices = h.replay_plan_for_step(step)
        if rslot >= len(qids):  # coverage shortfall -> uniform statement fill row
            parquet_idx = fill_indices[rslot - len(qids)]
            # COLD BOOTSTRAP (as in AC2): an EMPTY buffer yields zero replay qids, so
            # every replay slot becomes a fill row. Those rows must be TRAINED -- tagged
            # "statement" they would be inflow-only, dropped before the loss, and the
            # step would have nothing left to train on.
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
            )

        qid = qids[rslot]
        entry = h.entry_for_slot(qid, step, rslot)
        prefix_ids, cut_fraction = h.prefix_cut(entry, step, rslot)
        parquet_idx = h.row_of_qid[qid]
        # NO route draw: AC2 consults q_harness.route_plan_for_step(step)[rslot] here. Every replay
        # row runs to termination.
        return self._finalize_row(
            super().__getitem__(parquet_idx),
            step=step,
            qid=qid,
            source_type="replay",
            prefix_ids=prefix_ids,
            cut_fraction=cut_fraction,
            entry_id=entry["entry_id"],
            fill=False,
        )

    _INFLOW_ONLY = os.environ.get("SP_SCRATCH_INFLOW_ONLY", "0") in ("1", "true", "True")

    @staticmethod
    def _finalize_row(row: dict, *, step, qid, source_type, prefix_ids, cut_fraction,
                      entry_id, fill) -> dict:
        row["sp_prefix_token_ids"] = [int(t) for t in prefix_ids]
        # Always 0 / always "full": the schema keys stay because sp_prefix_agent and the
        # trainer read them unconditionally, and because the dashboard's route histogram then
        # renders as 100% full rather than as missing data.
        row["sp_q_max_new_tokens"] = 0
        row["agent_name"] = "sp_prefix_agent"
        if SPReplayNoQDataset._INFLOW_ONLY:
            # Statement rows are inflow-only -> 1 rollout; 0 = config default n.
            # Stamped BOTH top-level and in extra_info: the top-level key has been
            # observed to vanish from the live dataloader batch (cause unresolved)
            # while extra_info survives to the trainer, so extra_info is authoritative.
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
                "sp_q_route": "full",
            }
        )
        if SPReplayNoQDataset._INFLOW_ONLY:
            ei["sp_rollout_n"] = 1 if source_type == "statement" else 0
        row["extra_info"] = ei
        return row
