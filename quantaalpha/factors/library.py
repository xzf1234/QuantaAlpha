"""
Factor library manager: save experiment output to unified JSON factor library.
Called from quantaalpha/pipeline/loop.py feedback step.
"""

import json
import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FACTOR_CACHE_DIR = os.environ.get(
    "FACTOR_CACHE_DIR",
    "data/results/factor_cache",
)


class FactorLibraryManager:
    """Manage unified factor library (CRUD)."""

    def __init__(self, library_path: str):
        self.library_path = Path(library_path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.library_path.exists():
            try:
                with open(self.library_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Factor library file corrupted, recreating: {e}")
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_factors": 0,
                "version": "1.0",
            },
            "factors": {},
        }

    def _save(self):
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        self.data["metadata"]["total_factors"] = len(self.data["factors"])
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.library_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2, default=str)

    def add_factors_from_experiment(
        self,
        experiment,
        experiment_id: str = "unknown",
        round_number: int = 0,
        hypothesis: Optional[str] = None,
        feedback: Any = None,
        initial_direction: Optional[str] = None,
        user_initial_direction: Optional[str] = None,
        planning_direction: Optional[str] = None,
        evolution_phase: str = "original",
        trajectory_id: str = "",
        parent_trajectory_ids: Optional[list] = None,
    ):
        """Extract factors from a QlibFactorExperiment and write to library."""
        if experiment is None:
            logger.warning("experiment is None, skip saving factors")
            return
        backtest_results = self._extract_backtest_results(experiment)
        feedback_dict = self._extract_feedback(feedback)
        sub_tasks = getattr(experiment, "sub_tasks", []) or []
        sub_workspaces = getattr(experiment, "sub_workspace_list", []) or []

        # Load return data once for the whole experiment (shared across factors)
        _shared_return_series = FactorLibraryManager._load_return_series()

        for idx, task in enumerate(sub_tasks):
            factor_name = getattr(task, "factor_name", getattr(task, "name", f"factor_{idx}"))
            factor_expr = getattr(task, "factor_expression", "")
            factor_desc = getattr(task, "factor_description", getattr(task, "description", ""))
            factor_form = getattr(task, "factor_formulation", "")

            factor_id = hashlib.md5(
                f"{factor_name}_{factor_expr}".encode()
            ).hexdigest()[:16]

            code = ""
            cache_location = {}
            ws_for_ic = None
            if idx < len(sub_workspaces):
                ws = sub_workspaces[idx]
                ws_for_ic = ws
                code_dict = getattr(ws, "code_dict", {})
                code = "\n".join(
                    f"File: {fname}\n\n{content}"
                    for fname, content in code_dict.items()
                )
                ws_path = getattr(ws, "workspace_path", None)
                if ws_path:
                    ws_path = Path(ws_path)
                    workspace_suffix = ""
                    for part in ws_path.parts:
                        if part.startswith("workspace_"):
                            workspace_suffix = part.replace("workspace_", "")
                            break
                    h5_file = ws_path / "result.h5"
                    cache_location = {
                        "workspace_suffix": workspace_suffix,
                        "workspace_path": str(ws_path.parent),
                        "factor_dir": ws_path.name,
                    }
                    if h5_file.exists():
                        cache_location["result_h5_path"] = str(h5_file)
                    else:
                        logger.warning(
                            f"result.h5 missing for {factor_name} ({h5_file}), will recompute from expression in backtest"
                        )

            # Build per-factor backtest results: start from combined experiment-level
            # metrics, then override IC metrics with individually computed values so
            # each factor shows distinct performance numbers rather than the same
            # combined-portfolio result.
            factor_backtest_results = dict(backtest_results)
            if ws_for_ic is not None:
                per_ic = self._compute_per_factor_ic_metrics(
                    ws_for_ic, factor_name, _shared_return_series
                )
                if per_ic:
                    factor_backtest_results.update(per_ic)

            factor_entry = {
                "factor_id": factor_id,
                "factor_name": factor_name,
                "factor_expression": factor_expr,
                "factor_implementation_code": code,
                "factor_description": factor_desc,
                "factor_formulation": factor_form,
                "cache_location": cache_location,
                "metadata": {
                    "experiment_id": experiment_id,
                    "round_number": round_number,
                    "evolution_phase": evolution_phase,
                    "trajectory_id": trajectory_id,
                    "parent_trajectory_ids": parent_trajectory_ids or [],
                    "hypothesis": str(hypothesis) if hypothesis else "",
                    "initial_direction": initial_direction or "",
                    "planning_direction": planning_direction or "",
                    "created_at": datetime.now().isoformat(),
                },
                "backtest_results": factor_backtest_results,
                "feedback": feedback_dict,
            }

            self.data["factors"][factor_id] = factor_entry

            if factor_expr and cache_location.get("result_h5_path"):
                self._sync_h5_to_md5_cache(factor_expr, cache_location["result_h5_path"])

        self._save()
        logger.info(
            f"Saved {len(sub_tasks)} factors to {self.library_path} (backtest_results: {len(backtest_results)} metrics)"
        )

    @staticmethod
    def _sync_h5_to_md5_cache(factor_expression: str, h5_path: str,
                                cache_dir: Optional[str] = None) -> bool:
        """Sync factor values from result.h5 to MD5 cache dir (.pkl). Returns True on success."""
        cache_dir = Path(cache_dir or DEFAULT_FACTOR_CACHE_DIR)
        h5_file = Path(h5_path)

        if not h5_file.exists():
            return False

        md5_key = hashlib.md5(factor_expression.encode()).hexdigest()
        pkl_file = cache_dir / f"{md5_key}.pkl"

        if pkl_file.exists():
            return True

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            result = pd.read_hdf(str(h5_file))
            result.to_pickle(pkl_file)
            logger.debug(f"Synced factor cache -> {pkl_file.name}")
            return True
        except Exception as e:
            logger.debug(f"Sync factor cache failed [{h5_path}]: {e}")
            return False

    @staticmethod
    def check_cache_status(library_path: str,
                           cache_dir: Optional[str] = None) -> dict:
        """Check cache status for each factor in library. Returns:
            {
                "total": int,
                "h5_cached": int,
                "md5_cached": int,
                "need_compute": int,
                "factors": [ { "factor_id", "factor_name", "status" }, ... ]
            }
        """
        cache_dir = Path(cache_dir or DEFAULT_FACTOR_CACHE_DIR)
        project_root = Path(library_path).resolve().parent.parent.parent

        def resolve_path(path: str) -> Path:
            candidate = Path(path)
            if candidate.is_absolute():
                return candidate
            root_candidate = project_root / candidate
            if root_candidate.exists():
                return root_candidate
            return candidate

        with open(library_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        factors = data.get("factors", {})
        total = len(factors)
        h5_cached = 0
        md5_cached = 0
        need_compute = 0
        details = []

        for fid, finfo in factors.items():
            expr = finfo.get("factor_expression", "")
            cloc = finfo.get("cache_location", {})
            h5_path = cloc.get("result_h5_path", "")

            status = "need_compute"
            # Check h5 cache
            if h5_path and resolve_path(h5_path).exists():
                status = "h5_cached"
                h5_cached += 1
            # Check MD5 cache
            elif expr:
                md5_key = hashlib.md5(expr.encode()).hexdigest()
                if (cache_dir / f"{md5_key}.pkl").exists():
                    status = "md5_cached"
                    md5_cached += 1

            if status == "need_compute":
                need_compute += 1

            details.append({
                "factor_id": fid,
                "factor_name": finfo.get("factor_name", fid),
                "status": status,
            })

        return {
            "total": total,
            "h5_cached": h5_cached,
            "md5_cached": md5_cached,
            "need_compute": need_compute,
            "factors": details,
        }

    @staticmethod
    def warm_cache_from_json(library_path: str,
                             cache_dir: Optional[str] = None) -> dict:
        """Walk factor library JSON and sync all available result.h5 to MD5 cache dir. Returns:
            { "total": int, "synced": int, "skipped": int, "failed": int,
              "already_cached": int, "no_source": int }
        """
        cache_dir_path = Path(cache_dir or DEFAULT_FACTOR_CACHE_DIR)
        project_root = Path(library_path).resolve().parent.parent.parent

        def resolve_path(path: str) -> Path:
            candidate = Path(path)
            if candidate.is_absolute():
                return candidate
            root_candidate = project_root / candidate
            if root_candidate.exists():
                return root_candidate
            return candidate

        with open(library_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        factors = data.get("factors", {})
        synced = 0
        skipped = 0
        failed = 0
        already_cached = 0
        no_source = 0

        for fid, finfo in factors.items():
            expr = finfo.get("factor_expression", "")
            cloc = finfo.get("cache_location", {})
            h5_path = cloc.get("result_h5_path", "")

            if not expr or not h5_path:
                no_source += 1
                skipped += 1
                continue

            md5_key = hashlib.md5(expr.encode()).hexdigest()
            pkl_file = cache_dir_path / f"{md5_key}.pkl"

            if pkl_file.exists():
                already_cached += 1
                skipped += 1
                continue

            resolved_h5_path = resolve_path(h5_path)
            if not resolved_h5_path.exists():
                failed += 1
                continue

            try:
                cache_dir_path.mkdir(parents=True, exist_ok=True)
                result = pd.read_hdf(str(resolved_h5_path))
                result.to_pickle(pkl_file)
                synced += 1
            except Exception:
                failed += 1

        return {
            "total": len(factors),
            "synced": synced,
            "skipped": skipped,
            "failed": failed,
            "already_cached": already_cached,
            "no_source": no_source,
        }

    @staticmethod
    def _load_return_series() -> Optional[pd.Series]:
        """Load the `$return` column from the configured factor data folder.

        Loaded once per experiment and passed to ``_compute_per_factor_ic_metrics``
        to avoid repeated HDF5 reads.  Returns ``None`` if unavailable.
        """
        try:
            from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
            data_dir = Path(FACTOR_COSTEER_SETTINGS.data_folder)
            if not data_dir.is_absolute():
                project_root = Path(__file__).resolve().parent.parent.parent
                data_dir = project_root / data_dir
            for fname in ("daily_pv.h5", "daily_pv_all.h5"):
                candidate = data_dir / fname
                if candidate.exists():
                    pv_df = pd.read_hdf(str(candidate))
                    if isinstance(pv_df, pd.DataFrame) and "$return" in pv_df.columns:
                        return pv_df["$return"]
        except Exception as e:
            logger.debug(f"_load_return_series: {e}")
        return None

    @staticmethod
    def _compute_per_factor_ic_metrics(
        ws, factor_name: str, return_series: Optional[pd.Series] = None
    ) -> dict:
        """Compute IC / RankIC / ICIR / RankICIR for a single factor.

        Factor values are obtained by calling ``ws.execute("All")``, which
        returns immediately from the pickle cache when available (the typical
        case: the runner already executed the factor before calling this method).
        Falls back to reading ``result.h5`` from the workspace directory when
        the cache is disabled.

        ``return_series`` should be pre-loaded once outside the per-factor loop
        via ``_load_return_series()``.  If ``None`` is passed, the method tries
        to load it from workspace hard-links / the data folder as a last resort.

        Returns an empty dict on any failure so callers always get a valid dict.
        """
        try:
            # ── 1. Get factor values ──────────────────────────────────────────
            factor_df = None

            # Primary: call execute("All") — uses pickle cache when available,
            # which is the common case after runner.py already ran the factors.
            try:
                result = ws.execute("All")
                if isinstance(result, tuple) and len(result) >= 2:
                    factor_df = result[1]
            except Exception:
                pass

            # Fallback: read result.h5 directly (no-cache / first-run path)
            if factor_df is None:
                ws_path = getattr(ws, "workspace_path", None)
                if ws_path is not None:
                    h5_file = Path(ws_path) / "result.h5"
                    if h5_file.exists():
                        try:
                            factor_df = pd.read_hdf(str(h5_file))
                        except Exception:
                            pass

            if factor_df is None:
                return {}

            # Normalise to a Series (first column if DataFrame)
            if isinstance(factor_df, pd.DataFrame):
                factor_series = factor_df.iloc[:, 0]
            elif isinstance(factor_df, pd.Series):
                factor_series = factor_df
            else:
                return {}

            # ── 2. Get return data ────────────────────────────────────────────
            if return_series is None:
                # Try hard-linked / symlinked files inside the workspace
                ws_path = getattr(ws, "workspace_path", None)
                if ws_path is not None:
                    for fname in ("daily_pv.h5", "daily_pv_all.h5", "daily_pv_debug.h5"):
                        pv_file = Path(ws_path) / fname
                        if pv_file.exists():
                            try:
                                pv_df = pd.read_hdf(str(pv_file))
                                if isinstance(pv_df, pd.DataFrame) and "$return" in pv_df.columns:
                                    return_series = pv_df["$return"]
                                    break
                            except Exception:
                                continue
                # Ultimate fallback: load from data folder
                if return_series is None:
                    return_series = FactorLibraryManager._load_return_series()

            if return_series is None:
                return {}

            # ── 3. Normalise MultiIndex to (datetime, instrument) ─────────────
            def _ensure_datetime_first(s: pd.Series) -> pd.Series:
                if s.index.nlevels == 2:
                    names = list(s.index.names)
                    if names[0] != "datetime" and "datetime" in names:
                        return s.swaplevel().sort_index()
                return s

            factor_series = _ensure_datetime_first(factor_series)
            return_series = _ensure_datetime_first(return_series)

            # ── 4. Cross-sectional IC per date (factor[t] vs return[t+1]) ─────
            dates = sorted(factor_series.index.get_level_values("datetime").unique())
            ic_list: list = []
            rank_ic_list: list = []

            for i, dt in enumerate(dates):
                if i + 1 >= len(dates):
                    continue
                next_dt = dates[i + 1]
                try:
                    f_cross = factor_series.xs(dt, level="datetime").dropna()
                    ret_dates = return_series.index.get_level_values("datetime")
                    if next_dt not in ret_dates:
                        continue
                    r_cross = return_series.xs(next_dt, level="datetime").dropna()

                    common = f_cross.index.intersection(r_cross.index)
                    if len(common) < 5:
                        continue

                    f_aligned = f_cross.loc[common]
                    r_aligned = r_cross.loc[common]
                    mask = np.isfinite(f_aligned) & np.isfinite(r_aligned)
                    f_aligned = f_aligned[mask]
                    r_aligned = r_aligned[mask]
                    if len(f_aligned) < 5:
                        continue

                    ic_val = float(f_aligned.corr(r_aligned, method="pearson"))
                    ric_val = float(f_aligned.corr(r_aligned, method="spearman"))
                    if np.isfinite(ic_val):
                        ic_list.append(ic_val)
                    if np.isfinite(ric_val):
                        rank_ic_list.append(ric_val)
                except Exception:
                    continue

            if not ic_list:
                return {}

            ic_arr = np.array(ic_list)
            ric_arr = np.array(rank_ic_list) if rank_ic_list else ic_arr
            ic_mean = float(np.mean(ic_arr))
            ic_std = float(np.std(ic_arr))
            ric_mean = float(np.mean(ric_arr))
            ric_std = float(np.std(ric_arr))

            return {
                "IC": round(ic_mean, 6),
                "ICIR": round(ic_mean / (ic_std + 1e-10), 4),
                "Rank IC": round(ric_mean, 6),
                "Rank ICIR": round(ric_mean / (ric_std + 1e-10), 4),
            }
        except Exception as e:
            logger.warning(f"Per-factor IC computation failed for {factor_name}: {e}")
            return {}

    @staticmethod
    def _extract_backtest_results(experiment) -> dict:
        """Extract backtest metrics from experiment.result (pandas Series) as dict."""
        result = getattr(experiment, "result", None)
        if result is None:
            return {}
        if isinstance(result, pd.Series):
            out = {}
            for key, val in result.items():
                # NaN/Inf -> None for JSON
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val) or np.isinf(val):
                        out[str(key)] = None
                    else:
                        out[str(key)] = round(float(val), 8)
                else:
                    out[str(key)] = val
            return out

        if isinstance(result, pd.DataFrame):
            try:
                return {
                    str(k): round(float(v), 8) if isinstance(v, (float, np.floating)) and not np.isnan(v) else None
                    for k, v in result.iloc[:, 0].items()
                }
            except Exception:
                pass

        if isinstance(result, dict):
            return result

        return {}

    @staticmethod
    def _extract_feedback(feedback) -> dict:
        """Convert feedback object to serializable dict."""
        if feedback is None:
            return {}
        if isinstance(feedback, dict):
            return feedback

        out = {}
        for attr in ["observations", "hypothesis_evaluation", "decision", "reason",
                      "new_hypothesis", "feedback_str"]:
            val = getattr(feedback, attr, None)
            if val is not None:
                out[attr] = str(val) if not isinstance(val, (bool, int, float)) else val
        if not out:
            out["raw"] = str(feedback)
        return out
