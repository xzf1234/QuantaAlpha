#!/usr/bin/env python3
"""Build a stricter reliable factor library from existing and template factors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Candidate:
    name: str
    expression: str
    description: str
    source: str
    original: Optional[dict[str, Any]] = None


def _metric(results: dict[str, Any], names: Iterable[str]) -> Optional[float]:
    for name in names:
        value = results.get(name)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _load_existing_candidates(paths: list[Path]) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for fid, finfo in data.get("factors", {}).items():
            expr = (finfo.get("factor_expression") or "").strip()
            if not expr or expr in seen:
                continue
            seen.add(expr)
            candidates.append(
                Candidate(
                    name=finfo.get("factor_name") or fid,
                    expression=expr,
                    description=finfo.get("factor_description") or "",
                    source=path.name,
                    original=finfo,
                )
            )
    return candidates


def _template_candidates() -> list[Candidate]:
    specs = [
        ("Amount_Momentum_20D", "ZSCORE(TS_SUM($amount, 20) / (TS_MEAN($amount, 60) + 1e-8))", "Amount expansion against medium-term amount baseline."),
        ("Amount_Momentum_60D", "ZSCORE(TS_SUM($amount, 60) / (TS_MEAN($amount, 120) + 1e-8))", "Persistent amount expansion over 60 days."),
        ("Amount_Price_Divergence_20D", "RANK(TS_PCTCHANGE($amount, 20)) - RANK(TS_PCTCHANGE($close, 20))", "Capital participation rising faster than price."),
        ("Amount_Price_Divergence_60D", "RANK(TS_PCTCHANGE($amount, 60)) - RANK(TS_PCTCHANGE($close, 60))", "Medium-term amount-price divergence."),
        ("Illiquidity_Reversal_20D", "RANK(INV(TS_MEAN($amount, 20) + 1e-8)) * RANK(-1 * TS_SUM($return, 20))", "Low-liquidity reversal composite."),
        ("Turnover_Proxy_Reversal_20D", "RANK(INV(TS_MEAN(DIVIDE($amount, $close + 1e-8), 20) + 1e-8)) * RANK(-1 * TS_SUM($return, 20))", "Volume turnover proxy reversal."),
        ("Vwap_Pressure_10D", "ZSCORE(TS_MEAN(($close - $vwap) / ($vwap + 1e-8), 10))", "Close versus VWAP buying pressure."),
        ("Vwap_Pressure_Reversal_20D", "RANK(-1 * TS_MEAN(($close - $vwap) / ($vwap + 1e-8), 20))", "Reversal of closing pressure versus VWAP."),
        ("Low_Vol_Amount_60D", "RANK(INV(TS_STD($return, 60) + 1e-8)) + RANK(TS_MEAN($amount, 20) / (TS_MEAN($amount, 120) + 1e-8))", "Low volatility with supportive amount."),
        ("Quality_Price_Stability_60D", "RANK(INV(TS_STD($return, 60) + 1e-8)) + RANK(INV($close / (TS_MAX($close, 252) + 1e-8)))", "Stable stocks near lower relative 52-week position."),
        ("MoneyFlow_Momentum_20D", "ZSCORE(TS_MEAN($main_net_inflow, 20) / (TS_MEAN(ABS($amount), 20) + 1e-8))", "Main money inflow intensity; active only when external field exists."),
        ("MoneyFlow_Reversal_20D", "RANK(TS_MEAN($main_net_inflow, 20)) + RANK(-1 * TS_SUM($return, 20))", "Main inflow combined with price reversal; active only when external field exists."),
        ("Valuation_Momentum_PE_60D", "RANK(INV($pe_ttm + 1e-8)) + RANK(TS_SUM($return, 60))", "Cheap valuation plus medium-term momentum; active only when external field exists."),
        ("PB_Quality_Momentum", "RANK(INV($pb + 1e-8)) + RANK($roe) + RANK(TS_SUM($return, 60))", "Value-quality-momentum composite; active only when external fields exist."),
        ("Unlock_Pressure_Reversal", "RANK(INV($unlock_ratio + 1e-8)) + RANK(-1 * TS_SUM($return, 20))", "Low unlock pressure reversal; active only when external field exists."),
    ]
    return [
        Candidate(name=name, expression=expr, description=desc, source="template")
        for name, expr, desc in specs
    ]


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _future_return(data: pd.DataFrame) -> pd.Series:
    ret = data["$close"] / data.groupby(level="instrument")["$close"].shift(1) - 1
    return ret.groupby(level="instrument").shift(-1)


def _daily_ic(factor: pd.Series, label: pd.Series, method: str) -> pd.Series:
    aligned = pd.concat([factor.rename("factor"), label.rename("label")], axis=1)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)

    values = []
    dates = []
    for dt, group in aligned.groupby(level="datetime"):
        if len(group) < 20:
            continue
        corr = group["factor"].corr(group["label"], method=method)
        if corr is not None and math.isfinite(corr):
            dates.append(pd.Timestamp(dt))
            values.append(float(corr))
    return pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()


def _split_stats(ic: pd.Series, ric: pd.Series, segments: dict[str, list[str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for split_name, (start, end) in segments.items():
        mask = (ric.index >= pd.Timestamp(start)) & (ric.index <= pd.Timestamp(end))
        split_ric = ric.loc[mask]
        split_ic = ic.loc[mask]
        if len(split_ric) == 0:
            out[f"{split_name}_rank_ic"] = np.nan
            out[f"{split_name}_rank_icir"] = np.nan
            out[f"{split_name}_ic"] = np.nan
            continue
        out[f"{split_name}_rank_ic"] = float(split_ric.mean())
        out[f"{split_name}_rank_icir"] = float(split_ric.mean() / (split_ric.std() + 1e-12))
        out[f"{split_name}_ic"] = float(split_ic.mean()) if len(split_ic) else np.nan
    return out


def _year_stats(ric: pd.Series) -> dict[str, float]:
    if ric.empty:
        return {"positive_year_ratio": 0.0, "min_year_rank_ic": np.nan}
    yearly = ric.groupby(ric.index.year).mean()
    return {
        "positive_year_ratio": float((yearly > 0).mean()),
        "min_year_rank_ic": float(yearly.min()),
    }


def _score(stats: dict[str, float]) -> float:
    return (
        stats.get("test_rank_ic", 0.0) * 2.5
        + stats.get("valid_rank_ic", 0.0) * 1.5
        + stats.get("all_rank_ic", 0.0)
        + max(stats.get("positive_year_ratio", 0.0) - 0.5, 0) * 0.02
    )


def _passes_gate(stats: dict[str, float], strict: bool) -> bool:
    if stats.get("coverage", 0.0) < 0.55:
        return False
    if stats.get("all_rank_ic", 0.0) < (0.008 if strict else 0.005):
        return False
    if stats.get("valid_rank_ic", 0.0) < (0.002 if strict else -0.001):
        return False
    if stats.get("test_rank_ic", 0.0) < (0.005 if strict else 0.002):
        return False
    if stats.get("positive_year_ratio", 0.0) < (0.60 if strict else 0.50):
        return False
    return True


def _safe_name(name: str, existing: set[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") if "re" in globals() else name
    if not cleaned:
        cleaned = "Factor"
    base = cleaned
    idx = 2
    while cleaned in existing:
        cleaned = f"{base}_{idx}"
        idx += 1
    existing.add(cleaned)
    return cleaned


def _build_factor_entry(candidate: Candidate, expression: str, stats: dict[str, float]) -> dict[str, Any]:
    original = candidate.original or {}
    factor_id = hashlib.md5(f"{candidate.name}_{expression}".encode()).hexdigest()[:16]
    backtest_results = dict(original.get("backtest_results") or {})
    backtest_results.update(
        {
            "IC": round(float(stats.get("all_ic", np.nan)), 8),
            "ICIR": round(float(stats.get("all_icir", np.nan)), 8),
            "Rank IC": round(float(stats.get("all_rank_ic", np.nan)), 8),
            "Rank ICIR": round(float(stats.get("all_rank_icir", np.nan)), 8),
            "valid_rank_ic": round(float(stats.get("valid_rank_ic", np.nan)), 8),
            "test_rank_ic": round(float(stats.get("test_rank_ic", np.nan)), 8),
            "positive_year_ratio": round(float(stats.get("positive_year_ratio", 0.0)), 4),
            "coverage": round(float(stats.get("coverage", 0.0)), 4),
        }
    )
    cache_location = original.get("cache_location", {})
    if expression != candidate.expression:
        cache_location = {}

    return {
        "factor_id": factor_id,
        "factor_name": candidate.name,
        "factor_expression": expression,
        "factor_description": candidate.description or original.get("factor_description", ""),
        "factor_formulation": original.get("factor_formulation", expression),
        "factor_implementation_code": original.get("factor_implementation_code", ""),
        "cache_location": cache_location,
        "metadata": {
            "source": candidate.source,
            "selected_at": datetime.now().isoformat(),
            "selection_score": round(_score(stats), 8),
            "strict_gate": True,
        },
        "quality_gate": stats,
        "backtest_results": backtest_results,
        "feedback": original.get("feedback", {}),
    }


def _cache_key(expr: str) -> str:
    return hashlib.md5(expr.encode()).hexdigest()


def _save_cache(cache_dir: Path, expr: str, series: pd.Series) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    series.to_pickle(cache_dir / f"{_cache_key(expr)}.pkl")


def _prune_cache(cache_dir: Path, selected_exprs: set[str], candidate_exprs: set[str]) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    project_root = PROJECT_ROOT.resolve()
    if project_root not in cache_dir.parents and cache_dir != project_root:
        raise ValueError(f"Refusing to prune cache outside project: {cache_dir}")

    selected_keys = {_cache_key(expr) for expr in selected_exprs}
    candidate_keys = {_cache_key(expr) for expr in candidate_exprs}
    deleted = []
    kept = []
    if not cache_dir.exists():
        return {"deleted": deleted, "kept": kept}

    for path in cache_dir.glob("*.pkl"):
        key = path.stem
        if key in selected_keys:
            kept.append(path.name)
        elif key in candidate_keys:
            deleted.append({"file": path.name, "bytes": path.stat().st_size})
            path.unlink()

    return {"deleted": deleted, "kept": kept}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--output-dir", default="data/factorlib")
    parser.add_argument("--cache-dir", default="data/results/factor_cache")
    parser.add_argument("--max-factors", type=int, default=12)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--prune-cache", action="store_true")
    args = parser.parse_args()

    config = _load_config(PROJECT_ROOT / args.config)
    source_paths = [PROJECT_ROOT / p for p in args.source]
    if not source_paths:
        source_paths = sorted((PROJECT_ROOT / "data/factorlib").glob("all_factors_library*.json"))

    from quantaalpha.backtest.custom_factor_calculator import CustomFactorCalculator, get_qlib_stock_data

    candidates = _load_existing_candidates(source_paths)
    candidates.extend(_template_candidates())
    by_expr = {}
    for cand in candidates:
        by_expr.setdefault(cand.expression, cand)
    candidates = list(by_expr.values())

    print(f"Loaded candidates: {len(candidates)}")
    data = get_qlib_stock_data(config)
    label = _future_return(data)
    calculator = CustomFactorCalculator(data_df=data, cache_dir=PROJECT_ROOT / args.cache_dir, config=config)

    factor_dicts = [
        {
            "factor_name": cand.name,
            "factor_expression": cand.expression,
            "cache_location": cand.original.get("cache_location", {}) if cand.original else {},
        }
        for cand in candidates
    ]
    factor_df = calculator.calculate_factors_batch(factor_dicts, use_cache=True, skip_compute=False)
    if factor_df.empty:
        raise RuntimeError("No factors could be computed")

    evaluated = []
    missing = []
    segments = dict(config["dataset"]["segments"])
    segments["all"] = [config["data"]["start_time"], config["data"]["end_time"]]
    total_rows = len(data)

    for cand in candidates:
        if cand.name not in factor_df.columns:
            missing.append(cand.name)
            continue
        series = factor_df[cand.name]
        coverage = float(np.isfinite(series).mean()) if total_rows else 0.0
        ic = _daily_ic(series, label, "pearson")
        ric = _daily_ic(series, label, "spearman")
        stats = _split_stats(ic, ric, segments)
        stats.update(_year_stats(ric))
        stats["coverage"] = coverage
        stats["all_rank_ic"] = stats.get("all_rank_ic", np.nan)
        stats["all_rank_icir"] = stats.get("all_rank_icir", np.nan)
        all_ic_std = ic.std() if len(ic) else np.nan
        stats["all_ic"] = float(ic.mean()) if len(ic) else np.nan
        stats["all_icir"] = float(ic.mean() / (all_ic_std + 1e-12)) if len(ic) else np.nan

        expr = cand.expression
        if stats.get("all_rank_ic", 0.0) < 0 and stats.get("valid_rank_ic", 0.0) < 0 and stats.get("test_rank_ic", 0.0) < 0:
            series = -series
            expr = f"-1 * ({cand.expression})"
            ic = -ic
            ric = -ric
            stats = _split_stats(ic, ric, segments)
            stats.update(_year_stats(ric))
            stats["coverage"] = coverage
            stats["all_ic"] = float(ic.mean()) if len(ic) else np.nan
            stats["all_icir"] = float(ic.mean() / (ic.std() + 1e-12)) if len(ic) else np.nan

        if _passes_gate(stats, strict=args.strict):
            evaluated.append((cand, expr, series, stats, _score(stats)))

    evaluated.sort(key=lambda item: item[-1], reverse=True)
    selected = []
    selected_series = []
    selected_exprs = set()
    used_names = set()

    for cand, expr, series, stats, score in evaluated:
        too_correlated = False
        sampled = series.dropna()
        if len(sampled) > 200000:
            sampled = sampled.iloc[:: max(1, len(sampled) // 200000)]
        for prev in selected_series:
            common = sampled.index.intersection(prev.index)
            if len(common) < 1000:
                continue
            corr = sampled.loc[common].corr(prev.loc[common], method="spearman")
            if corr is not None and math.isfinite(corr) and abs(corr) >= 0.85:
                too_correlated = True
                break
        if too_correlated:
            continue
        cand.name = _safe_name(cand.name, used_names)
        selected.append((cand, expr, series, stats))
        selected_exprs.add(expr)
        selected_series.append(sampled)
        if len(selected) >= args.max_factors:
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"all_factors_library_strict_{timestamp}.json"
    summary_path = output_dir / f"strict_factors_summary_{timestamp}.md"
    prune_path = output_dir / f"strict_cache_prune_{timestamp}.json"

    factors = {}
    cache_dir = PROJECT_ROOT / args.cache_dir
    for cand, expr, series, stats in selected:
        _save_cache(cache_dir, expr, series)
        entry = _build_factor_entry(cand, expr, stats)
        factors[entry["factor_id"]] = entry

    payload = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_factors": len(factors),
            "version": "1.1",
            "library_type": "strict_reliable_selection",
            "source_libraries": [p.name for p in source_paths],
            "selection_criteria": {
                "coverage_min": 0.55,
                "all_rank_ic_min": 0.008 if args.strict else 0.005,
                "valid_rank_ic_min": 0.002 if args.strict else -0.001,
                "test_rank_ic_min": 0.005 if args.strict else 0.002,
                "positive_year_ratio_min": 0.60 if args.strict else 0.50,
                "max_abs_spearman_corr": 0.85,
            },
            "missing_or_failed": missing,
        },
        "factors": factors,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        f"# Strict Reliable Factors {timestamp}",
        "",
        f"- Candidates: {len(candidates)}",
        f"- Passed gate before correlation filter: {len(evaluated)}",
        f"- Selected: {len(selected)}",
        "",
        "| Factor | RankIC(all) | RankIC(valid) | RankIC(test) | Positive Years | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cand, _, _, stats in selected:
        lines.append(
            "| {name} | {all_ric:.4f} | {valid:.4f} | {test:.4f} | {pyr:.2f} | {cov:.2f} |".format(
                name=cand.name,
                all_ric=stats.get("all_rank_ic", float("nan")),
                valid=stats.get("valid_rank_ic", float("nan")),
                test=stats.get("test_rank_ic", float("nan")),
                pyr=stats.get("positive_year_ratio", 0.0),
                cov=stats.get("coverage", 0.0),
            )
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    prune_report = {"deleted": [], "kept": []}
    if args.prune_cache:
        candidate_exprs = {c.expression for c in candidates}
        prune_report = _prune_cache(PROJECT_ROOT / args.cache_dir, selected_exprs, candidate_exprs)
        with open(prune_path, "w", encoding="utf-8") as f:
            json.dump(prune_report, f, ensure_ascii=False, indent=2)

    print(f"Output library: {output_path}")
    print(f"Summary: {summary_path}")
    if args.prune_cache:
        print(f"Cache prune report: {prune_path}")
        print(f"Deleted low-eff cache files: {len(prune_report['deleted'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
