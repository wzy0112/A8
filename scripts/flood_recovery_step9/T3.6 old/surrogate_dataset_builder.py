from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from surrogate_config import (
    FLOOD_STAGES,
    FORECAST_HORIZONS_S,
    LOCATIONS,
    SURROGATE_DATASET_VERSION,
    TACTICS,
    TACTIC_TIMING_BUCKET_EDGES_S,
)


STATE_META_COLUMNS = {
    "episode_id",
    "time_s",
    "decision_epoch",
}


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow. Install with: "
            "pip install pyarrow"
        ) from exc


def _safe_float(value, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _resolve_logged_path(
    raw_ref: str | None,
    *,
    summary_path: Path,
    fallback_name: str,
) -> Path:
    """
    Resolve a path stored in an episode summary.

    If logs were moved after generation, fall back to the summary directory
    using the expected episode filename.
    """
    if raw_ref:
        candidate = Path(raw_ref)
        if candidate.exists():
            return candidate

        local_candidate = summary_path.parent / candidate.name
        if local_candidate.exists():
            return local_candidate

    fallback = summary_path.parent / fallback_name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not resolve logged file for {summary_path.name}: "
        f"{raw_ref!r} or {fallback}"
    )


def _load_episode_files(summary_path: Path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episode_id = str(summary["episode_id"])

    state_path = _resolve_logged_path(
        summary.get("state_log_ref"),
        summary_path=summary_path,
        fallback_name=f"episode_{episode_id}_state_log.csv",
    )
    trajectory_path = _resolve_logged_path(
        summary.get("trajectory_ref"),
        summary_path=summary_path,
        fallback_name=f"episode_{episode_id}_second_trajectory.csv",
    )
    epoch_path = _resolve_logged_path(
        summary.get("epoch_log_ref"),
        summary_path=summary_path,
        fallback_name=f"episode_{episode_id}_epoch_log.csv",
    )

    state_df = pd.read_csv(state_path)
    trajectory_df = pd.read_csv(trajectory_path)
    epoch_df = pd.read_csv(epoch_path)

    # Current logger versions store the epoch-end time in time_s.
    # For tactic timing we need the decision/action start time.
    # New logs should write action_time_s explicitly; old logs are repaired
    # deterministically as time_s - decision_period_s.
    if (
        "action_time_s" not in epoch_df.columns
        or epoch_df["action_time_s"].isna().all()
    ):
        epoch_df["action_time_s"] = (
            pd.to_numeric(
                epoch_df["time_s"],
                errors="coerce",
            )
            - float(summary.get("decision_period_s", 300))
        )

    if state_df.empty:
        raise ValueError(f"Empty state log: {state_path}")
    if trajectory_df.empty:
        raise ValueError(f"Empty trajectory log: {trajectory_path}")

    return summary, state_df, trajectory_df, epoch_df


def _stage_at_time(t: float, d: dict) -> str:
    onset_t = float(d["onset_t"])
    peak_start_t = float(d["peak_start_t"])
    peak_end_t = float(d["peak_end_t"])
    recovery_end_t = float(d["recovery_end_t"])

    if t < onset_t:
        return "normal"
    if t < peak_start_t:
        return "onset"
    if t < peak_end_t:
        return "peak"
    if t < recovery_end_t:
        return "recovery"
    return "normal"


def _future_flood_features(
    *,
    current_time_s: float,
    d: dict,
    affected_mask: dict[str, float],
) -> dict[str, float]:
    """
    Oracle future-flood labels derived from the scripted scenario.

    These are T3.5 training targets now. Later, at runtime, the T3.5 predictor
    supplies the same feature schema to T3.6 / RL instead of this oracle.
    """
    row: dict[str, float] = {}

    for horizon in FORECAST_HORIZONS_S:
        future_t = current_time_s + horizon
        stage = _stage_at_time(future_t, d)
        active = stage in {"onset", "peak", "recovery"}

        for name in FLOOD_STAGES:
            row[
                f"target_flood::h{horizon}::stage::{name}"
            ] = float(stage == name)

        for segment_feature, mask_value in affected_mask.items():
            row[
                f"target_flood::h{horizon}::affected::{segment_feature}"
            ] = float(mask_value if active else 0.0)

    return row


def _timing_bucket(relative_time_s: float) -> int:
    if relative_time_s < 0:
        return 0

    for bucket_index, upper in enumerate(
        TACTIC_TIMING_BUCKET_EDGES_S,
        start=1,
    ):
        if relative_time_s < upper:
            return bucket_index

    return len(TACTIC_TIMING_BUCKET_EDGES_S) + 1


def encode_future_tactic_chain(
    epoch_df: pd.DataFrame,
    *,
    current_time_s: float,
    horizon_s: int,
) -> dict[str, float]:
    """
    Fixed quantitative M encoding:
        per tactic-location slot:
          - presence
          - max intensity
          - first application timing bucket

    HOLD and OFF/remove actions do not create an activation.
    """
    result: dict[str, float] = {}

    for tactic in TACTICS:
        for location in LOCATIONS:
            prefix = f"M_future::{tactic}::{location}"
            result[f"{prefix}::presence"] = 0.0
            result[f"{prefix}::max_intensity"] = 0.0
            result[f"{prefix}::first_timing_bucket"] = 0.0

    if epoch_df.empty:
        return result

    end_t = current_time_s + horizon_s
    window = epoch_df[
        (epoch_df["action_time_s"] >= current_time_s)
        & (epoch_df["action_time_s"] <= end_t)
    ].copy()

    if window.empty:
        return result

    for tactic in TACTICS:
        for location in LOCATIONS:
            rows = window[
                (window["tactic_name"] == tactic)
                & (window["location_id"] == location)
                & (window["intensity_index"] > 0)
                & (window["operation"] == "apply_or_modify")
            ]

            if rows.empty:
                continue

            prefix = f"M_future::{tactic}::{location}"
            first_t = float(rows["action_time_s"].min())
            result[f"{prefix}::presence"] = 1.0
            result[f"{prefix}::max_intensity"] = float(
                rows["intensity_index"].max()
            )
            result[f"{prefix}::first_timing_bucket"] = float(
                _timing_bucket(first_t - current_time_s)
            )

    return result


def encode_episode_tactic_chain(epoch_df: pd.DataFrame) -> dict[str, float]:
    if epoch_df.empty:
        return encode_future_tactic_chain(
            epoch_df,
            current_time_s=0.0,
            horizon_s=max(FORECAST_HORIZONS_S),
        )

    start_t = float(epoch_df["action_time_s"].min())
    end_t = float(epoch_df["action_time_s"].max())
    horizon = max(1, int(math.ceil(end_t - start_t + 1.0)))

    encoded = encode_future_tactic_chain(
        epoch_df,
        current_time_s=start_t,
        horizon_s=horizon,
    )

    return {
        key.replace("M_future::", "M::", 1): value
        for key, value in encoded.items()
    }


def _interpolate_pr(
    trajectory_df: pd.DataFrame,
    query_time_s: float,
) -> float:
    times = trajectory_df["time_s"].to_numpy(dtype=float)
    values = trajectory_df["pr_t"].to_numpy(dtype=float)

    finite = np.isfinite(times) & np.isfinite(values)
    times = times[finite]
    values = values[finite]

    if len(times) == 0:
        return math.nan

    order = np.argsort(times)
    times = times[order]
    values = values[order]

    if query_time_s < times[0] or query_time_s > times[-1]:
        return math.nan

    return float(np.interp(query_time_s, times, values))


def _state_feature_columns(state_df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in state_df.columns
        if col not in STATE_META_COLUMNS
    ]


def _affected_mask(state_df: pd.DataFrame) -> dict[str, float]:
    affected_columns = [
        col
        for col in state_df.columns
        if col.startswith("flood.affected.")
    ]

    return {
        col.replace("flood.affected.", "", 1): float(
            pd.to_numeric(
                state_df[col],
                errors="coerce",
            ).fillna(0.0).max()
        )
        for col in affected_columns
    }


def _episode_record(
    *,
    summary_path: Path,
    summary: dict,
    state_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
) -> dict:
    d = summary.get("D", {})
    x = summary.get("X", {})
    outcomes = summary.get("outcomes", {})
    first_state = state_df.iloc[0]
    state_columns = _state_feature_columns(state_df)

    started = str(summary.get("started_at_utc", ""))
    run_date = started[:10] if len(started) >= 10 else "unknown"

    row = {
        "dataset_version": SURROGATE_DATASET_VERSION,
        "episode_id": str(summary["episode_id"]),
        "policy_version": summary.get("policy_version"),
        "sumo_seed": summary.get("sumo_seed"),
        "run_date": run_date,
        "severity": d.get("severity"),
        "site_id": d.get("site_id"),
        "mechanism": d.get("mechanism"),
        "onset_t": d.get("onset_t"),
        "peak_start_t": d.get("peak_start_t"),
        "peak_end_t": d.get("peak_end_t"),
        "recovery_end_t": d.get("recovery_end_t"),
        "peak_speed_mps": d.get("peak_speed_mps"),
        "demand_phase": x.get(
            "demand_phase",
            d.get("demand_phase", "unknown"),
        ),
        "day_type": x.get("day_type", "unknown"),
        "time_of_day_bucket": x.get(
            "time_of_day_bucket",
            "unknown",
        ),
        "recovered": bool(outcomes.get("recovered", False)),
        "truncated": bool(outcomes.get("truncated", False)),
        "recovery_time_s": outcomes.get("recovery_time_s"),
        "reward_total": outcomes.get("reward_total"),
        "co2_total_kg": outcomes.get("co2_total_kg"),
        "safety_events": outcomes.get(
            "hard_braking_events_total",
            outcomes.get("safety_events"),
        ),
        "total_time_loss_veh_h": outcomes.get(
            "total_time_loss_veh_h"
        ),
        "queue_exposure_veh_h": outcomes.get(
            "queue_exposure_veh_h"
        ),
        "summary_ref": str(summary_path),
        "state_log_ref": summary.get("state_log_ref"),
        "trajectory_ref": summary.get("trajectory_ref"),
        "epoch_log_ref": summary.get("epoch_log_ref"),
        "tactic_chain_json": json.dumps(
            summary.get("M", []),
            sort_keys=True,
        ),
        "D_json": json.dumps(d, sort_keys=True, default=str),
        "X_json": json.dumps(x, sort_keys=True, default=str),
        "S0_json": json.dumps(
            summary.get("S0", {}),
            sort_keys=True,
            default=str,
        ),
    }

    for col in state_columns:
        row[f"S0::{col}"] = _safe_float(first_state[col])

    row.update(encode_episode_tactic_chain(epoch_df))
    return row


def _decision_samples(
    *,
    summary: dict,
    state_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    episode_id = str(summary["episode_id"])
    d = summary.get("D", {})
    outcomes = summary.get("outcomes", {})
    state_columns = _state_feature_columns(state_df)
    mask = _affected_mask(state_df)

    recovery_time_s = _safe_float(
        outcomes.get("recovery_time_s")
    )
    recovery_abs_t = (
        float(d["onset_t"]) + recovery_time_s
        if math.isfinite(recovery_time_s)
        else math.nan
    )

    flood_rows: list[dict] = []
    recovery_rows: list[dict] = []

    for _, state_row in state_df.iterrows():
        current_t = float(state_row["time_s"])
        decision_epoch = int(state_row["decision_epoch"])

        base = {
            "dataset_version": SURROGATE_DATASET_VERSION,
            "episode_id": episode_id,
            "decision_epoch": decision_epoch,
            "time_s": current_t,
        }

        for col in state_columns:
            base[f"state::{col}"] = _safe_float(state_row[col])

        future_flood_targets = _future_flood_features(
            current_time_s=current_t,
            d=d,
            affected_mask=mask,
        )

        flood_row = {
            **base,
            **future_flood_targets,
        }
        flood_rows.append(flood_row)

        future_flood_inputs = {
            key.replace(
                "target_flood::",
                "future_flood::",
                1,
            ): value
            for key, value in future_flood_targets.items()
        }

        tactic_features = encode_future_tactic_chain(
            epoch_df,
            current_time_s=current_t,
            horizon_s=max(FORECAST_HORIZONS_S),
        )

        rec_row = {
            **base,
            **future_flood_inputs,
            **tactic_features,
            "target_eventual_recovered": float(
                bool(outcomes.get("recovered", False))
            ),
        }

        if math.isfinite(recovery_abs_t):
            rec_row[
                "target_recovery_time_remaining_s"
            ] = max(0.0, recovery_abs_t - current_t)
        else:
            rec_row[
                "target_recovery_time_remaining_s"
            ] = math.nan

        for horizon in FORECAST_HORIZONS_S:
            query_t = current_t + horizon
            rec_row[f"target_pr_h{horizon}"] = (
                _interpolate_pr(
                    trajectory_df,
                    query_t,
                )
            )

        recovery_rows.append(rec_row)

    return flood_rows, recovery_rows


def build_datasets(
    *,
    episode_root: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, int]:
    _require_pyarrow()

    episode_root = Path(episode_root)
    output_dir = Path(output_dir)

    summaries = sorted(
        episode_root.rglob("episode_*_summary.json")
    )
    if not summaries:
        raise FileNotFoundError(
            f"No episode_*_summary.json found under {episode_root}"
        )

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict] = []
    flood_rows: list[dict] = []
    recovery_rows: list[dict] = []

    failures: list[dict] = []

    for summary_path in summaries:
        try:
            (
                summary,
                state_df,
                trajectory_df,
                epoch_df,
            ) = _load_episode_files(summary_path)

            episode_rows.append(
                _episode_record(
                    summary_path=summary_path,
                    summary=summary,
                    state_df=state_df,
                    epoch_df=epoch_df,
                )
            )

            f_rows, r_rows = _decision_samples(
                summary=summary,
                state_df=state_df,
                trajectory_df=trajectory_df,
                epoch_df=epoch_df,
            )
            flood_rows.extend(f_rows)
            recovery_rows.extend(r_rows)

        except Exception as exc:
            failures.append({
                "summary_path": str(summary_path),
                "error": repr(exc),
            })

    if not episode_rows:
        raise RuntimeError(
            "No valid episodes were converted. "
            f"Failures: {failures[:5]}"
        )

    episode_df = pd.DataFrame(episode_rows)
    flood_df = pd.DataFrame(flood_rows)
    recovery_df = pd.DataFrame(recovery_rows)

    # 8.1: partitioned, one-row-per-episode storage.
    episode_dataset_dir = output_dir / "episode_records"
    episode_df.to_parquet(
        episode_dataset_dir,
        engine="pyarrow",
        index=False,
        partition_cols=[
            "severity",
            "site_id",
            "run_date",
        ],
    )

    # Decision-level training tables used by T3.5/T3.6.
    flood_df.to_parquet(
        output_dir / "flood_forecast_samples.parquet",
        engine="pyarrow",
        index=False,
    )
    recovery_df.to_parquet(
        output_dir / "recovery_surrogate_samples.parquet",
        engine="pyarrow",
        index=False,
    )

    manifest = {
        "dataset_version": SURROGATE_DATASET_VERSION,
        "episode_root": str(episode_root),
        "episode_records": int(len(episode_df)),
        "flood_forecast_samples": int(len(flood_df)),
        "recovery_surrogate_samples": int(len(recovery_df)),
        "failed_episode_count": int(len(failures)),
        "failures": failures,
        "forecast_horizons_s": list(
            FORECAST_HORIZONS_S
        ),
    }

    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return {
        "episode_records": len(episode_df),
        "flood_forecast_samples": len(flood_df),
        "recovery_surrogate_samples": len(recovery_df),
        "failures": len(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-root",
        type=Path,
        required=True,
        help=(
            "Root directory containing episode_*_summary.json "
            "and their CSV logs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    counts = build_datasets(
        episode_root=args.episode_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    print("Surrogate datasets created:")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
