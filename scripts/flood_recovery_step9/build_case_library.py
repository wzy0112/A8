from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def _pareto_mask(values: np.ndarray) -> np.ndarray:
    """
    True = non-dominated for minimization objectives.

    A row is dominated when another row is <= in all objectives
    and strictly < in at least one.
    """
    n = len(values)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue

        for j in range(n):
            if i == j:
                continue

            if (
                np.all(values[j] <= values[i])
                and np.any(values[j] < values[i])
            ):
                keep[i] = False
                break

    return keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-records",
        type=Path,
        required=True,
        help=(
            "Partitioned episode_records directory "
            "created by surrogate_dataset_builder.py"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--onset-bucket-s",
        type=int,
        default=300,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.episode_records)

    df["onset_bucket"] = (
        pd.to_numeric(
            df["onset_t"],
            errors="coerce",
        )
        // args.onset_bucket_s
    ).astype("Int64")

    df["scenario_bucket_id"] = (
        df["severity"].astype(str)
        + "|"
        + df["site_id"].astype(str)
        + "|"
        + df["demand_phase"].astype(str)
        + "|onset_"
        + df["onset_bucket"].astype(str)
    )

    # Non-recovered episodes cannot dominate recovered solutions.
    recovery_obj = pd.to_numeric(
        df["recovery_time_s"],
        errors="coerce",
    )
    recovery_obj = (
        recovery_obj
        .where(
            df["recovered"].astype(bool),
            np.inf,
        )
        .fillna(np.inf)
    )

    co2_obj = pd.to_numeric(
        df["co2_total_kg"],
        errors="coerce",
    ).fillna(np.inf)

    safety_obj = pd.to_numeric(
        df["safety_events"],
        errors="coerce",
    ).fillna(np.inf)

    df["_recovery_obj"] = recovery_obj
    df["_co2_obj"] = co2_obj
    df["_safety_obj"] = safety_obj
    # ---------------------------------------------------------
    # Step 1: deduplicate BEFORE Pareto filtering.
    #
    # Within the same scenario bucket, repeated evaluations of
    # exactly the same tactic chain represent the same policy case.
    # Keep the best representative according to:
    #   1. recovery time
    #   2. CO2
    #   3. safety events
    #   4. reward (higher is better)
    # ---------------------------------------------------------

    df = df.sort_values(
        by=[
            "_recovery_obj",
            "_co2_obj",
            "_safety_obj",
            "reward_total",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
        na_position="last",
    )

    df = df.drop_duplicates(
        subset=[
            "scenario_bucket_id",
            "tactic_chain_json",
        ],
        keep="first",
    ).copy()

    # ---------------------------------------------------------
    # Step 2: Pareto filtering on deduplicated cases.
    # ---------------------------------------------------------

    df["is_pareto_optimal"] = False

    for _, index in df.groupby(
            "scenario_bucket_id"
    ).groups.items():
        idx = list(index)

        values = df.loc[
            idx,
            [
                "_recovery_obj",
                "_co2_obj",
                "_safety_obj",
            ],
        ].to_numpy(dtype=float)

        mask = _pareto_mask(values)

        df.loc[
            np.asarray(idx)[mask],
            "is_pareto_optimal",
        ] = True

    # ---------------------------------------------------------
    # Step 3: retain only Pareto-optimal cases.
    # ---------------------------------------------------------

    library = df.loc[
        df["is_pareto_optimal"]
    ].copy()

    library["tactic_chain"] = (
        library["tactic_chain_json"]
    )

    output_columns = [
        "scenario_bucket_id",
        "episode_id",
        "severity",
        "site_id",
        "demand_phase",
        "onset_t",
        "tactic_chain",
        "recovery_time_s",
        "co2_total_kg",
        "safety_events",
        "reward_total",
        "is_pareto_optimal",
        "policy_version",
    ]

    library = library[output_columns]

    parquet_path = (
        args.output_dir
        / "recovery_case_library.parquet"
    )
    sqlite_path = (
        args.output_dir
        / "recovery_case_library.sqlite"
    )

    library.to_parquet(
        parquet_path,
        engine="pyarrow",
        index=False,
    )

    with sqlite3.connect(
            sqlite_path
    ) as connection:
        connection.execute(
            """
            DROP TABLE IF EXISTS
            recovery_case_library
            """
        )

        connection.execute(
            """
            CREATE TABLE recovery_case_library (
                scenario_bucket_id TEXT NOT NULL,
                episode_id TEXT,
                severity TEXT,
                site_id TEXT,
                demand_phase TEXT,
                onset_t REAL,
                tactic_chain TEXT NOT NULL,
                recovery_time_s REAL,
                co2_total_kg REAL,
                safety_events INTEGER,
                reward_total REAL,
                is_pareto_optimal BOOLEAN,
                policy_version TEXT,
                PRIMARY KEY (
                    scenario_bucket_id,
                    tactic_chain
                )
            )
            """
        )

        library.to_sql(
            "recovery_case_library",
            connection,
            if_exists="append",
            index=False,
        )

    manifest = {
        "input_episode_count": int(len(df)),
        "pareto_case_count": int(len(library)),
        "scenario_bucket_count": int(
            library["scenario_bucket_id"].nunique()
        ),
        "parquet_path": str(parquet_path),
        "sqlite_path": str(sqlite_path),
    }
    (
        args.output_dir
        / "case_library_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
