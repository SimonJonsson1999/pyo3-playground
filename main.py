import pandas as pd
import time
import football_kernel
from typing import Tuple
import polars as pl
import numpy as np
position_map = {
                    "Goalkeeper": 0,
                    "Defender": 1,
                    "Midfielder": 2,
                    "Forward": 3,
                    }
# Lets define a custom 'impact score'
def compute_impact(row) -> float:
    score = (
        row.goals * 4.0
        + row.assists * 3.0
        + row.shots_on_target * 0.7
        + row.expected_goals_xg * 1.5
        + row.expected_assists_xa * 1.2
        + row.key_passes * 0.5
        + row.successful_passes * 0.01
        + row.successful_dribbles * 0.4
        + row.tackles * 0.3
        + row.interceptions * 0.3
        + row.recoveries * 0.1
        - row.fouls_committed * 0.4
        - row.yellow_cards * 1.0
        - row.red_cards * 3.0
    )

    if row.position == "Goalkeeper":
        score += row.saves * 0.8 + row.clean_sheet * 3.0 - row.goals_conceded * 0.7
    elif row.position == "Defender":
        score += row.clearances * 0.25 + row.blocks * 0.4 + row.aerial_duels_won * 0.2
    elif row.position == "Midfielder":
        score += row.pass_accuracy * 0.03 + row.possession_impact * 0.5 + row.creativity_score * 0.4
    elif row.position == "Forward":
        score += row.offensive_contribution * 0.6 + row.clutch_performance_score * 0.4
    else:
        print("No position found")
        return -1
    
    return score
    
def base_case(df) -> Tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    df["impact_score"] = df.apply(compute_impact, axis=1)
    compute_time = time.perf_counter() - t0
    return df, compute_time

def rust_to_list(df) -> Tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    scores = football_kernel.compute_scores(
        positions=df["position"].map(position_map).tolist(),
        goals=df["goals"].tolist(),
        assists=df["assists"].tolist(),
        shots_on_target=df["shots_on_target"].tolist(),
        expected_goals_xg=df["expected_goals_xg"].tolist(),
        expected_assists_xa=df["expected_assists_xa"].tolist(),
        key_passes=df["key_passes"].tolist(),
        successful_passes=df["successful_passes"].tolist(),
        successful_dribbles=df["successful_dribbles"].tolist(),
        tackles=df["tackles"].tolist(),
        interceptions=df["interceptions"].tolist(),
        recoveries=df["recoveries"].tolist(),
        fouls_committed=df["fouls_committed"].tolist(),
        yellow_cards=df["yellow_cards"].tolist(),
        red_cards=df["red_cards"].tolist(),
        saves=df["saves"].tolist(),
        clean_sheet=df["clean_sheet"].tolist(),
        goals_conceded=df["goals_conceded"].tolist(),
        clearances=df["clearances"].tolist(),
        blocks=df["blocks"].tolist(),
        aerial_duels_won=df["aerial_duels_won"].tolist(),
        pass_accuracy=df["pass_accuracy"].tolist(),
        possession_impact=df["possession_impact"].tolist(),
        creativity_score=df["creativity_score"].tolist(),
        offensive_contribution=df["offensive_contribution"].tolist(),
        clutch_performance_score=df["clutch_performance_score"].tolist(),
    )
    df["impact_score"] = scores
    compute_time = time.perf_counter() - t0
    return df, compute_time

def rust_to_numpy(df) -> Tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    scores = football_kernel.compute_scores_numpy(
        positions=df["position"].map(position_map).to_numpy(dtype=np.uint8),
        goals=df["goals"].to_numpy(dtype=np.float64),
        assists=df["assists"].to_numpy(dtype=np.float64),
        shots_on_target=df["shots_on_target"].to_numpy(dtype=np.float64),
        expected_goals_xg=df["expected_goals_xg"].to_numpy(dtype=np.float64),
        expected_assists_xa=df["expected_assists_xa"].to_numpy(dtype=np.float64),
        key_passes=df["key_passes"].to_numpy(dtype=np.float64),
        successful_passes=df["successful_passes"].to_numpy(dtype=np.float64),
        successful_dribbles=df["successful_dribbles"].to_numpy(dtype=np.float64),
        tackles=df["tackles"].to_numpy(dtype=np.float64),
        interceptions=df["interceptions"].to_numpy(dtype=np.float64),
        recoveries=df["recoveries"].to_numpy(dtype=np.float64),
        fouls_committed=df["fouls_committed"].to_numpy(dtype=np.float64),
        yellow_cards=df["yellow_cards"].to_numpy(dtype=np.float64),
        red_cards=df["red_cards"].to_numpy(dtype=np.float64),
        saves=df["saves"].to_numpy(dtype=np.float64),
        clean_sheet=df["clean_sheet"].to_numpy(dtype=np.float64),
        goals_conceded=df["goals_conceded"].to_numpy(dtype=np.float64),
        clearances=df["clearances"].to_numpy(dtype=np.float64),
        blocks=df["blocks"].to_numpy(dtype=np.float64),
        aerial_duels_won=df["aerial_duels_won"].to_numpy(dtype=np.float64),
        pass_accuracy=df["pass_accuracy"].to_numpy(dtype=np.float64),
        possession_impact=df["possession_impact"].to_numpy(dtype=np.float64),
        creativity_score=df["creativity_score"].to_numpy(dtype=np.float64),
        offensive_contribution=df["offensive_contribution"].to_numpy(dtype=np.float64),
        clutch_performance_score=df["clutch_performance_score"].to_numpy(dtype=np.float64),
    )
    df["impact_score"] = scores
    compute_time = time.perf_counter() - t0
    return df, compute_time


def polars_native_case(df: pl.DataFrame):
    t0 = time.perf_counter()
    df = pl.from_pandas(df)
    score = (
        pl.col("goals") * 4.0
        + pl.col("assists") * 3.0
        + pl.col("shots_on_target") * 0.7
        + pl.col("expected_goals_xg") * 1.5
        + pl.col("expected_assists_xa") * 1.2
        + pl.col("key_passes") * 0.5
        + pl.col("successful_passes") * 0.01
        + pl.col("successful_dribbles") * 0.4
        + pl.col("tackles") * 0.3
        + pl.col("interceptions") * 0.3
        + pl.col("recoveries") * 0.1
        - pl.col("fouls_committed") * 0.4
        - pl.col("yellow_cards")
        - pl.col("red_cards") * 3.0
    )

    df = df.with_columns(
        (
            score
            + pl.when(pl.col("position") == "Goalkeeper")
            .then(
                pl.col("saves") * 0.8
                + pl.col("clean_sheet") * 3.0
                - pl.col("goals_conceded") * 0.7
            )
            .when(pl.col("position") == "Defender")
            .then(
                pl.col("clearances") * 0.25
                + pl.col("blocks") * 0.4
                + pl.col("aerial_duels_won") * 0.2
            )
            .when(pl.col("position") == "Midfielder")
            .then(
                pl.col("pass_accuracy") * 0.03
                + pl.col("possession_impact") * 0.5
                + pl.col("creativity_score") * 0.4
            )
            .otherwise(
                pl.col("offensive_contribution") * 0.6
                + pl.col("clutch_performance_score") * 0.4
            )
        ).alias("impact_score")
    )

    compute_time = time.perf_counter() - t0
    return df, compute_time

def main():
    
    df = pd.read_csv("fifa_world_cup_2026_player_performance.csv")
    _, base_compute_time = base_case(df)
    df_rust1, rust1_compuute_time = rust_to_list(df)
    _, rust_numpy = rust_to_numpy(df)
    _, polars_compute_time = polars_native_case(df)
    print(f"Python compute time: {base_compute_time:.5f} s")
    print(f"Rust compute time: {rust1_compuute_time:.5f} s")
    print(f"Rust compute time: {rust_numpy:.5f} s")
    print(f"Polars compute time: {polars_compute_time:.5f} s")
    print(f"Speed up using rust instead of python: {(base_compute_time/rust1_compuute_time):.2f}x")
    print(f"Speed up using rust instead of polars: {(polars_compute_time/rust1_compuute_time):.2f}x")
    print(f"Speed up using rust_numpy instead of polars: {(polars_compute_time/rust_numpy):.2f}x")
    print(f"Speed up using rust_numpy instead of python: {(base_compute_time/rust_numpy):.2f}x")
    print(df_rust1[["player_name", "position", "impact_score", "nationality"]].sort_values("impact_score", ascending=False).head(20))


if __name__ == "__main__":
    main()
