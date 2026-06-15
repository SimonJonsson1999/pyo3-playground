import pandas as pd
import time
import football_kernel
from typing import Tuple

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

def main():
    
    df = pd.read_csv("fifa_world_cup_2026_player_performance.csv")
    _, base_compute_time = base_case(df)
    df_rust1, rust1_compuute_time = rust_to_list(df)
    print(f"Python compute time: {base_compute_time:.5f} s")
    print(f"Rust compute time: {rust1_compuute_time:.5f} s")
    print(f"Speed up using rust instead of python: {(base_compute_time/rust1_compuute_time):.2f}x")
    print(df_rust1[["player_name", "position", "impact_score", "nationality"]].sort_values("impact_score", ascending=False).head(20))


if __name__ == "__main__":
    main()
