import statistics
import pandas as pd

from main import (
    rust_to_numpy,
    rust_to_list,
    polars_native_case,
    base_case,
)

df = pd.read_csv("fifa_world_cup_2026_player_performance.csv")

results = []

for pct in range(10, 101, 10):
    n = max(1, int(len(df) * pct / 100))
    subset = df.iloc[:n].copy()

    t_base_acc = []
    t_rust_1_acc = []
    t_polar_acc = []
    t_rust_2_acc = []

    for _ in range(10):
        _, t_base = base_case(subset.copy())
        _, t_rust_1 = rust_to_list(subset.copy())
        _, t_polar = polars_native_case(subset.copy())
        _, t_rust_2 = rust_to_numpy(subset.copy())

        t_base_acc.append(t_base)
        t_rust_1_acc.append(t_rust_1)
        t_polar_acc.append(t_polar)
        t_rust_2_acc.append(t_rust_2)

    results.append({
        "pct": pct,

        "python_mean": statistics.mean(t_base_acc),
        "python_std": statistics.stdev(t_base_acc),

        "rust_list_mean": statistics.mean(t_rust_1_acc),
        "rust_list_std": statistics.stdev(t_rust_1_acc),

        "polars_mean": statistics.mean(t_polar_acc),
        "polars_std": statistics.stdev(t_polar_acc),

        "rust_numpy_mean": statistics.mean(t_rust_2_acc),
        "rust_numpy_std": statistics.stdev(t_rust_2_acc),
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "benchmark_results.csv",
    index=False,
)

print("Saved benchmark_results.csv")