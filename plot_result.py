# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

df = pd.read_csv("benchmark_results.csv")

# ----------------------------
# Runtime scaling (log scale)
# ----------------------------

fig, ax = plt.subplots()

ax.plot(
    df["pct"],
    df["python_mean"],
    marker="o",
    linewidth=2,
    label="Pandas Apply",
)

ax.plot(
    df["pct"],
    df["rust_list_mean"],
    marker="o",
    linewidth=2,
    label="Rust + Lists",
)

ax.plot(
    df["pct"],
    df["rust_numpy_mean"],
    marker="o",
    linewidth=2,
    label="Rust + NumPy",
)

ax.plot(
    df["pct"],
    df["polars_mean"],
    marker="o",
    linewidth=2,
    label="Polars",
)

ax.set_yscale("log")

ax.set_title(
    "Impact Score Computation Performance",
    fontsize=16,
    pad=20,
)

ax.set_xlabel("Dataset Size (%)")
ax.set_ylabel("Runtime (seconds, log scale)")

ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("runtime_scaling_log.png", dpi=300)

# ----------------------------
# Speedup chart
# ----------------------------

fig, ax = plt.subplots()

speedup_rust_lists = (
    df["python_mean"] / df["rust_list_mean"]
)

speedup_rust_numpy = (
    df["python_mean"] / df["rust_numpy_mean"]
)

speedup_polars = (
    df["python_mean"] / df["polars_mean"]
)

ax.plot(
    df["pct"],
    speedup_rust_lists,
    marker="o",
    linewidth=2,
    label="Rust + Lists",
)

ax.plot(
    df["pct"],
    speedup_rust_numpy,
    marker="o",
    linewidth=2,
    label="Rust + NumPy",
)

ax.plot(
    df["pct"],
    speedup_polars,
    marker="o",
    linewidth=2,
    label="Polars",
)

ax.set_title(
    "Speedup Relative to Pandas Apply",
    fontsize=16,
    pad=20,
)

ax.set_xlabel("Dataset Size (%)")
ax.set_ylabel("Speedup (×)")

ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("speedup_vs_python.png", dpi=300)

# ----------------------------
# Error bars
# ----------------------------

fig, ax = plt.subplots()

ax.errorbar(
    df["pct"],
    df["python_mean"],
    yerr=df["python_std"],
    fmt="-o",
    capsize=4,
    label="Pandas Apply",
)

ax.errorbar(
    df["pct"],
    df["rust_numpy_mean"],
    yerr=df["rust_numpy_std"],
    fmt="-o",
    capsize=4,
    label="Rust + NumPy",
)

ax.errorbar(
    df["pct"],
    df["polars_mean"],
    yerr=df["polars_std"],
    fmt="-o",
    capsize=4,
    label="Polars",
)

ax.set_yscale("log")

ax.set_title(
    "Runtime Distribution (Mean ± Std)",
    fontsize=16,
    pad=20,
)

ax.set_xlabel("Dataset Size (%)")
ax.set_ylabel("Runtime (seconds, log scale)")

ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("runtime_errorbars.png", dpi=300)

print("Saved:")
print("  runtime_scaling_log.png")
print("  speedup_vs_python.png")
print("  runtime_errorbars.png")