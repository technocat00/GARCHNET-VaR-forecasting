from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(RESULTS_DIR / "paper_window_summary.csv")

comp = summary[summary["model"].isin(["Original GARCHNet", "GARCHNet++"])].copy()
pivot = comp.pivot(index="period", columns="model", values="exceptions")
pivot = pivot.loc[["Period I", "Period II", "Period III", "Period IV"]]

ax = pivot.plot(kind="bar", figsize=(9, 5))
ax.set_title("VaR Exceptions by Period")
ax.set_xlabel("Out-of-sample period")
ax.set_ylabel("Number of VaR exceptions")
ax.axhline(252 * 0.025, linestyle="--", linewidth=1)
ax.text(3.02, 252 * 0.025 + 0.4, "Expected ≈ 6.3", fontsize=9)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(ASSETS_DIR / "var_exceptions_comparison.png", dpi=180)
plt.close()

agg = comp.groupby("model")[["exceptions", "GPL score", "LLF", "CRLF", "CFLF"]].sum().reset_index()

orig = agg[agg["model"] == "Original GARCHNet"].iloc[0]
plus = agg[agg["model"] == "GARCHNet++"].iloc[0]

normalized = pd.DataFrame({
    "Metric": ["VaR breaches", "GPL score", "LLF", "CRLF", "CFLF"],
    "Original GARCHNet": [100, 100, 100, 100, 100],
    "GARCHNet++": [
        plus["exceptions"] / orig["exceptions"] * 100,
        plus["GPL score"] / orig["GPL score"] * 100,
        plus["LLF"] / orig["LLF"] * 100,
        plus["CRLF"] / orig["CRLF"] * 100,
        plus["CFLF"] / orig["CFLF"] * 100,
    ],
})

ax = normalized.set_index("Metric").plot(kind="bar", figsize=(9, 5))
ax.set_title("Aggregate Metrics: Original GARCHNet vs GARCHNet++")
ax.set_ylabel("Metric value normalized to Original GARCHNet = 100")
ax.set_xlabel("")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(ASSETS_DIR / "original_vs_plus_normalized.png", dpi=180)
plt.close()

forecast_path = RESULTS_DIR / "paper_window_forecasts.csv"

if forecast_path.exists():
    forecasts = pd.read_csv(forecast_path)

    if {"period", "model", "return", "VaR"}.issubset(forecasts.columns):
        if "date" in forecasts.columns:
            forecasts["date"] = pd.to_datetime(forecasts["date"])
            x_col = "date"
        else:
            forecasts["step"] = forecasts.groupby(["period", "model"]).cumcount()
            x_col = "step"

        periods = ["Period I", "Period II", "Period III", "Period IV"]
        models = ["Original GARCHNet", "GARCHNet++"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
        axes = axes.ravel()

        for ax, period in zip(axes, periods):
            part = forecasts[forecasts["period"] == period].copy()

            base = part[part["model"] == "Original GARCHNet"]
            if base.empty:
                base = part.drop_duplicates(subset=[x_col])

            ax.plot(base[x_col], base["return"], label="Actual return", linewidth=1)

            for model in models:
                mdf = part[part["model"] == model]
                if not mdf.empty:
                    ax.plot(mdf[x_col], mdf["VaR"], label=model, linewidth=1.2)

            ax.set_title(period)
            ax.axhline(0, linewidth=0.8)
            ax.tick_params(axis="x", rotation=25)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.suptitle("One-Day-Ahead 2.5% VaR Forecasts", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(ASSETS_DIR / "var_forecast_paths.png", dpi=180)
        plt.close()

print("Saved plots in assets/")