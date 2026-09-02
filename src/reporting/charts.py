"""Chart generation for the performance report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.config import PLOTS_DIR


def plot_equity_curve(daily_nav: pl.DataFrame, path=None):
    path = path or PLOTS_DIR / "equity_curve.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_nav["Datetime"], daily_nav["PortfolioValue"], color="steelblue")
    ax.set_title("Portfolio Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (Rs)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_drawdown(daily_nav: pl.DataFrame, path=None):
    path = path or PLOTS_DIR / "drawdown.png"
    values = daily_nav["PortfolioValue"].to_numpy()
    drawdown = values / np.maximum.accumulate(values) - 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(daily_nav["Datetime"], drawdown, 0, color="crimson", alpha=0.3)
    ax.plot(daily_nav["Datetime"], drawdown, color="crimson")
    ax.set_title("Portfolio Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_annual_returns(daily_nav: pl.DataFrame, path=None):
    path = path or PLOTS_DIR / "annual_returns.png"
    df = daily_nav.sort("Datetime").with_columns(pl.col("Datetime").dt.year().alias("Year"))
    annual = df.group_by("Year", maintain_order=True).agg(
        (pl.col("PortfolioValue").last() / pl.col("PortfolioValue").first() - 1).alias("Return")
    ).sort("Year")

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["seagreen" if r >= 0 else "crimson" for r in annual["Return"]]
    ax.bar(annual["Year"].cast(pl.Utf8), annual["Return"], color=colors)
    ax.set_title("Annual Returns")
    ax.set_ylabel("Return")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_correlation(corr_matrix: pl.DataFrame, path=None):
    path = path or PLOTS_DIR / "feature_correlation.png"
    features = corr_matrix["Feature"].to_list()
    matrix = corr_matrix.select(features).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=6)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_feature_importance(experiment_results: pl.DataFrame, path=None):
    """Proxy for feature importance: average CAGR per signal family across the experiment matrix
    (a true per-feature information-coefficient study would need a dedicated IC analysis)."""
    path = path or PLOTS_DIR / "feature_importance.png"
    summary = experiment_results.group_by("Model").agg(pl.col("CAGR").mean().alias("AvgCAGR")).sort("AvgCAGR")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(summary["Model"], summary["AvgCAGR"], color="steelblue")
    ax.set_title("Signal Importance (Avg CAGR across formation dates)")
    ax.set_xlabel("Average CAGR")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_portfolio_contribution(contributions: pl.DataFrame, path=None):
    path = path or PLOTS_DIR / "portfolio_contribution.png"
    df = contributions.sort("PnL")
    colors = ["seagreen" if v >= 0 else "crimson" for v in df["PnL"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["Symbol"], df["PnL"], color=colors)
    ax.set_title("Per-Stock Contribution to P&L")
    ax.set_xlabel("P&L (Rs)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_all_charts(daily_nav: pl.DataFrame, corr_matrix: pl.DataFrame, experiment_results: pl.DataFrame, contributions: pl.DataFrame) -> dict:
    return {
        "equity_curve": plot_equity_curve(daily_nav),
        "drawdown": plot_drawdown(daily_nav),
        "annual_returns": plot_annual_returns(daily_nav),
        "feature_correlation": plot_feature_correlation(corr_matrix),
        "feature_importance": plot_feature_importance(experiment_results),
        "portfolio_contribution": plot_portfolio_contribution(contributions),
    }
