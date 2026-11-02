"""Build reports/full_methodology_report.html from the markdown report, existing plots and results CSVs."""

import base64
import io
import re

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from src.config import PLOTS_DIR, REPORTS_DIR, RESULTS_DIR

SRC = REPORTS_DIR / "full_methodology_report.md"
OUT = REPORTS_DIR / "full_methodology_report.html"
FINAL = "composite_quantitative"
COLOR, HILITE = "#4c78a8", "#e45756"

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.55;color:#1f2933;
max-width:1100px;margin:0 auto;padding:24px 28px;background:#fff}
h1{border-bottom:3px solid #4c78a8;padding-bottom:8px}
h2{margin-top:2.2em;border-bottom:1px solid #d0d7de;padding-bottom:4px;color:#17365d}
h3{margin-top:1.6em;color:#244b7a}
table{border-collapse:collapse;margin:14px 0;font-size:.9em;display:block;overflow-x:auto}
th,td{border:1px solid #d0d7de;padding:6px 10px;text-align:left;vertical-align:top}
th{background:#eef3f9}tr:nth-child(even) td{background:#f8fafc}
code{background:#f0f3f6;padding:1px 5px;border-radius:4px;font-size:.92em}
pre{background:#f6f8fa;padding:12px;border-radius:6px;overflow-x:auto}pre code{background:none;padding:0}
figure{margin:18px 0;text-align:center}figure img{max-width:100%;border:1px solid #e1e4e8;border-radius:6px}
figcaption{font-size:.85em;color:#57606a;margin-top:4px}
nav#toc{background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:8px 20px;margin:18px 0}
nav#toc ul{margin:4px 0;padding-left:20px}
blockquote{border-left:4px solid #4c78a8;margin:12px 0;padding:2px 14px;background:#f6f8fa}
hr{border:0;border-top:1px solid #d0d7de;margin:2em 0}
@media print{body{max-width:none}h2{page-break-before:auto}figure,table{page-break-inside:avoid}}
"""


def fig_to_img(fig, caption):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return img_html(base64.b64encode(buf.getvalue()).decode(), caption)


def img_html(b64, caption):
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def file_img(name, caption):
    path = PLOTS_DIR / name
    if not path.exists():
        return ""
    return img_html(base64.b64encode(path.read_bytes()).decode(), caption)


def bars(labels, values, title, xlabel, highlight=None, pct=False):
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(labels) + 1.2))
    colors = [HILITE if l == highlight else COLOR for l in labels]
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.1%}" if pct else f" {v:.2f}", va="center", fontsize=8)
    return fig


def build_charts():
    charts = {}

    mc = pl.read_csv(RESULTS_DIR / "model_comparison.csv").sort("Sharpe", descending=True)
    names = mc["Model"].to_list()
    charts["7.2"] = (
        fig_to_img(bars(names, mc["Sharpe"].to_list(), "Sharpe ratio by model (formation 2021-01-01)", "Sharpe", FINAL),
                   "Figure: Sharpe by model; final model highlighted in red")
        + fig_to_img(bars(names, mc.sort("Sharpe", descending=True)["CAGR"].to_list(), "CAGR by model", "CAGR", FINAL, True),
                     "Figure: CAGR by model, same order")
        + fig_to_img(bars(names, mc["MaxDrawdown"].to_list(), "Max drawdown by model", "Max drawdown", FINAL, True),
                     "Figure: Max drawdown by model, same order")
    )

    ex = pl.read_csv(RESULTS_DIR / "experiment_results.csv")
    by_model = ex.group_by("Model").agg(pl.col("Sharpe").mean()).sort("Sharpe", descending=True)
    by_date = (ex.with_columns(pl.col("FormationDate").cast(pl.Utf8).str.slice(0, 10))
               .group_by("FormationDate").agg(pl.col("Sharpe").mean()).sort("FormationDate"))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(by_date["FormationDate"].to_list(), by_date["Sharpe"].to_list(), color=COLOR)
    ax.set_title("Average Sharpe by formation date (2,160-run grid)")
    ax.set_ylabel("Sharpe")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    charts["7.3"] = (
        fig_to_img(bars(by_model["Model"].to_list(), by_model["Sharpe"].to_list(), "Average Sharpe by model across all grid runs", "Sharpe"),
                   "Figure: model averages across nine formation dates")
        + fig_to_img(fig, "Figure: later formation dates have less time to compound")
    )

    parts = []
    mom = pl.read_csv(RESULTS_DIR / "momentum_lookback_sensitivity.csv")
    tr = pl.read_csv(RESULTS_DIR / "trend_window_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for fam, g in mom.group_by("SignalFamily", maintain_order=True):
        g = g.group_by("LookbackDays").agg(pl.col("Sharpe").mean()).sort("LookbackDays")
        ax.plot(range(len(g)), g["Sharpe"].to_list(), marker="o", label=f"{fam[0]} momentum")
        for i, d in enumerate(g["LookbackDays"].to_list()):
            ax.annotate(str(d), (i, g["Sharpe"][i]), fontsize=7, xytext=(0, 5), textcoords="offset points", ha="center")
    t = tr.group_by("LookbackDays").agg(pl.col("Sharpe").mean()).sort("LookbackDays")
    ax.plot(range(len(t)), t["Sharpe"].to_list(), marker="s", ls="--", color="k", label="trend strength (windows 60/90/126/180)")
    ax.set_xticks([])
    ax.set_title("Average Sharpe across lookback windows (labels = days)")
    ax.set_ylabel("Sharpe")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    charts["7.5"] = fig_to_img(fig, "Figure: performance is a broad plateau across windows")

    wm = pl.read_csv(RESULTS_DIR / "weighting_method_comparison.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, col, ttl, pct in zip(axes, ["CAGR", "Sharpe", "MaxDrawdown"], ["CAGR", "Sharpe", "Max drawdown"], [True, False, True]):
        ax.bar(wm["WeightingMethod"].to_list(), wm[col].to_list(), color=[HILITE if m == "equal" else COLOR for m in wm["WeightingMethod"]])
        ax.set_title(ttl)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    charts["8.1"] = fig_to_img(fig, "Figure: weighting methods on the final model (equal weight in red)")

    sc = pl.read_csv(RESULTS_DIR / "strategy_comparison.csv").sort("Sharpe", descending=True)
    charts["8.3"] = fig_to_img(
        bars(sc["Strategy"].to_list(), sc["Sharpe"].to_list(), "Sharpe: static vs rebalanced vs Markowitz (momentum_3m)", "Sharpe", "static"),
        "Figure: rebalancing comparison; static in red")

    charts["4.2"] = file_img("feature_correlation.png", "Figure: feature correlation heatmap")
    charts["9.3"] = (file_img("equity_curve.png", "Figure: equity curve") + file_img("drawdown.png", "Figure: drawdown")
                     + file_img("annual_returns.png", "Figure: annual returns")
                     + file_img("portfolio_contribution.png", "Figure: contribution by holding"))
    return charts


def inject(html, charts):
    # Place each chart group after the first table that follows the matching heading.
    for key, block in charts.items():
        if not block:
            continue
        m = re.search(rf"<h[23][^>]*>\s*{re.escape(key)}[ .]", html)
        if not m:
            continue
        end = html.find("</table>", m.end())
        nxt = re.search(r"<h[23]", html[m.end():])
        if end == -1 or (nxt and end > m.end() + nxt.start()):
            pos = m.end() + (nxt.start() if nxt else len(html) - m.end())
        else:
            pos = end + len("</table>")
        html = html[:pos] + block + html[pos:]
    return html


def main():
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"], extension_configs={"toc": {"toc_depth": "2"}})
    body = md.convert(SRC.read_text())
    body = inject(body, build_charts())
    toc = f'<nav id="toc"><strong>Contents</strong>{md.toc}</nav>'
    body = body.replace("</h1>", "</h1>" + toc, 1)
    OUT.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                   f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                   f"<title>CitadelxFinesse Challenge - Full Methodology Report</title><style>{CSS}</style></head>"
                   f"<body>{body}</body></html>")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
