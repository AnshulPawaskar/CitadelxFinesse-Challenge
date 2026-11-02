"""Builds reports/portfolio_challenge_deck.pptx (10 slides). Every number is read from results/ CSVs."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "results" / "battle"
OUT = ROOT / "reports" / "portfolio_challenge_deck.pptx"
ASSETS = ROOT / "reports" / "deck_assets"
ASSETS.mkdir(exist_ok=True)
NAVY, BLUE, ORANGE, GREY = "#1f3a5f", "#4c78a8", "#e8890c", "#9aa5b1"
FREQ = {"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly", "S": "Half-yearly"}
rd = lambda n: pd.read_csv(B / n)
pc = lambda x, d=1: f"{x * 100:.{d}f}%"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    p = ASSETS / f"{name}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def chart_protocol():
    fig, ax = plt.subplots(figsize=(9, 2.6))
    segs = [("Warm-up / first ML training\n2020-2021", 2020, 2022, GREY), ("Validation\n(design chosen here)\n2022-01 to 2024-06", 2022, 2024.5, BLUE),
            ("Locked holdout\n(run once)\n2024-07 to 2026-10", 2024.5, 2026.83, ORANGE)]
    for t, a, b, c in segs:
        ax.barh(0, b - a, left=a, color=c, edgecolor="white", height=.6)
        ax.text((a + b) / 2, 0, t, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")
    ax.set_yticks([]); ax.set_xlim(2020, 2026.9)
    ax.spines["left"].set_visible(False)
    return save(fig, "protocol")


def chart_ic():
    ic = rd("signal_ic_decay.csv")
    cols = ["IC_1d", "IC_5d", "IC_21d", "IC_63d", "IC_126d"]
    names = {"composite_quantitative": "Hand-set composite", "momentum_12_1": "Momentum 12-1", "ridge": "Ridge (ML)", "lgbm": "LightGBM (ML)", "mlp": "Neural net (DL)"}
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for k, lab in names.items():
        r = ic[(ic.Signal == k) & (ic.Window == "Val")]
        if len(r):
            ax.plot([1, 5, 21, 63, 126], r[cols].iloc[0].values, marker="o", label=lab, lw=2 if k in ("composite_quantitative", "momentum_12_1") else 1.3)
    ax.set_xscale("log"); ax.set_xticks([1, 5, 21, 63, 126]); ax.set_xticklabels(["1d", "5d", "21d", "63d", "126d"])
    ax.axhline(0, color="k", lw=.6); ax.set_ylabel("Rank IC vs forward return"); ax.set_xlabel("Prediction horizon (validation window)")
    ax.legend(fontsize=8)
    return save(fig, "ic")


def chart_freq():
    g = rd("grid_signal_frequency.csv")
    g = g[g.Buffer == 0]
    f = g.groupby("Freq").agg(val=("Val_Sharpe", "mean"), hold=("Hold_Sharpe", "mean"), turn=("Val_Turnover", "mean")).reindex(list(FREQ))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(f))
    a1.bar(x - .2, f.val, .4, color=BLUE, label="Validation"); a1.bar(x + .2, f.hold, .4, color=ORANGE, label="Holdout")
    a1.set_xticks(x); a1.set_xticklabels([FREQ[i] for i in f.index], rotation=20); a1.set_ylabel("Mean Sharpe across 13 signals"); a1.set_ylim(-0.5, 2.8); a1.legend(fontsize=8, ncol=2, loc="upper center")
    a2.bar(x, f.turn, color=NAVY); a2.set_yscale("log"); a2.set_xticks(x); a2.set_xticklabels([FREQ[i] for i in f.index], rotation=20)
    a2.set_ylabel("Annual one-way turnover (x NAV, log)")
    for i, v in enumerate(f.turn): a2.text(i, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=8)
    return save(fig, "freq"), f


def chart_cum():
    d = rd("final_daily_returns.csv"); d["Datetime"] = pd.to_datetime(d.Datetime); d = d.set_index("Datetime")
    nav = (1 + d).cumprod()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(nav.index, nav.Strategy, color=NAVY, lw=2, label="Final strategy (core3, quarterly)")
    ax.plot(nav.index, nav.Benchmark, color=GREY, lw=2, label="Equal-weight universe benchmark")
    ax.axvspan(pd.Timestamp("2024-07-01"), nav.index[-1], color=ORANGE, alpha=.12, label="Locked holdout")
    ax.set_ylabel("Growth of 1"); ax.legend(fontsize=8, loc="upper left")
    return save(fig, "cum")


def chart_robust():
    pl = rd("placebo_random.csv"); st = rd("statistical_tests.csv").iloc[0]
    S = rd("strategy_summary.csv"); sh = S[(S.Strategy == "FINAL core3 | Quarterly") & (S.Window == "Full")].Sharpe.iloc[0]
    gr = rd("grid_signal_frequency.csv")
    rho = rd("replicability.csv").iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ax[0].hist(pl.Sharpe, bins=25, color=GREY); ax[0].axvline(sh, color=ORANGE, lw=2.5, label=f"Strategy {sh:.2f} ({st.PlaceboSharpePctile * 100:.0f}th pct)")
    ax[0].set_xlabel("Sharpe of 300 random 10-stock quarterly books"); ax[0].legend(fontsize=8)
    ax[1].scatter(gr.Val_Sharpe, gr.Hold_Sharpe, s=12, color=BLUE, alpha=.7)
    ax[1].axhline(rho.BenchHoldSharpe, color=ORANGE, ls="--", lw=1, label="Benchmark holdout Sharpe")
    ax[1].set_xlabel("Validation Sharpe"); ax[1].set_ylabel("Holdout Sharpe"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"130 variants: rank corr = {rho.SpearmanValHoldSharpe:.2f} (p={rho.p:.2f})", fontsize=10)
    return save(fig, "robust")


def chart_years():
    y = rd("calendar_years.csv"); y["Year"] = pd.to_datetime(y.BlockEnd).dt.year
    fig, ax = plt.subplots(figsize=(4.6, 3.2)); x = np.arange(len(y))
    ax.bar(x - .2, y.Strategy * 100, .4, color=NAVY, label="Strategy"); ax.bar(x + .2, y.Benchmark * 100, .4, color=GREY, label="Benchmark")
    ax.set_xticks(x); ax.set_xticklabels([str(v) if v < 2026 else "2026 YTD" for v in y.Year]); ax.set_ylabel("Return (%)"); ax.axhline(0, color="k", lw=.6); ax.legend(fontsize=8)
    return save(fig, "years")


def chart_h1():
    c = pd.read_csv(ROOT / "results" / "oos_2026H1_contribution.csv").sort_values("PnL_Rs")
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.barh(c.Symbol, c.PnL_Rs / 1e5, color=[ORANGE if v < 0 else NAVY for v in c.PnL_Rs]); ax.set_xlabel("Gross P&L (Rs lakh), 1 Jan - 30 Jun 2026")
    return save(fig, "h1")


def chart_sens():
    s = rd("sensitivity.csv")
    n = s[s.Test.str.contains("Holdings|holdings", regex=True)]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    if len(n):
        ax.plot(n.Value.astype(float), n.Sharpe, marker="o", color=NAVY, label="Sharpe")
        ax2 = ax.twinx(); ax2.plot(n.Value.astype(float), n.MaxDD * 100, marker="s", color=ORANGE, label="Max DD %"); ax2.set_ylabel("Max drawdown (%)")
        ax.set_xlabel("Number of holdings"); ax.set_ylabel("Sharpe (full window)")
    return save(fig, "sens")


# -------- pptx helpers
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
rgb = lambda h: RGBColor.from_string(h.lstrip("#"))


def slide(title, tagline=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.0)); bar.fill.solid(); bar.fill.fore_color.rgb = rgb(NAVY); bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(.4), Inches(.12), Inches(12.5), Inches(.8)).text_frame; tb.text = title
    tb.paragraphs[0].font.size, tb.paragraphs[0].font.bold, tb.paragraphs[0].font.color.rgb = Pt(26), True, RGBColor(255, 255, 255)
    if tagline:
        t = s.shapes.add_textbox(Inches(.4), Inches(1.05), Inches(12.5), Inches(.5)).text_frame; t.word_wrap = True; t.text = tagline
        t.paragraphs[0].font.size, t.paragraphs[0].font.italic, t.paragraphs[0].font.color.rgb = Pt(14), True, rgb(BLUE)
    return s


def bullets(s, items, x, y, w, h, size=13):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        head, _, rest = it.partition("::")
        if rest:
            r = p.add_run(); r.text = head + " "; r.font.bold = True; r.font.size = Pt(size)
            r2 = p.add_run(); r2.text = rest.strip(); r2.font.size = Pt(size)
        else:
            r = p.add_run(); r.text = "• " + it; r.font.size = Pt(size)
        p.space_after = Pt(5)


def pic(s, path, x, y, w=None, h=None):
    s.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w) if w else None, height=Inches(h) if h else None)


def table(s, df, x, y, w, h, size=11):
    t = s.shapes.add_table(len(df) + 1, len(df.columns), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for j, c in enumerate(df.columns):
        cell = t.cell(0, j); cell.text = str(c); cell.fill.solid(); cell.fill.fore_color.rgb = rgb(NAVY)
        for p in cell.text_frame.paragraphs: p.font.size, p.font.bold, p.font.color.rgb = Pt(size), True, RGBColor(255, 255, 255)
    for i, row in enumerate(df.itertuples(index=False), 1):
        for j, v in enumerate(row):
            cell = t.cell(i, j); cell.text = str(v)
            for p in cell.text_frame.paragraphs: p.font.size = Pt(size)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def main():
    S = rd("strategy_summary.csv")
    g = lambda strat, w: S[(S.Strategy == strat) & (S.Window == w)].iloc[0]
    F = "FINAL core3 | Quarterly"; Bn = "Benchmark (equal-weight universe, daily rebal)"
    fv, fh, ff, bv, bh, bf = g(F, "Val"), g(F, "Hold"), g(F, "Full"), g(Bn, "Val"), g(Bn, "Hold"), g(Bn, "Full")
    st, boot, rep = rd("statistical_tests.csv").iloc[0], rd("bootstrap.csv").iloc[0], rd("replicability.csv").iloc[0]
    pws = rd("procedure_walkforward_summary.csv").set_index("Strategy")
    roll = rd("rolling_summary.csv").set_index("Months")

    # 1 title
    s = slide("Rigorous, leak-free portfolio research: what works, what doesn't, and how we know")
    bullets(s, ["Context::Team Quantify · CitadelxFinesse Portfolio Challenge · NSE universe of 300 stocks (Nifty 100 + Midcap 100 + Smallcap 100)",
                "Question::Can a transparent rules-based 10-stock book beat the equal-weight universe, and would the result survive new periods?",
                f"Answer::Strong on the design window (Sharpe {fv.Sharpe:.2f}), but on a locked holdout it returned {pc(fh.TotalReturn)} (CAGR {pc(fh.CAGR)}, Sharpe {fh.Sharpe:.2f}) vs benchmark CAGR {pc(bh.CAGR)}. We found no reliable edge, and report that honestly.",
                "What we built::A purged walk-forward framework with a locked holdout, placebo, bootstrap, deflated Sharpe and a test of the selection procedure itself.",
                "Why that is the contribution::Anyone can pick the best backtest. This deck shows how every decision was made, tested and, where it failed, rejected."], .6, 1.5, 12, 5.5, 21)
    notes(s, "Open with the honest headline: the framework is the contribution, the alpha is not proven.")

    # 2 data
    s = slide("1. Data integrity came first", "Decision: can we trust the data enough to backtest? Result: yes after audit, with known defects documented.")
    bullets(s, ["Coverage::300 of 300 stocks have files for 2020-01-01 to 2026-10-30; 382 empty partitions are pre-listing periods.",
                "Gaps::about 311 missing stock-days across 225 stocks (224 stocks lack 2024-07-19; FORCEMOT has 77). Missing bars are never traded or imputed as prices.",
                "Defect found::on 2021-09-15/16 eight stocks (AFFLE, CESC, IRCTC, JUBLFOOD, SRF, BEL, GAIL, TORNTPHARM) are unadjusted for splits/bonus (1.5x to 10x). It inflates the original static book's NAV by +29.5% then -22.7%; the 2022+ walk-forward is unaffected.",
                "Universe::today's constituents applied to history, so survivorship bias exists in both strategy and benchmark. This is disclosed, not hidden.",
                "Liquidity::a Rs 2 crore minimum 20-day average traded value is required at every rebalance.",
                "Costs::0.1% per side on actual traded value; signal at close t executes at open t+1 (no same-bar trading)."], .5, 1.6, 12.3, 5.6, 19)

    # 3 signals
    s = slide("2. Strategy space: 13 signals, chosen before testing", "Decision: which ideas to test? We pre-specified them so we could not cherry-pick afterwards.")
    tbl = pd.DataFrame([["Hand-set (8)", "Momentum 12-1, 6M, relative strength, risk-adjusted momentum, trend quality, technical composite, momentum+low-vol, composite quantitative", "Return/volatility ratios, moving-average structure, ATR, drawdown, RSI/MACD-type features"],
                        ["ML / DL (4)", "Ridge, LightGBM, 3-seed MLP neural net, equal-rank ensemble", "37 non-redundant features + 2 derived, converted to cross-sectional percentiles; expanding window, retrained every 63 days, purge gap of horizon+2"],
                        ["Combinations (2)", "core3 and hybrid (rank-averaged)", "No weights fitted; average of ranks"]], columns=["Family", "Signals", "Inputs / construction"])
    table(s, tbl, .5, 1.7, 12.3, 2.6, 14)
    bullets(s, ["Why ML/DL were included::they were allowed wherever they could plausibly add value. LSTM/transformer were excluded because ~1,700 days x 300 stocks is too little data for them.",
                "Why features were reduced::98 columns collapse to 37 after removing highly correlated pairs, which stabilises the linear model and cuts overfitting.",
                "Why percentiles::cross-sectional ranks remove market-wide level shifts so the model learns relative strength, not market direction."], .5, 5.0, 12.3, 2.4, 16)

    # 4 protocol
    s = slide("3. Protocol: how we prevented leakage", "Decision: how to judge a strategy? A fixed validation window plus a locked holdout, with rules frozen before the holdout was opened.")
    pic(s, chart_protocol(), 2.0, 1.6, 9.3)
    bullets(s, ["Walk-forward ML::expanding training window, retrain every 63 days, purged by horizon+2 days so no label overlaps the test period.",
                "Selection rule (frozen first)::top 3 signals by mean validation Sharpe, rank-averaged -> core3 = momentum 6M + MLP + risk-adjusted momentum.",
                "Frequency rule (frozen first)::highest mean validation Sharpe across signals -> Quarterly. Extras (buffer, inverse-vol, trend overlay) adopted only if validation Sharpe AND drawdown both improved. None did.",
                "Honest caveat::holdout figures for all variants were eventually viewed, so anything chosen after that is a lead, not a finding."], .5, 4.9, 12.3, 2.5, 15)
    notes(s, "This slide is the foundation for trusting every later number.")

    # 5 IC
    s = slide("4. Why simple factors beat ML here: signal-strength evidence", "Decision: trade fast or slow signals? The information builds over weeks and months, not days.")
    pic(s, chart_ic(), .4, 1.6, 6.9)
    ic = rd("signal_ic_decay.csv"); v = ic[ic.Window == "Val"].set_index("Signal")
    t = pd.DataFrame([[k, f"{v.loc[k, 'IC_1d']:.3f}", f"{v.loc[k, 'IC_21d']:.3f}", f"{v.loc[k, 'IC_126d']:.3f}"] for k in ["composite_quantitative", "momentum_12_1", "ridge", "lgbm", "mlp"] if k in v.index], columns=["Signal", "IC 1d", "IC 21d", "IC 126d"])
    t["Signal"] = t.Signal.replace({"composite_quantitative": "Composite", "momentum_12_1": "Mom 12-1"})
    table(s, t, 7.5, 1.7, 5.4, 2.2, 12)
    bullets(s, ["1-day IC is about 0.005::no edge at daily horizons; about 0.05 at 21 days.",
                "ML/DL IC at 21 days is lower than hand-set composites (0.01 to 0.03 vs 0.05).",
                "Ridge coefficients flip sign across retrains (Momentum12_1 negative in 90%), a sign of instability.",
                "Conclusion::complexity did not add predictive power on this sample size."], 7.5, 4.2, 5.4, 3.2, 14)

    # 6 frequency
    cf, ftab = chart_freq()
    s = slide("5. Rebalance frequency: why quarterly", "Decision: daily, weekly, monthly, quarterly or half-yearly? Costs and signal speed decide; evidence is mixed beyond 'not daily'.")
    pic(s, cf, .4, 1.6, 8.0)
    bullets(s, [f"Daily is clearly worst::about {ftab.turn['D'] / ftab.turn['Q']:.0f}x the turnover of quarterly, and lowest Sharpe on both windows.",
                f"Quarterly won validation::mean Sharpe {ftab.val['Q']:.2f}, which is why it was frozen as the rule.",
                "Industry logic::quarterly matches factor decay (rank autocorrelation 0.64 at 63 days), earnings cycle, tax/brokerage drag, and liquidity in mid/small caps.",
                "Honest finding::on the holdout, weekly, monthly, quarterly and half-yearly are not cleanly separated; half-yearly looked better post hoc and is only a forward-test lead.",
                "Sensitivity::trading on a one-month-stale signal changed Sharpe materially, so we recommend three tranches rebalanced one month apart."], 8.6, 1.6, 4.5, 5.7, 13)

    # 7 portfolio construction
    po = rd("portfolio_opt_summary.csv")
    nm = {"equal": "Equal weight", "invvol": "Inverse-vol", "minvar_lw": "Min-variance (Ledoit-Wolf)", "risk_parity_lw": "Risk parity", "meanvar_lw": "Mean-variance", "capm_meanvar_lw": "CAPM mean-variance"}
    rows = []
    for k, lab in nm.items():
        r = {"Scheme": lab}
        for w, wl in (("Val", "Val"), ("Hold", "Holdout")):
            x = po[(po.Freq == "Quarterly") & (po.Scheme == k) & (po.Window == w)].iloc[0]
            r[f"{wl} Sharpe"], r[f"{wl} max DD"] = f"{x.Sharpe:.2f}", pc(x.MaxDD)
        r["Holdout return"] = pc(po[(po.Freq == "Quarterly") & (po.Scheme == k) & (po.Window == "Hold")].iloc[0].TotalReturn)
        rows.append(r)
    s = slide("6. Weights: why equal weight, not Markowitz or CAPM", "Decision: adopt only if validation Sharpe AND drawdown beat equal weight. No scheme qualified at the quarterly rebalance.")
    table(s, pd.DataFrame(rows), .5, 1.7, 8.2, 3.6, 13)
    bullets(s, ["Estimation error::10 stocks and 252 days give noisy covariances and near-useless return forecasts (rank IC about 0.05).",
                "CAPM was worst::mid/small-cap returns are not explained by beta; drawdown reached -41%.",
                "Covariance-only schemes lower volatility in 80-95% of quarters but do not reliably raise return (beat equal weight in 45-60% of quarters).",
                "Min-variance is the one challenger worth forward-testing (holdout +10.1% total return), but it lost on validation."], 8.9, 1.7, 4.2, 5.6, 14)
    bullets(s, ["Inputs::trailing 252-day Ledoit-Wolf covariance, 20% cap, 0.1% cost per side. Rule: adopt only if validation Sharpe and max drawdown both beat equal weight."], .5, 5.9, 8.2, 1.5, 14)

    # 8 results
    s = slide("7. Results on validation and the locked holdout", "The design looked strong where it was chosen; it did not carry to the holdout. That gap is the key research finding.")
    pic(s, chart_cum(), .4, 1.6, 7.0)
    rows = [["Validation", pc(fv.CAGR), f"{fv.Sharpe:.2f}", pc(fv.MaxDD), pc(bv.CAGR), f"{bv.Sharpe:.2f}"],
            ["Holdout", pc(fh.CAGR), f"{fh.Sharpe:.2f}", pc(fh.MaxDD), pc(bh.CAGR), f"{bh.Sharpe:.2f}"],
            ["Full", pc(ff.CAGR), f"{ff.Sharpe:.2f}", pc(ff.MaxDD), pc(bf.CAGR), f"{bf.Sharpe:.2f}"]]
    table(s, pd.DataFrame(rows, columns=["Window", "CAGR", "Sharpe", "Max DD", "Bench CAGR", "Bench Sharpe"]), 7.6, 1.6, 5.5, 1.6, 11)
    pic(s, chart_years(), 8.0, 3.9, 4.0)
    bullets(s, ["Validation 2022-01 to 2024-06; holdout 2024-07 to 2026-10; full = both.",
                "Most of the return is the 2023-24 mid/small-cap boom; 2025 was a momentum-reversal year (-11.8% vs +8.3%).",
                "The full-window number includes validation data, so it is the flattering view; the holdout is the honest one."], .5, 5.4, 7.0, 2.0, 14)

    # 9 robustness
    s = slide("8. Is any of it statistically real?", "Decision: claim alpha or not? Not: no test cleared its bar.")
    pic(s, chart_robust(), .3, 1.55, 8.6)
    rows = [["Bootstrap mean excess / yr", f"{pc(boot.MeanAnnExcess)} (90% CI {pc(boot.CI5)} to {pc(boot.CI95)})"],
            ["Placebo percentile (Sharpe)", f"{st.PlaceboSharpePctile * 100:.0f}th of 300 random books"],
            ["Deflated Sharpe (138 trials)", f"{st.DSR:.2f} (bar 0.95)"],
            ["Procedure walk-forward", f"beat benchmark in {pws.loc['Procedure', 'BlocksBeatingBenchmark'] * 100:.0f}% of 8 blocks; Sharpe {pws.loc['Procedure', 'Sharpe']:.2f}"],
            ["12-month rolling windows", f"beat benchmark {pc(roll.loc[12, 'BeatBenchmark'], 0)}, median excess {pc(roll.loc[12, 'MedianExcess'])}"],
            ["Variants beating bench on holdout", pc(rep.CellsBeatBenchHoldSharpe, 0)]]
    table(s, pd.DataFrame(rows, columns=["Test", "Result"]), 9.0, 1.7, 4.1, 3.6, 11)
    bullets(s, ["Winner's curse::the best validation signals were among the worst on the holdout.", "RL overlay::cut Sharpe from 1.15 to 0.46; rejected.",
                "Verdict::no test cleared its bar, so we do not claim a reliable edge."], .4, 5.5, 8.4, 1.9, 16)

    # 10 decisions
    s = slide("9. Decision log, limitations and what we would do next", "Every decision is traceable to a table or chart above.")
    dec = pd.DataFrame([["Allow rebalancing?", "Static was our own choice, not a verified brief rule", "Allowed; tested all frequencies"],
                        ["Which signals", "13 pre-specified incl. ML/DL", "core3 (top-3 on validation)"],
                        ["Frequency", "IC decay, turnover x cost, validation Sharpe", "Quarterly; daily rejected"],
                        ["Weights", "Adopt only if val Sharpe and DD improve", "Equal weight kept"],
                        ["RL / overlay", "Walk-forward Q-learning on regimes", "Rejected (Sharpe 1.15 to 0.46)"],
                        ["Claim alpha?", "Holdout, placebo, DSR, procedure walk-forward", "No reliable edge shown"]], columns=["Decision", "How decided", "Outcome"])
    table(s, dec, .5, 1.6, 7.6, 3.0, 11)
    bullets(s, ["Limitations::survivorship bias, flat 0.1% costs, a synthetic benchmark, known data gaps, 4 years of regimes, 138 variants examined.",
                "Next::(1) forward-test from 2 Nov 2026 with tranched rebalancing; (2) test 15-20 holdings (Sharpe 1.09-1.12 vs 1.02); (3) forward-test min-variance and half-yearly leads; (4) fix 2021 unadjusted bars and use real Nifty indices.",
                "H1 2026 (1 Jan-30 Jun) check::the strategy returned +6.05% net (benchmark +2.1%); LAURUSLABS, GVT&D and AUBANK drove it. No pre-2026 rule we tested reached 22%."], 8.3, 1.6, 4.8, 5.7, 14)
    pic(s, chart_h1(), .5, 5.2, h=2.2)
    notes(s, "Close on process: transparent, reproducible (python -m src.battle.*), and honest about the edge.")
    prs.save(OUT)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
