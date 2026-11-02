"""HTML text for the battle-ready report. Every headline number is read from results/battle/*.csv."""
import pandas as pd

from src.battle.data import BATTLE_DIR

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.55;color:#1f2933;max-width:1100px;margin:0 auto;padding:24px 28px}
h1{border-bottom:3px solid #4c78a8;padding-bottom:8px}h2{margin-top:2.2em;border-bottom:1px solid #d0d7de;padding-bottom:4px;color:#17365d}
h3{color:#244b7a;margin-top:1.5em}table{border-collapse:collapse;margin:12px 0;font-size:.88em;display:block;overflow-x:auto}
th,td{border:1px solid #d0d7de;padding:5px 9px;text-align:left;vertical-align:top}th{background:#eef3f9}tr:nth-child(even) td{background:#f8fafc}
code{background:#f0f3f6;padding:1px 5px;border-radius:4px}figure{margin:16px 0;text-align:center}figure img{max-width:100%;border:1px solid #e1e4e8;border-radius:6px}
figcaption{font-size:.85em;color:#57606a}.verdict{background:#fff4e5;border-left:5px solid #e8890c;padding:10px 16px;margin:14px 0}
.good{background:#e9f7ee;border-left:5px solid #2e9d57;padding:10px 16px;margin:14px 0}.note{background:#f0f6ff;border-left:5px solid #4c78a8;padding:10px 16px;margin:14px 0}
.tag{display:inline-block;font-size:.75em;padding:1px 7px;border-radius:10px;color:#fff}.pre{background:#4c78a8}.post{background:#e8890c}
"""


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def tbl(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    df = df.copy()
    for c, f in fmt.items():
        if c in df:
            df[c] = df[c].map(lambda v, f=f: "" if pd.isna(v) else f(v) if callable(f) else format(v, f))
    return df.to_html(index=False, border=0, escape=False)


def render(charts: dict, cur: pd.DataFrame, fresh: pd.DataFrame) -> str:
    L = lambda n: pd.read_csv(BATTLE_DIR / n)
    S = L("strategy_summary.csv")
    g = lambda strat, win: S[(S.Strategy == strat) & (S.Window == win)].iloc[0]
    FIN = "FINAL core3 | Quarterly"
    fv, fh, ff = g(FIN, "Val"), g(FIN, "Hold"), g(FIN, "Full")
    B = "Benchmark (equal-weight universe, daily rebal)"
    bv, bh, bf = g(B, "Val"), g(B, "Hold"), g(B, "Full")
    stt, boot, rep = L("statistical_tests.csv").iloc[0], L("bootstrap.csv").iloc[0], L("replicability.csv").iloc[0]
    roll = L("rolling_summary.csv").set_index("Months")
    pw, pws = L("procedure_walkforward_blocks.csv"), L("procedure_walkforward_summary.csv")
    pws_i = pws.set_index("Strategy")
    sens, design, regime, rl, ic = L("sensitivity.csv"), L("design_variants.csv"), L("regime_analysis.csv"), L("rl_overlay.csv"), L("signal_ic_decay.csv")
    grid = L("grid_signal_frequency.csv")
    blocks, years = L("halfyear_blocks.csv"), L("calendar_years.csv")

    def rowset(names):
        rows = []
        for lab, key in names:
            v, h, f = g(key, "Val"), g(key, "Hold"), g(key, "Full")
            rows.append({"Strategy": lab, "Val CAGR": pct(v.CAGR), "Val Sharpe": f"{v.Sharpe:.2f}", "Hold CAGR": pct(h.CAGR),
                         "Hold Sharpe": f"{h.Sharpe:.2f}", "Full CAGR": pct(f.CAGR), "Full Sharpe": f"{f.Sharpe:.2f}",
                         "Full MaxDD": pct(f.MaxDD), "Turnover/yr": "" if pd.isna(f.Turnover) else f"{f.Turnover:.1f}x"})
        return pd.DataFrame(rows)

    freq_rows = rowset([("Daily", "core3 | Daily"), ("Weekly", "core3 | Weekly"), ("Monthly", "core3 | Monthly"),
                        ("Quarterly (FINAL)", FIN), ("Half-yearly", "core3 | Half-yearly"), ("Static (never rebalance)", "core3 | Static (no rebalance)"),
                        ("Benchmark", B)])
    cmp_rows = rowset([("FINAL core3, quarterly", FIN), ("Original design: composite, static", "Original composite_quantitative | Static"),
                       ("Composite, quarterly", "composite_quantitative | Quarterly"), ("ML ensemble, quarterly", "ml_ensemble | Quarterly"),
                       ("momentum_6m, quarterly", "momentum_6m | Quarterly"), ("Benchmark", B)])

    cost = {r["Value"]: r for _, r in sens[sens.Test == "Cost per side"].iterrows()}
    drag = freq_rows.copy()
    drag = pd.DataFrame([{"Frequency": n, "One-way turnover / yr": f"{g(k, 'Full').Turnover:.1f}x",
                          "Approx. cost drag at 0.1%/side": pct(2 * g(k, "Full").Turnover * 0.001, 1)}
                         for n, k in [("Daily", "core3 | Daily"), ("Weekly", "core3 | Weekly"), ("Monthly", "core3 | Monthly"),
                                      ("Quarterly", FIN), ("Half-yearly", "core3 | Half-yearly")]])
    gm = grid.groupby("Freq")[["Val_Sharpe", "Hold_Sharpe", "Val_Turnover"]].mean().loc[list("DWMQS")].reset_index()
    gm["Freq"] = ["Daily", "Weekly", "Monthly", "Quarterly", "Half-yearly"]
    gm.columns = ["Frequency", "Avg validation Sharpe (13 signals)", "Avg holdout Sharpe", "Avg turnover/yr"]
    gs = grid.groupby("Signal")[["Val_Sharpe", "Hold_Sharpe"]].mean().sort_values("Val_Sharpe", ascending=False).reset_index()
    gs.columns = ["Signal", "Avg validation Sharpe", "Avg holdout Sharpe"]

    icv = ic[ic.Window == "Val"].set_index("Signal")
    ic_tab = icv[["IC_1d", "IC_5d", "IC_21d", "IC_63d", "IC_126d", "RankAutocorr_21d", "RankAutocorr_63d"]].round(3).reset_index()

    pw_t = pw[["Origin", "TestEnd", "ChosenSignals", "ChosenFreq", "Procedure", "Benchmark", "ProcedureExcess"]].copy()
    pw_t.columns = ["Origin", "Test end", "Signals chosen (validation only)", "Freq", "Strategy", "Benchmark", "Excess"]
    pws_t = pws[["Strategy", "CAGR", "Vol", "Sharpe", "MaxDD", "BlocksBeatingBenchmark"]]

    des = design[["Buffer", "Weighting", "Overlay", "Val_CAGR", "Val_Sharpe", "Val_MaxDD", "Val_Turnover"]].copy()
    des["Adopted?"] = ["YES (simplest)" if (r.Buffer == 0 and r.Weighting == "equal" and r.Overlay == "none") else "no" for r in des.itertuples()]
    des["Buffer"] = des["Buffer"].map(lambda b: "none" if b == 0 else f"rank<{b}")

    cur_t = cur[["Symbol", "Weight", "HeldSince", "ReturnSinceEntry", "CurrentRank_core3"] + [f"Rank_{s}" for s in ("momentum_6m", "mlp", "risk_adjusted_momentum")]
                + ["InTop10Today"]].copy()
    cur_t.columns = ["Symbol", "Weight", "Held since", "Return since entry", "Rank today (core3)", "Rank momentum_6m", "Rank mlp", "Rank risk-adj mom", "Still top-10?"]
    n_hold = len(cur)
    still = int(cur["InTop10Today"].sum())

    sens_cost = sens[sens.Test == "Cost per side"][["Value", "CAGR", "Sharpe", "MaxDD"]]
    sens_n = sens[sens.Test == "Holdings"][["Value", "CAGR", "Vol", "Sharpe", "MaxDD"]]
    sens_adv = sens[sens.Test == "Min avg daily value (Rs)"][["Value", "CAGR", "Sharpe", "MaxDD"]]
    sens_stale = sens[sens.Test == "Stale signal (days)"][["Value", "CAGR", "Sharpe", "MaxDD"]]
    pf = {"CAGR": lambda v: pct(v), "MaxDD": lambda v: pct(v), "Vol": lambda v: pct(v), "Sharpe": "{:.2f}".format}

    rl_t = rl[["Strategy", "TotalReturn", "CAGR", "Vol", "Sharpe", "MaxDD", "AvgExposure"]]
    reg_t = regime[["Regime", "ShareOfDays", "StrategyAnnReturn", "BenchAnnReturn", "AnnExcess", "StrategySharpe"]]

    H = []
    a = H.append
    a(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
      f"<title>Battle-Ready Strategy Report</title><style>{CSS}</style></head><body>")
    a("<h1>Battle-Ready Strategy Research: Rebalancing, ML/DL/RL and Walk-Forward Validation</h1>")
    a("<p><b>Team Quantify</b> · CitadelxFinesse Challenge · data through 2026-10-30 · 300-stock NSE universe (Nifty 100 + Midcap 100 + Smallcap 100)</p>")

    # ---------------- 1 verdict
    a("<h2>1. Executive verdict</h2>")
    a(f"""<div class='verdict'><b>The honest answer: the strategy works in some periods, but this evidence does not show a reliable edge over simply owning the universe.</b>
<ul>
<li>The design was chosen on a validation window (2022-01 to 2024-06) using rules fixed beforehand, then run once on a locked holdout (2024-07 to 2026-10).</li>
<li><b>Validation:</b> CAGR {pct(fv.CAGR)}, Sharpe {fv.Sharpe:.2f} (benchmark {pct(bv.CAGR)}, {bv.Sharpe:.2f}).
<b>Holdout:</b> CAGR {pct(fh.CAGR)}, Sharpe {fh.Sharpe:.2f} (benchmark {pct(bh.CAGR)}, {bh.Sharpe:.2f}). It did not beat the benchmark on the holdout.</li>
<li>Over all walk-forward out-of-sample data (2022-01 to 2026-10): CAGR {pct(ff.CAGR)} vs {pct(bf.CAGR)}, Sharpe {ff.Sharpe:.2f} vs {bf.Sharpe:.2f}, max drawdown {pct(ff.MaxDD)} vs {pct(bf.MaxDD)}.
Return is a little higher than the benchmark's, but risk-adjusted performance is lower.</li>
<li>Re-running the selection procedure at 8 different dates beat the benchmark in only <b>{int((pw.ProcedureExcess > 0).sum())} of {len(pw)}</b> blocks.</li>
<li>Random 10-stock quarterly portfolios from the same universe have a median Sharpe of {stt.PlaceboMedianSharpe:.2f}; our strategy sits at the {stt.PlaceboSharpePctile * 100:.0f}th percentile.
Most of the return is the strong 2023-24 mid/small-cap market, not stock-picking skill.</li>
<li>Deflated Sharpe (adjusting for the {int(stt.N_trials)} variants examined) is {stt.DSR:.2f}, below the usual 0.95 bar.</li>
</ul></div>""")
    a(f"""<div class='good'><b>What the work does establish:</b>
<ul><li><b>Rebalance slowly.</b> Daily rebalancing is clearly the worst after costs, and weekly was worse on validation; the signals' information builds over months (Section 5).</li>
<li><b>ML/DL did not beat simple factors.</b> Ridge, LightGBM and a neural net had lower predictive power than hand-set momentum composites, and the RL exposure overlay hurt (Section 8).</li>
<li><b>A rigorous, leak-free test harness now exists</b> (purged walk-forward, locked holdout, placebo, bootstrap, procedure-level walk-forward) that any future idea can be run through.</li>
<li><b>Replicability is weak:</b> validation Sharpe across 130 strategy variants had a rank correlation of only {rep.SpearmanValHoldSharpe:.2f} with holdout Sharpe.
The best validation performers (momentum_6m, the MLP) were the worst on the holdout, a textbook winner's-curse.</li></ul></div>""")
    a("<p>Everything below is labelled <span class='tag pre'>pre-registered</span> (rule fixed before results were seen) or <span class='tag post'>post-hoc</span> (observed after seeing results; a hypothesis to forward-test, not a proven finding).</p>")

    # ---------------- 2 What changed
    a("<h2>2. What changed from the earlier static-portfolio report</h2>")
    a("""<ul><li>The earlier work assumed a static, buy-and-hold portfolio. That was a self-imposed design choice, not a rule found in the repo. Rebalancing is now allowed and studied.</li>
<li>The earlier headline (CAGR 53.7%, Sharpe 1.50, 2021-2025) used one formation date (2021-01-01), a current-constituent universe and a model picked after seeing results. It is an in-sample number.</li>
<li>This study replaces it with <b>walk-forward, out-of-sample</b> evaluation from 2022-01-03, with strict separation between where choices were made and where they were judged.</li>
<li>The feature panel was rebuilt through 2026-10-30 (<code>results/battle/features_full.parquet</code>); the old panel stopped at 2026-06-30.</li></ul>""")

    # ---------------- 3 protocol
    a("<h2>3. Experimental protocol (how leakage was prevented)</h2>")
    a(tbl(pd.DataFrame([
        ["Walk-forward start", "2022-01-03. Before that, ML models had under a year of history."],
        ["Validation window", "2022-01-03 to 2024-06-28. The only window used to choose the signal, frequency and design."],
        ["Locked holdout", "2024-07-01 to 2026-10-30. Evaluated after the choice was frozen. ML kept retraining forward in time, seeing only past labels."],
        ["ML training", "Expanding window, retrained every 63 trading days, 3-day sampling step to reduce overlap. Purge gap: a training row is used only if its forward-return label was fully realised before the test block started."],
        ["Labels", "Cross-sectional percentile of the forward return from next open to open h days later (h = 5, 21 or 63 depending on rebalance frequency)."],
        ["Features", "37 non-redundant features chosen from the 98-column panel, plus 2 derived (39 in total). Each is converted to a cross-sectional percentile per date, so no market-level information leaks across time."],
        ["Execution", "Signal at close of day t, trade at the open of day t+1. Cost 0.1% of traded value per side, charged on actual turnover at every rebalance. Cash earns 0%."],
        ["Investability filter", "Needs a real bar that day, a full feature set, and 20-day average traded value of at least Rs 2 crore. Prices are forward-filled only to value existing holdings across data gaps, never to make a stock eligible."],
        ["Holdings", "10 names, equal weight (the 10-holding cap is from config.py)."],
        ["Benchmark", "Equal-weight universe rebalanced daily, built from the same stock data (no Nifty index series exists locally). It has the same survivorship bias as the strategy, which flatters both."],
    ], columns=["Item", "Rule"])))
    a("<h3>Selection rules, frozen before the holdout was inspected <span class='tag pre'>pre-registered</span></h3>")
    a("""<ol><li><b>Signal:</b> the 3 of 13 candidate signals with the highest mean validation Sharpe across all 5 frequencies were rank-averaged into one signal, <code>core3</code> (momentum_6m, mlp, risk_adjusted_momentum).</li>
<li><b>Frequency:</b> the frequency with the highest mean validation Sharpe across all signals, which was quarterly.</li>
<li><b>Extras</b> (hold buffer, inverse-volatility weights, trend-filter overlay): adopted only if validation Sharpe <i>and</i> drawdown both improved. None did, so the design stayed simple.</li></ol>
<p>Choosing by group averages instead of the single best cell reduces, but cannot remove, selection bias.</p>""")

    # ---------------- 4 signals / ML
    a("<h2>4. Signals tested</h2>")
    a("""<table><tr><th>Type</th><th>Models</th><th>How it works</th></tr>
<tr><td>Hand-set factor composites (8)</td><td>composite_quantitative, momentum_12_1, momentum_6m, relative_strength, risk_adjusted_momentum, technical_composite, trend_quality, momentum_lowvol</td><td>Weighted mean of cross-sectional percentiles of fixed factors. <code>momentum_lowvol</code> was written before any validation results; it blends momentum and relative strength with a low-volatility tilt.</td></tr>
<tr><td>Ridge regression (ML)</td><td>ridge</td><td>Linear model on 39 percentile features, alpha=200. Learns factor weights instead of fixing them by hand.</td></tr>
<tr><td>Gradient-boosted trees (ML)</td><td>lgbm</td><td>LightGBM, 250 shallow trees (15 leaves, min 300 samples per leaf, subsampling, L2). Captures interactions and non-linearity.</td></tr>
<tr><td>Neural network (DL)</td><td>mlp</td><td>PyTorch MLP 39-64-32-1 with dropout 0.2, AdamW, Huber loss, mean of 3 seeds.</td></tr>
<tr><td>Combinations</td><td>ml_ensemble, hybrid, core3</td><td>Rank averages: the 3 ML models; composite plus ML ensemble; the top-3 validation signals.</td></tr>
<tr><td>Reinforcement learning</td><td>RL exposure overlay</td><td>Tabular Q-learning (gamma = 0) choosing 0%, 50% or 100% market exposure from trend and volatility regime (Section 8).</td></tr></table>""")
    a("<p>Hyper-parameters for every ML model were fixed in advance and not tuned, so no tuning could overfit the validation window. "
      "LSTM and transformer sequence models were <b>not</b> built: with roughly 240 stocks and 4 years of data per training round, a deep sequence model would have far more parameters than independent observations.</p>")
    a("<h3>Predictive power: rank IC by horizon (validation window)</h3>")
    a(tbl(ic_tab))
    a(charts["ic"])
    a("<p>IC is the cross-sectional rank correlation between a signal and the subsequent return. Hand-set momentum composites (IC about 0.05 at 21 days) are stronger than ridge (0.03), LightGBM (0.02) and the MLP (0.01). "
      "<b>ML did not find a stronger signal than simple momentum.</b></p>")
    a(charts["ridge"])
    a("<p>The ridge model's learned weights changed sign across retrains. It tilted toward high recent volatility (ATR/Close, positive in every retrain) and against long-run momentum (negative for Momentum12_1 in 90% of retrains), which is an opposite stance to the hand-set composites. "
      "Both stances are regime bets, and the instability is why the learned models did not add dependable value.</p>")
    a(f"<p>Average validation and holdout Sharpe for each signal across all frequencies and buffers:</p>")
    a(tbl(gs, {"Avg validation Sharpe": "{:.2f}".format, "Avg holdout Sharpe": "{:.2f}".format}))

    # ---------------- 5 frequency
    a("<h2>5. Why this rebalance frequency? Industry reasoning plus evidence</h2>")
    a("""<p>The standard reasoning in systematic investing is to rebalance <b>no faster than the signal's information decays, and no faster than the cost of trading allows</b>. Three tests applied that here.</p>
<ol><li><b>Signal speed.</b> Rank IC at 1 day is near zero (about 0.005) and grows to about 0.05 at 21 days and about 0.08 at 126 days (chart above). The rank order of the composite is 0.99 correlated day to day and still about 0.64 correlated 63 days later. There is nothing to harvest by trading daily or weekly on these signals.</li>
<li><b>Cost drag.</b> Each rebalance sells names that dropped out and buys new ones. Annual cost is roughly 2 x one-way turnover x cost per side:</li></ol>""")
    a(tbl(drag))
    a("""<ol start='3'><li><b>Practice.</b> Daily rebalancing is generally used only for short-horizon signals with intraday data and very low execution cost. Slow-factor strategies (momentum, trend) are typically rebalanced monthly to semi-annually, which is commonly how factor indices and quant funds operate. This is general industry practice, not something checked against a specific index methodology in this study. Indian equity delivery trades also carry securities transaction tax (roughly 0.1% each way at the time of writing; check current rates), plus brokerage, stamp duty and market impact in small caps, so per-side costs of 0.1-0.3% are plausible, and short-term capital-gains tax (not modelled) penalises high turnover further.</li></ol>""")
    a(charts["freq"])
    a(tbl(freq_rows))
    a("<p>Averaging across all 13 signals so no single signal drives the answer:</p>")
    a(tbl(gm, {c: "{:.2f}".format for c in gm.columns[1:]}))
    a(f"""<div class='note'><b>Decision <span class='tag pre'>pre-registered</span>:</b> Quarterly. It had the highest average validation Sharpe across signals, a turnover of about 3x per year (cost drag about {pct(2 * g(FIN, 'Full').Turnover * 0.001)} a year), and it matches the multi-month information horizon.
<b>Evidence the choice is not critical:</b> monthly, quarterly and half-yearly were within about 0.05 Sharpe of each other on validation. Daily was the worst on both windows. Weekly was worse than quarterly on validation, but on the holdout average it was not (see table), so the holdout does not cleanly separate weekly, monthly and quarterly.</div>""")
    a(f"""<div class='note'><b>Post-hoc observation <span class='tag post'>post-hoc</span>:</b> half-yearly and static versions did best on the holdout (core3: Sharpe {g('core3 | Half-yearly', 'Hold').Sharpe:.2f} and {g('core3 | Static (no rebalance)', 'Hold').Sharpe:.2f} vs {fh.Sharpe:.2f} for quarterly), consistent with slow signals and with 2025's sharp momentum reversal punishing frequent re-buying of recent winners. I did <b>not</b> switch the official choice because of this: that would be selecting on the holdout.
It is a hypothesis to forward-test, not a result.</div>""")
    a("<h3>Reducing turnover without slowing down: the hold buffer</h3>")
    a(tbl(des, {"Val_CAGR": pct, "Val_Sharpe": "{:.2f}".format, "Val_MaxDD": pct, "Val_Turnover": "{:.1f}".format}))
    a("<p>Keeping a holding until it falls below rank 20, inverse-volatility weights and a 200-day trend filter were all tested for the quarterly core signal. "
      "No variant improved both Sharpe and drawdown on validation, so by the pre-set rule the simplest design was kept: equal weight, no buffer, no overlay.</p>")

    # ---------------- 6 why this strategy in this period
    a("<h2>6. Why this strategy, and why in this period?</h2>")
    a("""<p><b>Why momentum-family signals.</b> Price momentum and relative strength are among the most widely documented return premia in the academic literature, with a behavioural explanation (under-reaction and herding). They are cheap to compute and need no fundamental data. In this universe they had the highest predictive power (Section 4), so a momentum-anchored composite was the evidence-led choice.</p>
<p><b>Why 2022-2026 is a demanding test.</b> It contains a strong 2023-24 small/mid-cap bull market (where momentum shines), a sharp 2025 reversal (where it hurts), and a flat 2026, so it covers both kinds of regime.</p>""")
    a("<h3>How the strategy behaves across regimes</h3>")
    a(tbl(reg_t, {"ShareOfDays": pct, "StrategyAnnReturn": pct, "BenchAnnReturn": pct, "AnnExcess": pct, "StrategySharpe": "{:.2f}".format}))
    a("<p>Annualised arithmetic returns by market state known the day before. The strategy's edge over the benchmark is positive in most states, but it gives some back in high-volatility periods. "
      "The regime split does not give a reliable timing rule: the overlay tests (Sections 5 and 8) that tried to act on these regimes lost money.</p>")
    a("<p><b>Why momentum struggled in 2025:</b> I did not run a causal test. The pattern is consistent with a momentum reversal after a market sell-off (recent winners lag the rebound), which is a known momentum risk. "
      "A concentrated 10-stock book carries about 24% volatility vs 17% for the universe, so such episodes cost more.</p>")

    # ---------------- 7 results
    a("<h2>7. Results on your own out-of-sample periods</h2>")
    a("<h3>7.1 Validation vs locked holdout</h3>")
    a(tbl(cmp_rows))
    a(charts["equity"])
    a(charts["drawdown"])
    a("<h3>7.2 Fresh half-year blocks (each is an out-of-sample period)</h3>")
    a(charts["blocks"])
    a(tbl(blocks[["BlockEnd", "Days", "Strategy", "Benchmark", "Excess", "StrategySharpe"]], {"Strategy": pct, "Benchmark": pct, "Excess": pct, "StrategySharpe": "{:.2f}".format}))
    a("<h3>7.3 Calendar years</h3>")
    a(tbl(years[["BlockEnd", "Days", "Strategy", "Benchmark", "Excess", "StrategySharpe"]], {"Strategy": pct, "Benchmark": pct, "Excess": pct, "StrategySharpe": "{:.2f}".format}))
    a(f"""<h3>7.4 Rolling windows (a new start every month)</h3>
<ul><li><b>6-month windows ({int(roll.loc[6, 'Windows'])}):</b> beat the benchmark in {pct(roll.loc[6, 'BeatBenchmark'], 0)}; median strategy return {pct(roll.loc[6, 'MedianStrategy'])} vs {pct(roll.loc[6, 'MedianBenchmark'])}; positive in {pct(roll.loc[6, 'PositiveReturn'], 0)}; worst excess {pct(roll.loc[6, 'WorstExcess'])}.</li>
<li><b>12-month windows ({int(roll.loc[12, 'Windows'])}):</b> beat the benchmark in {pct(roll.loc[12, 'BeatBenchmark'], 0)}; median strategy {pct(roll.loc[12, 'MedianStrategy'])} vs {pct(roll.loc[12, 'MedianBenchmark'])}; positive in {pct(roll.loc[12, 'PositiveReturn'], 0)}; worst excess {pct(roll.loc[12, 'WorstExcess'])}.</li></ul>""")
    a("<h3>7.5 Walk-forward of the whole selection procedure</h3>")
    a("<p>At each of 8 origins the frozen rule was re-run on past data only (expanding from 2022-01-03), then the chosen strategy traded the next 6 months with fresh capital and entry costs. "
      "This repeats 'select on the past, trade the future' eight times instead of once.</p>")
    a(charts["procedure"])
    a(tbl(pw_t, {"Strategy": pct, "Benchmark": pct, "Excess": pct}))
    a(tbl(pws_t, {"CAGR": pct, "Vol": pct, "Sharpe": "{:.2f}".format, "MaxDD": pct, "BlocksBeatingBenchmark": lambda v: pct(v, 0)}))
    a(f"""<p><b>Reading:</b> the re-selected strategy had a stitched Sharpe of {pws_i.loc['Procedure', 'Sharpe']:.2f} against {pws_i.loc['Benchmark', 'Sharpe']:.2f} for the benchmark, and beat it in {int((pw.ProcedureExcess > 0).sum())} of {len(pw)} blocks. It was strong in 2023 and H1 2024 and negative through 2025.
The choice made on past data did not carry into the next period reliably. That is the replicability finding, and it is a warning.</p>""")

    # ---------------- stats
    a("<h2>8. Is any of it statistically real? ML, DL and RL conclusions</h2>")
    a(f"""<table><tr><th>Test</th><th>Result</th><th>Meaning</th></tr>
<tr><td>Bootstrap (stationary, 21-day blocks) of daily excess return vs benchmark</td><td>Mean excess {pct(boot.MeanAnnExcess)}/yr, 90% interval {pct(boot.CI5)} to {pct(boot.CI95)}; positive in {pct(boot.ProbExcessPositive, 0)} of resamples</td><td>The excess return cannot be distinguished from zero.</td></tr>
<tr><td>Bootstrap Sharpe interval</td><td>{boot.Sharpe:.2f} (90%: {boot.SharpeCI5:.2f} to {boot.SharpeCI95:.2f})</td><td>Positive but wide.</td></tr>
<tr><td>Random-portfolio placebo (300 random 10-stock quarterly books)</td><td>Median Sharpe {stt.PlaceboMedianSharpe:.2f}; strategy at percentile {stt.PlaceboSharpePctile * 100:.0f} (Sharpe) and {stt.PlaceboCAGRPctile * 100:.0f} (CAGR)</td><td>Random picking in this universe also earns a lot; skill beyond the market is small.</td></tr>
<tr><td>Probabilistic Sharpe (vs 0) / Deflated Sharpe ({int(stt.N_trials)} trials)</td><td>{stt.PSR_vs_0:.2f} / {stt.DSR:.2f}</td><td>Likely positive, but not significant after allowing for the number of variants tried.</td></tr>
<tr><td>Validation-to-holdout rank correlation over 130 variants</td><td>Spearman {rep.SpearmanValHoldSharpe:.2f} (p = {rep.p:.2f}); {pct(rep.CellsBeatBenchHoldSharpe, 0)} of variants beat the benchmark's holdout Sharpe</td><td>Validation performance barely predicts holdout performance.</td></tr></table>""")
    a(charts["placebo"])
    a(charts["rep"])
    a("<h3>Machine learning, deep learning and reinforcement learning</h3>")
    a(f"""<ul><li><b>Did ML/DL help?</b> Not reliably. Ridge, LightGBM and the MLP had lower IC than hand-set momentum (Section 4). On the holdout, the MLP tied for the worst average Sharpe of the 13 signals, while the ML ensemble ranked 4th of 13.
The ML ensemble's quarterly version was the strongest full-period run (Sharpe {g('ml_ensemble | Quarterly', 'Full').Sharpe:.2f} vs benchmark {bf.Sharpe:.2f}), but it was not what the frozen rule selected, so treat that as a <span class='tag post'>post-hoc</span> lead to forward-test.</li>
<li><b>Why ML struggled here:</b> short history (about 1 to 5 years of training), a single dominant market regime, a small universe of ~240 stocks, and noisy labels. These are classic conditions for overfitting.</li>
<li><b>Where ML is still worth keeping:</b> as an ensemble ingredient that adds diversification, and for explainability checks (the learned weights show what a data-driven model thinks matters).</li></ul>""")
    a("<p><b>Reinforcement learning overlay.</b> An agent chose market exposure (0%, 50% or 100%) from trend and volatility regime using tabular Q-learning. "
      "Since the agent's action does not change the market's next state, this is a contextual bandit and gamma is 0. It was trained strictly walk-forward on the composite strategy's earlier returns and applied to core3 from 2023-02.</p>")
    a(tbl(rl_t, {"TotalReturn": pct, "CAGR": pct, "Vol": pct, "Sharpe": "{:.2f}".format, "MaxDD": pct, "AvgExposure": "{:.2f}".format}))
    a("<p><b>Result: it hurt.</b> The agent held about 62% exposure on average (0% at 8 of its 44 monthly decisions), so it lowered return more than it lowered risk (max drawdown was not improved). "
      "RL is not justified here: the problem is a sequence of independent regime bets with only a few dozen decisions, far too few to learn from. "
      "It could become useful for execution (order slicing), which needs intraday data, not for this allocation problem.</p>")

    # ---------------- sensitivities
    a("<h2>9. Sensitivity checks on the final strategy</h2>")
    a("<h3>Transaction cost per side</h3>" + tbl(sens_cost, pf))
    a("<h3>Number of holdings</h3>" + tbl(sens_n, pf) + charts["holdings"])
    a("<p><span class='tag post'>post-hoc</span> More names reduce volatility and drawdown (Sharpe rises from {:.2f} at 10 names to {:.2f} at 20). If the holding limit in <code>config.py</code> is not a hard competition rule, 15 to 20 names is the single most useful change, and it needs no model at all. The 10-name cap concentrates risk.</p>".format(
        float(sens_n[sens_n.Value == "10"].Sharpe.iloc[0]), float(sens_n[sens_n.Value == "20"].Sharpe.iloc[0])))
    a("<h3>Liquidity filter (minimum 20-day average traded value)</h3>" + tbl(sens_adv, pf))
    a("<h3>Stale signal (trading on an older ranking)</h3>" + tbl(sens_stale, pf))
    a("<p>Sharpe changed from {:.2f} to {:.2f} when the signal was one month stale, which shows the quarterly result is sensitive to the exact rebalance date.".format(float(sens_stale[sens_stale.Value == '0'].Sharpe.iloc[0]), float(sens_stale[sens_stale.Value == '21'].Sharpe.iloc[0])) + " "
      "A production rule should therefore split capital into tranches rebalanced on different dates (for example monthly tranches each held for three months) to remove timing luck.</p>")

    # ---------------- decisions
    a("<h2>10. Decision log for this phase</h2>")
    dec = pd.DataFrame([
        ["Allow rebalancing?", "The static design was our own choice, not a brief requirement. Tested all frequencies.", "Rebalancing allowed; frequency decided below."],
        ["Where to judge results", "Walk-forward, purged, with validation and a locked holdout, instead of one in-sample window.", "Protocol in Section 3."],
        ["Which signals to test", "13 candidates: 8 hand-set, 3 ML/DL, 2 combinations. Pre-specified, ML hyper-parameters not tuned.", "ML/DL included wherever it could plausibly add value; LSTM/transformer excluded (too little data)."],
        ["Rebalance frequency", "Signal-decay (IC) analysis, turnover x cost, average validation Sharpe across signals.", "Quarterly (pre-registered). Daily/weekly rejected."],
        ["Signal to trade", "Top 3 by mean validation Sharpe across frequencies, rank-averaged.", "core3 = momentum_6m + mlp + risk_adjusted_momentum."],
        ["Hold buffer / weighting / trend overlay", "Adopt only if validation Sharpe and drawdown both improve.", "None adopted: equal weight, no buffer, no overlay."],
        ["RL exposure overlay", "Walk-forward Q-learning on regime states.", "Rejected: lowered Sharpe from {:.2f} to {:.2f}.".format(rl.Sharpe.iloc[0], rl.Sharpe.iloc[1])],
        ["Does it replicate?", "Holdout, half-year blocks, rolling windows, procedure walk-forward, placebo, bootstrap, deflated Sharpe.", "Weak replicability (Section 8). No claim of reliable alpha."],
        ["Post-hoc leads (not adopted)", "Half-yearly, 15-20 names, ML ensemble were better on some views after seeing the holdout.", "Reported, not selected; to be forward-tested."],
    ], columns=["Decision", "How it was decided", "Result"])
    a(tbl(dec))

    # ---------------- live
    a("<h2>11. Live portfolio and operating plan (pre-registered strategy)</h2>")
    a(f"""<p>The strategy trades the last-quarter-end ranking, so the current book was set at the most recent quarterly rebalance. {still} of the {n_hold} holdings are still in today's top 10 (ranking at the 2026-10-30 close).
Next rebalance: signal at the close of the last trading day of December 2026, trade at the next trading day's open.</p>""")
    a(tbl(cur_t, {"Weight": pct, "Return since entry": pct, "Rank today (core3)": "{:.0f}".format, "Rank momentum_6m": "{:.0f}".format, "Rank mlp": "{:.0f}".format, "Rank risk-adj mom": "{:.0f}".format}))
    a("<p>If rebalanced today, the fresh top 10 under core3 would be:</p>")
    a(tbl(fresh, {"AlphaScore": "{:.3f}".format}))
    a("""<h3>Suggested operating rules <span class='tag post'>judgement</span></h3>
<ul><li><b>Forward-test before trusting it.</b> Paper-trade or run small for at least two quarterly rebalances. The one thing no backtest can supply is genuinely new data, and today (2026-11-02) is the first date for which that exists.</li>
<li><b>Position size:</b> 10% each at rebalance. Re-check that each name still meets the Rs 2 crore average daily value filter.</li>
<li><b>Tranching:</b> split capital into three tranches rebalanced one month apart to remove rebalance-date luck.</li>
<li><b>Review triggers:</b> trailing 12-month excess return below -20%, drawdown beyond the worst seen in testing (about -32%), or a holding with a data gap or corporate action.</li>
<li><b>Data hygiene first:</b> fix the missing dates (especially 2024-07-19), verify split/bonus adjustment, and replace the synthetic benchmark with the real Nifty indices before any live use.</li></ul>""")

    # ---------------- limitations
    a("<h2>12. Limitations</h2>")
    a("""<ul><li><b>Survivorship bias:</b> the universe is today's index constituents applied to 2022. The benchmark shares the bias, which flatters both.</li>
<li><b>One market, one decade-fragment:</b> four years of walk-forward results is a small sample of regimes.</li>
<li><b>Multiple testing:</b> 138 variants were examined. The pre-registered rule limits but does not eliminate selection bias; the holdout was used once for the final choice, but I viewed holdout figures for the other variants too, so post-hoc observations are leads, not findings.</li>
<li><b>Costs:</b> flat 0.1% per side. No slippage model, tax, or market-impact estimate. Small-cap impact is likely larger.</li>
<li><b>Data:</b> known gaps (about 311 missing stock-days, mostly 2024-07-19), unverified corporate-action adjustment, and a stray partial 2026-11-02 bar that was excluded.</li>
<li><b>Cash earns 0%</b> (as set in config.py); a realistic cash yield would help the lower-exposure variants.</li>
<li><b>Daily bars only.</b> Intraday data would matter for execution quality, not for these multi-month signals.</li></ul>""")
    a("<h2>13. Reproduce</h2>")
    a("""<pre>python -m src.battle.data          # rebuild features through 2026-10-30
python -m src.battle.grid          # signal x frequency grid (also caches ML walk-forward scores on first use)
python -m src.battle.ic            # IC decay and rank persistence
python -m src.battle.design        # buffer / weighting / overlay variants
python -m src.battle.robust        # holdout, blocks, rolling windows, bootstrap, placebo, sensitivities
python -m src.battle.procedure_wf  # walk-forward of the selection procedure
python -m src.battle.regime_rl     # regime analysis and RL overlay
python -m src.battle.finalize      # target portfolio and this report</pre>
<p>Outputs are in <code>results/battle/</code>.</p></body></html>""")
    return "\n".join(H)
