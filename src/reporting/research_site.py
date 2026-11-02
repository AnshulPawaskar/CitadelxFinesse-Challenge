"""Builds reports/portfolio_challenge_deck.html: a self-contained 1920x1080 light-theme presentation (HTML + CSS + JS, no CDN).
All numbers are read from results/ CSVs; the competition-window curve is recomputed with the same engine."""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.battle.context import build_context
from src.battle.engine import metrics, run_backtest

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "results" / "battle"
OUT = ROOT / "reports" / "portfolio_challenge_deck.html"
rd = lambda n: pd.read_csv(B / n)


def competition_window():
    ctx = build_context()
    m = ctx.mkt
    a = int(m.dates.searchsorted(pd.Timestamp("2026-01-01")))
    b = int(m.dates.searchsorted(pd.Timestamp("2026-06-30"), side="right")) - 1
    r = run_backtest(m, ctx.signal("core3", "S"), "S", a, b)
    bench = np.cumprod(1 + m.bench_ret[a: b + 1])
    mt = metrics(r.returns, r.dates)
    return {"dates": [d.strftime("%Y-%m-%d") for d in r.dates], "strategy": [round(float(x), 5) for x in r.nav],
            "bench": [round(float(x), 5) for x in bench], "ret": float(r.nav[-1] - 1), "bret": float(bench[-1] - 1),
            "sharpe": float(mt["Sharpe"]), "maxdd": float(mt["MaxDD"])}


def build_data():
    d = {}
    d["h1"] = competition_window()
    c = pd.read_csv(ROOT / "results" / "oos_2026H1_contribution.csv").sort_values("PnL_Rs", ascending=False)
    d["contrib"] = [{"s": r.Symbol, "v": round(r.PnL_Rs / 1e5, 2)} for r in c.itertuples()]
    fr = rd("final_daily_returns.csv"); fr["Datetime"] = pd.to_datetime(fr.Datetime); fr = fr.set_index("Datetime")
    nav = (1 + fr).cumprod().resample("W-FRI").last().dropna()
    d["cum"] = {"dates": [x.strftime("%Y-%m-%d") for x in nav.index], "strategy": [round(float(x), 4) for x in nav.Strategy], "bench": [round(float(x), 4) for x in nav.Benchmark]}
    y = rd("calendar_years.csv"); y["Year"] = pd.to_datetime(y.BlockEnd).dt.year
    hs = fr.loc["2024-07-01":]
    d["holdb"] = float((1 + hs.Benchmark).prod() - 1)
    d["holds"] = float((1 + hs.Strategy).prod() - 1)
    d["years"] = [{"y": ("2026 YTD" if r.Year == 2026 else str(r.Year)), "s": round(r.Strategy * 100, 1), "b": round(r.Benchmark * 100, 1)} for r in y.itertuples()]
    g = rd("grid_signal_frequency.csv"); g0 = g[g.Buffer == 0]
    f = g0.groupby("Freq").agg(v=("Val_Sharpe", "mean"), h=("Hold_Sharpe", "mean"), t=("Val_Turnover", "mean")).reindex(["D", "W", "M", "Q", "S"])
    d["freq"] = [{"f": n, "v": round(r.v, 2), "h": round(r.h, 2), "t": round(r.t, 1)} for n, r in zip(["Daily", "Weekly", "Monthly", "Quarterly", "Half-yearly"], f.itertuples())]
    d["scatter"] = [[round(a, 3), round(b, 3)] for a, b in zip(g.Val_Sharpe, g.Hold_Sharpe)]
    ic = rd("signal_ic_decay.csv"); v = ic[ic.Window == "Val"].set_index("Signal")
    cols = ["IC_1d", "IC_5d", "IC_21d", "IC_63d", "IC_126d"]
    names = {"composite_quantitative": "Hand-set composite", "momentum_12_1": "Momentum 12-1", "ridge": "Ridge (ML)", "lgbm": "LightGBM (ML)", "mlp": "Neural net (DL)"}
    d["ic"] = [{"n": nm, "v": [round(float(v.loc[k, c]), 4) for c in cols]} for k, nm in names.items() if k in v.index]
    pl = rd("placebo_random.csv"); st = rd("statistical_tests.csv").iloc[0]
    S = rd("strategy_summary.csv"); g_ = lambda s, w: S[(S.Strategy == s) & (S.Window == w)].iloc[0]
    F, Bn = "FINAL core3 | Half-yearly", "Benchmark (equal-weight universe, daily rebal)"
    d["placebo"] = {"sharpe": [round(x, 3) for x in pl.Sharpe], "strat": float(g_(F, "Full").Sharpe), "pct": float(st.PlaceboSharpePctile), "dsr": float(st.DSR), "trials": int(st.N_trials)}
    d["win"] = {k: {"cagr": float(g_(F, w).CAGR), "sharpe": float(g_(F, w).Sharpe), "dd": float(g_(F, w).MaxDD), "tot": float(g_(F, w).TotalReturn),
                    "bcagr": float(g_(Bn, w).CAGR), "bsharpe": float(g_(Bn, w).Sharpe)} for k, w in (("val", "Val"), ("hold", "Hold"), ("full", "Full"))}
    po = rd("portfolio_opt_summary.csv"); q = po[(po.Freq == "Half-yearly") & (po.Window.isin(["Val", "Hold"]))]
    nm = {"equal": "Equal weight", "invvol": "Inverse-vol", "minvar_lw": "Min-variance", "risk_parity_lw": "Risk parity", "meanvar_lw": "Mean-variance", "capm_meanvar_lw": "CAPM"}
    d["weights"] = [{"n": nm[k], "vs": round(float(q[(q.Scheme == k) & (q.Window == "Val")].Sharpe.iloc[0]), 2), "hs": round(float(q[(q.Scheme == k) & (q.Window == "Hold")].Sharpe.iloc[0]), 2),
                     "vd": round(float(q[(q.Scheme == k) & (q.Window == "Val")].MaxDD.iloc[0]) * 100, 1)} for k in nm]
    gm = g0.groupby("Signal").agg(v=("Val_Sharpe", "mean"), dd=("Val_MaxDD", "mean")).sort_values("v", ascending=False)
    info = {"momentum_6m": ("Momentum 6m", "Factor"), "risk_adjusted_momentum": ("Risk-adjusted momentum", "Factor"), "mlp": ("Neural net (MLP)", "DL"),
            "momentum_12_1": ("Momentum 12-1", "Factor"), "momentum_lowvol": ("Momentum + low vol", "Factor"), "ml_ensemble": ("ML ensemble (3 models)", "ML + DL"),
            "ridge": ("Ridge regression", "ML"), "hybrid": ("Hybrid (factor + ML)", "Blend"), "relative_strength": ("Relative strength", "Factor"),
            "lgbm": ("LightGBM", "ML"), "technical_composite": ("Technical composite", "Factor"), "composite_quantitative": ("Hand-set composite", "Factor"),
            "trend_quality": ("Trend quality", "Factor")}
    top3 = list(gm.index[:3])
    d["sig"] = [{"k": k, "n": info[k][0], "t": info[k][1], "v": round(r.v, 2), "dd": round(r.dd * 100, 1), "ic": round(float(v.loc[k, "IC_21d"]), 3), "pick": k in top3} for k, r in gm.iterrows()]
    rl = rd("rl_overlay.csv")
    d["rl"] = [{"n": r.Strategy, "sh": round(r.Sharpe, 2), "cagr": round(r.CAGR * 100, 1), "dd": round(r.MaxDD * 100, 1), "ex": (None if pd.isna(r.AvgExposure) else round(r.AvgExposure * 100))} for r in rl.itertuples() if r.Strategy != "Benchmark"]
    ro = rd("rolling_summary.csv").set_index("Months"); bo = rd("bootstrap.csv").iloc[0]
    d["stats"] = {"roll6": float(ro.loc[6, "BeatBenchmark"]), "roll12": float(ro.loc[12, "BeatBenchmark"]), "boot": float(bo.ProbExcessPositive), "ci": [float(bo.CI5), float(bo.CI95)], "excess": float(bo.MeanAnnExcess)}
    fcs = {"Monthly": "core3 | Monthly", "Quarterly": "core3 | Quarterly", "Half-yearly": F}
    d["fc"] = [{"f": k, "vs": float(g_(v, "Val").Sharpe), "vd": float(g_(v, "Val").MaxDD), "hs": float(g_(v, "Hold").Sharpe), "hr": float(g_(v, "Hold").TotalReturn),
                "fc": float(g_(v, "Full").CAGR), "t": float(g_(v, "Val").Turnover)} for k, v in fcs.items()]
    se = rd("sensitivity.csv")
    d["sens"] = [{"t": r.Test, "v": r.Value, "sh": float(r.Sharpe), "dd": float(r.MaxDD), "cagr": float(r.CAGR)} for r in se.itertuples()]
    d["rank"] = pd.read_csv(ROOT / "results" / "oos_2026H1_contribution.csv").to_dict("records")
    cm = pd.read_csv(B / "signal_correlation.csv", index_col=0)
    d["corr"] = {"names": list(cm.columns), "v": cm.round(2).values.tolist()}
    return d



TEMPLATE = ROOT / "src" / "reporting" / "deck_template.html"
ASSETS = ROOT / "reports" / "assets"
LABNAME = 'Organised by Finesse, SJMSOM IIT Bombay<small>In collaboration with IITB Citadel Securities Quantitative Research Lab</small>'
SOURCES = {}


def logo_tag():
    # Drop the official lab logo at reports/assets/lab_logo.png (or .svg/.jpg) to show it next to the IIT Bombay mark.
    tags = []
    for name in ("finesse_logo.png", "sjmsom_iitb_logo.png", "lab_logo.png", "lab_logo.svg", "lab_logo.jpg"):
        p = ASSETS / name
        if p.exists():
            mime = {"png": "image/png", "svg": "image/svg+xml", "jpg": "image/jpeg"}[p.suffix[1:]]
            tags.append(f'<img alt="{p.stem}" src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}">')
    return "".join(tags)


def tables(d):
    tcls = {"Factor": "t-info", "ML": "t-ok", "DL": "t-pur", "ML + DL": "t-ok", "Blend": "t-grey"}
    rows = "".join(f'<tr{" class=hl" if x["pick"] else ""}><td>{i + 1}</td><td>{x["n"]}</td><td><span class="tag {tcls[x["t"]]}">{x["t"]}</span></td><td>{x["v"]:.2f}</td><td>{x["ic"]:.3f}</td><td>{x["dd"]:.0f}%</td><td>{"Yes" if x["pick"] else ""}</td></tr>' for i, x in enumerate(d["sig"]))
    sig = '<table class="t2"><tr><th>#</th><th>Signal</th><th>Type</th><th>Val. Sharpe</th><th>Rank IC 21d</th><th>Max DD</th><th>Pick</th></tr>' + rows + "</table>"
    by = {x["k"]: x for x in d["sig"]}
    ml = [("ridge", "Ridge regression", "ML", "Not selected"), ("lgbm", "LightGBM", "ML", "Not selected"), ("mlp", "Neural net (MLP)", "DL", "In final blend"), ("ml_ensemble", "ML ensemble (3 models)", "ML + DL", "Not selected")]
    r2 = "".join(f'<tr{" class=hl" if k == "mlp" else ""}><td>{n}</td><td><span class="tag {tcls[t]}">{t}</span></td><td>{by[k]["v"]:.2f}</td><td>{by[k]["ic"]:.3f}</td><td>{v}</td></tr>' for k, n, t, v in ml)
    r2 += f'<tr><td>Q-learning exposure overlay</td><td><span class="tag t-warn">RL</span></td><td>-</td><td>-</td><td>Rejected</td></tr>'
    mlt = '<table class="t2 t3"><tr><th>Model</th><th>Type</th><th>Val. Sharpe</th><th>Rank IC 21d</th><th>Decision</th></tr>' + r2 + "</table>"
    rl = "".join(f'<tr><td>{x["n"].replace("Core3 Q", "Strategy")}</td><td>{x["cagr"]:.1f}%</td><td>{x["dd"]:.1f}%</td><td>{"-" if x["ex"] is None else str(x["ex"]) + "%"}</td></tr>' for x in d["rl"])
    rlt = '<table class="t2 t3"><tr><th>Feb 2023 onward</th><th>CAGR</th><th>Max DD</th><th>Exposure</th></tr>' + rl + "</table>"
    pc = lambda x, n=1: f"{x * 100:.{n}f}%"
    fr = "".join(f'<tr{" class=hl" if x["f"] == "Half-yearly" else ""}><td>{x["f"]}</td><td>{x["vs"]:.2f}</td><td>{pc(x["vd"], 0)}</td><td>{x["hs"]:.2f}</td><td>{pc(x["hr"], 0)}</td><td>{pc(x["fc"], 0)}</td><td>{x["t"]:.1f}x</td></tr>' for x in d["fc"])
    fct = '<table class="t2 t3"><tr><th>Rebalance</th><th>Val. Sharpe</th><th>Val. Max DD</th><th>Holdout Sharpe</th><th>Holdout return</th><th>CAGR 2022-26</th><th>Turnover / year</th></tr>' + fr + "</table>"
    rows = "".join(f'<tr><td>{x["t"]}</td><td>{x["v"]}</td><td>{x["sh"]:.2f}</td><td>{pc(x["dd"], 0)}</td><td>{pc(x["cagr"], 0)}</td></tr>' for x in d["sens"] if x["t"] in ("Cost per side", "Stale signal (days)", "Holdings"))
    stt = '<table class="t2"><tr><th>Test</th><th>Value</th><th>Sharpe</th><th>Max DD</th><th>CAGR</th></tr>' + rows + "</table>"
    rk = "".join(f'<tr><td>{x["RankFinal"]}</td><td>{x["Symbol"]}</td><td>{x["RankMomentum6m"]}</td><td>{x["RankRiskAdjMomentum"]}</td><td>{x["RankMLP"]}</td><td>10%</td><td class="{"good" if x["PnL_Rs"] >= 0 else "bad"}">{x["PnL_Rs"] / 1e5:+.2f}</td></tr>' for x in d["rank"])
    rkt = '<table class="t2 t3"><tr><th>Final rank</th><th>Stock</th><th>Momentum 6m rank</th><th>Risk-adj. momentum rank</th><th>MLP rank</th><th>Weight</th><th>P&amp;L (Rs lakh)</th></tr>' + rk + "</table>"
    nm = {"momentum_6m": "Mom. 6m", "risk_adjusted_momentum": "Risk-adj. mom.", "mlp": "MLP", "momentum_12_1": "Mom. 12-1", "trend_quality": "Trend quality", "ridge": "Ridge", "lgbm": "LightGBM"}
    c = d["corr"]
    def cell(v):
        a = min(1, max(0, abs(v)))
        return f'<td style="background:rgba(37,99,235,{a * .55:.2f});text-align:center;font-weight:{700 if a > .7 else 500}">{v:.2f}</td>'
    ct = '<table class="t2 cm"><tr><th></th>' + "".join(f"<th>{nm[k]}</th>" for k in c["names"]) + "</tr>" + "".join(
        f'<tr><td><b>{nm[k]}</b></td>' + "".join(cell(v) for v in row) + "</tr>" for k, row in zip(c["names"], c["v"])) + "</table>"
    return sig, mlt, rlt, fct, stt, rkt, ct


def main():
    d = build_data()
    h1, w = d["h1"], d["win"]
    html = TEMPLATE.read_text(encoding="utf-8")
    sg, ml, rt, fct, stt, rkt, ct = tables(d)
    html = html.replace("__SIGTABLE__", sg).replace("__MLTABLE__", ml).replace("__RLTABLE__", rt).replace("__FCTABLE__", fct).replace("__STABTABLE__", stt).replace("__RANKTABLE__", rkt).replace("__CORRTABLE__", ct)
    n = html.count('<section class="slide')
    logo_html = logo_tag()
    logo = '<span class="lgs" style="display:flex;gap:16px;align-items:center"></span>'
    hdr = f'<div class="hdr"><div class="org">{logo}<div class="sep"></div><div class="nm">{LABNAME}</div></div><div class="team"><i></i>Team Quantify</div></div>'
    html = html.replace("__HDR__", hdr).replace("__ORGLOGO__", logo).replace("__LABNAME__", LABNAME)
    for i in range(2, n + 1):
        src = SOURCES.get(i - 1, "")
        ref = ""
        ftr = f'<div class="ftr"><span><b>Team Quantify</b> &nbsp;|&nbsp; CitadelxFinesse Portfolio Challenge{ref}</span><span class="pg">{i} / {n}</span></div>'
        html = html.replace(f"__FTR{i}__", ftr)
    rep = {
        "__DATA__": json.dumps(d), "__LOGOJS__": json.dumps(logo_html), "__N__": str(n),
        "__H1RET__": f"{h1['ret'] * 100:.2f}", "__H1B__": f"{h1['bret'] * 100:.2f}", "__MULT__": f"{h1['ret'] / h1['bret']:.1f}",
        "__H1RET_T__": f"+{h1['ret'] * 100:.2f}%", "__H1B_T__": f"+{h1['bret'] * 100:.2f}%",
        "__NET__": f"{h1['ret'] * 1e7:,.0f}", "__H1SH__": f"{h1['sharpe']:.2f}", "__H1DD__": f"{h1['maxdd'] * 100:.1f}%",
        "__BOOT__": f"{d['stats']['boot'] * 100:.0f}", "__R6__": f"{d['stats']['roll6'] * 100:.0f}", "__PCT__": f"{d['placebo']['pct'] * 100:.0f}",
        "__DSR__": f"{d['placebo']['dsr']:.2f}", "__TRIALS__": str(d["placebo"]["trials"]),
        "__HOLD__": f"+{d['holds'] * 100:.1f}", "__HOLDB__": f"+{d['holdb'] * 100:.1f}",
        "__TURNX__": f"{d['freq'][0]['t'] / d['freq'][4]['t']:.0f}", "__Y25__": f"{d['years'][3]['s']:.1f}", "__Y25B__": f"{d['years'][3]['b']:.1f}",
        "__FCAGR__": f"{w['full']['cagr'] * 100:.1f}", "__FB__": f"{w['full']['bcagr'] * 100:.1f}",
    }
    for k, v in rep.items():
        html = html.replace(k, v)
    OUT.write_text(html, encoding="utf-8")
    print("Wrote", OUT, f"({len(html) / 1e3:.0f} KB)", "slides", n, rep["__H1RET_T__"], rep["__H1B_T__"], "hold", rep["__HOLD__"], rep["__HOLDB__"])


if __name__ == "__main__":
    main()
