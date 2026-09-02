"""Render performance_report.html embedding the generated charts and key metrics tables.

Note: PDF export isn't generated here — it would require an extra heavyweight dependency
(e.g. weasyprint/wkhtmltopdf with system-level packages) that isn't installed in this venv.
Say the word if you want that added; the HTML report below can also just be printed to PDF
from a browser in the meantime.
"""
from src.config import REPORTS_DIR, PLOTS_DIR


def _metrics_table(metrics: dict) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items())
    return f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"


def build_html_report(metrics: dict, oos_metrics: dict, chart_paths: dict, output_path=None) -> str:
    output_path = output_path or REPORTS_DIR / "performance_report.html"

    charts_html = "".join(
        f'<h2>{name.replace("_", " ").title()}</h2>'
        f'<img src="../plots/{path.name}" style="max-width:900px;width:100%;">'
        for name, path in chart_paths.items()
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Static Portfolio Selection - Performance Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
table {{ border-collapse: collapse; margin-bottom: 24px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
th {{ background: #f0f0f0; }}
h1, h2 {{ color: #1a1a2e; }}
</style>
</head>
<body>
<h1>Static Portfolio Selection - Performance Report</h1>

<h2>Backtest Performance (Formation -> 2025-12-31)</h2>
{_metrics_table(metrics)}

<h2>Out-of-Sample Performance (2026 H1)</h2>
{_metrics_table(oos_metrics)}

{charts_html}
</body>
</html>
"""
    output_path.write_text(html)
    return str(output_path)
