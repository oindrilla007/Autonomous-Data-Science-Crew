# ============================================================
# Reporting Agent  Generates multi-format reports
# Runs after Evaluation Agent, final output stage
# ============================================================

from crewai import LLM, Agent, Task
from crewai.tools import tool
from tools.mlflow_tools import setup_mlflow, log_report_artifact
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path
from jinja2 import Template
from datetime import datetime
import pandas as pd
import json
import os

load_dotenv()


#  1. Init LLM 

def get_llm() -> LLM:
    return LLM(
        model=f"groq/{os.getenv('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')}",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", 300)),
    )


#  2. HTML Template 

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>{{ title }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: #f0f4f8;
            color: #2d3748;
            line-height: 1.7;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 48px 40px;
            margin-bottom: 40px;
        }
        header h1 { font-size: 2.2rem; margin-bottom: 8px; }
        header p  { opacity: 0.85; font-size: 1rem; }
        .container { max-width: 1100px; margin: 0 auto; padding: 0 24px 60px; }
        .card {
            background: white;
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 28px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        }
        .card h2 {
            font-size: 1.3rem;
            color: #5a67d8;
            margin-bottom: 18px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ebf4ff;
        }
        .card h3 {
            font-size: 1rem;
            color: #4a5568;
            margin: 16px 0 8px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin: 16px 0;
        }
        .metric-box {
            background: #f7fafc;
            border-radius: 8px;
            padding: 18px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        .metric-box .value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #5a67d8;
        }
        .metric-box .label {
            font-size: 0.78rem;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 0.9rem;
        }
        th {
            background: #5a67d8;
            color: white;
            padding: 10px 14px;
            text-align: left;
            font-weight: 600;
        }
        td { padding: 10px 14px; border-bottom: 1px solid #e2e8f0; }
        tr:hover td { background: #f7fafc; }
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-green  { background: #c6f6d5; color: #276749; }
        .badge-yellow { background: #fefcbf; color: #744210; }
        .badge-red    { background: #fed7d7; color: #822727; }
        .badge-blue   { background: #bee3f8; color: #2a69ac; }
        .narrative {
            background: #f7fafc;
            border-left: 4px solid #667eea;
            padding: 20px 24px;
            border-radius: 0 8px 8px 0;
            margin: 16px 0;
            font-size: 0.95rem;
            color: #4a5568;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 14px;
            margin-top: 12px;
        }
        .chart-card {
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            background: #fff;
        }
        .chart-card img {
            width: 100%;
            height: 220px;
            object-fit: contain;
            background: #f7fafc;
            display: block;
        }
        .chart-caption {
            padding: 8px 10px;
            font-size: 0.82rem;
            color: #4a5568;
            border-top: 1px solid #edf2f7;
        }
        footer {
            text-align: center;
            padding: 32px;
            color: #a0aec0;
            font-size: 0.82rem;
        }
        .section-icon { margin-right: 8px; }
    </style>
</head>
<body>
<header>
    <h1> {{ title }}</h1>
    <p>Generated by Autonomous Data Science Crew &nbsp;|&nbsp; {{ timestamp }}</p>
    <p>Dataset: <strong>{{ dataset_name }}</strong> &nbsp;|&nbsp;
       Task: <strong>{{ task_type }}</strong></p>
</header>

<div class="container">

    <!-- Dataset Overview -->
    <div class="card">
        <h2><span class="section-icon"></span>Dataset Overview</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="value">{{ dataset_rows }}</div>
                <div class="label">Rows</div>
            </div>
            <div class="metric-box">
                <div class="value">{{ dataset_cols }}</div>
                <div class="label">Columns</div>
            </div>
            <div class="metric-box">
                <div class="value">{{ missing_pct }}%</div>
                <div class="label">Missing Values</div>
            </div>
            <div class="metric-box">
                <div class="value">{{ duplicate_rows }}</div>
                <div class="label">Duplicates</div>
            </div>
        </div>
        <div class="narrative" style="white-space: pre-line;">{{ dataset_narrative }}</div>
    </div>

    <!-- EDA Findings -->
    <div class="card">
        <h2><span class="section-icon"></span>Exploratory Data Analysis</h2>
        <div class="narrative" style="white-space: pre-line;">{{ eda_narrative }}</div>
        {% if chart_images %}
        <h3>Generated Charts</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 16px;">
        {% for name, img_data in chart_images.items() %}
            {% if img_data %}
            <div style="background: white; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
                <h4 style="font-size: 0.9rem; margin-bottom: 8px; color: #4a5568;">{{ name }}</h4>
                <img src="data:image/png;base64,{{ img_data }}" style="width: 100%; height: auto; border-radius: 4px;" alt="{{ name }}"/>
            </div>
            {% endif %}
        {% endfor %}
        </div>
        {% endif %}
    </div>

    <!-- Model Performance -->
    <div class="card">
        <h2><span class="section-icon"></span>Model Performance</h2>
        <h3>Best Model: {{ best_model_name }}</h3>
        <div class="metrics-grid">
            {% for key, val in best_metrics.items() %}
            <div class="metric-box">
                <div class="value">{{ val }}</div>
                <div class="label">{{ key }}</div>
            </div>
            {% endfor %}
        </div>

        {% if leaderboard %}
        <h3>Model Leaderboard</h3>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>CV Score</th>
                    <th>Test Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
            {% for i, row in leaderboard %}
                <tr>
                    <td>{{ i + 1 }}</td>
                    <td><strong>{{ row.model }}</strong></td>
                    <td>{{ row.cv_mean }}</td>
                    <td>{{ row.test_score }}</td>
                    <td>
                        {% if i == 0 %}
                        <span class="badge badge-green"> Best</span>
                        {% else %}
                        <span class="badge badge-blue">Evaluated</span>
                        {% endif %}
                    </td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        {% endif %}

        <div class="narrative" style="white-space: pre-line;">{{ modeling_narrative }}</div>
    </div>

    <!-- Overfitting Report -->
    <div class="card">
        <h2><span class="section-icon"></span>Overfitting Assessment</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="value">{{ train_score }}</div>
                <div class="label">Train Score</div>
            </div>
            <div class="metric-box">
                <div class="value">{{ test_score }}</div>
                <div class="label">Test Score</div>
            </div>
            <div class="metric-box">
                <div class="value">{{ gap_pct }}%</div>
                <div class="label">Gap</div>
            </div>
            <div class="metric-box">
                <div class="value">
                    <span class="badge
                        {% if severity == 'none' %}badge-green
                        {% elif severity == 'mild' %}badge-yellow
                        {% else %}badge-red{% endif %}">
                        {{ severity|upper }}
                    </span>
                </div>
                <div class="label">Severity</div>
            </div>
        </div>
        <div class="narrative" style="white-space: pre-line;">{{ overfit_narrative }}</div>
    </div>

    <!-- Key Insights -->
    <div class="card">
        <h2><span class="section-icon"></span>Key Insights & Recommendations</h2>
        <div class="narrative" style="white-space: pre-line;">{{ insights_narrative }}</div>
    </div>

    <!-- Feature Importance -->
    {% if top_features %}
    <div class="card">
        <h2><span class="section-icon"></span>Top Predictive Features</h2>
        <table>
            <thead>
                <tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>
            </thead>
            <tbody>
            {% for i, feat in top_features %}
                <tr>
                    <td>{{ i + 1 }}</td>
                    <td>{{ feat.feature }}</td>
                    <td>{{ feat.importance }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

</div>

<footer>
    <p>Autonomous Data Science Crew &nbsp;|&nbsp; Powered by CrewAI + Groq + MLflow</p>
    <p>Report generated: {{ timestamp }}</p>
</footer>
</body>
</html>
"""


#  3. Reporting Tools 

@tool("generate_html_report")
def generate_html_report(report_data_json: str) -> str:
    """
    Generate a polished HTML report from a JSON data dict.
    The JSON should contain all EDA, modeling, and evaluation
    results collected by prior agents.
    Returns the path to the saved HTML report.
    """
    try:
        data       = json.loads(report_data_json)
        output_dir = Path(os.getenv("REPORTS_DIR", "./reports"))
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path  = str(output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")

        # Extract leaderboard as list of tuples for template
        leaderboard_raw = data.get("leaderboard", [])
        leaderboard     = list(enumerate(
            [type("Row", (), r)() for r in leaderboard_raw]
        ))

        # Extract top features
        top_features_raw = data.get("top_features", [])
        top_features     = list(enumerate(
            [type("Feat", (), f)() for f in top_features_raw]
        ))

        # Convert charts to base64 images for PDF embedding
        import base64
        import re
        import plotly.graph_objects as go
        import plotly.io as pio

        def _figure_from_plotly_html(path: Path):
            try:
                html_text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return None

            match = re.search(
                r"Plotly\.newPlot\(\s*['\"][^'\"]+['\"]\s*,\s*(\[[\s\S]*?\])\s*,\s*({[\s\S]*?})\s*,\s*({[\s\S]*?})\s*\)",
                html_text,
            )
            if not match:
                return None
            try:
                data = json.loads(match.group(1))
                layout = json.loads(match.group(2))
                return go.Figure(data=data, layout=layout)
            except Exception:
                return None

        chart_paths = data.get("chart_paths", {})
        chart_images = {}
        if isinstance(chart_paths, dict):
            for k, v in chart_paths.items():
                if isinstance(v, list):
                    for i, p in enumerate(v):
                        name = f"{k}_{i+1}"
                        if not p:
                            continue
                        path = Path(str(p))
                        try:
                            if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                                chart_images[name] = base64.b64encode(path.read_bytes()).decode()
                            elif path.suffix.lower() == ".html":
                                fig = _figure_from_plotly_html(path)
                                if fig is not None:
                                    img_bytes = pio.to_image(fig, format="png", width=900, height=520)
                                    chart_images[name] = base64.b64encode(img_bytes).decode()
                        except Exception as e:
                            logger.warning(f"Could not convert chart {name}: {e}")
                elif v:
                    name = k
                    path = Path(str(v))
                    try:
                        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                            chart_images[name] = base64.b64encode(path.read_bytes()).decode()
                        elif path.suffix.lower() == ".html":
                            fig = _figure_from_plotly_html(path)
                            if fig is not None:
                                img_bytes = pio.to_image(fig, format="png", width=900, height=520)
                                chart_images[name] = base64.b64encode(img_bytes).decode()
                    except Exception as e:
                        logger.warning(f"Could not convert chart {name}: {e}")

        template = Template(HTML_TEMPLATE)
        html     = template.render(
            title            = data.get("title", "Data Science Report"),
            timestamp        = timestamp,
            dataset_name     = data.get("dataset_name", "Unknown"),
            task_type        = data.get("task_type", "Unknown"),
            dataset_rows     = data.get("dataset_rows", "N/A"),
            dataset_cols     = data.get("dataset_cols", "N/A"),
            missing_pct      = data.get("missing_pct", "N/A"),
            duplicate_rows   = data.get("duplicate_rows", "N/A"),
            dataset_narrative= data.get("dataset_narrative", ""),
            eda_narrative    = data.get("eda_narrative", ""),
            chart_images     = chart_images,
            best_model_name  = data.get("best_model_name", "N/A"),
            best_metrics     = data.get("best_metrics", {}),
            leaderboard      = leaderboard,
            modeling_narrative= data.get("modeling_narrative", ""),
            train_score      = data.get("train_score", "N/A"),
            test_score       = data.get("test_score", "N/A"),
            gap_pct          = data.get("gap_pct", "N/A"),
            severity         = data.get("severity", "none"),
            overfit_narrative= data.get("overfit_narrative", ""),
            insights_narrative= data.get("insights_narrative", ""),
            top_features     = top_features,
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        # Log to MLflow
        setup_mlflow()
        log_report_artifact(report_path, report_type="html")

        logger.success(f"HTML report saved  {report_path}")
        return json.dumps({
            "status":      "success",
            "report_path": report_path,
            "report_type": "html",
        })

    except Exception as e:
        logger.error(f"HTML report generation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("generate_markdown_report")
def generate_markdown_report(report_data_json: str) -> str:
    """
    Generate a Markdown report from a JSON data dict.
    Produces a clean .md file with all findings,
    metrics tables, and narrative summaries.
    Returns the path to the saved Markdown report.
    """
    try:
        data       = json.loads(report_data_json)
        output_dir = Path(os.getenv("REPORTS_DIR", "./reports"))
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = str(output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

        lines = []

        # Header
        lines += [
            f"# {data.get('title', 'Data Science Report')}",
            f"> Generated: {timestamp}  ",
            f"> Dataset: **{data.get('dataset_name', 'Unknown')}**  ",
            f"> Task: **{data.get('task_type', 'Unknown')}**",
            "",
            "---",
            "",
        ]

        # Dataset Overview
        lines += [
            "##  Dataset Overview",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Rows | {data.get('dataset_rows', 'N/A')} |",
            f"| Columns | {data.get('dataset_cols', 'N/A')} |",
            f"| Missing Values | {data.get('missing_pct', 'N/A')}% |",
            f"| Duplicate Rows | {data.get('duplicate_rows', 'N/A')} |",
            "",
            data.get("dataset_narrative", ""),
            "",
        ]

        # EDA
        lines += [
            "##  Exploratory Data Analysis",
            "",
            data.get("eda_narrative", ""),
            "",
        ]

        # Model Performance
        lines += [
            "##  Model Performance",
            "",
            f"**Best Model:** {data.get('best_model_name', 'N/A')}",
            "",
            "| Metric | Score |",
            "|--------|-------|",
        ]
        for k, v in data.get("best_metrics", {}).items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

        # Leaderboard
        leaderboard = data.get("leaderboard", [])
        if leaderboard:
            lines += [
                "### Model Leaderboard",
                "",
                "| Rank | Model | CV Score | Test Score |",
                "|------|-------|----------|------------|",
            ]
            for i, row in enumerate(leaderboard):
                lines.append(
                    f"| {i+1} | {row.get('model','N/A')} | "
                    f"{row.get('cv_mean','N/A')} | "
                    f"{row.get('test_score','N/A')} |"
                )
            lines.append("")

        lines.append(data.get("modeling_narrative", ""))
        lines.append("")

        # Overfitting
        lines += [
            "##  Overfitting Assessment",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Train Score | {data.get('train_score', 'N/A')} |",
            f"| Test Score | {data.get('test_score', 'N/A')} |",
            f"| Gap | {data.get('gap_pct', 'N/A')}% |",
            f"| Severity | **{data.get('severity', 'N/A').upper()}** |",
            "",
            data.get("overfit_narrative", ""),
            "",
        ]

        # Top Features
        top_features = data.get("top_features", [])
        if top_features:
            lines += [
                "##  Top Predictive Features",
                "",
                "| Rank | Feature | Importance |",
                "|------|---------|------------|",
            ]
            for i, feat in enumerate(top_features[:10]):
                lines.append(
                    f"| {i+1} | {feat.get('feature','N/A')} | "
                    f"{feat.get('importance','N/A')} |"
                )
            lines.append("")

        # Key Insights
        lines += [
            "##  Key Insights & Recommendations",
            "",
            data.get("insights_narrative", ""),
            "",
            "---",
            "_Report generated by Autonomous Data Science Crew_",
        ]

        content = "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.success(f"Markdown report saved  {report_path}")
        return json.dumps({
            "status":      "success",
            "report_path": report_path,
            "report_type": "markdown",
        })

    except Exception as e:
        logger.error(f"Markdown report generation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("generate_pdf_report")
def generate_pdf_report(html_report_path: str) -> str:
    """Convert HTML report to PDF using playwright (more reliable on Windows)."""
    try:
        from playwright.sync_api import sync_playwright
        
        html_path = Path(html_report_path)
        if not html_path.exists():
            return json.dumps({
                "status": "error",
                "message": f"HTML file not found: {html_report_path}",
            })

        pdf_path = str(html_path.with_suffix(".pdf"))
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file:///{html_path.absolute()}")
            page.pdf(path=pdf_path, format="A4", print_background=True)
            browser.close()

        # Log to MLflow
        setup_mlflow()
        log_report_artifact(pdf_path, report_type="pdf")

        logger.success(f"PDF report saved -> {pdf_path}")
        return json.dumps({
            "status": "success",
            "report_path": pdf_path,
            "report_type": "pdf",
        })
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})

@tool("compile_report_data")
def compile_report_data(
    dataset_filepath: str,
    target_col: str,
    eda_summary_json: str  = "{}",
    automl_results_json: str = "{}",
    evaluation_json: str  = "{}",
    overfit_json: str     = "{}",
    chart_paths_json: str = "{}",
) -> str:
    """
    Compile all agent outputs into unified report with REAL insights.
    """
    try:
        from tools.eda_tools import load_dataset, basic_summary

        df      = load_dataset(dataset_filepath)
        summary = basic_summary(df)
        shape   = summary.get("shape", {})

        # Load saved results from disk
        automl_path = Path("./models/automl_results.json")
        eda_path    = Path("./models/eda_results.json")
        eval_path   = Path("./models/evaluation_results.json")
        
        if automl_path.exists():
            with open(automl_path) as f:
                automl_data = json.load(f)
        else:
            automl_data = json.loads(automl_results_json)
        
        if eda_path.exists():
            with open(eda_path) as f:
                eda_data = json.load(f)
        else:
            eda_data = json.loads(eda_summary_json)
        
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)
        else:
            eval_data = json.loads(evaluation_json)
        
        overfit_data = json.loads(overfit_json) if overfit_json != "{}" else {}
        chart_paths  = json.loads(chart_paths_json)

        # Extract metrics
        best_metrics = (
            automl_data.get("best_metrics") or
            eval_data.get("metrics") or {}
        )

        # Build leaderboard
        leaderboard = []
        all_scores  = automl_data.get("all_model_scores", {})
        task_type   = automl_data.get("task_type", "unknown")
        primary     = "accuracy" if task_type == "classification" else "r2"

        for name, metrics in all_scores.items():
            if "error" not in metrics:
                leaderboard.append({
                    "model":      name,
                    "cv_mean":    metrics.get("cv_mean", "N/A"),
                    "test_score": metrics.get(primary, "N/A"),
                })
        
        leaderboard_sorted = sorted(
            leaderboard,
            key=lambda x: float(x["test_score"])
                if isinstance(x["test_score"], (int, float)) else 0,
            reverse=True,
        )

        # Extract correlation insights
        correlations = eda_data.get("correlation_analysis", {})
        top_corr_pairs = correlations.get("top_pairs", [])[:3]
        high_corr = correlations.get("high_corr_pairs", [])
        
        # Extract numeric stats for dataset explanation
        numeric_stats = eda_data.get("numeric_statistics", {})
        num_cols = summary.get("numeric_columns", [])
        
        # Extract target insights
        target_info = eda_data.get("target_analysis", {})
        
        # Extract features
        top_features = automl_data.get("top_features", [])

        # BUILD NARRATIVES WITH BETTER STRUCTURE
        
        # 1. DATASET EXPLANATION (What is this data about?)
        dataset_explanation = f"""
**What is this dataset?**

This dataset contains **{shape.get('rows', 'N/A')} observations** across **{shape.get('columns', 'N/A')} variables**. 
It appears to be a **{task_type}** problem focused on predicting the variable: **{target_col}**.

**Data Composition:**
- **Numeric features:** {len(num_cols)} ({', '.join(num_cols[:4])}{'...' if len(num_cols) > 4 else ''})
- **Categorical features:** {len(summary.get('categorical_columns', []))}
- **Missing values:** {sum(summary.get('null_counts', {}).values())} total ({round(sum(summary.get('null_percentages', {}).values()), 1)}%)
- **Duplicate rows:** {summary.get('duplicate_rows', 0)}
"""

        if target_info.get("task_type") == "classification":
            class_dist = target_info.get("class_distribution", {})
            if class_dist:
                dataset_explanation += f"\n**Target Distribution:**\n"
                for cls, count in class_dist.items():
                    pct = round(count / shape.get('rows', 1) * 100, 1)
                    dataset_explanation += f"- Class {cls}: {count} samples ({pct}%)\n"
                
                if target_info.get("is_imbalanced"):
                    dataset_explanation += f"\n⚠️ **Imbalance detected** (ratio: {target_info.get('class_balance_ratio')})"

        # 2. EDA INSIGHTS (What patterns were found?)
        eda_insights = "**Key Statistical Findings:**\n\n"
        
        if top_corr_pairs:
            eda_insights += "**Strong Correlations Detected:**\n"
            for p in top_corr_pairs:
                eda_insights += f"- **{p['col_a']}** ↔ **{p['col_b']}**: r={p['correlation']}\n"
        
        if high_corr:
            eda_insights += f"\n⚠️ **Multicollinearity Warning:** {len(high_corr)} feature pairs show r>0.8, which may reduce model interpretability.\n"
        
        if top_features:
            eda_insights += f"\n**Most Predictive Features:**\n"
            for i, feat in enumerate(top_features[:5], 1):
                eda_insights += f"{i}. **{feat.get('feature', 'N/A')}** (importance: {feat.get('importance', 0):.3f})\n"

        # 3. MODEL PERFORMANCE (How well did models perform?)
        model_narrative = f"""
**AutoML Results:**

Trained and evaluated **{len(all_scores)} machine learning models** using 5-fold cross-validation.

**🏆 Winner: {automl_data.get('best_model_name', 'N/A')}**
- **{primary.upper()}:** {best_metrics.get(primary, 'N/A')}
- **Cross-Validation Score:** {best_metrics.get('cv_mean', 'N/A')} (±{best_metrics.get('cv_std', 'N/A')})
"""

        if len(leaderboard_sorted) >= 3:
            model_narrative += f"\n**Top 3 Models:**\n"
            for i, m in enumerate(leaderboard_sorted[:3], 1):
                model_narrative += f"{i}. {m['model']}: {m['test_score']}\n"

        # 4. OVERFITTING ASSESSMENT
        overfit_narrative = ""
        if overfit_data:
            gap = overfit_data.get("gap_pct", 0)
            severity = overfit_data.get("severity", "none")
            
            if severity == "none":
                overfit_narrative = f"✅ **No overfitting detected.** Train-test gap: {gap}%. Model generalizes well."
            elif severity == "mild":
                overfit_narrative = f"⚠️ **Mild overfitting.** Train-test gap: {gap}%. Consider light regularization."
            else:
                overfit_narrative = f"❌ **{severity.capitalize()} overfitting.** Train-test gap: {gap}%. {overfit_data.get('recommendation', '')}"

        # 5. ACTIONABLE RECOMMENDATIONS
        recommendations = []
        
        if target_info.get("is_imbalanced"):
            recommendations.append("**Address class imbalance:** Use SMOTE, class weights, or stratified sampling")
        
        if len(high_corr) > 2:
            recommendations.append("**Reduce multicollinearity:** Drop redundant features or use PCA")
        
        if overfit_data.get("severity") in ["moderate", "severe"]:
            recommendations.append("**Fix overfitting:** Add regularization, reduce model complexity, or collect more data")
        
        if task_type == "classification" and best_metrics.get(primary, 0) < 0.85:
            recommendations.append("**Improve accuracy:** Try feature engineering, ensemble methods, or hyperparameter tuning")
        
        recommendations.append("**Next steps:** Deploy model, monitor performance, and retrain with new data")

        insights_narrative = "**Recommendations:**\n\n" + "\n".join(f"{i+1}. {r}" for i, r in enumerate(recommendations))

        # FINAL REPORT DATA
        report_data = {
            "title":             f"Autonomous DS Crew — {Path(dataset_filepath).stem.title()} Analysis",
            "dataset_name":      Path(dataset_filepath).stem,
            "task_type":         task_type.capitalize(),
            "dataset_rows":      shape.get("rows", "N/A"),
            "dataset_cols":      shape.get("columns", "N/A"),
            "missing_pct":       round(
                sum(summary.get("null_percentages", {}).values()) /
                max(len(summary.get("null_percentages", {})), 1), 2
            ),
            "duplicate_rows":    summary.get("duplicate_rows", 0),
            "best_model_name":   automl_data.get("best_model_name", "N/A"),
            "best_metrics":      best_metrics,
            "leaderboard":       leaderboard_sorted,
            "top_features":      top_features,
            "train_score":       overfit_data.get("train_score", eval_data.get("train_score", "N/A")),
            "test_score":        overfit_data.get("test_score", eval_data.get("test_score", "N/A")),
            "gap_pct":           overfit_data.get("gap_pct", "N/A"),
            "severity":          overfit_data.get("severity", "none"),
            "chart_paths":       chart_paths,
            
            # UPGRADED NARRATIVES
            "dataset_narrative":  dataset_explanation.strip(),
            "eda_narrative":      eda_insights.strip(),
            "modeling_narrative": model_narrative.strip(),
            "overfit_narrative":  overfit_narrative.strip(),
            "insights_narrative": insights_narrative.strip(),
        }

        logger.success("Report data compiled with structured insights")
        return json.dumps(report_data, default=str)

    except Exception as e:
        logger.error(f"Report data compilation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


#  4. Build Reporting Agent 

def build_reporting_agent(llm: LLM = None) -> Agent:
    """Build the CrewAI Reporting Agent with all tools."""
    if llm is None:
        llm = get_llm()

    agent = Agent(
        role="Data Science Report Writer & Communicator",
        goal=(
            "Transform raw analysis results into polished, professional "
            "multi-format reports (HTML, Markdown, PDF) that communicate "
            "findings clearly to both technical and non-technical audiences. "
            "Every report must tell a coherent story with actionable insights."
        ),
        backstory=(
            "You are a senior data science communicator who bridges the gap "
            "between complex ML results and business stakeholders. You have "
            "written hundreds of reports that turned confusing model outputs "
            "into clear, actionable narratives. You believe a great report "
            "is as important as a great model."
        ),
        tools=[
            compile_report_data,
            generate_html_report,
            generate_markdown_report,
            generate_pdf_report,
        ],
        llm=llm,
        verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        allow_delegation=False,
        max_iter=int(os.getenv("MAX_ITERATIONS", 1)),
    )

    logger.info("Reporting Agent built")
    return agent


#  5. Reporting Task 

def build_reporting_task(
    agent: Agent,
    dataset_filepath: str,
    target_col: str,
    context_tasks: list = None,
) -> Task:
    """Build a CrewAI Task for the Reporting Agent."""
    return Task(
        description=f"""
        Generate comprehensive multi-format reports for: {dataset_filepath}
        Target column: {target_col}

        Complete ALL steps in order:

        1. Compile all prior agent outputs into unified report data
           using compile_report_data  pass in results from the
           EDA, Modeling, and Evaluation agents

        2. Generate the HTML report using generate_html_report
           with the compiled report data JSON

        3. Generate the Markdown report using generate_markdown_report
           with the same compiled report data JSON

        4. Convert the HTML report to PDF using generate_pdf_report
           passing the HTML report path from step 2

        Write compelling narratives for each section that explain
        the findings in plain language. Make every insight actionable.
        Ensure all three report formats are saved and paths confirmed.
        """,
        expected_output=(
            "Confirmation that three report formats were successfully "
            "generated and saved: HTML report path, Markdown report path, "
            "and PDF report path. Plus a brief executive summary of the "
            "key findings written in plain business language."
        ),
        agent=agent,
        context=context_tasks or [],
    )


#  6. Standalone Test 

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from pathlib import Path

    logger.info("Testing Reporting Agent standalone...")
    Path("./reports").mkdir(exist_ok=True)
    Path("./data").mkdir(exist_ok=True)

    # Create sample data
    iris = load_iris(as_frame=True)
    iris.frame.to_csv("./data/sample_iris.csv", index=False)

    # Sample report data
    sample_data = json.dumps({
        "title":              "Test Report  Iris Dataset",
        "dataset_name":       "sample_iris",
        "task_type":          "classification",
        "dataset_rows":       150,
        "dataset_cols":       5,
        "missing_pct":        0.0,
        "duplicate_rows":     0,
        "best_model_name":    "RandomForest",
        "best_metrics":       {"accuracy": 0.9667, "f1_macro": 0.9667},
        "leaderboard":        [
            {"model": "RandomForest", "cv_mean": 0.96, "test_score": 0.9667},
            {"model": "XGBoost",      "cv_mean": 0.95, "test_score": 0.9600},
        ],
        "top_features":       [
            {"feature": "petal length (cm)", "importance": 0.45},
            {"feature": "petal width (cm)",  "importance": 0.38},
        ],
        "train_score":        0.99,
        "test_score":         0.9667,
        "gap_pct":            2.3,
        "severity":           "none",
        "chart_paths":        {},
        "dataset_narrative":  "The Iris dataset is a classic classification dataset with 150 samples.",
        "eda_narrative":      "Features show strong correlation with target classes.",
        "modeling_narrative": "RandomForest achieved best performance across all metrics.",
        "overfit_narrative":  "Model generalises well with minimal gap.",
        "insights_narrative": "Petal measurements are the strongest predictors of iris species.",
    })

    # Test HTML report
    html_result = generate_html_report.run(report_data_json=sample_data)
    html_data   = json.loads(html_result)
    logger.info(f"HTML report: {html_data.get('report_path')}")

    # Test Markdown report
    md_result = generate_markdown_report.run(report_data_json=sample_data)
    logger.info(f"Markdown report: {json.loads(md_result).get('report_path')}")

    logger.success("Reporting Agent test complete")


