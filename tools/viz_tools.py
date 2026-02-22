# ============================================================
# Viz Tools — Narrative chart generation
# Used by the EDA Agent & Reporting Agent
# ============================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger


# ── Output Directory ─────────────────────────────────────────

def _ensure_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── 1. Distribution Plots ────────────────────────────────────

def plot_distributions(df: pd.DataFrame, output_dir: str = "./reports/charts") -> list:
    """Histogram + KDE for every numeric column. Returns list of saved paths."""
    saved = []
    out = _ensure_dir(output_dir)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in num_cols:
        fig = px.histogram(
            df, x=col,
            marginal="kde",
            title=f"Distribution of {col}",
            template="plotly_white",
            color_discrete_sequence=["#6366f1"],
        )
        fig.update_layout(
            title_font_size=16,
            xaxis_title=col,
            yaxis_title="Count",
            bargap=0.05,
        )
        path = str(out / f"dist_{col}.html")
        fig.write_html(path)
        saved.append(path)
        logger.info(f"Distribution chart saved → {path}")

    return saved


# ── 2. Correlation Heatmap ───────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = "./reports/charts") -> str:
    """Interactive correlation heatmap for numeric columns."""
    out = _ensure_dir(output_dir)
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Heatmap",
        template="plotly_white",
    )
    fig.update_layout(title_font_size=16, width=800, height=700)

    path = str(out / "correlation_heatmap.html")
    fig.write_html(path)
    logger.info(f"Correlation heatmap saved → {path}")
    return path


# ── 3. Missing Values Chart ──────────────────────────────────

def plot_missing_values(df: pd.DataFrame, output_dir: str = "./reports/charts") -> str:
    """Bar chart of missing value percentages per column."""
    out = _ensure_dir(output_dir)

    missing = (df.isnull().mean() * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        logger.info("No missing values found — skipping chart")
        return None

    fig = px.bar(
        x=missing.index,
        y=missing.values,
        title="Missing Values by Column (%)",
        labels={"x": "Column", "y": "Missing (%)"},
        template="plotly_white",
        color=missing.values,
        color_continuous_scale="Reds",
    )
    fig.update_layout(title_font_size=16, showlegend=False)

    path = str(out / "missing_values.html")
    fig.write_html(path)
    logger.info(f"Missing values chart saved → {path}")
    return path


# ── 4. Target Distribution ───────────────────────────────────

def plot_target_distribution(df: pd.DataFrame, target_col: str,
                              output_dir: str = "./reports/charts") -> str:
    """Bar chart (classification) or histogram (regression) for target column."""
    out = _ensure_dir(output_dir)

    unique_vals = df[target_col].nunique()
    is_classification = unique_vals <= 20 or not pd.api.types.is_numeric_dtype(df[target_col])

    if is_classification:
        counts = df[target_col].value_counts().reset_index()
        counts.columns = [target_col, "count"]
        fig = px.bar(
            counts, x=target_col, y="count",
            title=f"Target Distribution — {target_col}",
            template="plotly_white",
            color="count",
            color_continuous_scale="Blues",
        )
    else:
        fig = px.histogram(
            df, x=target_col,
            marginal="box",
            title=f"Target Distribution — {target_col}",
            template="plotly_white",
            color_discrete_sequence=["#10b981"],
        )

    fig.update_layout(title_font_size=16)
    path = str(out / f"target_{target_col}.html")
    fig.write_html(path)
    logger.info(f"Target distribution chart saved → {path}")
    return path


# ── 5. Outlier Box Plots ─────────────────────────────────────

def plot_outliers(df: pd.DataFrame, output_dir: str = "./reports/charts") -> str:
    """Combined box plot for all numeric columns to visualise outliers."""
    out = _ensure_dir(output_dir)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        return None

    # Normalise for side-by-side comparison
    df_norm = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
    df_melted = df_norm.melt(var_name="Feature", value_name="Normalised Value")

    fig = px.box(
        df_melted, x="Feature", y="Normalised Value",
        title="Outlier Detection — Normalised Box Plots",
        template="plotly_white",
        color="Feature",
    )
    fig.update_layout(title_font_size=16, showlegend=False)

    path = str(out / "outliers_boxplot.html")
    fig.write_html(path)
    logger.info(f"Outlier box plot saved → {path}")
    return path


# ── 6. Feature Importance Chart ──────────────────────────────

def plot_feature_importance(feature_names: list, importances: list,
                             output_dir: str = "./reports/charts") -> str:
    """Horizontal bar chart of model feature importances."""
    out = _ensure_dir(output_dir)

    df_imp = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        df_imp, x="Importance", y="Feature",
        orientation="h",
        title="Feature Importance",
        template="plotly_white",
        color="Importance",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(title_font_size=16, showlegend=False)

    path = str(out / "feature_importance.html")
    fig.write_html(path)
    logger.info(f"Feature importance chart saved → {path}")
    return path


# ── 7. Model Comparison Chart ────────────────────────────────

def plot_model_comparison(results: dict, metric: str = "accuracy",
                           output_dir: str = "./reports/charts") -> str:
    """Bar chart comparing multiple models on a given metric."""
    out = _ensure_dir(output_dir)

    models  = list(results.keys())
    scores  = [results[m].get(metric, 0) for m in models]

    fig = px.bar(
        x=models, y=scores,
        title=f"Model Comparison — {metric.capitalize()}",
        labels={"x": "Model", "y": metric.capitalize()},
        template="plotly_white",
        color=scores,
        color_continuous_scale="Blues",
    )
    fig.update_layout(title_font_size=16, showlegend=False)

    path = str(out / f"model_comparison_{metric}.html")
    fig.write_html(path)
    logger.info(f"Model comparison chart saved → {path}")
    return path


# ── 8. Pairplot (static, seaborn) ───────────────────────────

def plot_pairplot(df: pd.DataFrame, target_col: str = None,
                  output_dir: str = "./reports/charts") -> str:
    """Seaborn pairplot saved as PNG for report embedding."""
    out = _ensure_dir(output_dir)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Cap at 6 columns to keep it readable
    cols_to_plot = num_cols[:6]
    if target_col and target_col in df.columns:
        plot_df = df[cols_to_plot + [target_col]].dropna()
        hue = target_col if df[target_col].nunique() <= 10 else None
    else:
        plot_df = df[cols_to_plot].dropna()
        hue = None

    sns.set_theme(style="whitegrid")
    pairplot = sns.pairplot(plot_df, hue=hue, plot_kws={"alpha": 0.5})
    pairplot.fig.suptitle("Pairplot — Feature Relationships", y=1.02, fontsize=14)

    path = str(out / "pairplot.png")
    pairplot.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Pairplot saved → {path}")
    return path


# ── 9. Run All Charts ────────────────────────────────────────

def generate_all_charts(df: pd.DataFrame, target_col: str = None,
                         output_dir: str = "./reports/charts") -> dict:
    """Generate the full suite of charts. Returns dict of all paths."""
    logger.info("Generating all charts...")

    chart_paths = {
        "distributions":         plot_distributions(df, output_dir),
        "correlation_heatmap":   plot_correlation_heatmap(df, output_dir),
        "missing_values":        plot_missing_values(df, output_dir),
        "outliers":              plot_outliers(df, output_dir),
        "pairplot":              plot_pairplot(df, target_col, output_dir),
    }

    if target_col and target_col in df.columns:
        chart_paths["target_distribution"] = plot_target_distribution(
            df, target_col, output_dir
        )

    logger.success(f"All charts generated → {output_dir}")
    return chart_paths