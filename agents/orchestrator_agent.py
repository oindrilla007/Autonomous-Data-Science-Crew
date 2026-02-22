# ============================================================
# Orchestrator Agent — Coordinates the entire DS crew
# The brain that sequences all 7 agents end-to-end
# ============================================================

from crewai import LLM, Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from crewai.tools import tool
from agents.memory_agent import build_memory_agent, MemoryStore
from agents.ingestion_agent import build_ingestion_agent, build_ingestion_task
from agents.eda_agent import build_eda_agent, build_eda_task
from agents.modeling_agent import build_modeling_agent, build_modeling_task
from agents.evaluation_agent import build_evaluation_agent, build_evaluation_task
from agents.reporting_agent import build_reporting_agent, build_reporting_task
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path
import re
import json
import os

load_dotenv()


# ── 1. Init LLM ─────────────────────────────────────────────

def get_llm() -> LLM:
    llm = LLM(
        model=f"groq/{os.getenv('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')}",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", 300)),
    )
    # Force ReAct tool execution for stability with Groq models that
    # intermittently fail native function-calling with `tool_use_failed`.
    llm.supports_function_calling = lambda: False
    return llm


# ── 2. Orchestrator Tools ────────────────────────────────────

@tool("validate_pipeline_inputs")
def validate_pipeline_inputs(filepath: str, target_col: str) -> str:
    """
    Validate that pipeline inputs are correct before
    kicking off the full crew run.
    Checks file exists, is readable, and target column
    is present in the dataset.
    Returns a JSON validation report.
    """
    try:
        from tools.eda_tools import load_dataset

        checks  = {}
        issues  = []

        # File existence
        checks["file_exists"] = Path(filepath).exists()
        if not checks["file_exists"]:
            issues.append(f"File not found: {filepath}")
            return json.dumps({
                "status": "error",
                "checks": checks,
                "issues": issues,
            })

        # File readable
        try:
            df = load_dataset(filepath)
            checks["file_readable"] = True
        except Exception as e:
            checks["file_readable"] = False
            issues.append(f"File cannot be read: {e}")
            return json.dumps({
                "status": "error",
                "checks": checks,
                "issues": issues,
            })

        # Target column present
        checks["target_col_exists"] = target_col in df.columns
        if not checks["target_col_exists"]:
            issues.append(
                f"Target column '{target_col}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Minimum rows
        checks["sufficient_rows"] = df.shape[0] >= 50
        if not checks["sufficient_rows"]:
            issues.append(
                f"Dataset has only {df.shape[0]} rows. "
                f"Minimum 50 rows recommended."
            )

        # Not all nulls in target
        if target_col in df.columns:
            checks["target_not_all_null"] = not df[target_col].isnull().all()
            if not checks["target_not_all_null"]:
                issues.append(f"Target column '{target_col}' is all nulls")

        status = "ready" if not issues else "warning"

        result = {
            "status":    status,
            "filepath":  filepath,
            "target_col": target_col,
            "shape":     {"rows": df.shape[0], "cols": df.shape[1]},
            "checks":    checks,
            "issues":    issues,
            "columns":   df.columns.tolist(),
        }

        logger.info(f"Pipeline validation | Status: {status} | Issues: {len(issues)}")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Pipeline validation failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("get_pipeline_status")
def get_pipeline_status(run_dir: str = "./") -> str:
    """
    Check the current status of the pipeline by inspecting
    which output files have been generated so far.
    Returns a JSON status report showing which stages
    are complete and which are pending.
    """
    try:
        checks = {
            "data_cleaned":      Path("./data/cleaned.csv").exists(),
            "model_saved":       Path("./models/best_model.pkl").exists(),
            "mlruns_exists":     Path("./mlruns").exists(),
            "chroma_db_exists":  Path("./chroma_db").exists(),
            "reports_dir":       Path("./reports").exists(),
        }

        # Count reports generated
        reports_dir  = Path("./reports")
        html_reports = list(reports_dir.glob("*.html")) if reports_dir.exists() else []
        md_reports   = list(reports_dir.glob("*.md"))   if reports_dir.exists() else []
        pdf_reports  = list(reports_dir.glob("*.pdf"))  if reports_dir.exists() else []
        charts       = list((reports_dir / "charts").glob("*")) \
                       if (reports_dir / "charts").exists() else []

        stages = {
            "ingestion":  checks["data_cleaned"],
            "eda":        len(charts) > 0,
            "modeling":   checks["model_saved"],
            "evaluation": checks["mlruns_exists"],
            "reporting":  len(html_reports) > 0,
        }

        completed = sum(stages.values())
        total     = len(stages)

        result = {
            "status":          "complete" if completed == total else "in_progress",
            "progress":        f"{completed}/{total} stages complete",
            "stages":          stages,
            "file_checks":     checks,
            "reports": {
                "html":   [str(p) for p in html_reports],
                "md":     [str(p) for p in md_reports],
                "pdf":    [str(p) for p in pdf_reports],
                "charts": len(charts),
            },
        }

        logger.info(f"Pipeline status | {completed}/{total} stages complete")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Pipeline status check failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("retrieve_memory_context")
def retrieve_memory_context(query: str) -> str:
    """
    Query the vector memory store for context relevant
    to a given topic or question.
    Useful for the orchestrator to gather context
    before making pipeline decisions.
    Returns a JSON list of relevant memory documents.
    """
    try:
        memory  = MemoryStore()
        results = memory.recall(query=query, n_results=5)

        if not results:
            return json.dumps({
                "status":   "empty",
                "message":  "No relevant context found in memory",
                "results":  [],
            })

        return json.dumps({
            "status":  "success",
            "query":   query,
            "count":   len(results),
            "results": results,
        }, default=str)

    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@tool("summarise_pipeline_results")
def summarise_pipeline_results() -> str:
    """
    Generate a high-level executive summary of everything
    the pipeline has produced so far by scanning all
    output files and the memory store.
    Returns a JSON executive summary.
    """
    try:
        memory  = MemoryStore()
        context = memory.recall_full_context(
            query="dataset analysis model results evaluation report"
        )

        # Scan outputs
        reports_dir  = Path("./reports")
        html_reports = list(reports_dir.glob("*.html")) if reports_dir.exists() else []
        md_reports   = list(reports_dir.glob("*.md"))   if reports_dir.exists() else []
        pdf_reports  = list(reports_dir.glob("*.pdf"))  if reports_dir.exists() else []

        model_exists = Path("./models/best_model.pkl").exists()
        data_cleaned = Path("./data/cleaned.csv").exists()

        summary = {
            "pipeline_complete": len(html_reports) > 0 and model_exists,
            "outputs": {
                "cleaned_dataset":  str(Path("./data/cleaned.csv"))
                                    if data_cleaned else None,
                "trained_model":    "./models/best_model.pkl"
                                    if model_exists else None,
                "html_reports":     [str(p) for p in html_reports],
                "md_reports":       [str(p) for p in md_reports],
                "pdf_reports":      [str(p) for p in pdf_reports],
            },
            "memory_context":   context[:1000] + "..."
                                if len(context) > 1000 else context,
            "memory_documents": memory.count(),
        }

        logger.info("Pipeline results summarised")
        return json.dumps(summary, default=str)

    except Exception as e:
        logger.error(f"Pipeline summary failed: {e}")
        return json.dumps({"status": "error", "message": str(e)})


# ── 3. Build Orchestrator Agent ──────────────────────────────

def build_orchestrator_agent(llm: ChatGroq = None) -> Agent:
    """Build the CrewAI Orchestrator Agent."""
    if llm is None:
        llm = get_llm()

    agent = Agent(
        role="Data Science Pipeline Orchestrator",
        goal=(
            "Coordinate and oversee the entire autonomous data science "
            "pipeline from raw data ingestion through to final report "
            "delivery. Ensure every agent runs in the correct sequence, "
            "validate inputs and outputs at each stage, and deliver a "
            "complete end-to-end analysis with zero manual intervention."
        ),
        backstory=(
            "You are the lead data science team manager who has run "
            "hundreds of end-to-end ML projects. You know exactly which "
            "agent to call, in what order, and how to handle failures "
            "gracefully. You never let a pipeline stall — you always "
            "find a way forward and keep the team aligned on the goal."
        ),
        tools=[
            validate_pipeline_inputs,
            get_pipeline_status,
            retrieve_memory_context,
            summarise_pipeline_results,
        ],
        llm=llm,
        verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        allow_delegation=True,
        max_iter=int(os.getenv("MAX_ITERATIONS", 1)),
    )

    logger.info("Orchestrator Agent built")
    return agent


# ── 4. Build Full Crew ───────────────────────────────────────

def build_full_crew(
    filepath: str,
    target_col: str,
) -> Crew:
    """
    Assemble the complete 7-agent crew with all tasks
    wired in the correct sequential order.
    Returns a ready-to-run CrewAI Crew.
    """
    logger.info("Assembling full DS crew...")

    llm = get_llm()

    # ── Build all agents
    memory_agent      = build_memory_agent(llm)
    ingestion_agent   = build_ingestion_agent(llm)
    eda_agent         = build_eda_agent(llm)
    modeling_agent    = build_modeling_agent(llm)
    evaluation_agent  = build_evaluation_agent(llm)
    reporting_agent   = build_reporting_agent(llm)
    orchestrator      = build_orchestrator_agent(llm)

    # ── Build tasks in sequence
    ingestion_task = build_ingestion_task(
        agent=ingestion_agent,
        filepath=filepath,
        cleaned_path="./data/cleaned.csv",
    )

    eda_task = build_eda_task(
        agent=eda_agent,
        filepath="./data/cleaned.csv",
        target_col=target_col,
        context_tasks=[],
    )

    modeling_task = build_modeling_task(
        agent=modeling_agent,
        filepath="./data/cleaned.csv",
        target_col=target_col,
        context_tasks=[],
    )

    evaluation_task = build_evaluation_task(
        agent=evaluation_agent,
        filepath="./data/cleaned.csv",
        target_col=target_col,
        task_type="auto",
        context_tasks=[],
    )

    reporting_task = build_reporting_task(
        agent=reporting_agent,
        dataset_filepath="./data/cleaned.csv",
        target_col=target_col,
        context_tasks=[],
    )

    # ── Orchestrator oversight task
    orchestrator_task = Task(
        description=f"""
        Oversee and validate the complete pipeline run for: {filepath}
        Target column: {target_col}

        1. Validate pipeline inputs using validate_pipeline_inputs
        2. Monitor pipeline status using get_pipeline_status
        3. Retrieve memory context using retrieve_memory_context
           with query: "dataset analysis model results"
        4. After all agents complete, summarise results
           using summarise_pipeline_results

        Ensure the pipeline completed successfully and all outputs
        were generated. Flag any issues or missing outputs.
        Provide a final executive summary of the entire run.
        """,
        expected_output=(
            "A final executive summary confirming the pipeline completed "
            "successfully, listing all generated outputs (model path, "
            "report paths, chart count), key performance metrics achieved, "
            "and any issues or recommendations for the next run."
        ),
        agent=orchestrator,
        context=[],
    )

    # ── Assemble crew
    crew = Crew(
        agents=[
            ingestion_agent,
            eda_agent,
            modeling_agent,
            evaluation_agent,
            reporting_agent,
            orchestrator,
        ],
        tasks=[
            ingestion_task,
            eda_task,
            modeling_task,
            evaluation_task,
            reporting_task,
            orchestrator_task,
        ],
        process=Process.sequential,
        verbose=os.getenv("AGENT_VERBOSE", "true").lower() == "true",
        memory=False,
    )

    logger.success(
        f"Full crew assembled | "
        f"Agents: {len(crew.agents)} | Tasks: {len(crew.tasks)}"
    )
    return crew


# ── 5. Run Full Pipeline ─────────────────────────────────────

def run_pipeline(filepath: str, target_col: str) -> dict:
    """
    Top-level function to run the complete autonomous
    data science pipeline end-to-end.
    """
    import time

    logger.info("=" * 60)
    logger.info("AUTONOMOUS DATA SCIENCE CREW — PIPELINE START")
    logger.info(f"Dataset:  {filepath}")
    logger.info(f"Target:   {target_col}")
    logger.info("=" * 60)

    # Create required directories
    for d in ["./data", "./models", "./reports", "./reports/charts",
              "./logs", "./chroma_db", "./mlruns"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Validate inputs first
    validation = json.loads(
        validate_pipeline_inputs.run(filepath=filepath, target_col=target_col)
    )
    if validation.get("status") == "error":
        logger.error(f"Pipeline validation failed: {validation.get('issues')}")
        return {"status": "error", "validation": validation}

    low_cost_mode = os.getenv("LOW_COST_MODE", "false").lower() == "true"
    if low_cost_mode:
        return run_pipeline_low_cost(filepath, target_col, validation)

    # Build and run crew with 429-aware retry logic
    result = None
    attempts = int(os.getenv("PIPELINE_MAX_RETRIES", 3))
    for attempt in range(1, attempts + 1):
        try:
            crew = build_full_crew(filepath, target_col)
            logger.info("Throttling 2s before crew kickoff to avoid rate limits...")
            time.sleep(2)
            result = crew.kickoff()
            break
        except Exception as e:
            is_rate_limit = "rate_limit" in str(e).lower() or "429" in str(e)
            if not is_rate_limit or attempt == attempts:
                raise
            wait_s = _extract_retry_seconds(str(e), fallback=12)
            logger.warning(
                f"Rate limited (attempt {attempt}/{attempts}). "
                f"Waiting {wait_s:.1f}s before retry."
            )
            time.sleep(wait_s)

    if result is None:
        raise RuntimeError("Crew execution failed after retries")

    generated_reports = _finalise_reports(cleaned_path="./data/cleaned.csv", target_col=target_col)

    # Final status
    status = json.loads(get_pipeline_status.run())
    summary= json.loads(summarise_pipeline_results.run())

    logger.info("=" * 60)
    logger.success("PIPELINE COMPLETE")
    logger.info(f"Progress: {status.get('progress')}")
    logger.info("=" * 60)

    return {
        "status":       "complete",
        "crew_output":  str(result),
        "generated_reports": generated_reports,
        "pipeline_status": status,
        "summary":      summary,
    }


def _extract_retry_seconds(error_text: str, fallback: float = 10.0) -> float:
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", error_text, re.I)
    if match:
        return float(match.group(1)) + 1.0
    return fallback


def run_pipeline_low_cost(filepath: str, target_col: str, validation: dict) -> dict:
    """
    Low-cost mode: run deterministic tool chain directly to minimise LLM calls.
    """
    from agents.ingestion_agent import (
        load_and_validate_dataset,
        detect_column_types,
        suggest_target_column,
        clean_dataset,
    )
    from agents.eda_agent import (
        run_exploratory_analysis,
        analyse_missing_values,
        analyse_correlations,
        analyse_outliers,
        generate_visualisations,
        generate_profiling_report,
    )
    from agents.modeling_agent import run_automl_training
    from agents.evaluation_agent import (
        evaluate_classification_model,
        evaluate_regression_model,
        check_overfitting,
        generate_evaluation_charts,
    )
    from agents.reporting_agent import (
        compile_report_data,
        generate_html_report,
        generate_markdown_report,
        generate_pdf_report,
    )

    logger.info("LOW_COST_MODE enabled: running direct tool pipeline")

    # Ingestion
    ingestion_validate = json.loads(load_and_validate_dataset.run(filepath=filepath))
    ingestion_types = json.loads(detect_column_types.run(filepath=filepath))
    ingestion_target = json.loads(suggest_target_column.run(filepath=filepath))
    cleaned = json.loads(clean_dataset.run(filepath=filepath, output_path="./data/cleaned.csv"))

    cleaned_path = cleaned.get("output_path", "./data/cleaned.csv")

    # EDA
    eda_main = json.loads(
        run_exploratory_analysis.run(filepath=cleaned_path, target_col=target_col)
    )
    eda_missing = json.loads(analyse_missing_values.run(filepath=cleaned_path))
    eda_corr = json.loads(analyse_correlations.run(filepath=cleaned_path))
    eda_outliers = json.loads(analyse_outliers.run(filepath=cleaned_path))
    eda_viz = json.loads(
        generate_visualisations.run(filepath=cleaned_path, target_col=target_col)
    )
    eda_profile = json.loads(generate_profiling_report.run(filepath=cleaned_path))

    # Modeling + evaluation
    modeling = json.loads(run_automl_training.run(filepath=cleaned_path, target_col=target_col))
    task_type = modeling.get("task_type", "classification")
    if task_type == "regression":
        evaluation = json.loads(
            evaluate_regression_model.run(filepath=cleaned_path, target_col=target_col)
        )
    else:
        evaluation = json.loads(
            evaluate_classification_model.run(filepath=cleaned_path, target_col=target_col)
        )

    overfit = json.loads(check_overfitting.run(filepath=cleaned_path, target_col=target_col))
    eval_charts = json.loads(
        generate_evaluation_charts.run(filepath=cleaned_path, target_col=target_col)
    )

    # Reporting (template-based, compact)
    eda_narrative = (
        f"Rows={eda_main.get('shape', {}).get('rows', 'n/a')}, "
        f"Cols={eda_main.get('shape', {}).get('columns', 'n/a')}, "
        f"MissingCols={eda_missing.get('columns_with_missing', 0)}, "
        f"HighCorrPairs={len(eda_corr.get('high_corr_pairs', []))}, "
        f"Outliers={eda_outliers.get('total_outliers', 0)}."
    )
    chart_paths = {}
    if isinstance(eda_viz.get("chart_paths"), dict):
        chart_paths.update(eda_viz["chart_paths"])
    if isinstance(eval_charts.get("chart_paths"), dict):
        chart_paths.update(eval_charts["chart_paths"])

    report_data_json = compile_report_data.run(
        dataset_filepath=cleaned_path,
        target_col=target_col,
        eda_summary_json=json.dumps({"narrative": eda_narrative}),
        automl_results_json=json.dumps(modeling),
        evaluation_json=json.dumps(evaluation),
        overfit_json=json.dumps(overfit),
        chart_paths_json=json.dumps(chart_paths),
    )
    html_result = json.loads(generate_html_report.run(report_data_json=report_data_json))
    md_result = json.loads(generate_markdown_report.run(report_data_json=report_data_json))
    pdf_result = {"status": "skipped"}
    if html_result.get("report_path"):
        pdf_result = json.loads(
            generate_pdf_report.run(html_report_path=html_result["report_path"])
        )

    status = json.loads(get_pipeline_status.run())
    summary = json.loads(summarise_pipeline_results.run())

    return {
        "status": "complete",
        "mode": "low_cost_direct_tools",
        "validation": validation,
        "ingestion": {
            "validate": ingestion_validate,
            "types": ingestion_types,
            "target_suggestion": ingestion_target,
            "cleaning": cleaned,
        },
        "eda": {
            "summary": eda_main,
            "missing": eda_missing,
            "correlations": eda_corr,
            "outliers": eda_outliers,
            "visuals": eda_viz,
            "profiling": eda_profile,
        },
        "modeling": modeling,
        "evaluation": evaluation,
        "overfitting": overfit,
        "reports": {
            "html": html_result,
            "markdown": md_result,
            "pdf": pdf_result,
        },
        "pipeline_status": status,
        "summary": summary,
    }


def _finalise_reports(cleaned_path: str, target_col: str) -> dict:
    """
    Deterministically generate HTML/MD/PDF reports after crew kickoff.
    This avoids partial reporting outputs when the reporting agent stops early.
    """
    from tools.eda_tools import load_dataset
    from pipelines.automl_pipeline import detect_task_type
    from agents.eda_agent import run_exploratory_analysis
    from agents.evaluation_agent import (
        evaluate_classification_model,
        evaluate_regression_model,
        check_overfitting,
        generate_evaluation_charts,
    )
    from agents.modeling_agent import load_saved_model, run_automl_training
    from agents.reporting_agent import (
        compile_report_data,
        generate_html_report,
        generate_markdown_report,
        generate_pdf_report,
    )

    try:
        df = load_dataset(cleaned_path)
        task_type = detect_task_type(df[target_col])

        eda_data = json.loads(
            run_exploratory_analysis.run(filepath=cleaned_path, target_col=target_col)
        )
        if task_type == "regression":
            eval_data = json.loads(
                evaluate_regression_model.run(filepath=cleaned_path, target_col=target_col)
            )
        else:
            eval_data = json.loads(
                evaluate_classification_model.run(filepath=cleaned_path, target_col=target_col)
            )
        overfit_data = json.loads(
            check_overfitting.run(filepath=cleaned_path, target_col=target_col)
        )
        chart_data = json.loads(
            generate_evaluation_charts.run(filepath=cleaned_path, target_col=target_col)
        )
        model_info = json.loads(load_saved_model.run(model_path="./models/best_model.pkl"))

        # Prefer real AutoML output if available; fallback to rerun AutoML once.
        automl_path = Path("./models/automl_results.json")
        automl_data = {}
        if automl_path.exists():
            try:
                with open(automl_path, "r", encoding="utf-8") as f:
                    automl_data = json.load(f)
            except Exception:
                automl_data = {}
        if not automl_data:
            try:
                automl_data = json.loads(
                    run_automl_training.run(filepath=cleaned_path, target_col=target_col)
                )
            except Exception:
                automl_data = {}

        if not isinstance(automl_data, dict) or automl_data.get("status") == "error":
            best_name = (
                model_info.get("pipeline_steps", {}).get("model")
                or model_info.get("model_type", "best_model")
            )
            automl_data = {
                "task_type": task_type,
                "best_model_name": best_name,
                "best_metrics": eval_data.get("metrics", {}),
                "all_model_scores": {},
                "top_features": [],
            }

        report_data_json = compile_report_data.run(
            dataset_filepath=cleaned_path,
            target_col=target_col,
            eda_summary_json=json.dumps(eda_data),
            automl_results_json=json.dumps(automl_data),
            evaluation_json=json.dumps(eval_data),
            overfit_json=json.dumps(overfit_data),
            chart_paths_json=json.dumps(chart_data.get("chart_paths", {})),
        )
        html_result = json.loads(generate_html_report.run(report_data_json=report_data_json))
        md_result = json.loads(generate_markdown_report.run(report_data_json=report_data_json))
        pdf_result = {"status": "skipped"}
        if html_result.get("report_path"):
            pdf_result = json.loads(
                generate_pdf_report.run(html_report_path=html_result["report_path"])
            )

        logger.success("Deterministic report finalisation complete")
        return {"html": html_result, "markdown": md_result, "pdf": pdf_result}
    except Exception as e:
        logger.error(f"Deterministic report finalisation failed: {e}")
        return {"status": "error", "message": str(e)}


# ── 6. Standalone Test ───────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from pathlib import Path

    logger.info("Testing Orchestrator standalone...")
    Path("./data").mkdir(exist_ok=True)

    # Create sample data
    iris = load_iris(as_frame=True)
    iris.frame.to_csv("./data/sample_iris.csv", index=False)

    # Test validation
    val = validate_pipeline_inputs.run(
        filepath="./data/sample_iris.csv", target_col="target"
    )
    logger.info(f"Validation: {val}")

    # Test status
    status = get_pipeline_status.run()
    logger.info(f"Status: {status}")

    logger.success("Orchestrator test complete")
