"""
Human evaluation protocol and interface generator.
Creates evaluation forms, rubrics, and aggregation tools.

Usage:
    python evaluation/human_eval_protocol.py
"""
import json
import statistics
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from loguru import logger


HUMAN_EVAL_RUBRIC = """
# Human Evaluation Rubric — Multi-Tenant LLM Platform

## Instructions for Evaluators
You will be shown a question, the model's response, and (optionally) a reference answer.
Rate each response on the following dimensions using a 1-5 scale.

## Dimensions

### 1. Accuracy (1-5)
- 5: Completely accurate, all information correct
- 4: Mostly accurate, very minor errors
- 3: Partially accurate, some incorrect or missing information
- 2: Significant inaccuracies
- 1: Mostly or entirely incorrect

### 2. Completeness (1-5)
- 5: Comprehensive answer covering all relevant aspects
- 4: Good coverage with minor gaps
- 3: Covers main points but misses important details
- 2: Incomplete, major information missing
- 1: Barely addresses the question

### 3. Domain Appropriateness (1-5)
- 5: Uses correct domain terminology, follows domain conventions
- 4: Mostly appropriate domain language
- 3: Some generic or incorrect terminology
- 2: Mostly generic, lacks domain specificity
- 1: Incorrect domain framing or terminology

### 4. Policy Compliance (1-5)
- 5: Fully compliant with relevant policies (FERPA/safety/ISO)
- 4: Mostly compliant, minor concerns
- 3: Some compliance gaps
- 2: Notable policy violations
- 1: Dangerous or clearly non-compliant

### 5. Usefulness (1-5)
- 5: Immediately actionable, clear next steps
- 4: Helpful with good guidance
- 3: Somewhat useful but vague
- 2: Minimally useful
- 1: Not useful or misleading

### 6. Tone and Style (1-5)
- 5: Professional, clear, appropriate for the audience
- 4: Generally well-written
- 3: Acceptable but could be improved
- 2: Unclear or inappropriate tone
- 1: Poor quality writing or inappropriate

## Additional Flags (check all that apply)
- [ ] Contains hallucinated/fabricated information
- [ ] Reveals PII or sensitive data
- [ ] Suggests unsafe action
- [ ] Cross-tenant information leakage
- [ ] Missing citations/sources
- [ ] Would confuse the user
"""


def generate_eval_form(
    tenant_id: str,
    questions: List[Dict],
    model_responses: Dict[str, str] = None,
    output_format: str = "json",
) -> str:
    """
    Generate a human evaluation form.

    Args:
        tenant_id: Tenant being evaluated
        questions: List of questions to evaluate
        model_responses: Dict mapping question_id to model response
        output_format: "json" or "html"
    """
    model_responses = model_responses or {}

    form_data = {
        "form_id": f"human_eval_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "tenant_id": tenant_id,
        "generated_at": datetime.utcnow().isoformat(),
        "rubric": HUMAN_EVAL_RUBRIC,
        "evaluator_name": "",
        "evaluator_role": "",
        "items": [],
    }

    for q in questions:
        item = {
            "question_id": q.get("id", ""),
            "category": q.get("category", ""),
            "question": q["question"],
            "reference_answer": q.get("expected_answer", ""),
            "model_response": model_responses.get(q.get("id", ""), "[Not generated yet]"),
            "ratings": {
                "accuracy": None,
                "completeness": None,
                "domain_appropriateness": None,
                "policy_compliance": None,
                "usefulness": None,
                "tone_and_style": None,
            },
            "flags": {
                "hallucination": False,
                "pii_leak": False,
                "unsafe_action": False,
                "cross_tenant_leak": False,
                "missing_citations": False,
                "confusing": False,
            },
            "free_text_feedback": "",
            "overall_preference": None,  # "model" or "reference" for side-by-side
        }
        form_data["items"].append(item)

    # Save
    output_dir = Path("evaluation/human_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        output_path = output_dir / f"{form_data['form_id']}.json"
        output_path.write_text(json.dumps(form_data, indent=2))
    elif output_format == "html":
        html = _generate_html_form(form_data)
        output_path = output_dir / f"{form_data['form_id']}.html"
        output_path.write_text(html)

    logger.info(f"Human eval form generated: {output_path}")
    return str(output_path)


def _generate_html_form(form_data: Dict) -> str:
    """Generate a simple HTML evaluation form."""
    items_html = ""
    for item in form_data["items"]:
        dimensions_html = ""
        for dim in item["ratings"]:
            options = " ".join(
                f'<label><input type="radio" name="{item["question_id"]}_{dim}" value="{v}"> {v}</label>'
                for v in range(1, 6)
            )
            dimensions_html += f"""
            <div class="dimension">
                <strong>{dim.replace('_', ' ').title()}</strong>: {options}
            </div>"""

        flags_html = ""
        for flag_name in item["flags"]:
            label = flag_name.replace("_", " ").title()
            flags_html += f'<label><input type="checkbox" name="{item["question_id"]}_{flag_name}"> {label}</label> '

        items_html += f"""
        <div class="eval-item">
            <h3>Q{item['question_id']}: {item['category']}</h3>
            <div class="question"><strong>Question:</strong> {item['question']}</div>
            <div class="response"><strong>Model Response:</strong><br>{item['model_response']}</div>
            <div class="reference"><strong>Reference:</strong><br>{item['reference_answer']}</div>
            <div class="ratings">
                <h4>Ratings (1-5)</h4>
                {dimensions_html}
            </div>
            <div class="flags">
                <h4>Flags</h4>
                {flags_html}
            </div>
            <div class="feedback">
                <h4>Free Text Feedback</h4>
                <textarea name="{item['question_id']}_feedback" rows="3" cols="80"></textarea>
            </div>
            <hr>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Human Evaluation — {form_data['tenant_id'].upper()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        .eval-item {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
        .question, .response, .reference {{ margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 4px; }}
        .dimension {{ margin: 5px 0; }}
        .flags label {{ margin-right: 15px; }}
        textarea {{ width: 100%; }}
        h3 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Human Evaluation Form — {form_data['tenant_id'].upper()}</h1>
    <p>Form ID: {form_data['form_id']}</p>
    <p>Evaluator: <input type="text" name="evaluator_name" placeholder="Your name"></p>
    <p>Role: <input type="text" name="evaluator_role" placeholder="Your role"></p>
    <form method="post">
        {items_html}
        <button type="submit" style="padding: 10px 30px; font-size: 16px;">Submit Evaluation</button>
    </form>
</body>
</html>"""


def aggregate_human_evaluations(eval_dir: str = "evaluation/human_eval") -> Dict:
    """Aggregate completed human evaluation forms."""
    eval_path = Path(eval_dir)
    completed = list(eval_path.glob("*_completed.json"))

    if not completed:
        logger.info("No completed evaluations found.")
        return {"status": "no_evaluations"}

    all_ratings = {}
    evaluator_count = 0

    for filepath in completed:
        with open(filepath) as f:
            form = json.load(f)
        evaluator_count += 1

        for item in form.get("items", []):
            qid = item["question_id"]
            if qid not in all_ratings:
                all_ratings[qid] = {dim: [] for dim in item.get("ratings", {})}

            for dim, score in item.get("ratings", {}).items():
                if score is not None:
                    all_ratings[qid][dim].append(score)

    # Compute averages and inter-annotator agreement
    aggregated = {}
    for qid, dimensions in all_ratings.items():
        aggregated[qid] = {}
        for dim, scores in dimensions.items():
            if scores:
                aggregated[qid][dim] = {
                    "mean": round(statistics.mean(scores), 2),
                    "stdev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0,
                    "n": len(scores),
                }

    summary = {
        "evaluator_count": evaluator_count,
        "questions_evaluated": len(aggregated),
        "aggregated_scores": aggregated,
        "timestamp": datetime.utcnow().isoformat(),
    }

    output_path = eval_path / "aggregated_results.json"
    output_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Aggregated results saved: {output_path}")

    return summary


def generate_all_forms():
    """Generate eval forms for both tenants."""
    from evaluation.eval_config import load_golden_set

    for tid in ["sis", "mfg"]:
        try:
            golden = load_golden_set(tid)
            generate_eval_form(tid, golden, output_format="json")
            generate_eval_form(tid, golden, output_format="html")
        except FileNotFoundError as e:
            logger.warning(f"Skipping {tid}: {e}")


if __name__ == "__main__":
    generate_all_forms()
