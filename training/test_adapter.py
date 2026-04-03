"""
Test trained adapter quality with domain-specific questions.
Compares responses with and without the adapter.

Usage:
    python training/test_adapter.py --tenant sis
    python training/test_adapter.py --tenant mfg
    python training/test_adapter.py --tenant sis --type dpo
"""
import argparse
import json
import os
from pathlib import Path

import torch
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

SIS_TEST_QUESTIONS = [
    "What are the FERPA requirements for releasing student records?",
    "How should a teacher handle a request for a student's grades from a parent?",
    "What is the process for enrolling a new student mid-semester?",
    "How do we record and report student attendance?",
    "What accommodations must be provided for students with documented disabilities?",
]

MFG_TEST_QUESTIONS = [
    "What is the lockout/tagout (LOTO) procedure before servicing equipment?",
    "How should a non-conforming product be handled in the quality control process?",
    "What is a CAPA and when must one be initiated?",
    "Describe the steps for a machine preventive maintenance inspection.",
    "What PPE is required when working with chemical solvents on the production floor?",
]

TENANT_SYSTEM_PROMPTS = {
    "sis": (
        "You are an AI assistant for a Student Information System. "
        "You help staff with enrollment, attendance, grading, FERPA compliance, "
        "and student records. Always follow FERPA privacy regulations."
    ),
    "mfg": (
        "You are an AI assistant for a manufacturing facility. "
        "You help staff with SOPs, quality control, safety protocols, "
        "and regulatory compliance. Always prioritize worker safety."
    ),
}


def ask(model, tokenizer, system_prompt: str, user_query: str, max_new_tokens: int = 256) -> str:
    """Generate a response for a single query."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        text = f"<|system|>\n{system_prompt}\n<|user|>\n{user_query}\n<|assistant|>\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def test_tenant_adapter(
    tenant_id: str,
    model_type: str = "sft",
    compare_base: bool = False,
):
    """
    Test a trained adapter against a set of domain questions.

    Args:
        tenant_id: "sis" or "mfg"
        model_type: "sft" or "dpo"
        compare_base: If True, also run questions against the base model for comparison
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    adapter_path = f"./models/adapters/{tenant_id}/{model_type}"

    if not Path(adapter_path).exists():
        logger.error(f"Adapter not found: {adapter_path}")
        logger.info("Run training first: python training/sft_train.py --tenant {tenant_id}")
        return

    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # Check adapter config for base model name
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            ac = json.load(f)
        base_model_name = ac.get("base_model_name_or_path", base_model_name)

    logger.info(f"Testing {tenant_id.upper()} {model_type.upper()} adapter")
    logger.info(f"Base model: {base_model_name}")
    logger.info(f"Adapter: {adapter_path}")

    # Load base model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        token=os.getenv("HF_TOKEN"),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,  # Use tokenizer from adapter dir (may have special tokens)
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    questions = SIS_TEST_QUESTIONS if tenant_id == "sis" else MFG_TEST_QUESTIONS
    system_prompt = TENANT_SYSTEM_PROMPTS[tenant_id]

    results = []

    # Optionally test base model first
    if compare_base:
        logger.info("\n" + "="*60)
        logger.info("BASE MODEL RESPONSES (no adapter)")
        logger.info("="*60)
        for i, q in enumerate(questions, 1):
            logger.info(f"\nQ{i}: {q}")
            response = ask(base_model, tokenizer, system_prompt, q)
            logger.info(f"A: {response[:300]}{'...' if len(response) > 300 else ''}")
            results.append({"model": "base", "question": q, "response": response})

    # Load adapter
    logger.info("\nLoading adapter...")
    adapted_model = PeftModel.from_pretrained(base_model, adapter_path)
    adapted_model.eval()

    logger.info("\n" + "="*60)
    logger.info(f"ADAPTED MODEL RESPONSES ({tenant_id.upper()} {model_type.upper()})")
    logger.info("="*60)

    for i, q in enumerate(questions, 1):
        logger.info(f"\nQ{i}: {q}")
        response = ask(adapted_model, tokenizer, system_prompt, q)
        logger.info(f"A: {response[:400]}{'...' if len(response) > 400 else ''}")
        results.append({"model": f"{tenant_id}_{model_type}", "question": q, "response": response})

    # Save results
    output_path = Path("evaluation/reports") / f"adapter_test_{tenant_id}_{model_type}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"\nTest results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test trained adapter")
    parser.add_argument("--tenant", type=str, required=True, choices=["sis", "mfg"])
    parser.add_argument("--type", type=str, default="sft", choices=["sft", "dpo"])
    parser.add_argument("--compare-base", action="store_true",
                        help="Also run base model for comparison")
    args = parser.parse_args()

    test_tenant_adapter(args.tenant, args.type, args.compare_base)


if __name__ == "__main__":
    main()
