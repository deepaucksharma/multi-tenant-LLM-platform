"""
Single source of truth for tenant configuration.
All modules import TENANTS, TenantConfig, and constants from here.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data"))

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 100


@dataclass
class TenantConfig:
    tenant_id: str
    domain: str
    raw_dir: Path
    processed_dir: Path
    chunks_dir: Path
    sft_dir: Path
    dpo_dir: Path
    eval_dir: Path
    system_prompt: str
    topics: List[str] = field(default_factory=list)


def _make_tenant(tenant_id: str, domain: str, system_prompt: str, topics: List[str]) -> TenantConfig:
    base = DATA_ROOT / tenant_id
    return TenantConfig(
        tenant_id=tenant_id,
        domain=domain,
        raw_dir=DATA_ROOT / "raw" / tenant_id,
        processed_dir=base / "processed",
        chunks_dir=base / "chunks",
        sft_dir=base / "sft",
        dpo_dir=base / "dpo",
        eval_dir=base / "eval",
        system_prompt=system_prompt,
        topics=topics,
    )


TENANTS = {
    "sis": _make_tenant(
        tenant_id="sis",
        domain="education",
        system_prompt=(
            "You are a Student Information System (SIS) assistant for an educational institution. "
            "Your role is to help staff, students, and parents with questions about enrollment, "
            "attendance, grading, transcripts, accommodations, and school policies. "
            "You must strictly follow FERPA privacy regulations — never disclose personally "
            "identifiable student information without proper authorization. "
            "Always cite the specific policy or document that supports your answer. "
            "If a request falls outside your knowledge base or involves unauthorized data access, "
            "politely decline and recommend the appropriate office or procedure."
        ),
        topics=[
            "enrollment",
            "attendance",
            "grading",
            "ferpa_compliance",
            "transcripts",
            "accommodations",
            "disciplinary",
            "parent_communication",
            "transfer",
            "records_management",
        ],
    ),
    "mfg": _make_tenant(
        tenant_id="mfg",
        domain="manufacturing",
        system_prompt=(
            "You are a Manufacturing Operations assistant for an industrial facility. "
            "Your role is to help operators, engineers, and supervisors with questions about "
            "standard operating procedures, quality control, CAPA processes, safety protocols, "
            "maintenance, and ISO documentation. "
            "Safety is the highest priority — never suggest bypassing lockout/tagout (LOTO) "
            "procedures, containment steps, or PPE requirements under any circumstances. "
            "Always reference the specific SOP, procedure number, or policy document. "
            "If a request involves bypassing safety controls or is outside your knowledge base, "
            "refuse and escalate to the appropriate supervisor or safety officer."
        ),
        topics=[
            "standard_operating_procedures",
            "quality_control",
            "capa",
            "safety_protocols",
            "maintenance",
            "production_scheduling",
            "regulatory_compliance",
            "defect_classification",
            "inventory_management",
            "iso_documentation",
        ],
    ),
}
