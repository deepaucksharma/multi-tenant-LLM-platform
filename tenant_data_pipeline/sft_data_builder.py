"""
Build Supervised Fine-Tuning (SFT) datasets from tenant documents.
Generates instruction-response pairs in chat format for TRL SFTTrainer.
"""
import json
import random
from typing import List, Dict
from loguru import logger

from tenant_data_pipeline.config import TENANTS

# ============================================================
# SIS SFT EXAMPLES
# ============================================================

SIS_SFT_EXAMPLES = [
    {"instruction": "What documents are required for student enrollment?", "input": "", "output": "For enrollment in District 42, students must provide: proof of residency, immunization records, and a birth certificate. Transfer students must also submit official transcripts from their previous institution. All documents can be uploaded via the SIS portal or presented to the registrar's office.", "topic": "enrollment"},
    {"instruction": "How long does the enrollment verification process take?", "input": "", "output": "The registrar verifies submitted enrollment documents within 3 business days. After verification, the student is assigned to the appropriate grade level and school based on address zoning, and a class schedule is generated.", "topic": "enrollment"},
    {"instruction": "What happens when a student enrolls mid-year?", "input": "", "output": "Students enrolling after the semester start date must complete an academic assessment. Credit transfer evaluation is completed within 5 business days. The receiving school counselor creates an individualized transition plan to support the student's academic integration.", "topic": "enrollment"},
    {"instruction": "Can a homeless student be denied enrollment?", "input": "", "output": "No. Under the McKinney-Vento Act, enrollment cannot be denied due to lack of a fixed address. Homeless students have the right to immediate enrollment, and the district must provide support services to ensure educational continuity.", "topic": "enrollment"},
    {"instruction": "What is the enrollment process for students with an IEP?", "input": "", "output": "Students with IEPs follow the standard enrollment process, with an additional step: the special education coordinator must review the student's IEP within 10 days of enrollment. Services outlined in the IEP must be provided during the review period.", "topic": "enrollment"},
    {"instruction": "How are English Language Learners processed during enrollment?", "input": "", "output": "English Language Learners must have the WIDA screener administered within 30 days of enrollment. Based on the screening results, appropriate language support services are assigned and documented in the SIS.", "topic": "enrollment"},
    {"instruction": "When must teachers record attendance in the SIS?", "input": "", "output": "Teachers must record attendance in the SIS within the first 15 minutes of each class period using codes: P (Present), A (Absent Unexcused), E (Excused Absence), T (Tardy), S (Suspended), F (Field Trip), M (Medical), R (Religious Observance).", "topic": "attendance"},
    {"instruction": "What happens after a student accumulates 5 unexcused absences?", "input": "", "output": "After 5 unexcused absences, the school counselor contacts the family as part of the progressive intervention protocol. If absences continue to 10, an attendance intervention team meeting is convened.", "topic": "attendance"},
    {"instruction": "How long do students have to complete make-up work after an excused absence?", "input": "", "output": "Students have one day per excused absence day to complete make-up work. Teachers must provide make-up work upon request within one school day. For extended absences of 5+ days, an individual make-up plan is created by the counselor.", "topic": "attendance"},
    {"instruction": "How long are attendance records retained?", "input": "", "output": "All attendance records are maintained in the SIS for 7 years after graduation. Attendance data is included in official transcripts when requested. FERPA protections apply to all attendance records.", "topic": "attendance"},
    {"instruction": "What is the grading scale used in District 42?", "input": "", "output": "District 42 uses: A (90-100, 4.0 GPA), B (80-89, 3.0 GPA), C (70-79, 2.0 GPA), D (60-69, 1.0 GPA), F (Below 60, 0.0 GPA). Honors and AP courses receive a 0.5 GPA weight bonus.", "topic": "grading"},
    {"instruction": "What is the process for changing a grade after report cards are issued?", "input": "", "output": "Grade changes after report card issuance require principal approval. A grade change form must be submitted documenting the reason and supporting evidence. All grade changes are logged in the SIS audit trail.", "topic": "grading"},
    {"instruction": "What happens to incomplete grades that are not resolved?", "input": "", "output": "Incomplete (I) grades must be resolved within 10 school days of the quarter end. If not resolved within that window, the SIS automatically converts the incomplete to an F grade.", "topic": "grading"},
    {"instruction": "What is FERPA and what does it protect?", "input": "", "output": "FERPA (Family Educational Rights and Privacy Act) protects the privacy of student education records. It applies to all schools receiving federal funding. Protected records include grades, transcripts, class lists, schedules, financial information, disciplinary records, and student IDs.", "topic": "ferpa_compliance"},
    {"instruction": "When can student records be disclosed without consent under FERPA?", "input": "", "output": "Records may be disclosed without consent to: school officials with legitimate educational interest, schools to which a student transfers, for financial aid purposes, to comply with a judicial order or subpoena, and in health and safety emergencies.", "topic": "ferpa_compliance"},
    {"instruction": "What access controls does the SIS enforce for FERPA compliance?", "input": "", "output": "The SIS enforces role-based access control, maintains audit logging of all record access, encrypts data at rest and in transit, requires annual FERPA training, and mandates breach notification within 72 hours.", "topic": "ferpa_compliance"},
    {"instruction": "Can parents opt out of directory information disclosure?", "input": "", "output": "Yes. Parents may opt out of directory information disclosure by submitting the opt-out form within 30 days of enrollment. Directory information includes student name, address, phone number, date of birth, enrollment status, and participation in activities.", "topic": "ferpa_compliance"},
    {"instruction": "How can a current student request an official transcript?", "input": "", "output": "Current students can request official transcripts via the SIS student portal. Requests are processed within 3 business days. The fee is $5 per official transcript, waived for students on free/reduced lunch for college applications.", "topic": "transcripts"},
    {"instruction": "Can transcripts be withheld for disciplinary reasons?", "input": "", "output": "No. Per district policy, transcript holds cannot be placed for disciplinary reasons. Transcripts may only be held for unpaid fees, unreturned materials, or an incomplete exit process.", "topic": "transcripts"},
    {"instruction": "How long are transcript records retained?", "input": "", "output": "Transcripts are permanent records and are never destroyed. Active records are in the SIS during enrollment. After graduation or withdrawal, records are archived for 99 years. All access is logged per FERPA audit requirements.", "topic": "transcripts"},
    {"instruction": "How long does the initial evaluation for accommodations take?", "input": "", "output": "The initial evaluation for accommodations must be completed within 60 calendar days of receiving parental consent. Referrals can be made by teachers, parents, or student self-referral, and the SIS tracks timelines throughout.", "topic": "accommodations"},
    {"instruction": "Who can access a student's IEP records in the SIS?", "input": "", "output": "Only IEP team members and authorized administrators can view full IEP records. Accommodation flags appear in teacher class rosters showing type only, not the full IEP. All records are protected under FERPA and IDEA.", "topic": "accommodations"},
    {"instruction": "What are the levels of progressive discipline?", "input": "", "output": "District 42 uses five levels: Level 1 (Minor) - verbal warning; Level 2 (Moderate) - office referral, parent contact; Level 3 (Serious) - in-school suspension 1-3 days; Level 4 (Severe) - out-of-school suspension up to 10 days; Level 5 (Critical) - expulsion hearing.", "topic": "disciplinary"},
    {"instruction": "What protections do special education students have in disciplinary proceedings?", "input": "", "output": "A manifestation determination is required before suspending a special education student beyond 10 cumulative days. The IEP team must convene within 10 school days. A student cannot be expelled for behavior determined to be a manifestation of their disability.", "topic": "disciplinary"},
    {"instruction": "Are disciplinary records included in official transcripts?", "input": "", "output": "No. Disciplinary records are NOT included in official transcripts per district policy. These records are maintained in the SIS for the duration of enrollment plus 5 years, with access restricted to administrators and designated counseling staff.", "topic": "disciplinary"},
    {"instruction": "What is the process for an intra-district transfer?", "input": "", "output": "Intra-district transfer requests are submitted through the SIS parent portal. They require approval from the receiving school principal. Transportation is not provided for transfers of choice. Decisions are communicated within 10 business days.", "topic": "transfer"},
    {"instruction": "How are grades handled during a mid-year transfer?", "input": "", "output": "For mid-year transfers, proportional grade calculation is applied for courses in progress. The sending school provides the current grade and attendance to date. The receiving teacher incorporates the proportional grade into the final calculation.", "topic": "transfer"},
    {"instruction": "What is the role-based access matrix for the SIS?", "input": "", "output": "Teachers access their own students' grades and attendance. Counselors view full academic records for assigned students. Administrators have full school-level access. Registrars handle enrollment and transcripts. Nurses access health records only.", "topic": "records_management"},
    {"instruction": "How are student records destroyed when required?", "input": "", "output": "Temporary records are destroyed 5 years after the student exits the district. Permanent records are maintained indefinitely. All destruction is logged in the SIS with an authorization chain. Electronic records undergo secure deletion with a certificate of destruction.", "topic": "records_management"},
    {"instruction": "What automated notifications does the SIS send to parents?", "input": "", "output": "The SIS sends: daily attendance alerts at 10:00 AM for absent students, grade alerts when any grade drops below C, and emergency notifications via SMS, email, and app push notification immediately.", "topic": "parent_communication"},
    {"instruction": "How does the SIS handle non-custodial parent access rights?", "input": "", "output": "Non-custodial parents retain FERPA rights unless a court order specifically restricts that access. Staff must verify custodial status in the SIS before providing information, and identity verification is required before disclosing any student information.", "topic": "parent_communication"},
]


# ============================================================
# MFG SFT EXAMPLES
# ============================================================

MFG_SFT_EXAMPLES = [
    {"instruction": "What is the startup sequence for the assembly line?", "input": "", "output": "Step 1: Activate main power (Circuit Breakers 1-4 ON). Step 2: Initialize PLC and wait for ready status. Step 3: Start conveyor at low speed (5 ft/min) for 2-minute warm-up. Step 4: Activate stations 1-12 in sequence. Step 5: Run 5 test units. Step 6: If all pass, increase to production speed. Step 7: Log startup in MES.", "topic": "standard_operating_procedures"},
    {"instruction": "What PPE is required on the assembly line?", "input": "", "output": "Minimum: safety glasses, steel-toe boots, hearing protection, gloves. Zone B (Welding): add full face shield, welding gloves, fire-resistant clothing. Zone C (Chemical): add chemical splash goggles, chemical-resistant gloves, apron.", "topic": "standard_operating_procedures"},
    {"instruction": "What should an operator do if a pre-startup check fails?", "input": "", "output": "Do NOT proceed with startup. Immediately contact the shift supervisor. Log the abnormal condition as an incident in the QMS. Production cannot begin until all pre-startup checks pass successfully.", "topic": "standard_operating_procedures"},
    {"instruction": "What items are on the pre-startup checklist?", "input": "", "output": "Review previous shift handoff notes, verify raw material availability, check tool calibration status (within 30-day window), inspect conveyor belt, verify air pressure (90-110 PSI), confirm PLC program version, test safety light curtains at stations 3, 7, and 12, verify torque tool settings.", "topic": "standard_operating_procedures"},
    {"instruction": "What are the defect severity levels?", "input": "", "output": "Class 1 (Critical): causes safety hazard or complete failure — immediate line stop, quarantine, CAPA required. Class 2 (Major): significantly reduces functionality — unit rejection, increased inspection. Class 3 (Minor): does not affect function or safety — log and repair if economically viable.", "topic": "defect_classification"},
    {"instruction": "What is the escalation matrix for defects?", "input": "", "output": "Class 1: Quality Manager notified within 1 hour, customer within 24 hours. Class 2: Quality Engineer within 4 hours, trend review at daily standup. Class 3: logged and reviewed weekly, action required if trend exceeds control limit.", "topic": "defect_classification"},
    {"instruction": "What information must be included when logging a defect?", "input": "", "output": "Include: defect code, severity classification, location on product, operator ID, timestamp, lot number, serial number, and corrective action taken. Photos required for all Class 1 and Class 2 defects. SPC charts updated automatically from MES data.", "topic": "defect_classification"},
    {"instruction": "What triggers a CAPA?", "input": "", "output": "CAPA triggers: customer complaints (all Class 1, repeat Class 2), internal audit findings, process capability below Cpk 1.33, SPC out-of-control conditions, regulatory findings, supplier quality issues, and product recalls.", "topic": "capa"},
    {"instruction": "What are the steps in the CAPA process?", "input": "", "output": "7 steps: 1) Identification within 24 hours. 2) Containment within 48 hours. 3) Root Cause Analysis within 15 business days. 4) Corrective Action Plan within 20 days. 5) Implementation. 6) Effectiveness Verification 30-90 days post-implementation. 7) Closure by Quality Manager.", "topic": "capa"},
    {"instruction": "What root cause analysis methods are approved for CAPA?", "input": "", "output": "Approved methods: 5-Why Analysis, Fishbone (Ishikawa) Diagram, 8D Problem Solving, and Fault Tree Analysis. Analysis must document the root cause with supporting evidence and distinguish between root cause and contributing factors.", "topic": "capa"},
    {"instruction": "How does the document control system work?", "input": "", "output": "All QMS documents use the format QMS-[TYPE]-[NUMBER]-[REV]. Review cycle is annual minimum. Changes require originator, reviewer, and approver signatures. Obsolete documents are archived with a 'SUPERSEDED' watermark.", "topic": "iso_documentation"},
    {"instruction": "What are the requirements for internal audits?", "input": "", "output": "Full system audit annually, process audits quarterly on a rotating schedule. Findings classified as: Major Nonconformity, Minor Nonconformity, or Observation. Auditors must be independent of the area audited and hold ISO 19011 certified internal auditor training.", "topic": "iso_documentation"},
    {"instruction": "How is risk-based thinking implemented?", "input": "", "output": "Risk assessment uses FMEA methodology. RPN = Severity × Occurrence × Detection. Actions required for RPN > 100 or Severity ≥ 8. Risk register maintained and reviewed quarterly.", "topic": "iso_documentation"},
    {"instruction": "What is the preventive maintenance schedule?", "input": "", "output": "Daily: visual inspection, lubrication checks, clean work areas. Weekly: belt tension, fluid levels, filter inspection. Monthly: full lubrication, calibration verification, electrical connections. Quarterly: comprehensive audit, bearing inspection. Annually: complete overhaul per manufacturer specs.", "topic": "maintenance"},
    {"instruction": "What is the lockout/tagout procedure?", "input": "", "output": "1) Notify affected personnel. 2) Shut down equipment. 3) Isolate all energy sources. 4) Apply personal lock and tag. 5) Verify zero energy state. 6) Perform maintenance. 7) Remove LOTO devices in reverse order. 8) Test equipment before returning to service.", "topic": "maintenance"},
    {"instruction": "What are the maintenance priority levels?", "input": "", "output": "P1-Emergency: respond within 1 hour. P2-Urgent: respond within 4 hours. P3-Standard: respond within 24 hours. P4-Planned: scheduled maintenance window. All activities require a work order in CMMS.", "topic": "maintenance"},
    {"instruction": "What are the emergency procedures for a chemical spill?", "input": "", "output": "Immediately evacuate the area. Notify EHS department. Do NOT attempt cleanup unless trained. Spill kit locations are at each department entrance. All incidents must be reported within 4 hours in the EHS system.", "topic": "safety_protocols"},
    {"instruction": "What are the safety metrics tracked?", "input": "", "output": "Total Recordable Incident Rate (TRIR) target <2.0, Days Away Restricted or Transferred (DART) target <1.0, near-miss reporting targeting >5 per month, and safety audit score targeting >90%.", "topic": "safety_protocols"},
    {"instruction": "What is the maximum lift weight without assistance?", "input": "", "output": "Maximum manual lift weight is 50 pounds. For repetitive tasks, rotation schedules are implemented. Anti-fatigue mats at all standing workstations. Ergonomic assessments required for all new setups and upon employee request.", "topic": "safety_protocols"},
    {"instruction": "What is the First Article Inspection process?", "input": "", "output": "FAI is required at the start of each production run and after any setup change. FAI checks all dimensions per the control plan. If the first article fails, the line must not proceed until the issue is resolved and a new FAI passes.", "topic": "quality_control"},
    {"instruction": "What are the calibration requirements for measuring equipment?", "input": "", "output": "All measurement equipment must be calibrated with NIST traceability. Green sticker = current; red sticker = out of calibration — do NOT use. Annual Gage R&R: %GRR must be <10% for critical, <30% for non-critical measurements.", "topic": "quality_control"},
    {"instruction": "How is non-conforming material handled?", "input": "", "output": "Non-conforming material is identified with a red tag and moved to the MRB area. Disposition options: Use As-Is (customer approval required), Rework, Scrap, Return to Supplier. All NCRs logged in QMS with defect code, quantity, and disposition.", "topic": "quality_control"},
    {"instruction": "How is the production schedule managed?", "input": "", "output": "Master Production Schedule generated weekly by planning using customer orders, demand forecasts, inventory levels, and capacity constraints. Loaded into MES by Friday for the following week. Daily adjustments only authorized by the production manager.", "topic": "production_scheduling"},
    {"instruction": "What is the changeover target time?", "input": "", "output": "Target changeover time is <30 minutes for the same product family. Changeover times tracked in MES for SMED improvement. A changeover checklist is required for all product changes, and verification includes first article inspection.", "topic": "production_scheduling"},
    {"instruction": "What regulations apply to the manufacturing facility?", "input": "", "output": "Applicable regulations: ISO 9001:2015 (Quality Management), ISO 14001:2015 (Environmental), OSHA 29 CFR 1910 (Safety), EPA regulations, and customer-specific requirements such as PPAP and APQP.", "topic": "regulatory_compliance"},
    {"instruction": "What are the record retention requirements?", "input": "", "output": "Retention requirements: quality records 7 years minimum, safety/OSHA records 5 years, environmental records 5 years, training records (employment + 5 years), product liability records (product lifetime + 10 years).", "topic": "regulatory_compliance"},
    {"instruction": "What is the cycle count schedule?", "input": "", "output": "A items (80% of value): counted monthly. B items (15% of value): counted quarterly. C items (5% of value): counted semi-annually. Annual physical inventory in December. Inventory accuracy target >98%.", "topic": "inventory_management"},
    {"instruction": "How is material traceability maintained?", "input": "", "output": "Lot traceability maintained from receipt through production to finished goods: Supplier lot → Receiving lot → Work order → Serial number. Both forward and reverse traceability achievable within 4 hours. Records maintained per retention schedule.", "topic": "inventory_management"},
    {"instruction": "What CAPA metrics are tracked?", "input": "", "output": "Key CAPA metrics: open CAPAs by age/department/category, average time to closure, first-pass effectiveness rate (target >85%), and recurrence rate (target <5%). Reviewed during management review meetings.", "topic": "capa"},
    {"instruction": "What are the LOTO requirements for contractors?", "input": "", "output": "Contractors must follow this facility's LOTO procedure AND their own company's program. The Facility Safety Manager coordinates with contractor supervisors before any maintenance work begins. Each contractor must have their own uniquely keyed lock.", "topic": "safety_protocols"},
    {"instruction": "When is a safety incident considered OSHA recordable?", "input": "", "output": "An incident is OSHA recordable if it results in: days away from work, restricted work or job transfer, medical treatment beyond first aid, loss of consciousness, or diagnosis of a significant injury. Near misses and first aid cases are not recordable but must be logged internally.", "topic": "regulatory_compliance"},
]


def augment_examples(examples: List[Dict], multiplier: int = 2) -> List[Dict]:
    """Create variations by rephrasing instructions."""
    prefixes = [
        "Can you explain ", "Please describe ", "What is the procedure for ",
        "Tell me about ", "I need to know about ", "Explain the process of ",
        "How does the system handle ", "What are the requirements for ",
        "Could you clarify ", "Walk me through ",
    ]
    augmented = list(examples)
    for _ in range(multiplier - 1):
        for ex in examples:
            original_q = ex["instruction"]
            prefix = random.choice(prefixes)
            clean_q = original_q.lower()
            for starter in ["what is ", "what are ", "how ", "when ", "can ", "what "]:
                if clean_q.startswith(starter):
                    clean_q = clean_q[len(starter):]
                    break
            new_q = f"{prefix}{clean_q}"
            if not new_q.endswith("?"):
                new_q += "?"
            augmented.append({
                "instruction": new_q,
                "input": ex.get("input", ""),
                "output": ex["output"],
                "topic": ex["topic"],
            })
    return augmented


def _get_system_prompt(tenant_id: str) -> str:
    from tenant_data_pipeline.config import TENANTS
    return TENANTS[tenant_id].system_prompt


def _to_chat_format(example: Dict) -> Dict:
    """Convert to TRL SFTTrainer chat format."""
    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    user_msg = example["instruction"]
    if example.get("input"):
        user_msg += f"\n\nContext: {example['input']}"
    messages.append({"role": "user", "content": user_msg})
    messages.append({"role": "assistant", "content": example["output"]})
    return {
        "messages": messages,
        "topic": example.get("topic", ""),
        "tenant_id": example.get("tenant_id", ""),
    }


def build_sft_dataset(tenant_id: str) -> List[Dict]:
    """Build SFT dataset for a tenant."""
    config = TENANTS[tenant_id]
    config.sft_dir.mkdir(parents=True, exist_ok=True)

    examples = SIS_SFT_EXAMPLES if tenant_id == "sis" else MFG_SFT_EXAMPLES

    target_count = 120
    multiplier = max(2, (target_count // len(examples)) + 1)
    augmented = augment_examples(examples, multiplier=multiplier)[:target_count]

    system_prompt = _get_system_prompt(tenant_id)
    formatted = []
    for ex in augmented:
        formatted.append({
            "system": system_prompt,
            "instruction": ex["instruction"],
            "input": ex.get("input", ""),
            "output": ex["output"],
            "topic": ex["topic"],
            "tenant_id": tenant_id,
        })

    random.seed(42)
    random.shuffle(formatted)
    split_idx = int(len(formatted) * 0.9)
    train_set = formatted[:split_idx]
    eval_set = formatted[split_idx:]

    (config.sft_dir / "train.json").write_text(json.dumps(train_set, indent=2), encoding="utf-8")
    (config.sft_dir / "eval.json").write_text(json.dumps(eval_set, indent=2), encoding="utf-8")

    chat_train = [_to_chat_format(ex) for ex in train_set]
    chat_eval = [_to_chat_format(ex) for ex in eval_set]
    (config.sft_dir / "train_chat.json").write_text(json.dumps(chat_train, indent=2), encoding="utf-8")
    (config.sft_dir / "eval_chat.json").write_text(json.dumps(chat_eval, indent=2), encoding="utf-8")

    logger.info(f"[{tenant_id}] SFT dataset: {len(train_set)} train, {len(eval_set)} eval")
    return formatted


def build_all_sft_datasets() -> Dict:
    results = {}
    for tenant_id in TENANTS:
        dataset = build_sft_dataset(tenant_id)
        results[tenant_id] = len(dataset)
    return results


if __name__ == "__main__":
    results = build_all_sft_datasets()
    print(f"SFT dataset sizes: {results}")
