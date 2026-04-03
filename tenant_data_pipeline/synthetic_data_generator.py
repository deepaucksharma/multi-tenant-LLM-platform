"""
Synthetic document generator for both tenants.
Creates realistic policy/procedure documents for SIS and MFG tenants.
"""
import json
from tenant_data_pipeline.config import TENANTS

# ============================================================
# SIS (Education) documents
# ============================================================

SIS_DOCUMENTS = [
    {
        "title": "Student Enrollment Policy",
        "topic": "enrollment",
        "doc_type": "policy",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """District 42 Student Enrollment Policy (Policy 5111)

PURPOSE
This policy establishes the procedures and requirements for enrolling students in District 42 schools.

REQUIRED DOCUMENTS
All enrolling students must submit the following:
- Proof of residency (utility bill, lease agreement, or notarized residency affidavit)
- Certified birth certificate
- Immunization records (per state requirements)
- Most recent report card or unofficial transcript

Transfer students must additionally provide:
- Official transcripts from previous institution
- Disciplinary records (last 2 years)
- Special education records, if applicable

ENROLLMENT PROCEDURES
Step 1: Submit all required documents via the SIS parent portal or in person at the District Office.
Step 2: Registrar verifies documentation within 3 business days.
Step 3: Student is assigned to grade level and home school based on current address zoning.
Step 4: School counselor generates an initial class schedule.
Step 5: Parents receive enrollment confirmation via email.

MID-YEAR ENROLLMENT
Students enrolling after the semester start date:
- Academic assessment administered within 5 business days
- Credit transfer evaluated by registrar and counselor
- Individualized transition plan created by counselor

SPECIAL CIRCUMSTANCES
Homeless Students: Per McKinney-Vento Act, enrollment cannot be denied due to lack of a fixed address. District must provide immediate enrollment and support services.
English Language Learners: WIDA screener must be administered within 30 days. Results determine language support services.
Students with IEPs: Special education coordinator reviews IEP within 10 days. Services must be provided during the review period.

RESIDENCY DISPUTES
If residency cannot be verified, the case is forwarded to the District Enrollment Officer. Provisional enrollment may be granted for up to 30 days pending investigation.

CONTACT
District Enrollment Office: enrollment@district42.edu | (555) 100-2000""",
    },
    {
        "title": "Student Attendance Policy",
        "topic": "attendance",
        "doc_type": "policy",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """District 42 Student Attendance Policy (Policy 5113)

PURPOSE
Regular school attendance is essential for academic success. This policy governs attendance requirements, absences, tardiness, and truancy.

ATTENDANCE REQUIREMENTS
Students must attend at least 90% of scheduled school days. Absences exceeding 10% (approximately 18 days) require a formal attendance review.

TYPES OF ABSENCES
Excused Absences:
- Illness (with doctor's note if absent 3+ consecutive days)
- Medical/dental appointments
- Approved family emergencies
- Religious observances (2 days per year)
- Court appearances
- College visits (juniors and seniors, pre-approved, maximum 2 days)

Unexcused Absences:
- Vacations not pre-approved by principal
- Truancy
- Any absence without documentation

ABSENCE REPORTING
Parents must notify the school before 9:00 AM on the day of absence via:
- SIS Parent Portal
- School attendance hotline: (555) 100-3000
- Email to attendance@school.district42.edu

Students must provide written documentation (signed by parent/guardian) within 2 days of return.

TARDY POLICY
3 unexcused tardies = 1 unexcused absence.
Chronic tardiness triggers a conference with the school counselor.

TRUANCY PROTOCOL
After 3 unexcused absences:
1. Counselor contacts parent/guardian
After 5 unexcused absences:
2. Formal attendance review meeting required
After 10 unexcused absences:
3. Referral to district attendance officer and possible involvement of county truancy court

MAKE-UP WORK
Students have one day per each day absent to complete make-up work (up to 5 days maximum).

CHRONIC ABSENTEEISM
Defined as missing 10% or more of school days. Triggers a Student Support Team (SST) meeting.

REPORTING
Attendance is recorded daily in the SIS by each teacher. Attendance reports are available to parents via the SIS portal in real-time.""",
    },
    {
        "title": "Grading and Assessment Policy",
        "topic": "grading",
        "doc_type": "policy",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """District 42 Grading and Assessment Policy (Policy 5124)

PURPOSE
Establish consistent, fair grading standards across all District 42 schools.

GRADING SCALE (Secondary)
A: 90-100% | Excellent
B: 80-89%  | Above Average
C: 70-79%  | Average
D: 60-69%  | Below Average
F: 0-59%   | Failing

GRADE POINT AVERAGE (GPA)
A = 4.0, B = 3.0, C = 2.0, D = 1.0, F = 0.0
Weighted GPA applies to AP/IB/Dual Enrollment courses (+0.5 per course).

GRADE COMPOSITION
Teachers must clearly communicate grading weights in the course syllabus at the start of each semester. Typical breakdown:
- Major assessments (tests, projects): 40-60%
- Minor assessments (quizzes, homework): 20-30%
- Participation/classwork: 10-20%

GRADE REPORTING
- Progress reports issued at 4.5 weeks
- Quarter/Semester grades issued within 5 business days of grading period close
- Parents and students can view grades in real-time via SIS portal

INCOMPLETE GRADES
An "I" (Incomplete) may be assigned when a student misses assessments due to excused extended absence. Incomplete work must be completed within 30 days or the grade converts to the earned grade (which may be an F for missing work).

GRADE CHALLENGES
Students or parents may challenge a grade within 10 school days of grade posting.
Process:
1. Request a meeting with the teacher
2. If unresolved, escalate to department head
3. Final appeal to school principal

ACADEMIC DISHONESTY
First offense: Zero on assessment, parent notification, written warning.
Second offense: Zero, parent conference, possible course failure.
Third offense: Referred to principal, potential suspension.

PROMOTION AND RETENTION
Middle school: Students failing 2 or more core courses may be retained with SST approval.
High school: Students must earn the required credits per grade level.
Retention requires written notification to parents by April 1.

HONOR ROLL
Principal's List: 4.0 GPA
Honor Roll: 3.5 GPA
Merit Roll: 3.0 GPA""",
    },
    {
        "title": "FERPA Compliance Guide",
        "topic": "ferpa_compliance",
        "doc_type": "compliance_guide",
        "source": "synthetic",
        "sensitivity": "confidential",
        "content": """FERPA Compliance Guide for District 42 Staff
Family Educational Rights and Privacy Act (20 U.S.C. § 1232g)

OVERVIEW
FERPA protects the privacy of student education records. It applies to all schools receiving federal funding. Violations can result in loss of federal funding.

STUDENT RIGHTS UNDER FERPA
1. Right to inspect and review education records within 45 days of request
2. Right to request amendment of inaccurate records
3. Right to consent before disclosure (with exceptions)
4. Right to file a complaint with the U.S. Department of Education

WHO HAS ACCESS TO STUDENT RECORDS?
Without consent:
- The student (if age 18+ or in college)
- Parents/guardians (if student is under 18 and not in college)
- School officials with legitimate educational interest
- Transferring schools (upon request)
- Accreditation organizations
- Financial aid authorities (limited)
- Judicial orders or subpoenas (with notification to student/parent)

DIRECTORY INFORMATION
District 42 has designated the following as directory information (may be released without consent unless parent opts out):
- Student name, address, phone number
- Date and place of birth
- Grade level
- Participation in officially recognized activities

Parents may opt out of directory information sharing by submitting a written request by September 30 each year.

DISCLOSURE WITH CONSENT
All other disclosures require signed, written consent specifying:
- Records to be disclosed
- Purpose of disclosure
- Party to whom disclosure is made

SPECIAL CIRCUMSTANCES
Health/Safety Emergency: Records may be disclosed without consent if necessary to protect the health or safety of the student or others. Document the emergency and reasoning.
Law Enforcement: Student records may be disclosed to the school's law enforcement unit. However, law enforcement records created by that unit are NOT education records under FERPA.

STAFF OBLIGATIONS
- Access only records needed for your educational role
- Never share student information via personal email or unsecured channels
- Verify identity before providing information over phone
- Log all non-routine disclosures in the FERPA disclosure log
- Report suspected violations immediately to the Privacy Officer

PENALTIES FOR VIOLATION
- Written reprimand
- Mandatory retraining
- Suspension
- Termination
- District loss of federal funding

FERPA Privacy Officer: privacy@district42.edu | (555) 100-1000""",
    },
    {
        "title": "Official Transcript Request Procedures",
        "topic": "transcripts",
        "doc_type": "procedure",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Official Transcript Request Procedures (Procedure 5125-P)

TYPES OF TRANSCRIPTS
1. Unofficial Transcript: Available immediately via SIS student portal. Stamped "UNOFFICIAL." Acceptable for personal use, scholarship applications (check requirements).
2. Official Transcript: Sealed envelope with registrar signature or electronic transmission via Parchment. Required for college applications, employment, graduate school.

REQUESTING AN OFFICIAL TRANSCRIPT
Current Students:
1. Log into SIS student portal
2. Navigate to Records > Transcript Request
3. Enter recipient name, address, and purpose
4. Pay processing fee: $5.00 per copy (fee waived for college applications for students receiving free/reduced lunch)
5. Allow 3-5 business days for processing

Alumni:
1. Submit form via district website: www.district42.edu/alumni-transcripts
2. Include signed photo ID copy
3. Processing fee: $10.00 per copy (check, money order, or online payment)
4. Allow 7-10 business days

RUSH TRANSCRIPTS
Available for an additional $15.00 fee. Same-day if requested before 10:00 AM. Next-day guaranteed. Available for current students and recent alumni (within 2 years of graduation) only.

ELECTRONIC TRANSCRIPTS
District 42 uses Parchment for secure electronic delivery. Available to most US colleges and universities. No additional fee.

HOLD ON TRANSCRIPT RELEASE
Transcripts will not be released if the student has:
- Outstanding textbook fines over $50
- Unreturned library books
- Financial obligations (athletic equipment, activity fees)

Students are notified of holds via the SIS portal. Holds must be cleared before transcript release.

FERPA AUTHORIZATION
Official transcripts are released only to:
- The student (age 18+)
- Parent/guardian (for students under 18)
- Third parties with signed FERPA release form on file

SPECIAL CIRCUMSTANCES
Special education transcripts must be redacted to remove confidential disability information unless explicitly authorized.
Grade records for students who were victims of crimes must be handled per the FERPA safety disclosure protocol.

CONTACT
Registrar's Office: registrar@district42.edu | (555) 100-4000 | Hours: M-F 8:00 AM - 4:30 PM""",
    },
    {
        "title": "Special Education and Accommodations Handbook",
        "topic": "accommodations",
        "doc_type": "handbook",
        "source": "synthetic",
        "sensitivity": "confidential",
        "content": """Special Education and Accommodations Handbook (Policy 6159)

LEGAL FRAMEWORK
- Individuals with Disabilities Education Act (IDEA)
- Section 504 of the Rehabilitation Act
- Americans with Disabilities Act (ADA)
- Every Student Succeeds Act (ESSA)

TYPES OF PLANS
Individualized Education Program (IEP):
For students with qualifying disabilities who need specialized instruction. Governed by IDEA.
Key components: Present levels, annual goals, services, placement, transition plan (age 16+).
Review: Annually, with triennial reevaluation.

Section 504 Plan:
For students with disabilities that substantially limit a major life activity but who do not need specialized instruction. Governed by Section 504.
Examples: Extended time, preferential seating, testing accommodations, attendance modifications.
Review: Annually.

REFERRAL PROCESS
1. Teacher or parent submits referral to Student Support Team (SST)
2. SST reviews data and determines if evaluation is warranted
3. If yes, written notice to parents and consent obtained
4. Multidisciplinary evaluation completed within 60 calendar days
5. Eligibility determined by team meeting
6. If eligible, IEP or 504 developed within 30 days of eligibility determination

COMMON ACCOMMODATIONS
Testing: Extended time (1.5x or 2x), separate testing room, read-aloud, scribe, calculator use
Instruction: Preferential seating, copy of notes, graphic organizers, chunked assignments, sensory breaks
Environmental: Low-stimulation workspace, fidget tools, noise-canceling headphones

CONFIDENTIALITY
IEP and 504 records are education records under FERPA. Access is limited to staff with legitimate educational interest. Records may NOT be shared with other students, parents of other students, or non-school parties without written consent.

PARENT RIGHTS
- Receive prior written notice before any change in identification, evaluation, or placement
- Participate in all IEP/504 team meetings
- Request an Independent Educational Evaluation (IEE) at public expense if disagreeing with district evaluation
- File a due process complaint or state complaint

MANIFESTATION DETERMINATION
Before a student with a disability is suspended more than 10 school days, a manifestation determination review must be conducted to determine if the behavior was caused by or substantially related to the disability.

CONTACT
Special Education Department: sped@district42.edu | (555) 100-5000""",
    },
    {
        "title": "Student Disciplinary Procedures",
        "topic": "disciplinary",
        "doc_type": "procedure",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Student Disciplinary Procedures (Policy 5144)

PHILOSOPHY
District 42 is committed to restorative practices. Discipline is designed to be corrective and instructional, not punitive.

LEVELS OF OFFENSES

Level 1 – Minor Infractions (Teacher-Managed):
- Classroom disruption, tardiness, dress code violation, phone use
Response: Teacher redirect, verbal warning, parent contact, detention

Level 2 – Moderate Infractions (Administrator-Managed):
- Defiance, repeated Level 1 violations, minor physical altercation, bullying
Response: 1-3 day suspension (in-school or out-of-school), parent conference, referral to counselor

Level 3 – Serious Infractions (Administrator + District):
- Fighting, harassment, vandalism, theft, substance use
Response: 3-10 day suspension, possible recommendation for expulsion, referral to alternative placement

Level 4 – Expellable Offenses (Zero Tolerance):
- Possession of weapons
- Assault on staff
- Sale/distribution of controlled substances
Response: Long-term suspension pending expulsion hearing before the Board of Education

DUE PROCESS
All suspended students have the right to:
- Informal hearing with administrator before suspension
- Written notice of suspension (mailed same day)
- Opportunity to respond to charges

Expulsion hearings include right to:
- Formal hearing with Board panel
- Present evidence and witnesses
- Appeal to Superintendent within 10 days

STUDENTS WITH DISABILITIES
Suspensions over 10 days require manifestation determination review (see Accommodations Handbook).

RECORDS
Disciplinary records are maintained in SIS and considered education records under FERPA.
Long-term suspensions and expulsions are noted on the student's official record.
Minor disciplinary records are purged after 3 years.

BULLYING AND HARASSMENT
Investigated within 3 school days of report. Investigators must be trained in trauma-informed practices. Retaliation against reporters is strictly prohibited.

RESTORATIVE PRACTICES
Available as alternative to suspension for Level 1-2 offenses. Requires agreement from all parties. Facilitator is trained counselor or administrator.""",
    },
    {
        "title": "Parent and Family Communication Protocol",
        "topic": "parent_communication",
        "doc_type": "policy",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Parent and Family Communication Protocol (Policy 1310)

PURPOSE
Establish clear, consistent, and equitable communication practices between District 42 and families.

COMMUNICATION CHANNELS
Primary:
- SIS Parent Portal: Grades, attendance, announcements (real-time, 24/7 access)
- SchoolMessenger: Automated calls, texts, and emails for district-wide announcements

Secondary:
- School website: www.school.district42.edu
- Teacher websites (linked from school site)
- School newsletters (bi-weekly)

Emergency:
- SchoolMessenger emergency broadcast (within 15 minutes of incident)
- District emergency website
- Local media channels

TEACHER RESPONSE TIME
Teachers must respond to parent communications (email, phone, portal message) within 2 school days.

TRANSLATION SERVICES
The district provides translation services in all languages spoken by 5% or more of the student population. Request translation at enrollment or any time through the Parent Engagement Office.
For immediate translation needs, call: (555) 100-7000.

RECORDS AND PRIVACY
All parent communications regarding a student's education records must comply with FERPA. Staff may not share student information with a parent of another student.

Divorced/Separated Parents: Both legal parents/guardians have equal rights to school information unless there is a court order on file limiting access. Courts orders must be on file with the registrar.

PARENT CONFERENCE PROTOCOL
Formal conferences must be held twice per year.
Informal conferences available upon request; scheduled within 5 school days.
Virtual conference options available.

NON-CUSTODIAL PARENT ACCESS
Non-custodial parents retain FERPA rights unless a court order specifying otherwise is on file. The court order must be submitted to the registrar annually.

COMPLAINTS AND GRIEVANCES
Informal complaints: Direct to teacher or principal within 10 school days of the event.
Formal complaints: Submit in writing to the Superintendent's office using Form 1312.
Timeline: District must respond within 15 school days of formal complaint.

PARENT PORTAL ACCESS
All families receive SIS portal credentials at enrollment. Portal access includes: grades, attendance, assignments, transcripts (unofficial), health records, and messaging.
Technical support: portal@district42.edu | (555) 100-6000""",
    },
    {
        "title": "Transfer Student Procedures",
        "topic": "transfer",
        "doc_type": "procedure",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Transfer Student Procedures (Procedure 5117-P)

INTRA-DISTRICT TRANSFERS
A student may request a transfer to a different school within District 42 for the following reasons:
- Program availability (specialized programs not offered at home school)
- Documented safety concerns (bully/victim situations)
- Childcare provider is in another school's zone
- Sibling already attending preferred school

PROCESS FOR INTRA-DISTRICT TRANSFER:
1. Parent submits Transfer Request Form via SIS portal (by March 15 for next year; by June 15 for August start)
2. Receiving school principal approves or denies within 15 school days
3. Transfer is conditional on space availability in the requested grade
4. Transportation is not provided for intra-district transfers

INTER-DISTRICT TRANSFERS (In-Bound)
Students from outside District 42 may request enrollment. Approval required from both districts.
Required documents: Same as standard enrollment + release form from resident district.
Timeline: Decision within 30 days.

INTER-DISTRICT TRANSFERS (Out-Bound)
District 42 students may request to attend a school in another district.
Process: Parent submits Form 5117 to the Superintendent's office.
Factors considered: Capacity of receiving district, educational program, student's academic history.
Timeline: Decision within 30 days.

CREDIT TRANSFER AND EVALUATION
Transfer credits are evaluated by the registrar and counselor within 10 days.
Accredited school credits: Accepted at face value.
Non-accredited school / homeschool credits: Evaluated on case-by-case basis; may require proficiency assessment.

SPECIAL EDUCATION TRANSFER
Students with IEPs: Receiving school must provide comparable services within 10 business days. A new IEP meeting must be held within 30 days.

SPORTS ELIGIBILITY
Transfers may affect athletic eligibility per state athletic association rules. Contact the Athletic Director immediately.

RECORDS REQUESTED FROM PREVIOUS SCHOOL
District 42 will request records within 24 hours of enrollment. If records are not received within 30 days, a follow-up request is sent. Student may attend while waiting for records.

WITHDRAWAL FROM DISTRICT 42
Parent must submit written withdrawal notice.
School will prepare records for release within 5 school days.
If student is under 18, withdrawal requires parent signature. Truancy concerns will be reported if appropriate.""",
    },
    {
        "title": "Student Records Management Policy",
        "topic": "records_management",
        "doc_type": "policy",
        "source": "synthetic",
        "sensitivity": "confidential",
        "content": """Student Records Management Policy (Policy 5125)

PURPOSE
Govern the creation, maintenance, access, retention, and destruction of student records in compliance with FERPA, state law, and district requirements.

TYPES OF STUDENT RECORDS
Education Records: Any records directly related to a student and maintained by the district.
Directory Information: See FERPA Compliance Guide (designated subset, opt-out available).
Sole Possession Records: Notes made by staff for their own use, not shared with others — NOT subject to FERPA.
Law Enforcement Records: Records created by the school's law enforcement unit — NOT subject to FERPA.

RECORD KEEPING SYSTEMS
The district maintains student records in:
1. SIS (Student Information System) — primary system for all active student records
2. Physical files at school sites — for documents requiring original signatures
3. Health Records System — maintained separately by school nurse, limited access

ACCESS CONTROLS
Who may access records:
- School officials with legitimate educational interest
- Student (18+ or enrolled in post-secondary)
- Parent/guardian (student under 18, not in post-secondary)
- Authorized third parties (with consent or legal exception)

Access is logged automatically by the SIS. Unusual access patterns are reviewed by the Privacy Officer quarterly.

RECORD RETENTION SCHEDULE
Permanent Records (keep forever): Transcripts, diplomas, graduation records
Long-term Records (keep 7 years after last enrollment): Health records, IEP/504 records, disciplinary records (Level 3+)
Standard Records (keep 3 years after graduation/last enrollment): Attendance, grades, general correspondence
Temporary Records (keep 1 year): Interim assessments, informal notes

RECORD DESTRUCTION
Physical records: Shredded via certified destruction vendor (Certificate of Destruction on file).
Electronic records: Purged from active SIS; backed up archives deleted per retention schedule.
Destruction logs maintained by the Records Manager.

SUBPOENAS AND LEGAL REQUESTS
All subpoenas or court orders for student records must be forwarded to the district's legal counsel immediately. Generally, district will attempt to notify parent/student before complying unless order prohibits notification.

DATA BREACH RESPONSE
Report suspected data breach to: dataprivacy@district42.edu within 24 hours.
District will notify affected families within 72 hours per state law.

RECORDS REQUESTS FROM PARENTS
Must be fulfilled within 45 days of written request.
District may charge reasonable copying fees (not to exceed $0.10/page for paper; no fee for electronic).
If records cannot be located, written notice must be provided within 45 days.""",
    },
    {
        "title": "FAQ: Enrollment and Registration",
        "topic": "enrollment",
        "doc_type": "faq",
        "source": "synthetic",
        "sensitivity": "public",
        "content": """Frequently Asked Questions: Enrollment and Registration

Q: When does enrollment open for the upcoming school year?
A: Open enrollment for the next school year begins March 1 and closes April 30. Families who miss the open enrollment window may register at any time; they will be placed on a waitlist for popular programs.

Q: Can I enroll my child online?
A: Yes. All enrollment forms are available on the SIS Parent Portal at portal.district42.edu. Required documents can be uploaded digitally. If you do not have internet access, visit the District Enrollment Office during business hours.

Q: What if I don't have a birth certificate?
A: You may submit an affidavit of age, a passport, or a baptismal record as alternative proof of age. A birth certificate must be submitted within 30 days.

Q: Does my child need to be fully vaccinated to enroll?
A: Students must meet state immunization requirements. Medical and religious exemptions are available. Contact the school nurse for the exemption form.

Q: How do I update my address after enrollment?
A: Address changes can be submitted via the SIS Parent Portal under Account Settings > Address Update. Proof of new residency must be re-submitted. If the address change moves the student to a different school zone, a transfer will be processed automatically.

Q: My child has a 504 plan. How do I make sure it is honored at the new school?
A: Include a copy of the 504 plan in the enrollment packet. The counselor will schedule a 504 meeting within 15 school days to confirm or update accommodations.

Q: Can grandparents or other relatives enroll a child?
A: Legal guardians may enroll a student. A guardianship or caregiver affidavit (notarized) must be on file. For McKinney-Vento situations (homeless, doubled-up housing), any adult responsible for the child may enroll.

Q: What happens if my child is denied enrollment?
A: The denial will be provided in writing with the reason and your appeal rights. You may appeal within 10 school days to the Superintendent's office.""",
    },
    {
        "title": "FAQ: Grades and Transcripts",
        "topic": "transcripts",
        "doc_type": "faq",
        "source": "synthetic",
        "sensitivity": "public",
        "content": """Frequently Asked Questions: Grades and Transcripts

Q: How can I access my child's grades?
A: Grades are available in real-time via the SIS Parent Portal. Log in at portal.district42.edu with your parent account credentials.

Q: What is the difference between a progress report and a report card?
A: Progress reports are informal mid-quarter summaries (not stored on official transcript). Report cards are issued at the end of each grading period and are recorded on the official transcript.

Q: My child received an Incomplete. What does that mean?
A: An Incomplete (I) is assigned when a student misses major assessments due to excused absence. The student has 30 days to complete the missing work. If not completed, the I converts to the earned grade.

Q: How do I request an official transcript for a college application?
A: Log in to the SIS portal > Records > Transcript Request. Select "Official - Electronic (Parchment)" for direct college delivery. Allow 3-5 business days.

Q: Is there a fee for transcripts?
A: Current students: $5.00 per official copy ($0 for students on free/reduced lunch for college applications). Electronic delivery via Parchment is free.

Q: My child's grade appears to be incorrect. What is the appeal process?
A: Contact the teacher first within 10 school days of the grade being posted. If unresolved, contact the department head, then the principal. All appeals must be initiated within 10 school days.

Q: Can colleges see my disciplinary record?
A: Disciplinary records are separate from academic transcripts. Some colleges request disciplinary information separately on their applications. Districts only release disciplinary records with student/parent consent unless required by law.

Q: How long does the school keep my child's records?
A: Permanent academic records (transcripts, diplomas) are kept forever. Other records follow a retention schedule ranging from 1-7 years depending on record type.""",
    },
    {
        "title": "IEP Process and Parent Guide",
        "topic": "accommodations",
        "doc_type": "guide",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """IEP Process and Parent Guide: District 42 Special Education

WHAT IS AN IEP?
An Individualized Education Program (IEP) is a legally binding document that describes the special education services and supports a student with a disability will receive. It is developed collaboratively by the IEP team, which includes the parent.

THE IEP TEAM
Required members:
- Parent/guardian
- General education teacher
- Special education teacher
- School psychologist or evaluator (at initial meeting)
- Administrator or LEA (Local Education Agency) representative
- Student (when appropriate, particularly at age 14+)

WHAT IS IN AN IEP?
1. Present Levels of Academic Achievement and Functional Performance (PLAAFP)
2. Annual Goals (measurable goals in each area of need)
3. Special Education Services (type, frequency, location)
4. Supplementary Aids and Services (in-class supports)
5. Accommodations (testing, instruction, environment)
6. Participation in General Education (how much time in regular classes)
7. State Assessment Participation (standard, alternate, or exempt)
8. Transition Services (starting at age 16): goals for post-secondary education, employment, independent living

THE IEP MEETING
You will receive written notice at least 10 calendar days before the meeting.
You have the right to participate meaningfully in the meeting.
You may bring a support person (friend, advocate, interpreter).
The meeting must be held at a mutually convenient time and place.

PARENT RIGHTS SUMMARY
Prior Written Notice: You must receive written notice before the district proposes or refuses to change your child's identification, evaluation, placement, or services.
Consent: You must provide written consent before the initial evaluation, initial placement, and some service changes.
Dispute Resolution: If you disagree with the IEP or placement, you may request mediation, a due process hearing, or file a state complaint. See the full Procedural Safeguards notice provided annually.

IEP REVIEW SCHEDULE
Annual: Full IEP review meeting required every 12 months.
Triennial: Full reevaluation every 3 years (or sooner if requested).
As Needed: IEP may be amended at any time with parent consent.

PROGRESS REPORTING
Progress on IEP goals is reported to parents as frequently as progress is reported for general education students (quarterly with report cards).

CONTACT
Special Education Coordinator: sped@district42.edu | (555) 100-5000""",
    },
    {
        "title": "Disciplinary FAQ and Student Rights",
        "topic": "disciplinary",
        "doc_type": "faq",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """FAQ: Student Discipline and Rights

Q: What are my rights if my child is suspended?
A: Students have the right to an informal hearing with an administrator before any suspension. Written notice must be mailed on the day of suspension. For suspensions over 5 days, parents may request a formal review.

Q: Can my child be suspended for posting something on social media?
A: Off-campus speech may be subject to discipline if it causes substantial disruption to school operations or threatens members of the school community. Each case is evaluated individually.

Q: What is in-school suspension (ISS) vs. out-of-school suspension (OSS)?
A: ISS: Student reports to a supervised room at school; instruction continues but separately from peers. OSS: Student is excluded from school grounds. Academic work must still be provided.

Q: How is bullying investigated?
A: All bullying reports are investigated within 3 school days by a trained investigator. Both the alleged perpetrator and reporter are interviewed. Parents of both students are notified. Retaliation is prohibited.

Q: What happens to my child's IEP during a suspension?
A: If suspended more than 10 cumulative school days, the district must hold a manifestation determination review (MDR). If the behavior is related to the disability, the student cannot be expelled unless the student brought a weapon, had drugs, or caused serious bodily injury.

Q: Can my child be searched at school?
A: School officials may search students with reasonable suspicion that school rules or laws have been violated. Locker searches may occur at any time. Searches must be reasonable in scope.

Q: How do I report a discipline concern?
A: Contact the school principal first. If unresolved, contact the Superintendent's Office at superintendent@district42.edu.""",
    },
    {
        "title": "Attendance FAQ and Truancy Prevention",
        "topic": "attendance",
        "doc_type": "faq",
        "source": "synthetic",
        "sensitivity": "public",
        "content": """FAQ: Attendance and Truancy Prevention

Q: How many absences are allowed before action is taken?
A: After 3 unexcused absences, the counselor contacts parents. After 5 unexcused absences, a formal attendance review is required. After 10 unexcused absences, the case is referred to the district attendance officer.

Q: What is the difference between an excused and unexcused absence?
A: Excused absences include illness (with documentation for 3+ days), medical appointments, family emergencies, religious observances (up to 2 days), court appearances, and pre-approved activities. All other absences are unexcused.

Q: What happens if my child misses school due to a long illness?
A: A doctor's note is required for absences of 3 or more consecutive days due to illness. For extended illness (7+ days), contact the school nurse to arrange homebound instruction.

Q: What is chronic absenteeism?
A: Missing 10% or more of school days (approximately 18 days). Chronically absent students receive a Student Support Team meeting to identify and address barriers.

Q: Can attendance affect my child's grades?
A: Yes. Some courses have attendance requirements. Excessive unexcused absences can also affect grades per teacher policy (outlined in the course syllabus).

Q: What if my family needs to travel during the school year?
A: Request a pre-approved educational absence from the principal at least 5 school days in advance for trips up to 5 days. Longer trips are generally classified as unexcused. Students must complete work in advance or upon return.

Q: I have a court date but my child needs to attend. Will it be excused?
A: Yes. Bring documentation of the court appearance. Court-ordered attendance is always excused. Legal proceedings related to the student are handled with confidentiality.""",
    },
]


# ============================================================
# MFG (Manufacturing) documents
# ============================================================

MFG_DOCUMENTS = [
    {
        "title": "SOP-001: Equipment Startup Procedure",
        "topic": "standard_operating_procedures",
        "doc_type": "sop",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Standard Operating Procedure: Equipment Startup
SOP Number: MFG-SOP-001 | Rev: 4 | Effective: 2024-01-15

PURPOSE
Define the steps required to safely start production equipment at the beginning of each shift, following any maintenance, or after an unplanned shutdown.

SCOPE
Applies to all production line operators on lines 1-6 and CNC machinists. Supervisors are responsible for ensuring operator compliance.

PRE-REQUISITES
- Operator must be trained and certified on the specific equipment (training record in HR system)
- PPE required: safety glasses, steel-toed boots, hearing protection (above 85 dB areas)
- LOTO authorization tag (if applicable — see LOTO procedure SOP-012)

PROCEDURE

Step 1: Pre-Startup Safety Check
1.1 Verify the machine is clear of all personnel, tools, and loose materials.
1.2 Inspect all guards and safety devices — they must be in place and functional.
1.3 Check for any "DO NOT OPERATE" or LOTO tags. If present, DO NOT START the machine. Contact the Maintenance Supervisor immediately.
1.4 Inspect fluid levels: hydraulic, coolant, lubricant (see equipment-specific checklist).
1.5 Verify emergency stop (E-stop) buttons are functional. Test by pressing and releasing.

Step 2: Power-On Sequence
2.1 Enable main power at the panel (switch must click into position).
2.2 Wait for the HMI display to fully initialize (approximately 45 seconds).
2.3 Verify no active fault codes on HMI. Clear faults only if authorized.

Step 3: Warm-Up Cycle
3.1 Run the warm-up macro (Program WU-001 on CNC) for 5 minutes minimum.
3.2 Do not load production materials during warm-up.
3.3 Monitor temperature, pressure, and RPM gauges for normal operating range.

Step 4: First-Article Inspection
4.1 Run one unit before starting full production.
4.2 Inspect against the part drawing or inspection checklist.
4.3 If unit passes inspection, log first-article in the QC log and begin production.
4.4 If unit fails, notify Quality Control and do NOT start production until resolved.

Step 5: Shift Log Entry
5.1 Record startup time, operator name, machine ID, and first-article result in the Digital Production Log.

EMERGENCY PROCEDURES
If any abnormal noise, smell, or vibration occurs: Press E-stop immediately. Evacuate area if fire or chemical hazard. Call extension 911 (internal). Do NOT attempt to diagnose or repair.

RELATED DOCUMENTS
SOP-002 (Shutdown Procedure), SOP-012 (LOTO), SOP-020 (Emergency Response)

REVISION HISTORY
Rev 4: Added HMI fault code step 2.3. Approved: Safety Manager 2024-01-15""",
    },
    {
        "title": "Quality Control Inspection Handbook",
        "topic": "quality_control",
        "doc_type": "handbook",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Quality Control Inspection Handbook
Revision 6 | Effective Date: 2024-03-01

1. SCOPE
This handbook governs all incoming, in-process, and final inspection activities at Facility 12. Compliance is mandatory for all QC technicians, production operators, and supervisors.

2. INSPECTION TYPES

2.1 Incoming Inspection
All raw materials and purchased components must be inspected before release to production.
- Random sampling per AQL Level II (ANSI/ASQ Z1.4-2008)
- Dimensional verification against Drawing/Specification
- Material certification review (Certificate of Conformance)
- Results recorded in ERP system within 4 hours of receipt

2.2 In-Process Inspection
- Frequency: Every 50 units (or as specified in Control Plan)
- Operator self-inspection: First and last unit of each run
- QC Technician verification: Start of shift, after material changes, after machine adjustment
- SPC charts maintained for critical characteristics
- If out-of-control signal detected on SPC: Stop production, notify Quality Engineer

2.3 Final Inspection
- 100% inspection for critical-to-function (CTF) dimensions
- Sample inspection for standard dimensions per AQL
- Functional testing per Test Specification
- Cosmetic inspection per Defect Classification (see Section 5)
- Results recorded in QC database; Certificate of Conformance issued for release

3. MEASUREMENT EQUIPMENT
All gages must be within calibration date (sticker on gage).
Never use an out-of-calibration gage. Tag and remove from service immediately.
Calibration records maintained by the Metrology Lab (ext. 4401).
Gage R&R must be performed annually for all critical measurement equipment.

4. NONCONFORMING MATERIAL
If nonconforming material is identified:
1. Stop production (if in-process nonconformance)
2. Quarantine the material with a RED nonconformance tag
3. Enter NCR (Nonconformance Report) in the QC system within 1 hour
4. Notify the Quality Engineer and Production Supervisor
5. Await disposition decision before further processing

Disposition options: Use As Is (QE approval required), Rework, Scrap, Return to Vendor.

5. DEFECT CLASSIFICATION
Critical Defect: Could cause injury, safety hazard, or field failure. ZERO tolerance. All units in lot are quarantined.
Major Defect: Likely to cause product failure or customer dissatisfaction. Lot disposition required.
Minor Defect: Unlikely to affect function, small cosmetic issue. May be accepted per sampling plan.

6. MEASUREMENT SYSTEM ANALYSIS
Gage R&R acceptance criteria: Total Gage R&R < 10% (acceptable), 10-30% (conditional), >30% (unacceptable — gage must be replaced or repaired).

7. DOCUMENTATION
All inspections must be documented — undocumented inspection is no inspection. Electronic records in QC system; paper backup for critical dimensions maintained for 10 years.""",
    },
    {
        "title": "CAPA Process Guide",
        "topic": "capa",
        "doc_type": "guide",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Corrective Action and Preventive Action (CAPA) Process Guide
Document: QMS-CAPA-001 | Rev: 5

1. PURPOSE
Define the process for identifying, investigating, and resolving quality escapes, customer complaints, internal nonconformances, and audit findings through a systematic corrective and preventive action process.

2. WHEN TO INITIATE A CAPA
- Customer complaints (severity: Major or Critical)
- Internal nonconformances (recurring or systemic)
- Supplier-related defects escaping to production
- Audit findings (internal or external)
- Product returns and warranty claims
- Significant process excursions

Minor, isolated one-time issues may be resolved via Quick Fix (QF) without a full CAPA.

3. CAPA PHASES

Phase 1: Problem Definition (Due: Day 1)
- Describe the defect/issue clearly using the 5W1H format (Who, What, When, Where, Why, How)
- Quantify scope: How many affected? What time period? What customers?
- Enter CAPA record in QMS system; assign CAPA number

Phase 2: Containment (Due: Day 2)
- Immediately prevent escape of defective product to customers
- Actions: 100% inspection, sorting, customer notification (if product escaped), temporary hold
- Containment effectiveness verified by Quality Engineer
- Document containment actions and results in CAPA record

Phase 3: Root Cause Analysis (Due: Day 10)
Methods (use the appropriate method based on complexity):
- 5-Why Analysis (for simple, single-cause issues)
- Fishbone / Ishikawa Diagram (for multi-cause issues)
- Fault Tree Analysis (for complex or safety-critical issues)
- Design of Experiments (for process parameter issues)

Root cause must be verified — do not proceed to corrective action until root cause is confirmed.

Phase 4: Corrective Action Plan (Due: Day 15)
- Define specific actions to eliminate the root cause
- Assign owners and due dates for each action
- Actions should be permanent (process change, design change, training revision)
- Avoid temporary or inspection-based corrective actions

Phase 5: Implementation and Verification (Due: Day 30)
- Implement all corrective actions
- Collect objective evidence (photos, data, updated procedures)
- Verify effectiveness: Run production, collect data, confirm defect rate returns to acceptable level
- If corrective action is ineffective, return to Phase 3

Phase 6: Closure and Preventive Action (Due: Day 45)
- Document lessons learned
- Identify similar processes or products that may have the same root cause
- Update FMEAs, control plans, work instructions as needed
- Share learnings with relevant teams

4. RESPONSIBILITIES
CAPA Owner: Assigned at opening; accountable for meeting all due dates.
Quality Engineer: Verifies root cause and corrective action effectiveness.
Quality Manager: Approves closure of all CAPAs.

5. METRICS
Open CAPAs by age (target: 80% closed within 45 days)
On-time closure rate (target: >90%)
Recurrence rate (target: <5%)""",
    },
    {
        "title": "Safety Protocols and Lockout/Tagout Procedure",
        "topic": "safety_protocols",
        "doc_type": "procedure",
        "source": "synthetic",
        "sensitivity": "confidential",
        "content": """Safety Protocols and Lockout/Tagout (LOTO) Procedure
Document: EHS-SOP-012 | Rev: 8 | OSHA 29 CFR 1910.147 Compliant

WARNING: FAILURE TO FOLLOW LOTO PROCEDURES CAN RESULT IN SERIOUS INJURY OR DEATH. THERE ARE NO EXCEPTIONS TO LOTO REQUIREMENTS.

1. PURPOSE
Protect workers from the unexpected energization, startup, or release of stored energy during servicing and maintenance of equipment.

2. SCOPE
Required ANY TIME a worker is exposed to:
- Electrical energy (during electrical work or contact with energized circuits)
- Mechanical motion (removing guards, working in machine zones)
- Hydraulic/pneumatic energy (breaking into pressurized systems)
- Thermal energy (working on hot surfaces or steam systems)
- Chemical energy (working with pressurized chemical systems)
- Gravitational energy (working under suspended loads)

3. AUTHORIZED EMPLOYEES
Only employees who have completed LOTO training and are listed on the authorized employee list may apply energy isolation devices. Training records maintained in HR system (LOTO-AUTH-LIST).

4. LOTO PROCEDURE — STEP BY STEP

Step 1: Notify affected employees that equipment is going down for service.
Step 2: Review the equipment's Energy Control Procedure (ECP) — posted on machine or available in LOTO binder.
Step 3: Identify ALL energy sources for the equipment (electrical, pneumatic, hydraulic, steam, gravity, spring-tension).
Step 4: Shut down equipment using normal stopping procedure.
Step 5: Isolate each energy source at each energy isolation point:
  - Electrical: Turn off and lock out at disconnect. Verify voltage with meter.
  - Pneumatic/Hydraulic: Close isolation valve, bleed pressure. Verify pressure gauge reads 0.
  - Steam/Thermal: Close isolation valve, allow to cool. Verify temperature.
Step 6: Apply your PERSONAL lockout lock (one per person). Write your name on the lock tag.
Step 7: Attempt to start machine — verify it does not start.
Step 8: Test for de-energization with appropriate testing equipment.
Step 9: Perform maintenance or service work.

RESTORATION AFTER SERVICE:
Step 1: Ensure all tools and materials are removed from machine.
Step 2: Ensure all guards and safety devices are reinstalled.
Step 3: Notify all personnel to clear the area.
Step 4: Remove YOUR lock only. Never remove another worker's lock.
Step 5: Restore energy sources in reverse order of isolation.
Step 6: Verify machine operates correctly.

5. GROUP LOCKOUT
When multiple workers service the same equipment:
- Each worker applies THEIR OWN personal lock to the hasp
- Machine cannot be restored until ALL workers remove their locks
- A group lockout coordinator is designated; records names of all workers

6. CONTRACTOR LOTO
Contractors must follow this procedure AND their own company's LOTO program. Facility Safety Manager coordinates with contractor supervisors before work begins.

7. LOTO VIOLATIONS
Any violation of LOTO procedure is grounds for immediate removal from work area and disciplinary action up to and including termination. Immediate supervisor is notified. EHS investigation initiated within 24 hours.

8. ANNUAL REVIEW
Authorized employees must review all energy control procedures annually. Review documented in HR training records.""",
    },
    {
        "title": "Preventive Maintenance Procedures",
        "topic": "maintenance",
        "doc_type": "procedure",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Preventive Maintenance Procedures
Document: MAINT-001 | Rev: 3

1. PURPOSE
Ensure equipment reliability and longevity through scheduled preventive maintenance, minimizing unplanned downtime and production losses.

2. PM SCHEDULE OVERVIEW
All PM tasks are generated by the CMMS (Computerized Maintenance Management System). Operators and maintenance technicians receive automated work orders 3 days before PM is due.

FREQUENCY CATEGORIES:
- Daily (D): Operator-performed checks at shift start
- Weekly (W): Maintenance technician inspection
- Monthly (M): Comprehensive inspection and lubrication
- Quarterly (Q): Calibration and precision adjustments
- Annual (A): Major overhaul, bearing replacement, full inspection

3. OPERATOR DAILY PM CHECKS (All Production Equipment)
Before first startup:
- Visual inspection: look for leaks, damage, missing guards
- Fluid levels: hydraulic, coolant (check sight glass; do not open reservoirs)
- Lubrication points: Apply grease per lube chart (posted on machine)
- Record findings in the equipment log (paper or digital)
- Report any abnormality to shift supervisor

4. MAINTENANCE WEEKLY INSPECTION
Maintenance technician performs:
- Vibration check on motors and gearboxes (handheld analyzer, record readings)
- Belt/chain tension check and adjustment
- Filter inspection (clean or replace per condition)
- Pneumatic/hydraulic system inspection (look for leaks, check pressure)
- Safety device function test (light curtains, E-stops)
- Review operator log for reported issues; follow up on all items

5. MONTHLY PM (Technician)
- Full lubrication per machine lube chart
- Coupling alignment check (laser alignment on critical machines)
- Electrical connection torque check
- Heat scan (infrared gun) on electrical panels and motors
- Coolant pH and concentration check; adjust or replace
- Air filter replacement (production areas)

6. FAILURE MODE TRACKING
All PM findings entered in CMMS under equipment ID. Recurring failures trigger a CAPA (see CAPA Process Guide). Maintenance Supervisor reviews CMMS data weekly for emerging trends.

7. EMERGENCY MAINTENANCE REQUEST
If production equipment fails during shift:
1. Operator presses E-stop and notifies supervisor
2. Supervisor calls maintenance dispatch: ext. 4400
3. Maintenance technician responds within 15 minutes (production areas)
4. If repair >2 hours, supervisor notifies production planner to reassign work

8. DOCUMENTATION
All PM activities must be documented in CMMS within 4 hours of completion. Incomplete PM work orders must have notes explaining delay and rescheduled date. PM completion rate tracked monthly (target: >95%).""",
    },
    {
        "title": "Production Scheduling Guidelines",
        "topic": "production_scheduling",
        "doc_type": "guideline",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Production Scheduling Guidelines
Document: PROD-SCH-001 | Rev: 2

1. PURPOSE
Establish guidelines for creating, communicating, and managing production schedules to meet customer delivery commitments while optimizing resource utilization.

2. SCHEDULING INPUTS
Customer Orders: Released to ERP by Customer Service. Orders with confirmed ship dates take priority.
Material Availability: Confirmed by Inventory Planner before scheduling into production.
Machine Capacity: Available hours per machine per shift (from capacity planning model).
Labor: Headcount plan from HR; certified operator assignments for specialized equipment.
PM and Downtime: Planned maintenance windows from CMMS; scheduled into calendar.

3. SCHEDULING HORIZON
Daily Schedule: Finalized by 3:00 PM for the following day.
Weekly Schedule: Published every Friday by 4:00 PM for the following week.
Monthly Forecast: Updated first business day of the month for the next 6 weeks.

4. PRIORITY RULES (in order)
1. Customer-designated CRITICAL orders (OTIF commitment)
2. Orders past due date
3. Orders at risk of becoming past due (considering lead time)
4. First-In, First-Scheduled (FIFS) for equal priority items

5. SCHEDULING PROCESS
Step 1: Production Planner downloads open orders from ERP (priority-sorted).
Step 2: Planner checks material availability in ERP; flags any shortages.
Step 3: Planner assigns orders to machines based on routing, capacity, and setup time.
Step 4: Supervisor reviews draft schedule for operator availability and practical constraints.
Step 5: Final schedule uploaded to ERP and distributed via email by 3:00 PM.
Step 6: Shop floor boards updated at each line by 3:30 PM.

6. CHANGE MANAGEMENT
Rush orders: Must be approved by Production Manager. Planner notifies affected customers of potential delays caused by the rush insertion.
Customer changes (quantity/date): Customer Service enters change in ERP; Planner evaluates impact within 2 hours. If change cannot be accommodated, Customer Service escalates to Sales Manager.
Machine downtime: If unplanned downtime occurs, Planner is notified within 15 minutes. Rerouting options evaluated; customer notification if delivery impact > 1 day.

7. KEY PERFORMANCE INDICATORS
On-Time-In-Full (OTIF): Target 98%
Schedule Attainment: Percentage of daily schedule completed as planned. Target: 90%
Machine Utilization: Target: 80% (production time / available time)

Reports published weekly in the KPI dashboard (SharePoint: Production > KPIs).""",
    },
    {
        "title": "Regulatory Compliance Reference Guide",
        "topic": "regulatory_compliance",
        "doc_type": "reference",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Regulatory Compliance Reference Guide
Facility 12 | Revision: 3 | Updated: 2024-02-01

1. APPLICABLE REGULATIONS AND STANDARDS
1.1 Occupational Safety and Health
- OSHA 29 CFR 1910 (General Industry): Machine guarding, hazard communication, PPE, LOTO, electrical, emergency action plans
- OSHA 29 CFR 1926: Applicable to construction activities on site
- State OSHA Plan: [State Name] Occupational Safety standards (equivalent or stricter than federal)

1.2 Environmental
- EPA Clean Air Act: VOC emissions from painting and coating operations. Air permit on file with EHS Manager.
- EPA Clean Water Act: Stormwater permit (NPDES). Spill Prevention, Control, and Countermeasure (SPCC) plan on file.
- RCRA (Resource Conservation and Recovery Act): Hazardous waste management and disposal.

1.3 Quality
- ISO 9001:2015: Quality Management System (certified). Certificate valid until 2026.
- IATF 16949: Automotive quality requirements (applicable to automotive product lines only).
- AS9100D: Aerospace requirements (applicable to aerospace product lines only).

1.4 Product Compliance
- RoHS: Restriction of Hazardous Substances (electronics components; customer-specified requirements)
- REACH: Registration, Evaluation, Authorization of Chemicals (EU export requirements)

2. COMPLIANCE MANAGEMENT
2.1 Permits and Licenses
All regulatory permits maintained by EHS Manager. Permit conditions are incorporated into standard operating procedures. Annual permit renewal calendar managed by EHS.

2.2 Training Requirements
New Employee: OSHA 10 (production), OSHA 30 (supervisors), Hazard Communication, PPE
Annual Refresher: LOTO, Emergency Response, Hazard Communication
Specialized: Forklift, Crane, Confined Space Entry, First Aid/CPR

2.3 Recordkeeping
OSHA 300 Log: Maintained by EHS. Posted Feb 1-Apr 30 annually.
Air Monitoring Records: Retained 30 years.
Hazardous Waste Manifests: Retained 3 years.
Training Records: Retained for duration of employment + 5 years.

3. REPORTING OBLIGATIONS
3.1 Injury/Illness Reporting
- Near misses and first aids: Reported same day to supervisor, logged in EHS system
- Recordable injuries: OSHA 300 log entry within 7 calendar days
- Hospitalization: OSHA notification within 24 hours (1-800-321-OSHA)
- Fatality: OSHA notification within 8 hours

3.2 Environmental Incidents
- Spills > reportable quantity: Notify EHS Manager immediately. EHS Manager contacts state and federal agencies per SPCC plan.
- Air permit exceedance: Report to state environmental agency within 2 business days.

4. THIRD-PARTY AUDITS
Customer Audits: Sales/Quality coordinates schedule. EHS participates as needed.
ISO/IATF Audits: Scheduled annually with registrar. QMS Manager owns preparation.
OSHA Inspections: Notify Legal and EHS Manager immediately. Cooperate professionally; do not volunteer information beyond what is requested.""",
    },
    {
        "title": "Defect Classification Taxonomy",
        "topic": "defect_classification",
        "doc_type": "reference",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Defect Classification Taxonomy
Document: QA-REF-003 | Rev: 4

PURPOSE
Provide a standardized classification system for product defects to ensure consistent evaluation, reporting, and disposition across all inspection activities.

CLASSIFICATION LEVELS

CRITICAL DEFECT (Class 1)
Definition: A defect that could cause injury, death, or product liability; a defect that renders the product unsafe for its intended use; a defect that creates a regulatory non-compliance.
Examples:
- Structural failure (crack, fracture, delamination in load-bearing component)
- Electrical fault (exposed wiring, wrong voltage, short circuit creating fire risk)
- Missing or incorrect safety label
- Contamination with hazardous material
- Dimension out of tolerance causing unsafe assembly
Action: 100% quarantine of all units in lot. CAPA required. Customer notification mandatory if product has shipped.

MAJOR DEFECT (Class 2)
Definition: A defect that is likely to cause product failure, significant customer dissatisfaction, or inability to perform intended function. Does not create immediate safety risk.
Examples:
- Dimension out of tolerance (functional impact, not safety)
- Wrong material or finish
- Incorrect part number used
- Significant cosmetic defect on a functional surface
- Missing or incorrect assembly
- Software/firmware defect causing incorrect operation
Action: Quarantine affected lot. QE disposition required. CAPA may be required based on frequency.

MINOR DEFECT (Class 3)
Definition: A defect unlikely to materially affect the function or service life of the product. Cosmetic imperfection on non-functional surface.
Examples:
- Superficial scratches on non-critical surfaces
- Minor dimensional out-of-tolerance on non-critical features
- Slight color variation within acceptable range
- Minor labeling imperfection (not a safety label)
Action: Evaluated per sampling plan. May be accepted with QE approval. Document in QC record.

DEFECT CODING SYSTEM
All defects logged in QC system with:
- Defect Code (from approved list)
- Classification (1, 2, or 3)
- Location on part (use part drawing reference)
- Quantity affected
- Inspector ID

DEFECT CODE EXAMPLES
D-001: Dimensional out of tolerance (specify dimension)
D-002: Surface finish defect
D-003: Wrong material
D-004: Assembly error
D-005: Contamination
D-006: Cracking / fracture
D-007: Missing component
D-008: Label defect
D-009: Electrical fault

PARETO ANALYSIS
QC team generates monthly Pareto chart of top defect codes. Top 3 defect codes trigger CAPA evaluation. Pareto results shared in Monthly Quality Review meeting.""",
    },
    {
        "title": "Inventory Management SOP",
        "topic": "inventory_management",
        "doc_type": "sop",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Inventory Management Standard Operating Procedure
Document: INVENT-001 | Rev: 2

PURPOSE
Establish procedures for receiving, storing, issuing, and tracking raw materials, work-in-process, and finished goods at Facility 12.

RECEIVING PROCEDURE
Step 1: Receiving clerk verifies delivery against Purchase Order (PO) in ERP.
Step 2: Count and compare quantity received vs. PO quantity. Note any discrepancy.
Step 3: Send sample to QC for incoming inspection (see QC Inspection Handbook).
Step 4: If QC passes: Enter receipt in ERP, print location label, move to assigned storage location.
Step 5: If QC fails: Quarantine material in red-tagged nonconforming area. Enter NCR in QC system.
Step 6: Email receipt confirmation to Buyer and Inventory Planner.

STORAGE REQUIREMENTS
Raw Materials:
- Store by material type and lot number (FIFO — First In, First Out)
- Temperature/humidity sensitive materials: stored in conditioned warehouse (see materials list)
- Hazardous materials: stored per SDS requirements in designated HazMat area
- Min/max quantity labels on all storage locations (trigger replenishment order at min)

Work-in-Process (WIP):
- Each traveler/job order has a physical job folder that stays with the part
- WIP stored in designated WIP lanes by operation sequence
- Maximum WIP accumulation at any work center: 2 days' worth

Finished Goods:
- Sorted by part number and customer
- FIFO strictly enforced — use FIFO labels
- Minimum 18 inches clearance from sprinkler heads
- Do not stack pallets more than 3 high without racking approval

ISSUING MATERIAL TO PRODUCTION
Operator requests material via ERP production order. Inventory attendant picks material by FIFO. Material issued against production order in ERP (backflush or manual). Discrepancies (quantity, lot number) reported to Inventory Planner within 1 hour.

CYCLE COUNTING
Full physical inventory: Annually (fiscal year-end)
Cycle counting: A-class items (top 20% by value): monthly. B-class: quarterly. C-class: annually.
Count accuracy target: 99% for A-class, 97% for B-class, 95% for C-class.
Discrepancies >$500: Investigated within 24 hours.

OBSOLETE AND EXPIRED MATERIAL
Materials past expiration date or declared obsolete are tagged and moved to quarantine.
Disposition (scrap or return to vendor) approved by Purchasing Manager.
Written-off in ERP by Accounting.

KEY METRICS
Inventory Accuracy (cycle count), Days on Hand, FIFO Compliance, On-Time Delivery from Stores""",
    },
    {
        "title": "ISO 9001:2015 Documentation Requirements",
        "topic": "iso_documentation",
        "doc_type": "policy",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """ISO 9001:2015 Documentation Requirements
Quality Management System (QMS) Overview
Document: QMS-DOC-001 | Rev: 6

PURPOSE
Define the documentation structure and control requirements for the Quality Management System at Facility 12, in compliance with ISO 9001:2015.

QMS DOCUMENTATION HIERARCHY
Level 1 — Quality Manual: High-level description of the QMS, organizational context, leadership commitment, and process interactions.
Level 2 — Procedures: How key processes are carried out (who, what, when).
Level 3 — Work Instructions / SOPs: Step-by-step instructions for specific tasks.
Level 4 — Records / Forms: Evidence that processes have been carried out.

MANDATORY ISO 9001:2015 RECORDS (must maintain)
- Monitoring and measuring equipment calibration records
- Evidence of employee competence (training records)
- Product/service requirements review records
- Design and development outputs records
- External provider qualification records
- Production and service provision records
- Inspection and test results
- Nonconforming output records
- Internal audit results
- Management review records
- Customer complaints and CAPA records

DOCUMENT CONTROL PROCEDURE (QMS-PRO-001)
Creation: Author prepares document using approved template (available on SharePoint: QMS > Templates).
Review: Technical review by subject matter expert; QMS review by Quality Engineer.
Approval: Quality Manager approves all Level 1-2 documents. Department Manager approves Level 3.
Issuance: Documents issued via QMS document control system (DocTrack). Paper copies are controlled copies only if stamped "CONTROLLED."
Revision: All changes follow the same review and approval process. Revision number incremented.
Obsolescence: Superseded documents removed from use and archived. Archive retained 10 years.

EXTERNAL DOCUMENTS
Customer drawings, specs, and standards: Controlled by Engineering per Drawing Control procedure.
Regulatory documents: Maintained by EHS Manager.

RECORD RETENTION
Active records: Accessible in QMS system or designated SharePoint folders.
Retention periods: Per regulatory requirements (minimum) or customer requirements (if longer).
Destruction: Per Records Retention Schedule (QMS-DOC-010). Destruction logged.

INTERNAL AUDITS
Minimum annually; audit schedule approved by QMS Manager.
All processes audited within each 3-year certification cycle.
Audit findings: CAPAs issued for all Major nonconformances. OFIs (Opportunities for Improvement) tracked.
External (Registrar) Audit: Conducted by certified ISO registrar. Surveillance annually; Re-certification every 3 years.

MANAGEMENT REVIEW
Conducted at minimum annually by senior leadership.
Inputs: Audit results, CAPA status, customer feedback, KPI trends, resource adequacy, risk.
Outputs: Actions, resource decisions, improvement opportunities.""",
    },
    {
        "title": "FAQ: Machine Safety and Emergency Response",
        "topic": "safety_protocols",
        "doc_type": "faq",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """FAQ: Machine Safety and Emergency Response

Q: What do I do if I see a machine running with a guard removed?
A: Stop the machine using the nearest E-stop. Do NOT restart it. Tag it with a "DO NOT OPERATE" tag. Notify your supervisor immediately and the Safety Officer.

Q: Can I bypass an E-stop if it's "just for a second" during startup?
A: Absolutely not. Bypassing, defeating, or ignoring any safety device is a serious safety violation and grounds for disciplinary action up to termination. E-stops and safety guards are never optional.

Q: Someone removed their LOTO lock before I finished my maintenance work. What should I do?
A: Do not attempt to re-energize the machine. Notify your supervisor and the Safety Manager immediately. This is a serious LOTO violation. An investigation will be conducted. Your safety is the priority.

Q: What PPE is required on the production floor?
A: Minimum: Safety glasses, steel-toed boots. Additional requirements by zone: Hearing protection (posted where noise exceeds 85 dB), face shield (grinding, welding), gloves (per task risk assessment). PPE requirements are posted at each zone entry.

Q: There's a chemical spill on the floor. What do I do?
A: Alert everyone nearby to evacuate the area. Activate the fire alarm if the spill is large or involves a flammable material. Do NOT attempt to clean up a hazardous material spill without proper training and equipment. Call the EHS emergency line: ext. 911.

Q: How do I report a near miss?
A: Report all near misses (incidents with injury potential, even if no one was hurt) to your supervisor within the same shift. Your supervisor enters the report in the EHS system. Near misses are reviewed in safety meetings — they are learning opportunities, not punishments.

Q: Can a visitor or contractor work on a machine without following our LOTO procedure?
A: No. All personnel — employees, contractors, temporary workers, visitors — must follow site safety rules. Contractors must also follow their own LOTO program if stricter. The Site Safety Manager coordinates contractor safety requirements.""",
    },
    {
        "title": "FAQ: Quality Control and Nonconformances",
        "topic": "quality_control",
        "doc_type": "faq",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """FAQ: Quality Control and Nonconformances

Q: I found a defective part during inspection. What do I do?
A: Stop production immediately if an in-process defect. Tag the nonconforming part with a RED nonconformance tag. Move it to the quarantine area. Enter an NCR in the QC system within 1 hour. Notify the Quality Engineer and Production Supervisor.

Q: Can I continue running production while waiting for the QE to evaluate a nonconformance?
A: Only if the QE gives explicit verbal or written authorization to continue. Otherwise, stop until the nonconformance is dispositioned. Running with a known nonconformance without authorization is a quality violation.

Q: What is the difference between a rework and a repair?
A: Rework: Restores the part to the original specification using standard processes. Repair: Restores function but does not restore the part to original specification — always requires customer approval and must be documented.

Q: How do I know if a gage is in calibration?
A: Check the calibration sticker on the gage. It shows the last calibration date and next due date. If no sticker, the calibration date has passed, or the gage is physically damaged — do NOT use it. Tag it and send it to the Metrology Lab.

Q: A customer called about a complaint. What should I do?
A: Do not make any commitments to the customer. Document the complaint details (part number, lot, quantity, nature of defect, date detected). Contact the Quality Manager and Sales Manager immediately. They will lead the customer response.

Q: I accidentally shipped a nonconforming part. Who do I notify?
A: Notify the Quality Manager and Customer Service Manager immediately. Do not wait. A containment plan and customer notification must be initiated within 24 hours. This may trigger a CAPA.

Q: What happens if the same defect keeps occurring?
A: Recurring defects (same defect code appearing 3+ times) automatically trigger a CAPA. The CAPA process includes root cause analysis and corrective action to eliminate the root cause permanently.""",
    },
    {
        "title": "CAPA FAQ and Root Cause Analysis Guide",
        "topic": "capa",
        "doc_type": "faq",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """FAQ: CAPA Process and Root Cause Analysis

Q: When is a CAPA required versus a Quick Fix?
A: A Quick Fix is appropriate for isolated, minor, one-time issues with a clear, simple solution. A CAPA is required for any customer complaint classified as Major or Critical, recurring defects, systemic process failures, and all audit findings.

Q: How do I write a good 5-Why?
A: Start with the specific problem statement (not a symptom). Ask "Why did this happen?" and answer with a specific, verifiable fact. Repeat 5 times (or until you reach a root cause you can act on). Each answer should become the next "why" question. Poor example: "Why did it fail? — Quality was not checked." Good example: "Why did it fail? — The in-process inspection step was skipped. Why was it skipped? — The operator was not aware it was required. Why? — The work instruction did not clearly specify the inspection step..."

Q: Who can close a CAPA?
A: The Quality Manager must approve all CAPA closures. The CAPA owner and their supervisor must sign off before submitting for QM approval.

Q: What counts as "verification of effectiveness"?
A: You must collect objective data showing the defect has not recurred after implementing the corrective action. This typically means running 30+ units or monitoring the process for 30 days and confirming no recurrence. A simple statement like "training was completed" is not sufficient.

Q: Can I reopen a closed CAPA if the problem comes back?
A: Yes. If a previously closed CAPA defect recurs, reopen the original CAPA and document that the corrective action was insufficient. A new, more rigorous root cause analysis is required.

Q: What is a Preventive Action and when is it required?
A: A Preventive Action addresses a potential problem before it occurs, based on analysis of trends, risk, or lessons learned from similar issues. ISO 9001 requires organizations to address risks and opportunities — preventive actions help do this. Include preventive actions in the CAPA when you identify a similar process or product that could have the same root cause.""",
    },
    {
        "title": "Production Scheduling and Capacity Planning Guide",
        "topic": "production_scheduling",
        "doc_type": "guide",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Production Scheduling and Capacity Planning — Facility 12 Operations Guide

SCHEDULING PHILOSOPHY
Facility 12 uses a pull-based scheduling system driven by customer demand signals. The Master Production Schedule (MPS) is maintained by the Production Planning team and updated weekly. Daily schedules are managed by the Shift Supervisor using the Manufacturing Execution System (MES).

SCHEDULE CREATION PROCESS
1. Demand Input: Customer orders and forecasts are entered into the ERP system by the Sales team. Confirmed orders take priority over forecast-based planning.
2. Capacity Check: The Planning team checks available machine capacity, labor hours, and raw material availability before confirming order commitments.
3. MPS Generation: The MPS is generated weekly for a 6-week rolling horizon. The first 2 weeks are firm; weeks 3-6 are flexible.
4. Daily Schedule: The Shift Supervisor downloads the daily work order list from MES each morning. Sequence is determined by: (1) Promise date, (2) Setup similarity (to minimize changeovers), (3) Material availability.

CAPACITY PLANNING RULES
- Machine OEE target: 85% minimum. Capacity calculations assume 85% availability.
- Buffer stock: Maintain 3 days of finished goods buffer for A-class customers.
- Overtime policy: Overtime requires Plant Manager approval. Planned overtime must be communicated to employees 48 hours in advance.
- Subcontracting: Secondary operations (heat treat, plating, grinding) have 5-day lead times. Include in schedule calculations.

CHANGEOVER MANAGEMENT
Minimize total changeover time by grouping similar part families in sequence. The standard changeover matrix is maintained on the production floor and in the MES. Target changeover time by line: Line 1 = 45 min, Line 2 = 30 min, Line 3 = 20 min.

SCHEDULE DISRUPTION RESPONSE
When an unplanned event (machine breakdown, material shortage, quality hold) disrupts the schedule:
1. Shift Supervisor assesses impact and estimates recovery time.
2. If the delay exceeds 2 hours, notify the Planning team to re-sequence or expedite.
3. Customer Service is notified of any orders at risk of missing promise dates within 4 hours of identifying the delay.
4. Root cause and recovery action documented in the MES downtime log.

KEY PERFORMANCE INDICATORS
- On-Time Delivery (OTD): Target >= 95% of orders shipped on or before promise date.
- Schedule Adherence: Target >= 90% of work orders completed within scheduled time window.
- Changeover Efficiency: Actual vs standard changeover time, target <= 110% of standard.""",
    },
    {
        "title": "Inventory Management and Material Control Procedures",
        "topic": "inventory_management",
        "doc_type": "procedure",
        "source": "synthetic",
        "sensitivity": "internal",
        "content": """Inventory Management and Material Control — Standard Operating Procedure IMS-001

PURPOSE
To define the processes for receiving, storing, controlling, and issuing raw materials, work-in-process, and finished goods at Facility 12, ensuring traceability and preventing use of nonconforming or expired materials.

RECEIVING PROCESS
All incoming materials must be received through the designated Receiving Dock. Steps:
1. Verify the packing slip matches the Purchase Order (part number, quantity, revision level).
2. Inspect packaging for damage. Photograph and document any damage.
3. Print and apply a Receiving Lot tag (includes PO number, supplier lot, date received, quantity, part number).
4. Move materials to the Receiving Inspection area — do NOT move to stock until inspection is complete.
5. Quality Receiving Inspector performs inspection per the approved Incoming Inspection Plan.
6. Approved materials receive a GREEN Approved sticker and are moved to assigned stock location.
7. Rejected materials receive a RED Hold sticker and are moved to the Quarantine area.

STORAGE AND LOCATION CONTROL
- All stock locations are labeled and maintained in the ERP system.
- First-In, First-Out (FIFO) rotation is mandatory. Older lot numbers must be consumed before newer lots.
- Temperature-sensitive materials are stored in the climate-controlled storage room (temp: 65-75°F, humidity: <60%).
- Shelf-life controlled materials: expiration date tracked in ERP. Materials within 30 days of expiration are flagged for priority use or disposition.
- Hazardous materials stored in approved flammable storage cabinets, segregated by chemical compatibility.

MATERIAL ISSUANCE
Materials are issued to production only via MES work order. The Warehouse Technician scans the part and lot barcode to confirm the correct material and record the issuance transaction. No material may be removed from stock without a valid work order.

INVENTORY ACCURACY
- Cycle counting: All A-class parts counted monthly, B-class quarterly, C-class annually.
- Annual physical inventory: Full count in the first week of January. Production is stopped during count.
- Inventory accuracy target: >= 98% for A-class parts.
- Discrepancies > 5% require root cause investigation and corrective action.

TRACEABILITY
The lot traceability chain — Supplier lot → Receiving lot → Work order → Finished goods serial — must be maintained intact in the ERP at all times. Traceability is required for customer recalls and regulatory audits. Loss of traceability is a Major nonconformance requiring immediate CAPA.""",
    },
]


def save_synthetic_documents():
    """Save all synthetic documents to tenant raw directories."""
    for tenant_id, docs in [("sis", SIS_DOCUMENTS), ("mfg", MFG_DOCUMENTS)]:
        config = TENANTS[tenant_id]
        config.raw_dir.mkdir(parents=True, exist_ok=True)

        for i, doc in enumerate(docs):
            filename = f"{doc['topic']}_{i:03d}.txt"
            filepath = config.raw_dir / filename
            content = f"TITLE: {doc['title']}\nTOPIC: {doc['topic']}\n\n{doc['content'].strip()}"
            filepath.write_text(content, encoding="utf-8")

        manifest = {
            "tenant_id": tenant_id,
            "domain": config.domain,
            "document_count": len(docs),
            "topics": list({d["topic"] for d in docs}),
            "documents": [
                {"title": d["title"], "topic": d["topic"], "filename": f"{d['topic']}_{i:03d}.txt"}
                for i, d in enumerate(docs)
            ],
        }
        manifest_path = config.raw_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[{tenant_id}] Saved {len(docs)} documents to {config.raw_dir}")

    return {"sis": len(SIS_DOCUMENTS), "mfg": len(MFG_DOCUMENTS)}


if __name__ == "__main__":
    counts = save_synthetic_documents()
    print(f"Generated synthetic data: {counts}")
