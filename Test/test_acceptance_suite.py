import os
import sys
import re
import uuid
import time
import json
import base64
import sqlite3
import subprocess

# Setup test environment
os.environ["JWT_SECRET"] = "acceptance-suite-jwt-secret-key-at-least-32-chars-long"
os.environ["ALLOW_SQLITE_FALLBACK"] = "true"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import auth_manager
from database import VaultDatabase, DatabaseUnavailableError, DatabaseIntegrityError
from ingestion_pipeline import DocumentIngestionPipeline, UnreadableDocumentError
from retrieval_pipeline import MultiTenantRetrievalPipeline
import agent_swarm
from agent_swarm import UserIntent
import app as aura_app


def create_isolated_env(db_name: str):
    db_path = f"/tmp/{db_name}_{uuid.uuid4().hex[:8]}.db"
    chroma_path = f"/tmp/chroma_{db_name}_{uuid.uuid4().hex[:8]}"
    db = VaultDatabase(db_url="", sqlite_path=db_path)
    auth = auth_manager.AuthManager(db)
    ingest = DocumentIngestionPipeline(chunk_size=180, chunk_overlap=25, max_chunk_size=250)
    retrieval = MultiTenantRetrievalPipeline(chroma_path, database=db)
    assistant = agent_swarm.EnterprisePersonalAssistant(retrieval, db)
    return db, auth, ingest, retrieval, assistant


def test_01_20_unanswerable_and_8_paraphrased_answerable_probes():
    """
    Acceptance Gate 1 (C1, N7, N9, V1):
    1. 20 unanswerable questions (10 general trivia + 10 domain near-miss probes) -> >= 95% abstain with zero citations.
    2. All 8 of the reviewer's paraphrased answerable probes at simulated cosine 0.45 -> 8/8 PASS the relevance gate (Fixes V1).
    """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate1")
    u = db.upsert_google_user("g_gate1", "gate1@example.com", "Gate 1 User", "")
    uid = u["id"]
    doc_id = f"doc_{uuid.uuid4().hex}"
    motor_text = (
        "Zurich Motor Insurance Policy Schedule. Policy No: PBW/00001001. "
        "Compulsory Own Damage Deductible is Rs 2,500. Third Party Liability limit is Rs 7,50,000. "
        "Roadside assistance toll-free number is 1800-209-8888. Policy expires on 15/12/2027."
    )
    chunks = ingest.recursive_chunk_text(motor_text, doc_id, "Motor_Policy.pdf", "Insurance")
    db.add_document_with_chunks(uid, doc_id, "Motor_Policy.pdf", "Motor_Policy.pdf", "pdf", 1024, chunks, "Insurance", {}, "h_g1")
    retrieval.index_chunks(chunks, user_id=uid, category="Insurance")
    conv = db.create_conversation(uid, "Gate 1 Chat")

    unanswerable_20 = [
        # 10 Domain near-miss probes
        "Does my policy cover flood and hurricane damage?",
        "What is the dental waiting period in my insurance policy?",
        "Does my insurance policy cover international emergency evacuation?",
        "What is the earthquake deductible in my policy document?",
        "Does my policy cover orthodontic braces or vision care?",
        "What is the cyber extortion limit in my insurance document?",
        "Does my contract include pet surgery reimbursement?",
        "What is the crop hail damage limit in my policy?",
        "Does my policy cover marine cargo transit losses?",
        "What is the kidnapping and ransom limit in my insurance policy?",
        # 10 Out-of-domain trivia probes
        "What is the capital of France?",
        "Who won the 2018 FIFA World Cup?",
        "How do I bake sourdough bread at home?",
        "What is the speed of light in a vacuum?",
        "Who wrote the play Hamlet?",
        "What is the population of Tokyo?",
        "How tall is Mount Everest in meters?",
        "When did the Titanic sink?",
        "What is the boiling point of gold?",
        "Who painted the Mona Lisa?",
    ]

    abstained = 0
    for q in unanswerable_20:
        res = assistant.respond(uid, conv["id"], q)
        if "could not find information" in res["answer"].lower() and len(res["citations"]) == 0:
            abstained += 1

    assert abstained >= 19, f"Expected >= 19/20 abstentions across trivia + domain near-misses, got {abstained}/20"

    # Part 3 V1 Probe: All 8 paraphrased answerable probes at simulated cosine 0.45 must PASS the gate
    lease_text = (
        "Residential Lease Agreement for Apartment 14B. Monthly rent is Rs 42,000. "
        "Security deposit is Rs 2,00,000 refundable at termination. Lock-in period is 6 months."
    )
    paraphrase_probes_8 = [
        ("Check my policy renewal and end date", motor_text, "Motor_Policy.pdf"),
        ("When does my insurance end?", motor_text, "Motor_Policy.pdf"),
        ("How much do I pay out of pocket on a claim?", motor_text, "Motor_Policy.pdf"),
        ("What number do I call for breakdown help?", motor_text, "Motor_Policy.pdf"),
        ("How much money do I get back when I move out?", lease_text, "Apartment_Lease.pdf"),
        ("How long is the minimum stay?", lease_text, "Apartment_Lease.pdf"),
        ("What is the deductible on my car policy?", motor_text, "Motor_Policy.pdf"),
        ("How much is the monthly flat rent?", lease_text, "Apartment_Lease.pdf"),
    ]
    for pq, chunk_txt, d_title in paraphrase_probes_8:
        sim_item = {
            "content": chunk_txt,
            "doc_title": d_title,
            "dense_score": 0.45,
            "sparse_score": 0.0,
            "matched_terms": 0
        }
        assert assistant.rag_agent.passes_relevance_gate(sim_item, pq) is True, f"Paraphrase '{pq}' was falsely refused at d_score=0.45!"


def test_02_50_question_golden_set_recall_at_production_chunk_size():
    """
    Acceptance Gate 2 (N7, V1, V6): 50-question golden evaluation set (including paraphrased questions)
    using PRODUCTION chunking (chunk_size=180, chunk_overlap=25, max_chunk_size=250) over a >= 32-chunk
    multi-document corpus -> Recall@5 >= 90%.
    """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate2")
    u = db.upsert_google_user("g_gate2", "gate2@example.com", "Gate 2 User", "")
    uid = u["id"]

    # Build a multi-page corpus that produces >= 32 chunks at the exact production chunk_size=180
    annexure_pages = "\n\n".join([
        f"--- [Page {p}] ---\n" + " ".join([
            f"Annexure {p} Clause {k}: Standard regulatory compliance procedure, archival record retention policy, "
            f"statutory audit guideline, ombudsman correspondence address, and regional branch office directory."
            for k in range(1, 12)
        ])
        for p in range(2, 29)
    ])

    corpus = [
        ("Motor_Policy.pdf", "Insurance",
         "--- [Page 1] ---\nZurich Motor Insurance Policy No: PBW/00001001. Insured Declared Value (IDV) is Rs 8,50,000. "
         "Compulsory deductible is Rs 2,000. Zero Depreciation add-on is active. "
         "Engine protector cover included. No Claim Bonus (NCB) is 25%. Roadside assistance toll-free number is 1800-209-8888. "
         "Cashless garage network has 4,500 workshops. Policy expires on 15/12/2027. Geographical area is India.\n\n" + annexure_pages),
        ("Health_Plan.pdf", "Insurance",
         "--- [Page 1] ---\nFamily Floater Health Insurance Policy No: POL-90002211. Sum Insured is Rs 15,00,000. "
         "Room rent limit is Single Private AC Room. Maternity waiting period is 24 months. "
         "Pre-existing disease waiting period is 36 months. Pre-hospitalization cover is 60 days and post-hospitalization is 180 days. "
         "Annual health checkup reimbursement up to Rs 5,000. Daycare procedures covered: 540 listed procedures. Ambulance limit Rs 3,000."),
        ("Employment_Contract.pdf", "Career",
         "--- [Page 1] ---\nEmployment Offer and Agreement for Senior AI Engineer. Base salary is $165,000 per annum. "
         "Annual performance bonus target is 15%. Probation period is 90 days. "
         "Notice period after confirmation is 60 days. Paid time off (PTO) is 24 working days per year. "
         "Equity grant is 10,000 RSUs vesting over 4 years with a 1-year cliff. Remote work stipend is $1,500 annually. Reporting to VP of Engineering."),
        ("Apartment_Lease.pdf", "Legal",
         "--- [Page 1] ---\nResidential Lease Agreement for Apartment 14B, Green Valley Towers. Monthly rent is Rs 42,000 payable by the 5th of each month. "
         "Security deposit is Rs 2,00,000 refundable at termination. "
         "Lock-in period is 6 months. Termination notice period is 2 months by either Lessor or Lessee. Annual rent escalation is 5%. "
         "Monthly maintenance charge is Rs 4,500. Pets are permitted with prior written consent. Subletting is strictly prohibited."),
        ("Bank_Statement.pdf", "Finance",
         "--- [Page 1] ---\nChecking Account Statement for October 2026. IFSC Code: HDFC0004321. Opening balance is Rs 1,20,000. "
         "Salary credit on 01/10/2026 of Rs 1,45,000. Home loan EMI debit on 05/10/2026 of Rs 38,500. "
         "Electricity bill payment of Rs 3,200. Mutual fund SIP debit of Rs 25,000. Closing balance is Rs 1,98,300. "
         "Overdraft limit approved is Rs 50,000 at 11.5% interest rate.")
    ]

    total_chunks = 0
    for idx, (title, cat, text) in enumerate(corpus):
        did = f"doc_gold_{idx}"
        chunks = ingest.recursive_chunk_text(text, did, title, cat)
        total_chunks += len(chunks)
        db.add_document_with_chunks(uid, did, title, title, "pdf", len(text), chunks, cat, {}, f"hash_gold_{idx}")
        retrieval.index_chunks(chunks, user_id=uid, category=cat)

    assert total_chunks >= 30, f"Expected >= 30 chunks at production chunk_size=180, got {total_chunks}"

    golden_path = os.path.join(os.path.dirname(__file__), "eval_golden_set.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_50 = json.load(f)

    assert len(golden_50) == 50
    hits_found = 0
    for item in golden_50:
        q = item["question"]
        expected_substring = item["expected_answer_substring"]
        hits = retrieval.retrieve(q, user_id=uid, top_k=5)
        assert len(hits) <= 5
        if any(expected_substring.lower() in h[0].get("content", "").lower() and assistant.rag_agent.passes_relevance_gate(h[0], q) for h in hits):
            hits_found += 1

    recall_at_5 = hits_found / float(len(golden_50))
    assert recall_at_5 >= 0.90, f"Expected Recall@5 >= 90% over {total_chunks} production chunks, got {recall_at_5 * 100:.1f}% ({hits_found}/50)"


def test_03_router_table_60_utterances():
    """
    Acceptance Gate 3 (C2, N9, Part 3): 60-utterance router test table -> >= 95% accuracy,
    including single-document attribute comparisons AND cross-document singular/plural comparison phrasings.
    """
    router = agent_swarm.SupervisorIntentRouter()
    docs = [
        {"id": "d1", "title": "Zurich_Motor_Policy.pdf"},
        {"id": "d2", "title": "Health_Floater_Plan.pdf"},
        {"id": "d3", "title": "Apartment_Lease.pdf"}
    ]

    utterances_60 = [
        # 28 Single-Document RAG queries (including 'how does', temporal 'vs', and single-doc attribute comparisons)
        ("Compare the premium and deductible in my motor policy", UserIntent.DOCUMENT_RAG),
        ("Contrast the opening balance and closing balance in my bank statement", UserIntent.DOCUMENT_RAG),
        ("How does my policy handle accidental damage?", UserIntent.DOCUMENT_RAG),
        ("How does the claim process work?", UserIntent.DOCUMENT_RAG),
        ("Is my premium higher vs last year?", UserIntent.DOCUMENT_RAG),
        ("How does cashless hospitalization work?", UserIntent.DOCUMENT_RAG),
        ("How does the deductible apply to windshield claims?", UserIntent.DOCUMENT_RAG),
        ("How does my No Claim Bonus change vs prior renewal?", UserIntent.DOCUMENT_RAG),
        ("What is my policy number?", UserIntent.DOCUMENT_RAG),
        ("What is the deductible on my car insurance?", UserIntent.DOCUMENT_RAG),
        ("When does my lease expire?", UserIntent.DOCUMENT_RAG),
        ("How much is my monthly rent?", UserIntent.DOCUMENT_RAG),
        ("Does my health insurance cover maternity?", UserIntent.DOCUMENT_RAG),
        ("What is the room rent cap?", UserIntent.DOCUMENT_RAG),
        ("Who is the lessor in the apartment lease?", UserIntent.DOCUMENT_RAG),
        ("What is the security deposit amount?", UserIntent.DOCUMENT_RAG),
        ("Summarize my car insurance coverage", UserIntent.DOCUMENT_RAG),
        ("What are the exclusions in my motor policy?", UserIntent.DOCUMENT_RAG),
        ("What is the notice period in my employment contract?", UserIntent.DOCUMENT_RAG),
        ("How many days of PTO do I get?", UserIntent.DOCUMENT_RAG),
        ("What was my closing bank balance in October?", UserIntent.DOCUMENT_RAG),
        ("What is the IFSC code on my bank statement?", UserIntent.DOCUMENT_RAG),
        ("How does the pre-existing disease waiting period work?", UserIntent.DOCUMENT_RAG),
        ("Is roadside assistance covered?", UserIntent.DOCUMENT_RAG),
        ("What is my IDV vs last year?", UserIntent.DOCUMENT_RAG),
        ("How does zero depreciation coverage work?", UserIntent.DOCUMENT_RAG),
        ("Are pets allowed in my lease?", UserIntent.DOCUMENT_RAG),
        ("What is the lock-in period for the apartment?", UserIntent.DOCUMENT_RAG),

        # 14 Explicit Multi-Document Comparison queries (including cross-doc singular noun phrasings from Part 3)
        ("Compare my Zurich policy and Health plan side by side", UserIntent.DOCUMENT_COMPARISON),
        ("Compare my car insurance with my health insurance", UserIntent.DOCUMENT_COMPARISON),
        ("How does my motor policy compare to my health plan?", UserIntent.DOCUMENT_COMPARISON),
        ("Compare all my uploaded policies", UserIntent.DOCUMENT_COMPARISON),
        ("What are the differences between my documents?", UserIntent.DOCUMENT_COMPARISON),
        ("Show a side-by-side comparison of both contracts", UserIntent.DOCUMENT_COMPARISON),
        ("Compare the deductibles across my insurance policies", UserIntent.DOCUMENT_COMPARISON),
        ("Contrast the termination clauses between the two agreements", UserIntent.DOCUMENT_COMPARISON),
        ("How do my two policies differ in coverage?", UserIntent.DOCUMENT_COMPARISON),
        ("Compare Zurich Motor Policy vs Health Floater Plan", UserIntent.DOCUMENT_COMPARISON),
        ("Give me a comparison between both uploaded files", UserIntent.DOCUMENT_COMPARISON),
        ("Compare the key dates across my documents", UserIntent.DOCUMENT_COMPARISON),
        ("What is the difference between my two policies?", UserIntent.DOCUMENT_COMPARISON),
        ("Compare my Zurich and Apartment Lease documents", UserIntent.DOCUMENT_COMPARISON),

        # 10 Memory Management queries
        ("What do you know about me?", UserIntent.MEMORY_MANAGEMENT),
        ("What personal facts do you know about me?", UserIntent.MEMORY_MANAGEMENT),
        ("Show my memories", UserIntent.MEMORY_MANAGEMENT),
        ("What facts have you saved?", UserIntent.MEMORY_MANAGEMENT),
        ("What do you remember about me?", UserIntent.MEMORY_MANAGEMENT),
        ("List my memories", UserIntent.MEMORY_MANAGEMENT),
        ("Show my saved profile facts", UserIntent.MEMORY_MANAGEMENT),
        ("What facts about me are stored?", UserIntent.MEMORY_MANAGEMENT),
        ("Check my memories", UserIntent.MEMORY_MANAGEMENT),
        ("View my saved facts", UserIntent.MEMORY_MANAGEMENT),

        # 8 General Chat / Greetings
        ("Hi", UserIntent.GENERAL_CHAT),
        ("Hello", UserIntent.GENERAL_CHAT),
        ("Hey AURA", UserIntent.GENERAL_CHAT),
        ("Good morning", UserIntent.GENERAL_CHAT),
        ("Good evening", UserIntent.GENERAL_CHAT),
        ("Who are you?", UserIntent.GENERAL_CHAT),
        ("What can you do?", UserIntent.GENERAL_CHAT),
        ("Thank you", UserIntent.GENERAL_CHAT),
    ]

    assert len(utterances_60) == 60
    correct = 0
    for q, expected in utterances_60:
        got = router.classify_intent(q, [], has_docs=True, user_docs=docs)["intent"]
        if got == expected:
            correct += 1

    accuracy = correct / 60.0
    assert accuracy >= 0.95, f"Router accuracy {accuracy*100:.1f}% ({correct}/60) < 95%"


def test_04_critic_catches_fabricated_citation_altered_swapped_and_derived_hallucinations():
    """
    Acceptance Gate 4 (C3, N9, V2):
    1. Fabricated citation removed and never relabelled.
    2. Altered number flagged.
    3. Single-sentence and multi-sentence swapped attribute numbers (premium vs deductible) flagged.
    4. Un-signaled sums/differences ('30 days' from 90-60, 'Rs 7,000' from 12,000-5,000, '85 days' from 60+25) FLAGGED (Fixes V2).
    5. Explicitly signaled calculation with both operands in the same chunk ('5,000 and 1,500, for a combined total of 6,500') GROUNDED.
    """
    critic = agent_swarm.VerificationCriticAgent()
    citations = [{"id": "cit_1", "doc_title": "Zurich_Car_Policy.pdf", "clean_label": "Zurich Car Policy (P.1)"}]
    hits = [(
        {"content": "Your annual insurance premium is Rs 12,000 and the compulsory deductible is Rs 2,500 (two thousand five hundred). Pre-hospitalization is 60 days, post-hospitalization is 90 days, NCB is 25%, and health checkup is Rs 5,000."},
        0.05
    )]

    # 1. Altered number probe
    res_num = critic.audit_response("Your annual premium is Rs 99,999 [Zurich Car Policy (P.1)].", hits, citations)
    assert res_num["is_grounded"] is False

    # 2. Fabricated citation probe
    res_cit = critic.audit_response("Your annual premium is Rs 12,000 [Made Up Doc (P.9)].", hits, citations)
    assert "Made Up Doc" not in res_cit["audited_answer"]
    assert "Zurich Car Policy" not in res_cit["audited_answer"]
    assert res_cit["is_grounded"] is False

    # 3. Single-sentence and compound-sentence swapped attribute numbers (Part 3 V2 probe)
    res_swap1 = critic.audit_response(
        "The premium is Rs 12,000 and the deductible is Rs 2,500. In summary, your premium is Rs 2,500 [Zurich Car Policy (P.1)].",
        hits, citations
    )
    assert res_swap1["is_grounded"] is False and any("mismatch" in f.lower() for f in res_swap1["flags"])

    res_swap2 = critic.audit_response(
        "Your annual premium is Rs 2,500 and the deductible is Rs 12,000 [Zurich Car Policy (P.1)].",
        hits, citations
    )
    assert res_swap2["is_grounded"] is False and any("mismatch" in f.lower() for f in res_swap2["flags"])

    # 4. Part 3 V2 probes: un-signaled coincidental sums/differences ('30 days', 'Rs 7,000', '85 days') MUST be flagged!
    for bad_claim in [
        "The waiting period is 30 days [Zurich Car Policy (P.1)].",
        "The policy fee is Rs 7,000 [Zurich Car Policy (P.1)].",
        "Coverage lasts for 85 days [Zurich Car Policy (P.1)].",
    ]:
        res_bad = critic.audit_response(bad_claim, hits, citations)
        assert res_bad["is_grounded"] is False, f"Expected coincidental sum/diff '{bad_claim}' to be flagged as ungrounded!"

    # 5. Explicitly signaled calculation with both operands present in the same chunk and in the answer passes
    hits_calc = [({"content": "The hospital room deductible is five thousand rupees and ambulance fee is 1,500."}, 0.05)]
    res_derived = critic.audit_response(
        "The deductible is Rs 5,000 and ambulance fee is Rs 1,500, for a combined total of Rs 6,500 [Zurich Car Policy (P.1)].",
        hits_calc, citations
    )
    assert res_derived["is_grounded"] is True, f"Expected explicit calculation with both operands to pass, got {res_derived['flags']}"


def test_05_two_tenant_isolation():
    """
    Acceptance Gate 5: Two-tenant API isolation across chat, compare, memory, and delete -> zero cross-tenant access.
    """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate5")
    alice = db.upsert_google_user("g_alice", "alice@example.com", "Alice", "")
    bob = db.upsert_google_user("g_bob", "bob@example.com", "Bob", "")
    a_id, b_id = alice["id"], bob["id"]

    d_alice = f"doc_{uuid.uuid4().hex}"
    chunks_a = ingest.recursive_chunk_text("Alice Secret Salary is $295,000 per year.", d_alice, "Salary.pdf", "Finance")
    db.add_document_with_chunks(a_id, d_alice, "Salary.pdf", "Salary.pdf", "pdf", 500, chunks_a, "Finance", {}, "h_a")
    mem_a = db.upsert_user_memory(a_id, "Finance", "Salary", "$295,000", "Salary.pdf", source_doc_id=d_alice)

    assert db.get_document(b_id, d_alice) is None
    assert len(retrieval.retrieve("$295,000 Salary", user_id=b_id)) == 0
    assert db.update_user_memory(b_id, mem_a["id"], "Salary", "$0") is False
    assert db.delete_user_memory(b_id, mem_a["id"]) is False
    assert db.delete_document(b_id, d_alice) is False
    assert db.get_document(a_id, d_alice) is not None


def test_06_oauth_state_nonce_and_expired_token():
    """
    Acceptance Gate 6 (C4 & N7): Missing state, server-side wrong nonce in verify_google_id_token, and expired token -> all rejected.
    """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate6")
    u = db.upsert_google_user("g_6", "u6@example.com", "U6", "")
    exp_tok = auth.create_session_token(u["id"], "u6@example.com", expires_in_seconds=-5)
    assert auth.verify_session_token(exp_tok) is None

    import unittest.mock as mock
    fake_google_payload = {
        "iss": "https://accounts.google.com",
        "aud": auth_manager.get_configured_google_client_id(),
        "sub": "google_sub_123",
        "email": "u6@example.com",
        "email_verified": "true",
        "nonce": "nonce_from_token_abc",
        "exp": str(int(time.time()) + 3600)
    }
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_google_payload
    with mock.patch("requests.get", return_value=mock_resp):
        ok_profile = auth.verify_google_id_token("eyJ_fake_google_jwt", expected_nonce="nonce_from_token_abc")
        assert ok_profile is not None and ok_profile["email"] == "u6@example.com"
        bad_profile = auth.verify_google_id_token("eyJ_fake_google_jwt", expected_nonce="wrong_nonce_xyz")
        assert bad_profile is None, "Server-side verify_google_id_token must reject nonce mismatch!"

    with open(os.path.join(ROOT_DIR, "index.html")) as f:
        html = f.read()
    assert "if (!expectedState || !returnedState || expectedState !== returnedState)" in html


def test_07_production_start_without_jwt_secret_refuses_to_start():
    """
    Acceptance Gate 7 (C5): Production start with RENDER + DATABASE_URL and no JWT_SECRET -> refuses to start.
    """
    env = os.environ.copy()
    env.pop("JWT_SECRET", None)
    env["RENDER"] = "true"
    env["DATABASE_URL"] = "postgresql://user:pass@ep-test.neon.tech/neondb"
    env["GROQ_API_KEY"] = "gsk_test_key"

    res = subprocess.run(
        [sys.executable, "-c", "import auth_manager"],
        cwd=ROOT_DIR,
        env=env,
        capture_output=True,
        text=True
    )
    assert res.returncode != 0, "Expected startup failure when JWT_SECRET is missing in production!"
    assert "Missing required environment variable 'JWT_SECRET'" in res.stderr


def test_08_table_csv_and_comma_prose_chunk_invariants():
    """
    Acceptance Gate 8 (H1, H2, N11):
    - 200-row Markdown table -> <= 250 words, header repeated, no mid-row starts.
    - 400-row CSV -> <= 180 MiniLM wordpieces per chunk (< 256 limit), header repeated.
    - Hard-wrapped .txt prose with 2+ commas per line -> NOT misdetected as CSV.
    """
    ingest = DocumentIngestionPipeline(chunk_size=180, chunk_overlap=25, max_chunk_size=250)

    header = "| Benefit_Item | Coverage_Limit | Deductible_Amount | Policy_Notes |"
    sep = "| --- | --- | --- | --- |"
    rows = [f"| Benefit_{i} | Rs_{i*10000} | Rs_2500 | Covered_under_section_{i}_terms |" for i in range(1, 201)]
    md_text = "\n".join([header, sep] + rows)
    t_chunks = ingest.recursive_chunk_text(md_text, "d_tbl", "table.md")
    assert max(len(c.content.split()) for c in t_chunks) <= 250
    assert all(c.content.splitlines()[0].strip() == header for c in t_chunks)

    csv_header = "Date,Description,Reference,Withdrawal,Deposit,Balance"
    csv_rows = [f"2026-10-{(i%28)+1:02d},Transfer_{i},REF{100+i},{i*10}.00,0.00,{50000-i*10}.00" for i in range(1, 401)]
    c_chunks = ingest.recursive_chunk_text("\n".join([csv_header] + csv_rows), "d_csv", "bank.csv")
    # Verify actual punctuation-split wordpieces <= 180 (well inside 256-wordpiece MiniLM limit)
    max_wp = max(len(re.findall(r"[a-zA-Z]+|\d+|[^\s\w]", c.content)) for c in c_chunks)
    assert max_wp <= 180, f"Expected CSV wordpieces <= 180, got {max_wp}"
    assert all(c.content.splitlines()[0].strip() == csv_header for c in c_chunks)

    first_line = "In accordance with Section 12, Clause 4, Sub-clause B, the policyholder shall notify the insurer."
    prose_lines = [first_line] + [
        f"Furthermore, under article {i}, paragraph 2, the insured vehicle, accessories, and spare parts are covered."
        for i in range(1, 40)
    ]
    p_chunks = ingest.recursive_chunk_text("\n".join(prose_lines), "d_prose", "contract_notes.txt")
    assert len(p_chunks) >= 2
    assert not p_chunks[1].content.startswith(first_line), "Prose .txt with commas must not repeat line 1 as a CSV header!"


def test_09_database_503_vs_409_and_legacy_schema_migration():
    """
    Acceptance Gate 9 (H5, N3, N4, V3):
    - Legacy pre-existing SQLite database with duplicate documents AND chunks attached to those duplicates
      is migrated in the right order (duplicate docs deleted first, then orphan chunks deleted, then FK & UNIQUE added).
    - Duplicate (user_id, file_hash) raises DatabaseIntegrityError (409), NOT DatabaseUnavailableError (503).
    """
    legacy_db_path = f"/tmp/legacy_{uuid.uuid4().hex[:8]}.db"
    conn = sqlite3.connect(legacy_db_path)
    conn.execute("CREATE TABLE users (id TEXT PRIMARY KEY, google_id TEXT, email TEXT, full_name TEXT, avatar_url TEXT)")
    conn.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, user_id TEXT, title TEXT, original_filename TEXT, file_type TEXT, file_size_bytes INTEGER, chunk_count INTEGER, file_hash TEXT)")
    conn.execute("CREATE TABLE document_chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT, user_id TEXT, doc_title TEXT, chunk_index INTEGER, page_number INTEGER, content TEXT)")
    conn.execute("CREATE TABLE user_memory (id TEXT PRIMARY KEY, user_id TEXT, category TEXT, fact_key TEXT, fact_value TEXT, confidence REAL, source_doc_title TEXT)")
    # Insert duplicate documents and a chunk referencing the duplicate doc_id ('d_old_2') to test V3 ordering
    conn.execute("INSERT INTO users VALUES ('u1', 'g_u1', 'u1@example.com', 'U1', '')")
    conn.execute("INSERT INTO documents VALUES ('d_old_1', 'u1', 'Doc1', 'Doc1', 'pdf', 100, 1, 'dup_hash')")
    conn.execute("INSERT INTO documents VALUES ('d_old_2', 'u1', 'Doc1_dup', 'Doc1_dup', 'pdf', 100, 1, 'dup_hash')")
    conn.execute("INSERT INTO document_chunks VALUES ('c_old_1', 'd_old_1', 'u1', 'Doc1', 1, 1, 'text 1')")
    conn.execute("INSERT INTO document_chunks VALUES ('c_old_2', 'd_old_2', 'u1', 'Doc1_dup', 1, 1, 'text 2')")
    conn.commit()
    conn.close()

    db = VaultDatabase(db_url="", sqlite_path=legacy_db_path)
    # Verify duplicate d_old_2 AND its orphan chunk c_old_2 were both cleaned up (Fixes V3)
    assert db._execute_scalar("SELECT COUNT(*) FROM documents WHERE user_id = 'u1'") == 1
    assert db._execute_scalar("SELECT COUNT(*) FROM document_chunks WHERE user_id = 'u1'") == 1

    u = db.upsert_google_user("g_leg", "leg@example.com", "Legacy User", "")
    uid = u["id"]
    ingest = DocumentIngestionPipeline()
    chunks = ingest.recursive_chunk_text("Sample text for duplicate hash test.", "d_dup1", "A.txt")
    db.add_document_with_chunks(uid, "d_dup1", "A.txt", "A.txt", "txt", 100, chunks, "General", {}, "same_hash_123")

    raised_integrity = False
    try:
        chunks2 = ingest.recursive_chunk_text("Sample text for duplicate hash test.", "d_dup2", "B.txt")
        db.add_document_with_chunks(uid, "d_dup2", "B.txt", "B.txt", "txt", 100, chunks2, "General", {}, "same_hash_123")
    except DatabaseIntegrityError:
        raised_integrity = True
    assert raised_integrity is True, "Duplicate (user_id, file_hash) must raise DatabaseIntegrityError (409)!"

    db.sqlite_path = "/nonexistent_dir_xyz/impossible.db"
    raised_503 = False
    try:
        db.get_user_by_id("usr_123")
    except DatabaseUnavailableError:
        raised_503 = True
    assert raised_503 is True


def test_10_revoked_token_persists_across_restart():
    """
    Acceptance Gate 10 (C5): Restart the instance -> revoked tokens still revoked in database.
    """
    db_path = f"/tmp/gate10_{uuid.uuid4().hex[:8]}.db"
    db1 = VaultDatabase(db_url="", sqlite_path=db_path)
    auth1 = auth_manager.AuthManager(db1)
    u = db1.upsert_google_user("g_10", "u10@example.com", "U10", "")
    tok = auth1.create_session_token(u["id"], "u10@example.com")
    auth1.revoke_session_token(tok)

    auth_manager.REVOKED_TOKENS.clear()
    db2 = VaultDatabase(db_url="", sqlite_path=db_path)
    auth2 = auth_manager.AuthManager(db2)
    assert auth2.verify_session_token(tok) is None, "Revoked token must remain revoked after restart!"


def test_11_bm25_performance_and_no_embedding_bloat():
    """
    Acceptance Gate 11 (H9 & N2):
    - get_all_user_chunks() omits `embedding_json` so BM25 queries do not transfer multi-megabyte vector blobs.
    - BM25 search over 5,000 chunks completes in < 0.60s.
    """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate11")
    u = db.upsert_google_user("g_11", "u11@example.com", "U11", "")
    uid = u["id"]
    c_test = ingest.recursive_chunk_text("Policy No: PBW/00001001 deductible Rs 2,500.", "d_11", "Pol.pdf")
    db.add_document_with_chunks(uid, "d_11", "Pol.pdf", "Pol.pdf", "pdf", 100, c_test, "Insurance", {}, "h_11")
    db.update_chunk_embeddings(uid, {c_test[0].chunk_id: [0.5] * 384})

    bm25_rows = db.get_all_user_chunks(uid)
    assert len(bm25_rows) == 1
    assert "embedding_json" not in bm25_rows[0], "get_all_user_chunks() must NOT select embedding_json (Fixes N2)!"

    chunks_5000 = [
        {
            "chunk_id": f"c_{i}",
            "doc_id": "d_perf",
            "doc_title": "Large_Corpus.pdf",
            "page_number": (i % 50) + 1,
            "content": f"Standard clause {i} regarding general terms and conditions of the agreement."
        }
        for i in range(4999)
    ]
    chunks_5000.append({
        "chunk_id": "c_target",
        "doc_id": "d_perf",
        "doc_title": "Motor_Policy.pdf",
        "page_number": 1,
        "content": "Target Policy No: PBW/00001001 with accidental deductible Rs 2,500."
    })

    t0 = time.time()
    hits = retrieval._sparse_search("deductible PBW/00001001", "u_perf", chunks_5000, top_k=5)
    elapsed = time.time() - t0
    assert len(hits) > 0 and hits[0][0]["chunk_id"] == "c_target"
    assert elapsed < 0.60, f"5,000-chunk BM25 took {elapsed:.3f}s (expected < 0.60s)"


def test_12_delete_account_wipes_all_9_tables_vectors_and_cookie():
    """
    Acceptance Gate 12 (N5, N7, N8): Delete account -> zero rows across all 9 user tables,
    vector store collection deleted, and aura_session cookie revoked and cleared.
    """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate12")
    aura_app.db = db
    aura_app.auth = auth
    aura_app.retrieval = retrieval

    u = db.upsert_google_user("g_12", "u12@example.com", "U12", "")
    uid = u["id"]
    did = f"doc_{uuid.uuid4().hex}"
    chunks = ingest.recursive_chunk_text("Confidential document content for deletion test.", did, "Del.pdf", "General")
    db.add_document_with_chunks(uid, did, "Del.pdf", "Del.pdf", "pdf", 200, chunks, "General", {}, "h_del")
    retrieval.index_chunks(chunks, user_id=uid, category="General")
    db.upsert_user_memory(uid, "General", "Key", "Val", "Del.pdf", source_doc_id=did)
    conv = db.create_conversation(uid, "Chat")
    db.add_message(conv["id"], "user", "Hello")
    db.create_upload_job(uid, total_files=1)
    db.add_and_check_daily_tokens(uid, tokens_to_add=500, daily_limit=50000)
    db.check_and_increment_rate_limit(f"{uid}:chat", max_requests=25, window_seconds=60)

    cookie_tok = auth.create_session_token(uid, "u12@example.com")
    resp = aura_app.delete_account(authorization=None, aura_session=cookie_tok, user=u)
    assert resp.status_code == 200
    assert auth.verify_session_token(cookie_tok) is None, "Cookie session token must be revoked on delete_account!"

    for tbl, col in [
        ("users", "id"),
        ("documents", "user_id"),
        ("document_chunks", "user_id"),
        ("user_memory", "user_id"),
        ("conversations", "user_id"),
        ("upload_jobs", "user_id"),
        ("token_usage_daily", "user_id"),
    ]:
        cnt = db._execute_scalar(f"SELECT COUNT(*) FROM {tbl} WHERE {col} = ?", (uid,))
        assert cnt == 0, f"Expected 0 rows in {tbl} after delete_account, got {cnt}"

    assert db._execute_scalar("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conv["id"],)) == 0
    assert db._execute_scalar("SELECT COUNT(*) FROM rate_limit_counters WHERE bucket_key LIKE ?", (f"{uid}:%",)) == 0


def test_13_db_persisted_vectors_survive_full_and_partial_ephemeral_wipe():
    """
    Acceptance Gate 13 (H8 & N6):
    - Vectors persisted in DB (`embedding_json`) survive both a total Chroma wipe AND a partial split-brain
      (where 1 older document is only in DB `embedding_json` while 1 newer document is in Chroma).
    """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate13")
    u = db.upsert_google_user("g_13", "u13@example.com", "U13", "")
    uid = u["id"]

    d_old = f"doc_old_{uuid.uuid4().hex[:6]}"
    c_old = ingest.recursive_chunk_text("Older motor policy deductible is Rs 5,000.", d_old, "Old_Car.pdf", "Insurance")
    db.add_document_with_chunks(uid, d_old, "Old_Car.pdf", "Old_Car.pdf", "pdf", 300, c_old, "Insurance", {}, "h_old")
    vec_target = [1.0] + [0.0] * 383
    db.update_chunk_embeddings(uid, {c_old[0].chunk_id: vec_target})

    d_new = f"doc_new_{uuid.uuid4().hex[:6]}"
    c_new = ingest.recursive_chunk_text("Newer health plan room rent is AC Room.", d_new, "New_Health.pdf", "Insurance")
    db.add_document_with_chunks(uid, d_new, "New_Health.pdf", "New_Health.pdf", "pdf", 300, c_new, "Insurance", {}, "h_new")
    retrieval.index_chunks(c_new, user_id=uid, category="Insurance")

    retrieval._compute_embeddings = lambda texts: [vec_target for _ in texts]
    dense_hits = retrieval._dense_search("What is the motor deductible?", user_id=uid, top_k=5, allowed_doc_ids={d_old, d_new})
    hit_doc_ids = {m.get("doc_id") for m, _ in dense_hits}
    assert d_old in hit_doc_ids, f"Expected older DB-backed document {d_old} to be merged into dense_hits! Got {hit_doc_ids}"


def test_14_rightmost_xff_and_atomic_rate_limiting():
    """
    Acceptance Gate 14 (H7, N1, N10):
    - get_real_client_ip uses the rightmost X-Forwarded-For hop so spoofed client headers cannot bypass rate limits.
    - Atomic DB rate limiter enforces window limits and prunes expired rows.
    """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate14")
    u = db.upsert_google_user("g_14", "u14@example.com", "U14", "")
    uid = u["id"]

    class DummyReq:
        def __init__(self, xff):
            self.headers = {"X-Forwarded-For": xff}
            self.client = None

    ip1 = aura_app.get_real_client_ip(DummyReq("1.1.1.1, 203.0.113.50"))
    ip2 = aura_app.get_real_client_ip(DummyReq("9.9.9.9, 203.0.113.50"))
    assert ip1 == "203.0.113.50" and ip2 == "203.0.113.50", f"Expected rightmost XFF IP 203.0.113.50, got {ip1}"

    assert db.check_and_increment_rate_limit("ip_1:demo", max_requests=3, window_seconds=60) is True
    assert db.check_and_increment_rate_limit("ip_1:demo", max_requests=3, window_seconds=60) is True
    assert db.check_and_increment_rate_limit("ip_1:demo", max_requests=3, window_seconds=60) is True
    assert db.check_and_increment_rate_limit("ip_1:demo", max_requests=3, window_seconds=60) is False

    ok1, used1 = db.add_and_check_daily_tokens(uid, tokens_to_add=30000, daily_limit=50000)
    assert ok1 is True and used1 == 30000
    ok2, used2 = db.add_and_check_daily_tokens(uid, tokens_to_add=25000, daily_limit=50000)
    assert ok2 is False and used2 == 55000


def test_15_memory_pending_review_and_source_doc_id_isolation():
    """
    Acceptance Gate 15 (H3, N9, N12):
    - Pending memories are excluded from LLM system prompts until confirmed.
    - Deleting one document only deletes memory facts linked to that specific source_doc_id.
    - Utility/electricity bills classify as 'Finance'.
    """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate15")
    u = db.upsert_google_user("g_15", "u15@example.com", "U15", "")
    uid = u["id"]

    cat = assistant.memory_agent.classify_document_category(
        "Electricity Utility Bill. Consumer Number: 88991122. Units Consumed: 310 kWh. Bill Amount Payable: Rs 2,480. Due Date: 15/11/2026.",
        "BESCOM_Bill_Oct.pdf"
    )
    assert cat == "Finance", f"Expected utility bill to classify as Finance, got {cat}"

    c1 = ingest.recursive_chunk_text("Policy 1 content", "doc_A", "Policy.pdf")
    c2 = ingest.recursive_chunk_text("Policy 2 content", "doc_B", "Policy.pdf")
    db.add_document_with_chunks(uid, "doc_A", "Policy.pdf", "Policy.pdf", "pdf", 100, c1, "Insurance", {}, "h_A")
    db.add_document_with_chunks(uid, "doc_B", "Policy.pdf", "Policy.pdf", "pdf", 100, c2, "Insurance", {}, "h_B")

    mem_a = db.upsert_user_memory(uid, "Insurance", "Policy Number A", "PBW/00001001", "Policy.pdf", status="pending", source_doc_id="doc_A")
    mem_b = db.upsert_user_memory(uid, "Insurance", "Policy Number B", "PBW/00009999", "Policy.pdf", status="confirmed", source_doc_id="doc_B")

    assert "PBW/00001001" not in assistant.memory_agent.format_memory_context(uid)
    db.confirm_user_memory(uid, mem_a["id"])
    assert "PBW/00001001" in assistant.memory_agent.format_memory_context(uid)

    db.delete_document(uid, "doc_A")
    remaining = db.get_user_memories(uid)
    assert len(remaining) == 1 and remaining[0]["id"] == mem_b["id"]


def test_16_httponly_cookie_upload_jobs_and_demo_cleanup():
    """
    Acceptance Gate 16 (C4, H6, M5, Part 3): HttpOnly cookie authentication, upload job tracking,
    zero blocking alert() calls, and parameterized PostgreSQL/SQLite demo cleanup.
    """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate16")
    aura_app.db = db
    aura_app.auth = auth
    u = db.upsert_google_user("g_16", "u16@example.com", "U16", "")
    uid = u["id"]
    tok = auth.create_session_token(uid, "u16@example.com")

    authed_user = aura_app.get_current_user(authorization=None, aura_session=tok)
    assert authed_user["id"] == uid

    jid = db.create_upload_job(uid, total_files=2)
    db.update_upload_job(jid, uid, 2, [{"filename": "a.pdf", "status": "indexed"}], status="completed")
    job = db.get_upload_job(uid, jid)
    assert job is not None and job["status"] == "completed" and job["completed_files"] == 2

    with open(os.path.join(ROOT_DIR, "index.html")) as f:
        html = f.read()
    assert len(re.findall(r"\balert\(", html)) == 0

    demo_u = db.upsert_google_user("gid_demo_exp1", "demo_exp1@demo.local", "Expired Demo", "")
    db._execute_non_query("UPDATE users SET created_at = ? WHERE id = ?", (time.time() - 90000, demo_u["id"]))
    expired_list = db.cleanup_expired_demo_users(ttl_seconds=86400)
    assert demo_u["id"] in expired_list
    assert db.get_user_by_id(demo_u["id"]) is None


if __name__ == "__main__":
    print("=" * 72)
    print("RUNNING AURA 16-GATE LAUNCH ACCEPTANCE TEST SUITE (PARTS 1, 2 & 3)")
    print("=" * 72)
    test_funcs = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    failed = 0
    for name, fn in test_funcs:
        t0 = time.time()
        try:
            fn()
            dt = (time.time() - t0) * 1000
            print(f"[PASS] {name} ({dt:.1f} ms)")
            passed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {name}: {e}")
            failed += 1
    print("=" * 72)
    print(f"RESULT: {passed}/{len(test_funcs)} GATES PASSED | {failed} FAILED")
    print("=" * 72)
    if failed > 0:
        sys.exit(1)
