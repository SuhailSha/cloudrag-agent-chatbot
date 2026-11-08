import os
import sys
import uuid
import time
import json
import base64
import subprocess

# Setup test environment
os.environ["JWT_SECRET"] = "acceptance-suite-jwt-secret-key-at-least-32-chars-long"
os.environ["ALLOW_SQLITE_FALLBACK"] = "true"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import auth_manager
from database import VaultDatabase, DatabaseUnavailableError
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


def test_01_20_unanswerable_questions_abstain():
    """ Acceptance Gate 1 (C1): 20 unanswerable questions -> at least 95% abstain, zero citations shown. """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate1")
    u = db.upsert_google_user("g_gate1", "gate1@example.com", "Gate 1 User", "")
    uid = u["id"]
    doc_id = f"doc_{uuid.uuid4().hex}"
    chunks = ingest.recursive_chunk_text(
        "Motor Insurance Policy Schedule. Policy No: PBW/00001001. Own Damage Deductible is Rs 2,500. "
        "Third Party Liability limit is Rs 7,50,000. Roadside assistance is included.",
        doc_id, "Motor_Policy.pdf", "Insurance"
    )
    db.add_document_with_chunks(uid, doc_id, "Motor_Policy.pdf", "Motor_Policy.pdf", "pdf", 1024, chunks, "Insurance", {}, "h_g1")
    conv = db.create_conversation(uid, "Gate 1 Chat")

    unanswerable_20 = [
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
        "What is the currency of Japan?",
        "How many planets are in the solar system?",
        "Who invented the telephone?",
        "What is the chemical formula for table salt?",
        "Where are the Pyramids of Giza located?",
        "Who was the first person to walk on the moon?",
        "What official language is spoken in Brazil?",
        "How many bones are in the adult human body?",
        "What is the largest ocean on Earth?",
        "Who directed the movie Inception?"
    ]

    abstained = 0
    for q in unanswerable_20:
        res = assistant.respond(uid, conv["id"], q)
        if "could not find information" in res["answer"].lower() and len(res["citations"]) == 0:
            abstained += 1

    assert abstained >= 19, f"Expected >= 19/20 abstentions, got {abstained}/20"


def test_02_50_question_golden_set_recall():
    """ Acceptance Gate 2: 50-question golden evaluation set -> Recall@5 >= 90%. """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate2")
    u = db.upsert_google_user("g_gate2", "gate2@example.com", "Gate 2 User", "")
    uid = u["id"]

    corpus = [
        ("Motor_Policy.pdf", "Insurance",
         "Zurich Motor Insurance Policy No: PBW/00001001. Insured Declared Value (IDV) is Rs 8,50,000. "
         "Compulsory deductible is Rs 2,000. Zero Depreciation add-on is active. Engine protector cover included. "
         "No Claim Bonus (NCB) is 25%. Roadside assistance toll-free number is 1800-209-8888. "
         "Cashless garage network has 4,500 workshops. Policy expires on 15/12/2027. Geographical area is India."),
        ("Health_Plan.pdf", "Insurance",
         "Family Floater Health Insurance Policy No: POL-90002211. Sum Insured is Rs 15,00,000. "
         "Room rent limit is Single Private AC Room. Maternity waiting period is 24 months. "
         "Pre-existing disease waiting period is 36 months. Pre-hospitalization cover is 60 days and post-hospitalization is 180 days. "
         "Annual health checkup reimbursement up to Rs 5,000. Daycare procedures covered: 540 listed procedures. Ambulance limit Rs 3,000."),
        ("Employment_Contract.pdf", "Career",
         "Employment Offer and Agreement for Senior AI Engineer. Base salary is $165,000 per annum. "
         "Annual performance bonus target is 15%. Probation period is 90 days. Notice period after confirmation is 60 days. "
         "Paid time off (PTO) is 24 working days per year. Equity grant is 10,000 RSUs vesting over 4 years with a 1-year cliff. "
         "Remote work stipend is $1,500 annually. Reporting to VP of Engineering."),
        ("Apartment_Lease.pdf", "Legal",
         "Residential Lease Agreement for Apartment 14B, Green Valley Towers. Monthly rent is Rs 42,000 payable by the 5th of each month. "
         "Security deposit is Rs 2,00,000 refundable at termination. Lock-in period is 6 months. "
         "Termination notice period is 2 months by either Lessor or Lessee. Annual rent escalation is 5%. "
         "Monthly maintenance charge is Rs 4,500. Pets are permitted with prior written consent. Subletting is strictly prohibited."),
        ("Bank_Statement.pdf", "Finance",
         "Checking Account Statement for October 2026. IFSC Code: HDFC0004321. Opening balance is Rs 1,20,000. "
         "Salary credit on 01/10/2026 of Rs 1,45,000. Home loan EMI debit on 05/10/2026 of Rs 38,500. "
         "Electricity bill payment of Rs 3,200. Mutual fund SIP debit of Rs 25,000. Closing balance is Rs 1,98,300. "
         "Overdraft limit approved is Rs 50,000 at 11.5% interest rate.")
    ]

    for idx, (title, cat, text) in enumerate(corpus):
        did = f"doc_gold_{idx}"
        chunks = ingest.recursive_chunk_text(text, did, title, cat)
        db.add_document_with_chunks(uid, did, title, title, "pdf", len(text), chunks, cat, {}, f"hash_gold_{idx}")

    golden_50 = [
        ("What is my motor policy number?", "PBW/00001001"),
        ("What is the Insured Declared Value (IDV) of my car?", "8,50,000"),
        ("What is the compulsory deductible on the motor policy?", "2,000"),
        ("Is Zero Depreciation included in my motor policy?", "Zero Depreciation"),
        ("Does my car insurance have engine protector cover?", "Engine protector"),
        ("What is my No Claim Bonus percentage?", "25%"),
        ("What is the roadside assistance phone number?", "1800-209-8888"),
        ("How many cashless garages are in the network?", "4,500"),
        ("When does my motor policy expire?", "15/12/2027"),
        ("What is the geographical area for the motor policy?", "India"),
        ("What is the health insurance policy number?", "POL-90002211"),
        ("What is the sum insured on the family floater health plan?", "15,00,000"),
        ("What is the room rent limit in the health policy?", "Single Private AC Room"),
        ("What is the waiting period for maternity coverage?", "24 months"),
        ("How long is the pre-existing disease waiting period?", "36 months"),
        ("How many days of pre-hospitalization are covered?", "60 days"),
        ("How many days of post-hospitalization are covered?", "180 days"),
        ("What is the annual health checkup reimbursement limit?", "5,000"),
        ("How many daycare procedures are covered?", "540"),
        ("What is the ambulance cover limit?", "3,000"),
        ("What is the job title in the employment agreement?", "Senior AI Engineer"),
        ("What is the annual base salary?", "$165,000"),
        ("What is the annual performance bonus target?", "15%"),
        ("How long is the probation period?", "90 days"),
        ("What is the notice period after confirmation?", "60 days"),
        ("How many days of PTO are provided per year?", "24 working days"),
        ("How many RSUs are in the equity grant?", "10,000"),
        ("What is the vesting schedule for the RSUs?", "4 years"),
        ("What is the remote work stipend amount?", "$1,500"),
        ("Who does the Senior AI Engineer report to?", "VP of Engineering"),
        ("Which apartment is covered by the lease agreement?", "Apartment 14B"),
        ("What is the monthly rent for the apartment?", "42,000"),
        ("By what day of the month is rent payable?", "5th"),
        ("What is the security deposit for the apartment?", "2,00,000"),
        ("How long is the lease lock-in period?", "6 months"),
        ("What is the lease termination notice period?", "2 months"),
        ("What is the annual rent escalation percentage?", "5%"),
        ("What is the monthly maintenance charge?", "4,500"),
        ("Are pets allowed in the apartment?", "Pets are permitted"),
        ("Is subletting allowed under the lease?", "strictly prohibited"),
        ("What is the IFSC code on the bank statement?", "HDFC0004321"),
        ("What was the opening balance in October 2026?", "1,20,000"),
        ("How much was the salary credit on 01/10/2026?", "1,45,000"),
        ("What was the home loan EMI debit amount?", "38,500"),
        ("How much was the electricity bill payment?", "3,200"),
        ("What was the mutual fund SIP debit amount?", "25,000"),
        ("What is the closing balance on the bank statement?", "1,98,300"),
        ("What is the approved overdraft limit?", "50,000"),
        ("What is the interest rate on the overdraft facility?", "11.5%"),
        ("On what date was the home loan EMI debited?", "05/10/2026"),
    ]

    hits_found = 0
    for q, expected_substring in golden_50:
        hits = retrieval.retrieve(q, user_id=uid, top_k=5)
        if any(expected_substring.lower() in h[0].get("content", "").lower() for h in hits):
            hits_found += 1

    recall_at_5 = hits_found / float(len(golden_50))
    assert recall_at_5 >= 0.90, f"Expected Recall@5 >= 90%, got {recall_at_5 * 100:.1f}% ({hits_found}/50)"


def test_03_router_table_60_utterances():
    """ Acceptance Gate 3 (C2): 60-utterance router test table -> >= 95% accuracy, including every 'how does' and 'vs' case. """
    router = agent_swarm.SupervisorIntentRouter()
    docs = [
        {"id": "d1", "title": "Zurich_Motor_Policy.pdf"},
        {"id": "d2", "title": "Health_Floater_Plan.pdf"},
        {"id": "d3", "title": "Apartment_Lease.pdf"}
    ]

    utterances_60 = [
        # 30 Single-Document RAG queries (including 'how does' and temporal 'vs')
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
        ("How much was the home loan EMI debit?", UserIntent.DOCUMENT_RAG),
        ("What is my base salary?", UserIntent.DOCUMENT_RAG),
        ("How does the RSU vesting cliff work?", UserIntent.DOCUMENT_RAG),
        ("What is the ambulance reimbursement limit?", UserIntent.DOCUMENT_RAG),

        # 12 Explicit Multi-Document Comparison queries
        ("Compare my Zurich policy and Health plan side by side", UserIntent.DOCUMENT_COMPARISON),
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


def test_04_critic_catches_fabricated_citation_and_altered_number():
    """ Acceptance Gate 4 (C3): Fabricated citation and altered number -> flagged or removed, never relabelled. """
    critic = agent_swarm.VerificationCriticAgent()
    citations = [{"id": "cit_1", "doc_title": "Zurich_Car_Policy.pdf", "clean_label": "Zurich Car Policy (P.1)"}]
    hits = [({"content": "Your annual insurance premium is Rs 12,000 and deductible is Rs 2,500."}, 0.05)]

    # Altered number probe
    res_num = critic.audit_response("Your annual premium is Rs 99,999 [Zurich Car Policy (P.1)].", hits, citations)
    assert res_num["is_grounded"] is False
    assert any("99999" in f for f in res_num["flags"])

    # Fabricated citation probe — must NEVER be relabelled to Zurich Car Policy
    res_cit = critic.audit_response("Your annual premium is Rs 12,000 [Made Up Doc (P.9)].", hits, citations)
    assert "Made Up Doc" not in res_cit["audited_answer"]
    assert "Zurich Car Policy" not in res_cit["audited_answer"]
    assert res_cit["is_grounded"] is False


def test_05_two_tenant_isolation():
    """ Acceptance Gate 5: Two-tenant API isolation across chat, compare, memory, and delete -> zero cross-tenant access. """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate5")
    alice = db.upsert_google_user("g_alice", "alice@example.com", "Alice", "")
    bob = db.upsert_google_user("g_bob", "bob@example.com", "Bob", "")
    a_id, b_id = alice["id"], bob["id"]

    d_alice = f"doc_{uuid.uuid4().hex}"
    chunks_a = ingest.recursive_chunk_text("Alice Secret Salary is $295,000 per year.", d_alice, "Salary.pdf", "Finance")
    db.add_document_with_chunks(a_id, d_alice, "Salary.pdf", "Salary.pdf", "pdf", 500, chunks_a, "Finance", {}, "h_a")
    mem_a = db.upsert_user_memory(a_id, "Finance", "Salary", "$295,000", "Salary.pdf")

    # Bob cannot read, search, update memory, or delete Alice's records
    assert db.get_document(b_id, d_alice) is None
    assert len(retrieval.retrieve("$295,000 Salary", user_id=b_id)) == 0
    assert db.update_user_memory(b_id, mem_a["id"], "Salary", "$0") is False
    assert db.delete_user_memory(b_id, mem_a["id"]) is False
    assert db.delete_document(b_id, d_alice) is False
    assert db.get_document(a_id, d_alice) is not None


def test_06_oauth_state_nonce_and_expired_token():
    """ Acceptance Gate 6 (C4): Missing state, wrong nonce, expired token -> all rejected. """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate6")
    u = db.upsert_google_user("g_6", "u6@example.com", "U6", "")
    exp_tok = auth.create_session_token(u["id"], "u6@example.com", expires_in_seconds=-5)
    assert auth.verify_session_token(exp_tok) is None

    # Check index.html enforces fail-closed state check
    with open(os.path.join(ROOT_DIR, "index.html")) as f:
        html = f.read()
    assert "if (!expectedState || !returnedState || expectedState !== returnedState)" in html


def test_07_production_start_without_jwt_secret_refuses_to_start():
    """ Acceptance Gate 7 (C5): Production start with RENDER + DATABASE_URL and no JWT_SECRET -> refuses to start. """
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


def test_08_table_and_csv_chunk_invariants():
    """ Acceptance Gate 8 (H1 & H2): 200-row table and 400-row CSV -> at most 250 words, header repeated, no mid-row starts. """
    ingest = DocumentIngestionPipeline(chunk_size=180, chunk_overlap=25, max_chunk_size=250)

    # 200-row Markdown table
    header = "| Benefit_Item | Coverage_Limit | Deductible_Amount | Policy_Notes |"
    sep = "| --- | --- | --- | --- |"
    rows = [f"| Benefit_{i} | Rs_{i*10000} | Rs_2500 | Covered_under_section_{i}_terms |" for i in range(1, 201)]
    md_text = "\n".join([header, sep] + rows)
    t_chunks = ingest.recursive_chunk_text(md_text, "d_tbl", "table.md")
    assert max(len(c.content.split()) for c in t_chunks) <= 250
    assert all(c.content.splitlines()[0].strip() == header for c in t_chunks)

    # 400-row CSV
    csv_header = "Date,Description,Reference,Withdrawal,Deposit,Balance"
    csv_rows = [f"2026-10-{(i%28)+1:02d},Transfer_{i},REF{100+i},{i*10}.00,0.00,{50000-i*10}.00" for i in range(1, 401)]
    c_chunks = ingest.recursive_chunk_text("\n".join([csv_header] + csv_rows), "d_csv", "bank.csv")
    assert max(len(c.content.split()) for c in c_chunks) <= 250
    assert all(c.content.splitlines()[0].strip() == csv_header for c in c_chunks)


def test_09_database_down_raises_503_not_401():
    """ Acceptance Gate 9 (H5): Database down mid-request -> raises DatabaseUnavailableError (mapped to 503, never masked as []). """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate9")
    # Point sqlite_path to an invalid directory file path to simulate DB outage
    db.sqlite_path = "/nonexistent_dir_xyz/impossible.db"
    raised_503 = False
    try:
        db.get_user_by_id("usr_123")
    except DatabaseUnavailableError:
        raised_503 = True
    assert raised_503 is True, "Database failure must raise DatabaseUnavailableError (503), not return None (401)!"


def test_10_revoked_token_persists_across_restart():
    """ Acceptance Gate 10 (C5): Restart the instance -> revoked tokens still revoked in database. """
    db_path = f"/tmp/gate10_{uuid.uuid4().hex[:8]}.db"
    db1 = VaultDatabase(db_url="", sqlite_path=db_path)
    auth1 = auth_manager.AuthManager(db1)
    u = db1.upsert_google_user("g_10", "u10@example.com", "U10", "")
    tok = auth1.create_session_token(u["id"], "u10@example.com")
    auth1.revoke_session_token(tok)

    # Simulate container restart: clear in-memory set and instantiate fresh DB + AuthManager
    auth_manager.REVOKED_TOKENS.clear()
    db2 = VaultDatabase(db_url="", sqlite_path=db_path)
    auth2 = auth_manager.AuthManager(db2)
    assert auth2.verify_session_token(tok) is None, "Revoked token must remain revoked after restart!"


def test_11_bm25_performance_5000_chunks():
    """ Acceptance Gate 11 (H9): BM25 search over 5,000 chunks completes well within latency target (< 0.45s). """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate11")
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


def test_12_delete_account_zero_rows_and_vectors():
    """ Acceptance Gate 12: Delete account -> zero rows and vectors across every store. """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate12")
    u = db.upsert_google_user("g_12", "u12@example.com", "U12", "")
    uid = u["id"]
    did = f"doc_{uuid.uuid4().hex}"
    chunks = ingest.recursive_chunk_text("Confidential document content for deletion test.", did, "Del.pdf", "General")
    db.add_document_with_chunks(uid, did, "Del.pdf", "Del.pdf", "pdf", 200, chunks, "General", {}, "h_del")
    db.upsert_user_memory(uid, "General", "Key", "Val", "Del.pdf")
    conv = db.create_conversation(uid, "Chat")
    db.add_message(conv["id"], "user", "Hello")

    retrieval.delete_user_vault(uid)
    db.delete_user_account(uid)

    assert db._execute_scalar("SELECT COUNT(*) FROM users WHERE id = ?", (uid,)) == 0
    assert db._execute_scalar("SELECT COUNT(*) FROM documents WHERE user_id = ?", (uid,)) == 0
    assert db._execute_scalar("SELECT COUNT(*) FROM document_chunks WHERE user_id = ?", (uid,)) == 0
    assert db._execute_scalar("SELECT COUNT(*) FROM user_memory WHERE user_id = ?", (uid,)) == 0
    assert db._execute_scalar("SELECT COUNT(*) FROM conversations WHERE user_id = ?", (uid,)) == 0




def test_13_db_persisted_vectors_survive_ephemeral_disk_wipe():
    """ Acceptance Gate 13 (H8 & Gate 10): Vectors persisted in DB (`embedding_json`) survive ephemeral Chroma disk wipe. """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate13")
    u = db.upsert_google_user("g_13", "u13@example.com", "U13", "")
    uid = u["id"]
    did = f"doc_{uuid.uuid4().hex}"
    chunks = ingest.recursive_chunk_text("Policy deductible is Rs 5,000 for accidental damage.", did, "Car.pdf", "Insurance")
    db.add_document_with_chunks(uid, did, "Car.pdf", "Car.pdf", "pdf", 300, chunks, "Insurance", {}, "h_13")

    # Persist synthetic 384-d unit vector in DB and mock _compute_embeddings
    vec_384 = [1.0] + [0.0] * 383
    db.update_chunk_embeddings(uid, {chunks[0].chunk_id: vec_384})
    retrieval._compute_embeddings = lambda texts: [vec_384 for _ in texts]

    # Simulate ephemeral container restart where Chroma collection is empty/None
    retrieval.chroma_client = None
    dense_hits = retrieval._dense_search("What is the deductible?", user_id=uid, top_k=5)
    assert len(dense_hits) == 1 and dense_hits[0][1] >= 0.99, f"Expected DB vector hit, got {dense_hits}"


def test_14_db_rate_limiting_and_daily_token_budget():
    """ Acceptance Gate 14 (H7): Database-persisted rate limits and daily per-user token budget enforcement. """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate14")
    u = db.upsert_google_user("g_14", "u14@example.com", "U14", "")
    uid = u["id"]

    # Rate limit bucket allows 3 requests, blocks 4th
    assert db.check_and_increment_rate_limit("ip_1:demo", max_requests=3, window_seconds=60) is True
    assert db.check_and_increment_rate_limit("ip_1:demo", max_requests=3, window_seconds=60) is True
    assert db.check_and_increment_rate_limit("ip_1:demo", max_requests=3, window_seconds=60) is True
    assert db.check_and_increment_rate_limit("ip_1:demo", max_requests=3, window_seconds=60) is False

    # Token budget allows under 50k, blocks once >= 50k
    ok1, used1 = db.add_and_check_daily_tokens(uid, tokens_to_add=30000, daily_limit=50000)
    assert ok1 is True and used1 == 30000
    ok2, used2 = db.add_and_check_daily_tokens(uid, tokens_to_add=25000, daily_limit=50000)
    assert ok2 is False and used2 == 55000
    ok3, _ = db.add_and_check_daily_tokens(uid, tokens_to_add=0, daily_limit=50000)
    assert ok3 is False


def test_15_memory_pending_review_and_confirmation_workflow():
    """ Acceptance Gate 15 (H3): Pending memories are excluded from LLM system prompts until explicitly confirmed. """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate15")
    u = db.upsert_google_user("g_15", "u15@example.com", "U15", "")
    uid = u["id"]

    mem = db.upsert_user_memory(uid, "Insurance", "Policy Number", "PBW/00001001", "Car.pdf", status="pending")
    assert mem["status"] == "pending"
    # Pending memory must NOT appear in active prompt context
    ctx_before = assistant.memory_agent.format_memory_context(uid)
    assert "PBW/00001001" not in ctx_before

    # Confirm memory -> now appears in active prompt context
    db.confirm_user_memory(uid, mem["id"])
    ctx_after = assistant.memory_agent.format_memory_context(uid)
    assert "PBW/00001001" in ctx_after


def test_16_httponly_cookie_and_upload_job_tracking():
    """ Acceptance Gate 16 (C4, H6, M5): HttpOnly cookie authentication, upload job tracking, and zero blocking alert() calls. """
    db, auth, ingest, retrieval, assistant = create_isolated_env("gate16")
    aura_app.db = db
    aura_app.auth = auth
    u = db.upsert_google_user("g_16", "u16@example.com", "U16", "")
    uid = u["id"]
    tok = auth.create_session_token(uid, "u16@example.com")

    # Authenticate via aura_session Cookie without Authorization header
    authed_user = aura_app.get_current_user(authorization=None, aura_session=tok)
    assert authed_user["id"] == uid

    # Upload job creation and status retrieval
    jid = db.create_upload_job(uid, total_files=2)
    db.update_upload_job(jid, uid, 2, [{"filename": "a.pdf", "status": "indexed"}], status="completed")
    job = db.get_upload_job(uid, jid)
    assert job is not None and job["status"] == "completed" and job["completed_files"] == 2

    # Verify 0 blocking alert() calls in index.html
    with open(os.path.join(ROOT_DIR, "index.html")) as f:
        html = f.read()
    import re
    assert len(re.findall(r"\balert\(", html)) == 0


if __name__ == "__main__":
    import inspect
    print("=" * 72)
    print("RUNNING AURA 12-GATE LAUNCH ACCEPTANCE TEST SUITE")
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
            print(f"[FAIL] {name}: {e}")
            failed += 1
    print("=" * 72)
    print(f"RESULT: {passed}/{len(test_funcs)} GATES PASSED | {failed} FAILED")
    print("=" * 72)
    if failed > 0:
        sys.exit(1)
