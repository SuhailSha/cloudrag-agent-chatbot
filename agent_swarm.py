import os
import json
import time
import requests
from typing import List, Dict, Any, Optional

class EnterprisePersonalAssistant:
    """
    Consumer-grade personal AI assistant.
    Provides human-like, beautifully formatted answers with citation pills,
    contextual memory across conversation sessions, and zero technical jargon.
    """
    def __init__(self, retrieval_pipe, database):
        self.retrieval = retrieval_pipe
        self.db = database
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    def _get_active_groq_models(self, api_key: str) -> List[str]:
        default_candidates = [
            os.getenv("GROQ_MODEL", "").strip(),
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "gemma2-9b-it"
        ]
        default_candidates = [m for m in default_candidates if m]

        if hasattr(self, '_cached_groq_models') and self._cached_groq_models:
            return self._cached_groq_models

        try:
            r = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=4
            )
            if r.status_code == 200:
                data = r.json().get("data", [])
                live_ids = {item["id"] for item in data if not item.get("id", "").startswith("whisper")}
                matched = [m for m in default_candidates if m in live_ids]
                for item in data:
                    mid = item.get("id", "")
                    if ("llama" in mid.lower() or "gpt" in mid.lower()) and mid not in matched:
                        matched.append(mid)
                if matched:
                    self._cached_groq_models = matched
                    return matched
        except Exception:
            pass

        self._cached_groq_models = default_candidates
        return default_candidates

    def _call_groq_llm(self, messages: List[Dict[str, str]], api_key: str) -> Optional[str]:
        candidate_models = self._get_active_groq_models(api_key)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        for model_id in candidate_models:
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": 0.2
            }
            try:
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=15
                )
                if r.status_code == 200:
                    data = r.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    if text:
                        return text
                else:
                    print(f"[Groq API Warning] Model {model_id} returned {r.status_code}. Trying next model...")
                    continue
            except Exception as e:
                print(f"[Groq Exception for {model_id}]: {e}")
                continue
        return None

    def respond(self, user_id: str, conversation_id: str, query: str, user_groq_key: str = None) -> Dict[str, Any]:
        """
        Executes personal document RAG and saves conversation messages.
        """
        active_key = (user_groq_key or self.groq_api_key or "").strip()

        # 1. Save User Message
        self.db.add_message(conversation_id, sender="user", content=query)

        # 2. Retrieve Relevant Context strictly from User's Isolated Namespace
        hits = self.retrieval.retrieve(query, user_id=user_id, top_k=5)
        
        citations = []
        context_blocks = []
        for i, (item, score) in enumerate(hits):
            doc_title = item.get("doc_title", "Document")
            # Create a clean, human-friendly citation label
            clean_title = doc_title.replace(".pdf", "").replace(".docx", "").replace("_", " ")
            cit = {
                "id": f"cit_{i+1}",
                "doc_title": doc_title,
                "clean_label": clean_title,
                "snippet": item.get("content", "")[:300],
                "full_content": item.get("content", "")
            }
            citations.append(cit)
            context_blocks.append(f"[Source: {clean_title}]\n{item.get('content', '')}")

        context_text = "\n\n---\n\n".join(context_blocks)

        # 3. Assemble Conversational History (Last 6 messages for context)
        past_msgs = self.db.get_conversation_messages(user_id, conversation_id)
        # Exclude the just-added user message
        history_msgs = past_msgs[:-1][-6:]

        system_instruction = (
            "You are a helpful, warm, and highly capable personal AI assistant. "
            "You help the user understand and manage their confidential personal documents (such as insurance policies, medical records, resumes, receipts, and agreements). "
            "Guidelines:\n"
            "- Answer the user's question directly, clearly, and concisely.\n"
            "- Present numbers, key clauses, dates, policies, or technical skills in clean, parallel bullet points.\n"
            "- Never mention technical jargon like 'vector embeddings', 'chunks', 'HNSW', or database IDs.\n"
            "- If citing facts from documents, cite naturally using the document's name in brackets, e.g. [Zurich Car Policy].\n"
            "- If the documents do not contain the answer, politely state what is missing and offer practical next steps."
        )

        llm_messages = [{"role": "system", "content": system_instruction}]
        for m in history_msgs:
            role = "user" if m["sender"] == "user" else "assistant"
            llm_messages.append({"role": role, "content": m["content"]})

        user_content = f"Personal Document Context:\n{context_text}\n\nUser Question: {query}" if context_text else query
        llm_messages.append({"role": "user", "content": user_content})

        # 4. Generate LLM Answer
        answer = None
        if active_key:
            answer = self._call_groq_llm(llm_messages, active_key)

        # High quality local fallback if Groq is offline
        if not answer:
            if citations:
                answer = f"Based on your uploaded documents, here is what I found regarding **\"{query}\"**:\n\n"
                for cit in citations[:3]:
                    answer += f"- **From {cit['clean_label']}**:\n  {cit['snippet'][:250]}...\n\n"
            else:
                answer = (
                    f"I searched your private documents for **\"{query}\"**, but couldn't find any matching details.\n\n"
                    "You can upload your policy, contract, or resume in **My Vault** to ask questions about it."
                )

        # 5. Save Assistant Message with Citations to Database
        saved_msg = self.db.add_message(conversation_id, sender="assistant", content=answer, citations=citations)

        return {
            "message_id": saved_msg["id"],
            "conversation_id": conversation_id,
            "answer": answer,
            "citations": citations
        }
