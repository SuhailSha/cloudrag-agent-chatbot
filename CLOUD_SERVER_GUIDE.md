# 24/7 Free Cloud Server Deployment Guide

This package runs your **complete personal RAG chatbot on a 24/7 cloud server for $0 (Zero Paid)**. You do NOT need to leave your computer running. You can access it securely from your phone, laptop, or anywhere on the internet.

---

## Why Hugging Face Spaces is the Best 100% Free Cloud Host

| Feature | Hugging Face Spaces (Free Tier) | Standard Paid VPS |
| :--- | :--- | :--- |
| **Monthly Cost** | **$0.00 / month forever** | $10 - $40 / month |
| **Credit Card Required?** | **NO** | Yes |
| **Server Uptime** | **Runs 24/7 on the cloud** | Runs 24/7 |
| **Hardware** | **2 vCPUs, 16 GB RAM, 50 GB Storage** | 1 vCPU, 1 GB RAM |
| **Privacy Mode** | **"Private" Space toggle** (Only you can access) | Manual firewall setup |
| **Mobile Access** | Public HTTPS Link (e.g. `https://username-bot.hf.space`) | Custom domain/IP |

---

## 3-Minute Cloud Deployment (No Code / No Terminal Needed)

### Step 1: Create a Free Hugging Face Account & Space
1. Sign up for free at [Hugging Face](https://huggingface.co/join) (takes 30 seconds, no credit card).
2. Go to [Hugging Face Spaces New](https://huggingface.co/new-space).
3. Fill in the details:
   - **Space name**: `my-personal-rag`
   - **License**: Apache 2.0
   - **Select the Space SDK**: Choose **Docker**
   - **Docker template**: Choose **Blank**
   - **Space Hardware**: **CPU basic • 2 vCPU • 16GB • Free**
   - **Privacy**: Select **Private** (Crucial: makes sure nobody on the internet can see your personal documents!).
4. Click **Create Space**.

---

### Step 2: Upload the Deployment Files
1. Download and unzip the **`cloud_personal_rag_server.zip`** package.
2. In your new Hugging Face Space page, click on the **Files** tab at the top.
3. Click **Add file** &rarr; **Upload files**.
4. Drag and drop all the unzipped files:
   - `Dockerfile`
   - `app.py`
   - `ingestion_pipeline.py`
   - `retrieval_pipeline.py`
   - `agent_swarm.py`
   - `requirements.txt`
   - `static/index.html`
5. Click **Commit changes to main**.

---

### Step 3: Access Your Chatbot on Your Phone & Computer
1. Hugging Face will automatically build your Docker container (takes about 2 minutes).
2. Once the status shows **Running**, you will see your public HTTPS URL at the top:
   👉 **`https://<your-username>-my-personal-rag.hf.space`**
3. Open this link on your phone, tablet, or laptop from anywhere in the world!
4. Upload your personal PDFs, text files, and policies. The system will:
   - Run the **Ingestion Pipeline** (extract text from PDFs, clean and chunk).
   - Run the **ChromaDB Vector Pipeline** (embed with Sentence-Transformers).
   - Coordinate the **Autonomous Multi-Agent Swarm** (Supervisor, RAG Retriever, Synthesizer, Task Extractor, and Critic).
