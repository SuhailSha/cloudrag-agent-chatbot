# Render Production Server Deployment Guide

This guide covers deploying the production-hardened **AURA – Personal AI Assistant Vault** on [Render](https://render.com) (or any containerized cloud host).

---

## 1. Prerequisites

1. **GitHub Account & Repository**: [github.com/SuhailSha/cloudrag-agent-chatbot](https://github.com/SuhailSha/cloudrag-agent-chatbot)
2. **Render Account**: [render.com](https://render.com)
3. **Database**: [Neon Serverless PostgreSQL](https://neon.tech)
4. **Google Cloud Console**: Authorized Web Client ID (`277695292361-386p40ds1cptp6bf1f8v1sr8mr5tbv8v.apps.googleusercontent.com`)
5. **Groq Cloud API Key**: [console.groq.com](https://console.groq.com)

---

## 2. Environment Variables Setup in Render

In your Render Web Service dashboard under **Environment**:

| KEY | VALUE | DESCRIPTION |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgres://user:password@ep-...neon.tech/neondb?sslmode=require` | Your Neon Postgres Connection String |
| `JWT_SECRET` | *(Generate a 64-char random hex string)* | Mandatory secret for JWT signing in production |
| `GOOGLE_CLIENT_ID` | `277695292361-386p40ds1cptp6bf1f8v1sr8mr5tbv8v.apps.googleusercontent.com` | Google OAuth Client ID |
| `GROQ_API_KEY` | `gsk_...` | Groq LLM API Key |
| `PINECONE_API_KEY` | *(Optional)* | Pinecone API key if using cloud vectors |
| `ALLOWED_ORIGINS` | `https://cloudrag-agent-chatbot.onrender.com` | Whitelisted CORS origins |
| `ALLOW_SQLITE_FALLBACK` | `false` | Fail closed if PostgreSQL connection drops |

---

## 3. Deployment Steps

1. **Commit and Push Code to GitHub**:
   ```bash
   git add .
   git commit -m "Production hardening: Phase 1-5 complete"
   git push origin main
   ```
2. **Deploy on Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com).
   - Select `cloudrag-agent-chatbot`.
   - Click **Manual Deploy** &rarr; **Clear build cache & deploy**.
3. **Verify Health Endpoints**:
   - Open `https://cloudrag-agent-chatbot.onrender.com/health/live` in your browser &rarr; should return `{"status": "alive"}`.
   - Open `https://cloudrag-agent-chatbot.onrender.com/health/ready` &rarr; should return `{"status": "ready", "database": "postgres", "vector_provider": "chroma_local"}`.
