# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

- GitHub: https://github.com/hamanorio/claude_exe (private)

## Projects in this repo

This monorepo contains two independent projects:

### 1. `myhome/` — Static Japanese personal website

A static HTML/CSS/JS personal site. All dynamic content is fetched at runtime from Supabase — **no build step**. See `myhome/CLAUDE.md` for full details.

**Local dev:**
```bash
python3 -m http.server 8080 --directory myhome/
```

**Content management:** Add/edit posts and projects directly in the Supabase dashboard (`posts` and `projects` tables). Changes reflect immediately on the live site without a redeploy.

**Deployed at:** `https://hamanorio.github.io/claude_exe/Claude_Code/myhome/`

---

### 2. `ai-chat/` — AI chat web application

A Next.js app using the App Router with Hono as the API layer, Prisma ORM, PostgreSQL, and Mastra to wrap the Claude API.

**Dev commands (run from `ai-chat/`):**
```bash
npm run dev       # Start dev server (localhost:3000)
npm run build     # Production build
npm run lint      # ESLint
npx prisma generate   # Regenerate Prisma client after schema changes
```

**Architecture:**
- `src/app/page.tsx` — Chat UI entry point
- `src/app/api/[...route]/route.ts` — Catches all API routes and delegates to the Hono app
- `src/lib/hono/index.ts` — All API route handlers (`POST /api/chat`, `GET /api/messages`, `DELETE /api/messages`)
- `src/lib/mastra/index.ts` — Defines the `chatAgent` using `@mastra/core/agent` with `anthropic/claude-sonnet-4-20250514`
- `src/lib/prisma.ts` — Prisma client singleton
- `src/components/Chat/` — UI components: `ChatContainer`, `MessageList`, `MessageItem`, `ChatInput`
- `prisma/schema.prisma` — `Message` model (PostgreSQL); stores `role`, `content`, optional `imageBase64`/`imageMediaType`

**API flow:** `ChatInput` → `POST /api/chat` → saves user message to DB → fetches full history → streams response via `chatAgent.stream()` → saves assistant response to DB on stream complete.

**Required environment variables (`ai-chat/.env.local`):**
```
ANTHROPIC_API_KEY=
DATABASE_URL=          # PostgreSQL connection string
```

**Deployment:** Vercel (set env vars in Vercel dashboard).

---

### 3. `todo/` — Vanilla JS todo app

A minimal static todo app (HTML/CSS/JS) with no build step. State persists in `localStorage`. No framework, no dependencies.

---

### 4. `todo-app/` and `todo2/` — Next.js todo apps

Two independent Next.js 16 (App Router) + Tailwind CSS 4 todo app experiments. No backend — each is a standalone frontend-only project.

**Dev commands (run from each directory):**
```bash
npm run dev   # localhost:3000
npm run build
npm run lint
```

---

### 5. `webapp/ai-chat/` — Python/FastAPI AI memo app

A stateless AI memo assistant using Gemini API + Notion as the persistent store. No application-level database.

**Stack:** FastAPI (Python), Gemini API, Notion API, deployed to Vercel.

**Dev commands (from `webapp/ai-chat/`):**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn api.index:app --reload --host 0.0.0.0   # localhost:8000
```

**Required env vars (`webapp/ai-chat/.env`):**
```
GEMINI_API_KEY=
NOTION_TOKEN=          # optional, for Notion save feature
NOTION_DATABASE_ID=    # optional
```

**Architecture:** `public/` (static frontend) ↔ `api/index.py` (FastAPI routes) ↔ `api/ai.py` (Gemini) / `api/notion.py` (Notion persistence).
