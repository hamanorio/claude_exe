# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Static Japanese personal website hosted on GitHub Pages. All dynamic content (blog posts, projects) is fetched at runtime from Supabase — there is no build step.

**Live URL:** `https://hamanorio.github.io/claude_exe/Claude_Code/myhome/`

## Local Development

```bash
python3 -m http.server 8080 --directory /Users/hamanorio/Claude_Code/myhome
# Open http://localhost:8080
```

> Do not use `npx serve` — it caches 301 redirects for `.html` URLs that break hash-based routing.

## Architecture

### Content is Supabase-driven (no build step)

All pages fetch data from Supabase at page load. `js/supabase.js` exposes three globals used across pages:
- `db` — Supabase client
- `formatDate(dateStr)` — formats ISO date to Japanese `YYYY年M月D日`
- `imageUrl(filename)` — returns public Storage URL from the `0505` bucket

The Supabase CDN (`@supabase/supabase-js@2`) is loaded via `<script>` tag before `supabase.js` in every page.

### Routing

`blog/post.html` is the **single page for all blog posts**. It reads the post ID from `location.hash` (`post.html#7` → fetches post with `id = 7`). The individual `blog/post1.html`, `post2.html`, `post3.html` are legacy files and are no longer used.

### Supabase Schema

**`posts` table**
| column | type | notes |
|--------|------|-------|
| id | serial PK | |
| title | text | |
| date | date | |
| summary | text | shown in list views |
| content | text | shown in post detail |
| images | text[] | filenames in the `0505` Storage bucket |
| videos | text[] | YouTube URLs (any standard YouTube/Shorts format) |

**`projects` table**
| column | type | notes |
|--------|------|-------|
| id | serial PK | |
| name | text | |
| description | text | |
| progress | integer | 0–100 |
| status | text | `ongoing` / `completed` / `paused` |
| updated_at | date | |

Both tables have RLS enabled with a public `SELECT` policy.

**Storage:** `0505` bucket (public) — stores blog post images.

### Navigation

Navigation is duplicated across all 5 HTML files: `index.html`, `about.html`, `blog.html`, `contact.html`, `blog/post.html`. When adding a nav item, update all five. Paths in `blog/post.html` use `../` prefix.

### Contact Form

Handled by Formspree (`https://formspree.io/f/xeenkpzo`). Submitted via `fetch` with `Accept: application/json` — no page redirect on success.

### Video Embedding

YouTube URLs in `posts.videos` are matched with a regex to extract the video ID, then embedded as a responsive 16:9 `<iframe>`. Supported URL formats: `youtube.com/watch?v=`, `youtu.be/`, `youtube.com/shorts/`.

## Deployment

Push to `main` → GitHub Pages auto-rebuilds (1–2 min). No CI/CD configuration needed.

```bash
git add <files>
git commit -m "message"
git push origin main
```

## Legacy Files (do not use)

- `build.js`, `add-post.js`, `new-post.sh` — old static-generation scripts replaced by Supabase
- `data/posts.json` — old data source, no longer read by the site
- `blog/post1.html`, `post2.html`, `post3.html` — replaced by `blog/post.html#<id>`
