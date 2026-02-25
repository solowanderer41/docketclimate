# The Docket Social

Multi-platform social media automation for [The Docket](https://www.anchor.is/docket), a climate journalism publication. Scrapes each issue, generates platform-specific posts via Claude API, validates against publishing law compliance checks, produces short-form video, and publishes to Bluesky, Twitter/X, Threads, and Instagram Reels on a weekly schedule.

## Quick Start

```bash
git clone https://github.com/solowanderer41/docketclimate.git
cd docketclimate
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python -m src.main test-connections
```

## How It Works

```
scrape ──> generate ──> schedule ──> post-today
  │            │            │            │
  │       Claude API    week queue    platform
  │       + exemplars   + compliance  publishers
  │                     checks
  ▼
anchor.is    ──>    queue/*.json    ──>    Bluesky / Twitter / Threads / Reels
```

1. **Scrape** fetches the latest issue from anchor.is, extracting articles across 6 sections (lived, systems, science, futures, archive, lab)
2. **Generate** produces platform-specific posts using Claude API, guided by exemplar few-shot examples and compliance guidelines
3. **Schedule** builds a weekly queue with time-slotted posts, runs compliance checks, and auto-fixes issues
4. **Post-today** publishes due items, retries failures, and logs results

## CLI Commands

All commands run via `.venv/bin/python -m src.main [command]`:

### Core Pipeline
| Command | Description |
|---------|-------------|
| `scrape` | Fetch the latest issue from anchor.is |
| `generate` | Generate posts from last scraped issue |
| `schedule` | Build a week's posting queue (interactive) |
| `schedule --auto` | Build queue with top-scored defaults |
| `post-today` | Post today's queued items |
| `post-today --dry` | Dry run (no actual posting) |
| `preview` | Preview generated content |
| `video` | Generate videos only |

### Analytics & Reporting
| Command | Description |
|---------|-------------|
| `collect-metrics` | Fetch engagement metrics for posted items |
| `metrics-report` | Show engagement analytics summary |
| `weekly-report` | Generate weekly engagement report |
| `review-week` | Weekly review: report + flag best work |

### Quality & Compliance
| Command | Description |
|---------|-------------|
| `compliance-check` | Run publishing law compliance checks |
| `compliance-check --fix` | Check + auto-fix queue items |
| `seed-exemplars` | Review top posts, edit, and save as exemplars |
| `seed-exemplars --top 12` | Show more candidates |
| `add-exemplar` | Write a manual gold-standard exemplar |
| `show-exemplars` | Show flagged exemplary content |
| `flag-good [ID]` | Flag a posted item as good copy/video |

### System
| Command | Description |
|---------|-------------|
| `test-connections` | Test all enabled platform connections |
| `queue-status` | Show current queue status |
| `health-check` | Run system health checks |
| `token-status` | Check Meta token health |
| `refresh-tokens` | Refresh Meta long-lived tokens |
| `repost-reel ID` | Delete and re-publish a Reel |

## Project Structure

```
src/
  main.py                 CLI orchestrator
  scraper.py              Anchor.is content scraping
  content_generator.py    Claude API post generation
  scheduler.py            Weekly queue builder (QueueItem dataclass)
  queue_runner.py         Daily posting executor with retry logic
  compliance.py           Fair-use / attribution / copyright checks
  exemplars.py            Quality exemplar management + prompt injection
  analytics.py            Multi-platform engagement metrics
  watchdog.py             System health monitoring
  notifier.py             Slack webhook notifications
  token_manager.py        Meta token refresh
  publishers/
    bluesky.py            AT Protocol
    twitter.py            Twitter API v2
    threads.py            Threads Graph API (3-step: create, poll, publish)
    reels.py              Instagram Reels Graph API
  video/
    generator.py          MoviePy vertical video (1080x1920, 30fps)
    imagegen.py           BFL Flux 2 Pro AI images
    voiceover.py          ElevenLabs text-to-speech
    stock.py              Pexels stock B-roll
    uploader.py           Cloudflare R2 CDN upload
config.yaml               All configuration (platforms, schedule, compliance, video)
queue/                    Weekly queue JSON files
output/                   Scraped issue data
analytics/                Engagement metrics + weekly reports
data/exemplars/           Flagged good-work examples (gitignored)
logs/                     Daily operation logs
```

## Configuration

### Platforms

Configured in `config.yaml`. Each platform has `enabled`, `max_chars`, and posting time slots:

- **Bluesky**: 300 chars, posts at 9:30am / 12:30pm / 3:30pm / 5:30pm PT
- **Twitter/X**: 280 chars, posts at 8am / 12pm / 5pm PT
- **Threads**: 500 chars, posts at 12:30pm / 7pm / 8:30pm PT
- **Instagram Reels**: 60s max video, 9:16 vertical, posts at 7:30am / 7:30pm PT

### Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

| Variable | Service |
|----------|---------|
| `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD` | Bluesky |
| `TWITTER_API_KEY`, `TWITTER_API_SECRET`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_TOKEN_SECRET` | Twitter/X |
| `META_THREADS_USER_ID`, `META_ACCESS_TOKEN` | Threads |
| `META_INSTAGRAM_ACCOUNT_ID`, `META_INSTAGRAM_ACCESS_TOKEN` | Instagram Reels |
| `ANTHROPIC_API_KEY` | Claude API (content generation) |
| `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` | ElevenLabs (voiceover) |
| `BFL_API_KEY` | Flux 2 Pro (AI images) |
| `PEXELS_API_KEY` | Pexels (stock B-roll) |
| `STORAGE_ENDPOINT_URL`, `STORAGE_ACCESS_KEY`, `STORAGE_SECRET_KEY`, `STORAGE_BUCKET`, `STORAGE_PUBLIC_URL` | Cloudflare R2 (video hosting) |
| `NOTIFICATION_WEBHOOK_URL` | Slack notifications |

### Compliance

The compliance engine validates posts against publishing law principles before publication:

- **Quote ratio**: Warns >50%, fails >70% direct quotation (*Campbell v. Acuff-Rose*)
- **Verbatim overlap**: Detects long word-for-word runs (*Warhol v. Goldsmith*)
- **Attribution**: Requires link or text credit (ICMJE standards)
- **Claim density**: Advisory count of unattributed factual claims (*Gertz v. Welch*)

Auto-fix attempts to repair posts before blocking. Configure thresholds in `config.yaml` under `compliance:`.

### Exemplars

The exemplar system stores hand-edited "gold standard" posts as few-shot examples for Claude API prompts. Seed them via `seed-exemplars` (review top-performing posts and edit to your ideal voice) or `add-exemplar` (write from scratch). Once 5+ exemplars exist, enable `exemplars.inject_into_prompts: true` in config.yaml.

## Automation (macOS)

Five launchd plists automate the daily workflow:

| Plist | Schedule | Command |
|-------|----------|---------|
| `com.docket.social.daily.plist` | Mon-Fri, 4x/day | `post-today` |
| `com.docket.social.weekly.plist` | Monday 12pm | `weekly-report` |
| `com.docket.social.healthcheck.plist` | Daily 10am | `health-check` |
| `com.docket.social.token-refresh.plist` | Periodic | `refresh-tokens` |

Install:
```bash
cp com.docket.social.*.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.docket.social.daily.plist
launchctl load ~/Library/LaunchAgents/com.docket.social.weekly.plist
launchctl load ~/Library/LaunchAgents/com.docket.social.healthcheck.plist
launchctl load ~/Library/LaunchAgents/com.docket.social.token-refresh.plist
```

## Video Pipeline

Generates vertical short-form video (1080x1920, 30fps) with:
- Per-slide AI images via BFL Flux 2 Pro or Pexels stock B-roll
- ElevenLabs voiceover synchronized per slide
- Breathing gradient backgrounds with floating particle effects
- Section-based color theming (coral, amber, teal, blue, taupe, sage)
- Hosted on Cloudflare R2 for Instagram Reels API ingestion

Two tiers: **cinematic** (per-slide AI images, Ken Burns zoom, subtitles) and **narrative** (single background, text overlay).

## Content Sections

| Section | Theme | Color |
|---------|-------|-------|
| lived | Human Stories | coral (#e07a5f) |
| systems | Cascading Effects | amber (#f2cc8f) |
| science | Emerging Science | teal (#4ecdc4) |
| futures | Future Tense | blue (#7b68ee) |
| archive | Climate Past | taupe (#a0937d) |
| lab | The Lab | sage (#81b29a) |
