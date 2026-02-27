# Docket Social — Project Rules

## Content Policy

- **Links must ALWAYS be included in text posts.** Never strip the article URL from post text or the QueueItem `url` field. CTA rotation may vary hashtags and follow prompts, but the source link is always present.
- **Twitter exception:** On Twitter/X the article link goes in a reply tweet (not the main tweet body) to avoid algorithmic reach penalties. This is the only acceptable pattern for omitting a link from the post body itself — the link is still present in the thread.

## Technical Notes

- Use `.venv/bin/python3` to run the project (system Python lacks dependencies).
- Queue files live in `queue/`, curation data in `output/`, metrics in `analytics/`.
- The pipeline has two text-building paths: `scheduler.py` (queue generation) and `content_generator.py:_build_post_text()` (used by curator). Both must follow the same link policy.
