"""
Publishing law compliance engine for The Docket.

Validates social media posts against fair-use, attribution, and copyright
principles before publication. Four checks run in sequence:

1. **Quote ratio** — flags posts where >50% of text is direct quotation
   (Fair Use Factor 3: amount and substantiality — *Campbell v. Acuff-Rose*,
   *Harper & Row v. Nation Enterprises*).

2. **Verbatim overlap** — detects long runs of text copied word-for-word
   from the source article (idea/expression dichotomy, transformative use
   requirement — *Warhol v. Goldsmith*).

3. **Attribution** — ensures every post credits the source via link or
   text attribution (ICMJE attribution standards, CRediT taxonomy).

4. **Claim density** — advisory-only count of unattributed factual claims
   containing specific numbers, dates, or named entities (*Gertz v. Welch*
   negligence standard for private-figure defamation).

Auto-fix behaviour:
    When ``auto_fix`` is enabled in config, the engine tries to repair
    failing posts (trim excessive quotes, append attribution) before
    resorting to a block.  Original text is preserved for audit.

Safety contract:
    Like all Docket modules, nothing in this file raises exceptions to
    the caller.  Every public function catches errors internally and
    returns a safe default (pass with warning).
"""

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# ── Regex patterns ───────────────────────────────────────────────────
# Matches text inside curly or straight double quotes.
_QUOTE_RE = re.compile(
    r'["\u201c]([^"\u201d]{5,})["\u201d]'
)

# Matches sentences containing specific factual claims (numbers,
# percentages, dates, dollar amounts).  Advisory check — false
# positives are acceptable since this never auto-fixes or blocks.
_CLAIM_RE = re.compile(
    r"[A-Z][^.!?]*?"                        # sentence start
    r"(?:\d[\d,.]*%?"                       # number or percentage
    r"|(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}"  # month + day
    r"|\$[\d,.]+[BMK]?"                     # dollar amounts
    r")"
    r"[^.!?]*[.!?]",                       # rest of sentence
    re.MULTILINE,
)


# ── Data structures ──────────────────────────────────────────────────
@dataclass
class ComplianceResult:
    """Result of running the full compliance check suite on a post."""

    status: str                           # "pass" | "warn" | "fixed" | "blocked"
    checks: list[dict] = field(default_factory=list)
    original_text: str | None = None      # pre-fix text (only when auto-fixed)
    fixed_text: str | None = None         # post-fix text (only when auto-fixed)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def summary(self) -> str:
        """One-line human-readable summary of all check results."""
        parts = []
        for c in self.checks:
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(c["status"], "?")
            parts.append(f"{icon} {c['name']}")
        return ", ".join(parts) if parts else "no checks run"


# ── Individual checks ────────────────────────────────────────────────

def check_quote_ratio(
    post_text: str,
    source_text: str | None,
    config: dict,
) -> dict:
    """Check what fraction of the post is direct quotation.

    Fair Use Factor 3: the amount and substantiality of the portion used.
    *Campbell v. Acuff-Rose* (510 U.S. 569, 1994) — all four factors
    weighed together; *Harper & Row v. Nation Enterprises* (471 U.S. 539,
    1985) — even 300 words can infringe if they are the "heart of the book".
    """
    try:
        check_cfg = config.get("compliance", {}).get("checks", {}).get(
            "quote_ratio", {}
        )
        if not check_cfg.get("enabled", True):
            return {"name": "Quote ratio", "status": "pass",
                    "detail": "disabled", "auto_fixed": False}

        warn_thresh = check_cfg.get("warn_threshold", 0.50)
        fail_thresh = check_cfg.get("fail_threshold", 0.70)

        quotes = _QUOTE_RE.findall(post_text)
        if not quotes:
            return {"name": "Quote ratio", "status": "pass",
                    "detail": "No direct quotes found", "auto_fixed": False}

        total_words = len(post_text.split())
        if total_words == 0:
            return {"name": "Quote ratio", "status": "pass",
                    "detail": "Empty post", "auto_fixed": False}

        quoted_words = sum(len(q.split()) for q in quotes)
        ratio = quoted_words / total_words

        if ratio > fail_thresh:
            return {"name": "Quote ratio", "status": "fail",
                    "detail": f"{ratio:.0%} quoted ({quoted_words}/{total_words} words)",
                    "auto_fixed": False}
        elif ratio > warn_thresh:
            return {"name": "Quote ratio", "status": "warn",
                    "detail": f"{ratio:.0%} quoted ({quoted_words}/{total_words} words)",
                    "auto_fixed": False}
        else:
            return {"name": "Quote ratio", "status": "pass",
                    "detail": f"{ratio:.0%} quoted", "auto_fixed": False}

    except Exception as e:
        return {"name": "Quote ratio", "status": "pass",
                "detail": f"Check error: {e}", "auto_fixed": False}


def check_verbatim_overlap(
    post_text: str,
    source_text: str | None,
    config: dict,
) -> dict:
    """Detect long runs of word-for-word text copied from the source.

    Grounded in the idea/expression dichotomy and transformative use
    requirement.  *Andy Warhol Foundation v. Goldsmith* (598 U.S. 508,
    2023) narrowed transformative use: when original and secondary use
    share the same purpose, the degree of transformation must be
    substantial.
    """
    try:
        check_cfg = config.get("compliance", {}).get("checks", {}).get(
            "verbatim_overlap", {}
        )
        if not check_cfg.get("enabled", True):
            return {"name": "Verbatim overlap", "status": "pass",
                    "detail": "disabled", "auto_fixed": False}

        if not source_text:
            return {"name": "Verbatim overlap", "status": "pass",
                    "detail": "No source text available for comparison",
                    "auto_fixed": False}

        warn_words = check_cfg.get("warn_words", 40)
        fail_words = check_cfg.get("fail_words", 60)

        # Tokenize into words for comparison
        post_words = post_text.lower().split()
        source_words = source_text.lower().split()

        if not post_words or not source_words:
            return {"name": "Verbatim overlap", "status": "pass",
                    "detail": "Insufficient text for comparison",
                    "auto_fixed": False}

        # Build a set of source n-grams for fast lookup, then find the
        # longest contiguous match by sliding a window over post words.
        # For efficiency, check from the longest possible match downward.
        max_overlap = 0
        overlap_text = ""

        # Build source word index for faster matching
        source_word_set = {}
        for i, w in enumerate(source_words):
            source_word_set.setdefault(w, []).append(i)

        for pi, pw in enumerate(post_words):
            if pw not in source_word_set:
                continue
            for si in source_word_set[pw]:
                # Extend match from this starting point
                length = 0
                while (pi + length < len(post_words) and
                       si + length < len(source_words) and
                       post_words[pi + length] == source_words[si + length]):
                    length += 1
                if length > max_overlap:
                    max_overlap = length
                    overlap_text = " ".join(
                        post_text.split()[pi:pi + min(length, 10)]
                    )
                    if length > 10:
                        overlap_text += "..."

        if max_overlap >= fail_words:
            return {"name": "Verbatim overlap", "status": "fail",
                    "detail": f"{max_overlap} consecutive words match source: \"{overlap_text}\"",
                    "auto_fixed": False}
        elif max_overlap >= warn_words:
            return {"name": "Verbatim overlap", "status": "warn",
                    "detail": f"{max_overlap} consecutive words match source: \"{overlap_text}\"",
                    "auto_fixed": False}
        else:
            return {"name": "Verbatim overlap", "status": "pass",
                    "detail": f"Max overlap: {max_overlap} words",
                    "auto_fixed": False}

    except Exception as e:
        return {"name": "Verbatim overlap", "status": "pass",
                "detail": f"Check error: {e}", "auto_fixed": False}


def check_attribution(
    post_text: str,
    article_url: str | None,
    platform: str,
    config: dict,
) -> dict:
    """Verify the post credits its source via link or text attribution.

    Grounded in ICMJE attribution standards and CRediT taxonomy
    principles (ANSI/NISO Z39.104-2022).  Academic and professional
    publishing norms require clear source identification.
    """
    try:
        check_cfg = config.get("compliance", {}).get("checks", {}).get(
            "attribution", {}
        )
        if not check_cfg.get("enabled", True):
            return {"name": "Attribution", "status": "pass",
                    "detail": "disabled", "auto_fixed": False}

        credit_text = check_cfg.get("credit_text", "via The Docket")

        # Check for URL presence
        has_url = bool(article_url) and article_url in post_text

        # Check for credit text (case-insensitive)
        has_credit = credit_text.lower() in post_text.lower()

        # Check for any URL in the post text
        has_any_url = bool(re.search(r"https?://\S+", post_text))

        if has_url or has_credit or has_any_url:
            return {"name": "Attribution", "status": "pass",
                    "detail": "Source link or credit present",
                    "auto_fixed": False}

        # No attribution found.  This is expected for no-link CTA rotation
        # posts — the pipeline intentionally omits links on some posts for
        # algorithmic distribution.  Flag as advisory warn, not fail.
        return {"name": "Attribution", "status": "warn",
                "detail": "No source link or credit text (may be CTA rotation)",
                "auto_fixed": False}

    except Exception as e:
        return {"name": "Attribution", "status": "pass",
                "detail": f"Check error: {e}", "auto_fixed": False}


def check_claim_density(
    post_text: str,
    config: dict,
) -> dict:
    """Count unattributed factual claims for defamation risk awareness.

    Advisory only — never auto-fixed or blocked.

    *Gertz v. Robert Welch, Inc.* (418 U.S. 323, 1974): private figures
    need only prove negligence for compensatory damages.  Specific,
    verifiable factual claims carry higher defamation risk than opinion
    or commentary.
    """
    try:
        check_cfg = config.get("compliance", {}).get("checks", {}).get(
            "claim_density", {}
        )
        if not check_cfg.get("enabled", True):
            return {"name": "Claim density", "status": "pass",
                    "detail": "disabled", "auto_fixed": False}

        warn_threshold = check_cfg.get("warn_threshold", 3)

        claims = _CLAIM_RE.findall(post_text)
        count = len(claims)

        if count > warn_threshold:
            examples = [c.strip()[:60] + "..." if len(c) > 60 else c.strip()
                        for c in claims[:2]]
            return {"name": "Claim density", "status": "warn",
                    "detail": f"{count} factual claims detected (e.g., {'; '.join(examples)})",
                    "auto_fixed": False}
        else:
            return {"name": "Claim density", "status": "pass",
                    "detail": f"{count} factual claims",
                    "auto_fixed": False}

    except Exception as e:
        return {"name": "Claim density", "status": "pass",
                "detail": f"Check error: {e}", "auto_fixed": False}


# ── Auto-fix helpers ─────────────────────────────────────────────────

def _auto_fix_quote_ratio(
    post_text: str, max_chars: int | None = None,
) -> tuple[str, bool]:
    """Trim the longest direct quote to its first clause.

    Returns (fixed_text, was_modified).
    """
    quotes = list(_QUOTE_RE.finditer(post_text))
    if not quotes:
        return post_text, False

    # Find the longest quote
    longest = max(quotes, key=lambda m: len(m.group(1)))
    full_quote = longest.group(1)

    # Trim to first clause (up to first comma, semicolon, or dash)
    clause_end = re.search(r"[,;\u2014—\-]", full_quote)
    if clause_end and clause_end.start() > 10:
        trimmed = full_quote[:clause_end.start()].rstrip()
    else:
        # If no clause boundary, take first ~40% of words
        words = full_quote.split()
        keep = max(3, len(words) * 2 // 5)
        trimmed = " ".join(words[:keep])

    # Replace in post text
    old_quoted = longest.group(0)  # includes quote marks
    # Reconstruct with the opening/closing marks
    open_mark = old_quoted[0]
    close_mark = old_quoted[-1]
    new_quoted = f"{open_mark}{trimmed} …{close_mark}"
    fixed = post_text.replace(old_quoted, new_quoted, 1)

    # Respect character limit if provided
    if max_chars and len(fixed) > max_chars:
        fixed = fixed[:max_chars].rstrip()

    return fixed, True


def _auto_fix_verbatim(
    post_text: str,
    source_text: str,
    max_overlap_words: int = 40,
) -> tuple[str, bool]:
    """Truncate a long verbatim passage and replace its tail with [...].

    Returns (fixed_text, was_modified).
    """
    post_words = post_text.split()
    source_words_lower = source_text.lower().split()

    # Find the longest overlap (same logic as check, but on original case)
    post_lower = [w.lower() for w in post_words]

    source_idx = {}
    for i, w in enumerate(source_words_lower):
        source_idx.setdefault(w, []).append(i)

    best_start = 0
    best_len = 0
    for pi, pw in enumerate(post_lower):
        if pw not in source_idx:
            continue
        for si in source_idx[pw]:
            length = 0
            while (pi + length < len(post_lower) and
                   si + length < len(source_words_lower) and
                   post_lower[pi + length] == source_words_lower[si + length]):
                length += 1
            if length > best_len:
                best_len = length
                best_start = pi

    if best_len < max_overlap_words:
        return post_text, False

    # Keep first ~60% of the overlap and replace the rest with [...]
    keep = max(5, best_len * 3 // 5)
    before = " ".join(post_words[:best_start + keep])
    after_idx = best_start + best_len
    after = " ".join(post_words[after_idx:]) if after_idx < len(post_words) else ""

    fixed = before + " [...]"
    if after:
        fixed += " " + after

    return fixed.strip(), True


def _auto_fix_attribution(
    post_text: str,
    platform: str,
    config: dict,
) -> tuple[str, bool]:
    """Append credit text if there's room within the platform limit.

    Returns (fixed_text, was_modified).
    """
    check_cfg = config.get("compliance", {}).get("checks", {}).get(
        "attribution", {}
    )
    credit = check_cfg.get("credit_text", "via The Docket")
    suffix = f"\n\n{credit}"

    # Platform character limits
    platform_limits = {
        "twitter": 280,
        "bluesky": 300,
        "threads": 500,
    }
    limit = platform_limits.get(platform, 500)

    if len(post_text) + len(suffix) <= limit:
        return post_text + suffix, True

    # Not enough room — can't auto-fix
    return post_text, False


# ── Orchestrator ─────────────────────────────────────────────────────

def run_compliance_check(
    post_text: str,
    source_text: str | None,
    article_url: str | None,
    platform: str,
    config: dict,
) -> ComplianceResult:
    """Run all enabled compliance checks on a post.

    Returns a :class:`ComplianceResult` with the worst status across all
    checks.  When ``auto_fix`` is enabled and a check fails, the engine
    attempts to repair the text before falling back to a block.

    Never raises exceptions — catches all errors internally and returns
    a "pass" result with a warning detail on error.
    """
    try:
        comp_cfg = config.get("compliance", {})
        if not comp_cfg.get("enabled", False):
            return ComplianceResult(status="pass")

        auto_fix = comp_cfg.get("auto_fix", True)
        block_on_fail = comp_cfg.get("block_on_fail", False)

        checks = []
        working_text = post_text
        was_fixed = False

        # 1. Quote ratio
        qr = check_quote_ratio(working_text, source_text, config)
        if auto_fix and qr["status"] == "fail":
            fixed, did_fix = _auto_fix_quote_ratio(working_text)
            if did_fix:
                working_text = fixed
                was_fixed = True
                qr["auto_fixed"] = True
                qr["status"] = "warn"
                qr["detail"] += " → auto-trimmed"
        checks.append(qr)

        # 2. Verbatim overlap
        vo = check_verbatim_overlap(working_text, source_text, config)
        if auto_fix and vo["status"] == "fail" and source_text:
            warn_words = comp_cfg.get("checks", {}).get(
                "verbatim_overlap", {}
            ).get("warn_words", 40)
            fixed, did_fix = _auto_fix_verbatim(
                working_text, source_text, warn_words
            )
            if did_fix:
                working_text = fixed
                was_fixed = True
                vo["auto_fixed"] = True
                vo["status"] = "warn"
                vo["detail"] += " → auto-truncated"
        checks.append(vo)

        # 3. Attribution
        attr = check_attribution(working_text, article_url, platform, config)
        if auto_fix and attr["status"] == "warn":
            # Only auto-fix attribution if the post actually lacks any
            # source indicator (not just a CTA rotation miss)
            has_any_url = bool(re.search(r"https?://\S+", working_text))
            if not has_any_url:
                fixed, did_fix = _auto_fix_attribution(
                    working_text, platform, config
                )
                if did_fix:
                    working_text = fixed
                    was_fixed = True
                    attr["auto_fixed"] = True
                    attr["status"] = "pass"
                    attr["detail"] = "Credit text auto-appended"
        checks.append(attr)

        # 4. Claim density (advisory only — never fixed or blocked)
        cd = check_claim_density(working_text, config)
        checks.append(cd)

        # Determine overall status
        statuses = [c["status"] for c in checks]
        if "fail" in statuses:
            if block_on_fail:
                overall = "blocked"
            else:
                overall = "warn"
        elif was_fixed:
            overall = "fixed"
        elif "warn" in statuses:
            overall = "warn"
        else:
            overall = "pass"

        return ComplianceResult(
            status=overall,
            checks=checks,
            original_text=post_text if was_fixed else None,
            fixed_text=working_text if was_fixed else None,
        )

    except Exception as e:
        return ComplianceResult(
            status="pass",
            checks=[{"name": "Compliance engine", "status": "pass",
                     "detail": f"Engine error: {e}", "auto_fixed": False}],
        )


# ── Prompt guidelines builder ────────────────────────────────────────

_HOOK_GUIDELINES = """CONTENT COMPLIANCE:
- Paraphrase and reframe rather than quoting verbatim from the source article
- If you include a direct quote, keep it under 15 words and attribute it
- Transform the story angle — don't summarize, find the unseen layer
- Each hook should add original framing that goes beyond the source text
- Never reproduce more than one sentence verbatim from the article"""

_VIDEO_GUIDELINES = """CONTENT COMPLIANCE:
- Slide text must be original — never copy sentences from the article verbatim
- Voiceover should paraphrase and reframe, not read the article aloud
- If citing a statistic or quote, keep it brief (under 10 words) and attribute the source
- Each slide should present an original visual/narrative angle, not a summary
- The script as a whole must be a transformative creative work, not a condensed version"""


def build_compliance_guidelines(check_type: str = "hook") -> str:
    """Return a compact compliance guidelines block for LLM prompt injection.

    Args:
        check_type: ``"hook"`` for text post prompts, ``"video"`` for
            video script prompts.

    Returns:
        A string block (~200 tokens) ready to be inserted into a Claude
        API prompt.  Returns empty string if check_type is unrecognised.
    """
    if check_type == "hook":
        return _HOOK_GUIDELINES
    elif check_type == "video":
        return _VIDEO_GUIDELINES
    return ""


# ── Rich terminal display ────────────────────────────────────────────

def print_compliance_report(results: list[tuple[str, ComplianceResult]]) -> None:
    """Display compliance check results for a list of queue items.

    Args:
        results: List of ``(item_id, ComplianceResult)`` tuples.
    """
    status_counts = {"pass": 0, "warn": 0, "fixed": 0, "blocked": 0}
    for _, r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    overall_parts = []
    if status_counts["blocked"]:
        overall_parts.append(f"[red]{status_counts['blocked']} blocked[/red]")
    if status_counts["fixed"]:
        overall_parts.append(f"[cyan]{status_counts['fixed']} auto-fixed[/cyan]")
    if status_counts["warn"]:
        overall_parts.append(f"[yellow]{status_counts['warn']} warnings[/yellow]")
    if status_counts["pass"]:
        overall_parts.append(f"[green]{status_counts['pass']} passed[/green]")

    console.print(f"\n[bold]Compliance Summary:[/bold] {', '.join(overall_parts)}")

    table = Table(show_lines=False)
    table.add_column("Item", style="bold", width=20)
    table.add_column("Status", width=10)
    table.add_column("Checks", width=55)

    status_styles = {
        "pass": "[green]PASS[/green]",
        "warn": "[yellow]WARN[/yellow]",
        "fixed": "[cyan]FIXED[/cyan]",
        "blocked": "[red]BLOCKED[/red]",
    }

    for item_id, result in results:
        style = status_styles.get(result.status, result.status)
        table.add_row(item_id[:20], style, result.summary)

    console.print(table)
