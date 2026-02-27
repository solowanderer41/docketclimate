"""
Content generator for The Docket social media automation.
Implements the "Unseen Reality" Social Strategy:
- Voice: The "Observational Insider" — calm, grounded, slightly weary, deeply observant
- Tone: Stark, atmospheric, punchy. Narrative weight over marketing hype.
- Energy: Low-frequency, high-impact. A quiet conversation while everyone else shouts.
Three Pillars:
1. The Sensory Detail — open with physical items/conditions that paint the struggle
2. The Statistical Gut-Punch — data as a weapon to expose failure
3. The "End of an Era" Sentiment — loss of vocation, landscape, certainty
"""
import os
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from src.scraper import Article, DocketIssue
load_dotenv(override=True)
# Lazy-loaded caches for engagement averages (feedback loop)
_section_avg_cache: dict[str, float] | None = None
_feature_vs_news_cache: dict[str, float] | None = None
# ---------------------------------------------------------------------------
# Hook variant A/B testing state
# ---------------------------------------------------------------------------
# Maps article title → chosen variant ID (0-4) for the current generation run.
# Populated by _build_unseen_reality_hook() when it selects a variant.
# Read by scheduler.py via get_hook_variant() to store on QueueItem.
_hook_variant_log: dict[str, int] = {}
# Variant selection weights loaded once per run from analytics/variant_weights.json
_variant_weights: list[float] | None = None
console = Console()
# Platform character limits
PLATFORM_LIMITS = {
    "bluesky": 300,
    "twitter": 280,
    "threads": 500,
}
# Hashtags — used sparingly per the strategy
SECTION_HASHTAGS = {
    "lived": ["#ClimateCrisis", "#Longform"],
    "systems": ["#Policy", "#ClimateCrisis"],
    "science": ["#ClimateCrisis", "#Education"],
    "futures": ["#ClimateCrisis", "#Longform"],
    "archive": ["#ClimateCrisis", "#Labor"],
    "lab": ["#ClimateCrisis", "#Longform"],
}
# ── Niche hashtag pools (Fix X6) ──────────────────────────────
# Mid-tier tags (10K-500K posts) where zero-authority accounts can surface.
# Never use mega-tags like #ClimateCrisis (>1M) or #ClimateChange (>2M).
NICHE_SECTION_TAGS = {
    "lived": ["#ClimateAdaptation", "#ClimateReality", "#ExtremeWeather", "#ClimateStories", "#ClimateMigration"],
    "systems": ["#ClimatePolicy", "#InsuranceCrisis", "#WaterRights", "#InfrastructureFailure", "#ClimateEconomy"],
    "science": ["#EarthScience", "#ClimateData", "#OceanWarming", "#ArcticIce", "#Biodiversity"],
    "futures": ["#CleanEnergy", "#ClimateTech", "#RenewableEnergy", "#GreenTransition", "#EnergyStorage"],
    "archive": ["#ClimateHistory", "#EnvironmentalJustice", "#LandUse", "#IndustrialPollution", "#HistoricalClimate"],
    "lab": ["#ClimateTech", "#Innovation", "#CarbonCapture", "#SustainableTech", "#ClimateInnovation"],
}
# Topic keywords → hashtag mappings for article-specific tags
TOPIC_HASHTAGS = {
    "drought": "#Drought", "flood": "#Flooding", "wildfire": "#Wildfire",
    "hurricane": "#Hurricane", "heat": "#ExtremeHeat", "insurance": "#InsuranceCrisis",
    "farm": "#Agriculture", "water": "#WaterCrisis", "coast": "#CoastalErosion",
    "ice": "#ArcticIce", "coral": "#CoralReefs", "forest": "#Deforestation",
    "solar": "#SolarEnergy", "wind": "#WindEnergy", "battery": "#EnergyStorage",
    "carbon": "#CarbonEmissions", "methane": "#Methane", "sea level": "#SeaLevelRise",
    "permafrost": "#Permafrost", "migration": "#ClimateMigration",
    "fish": "#Overfishing", "soil": "#SoilHealth", "glacier": "#GlacierRetreat",
}
# Platform-specific max hashtag counts
PLATFORM_HASHTAG_LIMITS = {
    "reels": 5,
    "threads": 2,
    "bluesky": 2,
    "twitter": 1,  # Twitter penalizes hashtag-heavy posts
}
def _select_hashtags(
    article_title: str,
    section_id: str,
    platform: str,
    summary: str = "",
) -> list[str]:
    """Select niche hashtags based on article content and platform limits.
    Strategy:
    1. Extract 1-2 topic-specific tags from title/summary keywords
    2. Fill remaining slots with curated mid-tier section tags
    3. Never use mega-tags (#ClimateCrisis, #ClimateChange)
    """
    max_tags = PLATFORM_HASHTAG_LIMITS.get(platform, 3)
    if max_tags == 0:
        return []
    tags = []
    combined_text = f"{article_title} {summary}".lower()
    # Step 1: Topic-specific tags from article content
    for keyword, tag in TOPIC_HASHTAGS.items():
        if keyword in combined_text and tag not in tags:
            tags.append(tag)
            if len(tags) >= 2:  # max 2 topic-specific
                break
    # Step 2: Fill with section niche tags (rotate based on title hash)
    section_pool = NICHE_SECTION_TAGS.get(section_id, NICHE_SECTION_TAGS["lived"])
    title_hash = sum(ord(c) for c in article_title)
    start = title_hash % len(section_pool)
    for i in range(len(section_pool)):
        if len(tags) >= max_tags:
            break
        tag = section_pool[(start + i) % len(section_pool)]
        if tag not in tags:
            tags.append(tag)
    return tags[:max_tags]
# Issue URL pattern
ISSUE_URL = "https://www.anchor.is/docket/issue/latest/"
SUBSCRIBE_URL = "https://www.anchor.is/docket/subscribe"
def _tag_url(
    url: str,
    platform: str,
    medium: str = "text",
    campaign: str = "",
    content: str = "",
) -> str:
    """Append UTM parameters to a URL for subscriber attribution.
    Every link posted to social gets tagged so we can attribute
    newsletter signups to specific platforms, post types, and sections.
    Args:
        url: The base URL (article link, subscribe page, etc.)
        platform: Social platform name (twitter, bluesky, threads, reels)
        medium: Content type — "text" or "video"
        campaign: Issue identifier, e.g. "issue_19"
        content: Post descriptor, e.g. "lived_feature", "teaser", "wrapup"
    Returns:
        The URL with UTM query parameters appended.
    """
    if not url:
        return url
    # Load config to check if UTM tracking is enabled
    try:
        with open(Path(__file__).parent.parent / "config.yaml") as f:
            config = yaml.safe_load(f)
        if not config.get("tracking", {}).get("utm_enabled", True):
            return url
    except Exception:
        pass  # Default to tagging if config can't be read
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    params["utm_source"] = [platform]
    params["utm_medium"] = [medium]
    if campaign:
        params["utm_campaign"] = [campaign]
    if content:
        params["utm_content"] = [content]
    new_query = urlencode({k: v[0] for k, v in params.items()})
    return urlunparse(parsed._replace(query=new_query))
# Conversation hooks for Threads (appended to posts to drive replies)
CONVERSATION_HOOKS = [
    "Has this hit your area yet?",
    "What's your city doing about this?",
    "Is anyone talking about this where you live?",
    "What would you do?",
    "Has anyone else noticed this?",
    "Does this match what you're seeing?",
]
# Rotating spoken video CTAs — documentary voice, understated
VIDEO_CTA_SPOKEN = [
    "That's The Docket. Follow for more stories like this.",
    "More at The Docket. Follow to keep seeing these.",
    "The Docket. Climate stories between the headlines. Follow for more.",
]
# Share-worthy closing slides for Reels (statement format — emotional/share-worthy)
SHARE_WORTHY_CLOSERS = [
    "This is what adaptation looks like now.",
    "This is the cost of doing nothing.",
    "The numbers don't care about the narrative.",
    "This is the gap between the policy and the ground.",
    "No one voted for this timeline.",
]
# On-screen CTA text for the final slide (follow + value prop)
VIDEO_CTA_TEXT = (
    "Follow @docketclimate for more\n"
    "Climate news \u00b7 Adaptation \u00b7 Speculation\n"
    "Link in bio"
)
# Legacy static video hashtags (kept for backward compat; _select_hashtags preferred)
VIDEO_HASHTAGS = {
    "lived": ["#ClimateAdaptation", "#ClimateReality", "#Documentary", "#ExtremeWeather", "#ClimateStories"],
    "systems": ["#ClimatePolicy", "#Infrastructure", "#ClimateEconomy", "#InsuranceCrisis", "#WaterRights"],
    "science": ["#EarthScience", "#OceanWarming", "#Biodiversity", "#ClimateData", "#ArcticIce"],
    "futures": ["#CleanEnergy", "#ClimateTech", "#RenewableEnergy", "#EnergyStorage", "#GreenTransition"],
    "archive": ["#ClimateHistory", "#EnvironmentalJustice", "#Documentary", "#LandUse", "#HistoricalClimate"],
    "lab": ["#ClimateTech", "#Innovation", "#CarbonCapture", "#SustainableTech", "#ClimateInnovation"],
}
def generate_pin_post(platform: str) -> str:
    """Generate a pinned introduction post for a platform profile.
    The pin post explains The Docket's value proposition in the
    Observational Insider voice and drives follows/subscriptions.
    Adapted per-platform for character limits and conventions.
    """
    subscribe_url = SUBSCRIBE_URL
    core = (
        "The Docket covers the climate stories that fall between the headlines.\n\n"
        "Every week: the insurance gaps nobody's filing on. The water maps cities "
        "voted against updating. The workers whose jobs just stopped existing.\n\n"
        "Free newsletter. Follow for the stories."
    )
    if platform == "twitter":
        # Twitter: tight, no link in tweet (bio link is the CTA)
        return (
            "Climate change has a texture. A sound. A price tag.\n\n"
            "The Docket covers the stories that fall between the headlines — "
            "the quiet costs, the slow exits, the ground truth.\n\n"
            "Free newsletter. Follow for the stories."
        )
    elif platform == "threads":
        # Threads: full 500 chars, end with conversation hook
        return (
            f"{core}\n\n"
            f"{subscribe_url}\n\n"
            "What climate story do you think is being undercovered?"
        )
    elif platform == "bluesky":
        # Bluesky: 300 chars, link card auto-generates
        return (
            "The Docket covers climate stories between the headlines. "
            "The quiet costs. The slow exits. The ground truth.\n\n"
            f"Free newsletter → {subscribe_url}"
        )
    else:
        # Instagram / generic
        return (
            f"{core}\n\n"
            f"🔗 Subscribe free — link in bio"
        )
@dataclass
class TextPost:
    platform: str
    text: str
    hashtags: list[str]
    article_title: str
    section: str
@dataclass
class VideoScript:
    title: str
    hook: str
    body_slides: list[str]
    cta: str
    voiceover_text: str
    article_title: str
    section: str
    url: str = ""
    cinematic_score: int = 0
    video_tier: str = "narrative"          # "cinematic" or "narrative"
    image_prompts: list[str] = field(default_factory=list)
    background_prompt: str = ""
    voiceover_lines: list[str] = field(default_factory=list)  # per-slide spoken text
    @property
    def caption(self) -> str:
        """
        The publishable caption for Reels / TikTok.
        SEO-optimized: hook + keyword-rich context line + link in bio + 5 hashtags.
        Instagram's 2025 algorithm weights keyword-rich captions for Explore/search.
        """
        hashtags = VIDEO_HASHTAGS.get(self.section, VIDEO_HASHTAGS.get("lived", []))
        tag_str = " ".join(hashtags[:5])
        return (
            f"{self.hook}\n\n"
            f"Climate news, adaptation, and speculation \u2014 every issue, free.\n"
            f"\U0001f517 Link in bio\n\n"
            f"{tag_str}"
        )
def _build_video_caption(hook: str, section: str, article_title: str = "") -> str:
    """Build an SEO-optimized Reels caption with max 5 niche hashtags.
    Instagram's 2025 algorithm weights keyword-rich captions for Explore/search
    distribution. The caption structure:
      hook → keyword-rich context line → link-in-bio CTA → 5 hashtags
    """
    hashtags = _select_hashtags(article_title or hook, section, "reels")
    tag_str = " ".join(hashtags[:5])
    return (
        f"{hook}\n\n"
        f"Climate news, adaptation, and speculation \u2014 every issue, free.\n"
        f"\U0001f517 Link in bio\n\n"
        f"{tag_str}"
    )
def _score_article(article: Article) -> float:
    """
    Score an article for social media appeal.
    Prefers articles with sensory detail, data, human stakes —
    the raw material for the "Unseen Reality" voice.
    """
    score = 0.0
    if article.author:
        score += 3.0
    if article.summary:
        summary = article.summary
        score += 2.0
        score += min(len(summary) / 200, 2.0)
        # Numbers/data = statistical gut-punch potential
        if re.search(r"\$[\d,]+|\d+%|\d+,\d{3}", summary):
            score += 3.0
        # Physical/sensory words
        sensory = ["dried", "flooded", "smoke", "heat", "storm", "water",
                    "fire", "wind", "ice", "snow", "drought", "rain", "dust"]
        for word in sensory:
            if word in summary.lower():
                score += 0.5
        # Quotes suggest real voices
        if '"' in summary or '\u201c' in summary:
            score += 2.0
    title = article.title or ""
    if title:
        if 20 <= len(title) <= 80:
            score += 2.0
        if re.search(r"\d+", title):
            score += 1.5
        if '"' in title or '\u201c' in title:
            score += 1.5
        if "?" in title:
            score += 1.0
    if article.full_text:
        score += 1.0
    # External source items are editorially curated — boost them
    if getattr(article, "is_external", False):
        score += 1.5
        if article.summary:
            score += 1.0
    if "Full Section" in (article.title or ""):
        score -= 5.0
    # Historical engagement boost (if analytics data exists)
    analytics_path = Path("analytics/metrics.json")
    _load_analytics_caches(analytics_path)
    # Section performance boost: +0 to +5 points
    # Sections that historically get more engagement are scored higher.
    if _section_avg_cache and hasattr(article, "section_id"):
        section_avg = _section_avg_cache.get(article.section_id, 0)
        max_avg = max(_section_avg_cache.values()) or 1
        score += (section_avg / max_avg) * 5.0
    # Feature vs news type boost: +0 to +3 points
    # If features outperform news (or vice versa), boost the winning type.
    if _feature_vs_news_cache and hasattr(article, "is_feature"):
        feat_avg = _feature_vs_news_cache.get("feature", 0)
        news_avg = _feature_vs_news_cache.get("news", 0)
        if article.is_feature and feat_avg > news_avg and news_avg > 0:
            score += min((feat_avg / news_avg - 1) * 3.0, 3.0)
        elif not article.is_feature and news_avg > feat_avg and feat_avg > 0:
            score += min((news_avg / feat_avg - 1) * 3.0, 3.0)
    return score
def _load_analytics_caches(analytics_path: Path):
    """
    Load engagement analytics caches from disk (once per process).
    Populates:
        _section_avg_cache: avg engagement per section
        _feature_vs_news_cache: avg engagement for features vs news
    """
    global _section_avg_cache, _feature_vs_news_cache
    if _section_avg_cache is not None:
        return  # Already loaded
    if not analytics_path.exists():
        _section_avg_cache = {}
        _feature_vs_news_cache = {}
        return
    try:
        from src.analytics import get_section_averages, load_metrics
        _section_avg_cache = get_section_averages(analytics_path)
        # Compute feature vs news averages
        metrics = load_metrics(analytics_path)
        feat_scores = [m.engagement_score for m in metrics if m.is_feature]
        news_scores = [m.engagement_score for m in metrics if not m.is_feature]
        _feature_vs_news_cache = {
            "feature": sum(feat_scores) / len(feat_scores) if feat_scores else 0,
            "news": sum(news_scores) / len(news_scores) if news_scores else 0,
        }
    except Exception:
        _section_avg_cache = {}
        _feature_vs_news_cache = {}
def _pick_best_articles(articles: list[Article], count: int) -> list[Article]:
    """Pick the most compelling articles from a list."""
    if not articles:
        return []
    scored = sorted(articles, key=_score_article, reverse=True)
    return scored[:count]
def _truncate_to_limit(text: str, limit: int) -> str:
    """Truncate text to fit within a character limit, breaking at sentence boundaries.
    Prefers cutting at a sentence end (period/question/exclamation) so the post
    reads as complete rather than trailing off with ``…``.  Falls back to word
    boundary + ellipsis only when no sentence break fits.
    """
    if len(text) <= limit:
        return text
    # Try to cut at the last sentence boundary that fits
    candidate = text[:limit]
    for delim in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
        idx = candidate.rfind(delim)
        if idx > limit // 3:  # must keep at least a third of the budget
            return text[: idx + 1]  # include the punctuation
    # Fallback: word boundary + ellipsis
    truncated = text[:limit - 1]
    last_space = truncated.rfind(" ")
    if last_space > limit // 2:
        truncated = truncated[:last_space]
    return truncated + "\u2026"
def _extract_data_points(text: str) -> list[str]:
    """Pull numbers, dollar amounts, percentages from text."""
    patterns = [
        r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?",
        r"\d+(?:\.\d+)?%",
        r"\d{1,3}(?:,\d{3})+",
        r"\d+(?:-fold|x)",
    ]
    points = []
    for p in patterns:
        points.extend(re.findall(p, text, re.IGNORECASE))
    return points
def _extract_sensory_fragment(text: str) -> str | None:
    """
    Try to extract a short, atmospheric fragment from article text.
    Looks for short concrete phrases that paint a picture.
    """
    if not text:
        return None
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Look for short, punchy sentences (under 60 chars) with concrete nouns
    for s in sentences:
        s = s.strip()
        if 15 < len(s) < 60:
            # Prefer sentences with physical/concrete language
            concrete = ["water", "pond", "field", "crop", "flood", "fire",
                        "smoke", "ice", "wind", "heat", "cold", "rain",
                        "house", "road", "hospital", "school", "farm",
                        "fish", "tree", "soil", "river", "coast", "sea",
                        "roof", "wall", "door", "window", "light"]
            if any(w in s.lower() for w in concrete):
                return s
    return None
# ---------------------------------------------------------------------------
# Narrative Structure Framework (Fix W)
# ---------------------------------------------------------------------------
# Five body-slide narrative structures. Each tells the LLM what ROLE each
# slide position plays in the sequence. The voice/texture rules remain
# separate — structure is "what this slide does", texture is "how it sounds".
# Selection mirrors the hook variant system: section affinity → content
# heuristics → hash-based rotation.
NARRATIVE_STRUCTURES = {
    "A": {
        "name": "The Zoom",
        "prompt_block": """NARRATIVE STRUCTURE — "The Zoom" (wide → narrow → wide)
Your 4-5 slides follow this arc:
  Slide 1: WIDE — Establish scope. A place, a number, a system. The big picture in one frame.
  Slide 2-3: NARROW — Zoom into a specific person, object, or moment inside that system.
  Slide 4-5: EVIDENCE — The data, the document, the measurement that makes it undeniable.
  Slide 6 (if needed): PULL BACK — Return to the system-level view, now reframed by what we saw up close.
  Final slide: THE WEIGHT — A single observation that makes the viewer sit with the gap between the system and the person.
Each slide should feel like a camera move: wide establishing shot, then close-up, then the reveal.""",
    },
    "B": {
        "name": "The Thread",
        "prompt_block": """NARRATIVE STRUCTURE — "The Thread" (character through-line)
Your 4-5 slides follow one person or place through time:
  Slide 1: ANCHOR — A person in a specific moment. Name, place, date. Make it concrete.
  Slide 2: THE BEFORE — What the world looked like before the disruption.
  Slide 3-4: THE DISRUPTION — What changed. Not explained, just shown.
  Slide 5: THE AFTER — Where that person or place is now. The new normal.
  Slide 6 (if needed): THE RETURN — Circle back to the opening image or moment, now with new meaning.
  Final slide: THE COST — What was lost, stated without commentary. Let the viewer feel it.
Every slide is connected to the same person or place. The viewer follows THEM, not a topic.""",
    },
    "C": {
        "name": "The Ratchet",
        "prompt_block": """NARRATIVE STRUCTURE — "The Ratchet" (escalation and turn)
Your 4-5 slides build pressure, then pivot:
  Slide 1: THE BASELINE — A single fact that seems manageable. A number. A status quo.
  Slide 2: TURN THE SCREW — A second fact that complicates the first. The picture gets worse.
  Slide 3: TURN AGAIN — A third fact. Now the pattern is undeniable.
  Slide 4 (optional): ONE MORE — If the article supports it, one more turn. The weight becomes absurd.
  Slide 5: THE PIVOT — Subvert expectations. The human cost, the irony, the thing nobody mentions.
  Final slide: THE VERDICT — A cold, declarative sentence that names what the accumulation reveals.
Each slide adds weight to the same scale. The pivot is the moment the scale breaks.""",
    },
    "D": {
        "name": "The Divide",
        "prompt_block": """NARRATIVE STRUCTURE — "The Divide" (two parallel realities)
Your 4-5 slides alternate between two realities until the gap is undeniable:
  Slide 1: REALITY A — One group, one place, one set of rules. Concrete detail.
  Slide 2: REALITY B — The other group, place, or set of rules. Same level of detail.
  Slide 3: REALITY A DEEPENED — More about the first side. The stakes become clear.
  Slide 4: REALITY B DEEPENED — More about the second side. The contrast sharpens.
  Slide 5-6: THE GAP — Name the distance between the two realities. Use a number, a comparison, a single image.
  Final slide: THE QUESTION — Not a literal question. A statement that makes the disparity impossible to ignore.
The power comes from juxtaposition. Don't explain the unfairness — let the viewer see it.""",
    },
    "E": {
        "name": "The Reveal",
        "prompt_block": """NARRATIVE STRUCTURE — "The Reveal" (hidden truth surfacing)
Your 4-5 slides withhold the key fact, then expose it:
  Slide 1: THE SURFACE — What the public sees. The press release, the official line, the assumption.
  Slide 2: THE CRACK — A detail that doesn't fit. Something small that hints at a different story.
  Slide 3-4: THE DIG — Evidence accumulates. Documents, data, testimony. Each slide adds one piece.
  Slide 5: THE REVEAL — The buried truth, stated plainly. No fanfare. Just the fact.
  Slide 6 (if needed): THE IMPLICATION — What this means going forward. The door that opened.
  Final slide: THE ECHO — A quiet callback to the surface from Slide 1, now understood differently.
Build toward the reveal like a slow zoom on a document. The audience should feel they discovered it.""",
    },
}
# Section → [primary, secondary] structure affinity
NARRATIVE_AFFINITY: dict[str, list[str]] = {
    "lived":    ["B", "D"],   # Thread (character), Divide (inequality)
    "systems":  ["A", "C"],   # Zoom (policy/systems), Ratchet (escalation)
    "science":  ["E", "A"],   # Reveal (discovery), Zoom (scope)
    "futures":  ["C", "E"],   # Ratchet (accumulation), Reveal (hidden)
    "archive":  ["B", "E"],   # Thread (historical), Reveal (buried history)
    "lab":      ["C", "A"],   # Ratchet (data), Zoom (scope)
}
def _select_narrative_structure(article: Article) -> tuple[str, str]:
    """
    Select a narrative structure for the article's body slides.
    Three-tier cascade (mirrors hook variant selection):
    1. Content heuristics — override when strong signal detected
    2. Section affinity — default mapping per section
    3. Hash rotation — tiebreaker between section's two affinities
    Returns (key, reason) where key is "A"-"E" and reason is a debug string.
    """
    section = getattr(article, "section_id", "") or ""
    title = article.title or ""
    text = article.full_text or article.summary or ""
    text_head = text[:500]
    # Tier 2: Content heuristics (check first — they override section default)
    # Named person → The Thread (B)
    person_pattern = (
        r"[A-Z][a-z]+ [A-Z][a-z]+ "
        r"(?:stood|said|walked|watched|held|drove|flew|sat|looked|"
        r"arrived|returned|waited|brought|told|asked|turned|knew|"
        r"was in|went to|lives in|works at|came to)"
    )
    if re.search(person_pattern, text_head):
        return ("B", "person_detected")
    # 3+ data points → The Ratchet (C)
    data_points = _extract_data_points(text)
    if len(data_points) >= 3:
        return ("C", "data_heavy")
    # Comparison language → The Divide (D)
    comparison_pattern = (
        r"\b(?:while|but|meanwhile|however|yet|whereas)\b"
    )
    comparison_hits = re.findall(comparison_pattern, text[:1000], re.IGNORECASE)
    if len(comparison_hits) >= 2:
        return ("D", "comparison_language")
    # Tier 1 + 3: Section affinity with hash rotation
    affinities = NARRATIVE_AFFINITY.get(section, ["A", "C"])
    title_hash = sum(ord(c) for c in title)
    key = affinities[title_hash % len(affinities)]
    return (key, f"section_{section}")
# ---------------------------------------------------------------------------
# Variant weight loading + hook variant tracking (A/B testing)
# ---------------------------------------------------------------------------
def _load_variant_weights() -> list[float]:
    """Load hook variant selection weights from disk (or return equal defaults)."""
    global _variant_weights
    if _variant_weights is not None:
        return _variant_weights
    weights_path = Path(__file__).parent.parent / "analytics" / "variant_weights.json"
    try:
        if weights_path.exists():
            with open(weights_path) as f:
                data = json.load(f)
            weights = data.get("weights", [1.0, 1.0, 1.0, 1.0, 1.0])
            if len(weights) == 5 and all(isinstance(w, (int, float)) for w in weights):
                _variant_weights = weights
                return _variant_weights
    except Exception:
        pass
    _variant_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    return _variant_weights
def get_hook_variant(title: str) -> int | None:
    """Return the hook variant used for a given article title (or None if untracked)."""
    return _hook_variant_log.get(title)
def _build_unseen_reality_hook(article: Article, variant: int | None = None) -> str:
    """
    DEPRECATED: All hooks now go through _generate_hooks_llm().
    Kept for backward compatibility — will be removed in a future release.
    Build a hook following the "Unseen Reality" strategy.
    The hook should be LONG and engaging — fill the available space.
    Five hook structures rotate to prevent predictability:
    - variant 0 (default): Sensory → Data → Stakes
    - variant 1: Data-first → Sensory → Stakes
    - variant 2: Question → Data → Scene
    - variant 3: Moral-stakes — collective failure, accountability
    - variant 4: Contrarian-reframe — everyone's looking at X, miss Y
    When variant is None, auto-selects based on article content
    (data-heavy → 1, question title → 2) with a hash-based rotation
    for articles that don't trigger specific heuristics.
    """
    title = article.title or ""
    summary = article.summary or ""
    full_text = article.full_text or ""
    source_text = summary or full_text
    if not source_text:
        # Title-only fallback — still make it weighted
        if title:
            return title.rstrip("?").strip() + "."
        return title
    # Gather all the raw material
    sensory = _extract_sensory_fragment(source_text)
    data_points = _extract_data_points(source_text)
    quote_match = re.search(r'["\u201c]([^"\u201d]{15,120})["\u201d]', source_text)
    # Extract the strongest sentences from the source
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', source_text)
                 if s.strip() and len(s.strip()) > 15]
    # Auto-select variant if not specified
    # Weighted random selection for A/B testing.  Content heuristics still
    # influence: data-heavy articles get a boost toward variant 1, question
    # titles toward variant 2.  But selection is randomised via weights
    # so we can measure which variants perform best.
    if variant is None:
        import random
        weights = list(_load_variant_weights())  # copy so we can adjust
        # Content-heuristic boosts (additive, not exclusive)
        if len(data_points) >= 2:
            weights[1] *= 2.0   # boost Data-first for data-heavy articles
        if "?" in title:
            weights[2] *= 2.0   # boost Question for question titles
        variant = random.choices([0, 1, 2, 3, 4], weights=weights, k=1)[0]
    # Record for A/B tracking (scheduler reads this via get_hook_variant)
    if title:
        _hook_variant_log[title] = variant
    # Build hook by layering elements — longer is better
    parts = []
    if variant == 1 and data_points and len(sentences) > 1:
        # DATA-FIRST: Lead with the hardest number, then scene
        for sentence in sentences:
            if any(dp in sentence for dp in data_points):
                parts.append(sentence)
                break
        if sensory:
            parts.append(sensory)
        elif sentences:
            parts.append(sentences[0] if sentences[0] not in parts else
                         (sentences[1] if len(sentences) > 1 else ""))
    elif variant == 2:
        # QUESTION: Lead with a provocative question derived from the title
        q_title = title.rstrip(".!").strip()
        if "?" not in q_title:
            # Transform title into a question
            q_title = f"What happens when {q_title.lower()}?"
        parts.append(q_title)
    elif variant == 3:
        # MORAL-STAKES: Find accountability/failure language in the source.
        # Only use moral framing when the article actually supports it.
        # Calm but damning — grounded in what the text says, not canned claims.
        moral_keywords = ["warned", "ignored", "preventable", "knew", "failed",
                          "refused", "defunded", "delayed", "overruled", "denied",
                          "rejected", "blocked", "lobbied", "suppressed", "buried"]
        moral_sentence = None
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in moral_keywords):
                moral_sentence = sentence
                break
        if moral_sentence:
            parts.append(moral_sentence)
        else:
            # No accountability language found — fall back to strongest lead sentence.
            # Don't force a moral frame the article doesn't support.
            if sentences:
                parts.append(sentences[0])
        # Follow with a second concrete sentence for context
        if sentences and len(sentences) > 1:
            for s in sentences[1:3]:
                if not _is_duplicate(s, parts):
                    parts.append(s)
                    break
    elif variant == 4:
        # CONTRARIAN-REFRAME: Lead with a less-obvious finding from the article.
        # The reframe comes from the CONTENT, not a canned formula.
        # Skip the first sentence (that's the obvious angle) and find a
        # deeper-layer detail — a secondary finding, a consequence, a mechanism.
        deeper = None
        for s in sentences[1:5]:
            # Prefer sentences with data, causation, or mechanism language
            if (re.search(r"\d", s) or
                any(w in s.lower() for w in ["because", "caused", "means", "leads to",
                                              "result", "driven by", "linked to"])):
                deeper = s
                break
        if deeper:
            parts.append(deeper)
        elif len(sentences) > 1:
            parts.append(sentences[1])
        elif sentences:
            parts.append(sentences[0])
    else:
        # DEFAULT (variant 0): Sensory opener
        pass
    # Layer 1: Lead with sensory or staccato list (for variant 0, or as fallback)
    if not parts:
        if sensory:
            parts.append(sensory)
        elif sentences:
            # Use first strong sentence as the scene-setter
            parts.append(sentences[0])
    # ── Dedup helper: reject sentences already present as substrings ──
    def _is_duplicate(candidate: str, existing: list[str]) -> bool:
        """Return True if candidate overlaps substantially with any existing part."""
        c = candidate.strip().rstrip(".")
        for ex in existing:
            e = ex.strip().rstrip(".")
            if c in e or e in c:
                return True
            # Also catch near-duplicates sharing a long prefix
            shared = min(len(c), len(e), 40)
            if shared > 20 and c[:shared] == e[:shared]:
                return True
        return False
    # Layer 2: Add data gut-punch if available
    if data_points and len(sentences) > 1:
        for sentence in sentences[1:]:
            if any(dp in sentence for dp in data_points) and not _is_duplicate(sentence, parts):
                parts.append(sentence)
                break
    # Layer 3: Add a quote for human weight
    if quote_match and len(parts) < 3:
        quote = quote_match.group(1)
        quote_text = f'\u201c{quote}.\u201d'
        if not _is_duplicate(quote_text, parts):
            parts.append(quote_text)
    # Layer 4: Add a second concrete sentence for depth
    if len(parts) < 3 and len(sentences) > 1:
        for s in sentences[1:4]:
            if not _is_duplicate(s, parts) and len(s) < 100:
                parts.append(s)
                break
    # Layer 5: If we still only have one part, pull more from summary
    if len(parts) == 1 and len(sentences) > 1:
        for s in sentences[1:3]:
            if not _is_duplicate(s, parts):
                parts.append(s)
    # Join with line breaks for atmospheric pacing
    hook = "\n\n".join(parts)
    # ── Quality gate (Fix X2) ─────────────────────────────────
    # Reject hooks that look broken, are too thin, or contain CTA artifacts.
    _cta_pattern = re.compile(
        r"CONTINUE READING|READ MORE|LEARN MORE|SEE MORE",
        re.IGNORECASE,
    )
    _title_clean = (title or "").strip().rstrip(".").lower()
    _hook_clean = hook.strip().rstrip(".").lower()
    is_bad = (
        len(hook.strip()) < 40
        or _cta_pattern.search(hook)
        or (_title_clean and _hook_clean == _title_clean)
    )
    if is_bad:
        # Try LLM refinement regardless of config
        try:
            refined = _refine_hook_with_exemplars(
                hook, article, "threads", 500,  # use generous limit
            )
            if refined and refined != hook and len(refined) > 40:
                return refined
        except Exception:
            pass
        # If LLM also fails, return empty string → caller should skip this article
        if _cta_pattern.search(hook):
            return ""  # signal to caller: unusable hook
        # For short-but-clean hooks, still return them (better than nothing)
    return hook
def _refine_hook_with_exemplars(
    hook: str,
    article: Article,
    platform: str,
    max_chars: int,
) -> str:
    """Optionally refine a template-generated hook using Claude + exemplars.
    DEPRECATED: No longer called — all hooks now generated directly by
    _generate_hooks_llm() with exemplars baked into the prompt. Kept for
    backward compatibility.
    This is a best-effort refinement — the template hook is always the
    safe fallback.
    """
    try:
        with open(Path(__file__).parent.parent / "config.yaml") as f:
            _cfg = yaml.safe_load(f)
        exemplar_cfg = _cfg.get("exemplars", {})
        if not exemplar_cfg.get("inject_into_prompts", False):
            return hook
    except Exception:
        return hook
    exemplar_block = _format_exemplars_for_prompt(
        "good_copy",
        section=article.section_id,
        limit=exemplar_cfg.get("max_per_prompt", 3),
    )
    if not exemplar_block:
        return hook  # No exemplars flagged yet — use raw hook
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return hook
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""You are a social media copywriter for The Docket, a climate journalism newsletter. Voice: "The Observational Insider" — calm, grounded, slightly weary, deeply observant. Stark, atmospheric, punchy. No hype. No emoji.
Below is a draft hook for a {platform} post about this article:
ARTICLE: {article.title}
DRAFT HOOK:
{hook}
{exemplar_block}
Rewrite the hook to match the quality and tone of the exemplars above. Keep the same factual content but sharpen the language, pacing, and impact. Lead with the most specific, surprising finding. Ensure the hook paraphrases rather than quoting verbatim from the article.
NEVER USE: "Everyone's talking about...", "They're missing the real story", "Nobody's talking about...", "Here's what they don't want you to know", or any formulaic engagement-bait opener. Lead with substance, not cleverness.
Must fit in {max_chars} characters (including line breaks). Return ONLY the rewritten hook text, no explanation."""
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        refined = response.content[0].text.strip()
        # Validate length — fall back if Claude exceeded the limit
        if len(refined) > max_chars or len(refined) < 20:
            return hook
        return refined
    except Exception:
        return hook  # Silent fallback
def _generate_hooks_llm(
    article: Article, platforms: dict[str, int],
) -> dict[str, str] | None:
    """Generate platform-specific hooks via a single Claude API call.
    One API call produces hooks for ALL text platforms at once — this
    saves cost and ensures variety across platforms.
    Args:
        article: The article to generate hooks for.
        platforms: Dict of ``{platform_name: max_chars}``.
    Returns:
        Dict mapping platform to hook text, or ``None`` on failure.
        Only includes platforms whose hooks pass validation.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    # Build exemplar block if available
    exemplar_block = ""
    try:
        with open(Path(__file__).parent.parent / "config.yaml") as f:
            _cfg = yaml.safe_load(f)
        exemplar_cfg = _cfg.get("exemplars", {})
        if exemplar_cfg.get("inject_into_prompts", False):
            exemplar_block = _format_exemplars_for_prompt(
                "good_copy",
                section=article.section_id,
                limit=exemplar_cfg.get("max_per_prompt", 3),
            )
    except Exception:
        pass
    # Build platform budget lines
    budget_lines = []
    for pname, limit in platforms.items():
        conventions = {
            "twitter": (
                "Punchy, self-contained. No link in body (goes in reply). "
                "One core finding, stated cold. Max impact in short space."
            ),
            "bluesky": (
                "Earnest and substantive — Bluesky's culture rewards sincerity over cleverness. "
                "Write like a smart colleague sharing a paper. Lead with the science or "
                "the specific finding. No engagement bait. Link card auto-generates."
            ),
            "threads": (
                "Most room — layer detail and context. Build a micro-narrative. "
                "End with a genuine question that invites informed replies, not just 'thoughts?' "
                "Threads rewards conversation depth."
            ),
            "video": (
                "Short-form vertical video hook — spoken aloud as voiceover. "
                "2-3 punchy sentences max. Must sound natural when read aloud. "
                "Lead with the most striking finding. No hashtags, no links."
            ),
        }
        convention = conventions.get(pname, "Concise and impactful.")
        budget_lines.append(f"- {pname}: {limit} chars. {convention}")
    platform_budgets = "\n".join(budget_lines)
    platform_names = ", ".join(platforms.keys())
    # Build compliance guidelines block (best-effort)
    compliance_block = ""
    try:
        from src.compliance import build_compliance_guidelines
        with open(Path(__file__).parent.parent / "config.yaml") as _ccf:
            _cc_cfg = yaml.safe_load(_ccf).get("compliance", {})
        if _cc_cfg.get("prompt_enrichment", False):
            compliance_block = build_compliance_guidelines("hook")
    except Exception:
        pass
    # Build the article context
    article_text = article.summary or article.text or ""
    if len(article_text) > 1500:
        article_text = article_text[:1500] + "..."
    prompt = f"""You are a social media copywriter for The Docket, a weekly climate journalism newsletter.
VOICE: "The Observational Insider" — calm, grounded, slightly weary, deeply observant.
TONE: Stark, atmospheric, punchy. Narrative weight over marketing hype. No emoji.
ENERGY: Low-frequency, high-impact. A quiet conversation while everyone else shouts.
ARTICLE TITLE: {article.title}
SECTION: {article.section_id}
ARTICLE TEXT:
{article_text}
{exemplar_block}
Write one unique hook for EACH platform below. Each hook should take a DIFFERENT angle on the story — don't repeat the same opening or framing across platforms.
PLATFORM BUDGETS:
{platform_budgets}
RULES:
- Each hook MUST fit within its platform's character limit (including line breaks)
- Each hook should be 2-4 short paragraphs separated by blank lines
- Lead with the most SPECIFIC, SURPRISING finding from the article — a number, a mechanism, a concrete detail. Specificity is inherently more engaging than vague framing.
- No hashtags, no links, no emoji, no "BREAKING" or "ICYMI"
- Each platform gets a DIFFERENT angle — vary the opening and structure
- Paraphrase — don't quote verbatim from the article text
- Don't editorialize beyond what the article supports
NEVER USE THESE PATTERNS:
- "Everyone's talking about..." / "Nobody's talking about..."
- "They're missing the real story" / "Here's what they're not telling you"
- "What they don't want you to know"
- "THREAD:" / "Let's talk about..."
- "The real story is..." / "Read past the headline"
- Any formulaic contrarian hook that promises a reframe without delivering one
- Claims of collective failure ("this was preventable") unless the article explicitly supports them
{compliance_block}
Return ONLY a JSON object mapping platform name to hook text. No markdown fences, no explanation.
Example format: {{"twitter": "hook text...", "bluesky": "hook text...", "threads": "hook text..."}}"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Parse JSON — strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        hooks = json.loads(raw)
        # Validate each hook
        valid = {}
        for pname, hook_text in hooks.items():
            if pname not in platforms:
                continue
            hook_text = hook_text.strip()
            limit = platforms[pname]
            if len(hook_text) < 40:
                continue
            if len(hook_text) > limit:
                # Try sentence-boundary truncation
                hook_text = _truncate_to_limit(hook_text, limit)
            if len(hook_text) <= limit:
                valid[pname] = hook_text
        # Require at least 2 valid hooks (otherwise fallback is better)
        if len(valid) < 2:
            return None
        # Log success
        counts = ", ".join(f"{p}({len(t)})" for p, t in valid.items())
        console.print(f"    [dim]LLM hooks: {counts}[/dim]")
        return valid
    except Exception as e:
        console.print(f"    [dim]LLM hooks failed ({e}), using template fallback[/dim]")
        return None
def _build_post_text(
    article: Article, platform: str, max_chars: int, hashtags: list[str],
    article_index: int = 0, issue_number: int | None = None,
    llm_hook: str | None = None,
) -> str:
    """
    Build a platform-specific post following the Unseen Reality strategy.
    Platform adaptations:
    - **Twitter**: Hook only (no link — link goes in a reply tweet).
      Full 280 chars for the hook. Hashtags appended if room.
    - **Threads**: Hook + link + conversation question.
      Uses the full 500 chars. Ends with a question to drive replies.
    - **Bluesky**: Hook + link (link card auto-generates, so no need
      for the URL in the body — but include it for fallback).
    CTA rotation (every 10 posts):
    - 7 posts (70%) → article link (default)
    - 2 posts (20%) → subscribe link (newsletter funnel)
    - 1 post  (10%) → no link at all (pure engagement play;
      algorithms reward link-free posts, profile bio link converts)
    When ``llm_hook`` is provided (from ``_generate_hooks_llm``), uses
    it directly. When ``llm_hook`` is None, returns empty string —
    caller should skip this platform.
    """
    if not llm_hook:
        return ""
    hook = llm_hook
    link = article.url or ISSUE_URL
    tag_str = " ".join(hashtags)
    # UTM content tag for attribution
    content_tag = f"{article.section_id}_{'feature' if article.is_feature else 'news'}"
    campaign = f"issue_{issue_number}" if issue_number else ""
    # ── CTA rotation ──────────────────────────────────────────────
    # Link is ALWAYS included. Rotation controls hashtags and follow
    # prompts only — never strips the article URL.
    # Cold-start: 50% clean (link only), 30% follow-driving, 20% full
    # Normal:     70% article, 20% subscribe, 10% clean (link only)
    try:
        with open(Path(__file__).parent.parent / "config.yaml") as _cf:
            _cold_cfg = yaml.safe_load(_cf).get("cold_start", {})
        _is_cold = _cold_cfg.get("enabled", False)
    except Exception:
        _is_cold = False
    rotation_slot = article_index % 10
    if _is_cold:
        # Cold-start rotation: vary CTA style, always keep link
        if rotation_slot in (0, 1, 2, 3, 4):
            # 50%: clean look — link but no hashtags
            tag_str = ""
        elif rotation_slot in (5, 6, 7):
            # 30%: follow-driving — link + follow prompt, no hashtags
            tag_str = ""
            _follow_prompts = [
                "Follow @docketclimate for more.",
                "More tomorrow. Follow @docketclimate.",
                "Follow for the stories that fall between the headlines.",
            ]
            hook = hook + "\n\n" + _follow_prompts[article_index % len(_follow_prompts)]
        else:
            # 20%: subscribe link + no hashtags
            if rotation_slot == 9:
                link = SUBSCRIBE_URL
                content_tag = f"{article.section_id}_subscribe"
                tag_str = ""
    else:
        # Normal rotation — always keep a link
        if rotation_slot in (3, 7):
            # 20%: subscribe link
            link = SUBSCRIBE_URL
            content_tag = f"{article.section_id}_subscribe"
            tag_str = ""  # drop hashtags for subscribe posts
        elif rotation_slot == 9:
            # 10%: clean look — article link, no hashtags
            tag_str = ""
    # Tag link with UTM parameters for subscriber attribution
    if link:
        link = _tag_url(link, platform=platform, medium="text",
                        campaign=campaign, content=content_tag)
    # ── Post-level quality gate: detect repeated phrases in final hook ──
    def _has_repeated_content(text: str) -> bool:
        """Return True if text contains a repeated sentence or long phrase."""
        if len(text) < 80:
            return False
        # Check if any 40+ char prefix appears more than once
        first_chunk = text[:40]
        if first_chunk in text[40:]:
            return True
        # Check for duplicate sentences
        sents = [s.strip() for s in text.split("\n\n") if s.strip()]
        seen = set()
        for s in sents:
            normalized = s.strip().rstrip(".!?…")
            if len(normalized) > 20:
                if normalized in seen:
                    return True
                seen.add(normalized)
        return False
    if _has_repeated_content(hook):
        # Deduplicate: keep only unique paragraphs
        seen_parts = []
        for part in hook.split("\n\n"):
            normalized = part.strip().rstrip(".!?…")
            if normalized and normalized not in [p.strip().rstrip(".!?…") for p in seen_parts]:
                seen_parts.append(part)
        hook = "\n\n".join(seen_parts)
    if platform == "twitter":
        # Twitter: hook + hashtags only. No link (link goes in reply).
        available = max_chars
        if tag_str:
            suffix = f"\n\n{tag_str}"
            available = max_chars - len(suffix)
            if available < 80:
                suffix = ""
                available = max_chars
        else:
            suffix = ""
        truncated = _truncate_to_limit(hook, available)
        return f"{truncated}{suffix}"
    elif platform == "threads":
        # Threads: hook + link + conversation question (link always present)
        convo_hook = CONVERSATION_HOOKS[article_index % len(CONVERSATION_HOOKS)]
        if tag_str:
            suffix = f"\n\n{link}\n{tag_str}\n\n{convo_hook}"
        else:
            suffix = f"\n\n{link}\n\n{convo_hook}"
        available = max_chars - len(suffix)
        if available < 60:
            # Drop tags + conversation hook to fit, keep link
            suffix = f"\n\n{link}"
            available = max_chars - len(suffix)
        truncated = _truncate_to_limit(hook, available)
        return f"{truncated}{suffix}"
    else:
        # Bluesky and other platforms: hook + link + hashtags (link always present)
        suffix_parts = [link]
        if tag_str:
            suffix_parts.append(tag_str)
        suffix = "\n".join(suffix_parts)
        suffix_with_gap = f"\n\n{suffix}"
        available = max_chars - len(suffix_with_gap)
        if available < 100:
            # UTM-tagged links can be 130+ chars — drop hashtags first
            suffix_with_gap = f"\n\n{link}"
            available = max_chars - len(suffix_with_gap)
        if available < 80:
            # Link is still too long (long UTM params). Strip UTM and use raw URL.
            raw_link = link.split("?")[0] if "?" in link else link
            suffix_with_gap = f"\n\n{raw_link}"
            available = max_chars - len(suffix_with_gap)
        truncated = _truncate_to_limit(hook, available)
        final = f"{truncated}{suffix_with_gap}"
        # Final safety — should never fire if budgets above are correct
        if len(final) > max_chars:
            truncated = _truncate_to_limit(hook, max_chars - len(suffix_with_gap))
            final = f"{truncated}{suffix_with_gap}"
        return final
def generate_text_posts(issue: DocketIssue, config: dict) -> list[TextPost]:
    """
    Generate platform-specific text posts from a DocketIssue.
    Feature articles get section-specific hashtags.
    News cards get #news and are text-only (no video).
    Uses the "Unseen Reality" voice: stark, atmospheric, data-driven hooks
    that expose the gap between official narratives and ground truth.
    """
    posts = []
    priority_sections = config.get("posting", {}).get(
        "priority_sections", ["lived", "systems", "science"]
    )
    news_posts_per_issue = config.get("posting", {}).get("news_posts_per_issue", 2)
    platforms_config = config.get("platforms", {})
    text_platforms = {}
    for pname, pconf in platforms_config.items():
        if pconf.get("type") == "text":
            limit = pconf.get("max_chars", PLATFORM_LIMITS.get(pname, 280))
            text_platforms[pname] = limit
    if not text_platforms:
        text_platforms = dict(PLATFORM_LIMITS)
    # --- Feature articles: one per priority section ---
    article_counter = 0
    for section_id in priority_sections:
        section_articles = issue.articles_by_section(section_id)
        features = [a for a in section_articles if a.is_feature]
        if not features:
            console.print(
                f"[yellow]No feature articles for section '{section_id}', skipping[/yellow]"
            )
            continue
        best = _pick_best_articles(features, 1)
        if not best:
            continue
        article = best[0]
        console.print(
            f"[cyan]Feature post:[/cyan] {article.title} "
            f"[dim]({section_id})[/dim]"
        )
        # LLM-generated hooks (one API call for all platforms)
        llm_hooks = _generate_hooks_llm(article, text_platforms)
        if not llm_hooks:
            # Retry once — transient API failures happen
            console.print(f"  [yellow]LLM hooks failed, retrying...[/yellow]")
            llm_hooks = _generate_hooks_llm(article, text_platforms)
        if not llm_hooks:
            console.print(
                f"  [red]Skipping {article.title[:50]} — LLM hook generation "
                f"failed after retry[/red]"
            )
            continue
        for platform, max_chars in text_platforms.items():
            hashtags = _select_hashtags(
                article.title, section_id, platform, article.summary or "",
            )
            text = _build_post_text(
                article, platform, max_chars, hashtags,
                article_counter, issue_number=issue.issue_number,
                llm_hook=llm_hooks.get(platform),
            )
            if not text:
                console.print(
                    f"  [yellow]{platform}: skipped (hook quality gate)[/yellow]"
                )
                continue
            post = TextPost(
                platform=platform,
                text=text,
                hashtags=hashtags,
                article_title=article.title,
                section=section_id,
            )
            posts.append(post)
            console.print(
                f"  [green]{platform}[/green] ({len(text)}/{max_chars} chars)"
            )
        article_counter += 1
    # --- News cards: text-only with #news ---
    all_news = [a for a in issue.articles if not a.is_feature and a.url
                and getattr(a, "is_external", False)]
    best_news = _pick_best_articles(all_news, news_posts_per_issue)
    for article in best_news:
        console.print(
            f"[cyan]News post:[/cyan] {article.title} "
            f"[dim]({article.section_id})[/dim]"
        )
        # LLM-generated hooks (one API call for all platforms)
        llm_hooks = _generate_hooks_llm(article, text_platforms)
        if not llm_hooks:
            console.print(f"  [yellow]LLM hooks failed, retrying...[/yellow]")
            llm_hooks = _generate_hooks_llm(article, text_platforms)
        if not llm_hooks:
            console.print(
                f"  [red]Skipping news {article.title[:50]} — LLM hook generation "
                f"failed after retry[/red]"
            )
            continue
        for platform, max_chars in text_platforms.items():
            hashtags = _select_hashtags(
                article.title, article.section_id, platform, article.summary or "",
            )
            text = _build_post_text(
                article, platform, max_chars, hashtags,
                article_counter, issue_number=issue.issue_number,
                llm_hook=llm_hooks.get(platform),
            )
            if not text:
                console.print(
                    f"  [yellow]{platform}: skipped (hook quality gate)[/yellow]"
                )
                continue
            post = TextPost(
                platform=platform,
                text=text,
                hashtags=hashtags,
                article_title=article.title,
                section=article.section_id,
            )
            posts.append(post)
            console.print(
                f"  [green]{platform}[/green] ({len(text)}/{max_chars} chars)"
            )
        article_counter += 1
    console.print(f"\n[bold green]Generated {len(posts)} text posts[/bold green]")
    return posts
def _generate_body_slides(article: Article) -> list[str]:
    """
    Generate 3-5 body slides in the Unseen Reality voice.
    Each slide is under 15 words. Stark. Concrete. No filler.
    """
    slides = []
    summary = article.summary or article.full_text or ""
    if not summary:
        slides.append(f"{article.title or 'The story'}.")
        slides.append("The data doesn\u2019t match the press release.")
        slides.append("Read the full reconstruction at The Docket.")
        return slides
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= 15:
            slides.append(sentence)
        else:
            slides.append(" ".join(words[:14]) + "\u2026")
        if len(slides) >= 5:
            break
    while len(slides) < 3:
        if len(slides) == 0:
            slides.append(f"{article.title or 'The story'}.")
        elif len(slides) == 1:
            slides.append("Official numbers tell one story. The ground tells another.")
        else:
            slides.append("The full deep-dive is at The Docket.")
    return slides[:5]
def _build_voiceover(hook: str, slides: list[str], cta: str) -> str:
    """
    Combine hook, slides, and CTA into flowing voiceover narration.
    Reads naturally spoken aloud — the Observational Insider voice.
    Calm. Measured. Slightly weary.
    """
    spoken_parts = [hook.replace("\n\n", " ")]
    for slide in slides:
        cleaned = slide.rstrip("\u2026").rstrip(".")
        spoken_parts.append(cleaned + ".")
    spoken_parts.append(cta)
    voiceover = " ".join(spoken_parts)
    voiceover = re.sub(r"\s{2,}", " ", voiceover)
    voiceover = re.sub(r"\.{2,}", ".", voiceover)
    return voiceover
def _format_exemplars_for_prompt(
    exemplar_type: str,
    section: str | None = None,
    limit: int = 3,
) -> str:
    """Format stored exemplars as few-shot context for LLM prompts.
    Returns a formatted reference block, or empty string if no
    exemplars are available. NOT wired into the active prompt yet —
    toggle via config ``exemplars.inject_into_prompts: true``.
    Args:
        exemplar_type: "good_copy" or "good_video".
        section: Prefer exemplars from this section.
        limit: Maximum number of exemplars to include.
    Returns:
        Formatted exemplar block, or "".
    """
    try:
        from src.exemplars import get_exemplars_for_prompt
    except ImportError:
        return ""
    exemplars = get_exemplars_for_prompt(exemplar_type, section=section, limit=limit)
    if not exemplars:
        return ""
    lines = [
        "REFERENCE — content you've previously flagged as excellent:",
        "",
    ]
    for i, ex in enumerate(exemplars, 1):
        if exemplar_type == "good_video" and ex.video_script:
            vs = ex.video_script
            slides = vs.get("body_slides", [])
            lines.append(f"Example {i} (section: {ex.section}, engagement: {ex.engagement_score:.0f}):")
            lines.append(f"Title: {ex.article_title}")
            if vs.get("hook"):
                lines.append(f"Hook: {vs['hook']}")
            lines.append("Slides:")
            for slide in slides[:6]:
                lines.append(f"  - {slide}")
            closing = slides[-1] if slides else ""
            if closing:
                lines.append(f"Closing: {closing}")
            score = vs.get("cinematic_score", "-")
            lines.append(f"Cinematic: {score}/10")
            lines.append("")
        else:
            lines.append(f"Example {i} (section: {ex.section}, engagement: {ex.engagement_score:.0f}):")
            lines.append(f"Title: {ex.article_title}")
            lines.append(f"Text: {ex.text[:200]}")
            lines.append("")
    lines.append(
        "WHAT MAKES THESE EXEMPLARS WORK:\n"
        "- They lead with a SPECIFIC finding, not a vague claim\n"
        "- They use concrete nouns and data, not abstract framing\n"
        "- They earn attention through substance, not engagement formulas\n"
        "- They paraphrase rather than quoting verbatim from the source\n"
        "- They match the platform's character and culture\n"
        "\n"
        "ANTI-PATTERNS — never do these:\n"
        '- "Everyone\'s talking about X. They\'re missing the real story."\n'
        '- "Nobody\'s talking about this."\n'
        '- "Here\'s what they don\'t want you to know."\n'
        "- Leading with a vague contrarian claim instead of a specific finding\n"
        "- Making moral claims the article doesn't support\n"
        "\n"
        "Extract PRINCIPLES from the exemplars and apply them to "
        "the current article. Do not copy their structure literally."
    )
    return "\n".join(lines)
def _generate_body_slides_llm(article: Article, hook: str) -> dict | None:
    """
    Generate video script body slides + voiceover using Claude API.
    Returns a dict with keys:
        body_slides: list[str]  — 5-7 on-screen text slides (≤15 words each)
        voiceover_lines: list[str] — spoken narration per slide (adds context)
        closing_slide: str — share-worthy closing statement
        closing_voiceover: str — spoken version of closing
    Returns None if the API call fails (caller falls back to regex extraction).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        console.print(f"[yellow]Claude API init failed: {e}[/yellow]")
        return None
    source_text = article.full_text or article.summary or ""
    if not source_text:
        return None
    # Exemplar injection: pull flagged "good_video" examples into the prompt
    exemplar_block = ""
    try:
        with open(Path(__file__).parent.parent / "config.yaml") as f:
            _cfg = yaml.safe_load(f)
        exemplar_cfg = _cfg.get("exemplars", {})
        if exemplar_cfg.get("inject_into_prompts", False):
            exemplar_block = _format_exemplars_for_prompt(
                "good_video",
                section=article.section_id,
                limit=exemplar_cfg.get("max_per_prompt", 3),
            )
    except Exception:
        pass  # Silent — exemplar injection is best-effort
    # Narrative structure selection (Fix W)
    narrative_key, narrative_reason = _select_narrative_structure(article)
    narrative_block = NARRATIVE_STRUCTURES[narrative_key]["prompt_block"]
    narrative_name = NARRATIVE_STRUCTURES[narrative_key]["name"]
    console.print(
        f"    [dim]Narrative: {narrative_name} ({narrative_key}) "
        f"[{narrative_reason}][/dim]"
    )
    # Cold-start mode: shorter videos for higher completion rates (Fix X4)
    try:
        with open(Path(__file__).parent.parent / "config.yaml") as _cvf:
            _cv_cfg = yaml.safe_load(_cvf).get("cold_start", {})
        _is_cold_video = _cv_cfg.get("enabled", False)
        _cold_video = _cv_cfg.get("video", {})
    except Exception:
        _is_cold_video = False
        _cold_video = {}
    if _is_cold_video:
        _slide_count = _cold_video.get("body_slides", 3)
        _max_words = _cold_video.get("max_words_per_slide", 15)
        _slide_range = f"{_slide_count - 1}-{_slide_count}"  # "2-3"
        _example_slides = ", ".join([f'"slide {i}"' for i in range(1, _slide_count + 1)])
        _example_vos = ", ".join([f'"spoken line {i}"' for i in range(1, _slide_count + 1)])
        _example_imgs = ", ".join([f'"prompt {i}"' for i in range(1, _slide_count + 2)])
        console.print(f"    [dim]Cold-start video: {_slide_range} body slides, max {_max_words} words[/dim]")
    else:
        _slide_count = 4
        _max_words = 18
        _slide_range = "4-5"
        _example_slides = '"slide 1", "slide 2", "slide 3", "slide 4"'
        _example_vos = '"spoken line 1", "spoken line 2", "spoken line 3", "spoken line 4"'
        _example_imgs = '"prompt 1", "prompt 2", "prompt 3", "prompt 4", "prompt 5"'
    # Build compliance guidelines for video scripts (best-effort)
    video_compliance_block = ""
    try:
        from src.compliance import build_compliance_guidelines
        with open(Path(__file__).parent.parent / "config.yaml") as _vcf:
            _vc_cfg = yaml.safe_load(_vcf).get("compliance", {})
        if _vc_cfg.get("prompt_enrichment", False):
            video_compliance_block = build_compliance_guidelines("video")
    except Exception:
        pass
    prompt = f"""You are an investigative documentarian and "The Observational Insider." Your task is to adapt a climate journalism story into a vertical video script that feels like a high-end cinematic short.
VOICE:
Grounded, spare, and haunting. You prefer the "cold truth" over "hot takes." You find the tension in the quiet details. Avoid journalistic clichés (e.g., "In a world where...", "The clock is ticking"). Use fragments. Use silence. No hype. No emojis.
ARTICLE TITLE: {article.title}
ARTICLE HOOK (already written — don't repeat this content):
{hook}
ARTICLE TEXT:
{source_text[:3000]}
Generate EXACTLY this JSON structure:
{{
  "body_slides": [{_example_slides}],
  "voiceover_lines": [{_example_vos}],
  "closing_slide": "A heavy realization",
  "closing_voiceover": "Spoken version of the closing",
  "cinematic_score": 7,
  "image_prompts": [{_example_imgs}],
  "background_prompt": "A single image prompt for the article theme",
  "narrative_structure": "{narrative_key}"
}}
RULES:
- body_slides: {_slide_range} slides that ADVANCE the narrative beyond the hook. Max {_max_words} words each. Find the "micro-moments" — a cracked gauge, a salt-stained boot, a flickering light. Focus on nouns and verbs. Fragments are good. Silence between thoughts is good.
- voiceover_lines: One per body slide. One sentence each, max {_max_words - 3} words. Spare. Let the image carry the weight.
- closing_slide: A "heavy" realization. Not a call to action, but a "cold water to the face" statement that demands a re-watch.
- closing_voiceover: Spoken version of the closing. Can be slightly longer.
{narrative_block}
CRITICAL: Follow the narrative structure above, but never sacrifice texture for structure. Each slide should still feel like a fragment — a moment, a detail, a cold fact. The structure tells you WHAT ROLE each slide plays. The voice tells you HOW it sounds. Both matter.
CINEMATIC SCORING:
Rate this article's visual potential 1-10 based on:
- Physical place? +3
- Visual subjects? +3
- Narrative arc? +2
- Specific scenes describable for AI image generation? +2
IMAGE PROMPTS:
If cinematic_score >= 6: generate "image_prompts" — one per body slide. Set "background_prompt" to "".
If cinematic_score < 6: generate one "background_prompt" for the overall article theme. Set "image_prompts" to [].
IMAGE PROMPT RULES:
- Photorealistic scene, 1-2 sentences
- Editorial documentary style, natural lighting, atmospheric
- No text, logos, UI elements, or watermarks
- Include specific visual details: colors, materials, weather, time of day, scale
- Portrait orientation (9:16 vertical)
- Each prompt should depict a different scene matching its body slide
{video_compliance_block}
{exemplar_block}
Return ONLY valid JSON. No markdown, no explanation."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,  # was 1024 — image prompts were getting truncated
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to recover truncated JSON by closing open structures
            _recovered = raw.rstrip()
            result = None
            for _closer in ['"}', '"]', '"}]', '"]}', '"}]}']:
                try:
                    result = json.loads(_recovered + _closer)
                    console.print("[dim]Recovered truncated JSON from Claude response[/dim]")
                    break
                except json.JSONDecodeError:
                    continue
            if result is None:
                console.print("[yellow]Claude returned invalid JSON for video script[/yellow]")
                return None
        # Validate structure
        if (isinstance(result.get("body_slides"), list) and
                len(result["body_slides"]) >= 3):
            # Validate and normalize cinematic scoring fields
            score = result.get("cinematic_score", 0)
            if not isinstance(score, int):
                try:
                    score = int(score)
                except (TypeError, ValueError):
                    score = 0
            result["cinematic_score"] = max(0, min(10, score))
            if score >= 6:
                result["video_tier"] = "cinematic"
                prompts = result.get("image_prompts", [])
                if not isinstance(prompts, list) or not prompts:
                    # LLM scored high but gave no prompts — downgrade
                    result["video_tier"] = "narrative"
                    result["image_prompts"] = []
                else:
                    # Pad prompts if fewer than body slides
                    while len(prompts) < len(result["body_slides"]):
                        prompts.append(f"Atmospheric photograph related to: {article.title}")
                    result["image_prompts"] = prompts
            else:
                result["video_tier"] = "narrative"
                result["image_prompts"] = []
                if not result.get("background_prompt"):
                    result["background_prompt"] = (
                        f"Atmospheric documentary photograph depicting the theme of: "
                        f"{article.title}"
                    )
            tier_label = result["video_tier"].upper()
            # Tag with the narrative structure used (for logging/debugging)
            result["narrative_structure"] = narrative_key
            console.print(
                f"    [green]Claude generated {len(result['body_slides'])} slides + "
                f"closing for {article.title[:40]} "
                f"[cinematic: {result['cinematic_score']}/10 → {tier_label}] "
                f"[narrative: {narrative_name}][/green]"
            )
            return result
        else:
            console.print("[yellow]Claude returned invalid structure, falling back[/yellow]")
            return None
    except Exception as e:
        console.print(f"[yellow]Claude API call failed: {e}[/yellow]")
        return None
def generate_video_scripts(issue: DocketIssue, config: dict) -> list[VideoScript]:
    """
    Generate video scripts from the most compelling articles.
    Voice: Observational Insider. Stark visuals, data that lands, no hype.
    """
    scripts = []
    video_count = config.get("posting", {}).get("video_posts_per_issue", 2)
    # Only feature articles get video — they're the marquee pieces
    feature_articles = [a for a in issue.articles if a.is_feature]
    best = _pick_best_articles(feature_articles, video_count)
    if not best:
        console.print("[yellow]No suitable articles found for video scripts[/yellow]")
        return scripts
    # Cold-start video config (for slide cap enforcement)
    _cold_cfg = config.get("cold_start", {})
    _cold_video = _cold_cfg.get("video", {}) if _cold_cfg.get("enabled", False) else {}
    _is_cold_video = bool(_cold_video)
    console.print(f"[cyan]Generating {len(best)} video scripts...[/cyan]")
    for script_idx, article in enumerate(best):
        # Generate video hook via LLM (single-platform call, video-optimized budget)
        _video_hooks = _generate_hooks_llm(article, {"video": 200})
        if not _video_hooks:
            _video_hooks = _generate_hooks_llm(article, {"video": 200})
        if not _video_hooks or "video" not in _video_hooks:
            console.print(
                f"  [red]Skipping video for {article.title[:50]} "
                f"— LLM hook generation failed[/red]"
            )
            continue
        hook = _video_hooks["video"]
        # Try LLM-generated slides first, fall back to regex extraction
        llm_result = _generate_body_slides_llm(article, hook)
        if llm_result:
            body_slides = llm_result["body_slides"]
            voiceover_lines = llm_result.get("voiceover_lines", body_slides)
            closing_slide = llm_result.get("closing_slide", "")
            closing_vo = llm_result.get("closing_voiceover", closing_slide)
        else:
            body_slides = _generate_body_slides(article)
            voiceover_lines = body_slides
            closing_slide = ""
            closing_vo = ""
        # Enforce cold-start slide cap (LLM sometimes overshoots)
        if _is_cold_video:
            _max_slides = _cold_video.get("body_slides", 3)
            if len(body_slides) > _max_slides:
                body_slides = body_slides[:_max_slides]
                voiceover_lines = voiceover_lines[:_max_slides]
        # Use article-specific URL, fall back to issue URL
        link = article.url or ISSUE_URL
        link = _tag_url(link, platform="reels", medium="video",
                        campaign=f"issue_{issue.issue_number}" if issue.issue_number else "",
                        content=f"{article.section_id}_video")
        # Rotate spoken CTA
        spoken_cta = VIDEO_CTA_SPOKEN[script_idx % len(VIDEO_CTA_SPOKEN)]
        # Use share-worthy closer if LLM didn't provide one
        if not closing_slide:
            closing_slide = SHARE_WORTHY_CLOSERS[script_idx % len(SHARE_WORTHY_CLOSERS)]
            closing_vo = closing_slide
        # Append closing slide to body_slides
        body_slides.append(closing_slide)
        voiceover_lines.append(closing_vo)
        cta = VIDEO_CTA_TEXT
        # Build voiceover: prefer LLM voiceover_lines (conversational) over
        # raw slide text concatenation from _build_voiceover()
        if llm_result and llm_result.get("voiceover_lines"):
            spoken_parts = [hook.replace("\n\n", " ")]
            spoken_parts.extend(voiceover_lines)
            spoken_parts.append(spoken_cta)
            voiceover = " ".join(p.strip() for p in spoken_parts if p.strip())
        else:
            voiceover = _build_voiceover(hook, body_slides, spoken_cta)
        # Extract cinematic scoring fields from LLM result
        cinematic_score = 0
        video_tier = "narrative"
        image_prompts = []
        background_prompt = ""
        if llm_result:
            cinematic_score = llm_result.get("cinematic_score", 0)
            video_tier = llm_result.get("video_tier", "narrative")
            image_prompts = llm_result.get("image_prompts", [])
            background_prompt = llm_result.get("background_prompt", "")
        script = VideoScript(
            title=article.title,
            hook=hook,
            body_slides=body_slides,
            cta=cta,
            voiceover_text=voiceover,
            article_title=article.title,
            section=article.section_id,
            url=link,
            cinematic_score=cinematic_score,
            video_tier=video_tier,
            image_prompts=image_prompts,
            background_prompt=background_prompt,
            voiceover_lines=voiceover_lines,
        )
        scripts.append(script)
        tier_label = video_tier.upper()
        console.print(f"  [green]Script:[/green] {article.title}")
        console.print(f"    Hook: {hook[:80]}{'...' if len(hook) > 80 else ''}")
        console.print(f"    Slides: {len(body_slides)} {'(LLM)' if llm_result else '(regex)'}")
        console.print(f"    Cinematic: {cinematic_score}/10 → {tier_label}")
        console.print(f"    Voiceover: {len(voiceover)} chars")
    console.print(
        f"\n[bold green]Generated {len(scripts)} video scripts[/bold green]"
    )
    return scripts
def generate_all_text_posts(issue: DocketIssue, config: dict) -> list[TextPost]:
    """
    Generate text posts for ALL feature articles and top news cards.
    Unlike generate_text_posts() which picks 1 feature per priority section,
    this generates posts for every feature across all 6 sections, plus a
    curated set of news cards — enough to fill a full week schedule.
    Returns one TextPost per article per platform. The scheduler/curator
    decides which actually get posted.
    """
    posts = []
    platforms_config = config.get("platforms", {})
    text_platforms = {}
    for pname, pconf in platforms_config.items():
        if pconf.get("type") == "text":
            limit = pconf.get("max_chars", PLATFORM_LIMITS.get(pname, 280))
            text_platforms[pname] = limit
    if not text_platforms:
        text_platforms = dict(PLATFORM_LIMITS)
    # All feature articles across all sections
    features = [a for a in issue.articles if a.is_feature]
    features_sorted = sorted(features, key=_score_article, reverse=True)
    console.print(f"[cyan]Generating drafts for {len(features_sorted)} features...[/cyan]")
    article_counter = 0
    for article in features_sorted:
        # Try LLM-generated hooks (one API call for all platforms)
        llm_hooks = _generate_hooks_llm(article, text_platforms)
        for platform, max_chars in text_platforms.items():
            hashtags = _select_hashtags(
                article.title, article.section_id, platform, article.summary or "",
            )
            text = _build_post_text(
                article, platform, max_chars, hashtags,
                article_counter, issue_number=issue.issue_number,
                llm_hook=llm_hooks.get(platform) if llm_hooks else None,
            )
            if not text:
                continue
            posts.append(TextPost(
                platform=platform,
                text=text,
                hashtags=hashtags,
                article_title=article.title,
                section=article.section_id,
            ))
        article_counter += 1
    # Top news cards (external source links only — internal snippet pages are too thin)
    news = [a for a in issue.articles if not a.is_feature and a.url
            and getattr(a, "is_external", False)]
    news_sorted = sorted(news, key=_score_article, reverse=True)
    # Cap at ~20 for curation — more than enough for a week
    news_candidates = news_sorted[:20]
    console.print(f"[cyan]Generating drafts for {len(news_candidates)} news cards...[/cyan]")
    for article in news_candidates:
        # Try LLM-generated hooks (one API call for all platforms)
        llm_hooks = _generate_hooks_llm(article, text_platforms)
        for platform, max_chars in text_platforms.items():
            hashtags = _select_hashtags(
                article.title, article.section_id, platform, article.summary or "",
            )
            text = _build_post_text(
                article, platform, max_chars, hashtags,
                article_counter, issue_number=issue.issue_number,
                llm_hook=llm_hooks.get(platform) if llm_hooks else None,
            )
            if not text:
                continue
            posts.append(TextPost(
                platform=platform,
                text=text,
                hashtags=hashtags,
                article_title=article.title,
                section=article.section_id,
            ))
        article_counter += 1
    console.print(
        f"[bold green]Generated {len(posts)} draft text posts "
        f"({len(features_sorted)} features + {len(news_candidates)} news × "
        f"{len(text_platforms)} platforms)[/bold green]"
    )
    return posts
def generate_all_video_scripts(issue: DocketIssue, config: dict) -> list[VideoScript]:
    """
    Generate video scripts for top feature articles — enough for a full week.
    Default: top 8 scored features. The curator narrows to ~5 (one per weekday).
    """
    scripts = []
    video_candidates = config.get("schedule", {}).get("videos_per_week", 5) + 3  # extras for curation
    feature_articles = [a for a in issue.articles if a.is_feature]
    best = _pick_best_articles(feature_articles, video_candidates)
    if not best:
        console.print("[yellow]No suitable feature articles for video scripts[/yellow]")
        return scripts
    # Cold-start video config (for slide cap enforcement)
    _cold_cfg = config.get("cold_start", {})
    _cold_video = _cold_cfg.get("video", {}) if _cold_cfg.get("enabled", False) else {}
    _is_cold_video = bool(_cold_video)
    console.print(f"[cyan]Generating {len(best)} video script drafts...[/cyan]")
    for script_idx, article in enumerate(best):
        # Generate video hook via LLM
        _video_hooks = _generate_hooks_llm(article, {"video": 200})
        if not _video_hooks:
            _video_hooks = _generate_hooks_llm(article, {"video": 200})
        if not _video_hooks or "video" not in _video_hooks:
            console.print(
                f"  [red]Skipping video draft for {article.title[:50]} "
                f"— LLM hook generation failed[/red]"
            )
            continue
        hook = _video_hooks["video"]
        # Try LLM-generated slides first, fall back to regex extraction
        llm_result = _generate_body_slides_llm(article, hook)
        if llm_result:
            body_slides = llm_result["body_slides"]
            closing_slide = llm_result.get("closing_slide", "")
            closing_vo = llm_result.get("closing_voiceover", closing_slide)
        else:
            body_slides = _generate_body_slides(article)
            closing_slide = ""
            closing_vo = ""
        # Enforce cold-start slide cap (LLM sometimes overshoots)
        if _is_cold_video:
            _max_slides = _cold_video.get("body_slides", 3)
            if len(body_slides) > _max_slides:
                body_slides = body_slides[:_max_slides]
        link = article.url or ISSUE_URL
        spoken_cta = VIDEO_CTA_SPOKEN[script_idx % len(VIDEO_CTA_SPOKEN)]
        if not closing_slide:
            closing_slide = SHARE_WORTHY_CLOSERS[script_idx % len(SHARE_WORTHY_CLOSERS)]
        body_slides.append(closing_slide)
        cta = VIDEO_CTA_TEXT
        voiceover = _build_voiceover(hook, body_slides, spoken_cta)
        # Extract cinematic scoring fields from LLM result
        cinematic_score = 0
        video_tier = "narrative"
        image_prompts = []
        background_prompt = ""
        if llm_result:
            cinematic_score = llm_result.get("cinematic_score", 0)
            video_tier = llm_result.get("video_tier", "narrative")
            image_prompts = llm_result.get("image_prompts", [])
            background_prompt = llm_result.get("background_prompt", "")
        script = VideoScript(
            title=article.title,
            hook=hook,
            body_slides=body_slides,
            cta=cta,
            voiceover_text=voiceover,
            article_title=article.title,
            section=article.section_id,
            url=link,
            cinematic_score=cinematic_score,
            video_tier=video_tier,
            image_prompts=image_prompts,
            background_prompt=background_prompt,
        )
        scripts.append(script)
    console.print(
        f"[bold green]Generated {len(scripts)} video script drafts[/bold green]"
    )
    return scripts
def _print_text_posts(posts: list[TextPost]) -> None:
    """Pretty-print generated text posts."""
    table = Table(title="Generated Text Posts", show_lines=True)
    table.add_column("Platform", style="cyan", width=10)
    table.add_column("Section", style="magenta", width=12)
    table.add_column("Article", style="yellow", width=30)
    table.add_column("Post Text", style="white", width=60)
    table.add_column("Chars", justify="right", width=6)
    for post in posts:
        table.add_row(
            post.platform,
            post.section,
            post.article_title[:28] + (".." if len(post.article_title) > 28 else ""),
            post.text,
            str(len(post.text)),
        )
    console.print(table)
def _print_video_scripts(scripts: list[VideoScript]) -> None:
    """Pretty-print generated video scripts."""
    for i, script in enumerate(scripts, 1):
        console.print(f"\n[bold cyan]━━━ Video Post {i} ━━━[/bold cyan]")
        console.print(f"[dim]Article:[/dim] {script.title} [dim]({script.section})[/dim]")
        # The publishable caption — no labels
        console.print(f"\n[bold]Post Caption:[/bold]")
        console.print(f"  {script.caption}")
        # Slide breakdown for video production
        console.print(f"\n[dim]Slides ({len(script.body_slides)}):[/dim]")
        for j, slide in enumerate(script.body_slides, 1):
            word_count = len(slide.split())
            color = "green" if word_count <= 15 else "red"
            console.print(f"  [{color}]{j}. {slide}[/{color}] [dim]({word_count}w)[/dim]")
        # Voiceover for ElevenLabs
        console.print(f"\n[dim]Voiceover ({len(script.voiceover_text)} chars):[/dim]")
        console.print(f"  [dim]{script.voiceover_text}[/dim]")
# ---------------------------------------------------------------------------
# Pre/Post Issue Posts (Fix M)
# ---------------------------------------------------------------------------
# Sunday teaser templates — tease one headline from the upcoming issue
TEASER_TEMPLATES = [
    "Issue #{issue_number} drops tomorrow.\n\nOne thing you'll find inside:\n\n{headline}\n\n{subscribe_link}",
    "Tomorrow's Docket has {article_count} stories you won't see anywhere else.\n\nIncluding: {headline}\n\n{subscribe_link}",
    "New issue incoming.\n\n\"{headline}\"\n\nGet it free tomorrow → {subscribe_link}",
    "Monday morning. New Docket.\n\nHere's a preview: {headline}\n\n{subscribe_link}",
]
# Friday wrap-up templates — summarize the week + subscribe link
WRAPUP_TEMPLATES = [
    "This week in The Docket:\n\n{headlines}\n\nGet next week's issue → {subscribe_link}",
    "Five days. {article_count} stories. Here's what The Docket covered this week:\n\n{headlines}\n\n{subscribe_link}",
    "Week in review.\n\n{headlines}\n\nNever miss an issue → {subscribe_link}",
]
def build_teaser_post(
    issue_number: int | None,
    teaser_article_title: str,
    article_count: int,
    platform: str,
    max_chars: int,
) -> str:
    """
    Build a Sunday evening teaser post for the upcoming issue.
    Rotates through teaser templates based on issue number.
    """
    idx = (issue_number or 0) % len(TEASER_TEMPLATES)
    template = TEASER_TEMPLATES[idx]
    # Tag subscribe link with UTM for attribution
    subscribe_link = _tag_url(
        SUBSCRIBE_URL, platform=platform, medium="text",
        campaign=f"issue_{issue_number}" if issue_number else "",
        content="teaser",
    )
    text = template.format(
        issue_number=issue_number or "?",
        headline=teaser_article_title,
        article_count=article_count,
        subscribe_link=subscribe_link,
    )
    # Platform-specific tweaks
    if platform == "threads":
        text += "\n\nWhat topics do you want us to cover?"
    elif platform == "twitter":
        # Keep it tight for Twitter — no subscribe link in tweet body
        # (link goes in reply if needed)
        text = text.replace(f"\n\n{SUBSCRIBE_URL}", "")
        text = text.replace(f"→ {SUBSCRIBE_URL}", "")
        text = text.replace(f"Get it free tomorrow → ", "Get it free tomorrow.")
        text = text.replace(f"Get next week's issue → ", "")
    if len(text) > max_chars:
        text = _truncate_to_limit(text, max_chars)
    return text
def build_wrapup_post(
    issue_number: int | None,
    top_titles: list[str],
    article_count: int,
    platform: str,
    max_chars: int,
) -> str:
    """
    Build a Friday wrap-up post summarizing the week's coverage.
    Lists top 3 headlines + subscribe link.
    """
    # Format headline list
    headlines = "\n".join(f"• {t}" for t in top_titles[:3])
    idx = (issue_number or 0) % len(WRAPUP_TEMPLATES)
    template = WRAPUP_TEMPLATES[idx]
    # Tag subscribe link with UTM for attribution
    subscribe_link = _tag_url(
        SUBSCRIBE_URL, platform=platform, medium="text",
        campaign=f"issue_{issue_number}" if issue_number else "",
        content="wrapup",
    )
    text = template.format(
        issue_number=issue_number or "?",
        headlines=headlines,
        article_count=article_count,
        subscribe_link=subscribe_link,
    )
    # Platform-specific tweaks
    if platform == "threads":
        text += "\n\nWhat was the biggest story for you this week?"
    elif platform == "twitter":
        text = text.replace(f"\n\n{SUBSCRIBE_URL}", "")
        text = text.replace(f"→ {SUBSCRIBE_URL}", "")
        text = text.replace(f"Get next week's issue → ", "")
        text = text.replace(f"Never miss an issue → ", "Never miss an issue.")
    if len(text) > max_chars:
        text = _truncate_to_limit(text, max_chars)
    return text
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    issue_files = sorted(output_dir.glob("issue_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not issue_files:
        console.print("[red]No issue files found in output/[/red]")
        console.print("[yellow]Run the scraper first to generate issue data.[/yellow]")
        raise SystemExit(1)
    issue_path = issue_files[0]
    config_path = project_root / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        console.print(f"[green]Loaded config from {config_path}[/green]")
    else:
        console.print("[yellow]No config.yaml found, using defaults[/yellow]")
        config = {
            "posting": {
                "priority_sections": ["lived", "systems", "science"],
                "video_posts_per_issue": 2,
            },
            "platforms": {
                "bluesky": {"type": "text", "max_chars": 300},
                "twitter": {"type": "text", "max_chars": 280},
                "threads": {"type": "text", "max_chars": 500},
            },
        }
    console.print(f"[cyan]Loading issue from {issue_path}[/cyan]")
    issue = DocketIssue.load(issue_path)
    console.print(
        f"[green]Loaded issue #{issue.issue_number or 'unknown'} "
        f"with {len(issue.articles)} articles[/green]"
    )
    console.rule("[bold]Text Posts[/bold]")
    text_posts = generate_text_posts(issue, config)
    _print_text_posts(text_posts)
    console.rule("[bold]Video Scripts[/bold]")
    video_scripts = generate_video_scripts(issue, config)
    _print_video_scripts(video_scripts)
