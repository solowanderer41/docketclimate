"""
Scraper for The Docket (anchor.is)

The site is a Next.js app with server-side rendering. We fetch each section
page and parse the rendered HTML to extract article content.
"""

import httpx
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import Optional
import yaml
import json
import re
from pathlib import Path
from rich.console import Console

console = Console()

SECTIONS = [
    {"id": "lived", "name": "Human Stories"},
    {"id": "systems", "name": "Cascading Effects"},
    {"id": "science", "name": "Emerging Science"},
    {"id": "futures", "name": "Future Tense"},
    {"id": "archive", "name": "Climate Past"},
    {"id": "lab", "name": "The Lab"},
]

BASE_URL = "https://www.anchor.is"


@dataclass
class Article:
    title: str
    section: str
    section_id: str
    author: Optional[str] = None
    date: Optional[str] = None
    summary: Optional[str] = None
    full_text: Optional[str] = None
    url: Optional[str] = None
    is_feature: bool = False
    tags: list[str] = field(default_factory=list)
    is_external: bool = False
    subsection: Optional[str] = None


@dataclass
class DocketIssue:
    issue_number: Optional[int] = None
    publish_date: Optional[str] = None
    theme: Optional[str] = None
    articles: list[Article] = field(default_factory=list)
    raw_sections: dict[str, str] = field(default_factory=dict)

    def articles_by_section(self, section_id: str) -> list[Article]:
        return [a for a in self.articles if a.section_id == section_id]

    def to_dict(self) -> dict:
        return {
            "issue_number": self.issue_number,
            "publish_date": self.publish_date,
            "theme": self.theme,
            "articles": [
                {
                    "title": a.title,
                    "section": a.section,
                    "section_id": a.section_id,
                    "author": a.author,
                    "date": a.date,
                    "summary": a.summary,
                    "full_text": a.full_text,
                    "url": a.url,
                    "is_feature": a.is_feature,
                    "tags": a.tags,
                    "is_external": a.is_external,
                    "subsection": a.subsection,
                }
                for a in self.articles
            ],
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        console.print(f"[green]Saved issue data to {path}[/green]")

    @classmethod
    def load(cls, path: Path) -> "DocketIssue":
        with open(path) as f:
            data = json.load(f)
        issue = cls(
            issue_number=data.get("issue_number"),
            publish_date=data.get("publish_date"),
            theme=data.get("theme"),
        )
        for a in data.get("articles", []):
            issue.articles.append(Article(**a))
        return issue


def _fetch_page(url: str) -> str:
    """Fetch a page and return the HTML content."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"
    }
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text


def _extract_issue_metadata(html: str) -> dict:
    """Extract issue number, date, and theme from the main page."""
    soup = BeautifulSoup(html, "lxml")
    meta = {}

    # Try to get issue number from meta tags or content
    og_url = soup.find("meta", property="og:url")
    if og_url:
        url_content = og_url.get("content", "")
        match = re.search(r"/issue/(\d+)", url_content)
        if match:
            meta["issue_number"] = int(match.group(1))

    # Try og:description for theme
    og_desc = soup.find("meta", property="og:description")
    if og_desc:
        meta["theme"] = og_desc.get("content", "")

    return meta


def _strip_category_suffix(title: str) -> str:
    """
    Strip trailing ' in {Category}' from news article aria-labels.

    News articles use: "Read article: {title} in {Category}"
    e.g. "North Carolina Builds Wetlands Instead of Walls in Natural Infrastructure"
    -> "North Carolina Builds Wetlands Instead of Walls"
    """
    # Known category patterns end with capitalized words after " in "
    # Try progressively shorter " in " splits from the right
    parts = title.rsplit(" in ", 1)
    if len(parts) == 2:
        potential_title, potential_category = parts
        # Category names are short (1-4 words), mostly title-cased
        # Allow lowercase connectors like "to", "and", "of", "the", "for"
        cat_words = potential_category.strip().split()
        connectors = {"to", "and", "of", "the", "for", "a", "an"}
        if 1 <= len(cat_words) <= 4 and all(
            w[0].isupper() or w.lower() in connectors for w in cat_words if w
        ):
            return potential_title.strip()
    return title


def _build_article_url_map(html: str, issue_number: int | None) -> dict[str, tuple[str, bool]]:
    """
    Build a mapping of article title -> (URL, is_feature) from aria-label attributes.

    The Docket uses three aria-label patterns:
    - "Read full article: {title}" — feature articles (clean title)
    - "Continue Reading: {title}" — secondary feature articles (clean title)
    - "Read article: {title} in {category}" — news cards (title + category suffix)

    Feature articles are the marquee pieces (video + text posts).
    News cards are brief items (text-only posts with #news).
    """
    title_to_info: dict[str, tuple[str, bool]] = {}

    # Combined pattern for both attribute orders
    aria_patterns = [
        # aria-label first, then href
        r'<a[^>]*?aria-label="(Read full article|Continue Reading|Read article):\s*([^"]+)"[^>]*?href="([^"]+)"[^>]*>',
        # href first, then aria-label
        r'<a[^>]*?href="([^"]+)"[^>]*?aria-label="(Read full article|Continue Reading|Read article):\s*([^"]+)"[^>]*>',
    ]

    for i, pattern in enumerate(aria_patterns):
        for match in re.finditer(pattern, html):
            if i == 0:
                label_type, raw_title, href = match.group(1), match.group(2), match.group(3)
            else:
                href, label_type, raw_title = match.group(1), match.group(2), match.group(3)

            if "/article/" not in href:
                continue

            title = _clean_html_entities(raw_title.strip())

            # "Read article:" pattern includes category suffix — strip it
            is_feature = label_type in ("Read full article", "Continue Reading")
            if label_type == "Read article":
                title = _strip_category_suffix(title)

            if title not in title_to_info:
                url = _normalize_article_url(href, issue_number)
                title_to_info[title] = (url, is_feature)

    return title_to_info


def _collect_all_article_urls(html: str, issue_number: int | None) -> list[str]:
    """Collect all unique /article/ URLs from the page."""
    urls = []
    seen = set()
    for match in re.finditer(r'href="(/docket/issue/[^"]*?/article/\d+)"', html):
        href = match.group(1)
        url = _normalize_article_url(href, issue_number)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _normalize_article_url(href: str, issue_number: int | None) -> str:
    """Convert a relative article href to a full URL, replacing 'latest' with issue number."""
    if issue_number:
        href = re.sub(r"/issue/latest/", f"/issue/{issue_number}/", href)
    if href.startswith("/"):
        return f"{BASE_URL}{href}"
    return href


def _clean_html_entities(text: str) -> str:
    """Decode common HTML entities in aria-label values."""
    import html as html_module
    return html_module.unescape(text)


def _fix_rsc_text(text: str) -> str:
    """
    Clean text extracted from Next.js RSC rendered HTML.

    When using get_text(' ') to avoid missing spaces between elements,
    drop-cap styling creates single-letter fragments like "N ew" or "F irst".
    This rejoins those fragments while keeping the space-separated extraction.
    Also normalizes whitespace.
    """
    # Rejoin single letter + space + lowercase continuation: "N ew" -> "New"
    text = re.sub(r"\b([A-Z]) ([a-z])", r"\1\2", text)
    # Normalize multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _parse_section_content(html: str, section_id: str, section_name: str,
                           issue_number: int | None = None) -> list[Article]:
    """
    Parse article content from a section page.

    Uses two strategies:
    1. Extract article URLs from aria-label patterns (most reliable)
    2. Parse headings for article metadata (title, author, date, summary)
    Then match articles to their URLs by title.
    """
    soup = BeautifulSoup(html, "lxml")
    articles = []

    # Step 1: Build title->URL map from aria-labels and collect all article URLs
    title_url_map = _build_article_url_map(html, issue_number)
    all_article_urls = _collect_all_article_urls(html, issue_number)

    # Step 2: Parse article content from headings
    headings = soup.find_all(["h1", "h2", "h3", "h4"])
    for heading in headings:
        title_text = heading.get_text(strip=True)
        # Skip navigation-like headings
        if title_text in ("The Docket", section_name, "") or len(title_text) < 5:
            continue

        author = None
        date = None
        summary = None

        # Match URL from the aria-label map (exact or fuzzy match)
        # Map returns (url, is_feature) tuples
        map_entry = title_url_map.get(title_text)
        url = map_entry[0] if map_entry else None
        is_feature = map_entry[1] if map_entry else False
        if not url:
            # Try case-insensitive / partial match
            title_lower = title_text.lower()
            for map_title, (map_url, map_is_feature) in title_url_map.items():
                if map_title.lower() == title_lower or title_lower in map_title.lower():
                    url = map_url
                    is_feature = map_is_feature
                    break

        # If still no URL, check if heading itself is inside an <a> with /article/ href
        external_url = None
        subsection_name = None
        if not url:
            parent_link = heading.find_parent("a", href=re.compile(r"/article/"))
            if parent_link:
                url = _normalize_article_url(parent_link["href"], issue_number)
            else:
                link = heading.find("a", href=re.compile(r"/article/"))
                if link:
                    url = _normalize_article_url(link["href"], issue_number)

        # If still no internal URL, check for external source links
        if not url:
            parent_link = heading.find_parent("a", href=True)
            if parent_link:
                href = parent_link.get("href", "")
                if href.startswith("http") and "anchor.is" not in href:
                    external_url = href
            if not external_url:
                child_link = heading.find("a", href=True)
                if child_link:
                    href = child_link.get("href", "")
                    if href.startswith("http") and "anchor.is" not in href:
                        external_url = href
            # Find subsection name from preceding h2/h3
            if external_url:
                for prev in heading.find_all_previous(["h2", "h3"]):
                    prev_text = prev.get_text(strip=True)
                    if 10 < len(prev_text) < 80 and prev_text != section_name:
                        subsection_name = prev_text
                        break

            # Extract editorial summary from parent <a> text (minus the title)
            if external_url and parent_link:
                link_text = parent_link.get_text(" ", strip=True)
                desc = link_text.replace(title_text, "", 1).strip()
                # Clean up common suffixes like "READ THE PAPER"
                desc = re.sub(r"\s*READ THE (?:PAPER|REPORT|STUDY|ARTICLE)\s*$", "", desc, flags=re.IGNORECASE)
                if len(desc) > 30:
                    summary = desc[:500]

        # Look for author and date in nearby elements
        parent = heading.parent
        if parent:
            text_content = parent.get_text(" ", strip=True)
            author_match = re.search(r"(?:By|by)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+)", text_content)
            if author_match:
                author = author_match.group(1)

            date_match = re.search(
                r"((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
                text_content,
            )
            if date_match:
                date = date_match.group(1)

        # Extract summary from nearby text
        next_sibling = heading.find_next_sibling("p")
        if next_sibling:
            summary = next_sibling.get_text(strip=True)[:500]

        if not summary and parent:
            paras = parent.find_all("p")
            para_texts = [p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 20]
            if para_texts:
                summary = " ".join(para_texts)[:500]

        if not summary and parent:
            container_text = parent.get_text(" ", strip=True)
            container_text = container_text.replace(title_text, "", 1).strip()
            if len(container_text) > 30:
                summary = container_text[:500]

        final_url = url or external_url
        is_ext = external_url is not None and url is None

        articles.append(
            Article(
                title=title_text,
                section=section_name,
                section_id=section_id,
                author=author,
                date=date,
                summary=summary,
                url=final_url,
                is_feature=is_feature,
                is_external=is_ext,
                subsection=subsection_name,
            )
        )

    # Step 3: Check for any article URLs from the page that weren't matched to headings
    # These are typically news card items with titles in the aria-label map
    matched_urls = {a.url for a in articles if a.url}
    unmatched_urls = [u for u in all_article_urls if u not in matched_urls]
    if unmatched_urls:
        # Reverse the title_url_map to get url->(title, is_feature)
        url_to_info = {url: (title, is_feat) for title, (url, is_feat) in title_url_map.items()}

        for unmatched_url in unmatched_urls:
            # First try the aria-label map (most reliable for clean titles)
            info = url_to_info.get(unmatched_url)
            title = info[0] if info else None
            is_feature = info[1] if info else False

            if not title:
                # Fallback: find the link in the DOM and extract title
                article_path = unmatched_url.replace(BASE_URL, "")
                latest_path = re.sub(r"/issue/\d+/", "/issue/latest/", article_path)
                for path_variant in [article_path, latest_path]:
                    link = soup.find("a", href=path_variant)
                    if link:
                        # Try aria-label first
                        aria = link.get("aria-label", "")
                        if aria:
                            # Strip "Read article: " prefix and category suffix
                            aria_title = re.sub(r"^(?:Read full article|Read article|Continue Reading):\s*", "", aria)
                            title = _strip_category_suffix(_clean_html_entities(aria_title))
                        if not title:
                            link_text = link.get_text(strip=True)
                            if link_text and len(link_text) > 5:
                                title = link_text
                        break

            if title and title not in {a.title for a in articles}:
                articles.append(
                    Article(
                        title=title,
                        section=section_name,
                        section_id=section_id,
                        url=unmatched_url,
                        is_feature=is_feature,
                    )
                )

    # Deduplicate by title, preferring articles with URLs
    seen_titles = set()
    unique_articles = []
    # Sort so articles with URLs come first
    articles.sort(key=lambda a: (a.url is None, a.title))
    for article in articles:
        if article.title not in seen_titles:
            seen_titles.add(article.title)
            unique_articles.append(article)

    return unique_articles


def _extract_from_rsc_payload(data, section_id: str, section_name: str, articles: list):
    """Recursively search RSC payload for article-like content."""
    if isinstance(data, dict):
        # Look for objects that have title-like keys
        if "title" in data or "headline" in data:
            title = data.get("title") or data.get("headline", "")
            if title and len(title) > 5:
                articles.append(
                    Article(
                        title=title,
                        section=section_name,
                        section_id=section_id,
                        author=data.get("author", {}).get("name") if isinstance(data.get("author"), dict) else data.get("author"),
                        summary=data.get("description") or data.get("summary") or data.get("excerpt"),
                        full_text=data.get("content") or data.get("body"),
                    )
                )
        for value in data.values():
            _extract_from_rsc_payload(value, section_id, section_name, articles)
    elif isinstance(data, list):
        for item in data:
            _extract_from_rsc_payload(item, section_id, section_name, articles)


def scrape_latest_issue() -> DocketIssue:
    """
    Scrape the latest issue of The Docket.

    Fetches the main page for metadata, then each section page for articles.
    """
    issue = DocketIssue()

    console.print("[bold cyan]Fetching The Docket latest issue...[/bold cyan]")

    # Fetch main page for metadata
    try:
        main_html = _fetch_page(f"{BASE_URL}/docket/issue/latest/")
        meta = _extract_issue_metadata(main_html)
        issue.issue_number = meta.get("issue_number")
        issue.theme = meta.get("theme")
        console.print(f"  Issue #{issue.issue_number}: {issue.theme[:80] if issue.theme else 'N/A'}...")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fetch main page: {e}[/yellow]")

    # Fetch each section
    for section in SECTIONS:
        section_url = f"{BASE_URL}/docket/issue/latest/{section['id']}"
        console.print(f"  Fetching [{section['name']}]...")

        try:
            html = _fetch_page(section_url)
            issue.raw_sections[section["id"]] = html
            articles = _parse_section_content(html, section["id"], section["name"], issue.issue_number)
            issue.articles.extend(articles)
            console.print(f"    Found {len(articles)} articles")
        except Exception as e:
            console.print(f"  [red]Error fetching {section['name']}: {e}[/red]")

    # Enrich articles that have URLs but no summaries by fetching individual pages
    thin_articles = [a for a in issue.articles if a.url and not a.summary and "/article/" in (a.url or "")]
    if thin_articles:
        console.print(f"\n  Enriching {len(thin_articles)} articles with missing summaries...")
        for article in thin_articles:
            try:
                html = _fetch_page(article.url)
                soup = BeautifulSoup(html, "lxml")

                # Try og:description first (cleanest source)
                og_desc = soup.find("meta", property="og:description")
                if og_desc and og_desc.get("content"):
                    desc = og_desc["content"].strip()
                    if len(desc) > 30 and "weekly guide" not in desc.lower():
                        article.summary = desc[:500]

                # Extract from <p> tags — skip nav boilerplate
                if not article.summary:
                    paras = soup.find_all("p")
                    content_paras = []
                    for p in paras:
                        # Use separator to avoid missing spaces between elements
                        text = _fix_rsc_text(p.get_text(" ", strip=True))
                        # Skip short lines and boilerplate
                        if len(text) < 30:
                            continue
                        if any(skip in text for skip in [
                            "weekly guide to living",
                            "SUBSCRIBE",
                            "Table of Contents",
                            "Human Stories Cascading Effects",
                        ]):
                            continue
                        content_paras.append(text)

                    # Deduplicate identical paragraphs (RSC often renders them twice)
                    seen_paras = set()
                    unique_paras = []
                    for cp in content_paras:
                        if cp not in seen_paras:
                            seen_paras.add(cp)
                            unique_paras.append(cp)

                    if unique_paras:
                        article.summary = " ".join(unique_paras)[:500]

                if article.summary:
                    article.summary = _fix_rsc_text(article.summary)
                    console.print(f"    Enriched: {article.title[:50]}")
            except Exception as e:
                console.print(f"    [dim]Could not enrich {article.title[:40]}: {e}[/dim]")

    # Clean up missing spaces in all summaries (RSC rendering artifact)
    for article in issue.articles:
        if article.summary:
            article.summary = _fix_rsc_text(article.summary)

    # Strip CTA artifacts from summaries and titles (e.g. "CONTINUE READING")
    _cta_artifacts = re.compile(
        r'\s*(CONTINUE READING|Continue Reading|READ MORE|Read More'
        r'|LEARN MORE|Learn More|SEE MORE|See More)\s*$',
        re.IGNORECASE,
    )
    for article in issue.articles:
        if article.summary:
            article.summary = _cta_artifacts.sub("", article.summary).strip()
            # If stripping left only the title repeated, clear it
            if article.summary and article.title and article.summary.strip().lower() == article.title.strip().lower():
                article.summary = None
        if article.title:
            article.title = _cta_artifacts.sub("", article.title).strip()

    console.print(f"\n[bold green]Total articles found: {len(issue.articles)}[/bold green]")
    return issue


if __name__ == "__main__":
    issue = scrape_latest_issue()
    output_path = Path(__file__).parent.parent / "output" / f"issue_{issue.issue_number or 'latest'}.json"
    issue.save(output_path)

    # Print summary
    for section in SECTIONS:
        articles = issue.articles_by_section(section["id"])
        if articles:
            console.print(f"\n[bold]{section['name']}[/bold]")
            for a in articles:
                console.print(f"  - {a.title}")
                if a.author:
                    console.print(f"    by {a.author}")
