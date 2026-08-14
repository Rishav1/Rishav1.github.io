#!/usr/bin/env python
"""Refresh _data/citations.yml with citation counts from Google Scholar.

Scholar has no API, so this reads the public profile page. The whole
publication list -- title, year, citation count and the citation_for_view id
that _layouts/bib.liquid keys on -- is present in that single page, so one
request is enough. Fetching pages per publication (what the `scholarly`
package does) is what gets CI runners captcha-walled, so this deliberately
does not do it.

Scholar still rate-limits aggressively from datacenter IPs. When a fetch is
blocked the existing citations.yml is left untouched rather than emptied, and
the script only reports failure once the data has gone stale for longer than
STALE_AFTER_DAYS. Since CI runs several times a week, one blocked run is
harmless; a sustained block is what actually needs attention.

Usage:
    python bin/update_scholar_citations.py [--force]
"""

from __future__ import annotations

import gzip
import html
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import yaml
from datetime import datetime, timedelta

OUTPUT_FILE: str = "_data/citations.yml"
SOCIALS_FILE: str = "_data/socials.yml"

PROFILE_URL: str = (
    "https://scholar.google.com/citations?hl=en&user={user}&cstart=0&pagesize=100"
)
MAX_ATTEMPTS: int = 4
# Scholar answers a blocked request instantly, so the pause between attempts
# is what gives the next one a chance rather than the request itself.
RETRY_BASE_DELAY: int = 20
# How long the file may keep its previous contents before a failed run is
# treated as a real outage instead of a transient block.
STALE_AFTER_DAYS: int = 10

USER_AGENTS: tuple[str, ...] = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
)

ROW_MARKER: str = 'class="gsc_a_tr"'
BLOCK_MARKERS: tuple[str, ...] = (
    "/sorry/",
    "unusual traffic",
    "not a robot",
    "captcha",
    "enablejs",
)


class ScholarBlocked(Exception):
    """Scholar served a captcha or rate-limit page instead of the profile."""


def load_scholar_user_id() -> str:
    """Load the Google Scholar user ID from the site configuration."""
    if not os.path.exists(SOCIALS_FILE):
        print(f"Configuration file {SOCIALS_FILE} not found.")
        sys.exit(1)
    try:
        with open(SOCIALS_FILE, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {SOCIALS_FILE}: {e}.")
        sys.exit(1)
    scholar_user_id = (config or {}).get("scholar_userid")
    if not scholar_user_id:
        print(f"No 'scholar_userid' found in {SOCIALS_FILE}.")
        sys.exit(1)
    return str(scholar_user_id).strip()


def fetch_profile(user_id: str) -> str:
    """Fetch the profile page, retrying when Scholar blocks the request."""
    url = PROFILE_URL.format(user=user_id)
    last_error: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            # Full jitter: a fixed retry cadence is itself a bot signal.
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            delay = round(random.uniform(delay / 2, delay))
            print(f"  Retrying in {delay}s ({attempt + 1}/{MAX_ATTEMPTS})...")
            time.sleep(delay)

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENTS[attempt % len(USER_AGENTS)],
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip",
                "Connection": "close",
                "Upgrade-Insecure-Requests": "1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                page = raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            last_error = e
            print(f"  HTTP {e.code} from Scholar.")
            continue
        except Exception as e:
            last_error = e
            print(f"  Request failed: {e}")
            continue

        if ROW_MARKER in page:
            return page

        lowered = page.lower()
        marker = next((m for m in BLOCK_MARKERS if m in lowered), None)
        last_error = ScholarBlocked(marker or "no publication rows in response")
        print(f"  Blocked by Scholar ({last_error}).")

    raise ScholarBlocked(f"giving up after {MAX_ATTEMPTS} attempts: {last_error}")


def parse_profile(page: str, user_id: str) -> dict[str, dict]:
    """Extract one record per publication from the profile page."""
    papers: dict[str, dict] = {}

    for row in page.split(ROW_MARKER)[1:]:
        key = re.search(r"citation_for_view=([\w-]+:[\w-]+)", row)
        title = re.search(r'class="gsc_a_at"[^>]*>(.*?)</a>', row, re.S)
        if not key or not title:
            continue

        citations = re.search(r'class="gsc_a_ac[^"]*"[^>]*>(\d*)<', row)
        year = re.search(r'class="gsc_a_h[^"]*"[^>]*>(\d*)<', row)
        year_text = year.group(1) if year else ""

        papers[html.unescape(key.group(1))] = {
            "title": html.unescape(re.sub(r"<[^>]+>", "", title.group(1))).strip(),
            # Scholar leaves the column blank for unpublished work; keep the
            # placeholder the site has always rendered for those.
            "year": year_text if year_text else "Unknown Year",
            "citations": int(citations.group(1)) if citations and citations.group(1) else 0,
        }

    if not papers:
        raise ScholarBlocked("profile page contained no parseable publications")

    stray = [k for k in papers if not k.startswith(f"{user_id}:")]
    if stray:
        print(f"Warning: {len(stray)} row(s) did not match user {user_id}; keeping anyway.")
    return papers


def load_existing() -> dict | None:
    """Read the current citations file, if it is present and readable."""
    if not os.path.exists(OUTPUT_FILE):
        return None
    try:
        with open(OUTPUT_FILE, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: could not read {OUTPUT_FILE}: {e}.")
        return None


def handle_failure(existing: dict | None, today: str, error: Exception) -> int:
    """Keep the last good data, and decide whether this is worth failing over.

    Returns the process exit code.
    """
    print(f"Could not refresh citations: {error}")

    last_updated = (existing or {}).get("metadata", {}).get("last_updated")
    if not last_updated:
        print("No previous citation data to fall back on.")
        return 1

    try:
        age = datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(
            str(last_updated), "%Y-%m-%d"
        )
    except ValueError:
        print(f"Existing last_updated ({last_updated!r}) is not a valid date.")
        return 1

    print(f"Keeping existing data from {last_updated} ({age.days}d old); file untouched.")
    if age > timedelta(days=STALE_AFTER_DAYS):
        print(
            f"That is over the {STALE_AFTER_DAYS}-day staleness budget - "
            "Scholar has been blocking this job for a while."
        )
        return 1
    print("Within the staleness budget, so treating this run as a transient block.")
    return 0


def main() -> int:
    user_id = load_scholar_user_id()
    today = datetime.now().strftime("%Y-%m-%d")
    force = "--force" in sys.argv

    existing = load_existing()
    last_updated = (existing or {}).get("metadata", {}).get("last_updated")
    if last_updated:
        print(f"Last updated on: {last_updated}")
        if str(last_updated) == today and not force:
            print("Citations data is already up-to-date. Skipping fetch.")
            return 0

    print(f"Fetching citations for Google Scholar ID: {user_id}")
    try:
        papers = parse_profile(fetch_profile(user_id), user_id)
    except (ScholarBlocked, urllib.error.URLError) as e:
        return handle_failure(existing, today, e)

    for key, paper in sorted(papers.items()):
        print(f"Found: {paper['title']} ({paper['year']}) - Citations: {paper['citations']}")

    if existing and existing.get("papers") == papers:
        print(f"No changes in citation data ({len(papers)} publications).")
        return 0

    citation_data = {"metadata": {"last_updated": today}, "papers": papers}
    try:
        with open(OUTPUT_FILE, "w") as f:
            yaml.dump(citation_data, f, width=1000, sort_keys=True)
    except Exception as e:
        print(f"Error writing citation data to {OUTPUT_FILE}: {e}.")
        return 1

    print(f"Citation data for {len(papers)} publications saved to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
