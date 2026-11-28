#!/usr/bin/env python3
"""FSS public voice-phishing media collector (Python 3.10+).

Install: uv add httpx beautifulsoup4 lxml
Requires ffprobe (FFmpeg). Run: uv run python fss_collector.py --max-pages 1
Full collection: uv run python fss_collector.py
Offline parser check: uv run python fss_collector.py --check-html list.html detail.html
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

BASE = "https://fss.or.kr"
BOARD = BASE + "/fss/bbs/B0000203/list.do"
PARAMS = dict(menuNo="200690", bbsId="", cl1Cd="", sdate="", edate="",
              searchCnd="1", searchWrd="")
MEDIA_EXTS = {".mp4", ".mp3", ".wav", ".m4a", ".wma", ".wmv", ".avi", ".mov", ".webm", ".ogg", ".aac", ".flac", ".mkv"}
LOG = logging.getLogger("fss")


class StopCollection(Exception):
    """Stop on explicit access restrictions instead of bypassing them."""


def official_url(base: str, href: str) -> str:
    url = urljoin(base, href)
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc not in {"fss.or.kr", "www.fss.or.kr", "vod.fss.or.kr"}:
        raise ValueError(f"Unexpected URL: {url}")
    return url


def parse_list(html: str | bytes, page_url: str):
    soup = BeautifulSoup(html, "lxml")
    count = soup.select_one(".total-count em")
    if count is None:
        raise ValueError("Missing total-count; unexpected HTML")
    total = int(count.get_text(strip=True).replace(",", ""))
    pagination = soup.select_one("#pagination-number-input")
    pages = int(pagination["max"]) if pagination else (1 if total <= 8 else 0)
    if pages < 1:
        raise ValueError("Missing pagination")
    posts = []
    for a in soup.select('.krds-bd-list-thumb-a li > a[href*="B0000203/view.do"]'):
        url = official_url(page_url, a["href"])
        post_id = parse_qs(urlsplit(url).query)["nttId"][0]
        if not post_id.isdigit():
            raise ValueError("Invalid post ID")
        title, date = a.select_one(".tit"), a.select_one(".date")
        posts.append(dict(post_id=post_id, title=title.get_text(" ", strip=True) if title else "",
                          published_at=date.get_text(strip=True) if date else None, page_url=url))
    if total and not posts:
        raise ValueError("No posts found despite nonzero total")
    return total, pages, posts


def parse_attachments(html: str | bytes, page_url: str):
    soup = BeautifulSoup(html, "lxml")
    result = {}
    for a in soup.select(".attach-file-box a.btn-attach-download[href]"):
        href = a["href"]
        if not href.startswith(("/", "https://")):
            continue
        url = official_url(page_url, href)
        query = parse_qs(urlsplit(url).query)
        name = re.sub(r"\s*\(파일크기\s*:[^)]*\)\s*$", "", a.get_text(" ", strip=True)).strip()
        ext = Path(name).suffix.lower()
        if ext not in MEDIA_EXTS:
            continue
        # Use the full URL to distinguish attachments and deduplicate repeated links.
        file_sn = query.get("fileSn", [""])[0]
        token = hashlib.sha256(url.encode()).hexdigest()[:12]
        result[url] = dict(download_url=url, original_filename=name, extension=ext,
                           media_id=query.get("mediaId", [None])[0],
                           attachment_id=query.get("atchFileId", [None])[0],
                           file_sn=file_sn, attachment_key=token)
    return list(result.values())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    output = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                             "-of", "json", str(path)], capture_output=True, text=True,
                            check=True, timeout=60)
    info = json.loads(output.stdout)
    streams = info.get("streams", [])
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise ValueError("Media has no audio stream")
    duration = float(info.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise ValueError("Media has no valid duration")
    return {"duration_sec": duration, "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "audio_codec": next(s.get("codec_name") for s in streams if s.get("codec_type") == "audio"),
            "has_video": any(s.get("codec_type") == "video" for s in streams)}


class Collector:
    def __init__(self, args):
        self.args = args
        self.root = args.output
        self.raw = self.root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "metadata.jsonl"
        self.client = httpx.Client(headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"},
                                   timeout=httpx.Timeout(30, connect=10), follow_redirects=True)
        self.last_request = 0.0
        self.records = {}
        if self.manifest.exists():
            for line in self.manifest.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self.records[row["record_id"]] = row
                except (ValueError, KeyError):
                    LOG.warning("Ignoring malformed manifest row")

    def close(self):
        self.client.close()

    def save(self, row):
        row = dict(row, updated_at=datetime.now(timezone.utc).isoformat())
        self.records[row["record_id"]] = row
        # Atomic snapshot, one row per item; safe to resume after interruption.
        temp = self.manifest.with_suffix(".jsonl.tmp")
        with temp.open("w", encoding="utf-8") as f:
            for value in self.records.values():
                f.write(json.dumps(value, ensure_ascii=False) + "\n")
        temp.replace(self.manifest)

    def pace(self):
        time.sleep(max(0, self.args.interval - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()

    @staticmethod
    def check_response(response):
        if response.status_code in (401, 403, 429):
            raise StopCollection(f"HTTP {response.status_code}: {response.url}; stopping collection")
        response.raise_for_status()
        official_url(BASE, str(response.url))

    def retry(self, func):
        for attempt in range(self.args.retries + 1):
            self.pace()
            try:
                return func()
            except StopCollection:
                raise
            except (httpx.HTTPError, ValueError, OSError, subprocess.SubprocessError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500 and exc.response.status_code != 408:
                    raise
                if attempt == self.args.retries:
                    raise
                LOG.warning("Attempt %d failed: %s; retrying", attempt + 1, exc)
                time.sleep(2 ** (attempt + 1))

    def get_html(self, url, referer, params=None):
        def request():
            r = self.client.get(url, params=params, headers={"Referer": referer})
            self.check_response(r)
            return r.content, str(r.url)
        return self.retry(request)

    def collect_posts(self):
        first_html, first_url = self.get_html(BOARD, BASE + "/", dict(PARAMS, pageIndex=1))
        total, pages, first = parse_list(first_html, first_url)
        limit = min(pages, self.args.max_pages or pages)
        LOG.info("Board: %d posts, %d pages; scanning %d pages", total, pages, limit)
        posts = {p["post_id"]: dict(p, list_url=first_url) for p in first}
        for page in range(2, limit + 1):
            html, url = self.get_html(BOARD, first_url, dict(PARAMS, pageIndex=page))
            _, _, found = parse_list(html, url)
            new = set(p["post_id"] for p in found) - posts.keys()
            if not new:
                raise ValueError(f"Page {page} yielded no new posts; check pagination")
            posts.update({p["post_id"]: dict(p, list_url=url) for p in found})
            LOG.info("Page %d/%d: %d unique posts", page, limit, len(posts))
        if limit == pages and len(posts) != total:
            LOG.warning("Board count mismatch: expected %d, found %d", total, len(posts))
        return list(posts.values()), total, limit == pages

    def download(self, post, item):
        record_id = f"{post['post_id']}_{item['attachment_key']}"
        target = self.raw / (record_id + item["extension"])
        part = target.with_suffix(target.suffix + ".part")
        row = dict(post, **item, record_id=record_id, source="FSS", file_path=str(target.relative_to(self.root)))
        previous = self.records.get(record_id, {})
        if target.exists():
            try:
                meta = probe(target)
                if not previous.get("sha256") or previous["sha256"] == meta["sha256"]:
                    self.save(dict(row, **meta, status="ok"))
                    LOG.info("SKIP verified %s", target.name)
                    return True
                LOG.warning("Hash mismatch: %s; redownloading", target.name)
            except (ValueError, OSError, subprocess.SubprocessError):
                LOG.warning("Invalid existing file %s; redownloading", target.name)
        if self.args.dry_run:
            LOG.info("DRY RUN %s -> %s", post["post_id"], item["original_filename"])
            return True

        def request():
            with self.client.stream("GET", item["download_url"],
                                    headers={"Referer": post["page_url"], "Accept-Encoding": "identity"},
                                    timeout=httpx.Timeout(120, connect=10)) as response:
                self.check_response(response)
                if response.status_code != 200:
                    raise ValueError(f"Expected complete HTTP 200, got {response.status_code}")
                content_type = response.headers.get("content-type", "").lower()
                if "text/" in content_type or "json" in content_type:
                    raise ValueError(f"Unexpected content type: {content_type}")
                with part.open("wb") as f:
                    for chunk in response.iter_bytes(1024 * 1024):
                        f.write(chunk)
                length = response.headers.get("content-length")
                if length and part.stat().st_size != int(length):
                    raise ValueError("Content-Length mismatch")
            meta = probe(part)
            part.replace(target)
            return meta

        try:
            meta = self.retry(request)
            self.save(dict(row, **meta, status="ok"))
            LOG.info("OK %s | %.1f s | %d bytes", target.name, meta["duration_sec"], meta["bytes"])
            return True
        except StopCollection:
            raise
        except Exception as exc:
            self.save(dict(row, status="failed", error=str(exc)))
            LOG.error("FAILED %s: %s", target.name, exc)
            return False
        finally:
            part.unlink(missing_ok=True)

    def run(self):
        posts, total, full = self.collect_posts()
        if self.args.limit:
            posts = posts[:self.args.limit]
        ok = failed = missing = attachments = 0
        for idx, post in enumerate(posts, 1):
            LOG.info("[%d/%d] %s %s", idx, len(posts), post["post_id"], post["title"])
            try:
                html, url = self.get_html(post["page_url"], post["list_url"])
                items = parse_attachments(html, url)
            except StopCollection:
                raise
            except Exception as exc:
                failed += 1
                if not self.args.dry_run:
                    self.save(dict(post, record_id="post_" + post["post_id"], status="detail_failed", error=str(exc)))
                LOG.error("Detail failed: %s", exc)
                continue
            if not self.args.dry_run:
                # Clear previous detail error when the detail page becomes available.
                self.records.pop("post_" + post["post_id"], None)
                self.save(dict(post, record_id="post_" + post["post_id"], status="listed" if items else "no_media", attachment_count=len(items)))
            if not items:
                missing += 1
                LOG.warning("No supported media attachment: %s", post["post_id"])
            for item in items:
                attachments += 1
                if self.args.dry_run:
                    LOG.info("DRY RUN %s %s", item["original_filename"], item["download_url"])
                elif self.download(post, item):
                    ok += 1
                else:
                    failed += 1
        # Mark exact byte duplicates without deleting provenance or raw files.
        if not self.args.dry_run:
            hashes = {}
            for key, row in sorted(self.records.items()):
                if row.get("status") == "ok" and row.get("sha256"):
                    duplicate = hashes.setdefault(row["sha256"], key)
                    row["duplicate_of"] = duplicate if duplicate != key else None
            if self.records:
                self.save(next(iter(self.records.values())))
        LOG.info("SUMMARY: board=%d scanned=%d attachments=%d ok=%d failed=%d no_media_posts=%d dry_run=%s",
                 total, len(posts), attachments, ok, failed, missing, self.args.dry_run)
        return 1 if failed or (full and not self.args.limit and len(posts) != total) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("voice_phishing/fss"))
    parser.add_argument("--max-pages", type=int, help="Scan only the first N pages")
    parser.add_argument("--limit", type=int, help="Process only the first N discovered posts")
    parser.add_argument("--interval", type=float, default=1.5, help="Minimum request interval (seconds)")
    parser.add_argument("--retries", type=int, default=2, help="Retries per failed request")
    parser.add_argument("--dry-run", action="store_true", help="List media links without downloading")
    parser.add_argument("--check-html", nargs=2, type=Path, metavar=("LIST", "DETAIL"), help="Offline parser test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.check_html:
        list_path, detail_path = args.check_html
        total, pages, posts = parse_list(list_path.read_bytes(), BOARD)
        items = parse_attachments(detail_path.read_bytes(), posts[0]["page_url"] if posts else BOARD)
        print(json.dumps(dict(total=total, pages=pages, posts=posts, attachments=items), ensure_ascii=False, indent=2))
        return 0
    if args.interval < 1 or args.retries < 0 or (args.limit is not None and args.limit < 1) or (args.max_pages is not None and args.max_pages < 1):
        parser.error("interval must be >= 1; retries >= 0; limits >= 1")
    if not args.dry_run and not shutil.which("ffprobe"):
        parser.error("ffprobe is required; install FFmpeg first")
    collector = Collector(args)
    try:
        return collector.run()
    except KeyboardInterrupt:
        LOG.warning("Interrupted; verified files are preserved")
        return 130
    except (StopCollection, httpx.HTTPError, ValueError, OSError) as exc:
        LOG.error("Collection stopped: %s", exc)
        return 1
    finally:
        collector.close()


if __name__ == "__main__":
    raise SystemExit(main())
