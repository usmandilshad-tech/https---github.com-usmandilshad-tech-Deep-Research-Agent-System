# tools.py
from __future__ import annotations
import os
from typing import Dict, List
from urllib.parse import urlparse
import httpx
import trafilatura
from pypdf import PdfReader
from io import BytesIO
from bs4 import BeautifulSoup
from agents import function_tool

# ---------- Config ----------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_ENDPOINT = "https://api.tavily.com/search"
SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")  # 'basic' | 'advanced'
INCLUDE_ANSWER = os.getenv("TAVILY_INCLUDE_ANSWER", "false").lower() in ("1", "true", "yes")
UA = "DSAS-ResearchBot/1.0 (+https://example.com)"

# ---------- Helpers ----------
def _is_pdf_response(resp: httpx.Response, url: str) -> bool:
    ct = (resp.headers.get("content-type") or "").lower()
    return "application/pdf" in ct or urlparse(url).path.lower().endswith(".pdf")

def _extract_text_from_pdf_bytes(b: bytes) -> str:
    reader = PdfReader(BytesIO(b))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(texts).strip()

def _extract_text_from_html(html: str, url: str) -> str:
    extracted = trafilatura.extract(
        html, url=url, include_formatting=False, include_tables=False, no_fallback=False
    )
    if extracted and extracted.strip():
        return extracted.strip()
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)

def _ddg_html_search(query: str, k: int) -> List[Dict[str, str]]:
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.8"}
    out: List[Dict[str, str]] = []
    try:
        with httpx.Client(timeout=30, headers=headers) as client:
            r = client.post(url, data=params)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            for res in soup.select("div.result"):
                a = res.select_one("a.result__a") or res.select_one("a[href]")
                if not a:
                    continue
                link = a.get("href") or ""
                if not link.startswith("http"):
                    continue
                title = a.get_text(" ", strip=True)
                snippet_el = res.select_one(".result__snippet") or res.select_one(".result__body")
                snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
                out.append({"title": title, "url": link, "snippet": snippet})
                if len(out) >= k:
                    break
    except Exception:
        pass
    return out

# ---------- Plain implementations you can call from Python ----------
def web_search_impl(query: str, k: int = 8) -> List[Dict[str, str]]:
    k = max(1, min(int(k or 8), 20))
    # 1) Try Tavily
    if TAVILY_API_KEY:
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": SEARCH_DEPTH,
            "max_results": k,
            "include_answer": INCLUDE_ANSWER,
            "include_images": False,
            "include_raw_content": False,
        }
        try:
            with httpx.Client(timeout=30, headers={"User-Agent": UA}) as client:
                resp = client.post(TAVILY_ENDPOINT, json=payload)
                resp.raise_for_status()
                data = resp.json()
                results = []
                for r in (data.get("results") or []):
                    results.append({
                        "title": r.get("title") or "",
                        "url": r.get("url") or "",
                        "snippet": r.get("content") or r.get("snippet") or ""
                    })
                if results:
                    return results[:k]
        except Exception:
            pass
    # 2) Fallback: DuckDuckGo HTML
    results = _ddg_html_search(query, k=k)
    if results:
        return results
    # 3) Last resort diagnostic stub
    return [{
        "title": "Search failed",
        "url": "https://html.duckduckgo.com/html/",
        "snippet": "No results returned. Check network/auth or try again."
    }]

def fetch_url_impl(url: str) -> str:
    if not url or not url.lower().startswith(("http://", "https://")):
        return ""
    headers = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.8"}
    try:
        with httpx.Client(follow_redirects=True, timeout=45, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            if _is_pdf_response(resp, url):
                try:
                    return _extract_text_from_pdf_bytes(resp.content)
                except Exception:
                    return "[PDF detected but text extraction failed]"
            html = resp.content.decode(resp.encoding or "utf-8", errors="ignore")
            return _extract_text_from_html(html, url)
    except Exception as e:
        return f"[fetch_error] {e}"

def citation_check_impl(claims_markdown: str, urls: List[str]) -> str:
    uniq = [u for u in dict.fromkeys(urls) if u.strip()]
    return f"[check] received {len(uniq)} URLs; deeper verification to follow."

# ---------- Tool wrappers the Agents can call ----------
@function_tool()
def web_search(query: str, k: int = 8) -> List[Dict[str, str]]:
    return web_search_impl(query, k)

@function_tool()
def fetch_url(url: str) -> str:
    return fetch_url_impl(url)

@function_tool()
def citation_check(claims_markdown: str, urls: List[str]) -> str:
    return citation_check_impl(claims_markdown, urls)
