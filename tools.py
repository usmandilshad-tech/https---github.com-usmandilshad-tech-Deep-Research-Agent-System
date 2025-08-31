# tools.py
from __future__ import annotations
import os
from typing import Dict, List
from urllib.parse import urlparse, urlencode
import httpx
import trafilatura
from pypdf import PdfReader
from io import BytesIO
from agents import function_tool

# ---- Optional BeautifulSoup; fall back to stdlib parser if lxml missing ----
try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except Exception:
    BeautifulSoup = None
    _HAS_BS4 = False

def _soup(html: str):
    if not _HAS_BS4:
        return None
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")

# ---------- Config ----------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_ENDPOINT = "https://api.tavily.com/search"
SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")
INCLUDE_ANSWER = os.getenv("TAVILY_INCLUDE_ANSWER", "false").lower() in ("1","true","yes")
UA = "DSAS-ResearchBot/1.0 (+https://example.com)"
MAX_FETCH_CHARS = int(os.getenv("MAX_FETCH_CHARS", "8000"))  # hard cap to reduce tokens

# ---------- Helpers ----------
def _looks_like_pdf_bytes(b: bytes) -> bool:
    return b.startswith(b"%PDF-")

def _is_pdf_response(resp: httpx.Response, url: str, content: bytes) -> bool:
    # header OR extension AND confirm magic header bytes
    ct = (resp.headers.get("content-type") or "").lower()
    ext_pdf = urlparse(url).path.lower().endswith(".pdf")
    header_says_pdf = "application/pdf" in ct or ext_pdf
    if header_says_pdf and _looks_like_pdf_bytes(content):
        return True
    return False

def _extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(b), strict=False)
    except Exception:
        return ""  # caller will fall back to HTML path if needed
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
    soup = _soup(html)
    if not soup:
        return html
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)

def _truncate(s: str, cap: int = MAX_FETCH_CHARS) -> str:
    if not s:
        return s
    if len(s) <= cap:
        return s
    return s[:cap] + "\n[...truncated...]"

# ---------- Search fallbacks (Tavily → DDG GET → DDG POST → Wikipedia → static) ----------
def _tavily_search(query: str, k: int, timeout: int = 25) -> List[Dict[str, str]]:
    if not TAVILY_API_KEY:
        return []
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
        with httpx.Client(timeout=timeout, headers={"User-Agent": UA}) as client:
            r = client.post(TAVILY_ENDPOINT, json=payload)
            r.raise_for_status()
            data = r.json()
        out = []
        for item in (data.get("results") or []):
            out.append({
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "snippet": item.get("content") or item.get("snippet") or ""
            })
        return out[:k]
    except Exception:
        return []

def _ddg_get(query: str, k: int, timeout: int = 20) -> List[Dict[str, str]]:
    base = "https://duckduckgo.com/html/"
    url = f"{base}?{urlencode({'q': query})}"
    headers = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.8"}
    try:
        with httpx.Client(timeout=timeout, headers=headers) as client:
            r = client.get(url)
            r.raise_for_status()
            soup = _soup(r.text)
            if not soup:
                return []
            results: List[Dict[str, str]] = []
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
                results.append({"title": title, "url": link, "snippet": snippet})
                if len(results) >= k:
                    break
            return results
    except Exception:
        return []

def _ddg_post_html(query: str, k: int, timeout: int = 20) -> List[Dict[str, str]]:
    base = "https://html.duckduckgo.com/html/"
    headers = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.8"}
    try:
        with httpx.Client(timeout=timeout, headers=headers) as client:
            r = client.post(base, data={"q": query})
            r.raise_for_status()
            soup = _soup(r.text)
            if not soup:
                return []
            results: List[Dict[str, str]] = []
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
                results.append({"title": title, "url": link, "snippet": snippet})
                if len(results) >= k:
                    break
            return results
    except Exception:
        return []

def _wikipedia_opensearch(query: str, k: int, timeout: int = 10) -> List[Dict[str, str]]:
    api = "https://en.wikipedia.org/w/api.php"
    params = {"action": "opensearch", "search": query, "limit": str(k), "namespace": "0", "format": "json"}
    try:
        with httpx.Client(timeout=timeout, headers={"User-Agent": UA}) as client:
            r = client.get(api, params=params)
            r.raise_for_status()
            data = r.json()
        titles = data[1] if len(data) > 1 else []
        descs  = data[2] if len(data) > 2 else []
        urls   = data[3] if len(data) > 3 else []
        out: List[Dict[str, str]] = []
        for t, d, u in zip(titles, descs, urls):
            if u.startswith("http"):
                out.append({"title": t or "", "url": u, "snippet": d or ""})
        return out[:k]
    except Exception:
        return []

def _static_seed(query: str, k: int) -> List[Dict[str, str]]:
    seed = [
        {"title": "Nature Review on Lithium Supply", "url": "https://www.nature.com/articles/s43017-022-00387-5", "snippet": "Peer-reviewed review article."},
        {"title": "NREL Lithium Resources", "url": "https://www.nrel.gov/docs/fy21osti/79178.pdf", "snippet": "US NREL report."},
        {"title": "USGS Lithium Summary", "url": "https://pubs.usgs.gov/periodicals/mcs2024/mcs2024-lithium.pdf", "snippet": "USGS commodity summary."},
        {"title": "World Bank Minerals for Climate Action", "url": "https://documents.worldbank.org/en/publication/documents-reports/documentdetail/906581585998632557/minerals-for-climate-action-the-mineral-intensity-of-the-clean-energy-transition", "snippet": "Macro context on critical minerals."},
    ]
    return seed[:max(1, min(k, len(seed)))]

# ---------- Plain implementations ----------
def web_search_impl(query: str, k: int = 3) -> List[Dict[str, str]]:
    k = max(1, min(int(k or 3), 10))  # smaller by default
    for layer in (
        lambda: _tavily_search(query, k),
        lambda: _ddg_get(query, k),
        lambda: _ddg_post_html(query, k),
        lambda: _wikipedia_opensearch(query, k),
        lambda: _static_seed(query, k),
    ):
        res = layer()
        if res:
            return res
    return _static_seed(query, k)

def fetch_url_impl(url: str) -> str:
    if not url or not url.lower().startswith(("http://", "https://")):
        return ""
    headers = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "en-US,en;q=0.8"}
    try:
        with httpx.Client(follow_redirects=True, timeout=45, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            content = resp.content
            if _is_pdf_response(resp, url, content):
                text = _extract_text_from_pdf_bytes(content)
                if not text:
                    # mis-labeled PDF; try as HTML
                    html = content.decode(resp.encoding or "utf-8", errors="ignore")
                    return _truncate(_extract_text_from_html(html, url))
                return _truncate(text)
            # HTML / other
            html = content.decode(resp.encoding or "utf-8", errors="ignore")
            return _truncate(_extract_text_from_html(html, url))
    except Exception as e:
        return f"[fetch_error] {e}"

def citation_check_impl(claims_markdown: str, urls: List[str]) -> str:
    uniq = [u for u in dict.fromkeys(urls) if u.strip()]
    return f"[check] received {len(uniq)} URLs; deeper verification to follow."

# ---------- Tool wrappers ----------
@function_tool()
def web_search(query: str, k: int = 3) -> List[Dict[str, str]]:
    return web_search_impl(query, k)

@function_tool()
def fetch_url(url: str) -> str:
    return fetch_url_impl(url)

@function_tool()
def citation_check(claims_markdown: str, urls: List[str]) -> str:
    return citation_check_impl(claims_markdown, urls)
