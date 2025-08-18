import os
import json
import requests

# Primary: Gemini grounded Google Search via generateContent (not Chat Completions).
# Fallback: Google Programmable Search (CSE).

def gemini_grounded_search(query: str, top_k: int = 5):
    use = os.getenv("GEMINI_USE_GOOGLE_SEARCH", "true").lower() == "true"
    api_key = os.getenv("GEMINI_API_KEY_RES") or os.getenv("GEMINI_API_KEY")
    if not use or api_key is None:
        return None

    # Gemini generateContent with google_search tool (v1beta)
    # Note: This path may change; verify with current Gemini API docs.
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-pro-exp-02-05:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    body = {
        "contents": [{"parts": [{"text": f"Search the web for: {query}"}]}],
        "tools": [{"google_search": {}}],  # request grounded search
    }
    try:
        resp = requests.post(url, params=params, headers=headers, data=json.dumps(body), timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            # Extract snippets if present
            excerpts = []
            # Gemini grounded responses often include citations and snippets in candidates
            # We'll try to pull reasonable text chunks
            for cand in data.get("candidates", []):
                parts = cand.get("content", {}).get("parts", [])
                for p in parts:
                    if "text" in p:
                        excerpts.append(p["text"])
            return excerpts[:top_k]
    except Exception:
        pass
    return None

def google_cse_search(query: str, top_k: int = 5):
    api_key = os.getenv("GOOGLE_CSE_API_KEY")
    cx = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cx:
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx, "q": query, "num": top_k}
    try:
        resp = requests.get(url, params=params, timeout=15)
        items = resp.json().get("items", [])
        return [f"{it.get('title','')} - {it.get('snippet','')}" for it in items]
    except Exception:
        return []

def web_search(query: str, provider: str = "gemini", top_k: int = 5):
    if provider == "gemini":
        res = gemini_grounded_search(query, top_k=top_k)
        if res is not None and len(res) > 0:
            return res
        # fallback
        return google_cse_search(query, top_k=top_k)
    else:
        return google_cse_search(query, top_k=top_k)