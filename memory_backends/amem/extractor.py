from __future__ import annotations

import logging
import re
import time

from agent.memory_backends.amem.llm import LLMClient
from agent.memory_backends.amem.prompts import ANALYZE_CONTENT_PROMPT

logger = logging.getLogger("amem.extractor")

# ---------------------------------------------------------------------------
# section-marker 解析辅助
# ---------------------------------------------------------------------------

def _extract_section(text: str, marker: str, next_markers: list[str] | None = None) -> str:
    """提取 marker: 到下一个 marker 之间的文本。"""
    pattern = re.compile(rf'^\s*{re.escape(marker)}\s*:\s*(.*)$', re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()
    first_line = match.group(1).strip()
    end = len(text)
    if next_markers:
        for nm in next_markers:
            nm_pat = re.compile(rf'^\s*{re.escape(nm)}\s*:', re.IGNORECASE | re.MULTILINE)
            nm_match = nm_pat.search(text, start)
            if nm_match and nm_match.start() < end:
                end = nm_match.start()
    rest = text[start:end].strip()
    if first_line and rest:
        return first_line + "\n" + rest
    return first_line or rest


def _parse_list_items(text: str) -> list[str]:
    """将文本解析为列表项（支持逗号分隔、bullet、每行一项）。"""
    if not text or not text.strip():
        return []
    items: list[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\-\*•]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        if ',' in line:
            for part in line.split(','):
                part = part.strip().strip('"').strip("'").strip()
                if part:
                    items.append(part)
        else:
            items.append(line)
    return items


def _heuristic_keywords(content: str, max_kw: int = 5) -> list[str]:
    """从内容中启发式提取关键词（无 LLM fallback）。"""
    stop = {
        'the','a','an','is','are','was','were','be','been','being','have','has',
        'had','do','does','did','will','would','could','should','may','might',
        'to','of','in','for','on','with','at','by','from','as','into','and',
        'or','but','if','it','its','i','me','my','you','your','he','she','they',
        'we','this','that','these','those','what','which','who','not','just',
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
    scored: list[tuple[str, int]] = []
    seen: set[str] = set()
    for w in words:
        wl = w.lower()
        if wl in stop or wl in seen:
            continue
        seen.add(wl)
        scored.append((wl, 2 if w[0].isupper() else 1))
    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:max_kw]]


def _heuristic_context(content: str) -> str:
    m = re.match(r'(.+?[.!?])\s', content)
    if m:
        return m.group(1).strip()
    return content[:200].strip()


# ---------------------------------------------------------------------------
# MetadataExtractor
# ---------------------------------------------------------------------------

class MetadataExtractor:
    """调用 LLM 提取 keywords / context / tags，解析失败时 graceful fallback。"""

    def __init__(self, llm: LLMClient, max_retries: int = 2) -> None:
        self._llm = llm
        self._max_retries = max_retries

    def extract(self, content: str) -> dict:
        """返回 {"keywords": [...], "context": "...", "tags": [...]}。
        任何异常都 fallback 到启发式结果，不抛出。
        """
        raw = self._call_with_retry(content)
        if raw is None:
            return self._fallback(content)
        return self._parse(raw, content)

    # ------------------------------------------------------------------

    def _call_with_retry(self, content: str) -> str | None:
        prompt = ANALYZE_CONTENT_PROMPT.format(content=content)
        messages = [
            {"role": "system", "content": "Follow the format specified in the prompt exactly."},
            {"role": "user", "content": prompt},
        ]
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._llm.complete(messages, max_tokens=512, temperature=0.3)
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(1.0 * (2 ** attempt))
                    logger.warning("MetadataExtractor retry %d/%d: %s", attempt + 1, self._max_retries, exc)
        logger.error("MetadataExtractor failed after %d attempts: %s", self._max_retries + 1, last_exc)
        return None

    def _parse(self, response: str, content: str) -> dict:
        keywords_text = _extract_section(response, "KEYWORDS", ["CONTEXT", "TAGS"])
        context_text  = _extract_section(response, "CONTEXT",  ["TAGS", "KEYWORDS"])
        tags_text     = _extract_section(response, "TAGS",     ["KEYWORDS", "CONTEXT"])

        keywords = _parse_list_items(keywords_text)
        context  = context_text.strip()
        tags     = _parse_list_items(tags_text)

        # 修复空字段
        if not keywords:
            keywords = _heuristic_keywords(content)
        if not context:
            context = _heuristic_context(content)
        if not tags:
            tags = keywords[:3]

        return {"keywords": keywords, "context": context, "tags": tags}

    def _fallback(self, content: str) -> dict:
        kw = _heuristic_keywords(content)
        return {
            "keywords": kw,
            "context": _heuristic_context(content),
            "tags": kw[:3],
        }
