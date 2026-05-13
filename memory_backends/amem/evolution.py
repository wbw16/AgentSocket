from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from agent.memory_backends.amem.llm import LLMClient
from agent.memory_backends.amem.note import MemoryNote
from agent.memory_backends.amem.prompts import (
    EVOLUTION_DECISION_PROMPT,
    STRENGTHEN_DETAILS_PROMPT,
    UPDATE_NEIGHBORS_PROMPT,
)

logger = logging.getLogger("amem.evolution")

# ---------------------------------------------------------------------------
# Update 数据结构
# ---------------------------------------------------------------------------

@dataclass
class NoteUpdate:
    """描述一次 evolution 对某个 note 的修改。"""
    note_id: str
    # 要合并进 payload 的字段（只更新非空值）
    patch: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 解析辅助（与 extractor.py 保持独立，避免循环依赖）
# ---------------------------------------------------------------------------

def _extract_section(text: str, marker: str, next_markers: list[str] | None = None) -> str:
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


def _parse_decision(response: str) -> str:
    """解析 DECISION: 行，返回标准化决策字符串。"""
    decision_text = _extract_section(response, "DECISION", ["REASON"]).strip().upper().replace(" ", "_")
    valid = {"NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"}
    if decision_text in valid:
        return decision_text
    # 关键词推断
    upper = response.upper()
    if "STRENGTHEN" in upper and "UPDATE" in upper:
        return "STRENGTHEN_AND_UPDATE"
    if "STRENGTHEN" in upper:
        return "STRENGTHEN"
    if "UPDATE" in upper:
        return "UPDATE_NEIGHBOR"
    return "NO_EVOLUTION"


def _parse_strengthen(response: str) -> tuple[list[int], list[str]]:
    """解析 CONNECTIONS 和 TAGS。"""
    conn_text = _extract_section(response, "CONNECTIONS", ["TAGS"])
    tags_text  = _extract_section(response, "TAGS", ["CONNECTIONS"])
    connections: list[int] = []
    for item in _parse_list_items(conn_text):
        try:
            connections.append(int(item.strip()))
        except (ValueError, TypeError):
            pass
    tags = _parse_list_items(tags_text)
    return connections, tags


def _parse_update_neighbors(response: str, num_neighbors: int) -> list[dict]:
    """解析每个 NEIGHBOR i: 块，返回 [{"context": ..., "tags": [...]}, ...]。"""
    results = []
    for i in range(num_neighbors):
        pattern = re.compile(rf'NEIGHBOR\s+{i}\s*:', re.IGNORECASE)
        match = pattern.search(response)
        if not match:
            results.append({"context": "", "tags": []})
            continue
        next_pat = re.compile(rf'NEIGHBOR\s+{i + 1}\s*:', re.IGNORECASE)
        next_match = next_pat.search(response, match.end())
        block_end = next_match.start() if next_match else len(response)
        block = response[match.end():block_end]
        ctx = _extract_section(block, "CONTEXT", ["TAGS"]).strip()
        tags = _parse_list_items(_extract_section(block, "TAGS", ["CONTEXT"]))
        results.append({"context": ctx, "tags": tags})
    return results


def _format_neighbors(neighbors: list[MemoryNote]) -> str:
    lines = []
    for i, n in enumerate(neighbors):
        lines.append(f"[{i}] content: {n.content}")
        if n.context:
            lines.append(f"    context: {n.context}")
        if n.keywords:
            lines.append(f"    keywords: {', '.join(n.keywords)}")
        if n.tags:
            lines.append(f"    tags: {', '.join(n.tags)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EvolutionPolicy
# ---------------------------------------------------------------------------

class EvolutionPolicy:
    """执行 A-MEM 的三步 evolution：决策 → strengthen → update_neighbors。"""

    def __init__(self, llm: LLMClient, max_retries: int = 2) -> None:
        self._llm = llm
        self._max_retries = max_retries

    def evolve(self, new_note: MemoryNote, neighbors: list[MemoryNote]) -> list[NoteUpdate]:
        """返回需要应用到 store 的 NoteUpdate 列表。失败时返回空列表（不抛出）。"""
        try:
            return self._evolve_inner(new_note, neighbors)
        except Exception as exc:
            logger.error("Evolution failed for note %s: %s — skipping", new_note.id, exc)
            return []

    # ------------------------------------------------------------------

    def _evolve_inner(self, new_note: MemoryNote, neighbors: list[MemoryNote]) -> list[NoteUpdate]:
        neighbor_text = _format_neighbors(neighbors)

        # ---- Call 1: 决策 ----
        decision_prompt = EVOLUTION_DECISION_PROMPT.format(
            context=new_note.context,
            content=new_note.content,
            keywords=", ".join(new_note.keywords),
            nearest_neighbors_memories=neighbor_text,
        )
        decision_resp = self._call(decision_prompt)
        if decision_resp is None:
            return []
        decision = _parse_decision(decision_resp)
        logger.debug("Evolution decision for %s: %s", new_note.id, decision)

        if decision == "NO_EVOLUTION":
            return []

        updates: list[NoteUpdate] = []
        should_strengthen = decision in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
        should_update = decision in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")

        # ---- Call 2: strengthen（更新 new_note 自身的 links/tags）----
        if should_strengthen:
            strengthen_prompt = STRENGTHEN_DETAILS_PROMPT.format(
                content=new_note.content,
                keywords=", ".join(new_note.keywords),
                nearest_neighbors_memories=neighbor_text,
            )
            strengthen_resp = self._call(strengthen_prompt)
            if strengthen_resp is not None:
                conn_indices, new_tags = _parse_strengthen(strengthen_resp)
                # 把邻居 id 加入 new_note.links
                new_links = list(new_note.links)
                for idx in conn_indices:
                    if 0 <= idx < len(neighbors):
                        nid = neighbors[idx].id
                        if nid not in new_links:
                            new_links.append(nid)
                patch: dict = {"links": new_links}
                if new_tags:
                    patch["tags"] = new_tags
                updates.append(NoteUpdate(note_id=new_note.id, patch=patch))

        # ---- Call 3: update_neighbors（改写邻居的 context/tags）----
        if should_update:
            update_prompt = UPDATE_NEIGHBORS_PROMPT.format(
                content=new_note.content,
                context=new_note.context,
                nearest_neighbors_memories=neighbor_text,
                max_neighbor_idx=len(neighbors) - 1,
                neighbor_count=len(neighbors),
            )
            update_resp = self._call(update_prompt)
            if update_resp is not None:
                neighbor_updates = _parse_update_neighbors(update_resp, len(neighbors))
                for i, upd in enumerate(neighbor_updates):
                    if i >= len(neighbors):
                        break
                    patch = {}
                    if upd.get("context"):
                        patch["context"] = upd["context"]
                    if upd.get("tags"):
                        patch["tags"] = upd["tags"]
                    if patch:
                        updates.append(NoteUpdate(note_id=neighbors[i].id, patch=patch))

        return updates

    def _call(self, prompt: str) -> str | None:
        messages = [
            {"role": "system", "content": "Follow the format specified in the prompt exactly."},
            {"role": "user", "content": prompt},
        ]
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._llm.complete(messages, max_tokens=1024, temperature=0.3)
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(1.0 * (2 ** attempt))
                    logger.warning("EvolutionPolicy retry %d/%d: %s", attempt + 1, self._max_retries, exc)
        logger.error("EvolutionPolicy call failed after %d attempts: %s", self._max_retries + 1, last_exc)
        return None
