"""
rulebook.py — 룰북(도메인 지식) 로딩 및 청킹
------------------------------------------------------------
industry_profiles.yaml + control_ledger.yaml 을 읽어 메모리에 올린다.
룰북은 도구의 '분기 로직'이자 LLM 에 주입되는 지식의 본체다.

또한 RAG 를 위한 '청크'를 생성한다. 각 청크는 문단 단위이며
{id, text, meta} 구조를 가진다. corpus/ 폴더의 .md/.txt 원문도 함께 청킹한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RULEBOOK_DIR = os.path.join(_ROOT, "rulebook")
CORPUS_DIR = os.path.join(_ROOT, "corpus")


@dataclass
class Chunk:
    id: str
    text: str
    meta: dict = field(default_factory=dict)


@lru_cache(maxsize=1)
def load() -> dict:
    """두 YAML 을 읽어 하나의 dict 로 반환 (캐시)."""
    with open(os.path.join(RULEBOOK_DIR, "industry_profiles.yaml"), encoding="utf-8") as f:
        profiles = yaml.safe_load(f)
    with open(os.path.join(RULEBOOK_DIR, "control_ledger.yaml"), encoding="utf-8") as f:
        ledger = yaml.safe_load(f)
    return {"profiles": profiles, "ledger": ledger}


def industries() -> dict:
    return load()["profiles"]["industries"]


def significance_levels() -> dict:
    return load()["profiles"]["significance_levels"]


def process_catalog() -> dict:
    return load()["profiles"]["processes"]


def itgc_gate() -> dict:
    return load()["ledger"]["itgc_gate"]


def ledger_processes() -> dict:
    return load()["ledger"]["processes"]


# ------------------------------------------------------------
# RAG 청킹 — 룰북의 통제/리스크/근거를 문단 단위 청크로 변환
# ------------------------------------------------------------
@lru_cache(maxsize=1)
def build_chunks() -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    lp = ledger_processes()
    cat = process_catalog()

    for pid, pdata in lp.items():
        pname = pdata.get("name", pid)
        std = pdata.get("standard", cat.get(pid, {}).get("standard", "-"))

        # 프로세스 핵심 논지 (deep 프로세스만 존재)
        for key in ("core_thesis", "why_it_breaks"):
            if pdata.get(key):
                chunks.append(Chunk(
                    id=f"{pid}::{key}",
                    text=f"[{pname}] {pdata[key].strip()}",
                    meta={"process_id": pid, "kind": key, "standard": std},
                ))

        # 리스크
        for i, r in enumerate(pdata.get("risks", [])):
            chunks.append(Chunk(
                id=f"{pid}::risk::{r.get('id', i)}",
                text=f"[{pname} 리스크] {r['risk']} (영향받는 경영진주장: {r.get('assertion_impact','-')})",
                meta={"process_id": pid, "kind": "risk", "standard": std},
            ))

        # 통제 (근거 조항 포함) — 검색·인용의 핵심 대상
        for c in pdata.get("controls", []):
            ev = c.get("evidence", []) or []
            ev_txt = "; ".join(
                f"{e.get('source','')}({e.get('paragraph','TBD')}, "
                f"{'검증완료' if e.get('verified') else '원문대조전'})"
                for e in ev
            )
            body = (
                f"[{pname} 통제 {c['id']}] {c['name']}. "
                f"설계: {c.get('design','')}. "
                f"{('비고: ' + c['design_note'] + '. ') if c.get('design_note') else ''}"
                f"근거: {ev_txt or '없음'}. "
                f"{('감사인 관점: ' + c['auditor_view']) if c.get('auditor_view') else ''}"
            )
            chunks.append(Chunk(
                id=f"{pid}::control::{c['id']}",
                text=body,
                meta={
                    "process_id": pid, "kind": "control", "standard": std,
                    "control_id": c["id"], "layer": c.get("layer"),
                    "evidence": ev,
                },
            ))

    # corpus/ 원문 문서 (감사기준서 발췌 등) — 있으면 문단 단위로 청킹
    if os.path.isdir(CORPUS_DIR):
        for fn in sorted(os.listdir(CORPUS_DIR)):
            if not fn.lower().endswith((".md", ".txt")):
                continue
            path = os.path.join(CORPUS_DIR, fn)
            with open(path, encoding="utf-8") as f:
                raw = f.read()
            for j, para in enumerate(_split_paragraphs(raw)):
                chunks.append(Chunk(
                    id=f"corpus::{fn}::{j}",
                    text=para,
                    meta={"process_id": None, "kind": "corpus", "source_file": fn},
                ))

    return tuple(chunks)


def _split_paragraphs(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split("\n\n")]
    return [p for p in parts if len(p) > 20 and not p.startswith("#")]
