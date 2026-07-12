"""
rag.py — 검색 증강 생성 (Retrieval-Augmented Generation)
------------------------------------------------------------
사용자의 첫 RAG 구현. 원리를 주석으로 설명한다.

파이프라인 (벡터DB 미사용 — 코퍼스가 작아 numpy 코사인으로 충분):
  1. 룰북 YAML + corpus/ 문서를 문단 단위 청크로 분할        (rulebook.build_chunks)
  2. 각 청크를 임베딩하여 (n, d) 행렬을 만든다               (Embedder.fit)
  3. 질문도 같은 공간으로 임베딩한다                          (Embedder.transform)
  4. 질문 벡터 vs 모든 청크의 코사인 유사도를 계산 (numpy 내적) 후 상위 k 반환

왜 벡터DB 가 없어도 되는가:
  임베딩을 L2 정규화하면 코사인 유사도 = 내적이다. 청크가 수백 개 규모라면
  (질문벡터) · (청크행렬.T) 한 번의 행렬곱으로 전수 계산이 끝난다. 인덱스 자료구조
  (HNSW 등)의 이득은 수십만 벡터 이상에서 나타난다. → README 설계 근거 참조.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm import make_embedder
from rulebook import Chunk, build_chunks


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


class RagIndex:
    """룰북+코퍼스 청크에 대한 인메모리 코사인 검색 인덱스."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = list(build_chunks())
        self.embedder = make_embedder()
        # 2) 전체 청크 임베딩 (정규화 완료 행렬)
        self.matrix: np.ndarray = self.embedder.fit([c.text for c in self.chunks])

    def search(self, query: str, k: int = 5,
               process_ids: list[str] | None = None) -> list[Retrieved]:
        """3~4) 질의 임베딩 → 코사인 유사도 → 상위 k."""
        qv = self.embedder.transform([query])[0]        # (d,)
        sims = self.matrix @ qv                          # 정규화됐으므로 내적 = 코사인
        order = np.argsort(-sims)

        results: list[Retrieved] = []
        for idx in order:
            ch = self.chunks[idx]
            if process_ids and ch.meta.get("process_id") not in process_ids \
                    and ch.meta.get("kind") != "corpus":
                continue  # 선택 프로세스로 우선 필터 (corpus 는 항상 허용)
            results.append(Retrieved(chunk=ch, score=float(sims[idx])))
            if len(results) >= k:
                break
        return results

    def grounded_sources_for(self, process_ids: list[str]) -> set[str]:
        """선택된 프로세스들의 지식(룰북 청크)이 담고 있는 근거 출처 전체 집합.
        Harness 통제 #2 는 이 집합에 인용을 대조한다. 여기 없는 인용
        (다른 프로세스 근거 또는 LLM 이 지어낸 조항)은 차단된다.
        top-k 절단으로 특정 통제 근거가 누락되는 것을 방지하기 위해
        '검색 대상 프로세스의 지식 범위' 전체를 접지 기준으로 삼는다."""
        srcs: set[str] = {"내부회계관리제도"}
        pset = set(process_ids)
        for ch in self.chunks:
            if ch.meta.get("process_id") not in pset:
                continue
            std = ch.meta.get("standard")
            if std and std != "-":
                srcs.add(std)
            for e in ch.meta.get("evidence", []) or []:
                if e.get("source"):
                    srcs.add(e["source"])
        return srcs

    def grounded_sources(self, retrieved: list[Retrieved]) -> set[str]:
        """검색된 청크들이 '근거로 인정할 수 있는' 출처 집합.
        Harness 통제 #2(근거 인용 강제)가 이 집합에 대조한다.
          - 청크에 명시된 evidence.source (예: '감사기준서 540')
          - 청크가 매핑된 기준서 standard (예: 'K-IFRS 1113')
          - 내부회계관리제도(ICFR) — 도구 전체가 그 위에서 동작하는 기반 프레임워크
        검색된 청크에 근거가 전혀 걸리지 않은 인용은 이 집합 밖 → Harness 가 차단한다."""
        srcs: set[str] = {"내부회계관리제도"}
        for r in retrieved:
            std = r.chunk.meta.get("standard")
            if std and std != "-":
                srcs.add(std)
            for e in r.chunk.meta.get("evidence", []) or []:
                if e.get("source"):
                    srcs.add(e["source"])
        return srcs
