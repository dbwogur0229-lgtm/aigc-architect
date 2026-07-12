"""
harness.py — 에이전트를 실행·감시·제어하는 틀
============================================================
★ 자기참조 설계: 이 Harness 는 이 도구가 '설계하는' AIGC 통제를
   '스스로에게' 적용한 것이다.

  AIGC 통제              →  Harness 구현
  ─────────────────────────────────────────────────────────
  설명가능성·근거 추적   →  근거 조항 인용 강제 (검색된 청크에 없는 인용 차단)
  Human-in-the-loop      →  신뢰도 낮은 판정은 확정 않고 격리
  감사추적               →  실행 로그 (스킬 호출 순서·검색 문서·프롬프트 전량 기록)
  할루시네이션 통제      →  출력 스키마 검증(pydantic), 미검증 근거는 배지 표시
  비결정성 관리          →  temperature 고정 (llm.py), 출력 일관성 검증
============================================================
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from schemas import ControlCard

# 사람 검토 게이트 임계값. 이 값 미만의 confidence 는 확정하지 않고 격리한다.
CONFIDENCE_THRESHOLD = 0.60
# 스텝 상한 — 무한루프 방지 (Harness 통제 #5)
MAX_STEPS = 12


@dataclass
class LogEntry:
    step: int
    skill: str
    inputs: str
    retrieved_ids: list[str]
    output_summary: str
    elapsed_ms: int


class RunLog:
    """Harness 통제 #4 — 감사추적. 모든 스킬 호출을 순서대로 기록."""

    def __init__(self) -> None:
        self.entries: list[LogEntry] = []
        self.notes: list[str] = []       # 근거 미확인 등 통제 이벤트
        self._step = 0

    def record(self, skill: str, inputs: str, retrieved_ids: list[str],
               output_summary: str, elapsed_ms: int) -> None:
        self._step += 1
        if self._step > MAX_STEPS:
            raise RuntimeError(f"스텝 상한({MAX_STEPS}) 초과 — 무한루프 방지 차단")
        self.entries.append(LogEntry(
            step=self._step, skill=skill, inputs=inputs,
            retrieved_ids=retrieved_ids, output_summary=output_summary,
            elapsed_ms=elapsed_ms,
        ))

    def note(self, msg: str) -> None:
        self.notes.append(msg)


class timed:
    """with timed() as t: ... ; t.ms 로 소요시간(ms)."""
    def __enter__(self):
        self._t = time.perf_counter()
        self.ms = 0
        return self

    def __exit__(self, *a):
        self.ms = int((time.perf_counter() - self._t) * 1000)


@dataclass
class HarnessResult:
    accepted: list[ControlCard] = field(default_factory=list)   # 정상 통제 카드
    quarantined: list[ControlCard] = field(default_factory=list)  # 사람 검토 필요
    rejected: list[str] = field(default_factory=list)           # 스키마/근거 실패 폐기


def enforce(candidates: list[dict], allowed_sources: set[str],
            log: RunLog) -> HarnessResult:
    """
    후보 통제(dict) 목록을 Harness 3중 게이트에 통과시킨다.

      게이트 1 — 스키마 검증 (pydantic): evidence_source 누락 등은 폐기
      게이트 2 — 근거 인용 강제: evidence_source 가 '검색된 청크'의 출처 집합에
                 없으면 폐기 + '근거 미확인' 로그
      게이트 3 — 사람 검토: confidence < 임계 → 확정 않고 격리
    """
    result = HarnessResult()
    for cand in candidates:
        cid = cand.get("control_id", "?")
        # 게이트 1: 스키마
        try:
            card = ControlCard(**cand)
        except ValidationError as e:
            msg = f"[스키마 폐기] {cid}: {e.errors()[0]['msg'] if e.errors() else e}"
            result.rejected.append(msg)
            log.note(msg)
            continue

        # 게이트 2: 근거 인용 강제 — 근거 중 하나라도 접지되면 통과
        srcs = [e.source for e in card.evidences]
        if allowed_sources and not any(s in allowed_sources for s in srcs):
            msg = (f"[근거 미확인] {cid}: 인용 {srcs} 이(가) "
                   f"검색된 지식 범위에 없음 → 출력 제외")
            result.rejected.append(msg)
            log.note(msg)
            continue

        # 게이트 3: 사람 검토 게이트
        if card.confidence < CONFIDENCE_THRESHOLD:
            result.quarantined.append(card)
            log.note(f"[격리] {cid}: confidence={card.confidence:.2f} "
                     f"< {CONFIDENCE_THRESHOLD} → 사람 검토 큐")
        else:
            result.accepted.append(card)
    return result
