"""
schemas.py — Harness 통제 #1: 출력 스키마 검증 (pydantic)
------------------------------------------------------------
이 파일은 에이전트가 만들어내는 모든 구조화된 출력의 계약(contract)이다.
LLM 이 반환한 자유 텍스트는 반드시 이 스키마를 통과해야만 화면에 노출된다.
근거(evidences)가 비어 있으면 ControlCard 자체가 성립하지 않도록 강제한다.
→ 이것이 "할루시네이션을 구조로 막는다"는 자기참조 설계의 첫 번째 구현이다.

하나의 통제는 여러 근거 조항을 가질 수 있다(예: aipc_01 = 감사기준서 330 + 1100).
따라서 근거는 단수가 아니라 Evidence 의 리스트(evidences)로 담는다.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class GateResult(BaseModel):
    """ITGC 게이트(층 1) 판정 결과."""
    passed: bool
    failed_checks: list[str] = Field(default_factory=list)
    message: str = ""


class Process(BaseModel):
    """산업 프로필에서 식별된, AI 통제 설계 대상 프로세스."""
    process_id: str
    name: str
    significance: str            # critical / significant / routine / low
    priority: int
    rationale: str
    standard: str = "-"
    assertions: list[str] = Field(default_factory=list)
    ai_intervention: str = ""


class RiskItem(BaseModel):
    """프로세스별 어서션 리스크 (control_ledger.risks[])."""
    process_id: str
    process_name: str
    risk_id: Optional[str] = None
    risk: str
    assertion_impact: str


class Evidence(BaseModel):
    """통제의 근거 조항 하나. source 는 필수(빈 값이면 검증 실패)."""
    source: str
    paragraph: str = "TBD"
    verified: bool = False

    @field_validator("source")
    @classmethod
    def _source_required(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("evidence.source 누락 — 근거 없는 통제는 출력 불가")
        return v.strip()


class ControlCard(BaseModel):
    """
    통제 스타터킷의 한 장(카드).

    Harness 규칙:
      - evidences 는 최소 1개 이상. 비면 카드 자체가 폐기된다.
      - layer 는 1/2/3 만 허용.
      - confidence 가 임계 미만이면 Agent 가 '사람 검토 큐'로 격리한다.
    """
    control_id: str
    process_id: str
    name: str
    why: str                     # 왜 이 통제가 필요한가
    implementation: str          # 구체적 실행 모습 (기준서 언어)
    evidences: list[Evidence]    # 근거 조항들 — 필수(최소 1)
    layer: Literal[1, 2, 3]
    confidence: float = 1.0
    forced: bool = False         # 비결정성 통제 등 Agent 가 강제 포함했는지
    activity_type: str | None = None  # 보론3-20 통제활동 유형 (권한부여·승인/대사/검증/물리적·논리적/업무분장)

    @field_validator("evidences")
    @classmethod
    def _at_least_one(cls, v: list) -> list:
        if not v:
            raise ValueError("근거 없는 통제는 출력 불가")
        return v

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @property
    def verified_all(self) -> bool:
        return all(e.verified for e in self.evidences)
