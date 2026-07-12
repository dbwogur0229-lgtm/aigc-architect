"""
agent.py — 스킬을 조건에 따라 순서 판단하며 호출하는 주체
------------------------------------------------------------
Agent 는 단순 파이프라인이 아니다. 입력에 따라 '조건 분기'가 살아있어야
도구가 실제로 '도는 것'이 증명된다. 핵심 분기:

  · ITGC 게이트 실패 & 강행 아님 → 하위 스킬 전면 중단, 경고 반환
  · ai_performed_control 선택   → 비결정성 통제(aipc_01, aipc_02) 강제 포함
                                   + '비결정성·출력 일관성' 추가 RAG 검색
  · critical 프로세스           → 통제 상술 / routine → 요약 (skills.contextualize)
  · 산업별 프로세스 목록이 실제로 달라짐 (industry_profiles.yaml 분기)

run_agent() 는 제너레이터다. 각 스텝마다 진행 이벤트를 yield 하여
UI 가 "[k/5] ..." 를 실시간 표시하게 한다(에이전트임을 시각적으로 증명).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import skills
from harness import HarnessResult, RunLog, enforce, timed
from rag import RagIndex
from schemas import ControlCard, GateResult, Process, RiskItem


@dataclass
class AgentInputs:
    industry: str
    selected_processes: list[str]
    itgc_answers: dict[str, bool]
    override_itgc: bool = False
    company_size: str = "중견기업"
    model_sourcing: str = "외부 API"


@dataclass
class AgentOutput:
    gate: GateResult | None = None
    processes: list[Process] = field(default_factory=list)
    risks: list[RiskItem] = field(default_factory=list)
    harness: HarnessResult = field(default_factory=HarnessResult)
    auditor_view: str = ""
    log: RunLog = field(default_factory=RunLog)
    stopped_reason: str = ""


# 비결정성 통제 — ai_performed_control 선택 시 강제 포함
FORCED_NONDETERMINISM = {"aipc_01", "aipc_02"}


def run_agent(inputs: AgentInputs, index: RagIndex):
    """제너레이터: ('step', 라벨) 이벤트들을 yield 하고, 마지막에 ('done', AgentOutput)."""
    out = AgentOutput()
    log = out.log

    # ── [1/5] ITGC 선결 조건 확인 ──────────────────────────────
    yield ("step", "[1/5] ITGC 선결 조건 확인 중...")
    with timed() as t:
        gate = skills.check_itgc_gate(inputs.itgc_answers)
    log.record("check_itgc_gate", f"answers={inputs.itgc_answers}", [],
               f"passed={gate.passed}, failed={gate.failed_checks}", t.ms)
    out.gate = gate

    if not gate.passed and not inputs.override_itgc:
        # 분기: 게이트 실패 → 하위 스킬 전면 중단
        out.stopped_reason = ("ITGC 전제 조건 미충족 — 층 1이 무너지면 층 2·3 통제 설계가 "
                              "무의미하므로 이후 단계를 중단했습니다.")
        log.note("[게이트 차단] ITGC 미비 → 이후 단계 중단")
        yield ("done", out)
        return
    if not gate.passed and inputs.override_itgc:
        log.note("[게이트 강행] ITGC 미비 상태를 사용자가 인지하고 진행 — 결과는 조건부")

    # ── [2/5] 유의적 계정 식별 ────────────────────────────────
    yield ("step", "[2/5] 유의적 계정 식별 중...")
    with timed() as t:
        all_procs = skills.identify_significant_accounts(inputs.industry)
        procs = [p for p in all_procs if p.process_id in inputs.selected_processes]
    stars = [p.name for p in procs if p.significance == "critical"]
    log.record("identify_significant_accounts", f"industry={inputs.industry}", [],
               f"선택 {len(procs)}개, critical={stars}", t.ms)
    out.processes = procs

    # ── [3/5] 어서션 리스크 매핑 ──────────────────────────────
    yield ("step", "[3/5] 경영진주장 리스크 매핑 중...")
    with timed() as t:
        risks = skills.map_assertion_risk(procs)
    log.record("map_assertion_risk", f"processes={[p.process_id for p in procs]}", [],
               f"리스크 {len(risks)}건", t.ms)
    out.risks = risks

    # ── [4/5] 통제 검색·구체화 (RAG + Harness) ────────────────
    yield ("step", "[4/5] 관련 기준·통제 검색 중... (RAG · 검색 증강)")
    # 분기: ai_performed_control 선택 시 비결정성 추가 검색
    force_ids: set[str] = set()
    if "ai_performed_control" in inputs.selected_processes:
        force_ids |= FORCED_NONDETERMINISM
        log.note("[분기] AI 지원 통제 선택 → 비결정성 통제 aipc_01·aipc_02 강제 포함")

    # 프로세스별로 검색해 각 프로세스의 근거가 grounded 집합에 반드시 포함되게 한다.
    # (단일 혼합 쿼리 top-k 는 특정 프로세스 근거를 누락할 수 있음)
    with timed() as t:
        retrieved = []
        seen: set[str] = set()
        for p in procs:
            for r in index.search(f"{p.name} 통제 근거 경영진주장", k=3,
                                  process_ids=[p.process_id]):
                if r.chunk.id not in seen:
                    seen.add(r.chunk.id)
                    retrieved.append(r)
        if force_ids:  # 비결정성 추가 검색
            for r in index.search("비결정성 출력 일관성 성능지표 운영유효성", k=3,
                                  process_ids=["ai_performed_control"]):
                if r.chunk.id not in seen:
                    seen.add(r.chunk.id)
                    retrieved.append(r)
        # 접지 기준: 검색 대상 프로세스들의 지식 범위 전체 (top-k 절단 방지)
        grounded = index.grounded_sources_for([p.process_id for p in procs])
    log.record("retrieve_controls",
               f"프로세스별 검색 {len(procs)}건" + (" +비결정성" if force_ids else ""),
               [r.chunk.id for r in retrieved],
               f"근거출처 {len(grounded)}종 확보", t.ms)

    with timed() as t:
        candidates = skills.contextualize_controls(
            procs, inputs.company_size, inputs.model_sourcing, force_ids=force_ids)
    log.record("contextualize_controls",
               f"size={inputs.company_size}, sourcing={inputs.model_sourcing}",
               [r.chunk.id for r in retrieved],
               f"후보 통제 {len(candidates)}개 생성", t.ms)

    # Harness 3중 게이트: 스키마 → 근거 인용 → 사람 검토
    with timed() as t:
        result = enforce(candidates, grounded, log)
    log.record("harness.enforce", f"후보 {len(candidates)}개",
               [r.chunk.id for r in retrieved],
               f"수용 {len(result.accepted)} / 격리 {len(result.quarantined)} "
               f"/ 폐기 {len(result.rejected)}", t.ms)
    out.harness = result

    # ── [5/5] 감사인 관점 생성 ────────────────────────────────
    yield ("step", "[5/5] 감사인 관점 생성 중...")
    with timed() as t:
        view = skills.generate_auditor_view(result.accepted)
    log.record("generate_auditor_view", f"수용통제 {len(result.accepted)}개", [],
               f"{len(view)}자 생성", t.ms)
    out.auditor_view = view

    yield ("done", out)
