"""
skills.py — 에이전트가 쥐는 능력 단위 모듈 (함수)
------------------------------------------------------------
각 스킬은 단일 책임을 갖는 순수 함수에 가깝다. Agent(agent.py)가
조건에 따라 이들을 순서대로 호출한다. LLM 호출 스킬은 키가 없으면
룰북 기반 템플릿으로 우아하게 대체된다(llm.complete 참조).

confidence 는 LLM 이 추측하지 않고 '룰북 depth'에서 결정론적으로 산출한다.
  - deep 프로세스 통제  : 면접 방어 가능 수준까지 상술됨 → 높은 신뢰도
  - outline 프로세스 통제: 골격 단계(원문 대조 전) → 낮은 신뢰도 → 사람 검토 큐
이렇게 하면 '신뢰도'가 비결정적 LLM 출력이 아니라 도메인 성숙도에 근거한다.
"""
from __future__ import annotations

import llm
import rulebook as rb
from rag import RagIndex, Retrieved
from schemas import GateResult, Process, RiskItem

# 신뢰도 매핑 (룰북 depth 기반, 결정론적)
CONF_DEEP = 0.85
CONF_OUTLINE = 0.50
UNVERIFIED_PENALTY = 0.0   # 근거 미검증은 '배지'로 표시하되 신뢰도 감점은 하지 않음
                            # (검증은 사용자 숙제이지 신뢰도 저하 사유가 아님)

COMPANY_SIZES = ["대기업", "중견기업", "중소기업"]
MODEL_SOURCING = ["자체개발", "외부 API", "벤더 모델"]


# ------------------------------------------------------------
# 스킬 1 — ITGC 게이트 (층 1 선결 조건)
# ------------------------------------------------------------
def check_itgc_gate(answers: dict[str, bool]) -> GateResult:
    """층 1 선결 조건 확인. 하나라도 '아니오'면 실패로 판정한다."""
    gate = rb.itgc_gate()
    failed = [c["id"] for c in gate["checks"] if not answers.get(c["id"], False)]
    if failed:
        return GateResult(passed=False, failed_checks=failed,
                          message=gate["fail_message"].strip())
    return GateResult(passed=True, message="ITGC 선결 조건 충족 — 층 2·3 통제 설계 진행 가능")


# ------------------------------------------------------------
# 스킬 2 — 유의적 계정 식별 (산업 프로필 조회)
# ------------------------------------------------------------
def identify_significant_accounts(industry: str) -> list[Process]:
    """industry_profiles.yaml 조회. not_applicable 제거, priority(critical 우선) 정렬."""
    industries = rb.industries()
    if industry not in industries:
        return []
    levels = rb.significance_levels()
    catalog = rb.process_catalog()
    out: list[Process] = []
    for pid, pd in industries[industry]["profile"].items():
        sig = pd["significance"]
        if sig == "not_applicable":
            continue  # 계정 자체가 부존재 → 선택지에서 제거
        cat = catalog.get(pid, {})
        out.append(Process(
            process_id=pid,
            name=cat.get("name", pid),
            significance=sig,
            priority=levels.get(sig, {}).get("priority", 50),
            rationale=pd.get("rationale", ""),
            standard=cat.get("standard", "-"),
            assertions=cat.get("assertions", []),
            ai_intervention=cat.get("ai_intervention", ""),
        ))
    out.sort(key=lambda p: p.priority)
    return out


# ------------------------------------------------------------
# 스킬 3 — 어서션 리스크 매핑 (통제 원장 risks[] 조회)
# ------------------------------------------------------------
def map_assertion_risk(processes: list[Process]) -> list[RiskItem]:
    lp = rb.ledger_processes()
    items: list[RiskItem] = []
    for p in processes:
        pdata = lp.get(p.process_id)
        if not pdata:
            continue  # 통제 원장 미수록 프로세스 (확장 예정)
        for r in pdata.get("risks", []):
            items.append(RiskItem(
                process_id=p.process_id, process_name=p.name,
                risk_id=r.get("id"), risk=r["risk"],
                assertion_impact=r.get("assertion_impact", "-"),
            ))
    return items


# ------------------------------------------------------------
# 스킬 4 — 통제 검색 (RAG)
# ------------------------------------------------------------
def retrieve_controls(index: RagIndex, processes: list[Process],
                      query: str, k: int = 6) -> list[Retrieved]:
    pids = [p.process_id for p in processes]
    return index.search(query, k=k, process_ids=pids)


# ------------------------------------------------------------
# 스킬 5 — 통제 구체화 (LLM, 룰북 폴백)
# ------------------------------------------------------------
def contextualize_controls(processes: list[Process], company_size: str,
                           model_sourcing: str,
                           force_ids: set[str] | None = None) -> list[dict]:
    """
    선택 프로세스의 통제를 기업 규모·모델 조달 방식에 맞게 구체화한다.
    반환은 dict 후보 목록 — Harness.enforce 가 스키마·근거·신뢰도 게이트를 적용한다.
    """
    force_ids = force_ids or set()
    lp = rb.ledger_processes()
    candidates: list[dict] = []

    for p in processes:
        pdata = lp.get(p.process_id)
        if not pdata:
            continue
        depth = pdata.get("depth", "outline")
        detail = p.significance in ("critical", "significant")  # critical→상술
        for c in pdata.get("controls", []):
            fallback_src = p.standard if p.standard != "-" else "내부회계관리제도"
            ev_list = c.get("evidence") or []
            if ev_list:
                evidences = [{
                    "source": e.get("source") or fallback_src,
                    "paragraph": e.get("paragraph", "TBD"),
                    "verified": bool(e.get("verified", False)),
                } for e in ev_list]
            else:  # outline 통제 등 명시 근거가 없으면 프로세스 기준서로 접지
                evidences = [{"source": fallback_src, "paragraph": "TBD", "verified": False}]
            why = _control_why(pdata, c, p)
            implementation = _control_implementation(c, company_size, model_sourcing, detail)
            candidates.append({
                "control_id": c["id"],
                "process_id": p.process_id,
                "name": c["name"],
                "why": why,
                "implementation": implementation,
                "evidences": evidences,
                "activity_type": c.get("activity_type"),
                "layer": c.get("layer", 2),
                "confidence": CONF_DEEP if depth == "deep" else CONF_OUTLINE,
                "forced": c["id"] in force_ids,
            })
    return candidates


def _control_why(pdata: dict, control: dict, p: Process) -> str:
    """왜 이 통제가 필요한가 — 매칭 리스크/감사인 관점/핵심논지에서 도출."""
    if control.get("auditor_view"):
        base = control["auditor_view"]
    elif pdata.get("risks"):
        base = pdata["risks"][0]["risk"]
    elif pdata.get("core_thesis"):
        base = pdata["core_thesis"].strip().split("\n")[0]
    else:
        base = f"{p.name}에서 AI 개입 시 경영진주장({', '.join(p.assertions) or '-'})이 위협받는다."
    if llm.has_llm():
        out = llm.complete(
            system="너는 재무보고 내부통제 전문가다. 한국어로 한 문장, 간결하게.",
            user=f"통제 '{control['name']}'가 필요한 이유를 아래 근거를 바탕으로 한 문장으로: {base}",
            max_tokens=180,
        )
        if out:
            return out
    return base


def _control_implementation(control: dict, size: str, sourcing: str, detail: bool) -> str:
    """구체적 실행 모습 — 기준서(KSA·K-IFRS) 언어로 기술.

    설계 원칙: 실무 운영 편의('전량 불가라 표본으로 조정' 등)를 서술하지 않는다.
    이 프로젝트의 명제는 'AI는 비결정적이라 벤치마킹(표본 축소) 논리가 붕괴한다'이므로,
    통제의 실행 모습은 기준서 개념(IPE 완전성·정확성, 경영진 편의, 재수행, 파라미터 고정 등)
    으로 기술되어야 한다. 룰북의 design 필드가 그 1차 근거다."""
    design = control.get("design", "")
    note = control.get("design_note", "")
    note_part = f" (비고: {note})" if note else ""
    if llm.has_llm() and detail:
        out = llm.complete(
            system=("너는 감사기준서(KSA)와 K-IFRS를 엄밀히 적용하는 내부통제 설계자다. "
                    "통제의 '구체적 실행 모습'을 한국어 2~3문장으로 기술하되, 실무 운영 편의가 아니라 "
                    "기준서 언어로 쓴다. 가능하면 다음 개념을 정확히 사용한다: AI 산출정보(IPE)의 "
                    "완전성·정확성, 재수행(reperformance), 경영진 편의(Management Bias), 파라미터 고정(Lock), "
                    "temperature 고정, 벤치마킹 논리 붕괴. '전량 불가라 표본으로 조정한다'는 식의 서술은 금지."),
            user=(f"통제명: {control['name']}\n룰북 설계: {design}{note_part}\n"
                  f"맥락(기업규모={size}, 모델조달={sourcing})을 반영하되 기준서 언어로 재기술."),
            max_tokens=260,
        )
        if out:
            return out
    return (design + note_part).strip()


# ------------------------------------------------------------
# 스킬 6 — 감사인 관점 생성 (LLM, 룰북 폴백)
# ------------------------------------------------------------
def generate_auditor_view(cards) -> str:
    """수용된 통제 카드들의 auditor_view 를 종합한다."""
    if not cards:
        return "설계된 통제가 없어 감사인 의존 판단을 생성할 수 없습니다."
    lp = rb.ledger_processes()
    views: list[str] = []
    for card in cards:
        pdata = lp.get(card.process_id, {})
        for c in pdata.get("controls", []):
            if c["id"] == card.control_id and c.get("auditor_view"):
                views.append(f"- {card.name}: {c['auditor_view']}")
                break
    joined = "\n".join(views) if views else "개별 통제의 감사인 관점 데이터가 제한적입니다."
    if llm.has_llm():
        out = llm.complete(
            system="너는 감사 파트너다. 아래 통제별 관점을 종합해 '이 통제들이 유효할 때 "
                   "감사인이 어디까지 통제에 의존하고 실증절차를 줄일 수 있는가'를 한국어 3~4문장으로.",
            user=joined,
            max_tokens=400,
        )
        if out:
            return out
    return ("이 통제들이 설계·운영상 유효하다면, 감사인은 해당 프로세스에 대해 통제 의존 접근을 "
            "취하고 실증절차 표본을 축소할 수 있습니다. 다만 AI 지원 통제는 비결정성 탓에 "
            "'표본 몇 건'이 아니라 '성능이 지속 담보되는가'로 운영유효성을 평가해야 합니다.\n\n"
            + joined)
