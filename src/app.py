"""
app.py — AIGC Architect Streamlit UI
------------------------------------------------------------
화면 흐름: [1] 입력(2단) → [2] 스텝별 실행(Agent 증명) → [3] 탭 출력
톤앤매너: 삼일회계법인(PwC) 오렌지 포인트 + 차분한 네이비/그레이 베이스.
통제 설계안은 3층 구조(층2 AIGC → 층3 프로세스)를 시각적으로 위계화한다.
"""
from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

import llm
import export as xl
import rulebook as rb
import skills
from agent import AgentInputs, run_agent
from rag import RagIndex
from skills import COMPANY_SIZES, MODEL_SOURCING

st.set_page_config(page_title="AIGC Architect", page_icon="🧭", layout="wide")

# ── PwC 계열 팔레트 ──────────────────────────────────────────
ORANGE = "#DB4E18"     # PwC 다크 오렌지 (포인트·층2 AIGC)
ORANGE_LT = "#E88D14"
NAVY = "#1F2A44"       # 헤더·층3
SLATE = "#33373D"
GREY = "#6B7280"
AMBER = "#B26A00"
GREEN = "#2E7D32"

# 3층 구조 도식 (입력 화면 좌측 — 처음 보는 사람도 프레임워크를 이해하도록)
def layer_diagram() -> str:
    tier = ("height:66px;display:flex;flex-direction:column;align-items:center;"
            "justify-content:center;color:#fff;text-align:center")
    return (
        f"<div style='margin-top:20px'>"
        f"<div style='font-weight:800;color:{NAVY};font-size:1.0rem;margin-bottom:10px'>"
        f"🏗️ 통제의 3층 구조</div>"
        f"<div style='position:relative'>"
        # 층 3 (꼭대기)
        f"<div style='background:{NAVY};clip-path:polygon(35% 0,65% 0,75% 100%,25% 100%);{tier}'>"
        f"<div style='font-weight:800;font-size:0.9rem'>층 3 · 프로세스 통제</div>"
        f"<div style='font-size:0.68rem;opacity:0.9'>경영진주장 수준 위험 대응</div></div>"
        # 층 2 (핵심)
        f"<div style='background:{ORANGE};clip-path:polygon(25% 0,75% 0,85% 100%,15% 100%);{tier}'>"
        f"<div style='font-weight:800;font-size:0.92rem'>층 2 · AIGC &#9733;</div>"
        f"<div style='font-size:0.68rem;opacity:0.95'>AI 고유 통제 · 본 도구 핵심</div></div>"
        # 층 1 (기반)
        f"<div style='background:{GREY};clip-path:polygon(15% 0,85% 0,95% 100%,5% 100%);{tier}'>"
        f"<div style='font-weight:800;font-size:0.9rem'>층 1 · ITGC</div>"
        f"<div style='font-size:0.68rem;opacity:0.9'>시스템 통제 · 선결 확인</div></div>"
        f"</div>"
        f"<div style='background:#FBEEE6;border:1px solid #F3D6C6;border-radius:6px;"
        f"padding:8px 11px;margin-top:10px;font-size:0.72rem;color:{SLATE};line-height:1.5'>"
        f"<b style='color:{ORANGE}'>AIGC = AI General Controls</b><br>"
        f"ITGC(IT일반통제)가 시스템 통제의 신뢰 기반이듯, AIGC는 AI 모델 자체"
        f"(모델·프롬프트·학습데이터·비결정성)를 규율하는 통제입니다. 아직 공식 기준서에 없는 "
        f"신개념으로, 본 프로젝트가 재무보고 통제 맥락에서 구체화합니다.</div>"
        f"<div style='font-size:0.75rem;color:{GREY};margin-top:10px;text-align:center;"
        f"font-style:italic'>&#9650; 아래 층이 위 층의 신뢰 근거 &mdash; "
        f"&ldquo;층 1 없이 층 2 없고, 층 2 없이 층 3 없다&rdquo;</div>"
        f"<div style='font-size:0.68rem;color:{GREY};margin-top:4px;text-align:center'>"
        f"층 3 근거 · 감사기준서 315 문단 26 (경영진주장 수준 위험 대처 통제)</div>"
        f"</div>"
    )


# 스캔용 핵심 감사 용어 — 출력 텍스트에서 볼드 처리
_TERMS = [
    "감사기준서 540", "감사기준서 330", "KSA 540", "KSA 330",
    "K-IFRS 1109", "K-IFRS 1115", "K-IFRS 1113", "K-IFRS 1116",
    "K-IFRS 1008", "K-IFRS 1002", "K-IFRS 1036",
    "경영진주장", "경영진 검토통제", "Management Bias", "경영진 편의", "MRC",
    "비결정성", "Non-determinism", "벤치마킹", "benchmarking", "Benchmarking",
    "IPE", "Human-in-the-loop", "temperature", "할루시네이션",
    "재수행", "reperformance", "파라미터", "백테스트",
]


@st.cache_resource(show_spinner="RAG 인덱스 구축 중 (룰북+코퍼스 임베딩)...")
def get_index() -> RagIndex:
    return RagIndex()


def highlight(text: str) -> str:
    """핵심 감사 용어를 볼드 처리 (긴 용어 우선, 중첩 방지)."""
    if not text:
        return text
    marks: dict[str, str] = {}
    for i, term in enumerate(sorted(_TERMS, key=len, reverse=True)):
        if term in text:
            key = f"\x00{i}\x00"
            text = text.replace(term, key)
            marks[key] = term
    for key, term in marks.items():
        text = text.replace(key, f"**{term}**")
    return text


def plainify(text: str) -> str:
    """결과 표시용 용어 정리 — 룰북 원본은 그대로 두고 화면에서만 변환.
    · '어서션'은 기준서 공식용어 '경영진주장'으로 전량 치환
    · AI 용어는 '첫 등장'에만 괄호 풀이를 붙여(중복 방지) 비전공자도 이해하게 함."""
    if not text:
        return text
    text = text.replace("어서션", "경영진주장")
    glosses = [
        ("비결정적(non-deterministic)", "비결정적(같은 입력에도 결과가 달라질 수 있음)"),
        ("비결정성", "비결정성(같은 입력에도 결과가 달라질 수 있음)"),
        ("드리프트", "드리프트(시간이 지나며 성능·판단이 변함)"),
        ("게이팅", "충족 여부만 확인"),
        ("할루시네이션", "할루시네이션(없는 사실을 지어내는 현상)"),
    ]
    for term, gloss in glosses:
        idx = text.find(term)
        if idx != -1:  # 첫 등장만 풀이
            text = text[:idx] + gloss + text[idx + len(term):]
    return text


def show(text: str) -> str:
    """결과 텍스트 표시 파이프라인: 용어 정리 → 핵심어 볼드."""
    return highlight(plainify(text))


def badge(text: str, bg: str, fg: str = "#FFFFFF") -> str:
    return (f"<span style='background:{bg};color:{fg};padding:2px 9px;border-radius:11px;"
            f"font-size:0.70rem;font-weight:700;margin-right:5px;white-space:nowrap;"
            f"letter-spacing:0.2px'>{text}</span>")


def layer_meta(layer: int):
    return {
        1: ("층1 · ITGC", GREY),
        2: ("층2 · AIGC", ORANGE),
        3: ("층3 · 프로세스", NAVY),
    }.get(layer, (f"층{layer}", GREY))


def section_bar(title: str, subtitle: str, color: str) -> str:
    return (
        f"<div style='border-left:6px solid {color};background:#F7F8FA;"
        f"padding:10px 14px;border-radius:0 6px 6px 0;margin:6px 0 12px 0'>"
        f"<div style='font-size:1.05rem;font-weight:800;color:{color}'>{title}</div>"
        f"<div style='font-size:0.82rem;color:{GREY};margin-top:2px'>{subtitle}</div></div>"
    )


SIZE_NOTE = {
    "중소기업": "소규모·저복잡 기업: 통제가 덜 공식화됐을 수 있어 <b>위험기반</b> 접근. 이상적 업무분장이 어려우면 <b>보완적 적발통제</b>로 대체(315 보론3-20). ITGC 문서화가 낮을 수 있어 관찰·검사로 보완(315 A33·A170).",
    "중견기업": "중간 복잡성: 핵심 계정 중심의 공식 통제와 위험기반 표본을 병행. 통제 공식화 정도에 따라 절차 범위 조정(315 A16).",
    "대기업": "전담 IT 조직·<b>공식 ITGC</b> 전제(315 A170). 자동화 의존이 커 <b>전수·상시 모니터링</b>(aipc_07) 적용 여지가 크다.",
}
SIZE_EMPH = {"대기업": {"aipc_07"}}
SRC_NOTE = {
    "외부 API": "<b>서비스조직 이용에 해당(감사기준서 402)</b>: <b>유형2 보고서</b>(설계+운영효과성, 실무: <b>SOC 1</b>·정보보호는 <b>SOC 2</b>)로 공급사 통제 의존 여부를 결정, <b>상호보완적 이용자기업통제</b> 식별, 잔여위험 평가. 모델 내부 검증 불가 → 산출물(<b>IPE</b>) 검증·사이버보안·DLP 강조(315 A174).",
    "자체개발": "내부 개발·운영: 시스템 개발·변경관리(315 보론6-2b), <b>학습데이터 거버넌스</b>(ecl_06), 모델 변경관리(aipc_05)가 내부 책임. 서비스조직(402) 미적용.",
    "벤더 모델": "구매 소프트웨어 성격(315 A126): <b>변경관리·벤더 버전관리</b>(aipc_05) 중심. 벤더가 클라우드로 운영하면 서비스조직(402) 적용 — SOC 보고서 요구.",
}
SRC_EMPH = {"외부 API": {"aipc_03", "aipc_02"}, "자체개발": {"ecl_06", "aipc_05"}, "벤더 모델": {"aipc_05"}}


def combo_context(size: str, sourcing: str):
    """선택한 기업 규모 × 모델 조달에 따른 기준서 근거 맥락 진단 + 강조 통제 id 집합."""
    parts = []
    if size in SIZE_NOTE:
        parts.append(f"<b style='color:{NAVY}'>기업 규모 · {size}</b> — {SIZE_NOTE[size]}")
    if sourcing in SRC_NOTE:
        parts.append(f"<b style='color:{NAVY}'>모델 조달 · {sourcing}</b> — {SRC_NOTE[sourcing]}")
    html = (
        f"<div style='background:#F2F4F8;border:1px solid #D9DEE5;border-radius:8px;"
        f"padding:11px 14px;margin-bottom:12px;font-size:0.82rem;color:{SLATE};line-height:1.55'>"
        f"<div style='font-weight:800;color:{NAVY};margin-bottom:4px'>🧭 조합 맥락 진단</div>"
        f"{'<br><br>'.join(parts)}"
        f"<div style='font-size:0.74rem;color:{GREY};margin-top:6px'>※ 규모는 복잡성의 지표일 뿐 — "
        f"소규모도 복잡할 수 있음(감사기준서 315 문단6).</div></div>"
    )
    emph = set(SIZE_EMPH.get(size, set())) | set(SRC_EMPH.get(sourcing, set()))
    return html, emph


def render_card(card, emphasis=frozenset()) -> None:
    label, lcolor = layer_meta(card.layer)
    with st.container(border=True):
        badges = badge(card.control_id, GREY) + badge(label, lcolor)
        if card.forced:
            badges += badge("🔒 강제포함", SLATE)
        _atype = getattr(card, "activity_type", None)
        if _atype:
            badges += badge(f"유형 · {_atype}", "#0F766E")
        if card.control_id in emphasis:
            badges += badge("★ 이 조합 중요", "#B45309")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:4px;flex-wrap:wrap'>"
            f"<span style='font-size:1.02rem;font-weight:800;color:{NAVY};margin-right:6px'>"
            f"{card.name}</span>{badges}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:{GREY};font-size:0.74rem;font-weight:700;"
            f"margin:12px 0 2px 0;text-transform:uppercase;letter-spacing:0.4px'>왜 필요한가</div>",
            unsafe_allow_html=True,
        )
        st.markdown(show(card.why))
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{GREY};font-size:0.74rem;font-weight:700;"
            f"margin:0 0 2px 0;text-transform:uppercase;letter-spacing:0.4px'>"
            f"구체적 실행 모습 · 기준서 언어</div>",
            unsafe_allow_html=True,
        )
        st.markdown(show(card.implementation))
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        rows = ""
        for e in card.evidences:
            ev = badge("검증완료", GREEN) if e.verified else badge("원문 대조 전", AMBER)
            rows += (f"<div style='margin-top:3px'>📚 <b>{e.source}</b> "
                     f"<span style='color:{GREY}'>({e.paragraph})</span> {ev}</div>")
        meta = badge(f"근거 {len(card.evidences)}건", "#EEF0F3", SLATE)
        meta += badge(f"신뢰도 {card.confidence:.2f}", "#EEF0F3", SLATE)
        st.markdown(
            f"<div style='border-top:1px solid #ECECEC;padding-top:8px;font-size:0.80rem;"
            f"color:{SLATE}'>{rows}<div style='margin-top:6px'>{meta}</div></div>",
            unsafe_allow_html=True,
        )


# ============================================================
# 헤더
# ============================================================
st.markdown(
    f"<div style='display:flex;align-items:baseline;gap:10px'>"
    f"<span style='font-size:1.9rem'>🧭</span>"
    f"<span style='font-size:1.9rem;font-weight:800;color:{NAVY}'>AIGC Architect</span></div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div style='border-left:5px solid {ORANGE};padding:6px 0 6px 12px;margin:4px 0 2px 0;"
    f"color:{SLATE};font-size:0.95rem'>"
    f"<b>ITGC가 자동통제의 신뢰 기반이었듯, AIGC는 AI 지원 통제의 신뢰 기반이다.</b><br>"
    f"AI를 회계 프로세스에 도입할 때 어떤 경영진주장(재무제표가 옳다는 회사의 주장)이 위협받고 어떤 통제가 필요한지 설계합니다.</div>",
    unsafe_allow_html=True,
)
c1, c2 = st.columns(2)
c1.caption(f"🤖 LLM 모드 `{llm.llm_mode()}` · 🔎 임베딩 `{llm.embed_mode()}`")
c2.caption("자기참조 설계 — 이 도구의 실행 통제 틀(Harness)은 이 도구가 설계하는 AIGC 통제를 스스로에게 적용")
st.divider()

# ============================================================
# [1] 입력 — 2단 레이아웃
# ============================================================
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.markdown(f"<div style='font-weight:800;color:{NAVY};font-size:1.05rem'>"
                f"① 전제 조건 확인 · ITGC (층 1)</div>", unsafe_allow_html=True)
    st.caption("층 1 없이 층 2 없고, 층 2 없이 층 3 없다. 하나라도 '아니오'면 하위 설계가 무의미합니다.")
    gate = rb.itgc_gate()
    itgc_answers: dict[str, bool] = {}
    _groups: dict[str, list] = {}
    for chk in gate["checks"]:
        _groups.setdefault(chk.get("category", "기타"), []).append(chk)
    for _cat, _checks in _groups.items():
        st.markdown(f"<div style='font-size:0.78rem;font-weight:700;color:{ORANGE};"
                    f"margin:8px 0 2px 0'>{_cat}</div>", unsafe_allow_html=True)
        for chk in _checks:
            itgc_answers[chk["id"]] = st.checkbox(
                chk["question"], value=True, key=f"gate_{chk['id']}",
                help=chk["why_prerequisite"],
            )
    st.caption("감사기준서 315 보론6의 IT 프로세스 대분류(접근·변경·운영)를 따릅니다. "
               "보론6은 AI 고유 리스크(비결정성·드리프트 등)를 다루지 않으므로 그 영역은 층2 AIGC가 담당합니다.")
    override = False
    if not all(itgc_answers.values()):
        st.warning("ITGC 미비 항목이 있습니다. 조건부로 진행하려면 아래를 체크하세요.")
        override = st.checkbox("⚠️ ITGC 미비를 인지하고 조건부로 진행", value=False)

    st.markdown(layer_diagram(), unsafe_allow_html=True)

with right:
    st.markdown(f"<div style='font-weight:800;color:{NAVY};font-size:1.05rem'>"
                f"② 산업 · 규모 · 조달</div>", unsafe_allow_html=True)
    industries = rb.industries()
    ind_keys = list(industries.keys())
    ic1, ic2, ic3 = st.columns(3)
    industry = ic1.selectbox("산업", ind_keys, format_func=lambda k: industries[k]["name"])
    company_size = ic2.selectbox("기업 규모", COMPANY_SIZES, index=1,
                                 help="통제 설계의 현실성을 조정합니다.")
    model_sourcing = ic3.selectbox("모델 조달", MODEL_SOURCING, index=1)
    st.caption(industries[industry]["description"])

    procs = skills.identify_significant_accounts(industry)
    st.markdown(f"<div style='font-weight:800;color:{NAVY};font-size:1.05rem;margin-top:6px'>"
                f"③ AI 개입 프로세스 선택</div>", unsafe_allow_html=True)
    st.caption("산업에 따라 목록이 실제로 달라집니다. ★=유의적 추정 · '해당 없음'은 계정 부존재로 제외됩니다.")
    st.markdown(
        f"<div style='background:#F7F8FA;border:1px solid #E2E6EB;border-radius:8px;"
        f"padding:9px 12px;margin:4px 0 8px 0;font-size:0.78rem;color:{SLATE};line-height:1.55'>"
        f"<b style='color:{ORANGE}'>🔬 DEEP</b> — 근거 조항 원문 대조·감사인 관점까지 완비한 프로세스. "
        f"설계안이 신뢰도 0.85로 <b>수용</b>됩니다.&nbsp;&nbsp;"
        f"<b style='color:{GREY}'>◻ OUTLINE</b> — 뼈대 수준 프로세스. 신뢰도 0.50으로 "
        f"자동으로 <b>'사람 검토 필요'</b>에 격리됩니다 — 덜 검증된 것은 덜 검증됐다고 표시하는 설계입니다."
        f"</div>", unsafe_allow_html=True)
    _depths = {pid: pd.get("depth", "outline") for pid, pd in rb.ledger_processes().items()}
    selected: list[str] = []
    for p in procs:
        default = p.significance in ("critical", "significant")
        star = " ★" if p.significance == "critical" else ""
        _d = _depths.get(p.process_id)
        _badge = " 🔬" if _d == "deep" else ""
        checked = st.checkbox(f"{p.name}{star}{_badge}", value=default,
                              key=f"proc_{p.process_id}",
                              help=p.rationale + (f"  [{'DEEP — 원문 대조 완비' if _d == 'deep' else 'OUTLINE — 뼈대, 사람 검토 격리' if _d else '통제 원장 미수록'}]"))
        if checked:
            selected.append(p.process_id)
            if p.significance == "low":
                st.caption("↳ ⚠️ 우선순위 하위 — 유의적 계정 설계 후 검토 권장")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
run = st.button("통제 설계 실행", type="primary", use_container_width=True, disabled=not selected)
if not selected:
    st.info("프로세스를 하나 이상 선택하세요.")

# ============================================================
# [2] 실행 — 스텝별 진행
# ============================================================
if run:
    index = get_index()
    inputs = AgentInputs(
        industry=industry, selected_processes=selected,
        itgc_answers=itgc_answers, override_itgc=override,
        company_size=company_size, model_sourcing=model_sourcing,
    )
    status = st.status("에이전트 실행 중...", expanded=True)
    output = None
    for kind, payload in run_agent(inputs, index):
        if kind == "step":
            status.write(payload)
            time.sleep(0.35)
        elif kind == "done":
            output = payload
    status.update(label="에이전트 실행 완료", state="complete", expanded=False)
    for _k in [k for k in st.session_state if k.startswith("promote_")]:
        del st.session_state[_k]
    st.session_state["output"] = output
    st.session_state["ctx"] = {"industry": industries[industry]["name"],
                               "size": company_size, "sourcing": model_sourcing}

# ============================================================
# [3] 출력 — 탭
# ============================================================
output = st.session_state.get("output")
if output is not None:
    ctx = st.session_state.get("ctx", {})
    st.markdown(
        f"<div style='background:{NAVY};color:#fff;padding:10px 16px;border-radius:6px;"
        f"font-weight:700;margin:4px 0 10px 0'>결과 · {ctx.get('industry','')} "
        f"<span style='opacity:0.6'>|</span> {ctx.get('size','')} "
        f"<span style='opacity:0.6'>|</span> {ctx.get('sourcing','')}</div>",
        unsafe_allow_html=True,
    )

    if output.stopped_reason:
        st.error(output.stopped_reason)
        with st.expander("ITGC 게이트 상세"):
            for cid in output.gate.failed_checks:
                q = next(c["question"] for c in rb.itgc_gate()["checks"] if c["id"] == cid)
                st.write(f"❌ {q}")
            st.info(rb.itgc_gate()["fail_rationale"])
    else:
        tabs = st.tabs([
            "✅ 전제 조건", "🗺️ 경영진주장 리스크 맵", "🎛️ 통제 설계안",
            "👓 감사인 관점", "⚠️ 사람 검토 필요", "🔍 실행 로그",
        ])

        # --- 선결 확인 ---
        with tabs[0]:
            g = output.gate
            (st.success if g.passed else st.warning)(g.message)
            _res_groups: dict[str, list] = {}
            for chk in rb.itgc_gate()["checks"]:
                _res_groups.setdefault(chk.get("category", "기타"), []).append(chk)
            for _cat, _checks in _res_groups.items():
                _n_ok = sum(1 for c in _checks if c["id"] not in g.failed_checks)
                _all_ok = _n_ok == len(_checks)
                st.markdown(
                    f"<div style='font-size:0.82rem;font-weight:700;"
                    f"color:{ORANGE if _all_ok else '#B42318'};margin:10px 0 2px 0'>"
                    f"{_cat} · {_n_ok}/{len(_checks)} 충족</div>",
                    unsafe_allow_html=True,
                )
                for chk in _checks:
                    ok = chk["id"] not in g.failed_checks
                    st.write(f"{'✅' if ok else '❌'} {chk['question']}")
                    st.caption(f"   → {chk['why_prerequisite']}")

        # --- 어서션 리스크 맵 ---
        with tabs[1]:
            if not output.risks:
                st.info("선택 프로세스의 통제 원장 리스크가 없습니다 (확장 예정 프로세스일 수 있음).")
            cur = None
            for r in output.risks:
                if r.process_name != cur:
                    cur = r.process_name
                    st.markdown(f"<div style='font-weight:800;color:{NAVY};margin-top:10px'>"
                                f"{cur}</div>", unsafe_allow_html=True)
                st.markdown(f"- {show(r.risk)}")
                st.caption(f"   영향받는 경영진주장: {plainify(r.assertion_impact)}")

        # --- 통제 설계안 (3층 위계) ---
        with tabs[2]:
            acc = output.harness.accepted
            l2 = [c for c in acc if c.layer == 2]
            l3 = [c for c in acc if c.layer == 3]
            l1 = [c for c in acc if c.layer == 1]
            _promoted = [c for c in output.harness.quarantined
                         if st.session_state.get(f"promote_{c.control_id}")]
            _xlsx = xl.build_excel(
                output, ctx, rb.itgc_gate()["checks"],
                {c.control_id for c in _promoted},
            )
            st.download_button(
                "📥 엑셀 다운로드 — 감사조서형 (전제조건·리스크맵·통제설계안·사람검토)",
                data=_xlsx,
                file_name=f"AIGC_통제설계안_{ctx.get('industry','')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.caption("'사람 검토 필요' 탭에서 확정한 통제는 '사람 검토 후 확정' 경로로 표시되어 엑셀에 포함됩니다.")
            st.caption(f"수용된 통제 {len(acc)}개 · 층2(AIGC) {len(l2)} / 층3(프로세스) {len(l3)}"
                       f"{f' / 층1 {len(l1)}' if l1 else ''}. "
                       "판정이 아니라 '이 프로필이면 이런 통제가 필요하다'는 설계 권고입니다. 근거 미검증 항목은 '원문 대조 전' 배지로 표시됩니다.")

            _banner, _emph = combo_context(ctx.get("size", ""), ctx.get("sourcing", ""))
            st.markdown(_banner, unsafe_allow_html=True)

            st.markdown(
                f"<div style='background:#F7F8FA;border:1px solid #E2E6EB;border-radius:8px;"
                f"padding:10px 14px;margin-bottom:12px;font-size:0.86rem;color:{SLATE};line-height:1.55'>"
                f"<b style='color:{ORANGE}'>AI 도입 후 통제 진화</b> — 사람의 역할은 "
                f"<b>검토(Check) → 위험평가·상시 모니터링</b>으로, 테스트는 <b>표본 → 전수</b>로, "
                f"기준은 <b>정성적 판단 → 명시적 정량화</b>로 전환됩니다. 아래 통제 카드는 이 전환을 반영합니다."
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(section_bar(
                "🧩 층 2 · AIGC — AI 고유 통제",
                "모델·프롬프트·학습데이터·비결정성, 툴셋·접근권한 제한, 전수·상시 모니터링 통제. AI를 신뢰 가능하게 만드는 기반.",
                ORANGE), unsafe_allow_html=True)
            for c in l2:
                render_card(c, _emph)
            if not l2:
                st.info("이 조합에서 수용된 층2 통제가 없습니다. '사람 검토 필요' 탭을 확인하세요.")

            st.markdown(
                f"<div style='text-align:center;color:{ORANGE};font-weight:700;"
                f"margin:14px 0 6px 0'>▲ 아래 층 3 통제는 위 층 2(AIGC)가 확보되어야 신뢰할 수 있다</div>",
                unsafe_allow_html=True,
            )
            st.markdown(section_bar(
                "🏛️ 층 3 · 프로세스 통제 — 경영진주장 수준 위험 대응",
                "회계 계정별 통제. 층 2가 확보되었다는 전제 위에서만 유효.",
                NAVY), unsafe_allow_html=True)
            for c in l3:
                render_card(c, _emph)
            if not l3:
                st.info("이 조합에서 수용된 층3 통제가 없습니다.")

            if _promoted:
                st.markdown(section_bar(
                    "✅ 사람 검토 후 확정된 통제",
                    "'사람 검토 필요' 탭에서 검토를 마치고 확정한 통제. 격리 → 검토 → 확정의 HITL 경로를 거쳤습니다.",
                    "#1A7F37"), unsafe_allow_html=True)
                for c in _promoted:
                    render_card(c, _emph)



        # --- 감사인 관점 (벤치마킹 붕괴 콜아웃) ---
        with tabs[3]:
            st.markdown(
                f"<div style='border-left:6px solid {ORANGE};background:#FBEEE6;"
                f"padding:14px 18px;border-radius:0 8px 8px 0;color:{SLATE};line-height:1.65'>"
                f"<div style='font-weight:800;color:{ORANGE};margin-bottom:4px'>"
                f"⚡ 벤치마킹 논리의 붕괴</div>"
                f"AI 기반 자동통제는 <b>비결정성</b>(같은 자료를 넣어도 결과가 달라질 수 있는 성질)으로 인해 기존 IT 환경의 "
                f"<b>‘벤치마킹(Benchmarking, Sample=1) 논리’</b>가 적용되지 않습니다. "
                f"나아가 AI 오류는 개별 건이 아니라 <b>모집단 전체로 동시에 확산(증폭성)</b>되므로 "
                f"표본 축소가 아닌 <b>전수 검증·상시 모니터링</b>(aipc_07)이 대안이 됩니다. 따라서 감사인은 "
                f"<b>AIGC(층 2)</b>의 운영 유효성을 먼저 테스트한 후, 예외 사항에 대한 "
                f"<b>경영진 검토통제(MRC)</b>에 <b>실증적 성격의 테스트</b>를 결합하여 접근해야 합니다."
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='border-left:6px solid {NAVY};background:#F2F4F8;padding:12px 16px;"
                f"border-radius:0 8px 8px 0;color:{SLATE};margin-top:12px;line-height:1.6'>"
                f"<div style='font-weight:800;color:{NAVY};margin-bottom:4px'>⚖️ 경영진 책임 관점</div>"
                f"상법 개정으로 이사의 내부통제 책임이 '감독'에서 '자기수행'으로 강화되었습니다. "
                f"통제 미비는 단순 오류를 넘어 <b>선관주의 의무 위반·공시 위반 리스크</b>로 직결될 수 있으며, "
                f"외부감사인보다 경영진이 선제적으로 취약점을 밝혀야 하는 방향입니다(핵심감사사항 KAM 반영 비율 높음)."
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown(show(output.auditor_view))

        # --- 사람 검토 필요 (격리) ---
        with tabs[4]:
            q = output.harness.quarantined
            st.caption("에이전트가 확신하지 못해 확정하지 않고 격리한 항목 (Human-in-the-loop). "
                       "대개 골격(outline) 단계로 원문 대조·상술이 필요합니다.")
            for card in q:
                label, lcolor = layer_meta(card.layer)
                with st.container(border=True):
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:4px;flex-wrap:wrap'>"
                        f"<span style='font-weight:800;color:{NAVY};margin-right:6px'>{card.name}</span>"
                        f"{badge(card.control_id, GREY)}{badge(label, lcolor)}"
                        f"{badge('유형 · '+getattr(card,'activity_type',None), '#0F766E') if getattr(card,'activity_type',None) else ''}"
                        f"{badge(f'신뢰도 {card.confidence:.2f}', '#EEF0F3', SLATE)}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(show(card.implementation))
                    st.checkbox(
                        "✅ 검토 완료 — 설계안으로 확정 (엑셀 '통제설계안' 시트에 포함)",
                        key=f"promote_{card.control_id}",
                        help="사람 검토(HITL)를 마친 통제만 확정하세요. 확정 시 통제 설계안 탭과 엑셀에 '사람 검토 후 확정'으로 표시됩니다.",
                    )
                    st.caption("📚 근거 " + ", ".join(e.source for e in card.evidences) + " — 검토·상술 후 확정 대상")
            if not q:
                st.success("격리된 항목이 없습니다.")
            if output.harness.rejected:
                with st.expander(f"🚫 Harness가 폐기한 통제 {len(output.harness.rejected)}건 "
                                 f"(스키마·근거 미확인)"):
                    for m in output.harness.rejected:
                        st.write(m)

        # --- 실행 로그 ---
        with tabs[5]:
            st.caption("각 기능(스킬) 호출 순서 · 입력 · 검색된 문단(청크) · 출력 · 소요시간 (감사추적)")
            for e in output.log.entries:
                st.markdown(f"**[{e.step}] `{e.skill}`** · {e.elapsed_ms}ms")
                st.caption(f"입력: {e.inputs}")
                if e.retrieved_ids:
                    st.caption(f"검색 청크: {', '.join(e.retrieved_ids)}")
                st.caption(f"출력: {e.output_summary}")
            if output.log.notes:
                st.markdown(f"<div style='font-weight:800;color:{ORANGE};margin-top:6px'>"
                            f"통제 이벤트 (분기·게이트·폐기)</div>", unsafe_allow_html=True)
                for n in output.log.notes:
                    st.write(f"- {n}")
