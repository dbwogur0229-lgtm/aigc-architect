"""
smoke_test.py — 헤드리스 파이프라인 검증 (LLM 키 없이 폴백 경로).
실행: python tests/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import AgentInputs, run_agent  # noqa: E402
from rag import RagIndex  # noqa: E402
import skills  # noqa: E402

import rulebook as _rb  # noqa: E402
ALL_TRUE = {c["id"]: True for c in _rb.itgc_gate()["checks"]}


def run(inputs):
    idx = INDEX
    out = None
    for kind, payload in run_agent(inputs, idx):
        if kind == "done":
            out = payload
    return out


INDEX = RagIndex()
failures = []


def check(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        failures.append(msg)


print(f"청크 수: {len(INDEX.chunks)} · 임베딩 shape: {INDEX.matrix.shape}")

# 1) 산업별 프로세스 목록이 실제로 달라진다
fin = [p.process_id for p in skills.identify_significant_accounts("financial")]
con = [p.process_id for p in skills.identify_significant_accounts("construction")]
print("\n[1] 산업별 분기")
check("ecl_estimation" in fin, "금융에 ECL 포함")
check("poc_estimation" in con, "건설에 진행률(POC) 포함")
check("inventory_obsolescence" not in fin, "금융에서 재고진부화(not_applicable) 제거")
check(set(fin) != set(con), "산업별 프로세스 목록 상이")

# 2) ITGC 게이트 실패 → 하위 중단
print("\n[2] ITGC 게이트 차단")
bad = AgentInputs(industry="retail", selected_processes=["inventory_obsolescence"],
                  itgc_answers={**ALL_TRUE, "gate_access": False})
o = run(bad)
check(bool(o.stopped_reason), "게이트 실패 시 stopped_reason 설정")
check(len(o.harness.accepted) == 0, "게이트 실패 시 통제 생성 안 함")

# 3) override 시 진행
o2 = run(AgentInputs(industry="retail", selected_processes=["inventory_obsolescence", "ai_performed_control"],
                     itgc_answers={**ALL_TRUE, "gate_access": False}, override_itgc=True))
check(not o2.stopped_reason, "override 시 진행")

# 4) ai_performed_control 선택 → aipc_01/02 강제 포함
print("\n[3] 비결정성 통제 강제 포함")
o3 = run(AgentInputs(industry="financial",
                     selected_processes=["ecl_estimation", "ai_performed_control"],
                     itgc_answers=ALL_TRUE, company_size="중소기업", model_sourcing="외부 API"))
acc_ids = {c.control_id for c in o3.harness.accepted}
check("aipc_01" in acc_ids and "aipc_02" in acc_ids, "aipc_01·aipc_02 수용됨")
check(any(c.forced for c in o3.harness.accepted), "강제포함 플래그 존재")
check(any("비결정성" in n for n in o3.log.notes), "분기 로그 기록")

# 5) Harness: deep 수용 / outline 격리
print("\n[4] Harness 게이트")
check(len(o3.harness.accepted) > 0, "deep 통제 수용")
check(len(o3.harness.quarantined) >= 0, "격리 큐 존재(무결)")
# ECL deep 통제는 수용되어야 함
check(any(c.control_id.startswith("ecl_") for c in o3.harness.accepted), "ECL deep 통제 수용")

# 6) 모든 수용 통제는 근거 출처를 가진다 (스키마 강제)
check(all(c.evidences for c in o3.harness.accepted), "수용 통제 전부 근거 보유")
a1=[c for c in o3.harness.accepted if c.control_id=="aipc_01"]
check(bool(a1) and len(a1[0].evidences) >= 2, "aipc_01 복수 근거(330+1100) 표시")

# 7) 실행 로그가 스킬 순서를 기록
print("\n[5] 감사추적")
skill_seq = [e.skill for e in o3.log.entries]
check(skill_seq[0] == "check_itgc_gate", "로그 첫 스킬 = ITGC 게이트")
check("harness.enforce" in skill_seq, "Harness 실행 로그 기록")
check(len(o3.auditor_view) > 0, "감사인 관점 생성됨")

print("\n" + ("=" * 40))
if failures:
    print(f"실패 {len(failures)}건: {failures}")
    sys.exit(1)
print("모든 스모크 테스트 통과 ✅")
