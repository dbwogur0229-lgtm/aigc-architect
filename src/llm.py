"""
llm.py — LLM 및 임베딩 접근 계층 (Anthropic 기준)
------------------------------------------------------------
설계 원칙 두 가지:

1) temperature 고정 (Harness 통제 #6 — 비결정성 관리)
   생성 호출은 모두 TEMPERATURE 상수를 사용한다. 도구가 스스로에게
   "AI 통제는 temperature 를 고정해 출력을 재현 가능하게 하라"는 AIGC 통제를 적용한다.

2) 우아한 성능저하 (graceful degradation)
   - ANTHROPIC_API_KEY 가 없으면 LLM 호출은 룰북 기반 템플릿으로 대체된다.
     → API 키 없이도 앱이 end-to-end 로 돌아가며(데모 GIF 용이) Agent·Harness·RAG 를 증명한다.
   - 임베딩은 VOYAGE_API_KEY 가 있으면 Voyage, 없으면 로컬 TF-IDF(코사인) 로 대체된다.
     → 벡터 검색 로직(numpy 코사인)은 두 경우 모두 동일하다. 아래 rag.py 참조.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter

import numpy as np

# 비결정성 관리: 생성 호출 temperature 고정
TEMPERATURE = 0.0
CLAUDE_MODEL = os.environ.get("AIGC_CLAUDE_MODEL", "claude-sonnet-4-20250514")
VOYAGE_MODEL = os.environ.get("AIGC_VOYAGE_MODEL", "voyage-3")


def has_llm() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def llm_mode() -> str:
    return "anthropic" if has_llm() else "fallback-template"


def embed_mode() -> str:
    return "voyage" if os.environ.get("VOYAGE_API_KEY") else "local-tfidf"


# ------------------------------------------------------------
# 1) 생성 (Claude)
# ------------------------------------------------------------
def complete(system: str, user: str, max_tokens: int = 700) -> str:
    """
    Claude 로 텍스트를 생성한다. 키가 없으면 빈 문자열을 반환하고,
    호출부(skills.py)가 룰북 기반 템플릿으로 대체한다.
    """
    if not has_llm():
        return ""
    try:
        import anthropic  # 지연 임포트: 키 없을 때 의존성 불필요
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=TEMPERATURE,        # ← 고정
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    except Exception as e:  # 네트워크·쿼터 실패 시에도 앱은 계속 돈다
        return f""  # 호출부가 템플릿 대체


# ------------------------------------------------------------
# 2) 임베딩 — fit(코퍼스) / transform(질의) 로 같은 벡터공간을 공유한다.
#    (로컬 TF-IDF 는 코퍼스에서 학습한 vocab·idf 로 질의를 투영해야
#     코사인 유사도가 성립하므로 stateful Embedder 로 설계한다.)
# ------------------------------------------------------------
def make_embedder() -> "Embedder":
    if os.environ.get("VOYAGE_API_KEY"):
        try:
            import voyageai  # noqa: F401  존재 확인
            return VoyageEmbedder()
        except Exception:
            pass
    return LocalTfidfEmbedder()


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Embedder:
    def fit(self, corpus: list[str]) -> np.ndarray:  # (n, d) 정규화 행렬
        raise NotImplementedError

    def transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class VoyageEmbedder(Embedder):
    """Voyage 임베딩. 문서/질의를 독립적으로 임베딩한다."""

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        import voyageai
        client = voyageai.Client()
        vecs = client.embed(texts, model=VOYAGE_MODEL, input_type=kind).embeddings
        return _l2_normalize(np.asarray(vecs, dtype=np.float32))

    def fit(self, corpus: list[str]) -> np.ndarray:
        return self._embed(corpus, "document")

    def transform(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "query")


# --- 로컬 TF-IDF 폴백 -----------------------------------------
# 외부 임베딩 키가 없어도 RAG 가 동작하도록 하는 순수 numpy 벡터라이저.
# 한국어/영문/숫자 토크나이저 + TF-IDF. 규모가 작아(청크 수백 개)
# 이 정도로 상위 k 검색 품질은 충분하다. (README 설계 근거 참조)
_TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z]+|[0-9]+")


def _tokenize(t: str) -> list[str]:
    return _TOKEN_RE.findall(t.lower())


class LocalTfidfEmbedder(Embedder):
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def fit(self, corpus: list[str]) -> np.ndarray:
        docs = [_tokenize(t) for t in corpus]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        n = max(1, len(corpus))
        self.idf = {w: math.log((1 + n) / (1 + df[w])) + 1.0 for w in self.vocab}
        return self._vectorize(docs)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self._vectorize([_tokenize(t) for t in texts])

    def _vectorize(self, docs: list[list[str]]) -> np.ndarray:
        m = np.zeros((len(docs), max(1, len(self.vocab))), dtype=np.float32)
        for i, d in enumerate(docs):
            tf = Counter(d)
            for w, c in tf.items():
                j = self.vocab.get(w)
                if j is not None:
                    m[i, j] = (c / max(1, len(d))) * self.idf.get(w, 0.0)
        return _l2_normalize(m)
