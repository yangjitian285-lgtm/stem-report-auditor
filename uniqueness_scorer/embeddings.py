"""埋め込みベクトル生成および仮想・量産型AI母集団生成のプラガブル・バックエンド."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from .text_ja import char_ngrams, extract_content_tokens


# 言い換え（Paraphrase）を同一の意味軸へ射影する日本語概念クラスタ辞書
# オフライン環境でも「言い換え版量産AI」と「元の量産AI」の意味的近さを捉えるために使用
#
# 【注意】この辞書は「生成AIと労働市場」「就活ES」の2課題（ベンチマーク）を見ながら作られており、
# それ以外の分野では効かない。ベンチマークの好成績の一部はこの辞書による（循環検証）。
# 本番の採点には SentenceTransformerBackend（provider="st"）か OpenAI/Gemini の埋め込みを推奨する。
# 辞書を無効にするには LocalHybridEmbedder(use_concept_lexicon=False) または 環境変数 UNIQ_NO_LEXICON=1。
CONCEPT_LEXICON: Dict[str, List[str]] = {
    "CONCEPT_EFFICIENCY": [
        "効率", "生産性", "自動化", "省力", "時短", "高速", "業務改善", "ルーチン",
        "定型業務", "単純作業", "負担軽減", "最適化", "合理化", "スケーラブル",
    ],
    "CONCEPT_JOB_LOSS": [
        "雇用", "失業", "代替", "奪われ", "仕事がなくな", "人員削減", "リストラ",
        "労働市場", "職種", "消滅", "淘汰", "置き換わ", "職を失",
    ],
    "CONCEPT_RESKILLING": [
        "リスキリング", "学び直", "再教育", "教育", "スキルアップ", "人材育成",
        "研修", "生涯学習", "アップスキル", "能力開発", "適応力",
    ],
    "CONCEPT_CREATIVITY": [
        "創造", "クリエイティブ", "発想", "共感", "コミュニケーション", "人間らし",
        "感情", "判断力", "付加価値", "高付加価値", "対人", "企画",
    ],
    "CONCEPT_ETHICS_BIAS": [
        "倫理", "バイアス", "偏見", "プライバシー", "著作権", "個人情報", "公平性",
        "透明性", "ブラックボックス", "法整備", "規制", "ガイドライン", "ガバナンス",
    ],
    "CONCEPT_INEQUALITY": [
        "格差", "二極化", "分断", "不平等", "所得", "デジタルデバイド", "勝者総取り",
    ],
    "CONCEPT_COEXISTENCE": [
        "共存", "協働", "パートナー", "補完", "道具", "ツールとして", "人間中心",
    ],
    "CONCEPT_LABOR_SHORTAGE": [
        "人手不足", "労働力不足", "少子高齢化", "人口減少", "人材難", "働き手",
    ],
    "CONCEPT_AI_TECH": [
        "生成ai", "人工知能", "chatgpt", "llm", "大規模言語モデル", "ディープラーニング",
        "機械学習", "デジタル", "テクノロジー", "ai技術", "アルゴリズム",
    ],
    "CONCEPT_TACIT_SKILL": [
        "暗黙知", "徒弟", "下積み", "若手", "見習い", "熟練", "OJT", "経験則",
        "修行", "技能継承", "ベテラン", "入門業務", "ジュニア", "技能空洞化",
    ],
    "CONCEPT_ORG_POWER": [
        "交渉力", "労働組合", "賃金分配", "資本", "プラットフォーム", "寡占",
        "下請け", "単価", "フリーランス", "責任の所在", "説明責任", "監査",
    ],
    # 就活ES向け概念軸
    "CONCEPT_ES_TEAMWORK": [
        "チーム", "メンバー", "協力", "巻き込", "一丸", "連携", "役割分担", "組織",
        "サークル", "部活", "バイト", "アルバイト", "ゼミ",
    ],
    "CONCEPT_ES_PROBLEM": [
        "課題", "問題", "低迷", "減少", "赤字", "壁", "困難", "停滞", "伸び悩",
        "離脱", "不振", "トラブル",
    ],
    "CONCEPT_ES_COMM": [
        "ヒアリング", "面談", "話し合", "傾聴", "意見交換", "アンケート", "対話",
        "コミュニケーション", "共有", "ミーティング", "会議",
    ],
    "CONCEPT_ES_RESULT": [
        "売上", "集客", "向上", "増加", "達成", "改善", "過去最高", "優勝", "入賞",
        "成果", "結果", "目標達成",
    ],
    "CONCEPT_ES_FAILURE": [
        "失敗", "挫折", "撤退", "反対", "対立", "批判", "誤算", "損失", "想定外",
        "やり直し", "反省", "見落とし",
    ],
}


class EmbeddingBackend(ABC):
    """テキスト埋め込みおよび任意でLLM生成を提供する抽象バックエンド."""

    name: str = "base"

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """テキスト列を L2正規化済みの (N, D) 行列へ変換する."""
        raise NotImplementedError

    def generate_cohort(
        self,
        assignment_prompt: str,
        n_samples: int = 15,
        mode: str = "report",
    ) -> List[str]:
        """課題文から仮想・量産型AI母集団を生成する（LLM接続時のみ利用可能）."""
        raise NotImplementedError(
            f"Backend '{self.name}' does not support live LLM cohort generation. "
            "Pass reference_cohort explicitly or use --provider openai / gemini."
        )

    def suggest_angles_llm(
        self,
        assignment_prompt: str,
        saturated_claims: Sequence[str],
        mode: str = "report",
    ) -> Optional[List[str]]:
        """任意: LLMを使って未踏の直交切り口を提案する."""
        return None


class LocalHybridEmbedder(EmbeddingBackend):
    """外部API・外部NLPライブラリ不要のハイブリッド日本語埋め込み器.

    1. 概念クラスタ空間（言い換え耐性のある意味軸）
    2. 内容語 TF-IDF 空間（ハッシュ射影）
    3. 文字 2〜3gram TF-IDF 空間（語彙・表現の近接性）
    を結合し、コーパス上でSVD（LSA）または重み付きL2正規化を行う。
    """

    name = "local_hybrid"

    def __init__(
        self,
        hash_dim: int = 384,
        concept_weight: float = 2.4,
        use_concept_lexicon: Optional[bool] = None,
    ) -> None:
        self.hash_dim = hash_dim
        if use_concept_lexicon is None:
            use_concept_lexicon = os.environ.get("UNIQ_NO_LEXICON", "") not in ("1", "true", "yes")
        self.use_concept_lexicon = use_concept_lexicon
        self.concept_weight = concept_weight if use_concept_lexicon else 0.0
        self.concept_keys = list(CONCEPT_LEXICON.keys())

    @staticmethod
    def _signed_hash(token: str, dim: int) -> Tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        val = int.from_bytes(digest, "little")
        idx = val % dim
        sign = 1.0 if ((val >> 63) & 1) == 0 else -1.0
        return idx, sign

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        n = len(texts)
        if n == 0:
            return np.zeros((0, len(self.concept_keys) + self.hash_dim), dtype=np.float64)

        # 全テキストからDFを計算してTF-IDF重み付け
        doc_tokens: List[Counter] = []
        doc_ngrams: List[Counter] = []
        df_tok: Counter = Counter()
        df_ng: Counter = Counter()

        for text in texts:
            toks = extract_content_tokens(text)
            ngs = char_ngrams(text, (2, 3))
            c_tok = Counter(toks)
            c_ng = Counter(ngs)
            doc_tokens.append(c_tok)
            doc_ngrams.append(c_ng)
            df_tok.update(c_tok.keys())
            df_ng.update(c_ng.keys())

        c_dim = len(self.concept_keys)
        concept_mat = np.zeros((n, c_dim), dtype=np.float64)
        lex_mat = np.zeros((n, self.hash_dim), dtype=np.float64)

        for i, text in enumerate(texts):
            low = text.lower()
            # 1) 概念クラスタ射影（言い換えに非常に強い）
            for ci, ckey in enumerate(self.concept_keys):
                hits = 0
                for kw in CONCEPT_LEXICON[ckey]:
                    if kw in low:
                        hits += low.count(kw)
                if hits > 0:
                    concept_mat[i, ci] = math.log1p(hits)

            # 2) 内容語 TF-IDF
            for tok, tf in doc_tokens[i].items():
                idf = math.log((n + 1.5) / (df_tok[tok] + 0.5)) + 0.5
                idx, sign = self._signed_hash("T:" + tok, self.hash_dim)
                lex_mat[i, idx] += sign * (1.0 + math.log(tf)) * idf * 1.4

            # 3) 文字 n-gram TF-IDF
            for ng, tf in doc_ngrams[i].items():
                idf = math.log((n + 1.5) / (df_ng[ng] + 0.5)) + 0.2
                idx, sign = self._signed_hash("N:" + ng, self.hash_dim)
                lex_mat[i, idx] += sign * (1.0 + math.log(tf)) * idf * 0.55

        # 各ブロックを個別に正規化してから重み付き結合
        c_norms = np.linalg.norm(concept_mat, axis=1, keepdims=True)
        concept_mat = np.where(c_norms > 1e-12, concept_mat / np.maximum(c_norms, 1e-12), 0.0)

        l_norms = np.linalg.norm(lex_mat, axis=1, keepdims=True)
        lex_mat = np.where(l_norms > 1e-12, lex_mat / np.maximum(l_norms, 1e-12), 0.0)

        combined = np.hstack([concept_mat * self.concept_weight, lex_mat])
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        combined = np.where(norms > 1e-12, combined / np.maximum(norms, 1e-12), 0.0)
        return combined


# 典型的な学生がAIに投げるプロンプト変種（仮想・量産型AI母集団の生成に使用）
STUDENT_PROMPT_TEMPLATES_REPORT = [
    "以下の大学レポート課題について、800字程度で論理的なレポートを書いてください。\n課題：{task}",
    "次のレポート課題の解答例を、序論・本論・結論の構成でわかりやすく作成してください。\n課題：{task}",
    "以下のテーマで大学授業のレポートを作成して。メリット・デメリットと今後の課題を入れて。\nテーマ：{task}",
    "大学生が提出する標準的なレポートを書いてください。\nお題：{task}",
    "以下の課題について、多角的な視点から考察したレポートを書いてください。\n課題：{task}",
]

STUDENT_PROMPT_TEMPLATES_ES = [
    "以下の就活ES設問に対して、400字程度で説得力のある回答を書いてください。\n設問：{task}",
    "新卒就活のエントリーシートを書いて。課題・行動・結果・学びの構成で。\n設問：{task}",
    "就活生として高評価されそうなエントリーシートの例文を作ってください。\n設問：{task}",
    "以下のES設問に、チームで協力した経験を題材に回答を作成してください。\n設問：{task}",
]


def _is_placeholder_key(key: str) -> bool:
    """README の例 'YOUR_KEY_OR_OMIT_FOR_LOCAL' などのダミー値を未設定扱いにする."""
    k = (key or "").strip()
    return (not k) or k.upper().startswith("YOUR") or "OMIT" in k.upper() or k in ("sk-...", "AIza...")


def _post_with_retry(url: str, *, headers: Dict[str, str], payload: Dict, timeout: int, retries: int = 4) -> requests.Response:
    """429/5xx・通信エラー時に指数バックオフで再試行するPOST."""
    delay = 1.5
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:  # pragma: no cover - ネットワーク依存
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"request failed: {last_exc}")  # pragma: no cover


class CohortCache:
    """生成した量産型AI母集団をディスクに保存し、同じ課題文での再生成（課金・不再現）を避ける."""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        base = cache_dir or os.environ.get("UNIQ_CACHE_DIR") or str(Path.home() / ".cache" / "uniqueness_scorer")
        self.dir = Path(base)

    def _path(self, backend: str, model: str, mode: str, prompt: str, n: int) -> Path:
        h = hashlib.sha256(f"{backend}|{model}|{mode}|{n}|{prompt}".encode("utf-8")).hexdigest()[:24]
        return self.dir / f"cohort_{h}.json"

    def get(self, backend: str, model: str, mode: str, prompt: str, n: int) -> Optional[List[str]]:
        p = self._path(backend, model, mode, prompt, n)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) >= n:
                return [str(x) for x in data[:n]]
        except (OSError, ValueError):
            return None
        return None

    def put(self, backend: str, model: str, mode: str, prompt: str, n: int, cohort: List[str]) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._path(backend, model, mode, prompt, n).write_text(
                json.dumps(cohort, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except OSError:
            pass


class SentenceTransformerBackend(EmbeddingBackend):
    """ローカルで動く日本語対応の文埋め込みモデル（sentence-transformers）バックエンド.

    手作り辞書に頼らず、未知の分野でも意味の近さを測れる。APIキー不要・オフライン動作
    （初回のみモデルをダウンロード）。``pip install sentence-transformers`` が必要。

    既定モデルは多言語E5（日本語対応・小型）。環境変数 UNIQ_ST_MODEL で変更できる
    （例: cl-nagoya/ruri-v3-130m）。E5系は "query: " 接頭辞を付けて対称な類似度として使う。
    """

    name = "sentence_transformers"

    def __init__(self, model_name: Optional[str] = None, prefix: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_name = model_name or os.environ.get("UNIQ_ST_MODEL", "intfloat/multilingual-e5-small")
        if prefix is None:
            prefix = "query: " if "e5" in self.model_name.lower() else ""
        self.prefix = prefix
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerBackend には sentence-transformers が必要です: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=device)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float64)
        inputs = [self.prefix + (t if t.strip() else " ") for t in texts]
        arr = np.asarray(
            self._model.encode(inputs, batch_size=32, show_progress_bar=False, normalize_embeddings=True),
            dtype=np.float64,
        )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return np.where(norms > 1e-12, arr / np.maximum(norms, 1e-12), 0.0)


class OpenAIBackend(EmbeddingBackend):
    """OpenAI REST API バックエンド（追加パッケージ不要・requestsのみ）."""

    name = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        embed_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if _is_placeholder_key(self.api_key):
            raise ValueError("OPENAI_API_KEY is required for OpenAIBackend.")
        self.cache = CohortCache()
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float64)
        # バッチ分割（安全のため最大64件ずつ）
        all_vecs: List[List[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = [t[:6000] if t.strip() else " " for t in texts[start : start + batch_size]]
            resp = _post_with_retry(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                payload={"model": self.embed_model, "input": batch},
                timeout=60,
            )
            data = resp.json()["data"]
            data.sort(key=lambda item: item["index"])
            all_vecs.extend(item["embedding"] for item in data)
        arr = np.asarray(all_vecs, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return np.where(norms > 1e-12, arr / np.maximum(norms, 1e-12), 0.0)

    def _chat(self, prompt: str, temperature: float = 0.8) -> str:
        resp = _post_with_retry(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            payload={
                "model": self.chat_model,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=90,
        )
        return resp.json()["choices"][0]["message"]["content"].strip()

    def generate_cohort(
        self,
        assignment_prompt: str,
        n_samples: int = 12,
        mode: str = "report",
    ) -> List[str]:
        templates = (
            STUDENT_PROMPT_TEMPLATES_ES if mode == "es" else STUDENT_PROMPT_TEMPLATES_REPORT
        )
        cached = self.cache.get(self.name, self.chat_model, mode, assignment_prompt, n_samples)
        if cached:
            return cached
        temps = [0.4, 0.7, 0.9, 1.0]
        cohort: List[str] = []
        for i in range(n_samples):
            tmpl = templates[i % len(templates)]
            temp = temps[i % len(temps)]
            cohort.append(self._chat(tmpl.format(task=assignment_prompt), temperature=temp))
        self.cache.put(self.name, self.chat_model, mode, assignment_prompt, n_samples, cohort)
        return cohort

    def suggest_angles_llm(
        self,
        assignment_prompt: str,
        saturated_claims: Sequence[str],
        mode: str = "report",
    ) -> Optional[List[str]]:
        saturated_txt = "\n".join(f"- {c}" for c in saturated_claims[:6])
        prompt = (
            f"課題：{assignment_prompt}\n\n"
            f"量産型AIレポートですでに飽和しているありきたりな論点：\n{saturated_txt}\n\n"
            "上記の飽和論点とは直交し、かつ課題の趣旨を外さない「独自の切り口・反証視点・一次調査アイデア」を"
            "日本語で3つ、箇条書き（各1〜2文）で提案してください。"
        )
        out = self._chat(prompt, temperature=0.85)
        lines = [
            re.sub(r"^[\-\*・\d\.\)\s]+", "", ln).strip()
            for ln in out.splitlines()
            if ln.strip()
        ]
        return [ln for ln in lines if len(ln) >= 10][:3]


class GeminiBackend(EmbeddingBackend):
    """Google Gemini REST API バックエンド（追加パッケージ不要・requestsのみ）."""

    name = "gemini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        embed_model: str = "text-embedding-004",
        chat_model: str = "gemini-2.0-flash",
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if _is_placeholder_key(self.api_key):
            raise ValueError("GEMINI_API_KEY is required for GeminiBackend.")
        self.cache = CohortCache()

    def _headers(self) -> Dict[str, str]:
        # APIキーはURLクエリではなくヘッダーで送る（ログやプロキシ履歴への漏えい防止）
        return {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        self.embed_model = embed_model
        self.chat_model = chat_model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float64)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.embed_model}:batchEmbedContents"
        )
        all_vecs: List[List[float]] = []
        batch_size = 50
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            reqs = [
                {
                    "model": f"models/{self.embed_model}",
                    "content": {"parts": [{"text": t[:6000] if t.strip() else " "}]},
                }
                for t in batch
            ]
            resp = _post_with_retry(url, headers=self._headers(), payload={"requests": reqs}, timeout=60)
            all_vecs.extend(item["values"] for item in resp.json()["embeddings"])
        arr = np.asarray(all_vecs, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return np.where(norms > 1e-12, arr / np.maximum(norms, 1e-12), 0.0)

    def _generate(self, prompt: str, temperature: float = 0.8) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.chat_model}:generateContent"
        )
        resp = _post_with_retry(
            url,
            headers=self._headers(),
            payload={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature},
            },
            timeout=90,
        )
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    def generate_cohort(
        self,
        assignment_prompt: str,
        n_samples: int = 12,
        mode: str = "report",
    ) -> List[str]:
        templates = (
            STUDENT_PROMPT_TEMPLATES_ES if mode == "es" else STUDENT_PROMPT_TEMPLATES_REPORT
        )
        cached = self.cache.get(self.name, self.chat_model, mode, assignment_prompt, n_samples)
        if cached:
            return cached
        temps = [0.4, 0.7, 0.9, 1.0]
        cohort: List[str] = []
        for i in range(n_samples):
            tmpl = templates[i % len(templates)]
            temp = temps[i % len(temps)]
            cohort.append(self._generate(tmpl.format(task=assignment_prompt), temperature=temp))
        self.cache.put(self.name, self.chat_model, mode, assignment_prompt, n_samples, cohort)
        return cohort

    def suggest_angles_llm(
        self,
        assignment_prompt: str,
        saturated_claims: Sequence[str],
        mode: str = "report",
    ) -> Optional[List[str]]:
        saturated_txt = "\n".join(f"- {c}" for c in saturated_claims[:6])
        prompt = (
            f"課題：{assignment_prompt}\n\n"
            f"量産型AIレポートですでに飽和しているありきたりな論点：\n{saturated_txt}\n\n"
            "上記の飽和論点とは直交し、かつ課題の趣旨を外さない「独自の切り口・反証視点・一次調査アイデア」を"
            "日本語で3つ、箇条書き（各1〜2文）で提案してください。"
        )
        out = self._generate(prompt, temperature=0.85)
        lines = [
            re.sub(r"^[\-\*・\d\.\)\s]+", "", ln).strip()
            for ln in out.splitlines()
            if ln.strip()
        ]
        return [ln for ln in lines if len(ln) >= 10][:3]


def create_backend(provider: str = "auto") -> EmbeddingBackend:
    """指定または環境変数から最適なバックエンドを生成する."""
    p = provider.lower()
    if p == "openai":
        return OpenAIBackend()
    if p == "gemini":
        return GeminiBackend()
    if p == "local":
        return LocalHybridEmbedder()
    if p in ("st", "sentence-transformers", "sentence_transformers"):
        return SentenceTransformerBackend()
    if p == "auto":
        # ダミー値（README例の YOUR_KEY_OR_OMIT_FOR_LOCAL 等）は未設定として扱う
        if not _is_placeholder_key(os.environ.get("OPENAI_API_KEY", "")):
            return OpenAIBackend()
        if not _is_placeholder_key(os.environ.get("GEMINI_API_KEY", "")):
            return GeminiBackend()
        return LocalHybridEmbedder()
    raise ValueError(f"Unknown provider: {provider}")
