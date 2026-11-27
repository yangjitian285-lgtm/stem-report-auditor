# STEM Report Auditor & De-Homogenizer（「脱・量産型AI」理系レポート数理監査＆独自性スコアラー）

理系の実験・計算レポート（Word `.docx` / PDF / テキスト）や論述・就活ESに対して、**LLMの暗算ミス・自己バイアスを排した決定論的な数理監査**と、**「量産型AIとの丸被り」からの脱却支援**を行う軽量Pythonツール＆MCPサーバーです。

外部の重いMLフレームワークやMCP SDKに依存せず、**Python標準ライブラリ + `numpy` + `requests`** だけで動作します。

---

## なぜ「素のLLMに採点させる」のと違うのか？

1. **検算・表間矛盾・線形外挿にハルシネーションがゼロ**
   - 表の全行に対する三角関数・格子定数・分散度（$M_w/M_n$）・平均値の検算、離れた表同士の同一試料数値矛盾、小数桁の不統一、系統バイアスと線形外挿をPython/NumPyで確定的に実行します。
2. **LLM自身の「量産型優等生バイアス」を外部から止める**
   - 素のLLMは「概ね一致」「読み取り誤差・温度変化が考えられる」「測定回数を増やす」といった量産型考察を高く評価しがちです。本ツールは外部の数理基準（限界Vendi Score増分 $\Delta\text{VS}$、正則化マハラノビス距離、主張単位被り率、理系逃げフレーズ減点、系統バイアスとランダム誤差の物理矛盾検知）で評価します。
3. **Claude Code 等のAIとMCPで分業できる（Linter＋自動検算機）**
   - 素のLLMにコードレビューさせるのではなく「コンパイラ＋Linter＋テストを持たせて直す」のと同じ構造を、理系レポート・論述執筆に適用できます。

---

## 主な機能（MCP 6ツール＋CLI）

| ツール / コマンド | 役割 |
| :--- | :--- |
| **`audit_stem_file`** (`cli.py audit`) | Word(`.docx`)/PDF/テキストを1ファイル丸投げで完全監査（表内検算・表間矛盾・有効数字・誤記・テンプレ消し忘れ・系統バイアス＆外挿自動発見・理系考察6軸スコア） |
| **`patch_docx_safe`** | 元 `.docx` を絶対に上書きせず、図表・書式・XML名前空間を保持したコピー版（`*_revised.docx`）へ承認済み修正を安全反映＆Before/After自動再監査 |
| **`score_stem_report`** (`cli.py stem`) | 理系考察テキスト＋実測数値データ（実測値・基準値・独立変数）の6軸診断 |
| **`score_report_uniqueness`** (`cli.py score`) | 自由論述レポート・就活ESの6軸数理採点（$\Delta\text{VS}$・LOOマハラノビス・k-NN脱出・主張被り率・骨格/AI臭・具体性＋課題適合ゲート $G^\alpha$） |
| **`diagnose_claims_heatmap`** | 主張（文）単位の量産型AI被り率（RED / YELLOW / GREEN）と差し替え提案を抽出 |
| **`compare_draft_revisions`** | 改訂前後のスコア・$\Delta\text{VS}$・主張被り率の変化幅を定量比較 |

---

## 数理モデルの概要

### 一般論述・就活ESモード（`math_engine.py`）
課題文 $q$ に対する仮想・量産型AI母集団 $\mathcal{R}(q)=\{y_1,\dots,y_M\}$ に対し、以下を算出します：

$$\text{UniquenessScore}(x \mid q) = 100 \times G(x \mid q)^{\alpha} \times \Bigl(w_1 S_{\text{vendi}} + w_2 S_{\text{maha}} + w_3 S_{\text{knn}} + w_4 S_{\text{claim}} + w_5 S_{\text{skeleton}} + w_6 S_{\text{spec}}\Bigr)$$

- **限界Vendi Score増分**：$\Delta \text{VS}(x \mid \mathcal{R}) = \text{VS}(\mathcal{R} \cup \{x\}) - \text{VS}(\mathcal{R})$（Friedman & Dieng, TMLR 2023）
- **LOO正則化マハラノビス距離**：Ledoit-Wolf型収縮＋Woodbury恒等式により高次元小標本でも厳密かつ高速に標本外校正
- **品質ゲート $G(x \mid q)$**：課題から逸脱した珍文が高得点になる距離ハックを遮断

### 理系計算・実験レポートモード（`stem_engine.py` / `stem_auditor.py`）
1. 脱・量産型AI考察（主張被り率＋$\Delta\text{VS}$）
2. 脱・理系逃げフレーズ（未定量の「読み取り誤差・環境要因・概ね一致・測定回数増加」を減点）
3. 誤差方向（バイアス符号）＆系統トレンド整合（全点が片側バイアスなのにランダム誤差のみで説明する物理矛盾を自動検知）
4. オーダー評価・誤差伝播の定量性
5. 物理メカニズム・理論近似破綻の深掘り
6. 結果再述からの脱却度

---

## クイックスタート

### インストール
```bash
pip install -r requirements.txt
# PDFファイルを直接監査したい場合は追加で:
# pip install pypdf
```

### 1. デモ＆ベンチマーク実行（APIキー不要）
```bash
python demo_stem.py
python benchmark.py
```

### 2. CLIで理系Wordレポートを1ファイル完全監査
```bash
python cli.py audit path/to/your_report.docx --html-out audit_report.html
```

### 3. CLIで論述・就活ESを採点（OpenAI / Gemini API またはローカル母集団）
```bash
# OpenAI / Gemini API を使う場合（仮想・量産型AI母集団を自動生成）
export GEMINI_API_KEY="YOUR_API_KEY"
python cli.py score --task task.txt --draft draft.md --mode report --provider gemini --html-out report.html

# APIキーなし・オフラインの場合
python cli.py score --task task.txt --draft draft.md --cohort cohort.json --provider local --html-out report.html
```

### 4. Claude Code へ MCP サーバーとして登録（推奨）
```bash
claude mcp add -s user uniqueness-scorer -- python "/path/to/this/repo/cli.py" mcp
```
登録後、Claude Code 上で以下のように依頼するだけで、**監査 → ユーザー確認 → 安全なコピー版 `.docx` 反映 → 再監査**の対話ループを回せます：
> `report.docx` を `audit_stem_file` で監査して。見つかったミスや考察改善案を私に確認しながら、OKを出した箇所だけ `patch_docx_safe` でコピー版へ反映して再監査して。

---

## ライセンス
MIT License — 詳細は [LICENSE](LICENSE) を参照してください。
