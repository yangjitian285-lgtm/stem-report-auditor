"""スタンドアロンHTML可視化レポート（2D PCA散布図＋主張単位ヒートマップ＋6指標バー）."""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .math_engine import ScoringResult
from .stem_engine import DataSeriesInput, StemScoringResult


STEM_METRIC_LABELS = {
    "dehomog_score": "脱・量産型AI考察（主張独自+ΔVS）",
    "anti_stem_slop_score": "脱・理系逃げフレーズ（反AI臭）",
    "bias_trend_consistency": "誤差方向（符号）＆系統トレンド整合",
    "quant_error_rigor": "オーダー評価・誤差伝播の定量性",
    "mechanism_depth": "物理メカニズム・近似破綻の深掘り",
    "beyond_restatement": "結果再述からの脱却度（なぜの説明）",
}


METRIC_LABELS = {
    "vendi_gain_score": "Vendi Score 増分 (ΔVS)",
    "mahalanobis_score": "マハラノビス直交逸脱度",
    "knn_escape_score": "最近傍クラスタ脱出度",
    "claim_novelty_score": "主張単位 独自性 (InfoGain)",
    "skeleton_antislop_score": "脱テンプレ骨格・反AI臭",
    "specificity_score": "一次情報・具体性・反論度",
}


def _build_pca_svg(
    cohort_points: Sequence[Dict[str, Any]],
    highlight_points: Sequence[Dict[str, Any]],
    explained: Sequence[float],
    width: int = 560,
    height: int = 340,
) -> str:
    """インラインSVGで2D PCAクラスタ図を描画する."""
    all_pts = list(cohort_points) + list(highlight_points)
    if not all_pts:
        return "<p>PCAデータなし</p>"

    xs = [p["x"] for p in all_pts]
    ys = [p["y"] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max((max_x - min_x) * 0.18, 0.1)
    pad_y = max((max_y - min_y) * 0.20, 0.1)
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    margin = 44

    def sx(x: float) -> float:
        return margin + (x - min_x) / max(max_x - min_x, 1e-9) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - (y - min_y) / max(max_y - min_y, 1e-9) * (height - 2 * margin)

    # 母集団重心と半径目安（楕円）
    cx = sum(p["x"] for p in cohort_points) / max(len(cohort_points), 1)
    cy = sum(p["y"] for p in cohort_points) / max(len(cohort_points), 1)
    rx = max(
        (sum((p["x"] - cx) ** 2 for p in cohort_points) / max(len(cohort_points), 1)) ** 0.5 * 1.6,
        0.04,
    )
    ry = max(
        (sum((p["y"] - cy) ** 2 for p in cohort_points) / max(len(cohort_points), 1)) ** 0.5 * 1.6,
        0.04,
    )
    svg_rx = abs(sx(cx + rx) - sx(cx))
    svg_ry = abs(sy(cy + ry) - sy(cy))

    lines: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;background:#0f172a;border-radius:12px;border:1px solid #334155;">',
        f'<line x1="{margin}" y1="{sy(0):.1f}" x2="{width-margin}" y2="{sy(0):.1f}" stroke="#1e293b" stroke-dasharray="4,4"/>',
        f'<line x1="{sx(0):.1f}" y1="{margin}" x2="{sx(0):.1f}" y2="{height-margin}" stroke="#1e293b" stroke-dasharray="4,4"/>',
        f'<ellipse cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" rx="{svg_rx:.1f}" ry="{svg_ry:.1f}" fill="rgba(239,68,68,0.12)" stroke="#ef4444" stroke-dasharray="5,4" stroke-width="1.5"/>',
        f'<text x="{sx(cx):.1f}" y="{sy(cy) - svg_ry - 6:.1f}" fill="#fca5a5" font-size="11" text-anchor="middle">量産型AI密集エリア</text>',
    ]

    for p in cohort_points:
        lines.append(
            f'<circle cx="{sx(p["x"]):.1f}" cy="{sy(p["y"]):.1f}" r="5" fill="#94a3b8" opacity="0.72">'
            f'<title>{html.escape(str(p.get("id", "AI")))}</title></circle>'
        )

    for hp in highlight_points:
        color = hp.get("color", "#22c55e")
        label = html.escape(str(hp.get("label", hp.get("id", "DRAFT"))))
        px, py = sx(hp["x"]), sy(hp["y"])
        lines.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="{color}" stroke="#ffffff" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{px + 10:.1f}" y="{py + 4:.1f}" fill="#f8fafc" font-size="11.5" font-weight="bold">{label}</text>'
        )

    ev1 = explained[0] * 100 if len(explained) > 0 else 0.0
    ev2 = explained[1] * 100 if len(explained) > 1 else 0.0
    lines.append(
        f'<text x="{width/2:.1f}" y="{height-12}" fill="#94a3b8" font-size="11" text-anchor="middle">PC1 ({ev1:.1f}%)</text>'
    )
    lines.append(
        f'<text x="14" y="{height/2:.1f}" fill="#94a3b8" font-size="11" text-anchor="middle" transform="rotate(-90 14 {height/2:.1f})">PC2 ({ev2:.1f}%)</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def _metric_bars_html(metrics: Dict[str, float]) -> str:
    rows: List[str] = []
    for key, val in metrics.items():
        label = METRIC_LABELS.get(key, key)
        color = "#22c55e" if val >= 65 else ("#eab308" if val >= 40 else "#ef4444")
        rows.append(
            f'<div style="margin-bottom:10px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;">'
            f'<span>{html.escape(label)}</span><strong>{val:.1f} / 100</strong></div>'
            f'<div style="background:#1e293b;height:9px;border-radius:999px;overflow:hidden;">'
            f'<div style="width:{min(100.0, max(0.0, val)):.1f}%;height:100%;background:{color};border-radius:999px;"></div>'
            f"</div></div>"
        )
    return "\n".join(rows)


def _claims_heatmap_html(result: ScoringResult) -> str:
    blocks: List[str] = []
    for c in result.claims:
        pct = c.cohort_overlap_rate * 100.0
        if c.status == "RED":
            bg, border, badge = "rgba(239,68,68,0.16)", "#ef4444", "量産型丸被り"
        elif c.status == "YELLOW":
            bg, border, badge = "rgba(234,179,8,0.16)", "#eab308", "要注意"
        else:
            bg, border, badge = "rgba(34,197,94,0.15)", "#22c55e", "独自視点"
        blocks.append(
            f'<div style="background:{bg};border-left:4px solid {border};padding:10px 12px;margin-bottom:8px;border-radius:6px;">'
            f'<div style="font-size:12px;color:#cbd5e1;margin-bottom:4px;">'
            f'<strong>主張 #{c.index}</strong> ｜ <span style="color:{border};font-weight:bold;">{badge}（AI被り推定 {pct:.0f}%）</span>'
            f'</div>'
            f'<div style="font-size:14px;color:#f8fafc;line-height:1.55;">{html.escape(c.text)}</div>'
            f"</div>"
        )
    return "\n".join(blocks)


def render_single_report_html(
    title: str,
    assignment_prompt: str,
    result: ScoringResult,
) -> str:
    """単一ドラフトの詳細診断HTMLを生成する."""
    dp = result.pca_projection.get("draft_point", {})
    dp_colored = {
        **dp,
        "label": f"あなたの原稿 ({result.final_score:.1f}点)",
        "color": "#38bdf8",
    }
    svg = _build_pca_svg(
        result.pca_projection.get("cohort_points", []),
        [dp_colored] if dp else [],
        result.pca_projection.get("explained_variance", [0.0, 0.0]),
    )
    suggestions_li = "\n".join(
        f"<li style='margin-bottom:8px;line-height:1.6;'>{html.escape(s)}</li>"
        for s in result.orthogonal_suggestions
    )
    slop_str = " / ".join(result.slop_hits) if result.slop_hits else "検出なし"
    q = getattr(result, "quality_scores", {}) or {}
    q_labels = {"task_relevance": "課題適合", "form_adequacy": "文量・体裁", "coherence": "論理のつながり"}
    quality_bars = _metric_bars_html({q_labels[k]: v for k, v in q.items() if k in q_labels}) if q else ""
    flags_li = "".join(
        f"<li style='margin-bottom:6px;'>{html.escape(f)}</li>" for f in (getattr(result, "quality_flags", []) or [])
    )
    verify_li = "".join(
        f"<li style='margin-bottom:6px;'><strong>{html.escape(v['reason'])}</strong><br/>"
        f"<span style='color:#cbd5e1;'>{html.escape(v['text'][:120])}</span></li>"
        for v in (getattr(result, "verification_needed", []) or [])
    )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>{html.escape(title)} - 脱・量産型AI 独自性スコアラー</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Sans", sans-serif; background:#020617; color:#e2e8f0; margin:0; padding:24px; }}
  .wrap {{ max-width:1040px; margin:0 auto; }}
  .card {{ background:#0f172a; border:1px solid #1e293b; border-radius:14px; padding:20px; margin-bottom:18px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  @media (max-width:820px) {{ .grid {{ grid-template-columns:1fr; }} }}
  .score-big {{ font-size:52px; font-weight:800; color:#38bdf8; line-height:1.1; }}
  .pill {{ display:inline-block; padding:4px 12px; border-radius:999px; background:#1e293b; font-size:13px; margin-right:8px; }}
  h1,h2,h3 {{ margin-top:0; color:#f8fafc; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1 style="font-size:22px;">「脱・量産型AI」独自性診断レポート：{html.escape(title)}</h1>
    <p style="color:#94a3b8;font-size:14px;">課題：{html.escape(assignment_prompt)}</p>
    <div style="display:flex;align-items:center;gap:28px;flex-wrap:wrap;margin-top:14px;">
      <div>
        <div class="score-big">{result.final_score:.1f}</div>
        <div style="font-size:13px;color:#94a3b8;">総合独自性スコア (0〜100)</div>
      </div>
      <div>
        <div style="font-size:20px;font-weight:700;margin-bottom:6px;">判定：{html.escape(result.grade)}</div>
        <span class="pill">純粋独自性：{result.raw_originality:.1f}</span>
        <span class="pill">課題適合ゲート G：{result.relevance_gate:.2f}</span>
        <span class="pill">限界Vendi増分 ΔVS：+{result.raw_stats.get("delta_vendi_score", 0):.3f}</span>
      </div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2 style="font-size:17px;">6軸数理メトリクス</h2>
      {_metric_bars_html(result.metrics)}
      <p style="font-size:12px;color:#94a3b8;margin-top:12px;">
        骨格：{' → '.join(result.skeleton)}<br/>
        量産型最頻骨格：{' → '.join(result.cohort_dominant_skeleton)}<br/>
        検出AI常套句：{html.escape(slop_str)}
      </p>
    </div>
    <div class="card">
      <h2 style="font-size:17px;">2D PCA 意味多様体マップ</h2>
      {svg}
      <p style="font-size:12px;color:#94a3b8;">赤い破線エリアが「他学生の量産型AI」密集地帯。外側に抜け出しつつ課題適合ゲートGが高い状態が理想です。</p>
    </div>
  </div>

  <div class="card">
    <h2 style="font-size:17px;">主張（Claim）単位ヒートマップ（どこが量産型AIと丸被りか）</h2>
    {_claims_heatmap_html(result)}
  </div>

  <div class="card">
    <h2 style="font-size:17px;">品質（独自性とは別軸・合計点には含めない）</h2>
    {quality_bars}
    {"<ul style='padding-left:20px;margin:8px 0 0;font-size:13.5px;'>" + flags_li + "</ul>" if flags_li else ""}
    <p style="font-size:12px;color:#94a3b8;">独自性が高くても、課題に答えていない・論理が飛ぶ・事実が誤っている文章は良いレポートではありません。事実の正確さはこのツールでは判定できません。</p>
  </div>

  {"<div class='card'><h2 style='font-size:17px;'>根拠の確認が必要な記述</h2><ul style='padding-left:20px;margin:0;font-size:13.5px;'>" + verify_li + "</ul><p style='font-size:12px;color:#94a3b8;'>具体的な数値や調査は独自性の加点対象ですが、ツールは真偽を確認できません。実在しないデータを書くことは捏造です。</p></div>" if verify_li else ""}

  <div class="card">
    <h2 style="font-size:17px;">脱・量産型ピボット提案</h2>
    <ul style="padding-left:20px;margin:0;">{suggestions_li}</ul>
  </div>
</div>
</body>
</html>"""


def render_benchmark_html(
    sections: Sequence[Dict[str, Any]],
) -> str:
    """4群ベンチマーク結果（複数課題）を一覧比較できるHTMLレポートを生成する."""
    section_blocks: List[str] = []
    group_colors = {
        "A": "#ef4444",
        "B": "#f97316",
        "C": "#a855f7",
        "D": "#22c55e",
    }

    for sec in sections:
        rows_html: List[str] = []
        highlights: List[Dict[str, Any]] = []
        cohort_pts = sec["pca_cohort_points"]
        explained = sec["pca_explained"]

        for item in sec["items"]:
            res: ScoringResult = item["result"]
            grp = item["group_key"]
            col = group_colors.get(grp, "#38bdf8")
            pt = item["pca_point"]
            highlights.append(
                {
                    "x": pt["x"],
                    "y": pt["y"],
                    "label": f"{grp}: {res.final_score:.1f}",
                    "color": col,
                }
            )
            rows_html.append(
                f"<tr>"
                f"<td><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{col};margin-right:6px;'></span>"
                f"<strong>{html.escape(item['name'])}</strong></td>"
                f"<td style='font-size:18px;font-weight:800;color:{col};'>{res.final_score:.1f}</td>"
                f"<td>{res.raw_originality:.1f}</td>"
                f"<td>{res.relevance_gate:.2f}</td>"
                f"<td>+{res.raw_stats['delta_vendi_score']:.3f}</td>"
                f"<td>{res.raw_stats['mahalanobis_z_score']:+.2f}σ</td>"
                f"<td>{res.raw_stats['mean_claim_overlap_rate']*100:.0f}%</td>"
                f"<td>{html.escape(res.grade)}</td>"
                f"</tr>"
            )

        svg = _build_pca_svg(cohort_pts, highlights, explained)
        # Group D と Group B のヒートマップ比較
        d_item = next((it for it in sec["items"] if it["group_key"] == "D"), None)
        b_item = next((it for it in sec["items"] if it["group_key"] == "B"), None)
        heatmap_compare = ""
        if b_item and d_item:
            heatmap_compare = (
                f'<div class="grid" style="margin-top:16px;">'
                f'<div><h3 style="font-size:15px;color:#fb923c;">Group B（言い換えリライト版）の主張ヒートマップ</h3>{_claims_heatmap_html(b_item["result"])}</div>'
                f'<div><h3 style="font-size:15px;color:#4ade80;">Group D（独自視点レポート）の主張ヒートマップ</h3>{_claims_heatmap_html(d_item["result"])}</div>'
                f"</div>"
            )

        section_blocks.append(
            f'<div class="card">'
            f'<h2 style="font-size:19px;">{html.escape(sec["task_title"])}</h2>'
            f'<p style="color:#94a3b8;font-size:13.5px;">課題：{html.escape(sec["prompt"])}</p>'
            f'<div class="grid" style="align-items:start;">'
            f'<div style="overflow-x:auto;"><table><thead><tr>'
            f'<th>テスト群</th><th>総合点</th><th>純独自</th><th>適合G</th><th>ΔVS</th><th>Maha-Z</th><th>主張被り</th><th>判定</th>'
            f'</tr></thead><tbody>{"".join(rows_html)}</tbody></table></div>'
            f'<div>{svg}</div>'
            f"</div>"
            f"{heatmap_compare}"
            f"</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>「脱・量産型AI」独自性スコアラー 4群ベンチマーク実証レポート</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Sans", sans-serif; background:#020617; color:#e2e8f0; margin:0; padding:24px; }}
  .wrap {{ max-width:1180px; margin:0 auto; }}
  .card {{ background:#0f172a; border:1px solid #1e293b; border-radius:14px; padding:20px; margin-bottom:22px; }}
  .grid {{ display:grid; grid-template-columns:1.2fr 1fr; gap:18px; }}
  @media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:9px 8px; border-bottom:1px solid #1e293b; text-align:left; }}
  th {{ color:#94a3b8; font-weight:600; }}
  h1,h2,h3 {{ margin-top:0; color:#f8fafc; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1 style="font-size:24px;">「脱・量産型AI」独自性スコアラー：4群ベンチマーク実証結果</h1>
    <p style="color:#94a3b8;font-size:14px;margin-bottom:0;">
      検証目標：<strong>Group D（独自視点）</strong> が最高評価となり、<strong>Group A（素の量産AI）</strong>・<strong>Group B（AI検知回避の言い換え版）</strong>・<strong>Group C（課題ズレの珍文）</strong> のすべてを明確に分離できるかを確認。
    </p>
  </div>
  {"".join(section_blocks)}
</div>
</body>
</html>"""


def _build_trend_svg(
    data: DataSeriesInput,
    extrap_zero: Any,
    r2: Any,
    width: int = 500,
    height: int = 270,
) -> str:
    """実測値 vs 独立変数（例: a vs cos²θ）の系統トレンド＆外挿SVGプロット."""
    if not data.independent_var or len(data.independent_var) != len(data.observed):
        return ""
    xs = [0.0] + list(data.independent_var)
    ys = list(data.observed)
    ref_val = float(data.reference) if not isinstance(data.reference, (list, tuple)) else None
    if ref_val is not None:
        ys.append(ref_val)
    if extrap_zero is not None:
        ys.append(float(extrap_zero))

    min_x, max_x = 0.0, max(xs) * 1.12
    min_y, max_y = min(ys), max(ys)
    pad_y = max((max_y - min_y) * 0.22, abs(max_y) * 0.001, 1e-6)
    min_y -= pad_y
    max_y += pad_y
    m_l, m_r, m_t, m_b = 64, 24, 26, 40

    def sx(v: float) -> float:
        return m_l + (v - min_x) / max(max_x - min_x, 1e-12) * (width - m_l - m_r)

    def sy(v: float) -> float:
        return height - m_b - (v - min_y) / max(max_y - min_y, 1e-12) * (height - m_t - m_b)

    lines = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;background:#0f172a;border-radius:12px;border:1px solid #334155;">',
        f'<line x1="{m_l}" y1="{height-m_b}" x2="{width-m_r}" y2="{height-m_b}" stroke="#475569"/>',
        f'<line x1="{m_l}" y1="{m_t}" x2="{m_l}" y2="{height-m_b}" stroke="#475569"/>',
    ]
    if ref_val is not None:
        ry = sy(ref_val)
        lines.append(
            f'<line x1="{m_l}" y1="{ry:.1f}" x2="{width-m_r}" y2="{ry:.1f}" stroke="#22c55e" stroke-dasharray="6,4" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{width-m_r-4}" y="{ry-5:.1f}" fill="#4ade80" font-size="11" text-anchor="end">文献/理論値 {ref_val:.5g}</text>'
        )

    # 回帰直線
    if extrap_zero is not None:
        import numpy as _np
        slope, intercept = _np.polyfit(data.independent_var, data.observed, 1)
        x_end = max(data.independent_var)
        y_end = slope * x_end + intercept
        lines.append(
            f'<line x1="{sx(0.0):.1f}" y1="{sy(float(intercept)):.1f}" x2="{sx(x_end):.1f}" y2="{sy(float(y_end)):.1f}" stroke="#38bdf8" stroke-width="2" stroke-dasharray="4,3"/>'
        )
        lines.append(
            f'<circle cx="{sx(0.0):.1f}" cy="{sy(float(intercept)):.1f}" r="6" fill="#38bdf8" stroke="#fff" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{sx(0.0)+8:.1f}" y="{sy(float(intercept))+14:.1f}" fill="#38bdf8" font-size="11" font-weight="bold">外挿値 {float(intercept):.5g} (R²={r2})</text>'
        )

    for xv, yv in zip(data.independent_var, data.observed):
        lines.append(
            f'<circle cx="{sx(xv):.1f}" cy="{sy(yv):.1f}" r="5.5" fill="#f97316" stroke="#fff" stroke-width="1.2"/>'
        )

    lines.append(
        f'<text x="{(m_l+width-m_r)/2:.1f}" y="{height-10}" fill="#94a3b8" font-size="11" text-anchor="middle">{html.escape(data.independent_name or "独立変数")}</text>'
    )
    lines.append(
        f'<text x="15" y="{(m_t+height-m_b)/2:.1f}" fill="#94a3b8" font-size="11" text-anchor="middle" transform="rotate(-90 15 {(m_t+height-m_b)/2:.1f})">{html.escape(data.name)} ({html.escape(data.unit)})</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def render_stem_benchmark_html(sections: Sequence[Dict[str, Any]]) -> str:
    """理系計算・実験レポートの実戦検証HTMLレポートを生成する."""
    blocks: List[str] = []
    colors = {"AI": "#ef4444", "REAL": "#eab308", "OFF": "#a855f7", "S_PLUS": "#22c55e"}

    for sec in sections:
        rows: List[str] = []
        for it in sec["items"]:
            r: StemScoringResult = it["result"]
            c = colors.get(it["tag"], "#38bdf8")
            rows.append(
                f"<tr>"
                f"<td><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{c};margin-right:6px;'></span><strong>{html.escape(it['name'])}</strong></td>"
                f"<td style='font-size:18px;font-weight:800;color:{c};'>{r.final_score:.1f}</td>"
                f"<td>{r.metrics['dehomog_score']:.0f}</td>"
                f"<td>{r.metrics['anti_stem_slop_score']:.0f}</td>"
                f"<td>{r.metrics['bias_trend_consistency']:.0f}</td>"
                f"<td>{r.metrics['quant_error_rigor']:.0f}</td>"
                f"<td>{r.metrics['mechanism_depth']:.0f}</td>"
                f"<td>{r.metrics['beyond_restatement']:.0f}</td>"
                f"<td>{r.relevance_gate:.2f}</td>"
                f"<td>{html.escape(r.grade)}</td>"
                f"</tr>"
            )

        real_item = next((it for it in sec["items"] if it["tag"] == "REAL"), None)
        audit_html = ""
        trend_svg = ""
        if real_item:
            na = real_item["result"].numeric_audit
            if na.has_data:
                findings_li = "".join(f"<li>{html.escape(f)}</li>" for f in na.findings)
                audit_html = f"<div style='background:#1e293b;padding:12px 14px;border-radius:10px;margin:12px 0;'><strong>数値データ自動監査（NumPy解析）</strong><ul style='margin:6px 0 0 18px;font-size:13px;line-height:1.6;'>{findings_li}</ul></div>"
                if sec.get("data_series") is not None:
                    trend_svg = _build_trend_svg(sec["data_series"], na.extrapolated_at_zero, na.trend_r2)
            sugg_li = "".join(
                f"<li style='margin-bottom:6px;'>{html.escape(s)}</li>"
                for s in real_item["result"].scientific_suggestions
            )
            real_sugg_html = f"<div style='margin-top:12px;'><h3 style='font-size:15px;color:#facc15;'>実提出レポートに対する自動ピボット提案</h3><ul style='font-size:13.5px;line-height:1.6;padding-left:20px;'>{sugg_li}</ul></div>"
        else:
            real_sugg_html = ""

        blocks.append(
            f'<div class="card">'
            f'<h2 style="font-size:19px;">{html.escape(sec["title"])}</h2>'
            f'<p style="color:#94a3b8;font-size:13px;">課題：{html.escape(sec["prompt"])}</p>'
            f'{audit_html}'
            f'<div style="overflow-x:auto;"><table><thead><tr>'
            f'<th>バージョン</th><th>総合</th><th>脱量産</th><th>反逃句</th><th>符号整合</th><th>定量誤差</th><th>機構深掘</th><th>脱再述</th><th>適合G</th><th>判定</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            f'<div class="grid" style="margin-top:14px;">'
            f'<div>{real_sugg_html}</div>'
            f'<div>{trend_svg}</div>'
            f'</div></div>'
        )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>理系計算・実験レポート 脱量産型考察スコアラー 実戦検証</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Sans", sans-serif; background:#020617; color:#e2e8f0; margin:0; padding:24px; }}
  .wrap {{ max-width:1200px; margin:0 auto; }}
  .card {{ background:#0f172a; border:1px solid #1e293b; border-radius:14px; padding:20px; margin-bottom:22px; }}
  .grid {{ display:grid; grid-template-columns:1.15fr 1fr; gap:18px; align-items:start; }}
  @media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:8px 7px; border-bottom:1px solid #1e293b; text-align:left; }}
  th {{ color:#94a3b8; }}
  h1,h2,h3 {{ margin-top:0; color:#f8fafc; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1 style="font-size:23px;">理系計算・実験レポート特化「脱・量産型考察」スコアラー 実戦検証</h1>
    <p style="color:#94a3b8;font-size:14px;margin-bottom:0;">
      実提出レポート（結晶構造解析・ラジカル共重合）の数値データと考察を用い、<strong>量産型AI考察</strong>・<strong>実提出レポート</strong>・<strong>論点ズレ</strong>・<strong>Sランク改善考察</strong>の4つを6軸数理モデルで比較検証。
    </p>
  </div>
  {"".join(blocks)}
</div>
</body>
</html>"""


def render_full_audit_html(report: Any) -> str:
    """1ファイル丸投げ決定論的監査＋理系考察スコアの統合HTMLダッシュボードを生成する."""
    sev_style = {
        "CRITICAL": ("rgba(239,68,68,0.18)", "#ef4444", "CRITICAL"),
        "WARNING": ("rgba(234,179,8,0.16)", "#eab308", "WARNING"),
        "INFO": ("rgba(56,189,248,0.15)", "#38bdf8", "INFO"),
        "PASS": ("rgba(34,197,94,0.15)", "#22c55e", "PASS"),
    }

    finding_rows: List[str] = []
    for f in report.findings:
        bg, col, label = sev_style.get(f.severity, ("#1e293b", "#94a3b8", f.severity))
        sugg_html = (
            f"<div style='color:#94a3b8;font-size:12px;margin-top:3px;'>→ {html.escape(f.suggestion)}</div>"
            if f.suggestion
            else ""
        )
        finding_rows.append(
            f"<div style='background:{bg};border-left:4px solid {col};padding:10px 12px;margin-bottom:8px;border-radius:6px;'>"
            f"<div style='font-size:12px;margin-bottom:3px;'><strong style='color:{col};'>[{label}]</strong> "
            f"<span style='color:#cbd5e1;'>{html.escape(f.category)} ｜ {html.escape(f.location)}</span></div>"
            f"<div style='font-size:13.5px;color:#f8fafc;'>{html.escape(f.message)}</div>"
            f"{sugg_html}</div>"
        )

    # 6軸バー（理系考察）
    disc = report.stem_discussion_result
    stem_bars: List[str] = []
    disc_metrics = disc.metrics if disc is not None else {}
    disc_grade = disc.grade if disc is not None else (getattr(report, "discussion_note", "") or "対象外")

    def _fmt(v: Optional[float]) -> str:
        return "N/A" if v is None else f"{v:.1f}"

    if disc is None:
        stem_bars.append(
            f"<p style='color:#94a3b8;font-size:13px;'>{html.escape(getattr(report, 'discussion_note', '') or '考察スコアは算出していません。')}</p>"
        )
    for k, v in disc_metrics.items():
        lbl = STEM_METRIC_LABELS.get(k, k)
        col = "#22c55e" if v >= 65 else ("#eab308" if v >= 40 else "#ef4444")
        stem_bars.append(
            f"<div style='margin-bottom:9px;'>"
            f"<div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;'>"
            f"<span>{html.escape(lbl)}</span><strong>{v:.1f} / 100</strong></div>"
            f"<div style='background:#1e293b;height:9px;border-radius:999px;overflow:hidden;'>"
            f"<div style='width:{min(100.0, max(0.0, v)):.1f}%;height:100%;background:{col};'></div>"
            f"</div></div>"
        )

    # 自動発見データ系列のトレンドSVG（あれば描画）
    trend_svg = ""
    if report.auto_discovered_series and report.auto_discovered_audits:
        ds_dict = report.auto_discovered_series[0]
        na = report.auto_discovered_audits[0]
        ds_obj = DataSeriesInput(**ds_dict)
        trend_svg = _build_trend_svg(ds_obj, na.extrapolated_at_zero, na.trend_r2)

    checklist_li = "".join(
        f"<li style='margin-bottom:7px;line-height:1.55;'>{html.escape(c)}</li>"
        for c in report.priority_checklist
    )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>理系レポート決定論的監査レポート - {html.escape(report.source_path)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Hiragino Sans", sans-serif; background:#020617; color:#e2e8f0; margin:0; padding:24px; }}
  .wrap {{ max-width:1120px; margin:0 auto; }}
  .card {{ background:#0f172a; border:1px solid #1e293b; border-radius:14px; padding:20px; margin-bottom:18px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  @media (max-width:860px) {{ .grid {{ grid-template-columns:1fr; }} }}
  .kpi {{ background:#1e293b; padding:14px 18px; border-radius:12px; min-width:160px; }}
  .kpi-num {{ font-size:34px; font-weight:800; line-height:1.1; }}
  h1,h2 {{ margin-top:0; color:#f8fafc; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1 style="font-size:21px;">理系レポート決定論的監査＋脱・量産型考察レポート</h1>
    <p style="color:#94a3b8;font-size:13px;">対象ファイル：{html.escape(report.source_path)}（抽出テーブル数：{report.extracted_tables_count}）</p>
    <div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:12px;">
      <div class="kpi"><div class="kpi-num" style="color:#38bdf8;">{_fmt(report.overall_score)}</div><div style="font-size:12px;color:#94a3b8;">総合監査スコア</div></div>
      <div class="kpi"><div class="kpi-num" style="color:#22c55e;">{_fmt(report.deterministic_score)}</div><div style="font-size:12px;color:#94a3b8;">決定論的整合点（検算/誤記/矛盾）</div></div>
      <div class="kpi"><div class="kpi-num" style="color:#facc15;">{_fmt(report.discussion_score)}</div><div style="font-size:12px;color:#94a3b8;">理系考察 脱量産スコア</div></div>
      <div class="kpi" style="flex:1;"><div style="font-size:17px;font-weight:700;margin-top:4px;">{html.escape(report.overall_grade)}</div><div style="font-size:12px;color:#94a3b8;margin-top:4px;">考察判定：{html.escape(disc_grade)}</div></div>
    </div>
  </div>

  <div class="card">
    <h2 style="font-size:17px;">提出前 優先修正チェックリスト</h2>
    <ol style="padding-left:20px;margin:0;font-size:13.5px;">{checklist_li}</ol>
  </div>

  <div class="grid">
    <div class="card">
      <h2 style="font-size:17px;">理系考察 6軸スコア</h2>
      {"".join(stem_bars)}
    </div>
    <div class="card">
      <h2 style="font-size:17px;">自動発見データ・系統トレンド解析</h2>
      {trend_svg if trend_svg else "<p style='color:#94a3b8;font-size:13px;'>外挿可能な独立変数付きデータ系列は未検出（系統バイアス等は下記監査詳細を参照）。</p>"}
    </div>
  </div>

  <div class="card">
    <h2 style="font-size:17px;">決定論的監査ログ（検算・表間矛盾・有効数字・誤記・テンプレ残存）</h2>
    {"".join(finding_rows) if finding_rows else "<p style='color:#4ade80;'>重大な矛盾・誤記は検出されませんでした。</p>"}
  </div>
</div>
</body>
</html>"""
