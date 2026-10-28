"""
Clinical PDF Assessment Report Generator utilizing ReportLab.
Produces professional, printable medical summary sheets including indicators,
model probabilities, SHAP explanations, and mandatory clinical disclaimers.
"""

import io
from datetime import datetime
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from src.config import FEATURE_METADATA, MODEL_VERSION, CLINICAL_DISCLAIMER


def generate_patient_pdf_report(
    patient_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    explanation_result: Dict[str, Any] = None,
    patient_id: str = "SYNTH-PATIENT-001"
) -> bytes:
    """
    Generate a formatted multi-section clinical evaluation PDF report in memory.
    Returns bytes buffer ready for HTTP streaming or local disk storage.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a")
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b")
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#334155")
    )
    badge_high = ParagraphStyle(
        "BadgeHigh",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#dc2626")
    )
    badge_low = ParagraphStyle(
        "BadgeLow",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#16a34a")
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748b")
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("CARDIOVASCULAR RISK EVALUATION REPORT", title_style))
    story.append(Paragraph(
        f"Decision-Support Prototype &bull; Model Version: {MODEL_VERSION} &bull; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        subtitle_style
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=12))

    # 2. Executive Assessment Summary Box
    is_positive = prediction_result.get("prediction") == 1
    badge_style = badge_high if is_positive else badge_low
    diag_text = "ELEVATED RISK DETECTED (POSITIVE)" if is_positive else "LOW RISK / NORMAL BASELINE (NEGATIVE)"
    prob_str = f"{prediction_result.get('confidence_percentage', 0.0):.1f}%"
    risk_tier = prediction_result.get("risk_level", "Unknown")

    summary_data = [
        [
            Paragraph("<b>Diagnostic Classification:</b>", body_style),
            Paragraph(diag_text, badge_style),
            Paragraph("<b>Calibrated Probability:</b>", body_style),
            Paragraph(f"<b>{prob_str}</b>", badge_style)
        ],
        [
            Paragraph("<b>Risk Severity Tier:</b>", body_style),
            Paragraph(f"<b>{risk_tier.upper()}</b>", body_style),
            Paragraph("<b>Patient Identifier:</b>", body_style),
            Paragraph(f"{patient_id} (Synthetic / Demo)", body_style)
        ]
    ]

    bg_color = colors.HexColor("#fef2f2") if is_positive else colors.HexColor("#f0fdf4")
    border_color = colors.HexColor("#f87171") if is_positive else colors.HexColor("#86efac")

    summary_table = Table(summary_data, colWidths=[120, 150, 120, 140])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_color),
        ("BOX", (0, 0), (-1, -1), 1.2, border_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # 3. Patient Clinical Indicators Table
    story.append(Paragraph("1. Submitted Clinical Parameters", section_heading))
    
    rows = [["Indicator", "Patient Value", "Standard Clinical Reference Range", "Status"]]
    for f in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]:
        meta = FEATURE_METADATA.get(f, {})
        val = patient_data.get(f, "N/A")
        label = meta.get("label", f)
        unit = meta.get("unit", "")
        ref = meta.get("reference_range", meta.get("choices", {}).get(val, "Standard"))
        if isinstance(ref, dict):
            ref = str(ref)

        val_display = f"{val} {unit}".strip()
        status = "Evaluated"
        rows.append([label, val_display, str(ref)[:50], status])

    param_table = Table(rows, colWidths=[160, 90, 210, 70])
    param_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])
    ]))
    story.append(param_table)
    story.append(Spacer(1, 14))

    # 4. Clinical Alert Findings
    alerts = prediction_result.get("risk_factors", [])
    story.append(Paragraph("2. Clinical Heuristic Flags (AHA Guidelines)", section_heading))
    if alerts:
        alert_rows = [["Flagged Parameter", "Observed Level", "Clinical Context"]]
        for a in alerts:
            alert_rows.append([a["feature"], a["value"] + f" ({a['status']})", a["clinical_note"]])
        alert_table = Table(alert_rows, colWidths=[140, 140, 250])
        alert_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fee2e2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#991b1b")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#fca5a5")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(alert_table)
    else:
        story.append(Paragraph("No critical physiological risk alerts identified based on standard threshold checks.", body_style))
    story.append(Spacer(1, 14))

    # 5. Explainable AI: SHAP Attribution
    if explanation_result:
        story.append(Paragraph("3. Explainable AI: Model Reasoning Attribution", section_heading))
        narrative = explanation_result.get("clinical_narrative", "")
        if narrative:
            story.append(Paragraph(narrative, body_style))
            story.append(Spacer(1, 6))

        top_inc = explanation_result.get("top_risk_drivers", [])
        top_dec = explanation_result.get("top_protective_factors", [])
        
        xai_rows = [["Primary Risk Drivers (Pushing Risk UP)", "Protective Factors (Pulling Risk DOWN)"]]
        max_len = max(len(top_inc), len(top_dec), 1)
        for i in range(max_len):
            inc_text = f"+ {top_inc[i]['label']} (+{top_inc[i]['shap_value']:.3f})" if i < len(top_inc) else "-"
            dec_text = f"- {top_dec[i]['label']} ({top_dec[i]['shap_value']:.3f})" if i < len(top_dec) else "-"
            xai_rows.append([inc_text, dec_text])

        xai_table = Table(xai_rows, colWidths=[265, 265])
        xai_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(xai_table)
        story.append(Spacer(1, 14))

    # 6. Medical Disclaimer Footer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
    story.append(Paragraph(f"<b>LEGAL & CLINICAL DISCLAIMER:</b> {CLINICAL_DISCLAIMER}", disclaimer_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
