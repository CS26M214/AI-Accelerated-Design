"""Build the concise Assignment 3 PDF report from the measured experiments."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from tensor_flat import print_results, run_experiments


OUTPUT_PATH = Path(__file__).with_name("AIAD_Assignment3_Report.pdf")


def _draw_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8DEE8"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#526174"))
    canvas.drawString(18 * mm, 10 * mm, "AI Accelerator Design | Assignment 3")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> Path:
    results = run_experiments()
    print_results(results)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#17365D"),
            alignment=TA_CENTER,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#526174"),
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.5,
            alignment=TA_LEFT,
            spaceAfter=5,
        )
    )
    code_style = ParagraphStyle(
        name="PseudoCode",
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        leftIndent=7,
        rightIndent=5,
        textColor=colors.HexColor("#172B4D"),
    )

    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=21 * mm,
        title="AI Accelerator Design Assignment 3 - Tensor Flattening",
        author="IIT Tirupati M.Tech DS and AI",
    )

    story = [
        Paragraph("AI Accelerator Design | Assignment 3", styles["ReportTitle"]),
        Paragraph(
            "Manual BCHW to B x (C*H*W) mapping and reconstruction",
            styles["ReportSubtitle"],
        ),
        Paragraph("1. Objective and address mapping", styles["SectionHeading"]),
        Paragraph(
            "This work maps a batch of channel-first image or feature-map tensors "
            "from shape B x C x H x W into B x (C*H*W), then reconstructs the "
            "original layout using explicit loops and index arithmetic. NumPy "
            "provides storage and element access; both conversions are implemented "
            "element by element.",
            styles["ReportBody"],
        ),
        Paragraph(
            "For a fixed batch index b, the flattened position is "
            "j = c*H*W + h*W + w. The inverse is c = j // (H*W), "
            "r = j % (H*W), h = r // W, and w = r % W. Batch index b is "
            "preserved as the row index of the two-dimensional result.",
            styles["ReportBody"],
        ),
        Paragraph("2. Pseudocode", styles["SectionHeading"]),
        KeepTogether(
            [
                Preformatted(
                    "FLATTEN(I, B, C, H, W):\n"
                    "  allocate F[B, C*H*W]\n"
                    "  for b = 0..B-1, c = 0..C-1:\n"
                    "    for h = 0..H-1, w = 0..W-1:\n"
                    "      j = c*H*W + h*W + w\n"
                    "      F[b,j] = I[b,c,h,w]\n"
                    "  return F\n\n"
                    "RECONSTRUCT(F, B, C, H, W):\n"
                    "  allocate R[B, C, H, W]\n"
                    "  for b = 0..B-1, j = 0..C*H*W-1:\n"
                    "    c = j // (H*W); r = j % (H*W)\n"
                    "    h = r // W;       w = r % W\n"
                    "    R[b,c,h,w] = F[b,j]\n"
                    "  return R",
                    code_style,
                ),
                Spacer(1, 3),
            ]
        ),
        Paragraph("3. Manual indexing example", styles["SectionHeading"]),
        Paragraph(
            "Let B=1, C=2, H=2, W=3, with channel 0 equal to "
            "[[a,b,c],[d,e,f]] and channel 1 equal to [[g,h,i],[j,k,l]]. "
            "The flattened row is [a,b,c,d,e,f,g,h,i,j,k,l]. For the element "
            "I[0,1,1,2]=l, j=1*2*3 + 1*3 + 2=11, so F[0,11]=l. In reverse, "
            "11 // 6=1, remainder 5, 5 // 3=1, and 5 % 3=2, recovering "
            "(c,h,w)=(1,1,2).",
            styles["ReportBody"],
        ),
        Paragraph("4. Experiments and results", styles["SectionHeading"]),
        Paragraph(
            "The grayscale and RGB inputs are 64 x 96 crops from scikit-learn's "
            "bundled china.jpg sample image; grayscale values use the standard "
            "0.299R + 0.587G + 0.114B luminance formula. Synthetic feature maps "
            "use B=2, H=W=8 and seeded uniform floating-point values. The seed "
            "is 20261005 for reproducibility.",
            styles["ReportBody"],
        ),
    ]

    table_data = [["Dataset / input", "B", "C", "H", "W", "Emax", "MAE"]]
    for row in results:
        table_data.append(
            [
                row["name"],
                str(row["B"]),
                str(row["C"]),
                str(row["H"]),
                str(row["W"]),
                f'{row["Emax"]:.1e}',
                f'{row["MAE"]:.1e}',
            ]
        )

    results_table = Table(
        table_data,
        colWidths=[64 * mm, 10 * mm, 10 * mm, 10 * mm, 10 * mm, 23 * mm, 23 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )
    results_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD3DF")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend(
        [
            results_table,
            Spacer(1, 6),
            Paragraph("5. Discussion", styles["SectionHeading"]),
            Paragraph(
                "All tested configurations reconstructed exactly: Emax=0 and "
                "MAE=0 in every row. This includes the single-channel grayscale "
                "case, three-channel RGB case, and feature maps up to 500 channels. "
                "The result confirms that the C-H-W ordering and the inverse "
                "quotient/remainder calculations agree for all tested indices. "
                "The experiments check data-layout correctness; they do not measure "
                "accelerator throughput or memory bandwidth.",
                styles["ReportBody"],
            ),
        ]
    )

    document.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(f"Wrote {build_report()}")
