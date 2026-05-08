"""Append a 'How the project works' pipeline slide to the existing deck.

Opens Unit_Test_Data/FR_Thor_Unit_Testing.pptx in place, adds one new slide
styled to match the dark theme used on slide 3, then reorders the new slide
into position 4 (right after 'About the project'). All other slides are
untouched.

Run:
    python Unit_Test_Data/presentation/add_pipeline_slide.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent.parent
PPT_PATH = ROOT / "Unit_Test_Data" / "FR_Thor_Unit_Testing.pptx"

# Palette pulled from existing slide 3
BG       = RGBColor(0x10, 0x14, 0x1A)
HEADER   = RGBColor(0x4F, 0xC3, 0xF7)
SUB      = RGBColor(0x81, 0xD4, 0xFA)
CARD     = RGBColor(0x1B, 0x3A, 0x4D)
BODY     = RGBColor(0xEC, 0xEF, 0xF4)
ARROW    = RGBColor(0x90, 0xA4, 0xAE)
NAV_BG   = RGBColor(0x22, 0x2A, 0x36)
NAV_BLUE = RGBColor(0x4F, 0xC3, 0xF7)
NAV_GOLD = RGBColor(0xFF, 0xD5, 0x4F)
COUNTER  = RGBColor(0x90, 0xA4, 0xAE)

# Per-phase accent colours (matching the FD/FT/FR cards on slide 3)
FD_BLUE   = RGBColor(0x4F, 0xC3, 0xF7)
FT_GREEN  = RGBColor(0x81, 0xC7, 0x84)
FR_PURPLE = RGBColor(0xCE, 0x93, 0xD8)
RPT_GOLD  = RGBColor(0xFF, 0xD5, 0x4F)


def add_text(slide, x, y, w, h, text, *, size=14, bold=False,
             color=BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return tb


def add_rect(slide, x, y, w, h, fill, *, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
    shp.shadow.inherit = False
    # clear default text frame so we can layer our own boxes
    shp.text_frame.text = ""
    return shp


def add_phase_card(slide, x, y, w, h, accent, num, title, desc):
    add_rect(slide, x, y, w, h, CARD)
    # accent strip on top
    add_rect(slide, x, y, w, Inches(0.18), accent)
    add_text(slide, x + Inches(0.15), y + Inches(0.30),
             w - Inches(0.30), Inches(0.30),
             f"PHASE {num}", size=10, bold=True, color=accent)
    add_text(slide, x + Inches(0.15), y + Inches(0.58),
             w - Inches(0.30), Inches(0.45),
             title, size=16, bold=True, color=accent)
    add_text(slide, x + Inches(0.15), y + Inches(1.08),
             w - Inches(0.30), h - Inches(1.20),
             desc, size=11, color=BODY)


def add_arrow(slide, x, y, w, h):
    shp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = ARROW
    shp.line.fill.background()


def add_nav(slide, x, y, label, color):
    btn = add_rect(slide, x, y, Inches(1.10), Inches(0.40), NAV_BG)
    add_text(slide, x, y, Inches(1.10), Inches(0.40),
             label, size=11, bold=True, color=color,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return btn


def build_pipeline_slide(prs):
    blank = prs.slide_layouts[6]
    s = prs.slides.add_slide(blank)

    # full-bleed background
    add_rect(s, Inches(0.03), Inches(0.0), Inches(13.33), Inches(7.50), BG)

    # header
    add_text(s, Inches(0.50), Inches(0.50), Inches(12.30), Inches(0.90),
             "How the pipeline works", size=36, bold=True, color=HEADER)
    add_text(s, Inches(0.50), Inches(1.40), Inches(12.30), Inches(0.50),
             "Four phases per day, fully offline, one Excel report at the end",
             size=18, bold=True, color=SUB)

    # input strip
    add_rect(s, Inches(0.50), Inches(2.20), Inches(12.30), Inches(0.55), CARD)
    add_text(s, Inches(0.50), Inches(2.20), Inches(12.30), Inches(0.55),
             "INPUT   ·   CCTV clips pulled from the recording Jetson over SSH/SCP into data/<date>/videos/",
             size=12, bold=True, color=BODY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # four phase cards across the slide
    card_w = Inches(2.85)
    card_h = Inches(2.55)
    gap    = Inches(0.20)
    arrow_w = Inches(0.25)
    arrow_h = Inches(0.40)
    y_card  = Inches(3.05)

    x = Inches(0.50)
    phases = [
        (FD_BLUE,   "1", "Tracking",
         "YOLO finds people in each frame. OSNet turns each person into a 512-d "
         "fingerprint. PersonTracker uses a 3-phase greedy match to keep the "
         "same track ID across frames and across clips."),
        (FT_GREEN,  "2", "Face Capture",
         "YOLO-face finds faces inside each tracked person. score_face grades "
         "each crop on blur, size, brightness and pose. The best crops are "
         "saved as face_*_qNN.jpg, capped per person."),
        (FR_PURPLE, "3", "Face Recognition",
         "Each saved face passes 3 filters (quality / skin / keypoints). "
         "InsightFace embeds the survivors. FAISS matches against the staff "
         "registry; unmatched faces are clustered into customers."),
        (RPT_GOLD,  "4", "Reporting",
         "Per-person results are written to state.json after every clip. "
         "At the end of the day, a 6-sheet Excel report with thumbnails is "
         "written to data/<date>/report/."),
    ]
    for i, (accent, num, title, desc) in enumerate(phases):
        add_phase_card(s, x, y_card, card_w, card_h, accent, num, title, desc)
        if i < len(phases) - 1:
            ax = x + card_w + Inches(-0.025)
            ay = y_card + (card_h - arrow_h) / 2
            add_arrow(s, ax, ay, arrow_w + gap - Inches(0.05), arrow_h)
        x += card_w + gap

    # output strip
    add_rect(s, Inches(0.50), Inches(5.85), Inches(12.30), Inches(0.55), CARD)
    add_text(s, Inches(0.50), Inches(5.85), Inches(12.30), Inches(0.55),
             "OUTPUT   ·   data/<date>/report/fr_report_<date>.xlsx   +   "
             "state.json (crash-safe, written after every clip)",
             size=12, bold=True, color=BODY,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

    # footer note
    add_text(s, Inches(0.50), Inches(6.55), Inches(12.30), Inches(0.35),
             "Everything below the input runs on the local machine. No cloud "
             "calls. Models: YOLO11 (person + face), OSNet ReID, InsightFace "
             "antelopev2, FAISS IndexFlatIP.",
             size=11, color=SUB, align=PP_ALIGN.CENTER)

    # bottom-left page counter (matches slide 3 style)
    add_text(s, Inches(0.50), Inches(7.05), Inches(2.00), Inches(0.30),
             "4 / 10", size=11, color=COUNTER)

    # bottom-right nav cluster
    add_nav(s, Inches(9.33),  Inches(7.00), "◀  Prev", NAV_BLUE)
    add_nav(s, Inches(10.58), Inches(7.00), "⌂  Home", NAV_GOLD)
    add_nav(s, Inches(11.83), Inches(7.00), "Next  ▶", NAV_BLUE)

    return s


def move_slide(prs, old_idx, new_idx):
    """Reorder slides by editing the sldIdLst in presentation.xml."""
    sldIdLst = prs.slides._sldIdLst
    slides = list(sldIdLst)
    sldIdLst.remove(slides[old_idx])
    sldIdLst.insert(new_idx, slides[old_idx])


def main():
    prs = Presentation(str(PPT_PATH))
    before = len(prs.slides)
    build_pipeline_slide(prs)
    # newly-added slide is at the end; move it to index 3 (becomes slide 4)
    move_slide(prs, before, 3)
    prs.save(str(PPT_PATH))
    print(f"Added pipeline slide at position 4. Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
