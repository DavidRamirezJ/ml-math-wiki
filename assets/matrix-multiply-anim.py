#!/usr/bin/env python3
"""
Generate an animated GIF showing how matrix multiplication works as
row-dot-column: C = A @ B, where c_ij = sum_l a_il * b_lj.

Pure Pillow (no ffmpeg / matplotlib). Writes matrix-multiply.gif next to this file.
Reproducible companion to wiki/concepts/matrix.md.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---- the example (matches Example 2.3 in the MML book) ----
A = np.array([[1, 2, 3],
              [3, 2, 1]])              # 2x3
B = np.array([[0, 2],
              [1, -1],
              [0, 1]])                 # 3x2
C = A @ B                              # 2x2  -> [[2,3],[2,5]]

# ---- layout ----
CELL = 64                # cell size px
PAD = 28                 # padding inside a matrix grid
W, H = 1040, 560
BG = (250, 250, 252)
INK = (33, 37, 41)
GRID = (170, 176, 186)
ROW_HL = (255, 214, 102)     # A row highlight (amber)
COL_HL = (130, 200, 255)     # B col highlight (blue)
OUT_HL = (167, 243, 169)     # C cell highlight (green)
PROD = (200, 60, 60)         # product term color
DIM = (210, 212, 218)

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
f_cell = ImageFont.truetype(FONT_B, 26)
f_lab  = ImageFont.truetype(FONT_B, 30)
f_txt  = ImageFont.truetype(FONT, 24)
f_small= ImageFont.truetype(FONT, 20)
f_calc = ImageFont.truetype(FONT_B, 30)

# top-left origins of each matrix grid
A_xy = (70, 170)
B_xy = (70 + 3*CELL + 130, 128)         # B staggered up (its rows align with A's cols)
C_xy = (B_xy[0] + 2*CELL + 150, 170)

def grid_size(m):  # (cols, rows) -> pixel w,h
    r, c = m.shape
    return c*CELL, r*CELL

def cell_box(origin, i, j):
    x = origin[0] + j*CELL
    y = origin[1] + i*CELL
    return [x, y, x+CELL, y+CELL]

def center_text(d, box, text, font, fill):
    tb = d.textbbox((0,0), text, font=font)
    tw, th = tb[2]-tb[0], tb[3]-tb[1]
    cx = (box[0]+box[2])/2 - tw/2 - tb[0]
    cy = (box[1]+box[3])/2 - th/2 - tb[1]
    d.text((cx, cy), text, font=font, fill=fill)

def draw_bracket(d, origin, m, color=INK, wdt=3):
    w, h = grid_size(m)
    x0, y0 = origin
    x1, y1 = x0+w, y0+h
    ear = 12
    for (xa, xb, ear_dir) in [(x0, x0, 1), (x1, x1, -1)]:
        d.line([(xa, y0), (xa, y1)], fill=color, width=wdt)
        d.line([(xa, y0), (xa+ear*ear_dir, y0)], fill=color, width=wdt)
        d.line([(xa, y1), (xa+ear*ear_dir, y1)], fill=color, width=wdt)

def draw_matrix(d, origin, m, label, hl_row=None, hl_col=None, hl_cell=None,
                fade_unhighlighted=False, shown_mask=None):
    r, c = m.shape
    for i in range(r):
        for j in range(c):
            box = cell_box(origin, i, j)
            fill = None
            active = False
            if hl_cell is not None and (i, j) == hl_cell:
                fill = OUT_HL; active = True
            elif hl_row is not None and i == hl_row:
                fill = ROW_HL; active = True
            elif hl_col is not None and j == hl_col:
                fill = COL_HL; active = True
            if fill:
                d.rectangle(box, fill=fill)
            d.rectangle(box, outline=GRID, width=2)
            # value
            val_color = INK
            if fade_unhighlighted and not active:
                val_color = DIM
            show = True
            if shown_mask is not None:
                show = shown_mask[i][j]
            if show:
                center_text(d, box, str(m[i, j]), f_cell, val_color)
    draw_bracket(d, origin, m)
    # label above
    w, _ = grid_size(m)
    center_text(d, [origin[0], origin[1]-44, origin[0]+w, origin[1]-12], label, f_lab, INK)

def base_frame(hlrow=None, hlcol=None, hlcell=None, cmask=None, fade=False, caption=""):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # title
    center_text(d, [0, 8, W, 48], "Matrix Multiplication:  C = A · B", f_lab, INK)
    center_text(d, [0, 50, W, 78],
                "each entry  cᵢⱼ  =  (row i of A)  ·  (column j of B)", f_txt, (90,94,102))
    draw_matrix(d, A_xy, A, "A  (2×3)", hl_row=hlrow, fade_unhighlighted=fade)
    draw_matrix(d, B_xy, B, "B  (3×2)", hl_col=hlcol, fade_unhighlighted=fade)
    draw_matrix(d, C_xy, C, "C  (2×2)", hl_cell=hlcell, shown_mask=cmask)
    # multiply / equals signs
    center_text(d, [A_xy[0]+3*CELL, A_xy[1], B_xy[0], A_xy[1]+2*CELL], "×", f_lab, INK)
    center_text(d, [B_xy[0]+2*CELL, C_xy[1], C_xy[0], C_xy[1]+2*CELL], "=", f_lab, INK)
    if caption:
        d.rectangle([40, H-78, W-40, H-20], fill=(238,240,245), outline=GRID, width=2)
        center_text(d, [40, H-78, W-40, H-20], caption, f_calc, INK)
    return img, d

frames = []
def hold(img, n): frames.extend([img]*n)

r, c = C.shape
inner = A.shape[1]
shown = [[False]*c for _ in range(r)]

# intro — C starts empty
img, _ = base_frame(cmask=shown, caption="C is 2×2.  We fill it one entry at a time.")
hold(img, 14)

for i in range(r):
    for j in range(c):
        # 1) highlight row i and col j
        img, d = base_frame(hlrow=i, hlcol=j, hlcell=(i,j), cmask=shown, fade=True,
                            caption=f"c[{i+1},{j+1}] = row {i+1} of A  ·  col {j+1} of B")
        hold(img, 12)
        # 2) build the sum term by term
        terms = []
        running = 0
        for l in range(inner):
            a, b = A[i, l], B[l, j]
            running += a*b
            terms.append((a, b))
            img, d = base_frame(hlrow=i, hlcol=j, hlcell=(i,j), cmask=shown, fade=True)
            # render the running calculation in the caption bar manually for color
            d.rectangle([40, H-78, W-40, H-20], fill=(238,240,245), outline=GRID, width=2)
            parts = "  +  ".join(f"{aa}×({bb})" if bb < 0 else f"{aa}×{bb}" for aa,bb in terms)
            txt = f"c[{i+1},{j+1}]  =  {parts}"
            if l == inner-1:
                txt += f"  =  {running}"
            center_text(d, [40, H-78, W-40, H-20], txt, f_calc, PROD if l<inner-1 else INK)
            hold(img, 11)
        # 3) commit the value into C
        shown[i][j] = True
        img, d = base_frame(hlcell=(i,j), cmask=shown,
                            caption=f"✓  c[{i+1},{j+1}] = {C[i,j]}")
        hold(img, 12)

# final: full result
img, _ = base_frame(cmask=[[True]*c for _ in range(r)],
                    caption="Done — every entry is a row·column dot product.")
hold(img, 26)

out = __file__.rsplit("/",1)[0] + "/matrix-multiply.gif"
frames[0].save(out, save_all=True, append_images=frames[1:],
               duration=90, loop=0, optimize=True)
print("wrote", out, "frames:", len(frames))
