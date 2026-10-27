"""Matrix — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
import contextlib
import io
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.path.expanduser("~/.claude/skills/explainer-video/scripts"))
sys.path.insert(0, str(HERE))

from studio import *  # noqa: E402,F403
from narration import N  # noqa: E402
sys.path.insert(0, str(HERE.parent))
from common import Credits  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    import demo  # real numbers for the animations

SERIES = "ML Math Atoms · Video 2 of 16"
TITLE = "Matrix — a grid of numbers that moves vectors"


def arr(plane, a, b, color, width=6):
    return Arrow(plane.c2p(*a), plane.c2p(*b), buff=0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.18)


def grid_plane(xr, yr, xl, yl):
    return NumberPlane(x_range=xr, y_range=yr, x_length=xl, y_length=yl,
                       background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.35,
                                              "stroke_width": 1},
                       axis_config={"stroke_color": MUTED})


def mat(rows, scale=1.0, color=FG, h_buff=1.0, v_buff=0.8):
    m = Matrix([[str(e) for e in r] for r in rows], h_buff=h_buff, v_buff=v_buff)
    m.set_color(color)
    m.scale(scale)
    return m


def colour_cols(m, colours):
    for c, k in zip(m.get_columns(), colours):
        c.set_color(k)
    return m


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        if title.width > 12.5:
            title.scale_to_fit_width(12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.4)
        X = mat(demo.X.tolist(), 0.8, BLUE, h_buff=1.5).shift(LEFT * 3.2 + DOWN * 0.9)
        rows = X.get_rows()
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            self.wait_until(v, "vectors")
            self.play(LaggedStart(*[FadeIn(r, shift=DOWN * 0.2) for r in rows], lag_ratio=0.3),
                      run_time=1.2)
            self.wait_until(v, "table")
            self.play(FadeIn(X.get_brackets()), run_time=0.6)
            self.wait_until(v, "matrix")
            self.play(Indicate(X, color=YELLOW, scale_factor=1.08), run_time=1)
        tlab = T("a table that stores data", 26, BLUE).next_to(X, DOWN, buff=0.35)
        xin = MathTex("x", color=BLUE, font_size=54)
        box = RoundedRectangle(corner_radius=0.15, width=1.6, height=1.2, color=PURPLE)
        bl = MathTex("A", color=PURPLE, font_size=60).move_to(box)
        xout = MathTex("Ax", color=YELLOW, font_size=54)
        mach = VGroup(xin, VGroup(box, bl), xout).arrange(RIGHT, buff=1.1)
        mach.move_to(RIGHT * 3.4 + DOWN * 0.9)
        a1 = Arrow(xin.get_right(), box.get_left(), buff=0.15, color=MUTED)
        a2 = Arrow(box.get_right(), xout.get_left(), buff=0.15, color=MUTED)
        mlab = T("a machine that moves vectors", 26, PURPLE).next_to(mach, DOWN, buff=0.35)
        mlab.align_to(tlab, DOWN)
        pre = T("You need: video 1 · Vector", 24, MUTED).to_edge(DOWN, buff=0.4)
        with self.voice(N["intro2"]) as v:
            self.wait_until(v, "table")
            self.play(FadeIn(tlab, shift=UP * 0.2), run_time=0.6)
            self.wait_until(v, "machine")
            self.play(FadeIn(xin), Create(box), Write(bl), GrowArrow(a1), GrowArrow(a2),
                      FadeIn(xout), FadeIn(mlab), run_time=1.4)
            self.wait_until(v, "need")
            self.play(FadeIn(pre), run_time=0.6)


class C1_Table(Studio):
    def construct(self):
        self.heading("1 · A matrix is a table")
        X = mat(demo.X.tolist(), 0.95, FG, h_buff=1.6).shift(LEFT * 2.6 + DOWN * 0.3)
        rows, cols = X.get_rows(), X.get_columns()
        heads = VGroup(*[T(s, 22, ORANGE).move_to([c.get_center()[0], X.get_top()[1] + 0.45, 0])
                         for s, c in zip(["area m²", "bedrooms", "age"], cols)])
        rlabs = VGroup(*[T(f"house {i}", 22, BLUE).next_to(X, LEFT, buff=0.3)
                         .set_y(r.get_center()[1]) for i, r in zip((1, 2, 3), rows)])
        rem = self.reminder_card("vector", "A list of numbers you can add and scale "
                                 "(video 1).", pos=DR, width=4.6)
        with self.voice(N["table1"]) as v:
            for h, w in zip(heads, ["area", "bedrooms", "age"]):
                self.wait_until(v, w)
                self.play(FadeIn(h, shift=DOWN * 0.15), run_time=0.5)
            self.wait_until(v, "vector")
            self.play(FadeIn(rem, shift=UP * 0.3),
                      LaggedStart(*[AnimationGroup(FadeIn(l), FadeIn(r, shift=RIGHT * 0.2))
                                    for l, r in zip(rlabs, rows)], lag_ratio=0.4), run_time=1.6)
            self.wait_until(v, "stack")
            self.play(FadeIn(X.get_brackets()), run_time=0.8)
        card = self.term_card("matrix", "A rectangular grid of numbers, arranged in rows and "
                              "columns.", pos=DR, width=5.0)
        br = Brace(X, RIGHT, color=MUTED)
        brl = T("3 rows", 22, MUTED).next_to(br, RIGHT, buff=0.15)
        bd = Brace(X, DOWN, color=MUTED)
        bdl = T("3 columns", 22, MUTED).next_to(bd, DOWN, buff=0.12)
        shape = MathTex(r"3\times 3", font_size=72, color=YELLOW).move_to(RIGHT * 4.2 + UP * 1.6)
        shl = T("rows × columns", 24, MUTED).next_to(shape, DOWN, buff=0.25)
        with self.voice(N["table2"]) as v:
            self.play(FadeOut(rem), run_time=0.4)
            self.wait_until(v, "matrix")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "rows", 2)
            self.play(GrowFromCenter(br), FadeIn(brl), run_time=0.6)
            self.wait_until(v, "columns", 2)
            self.play(GrowFromCenter(bd), FadeIn(bdl), run_time=0.6)
            self.wait_until(v, "by")
            self.play(Write(shape), FadeIn(shl), run_time=0.8)
        ent = X.get_entries()[3]   # row 2, column 1
        box = SurroundingRectangle(ent, color=YELLOW, buff=0.1)
        addr = MathTex("a_{21}", "=", "80", font_size=60).move_to(RIGHT * 4.2 + DOWN * 0.6)
        addr[0].set_color(YELLOW); addr[2].set_color(YELLOW)
        leg = T("a  row, column", 24, MUTED).next_to(addr, DOWN, buff=0.3)
        with self.voice(N["table3"]) as v:
            self.play(FadeOut(card), run_time=0.4)
            self.wait_until(v, "address")
            self.play(Write(addr[0]), run_time=0.8)
            self.wait_until(v, "row")
            self.play(FadeIn(leg), run_time=0.5)
            self.wait_until(v, "two", 2)
            self.play(Create(box), run_time=0.6)
            self.wait_until(v, "eighty")
            self.play(Write(addr[1:]), Indicate(ent, color=YELLOW), run_time=0.8)
        rbox = SurroundingRectangle(VGroup(rows[0], rlabs[0]), color=BLUE, buff=0.12)
        cbox = SurroundingRectangle(VGroup(cols[1], heads[1]), color=ORANGE, buff=0.12)
        rl = T("row = one data point", 26, BLUE)
        cl = T("column = one feature", 26, ORANGE)
        bl = T("a batch of data: one row per example", 24, MUTED)
        notes = VGroup(rl, cl, bl).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        notes.move_to(RIGHT * 3.9 + DOWN * 1.2)
        with self.voice(N["table4"]) as v:
            self.play(FadeOut(VGroup(addr, leg, box, shape, shl, br, brl)), run_time=0.5)
            self.wait_until(v, "row")
            self.play(Create(rbox), FadeIn(rl), run_time=0.7)
            self.wait_until(v, "column")
            self.play(Create(cbox), FadeIn(cl), run_time=0.7)
            self.wait_until(v, "three")
            self.play(Indicate(cols[1], color=ORANGE), run_time=1)
            self.wait_until(v, "batch")
            self.play(FadeIn(bl), run_time=0.6)


class C2_Machine(Studio):
    def construct(self):
        self.heading("2 · A matrix is a machine")
        A = colour_cols(mat(demo.A.tolist(), 1.0), [BLUE, GREEN])
        x = mat([[demo.x[0]], [demo.x[1]]], 1.0, PURPLE)
        eq = MathTex("=", font_size=60)
        q = MathTex("?", font_size=60, color=YELLOW)
        al = MathTex("A", font_size=56)
        top = VGroup(al, MathTex("=", font_size=56), A, x, eq, q).arrange(RIGHT, buff=0.3)
        top.move_to(LEFT * 2.6 + UP * 1.6)
        with self.voice(N["mach1"]) as v:
            self.wait_until(v, "matrix", 2)
            self.play(FadeIn(top[:2]), FadeIn(A.get_brackets()), run_time=0.6)
            self.wait_until(v, "first")
            self.play(FadeIn(A.get_columns()[0], shift=DOWN * 0.2), run_time=0.6)
            self.wait_until(v, "second", 2)
            self.play(FadeIn(A.get_columns()[1], shift=DOWN * 0.2), run_time=0.6)
            self.wait_until(v, "x")
            self.play(FadeIn(x), run_time=0.6)
            self.wait_until(v, "times")
            self.play(FadeIn(eq), FadeIn(q), run_time=0.6)
        cv = MathTex("1", r"\begin{bmatrix}2\\1\end{bmatrix}", "+", "2",
                     r"\begin{bmatrix}1\\3\end{bmatrix}", "=",
                     r"\begin{bmatrix}2\\1\end{bmatrix}", "+", r"\begin{bmatrix}2\\6\end{bmatrix}",
                     "=", r"\begin{bmatrix}%d\\%d\end{bmatrix}" % tuple(demo.colview), font_size=48)
        for i, c in [(0, PURPLE), (1, BLUE), (3, PURPLE), (4, GREEN), (6, BLUE), (8, GREEN),
                     (10, YELLOW)]:
            cv[i].set_color(c)
        cv.move_to(LEFT * 2.2 + DOWN * 0.4)
        lab = T("column view: weights × columns", 22, MUTED).next_to(cv, UP, buff=0.3).align_to(cv, LEFT)
        rem = self.reminder_card("linear combination", "Scale some vectors, then add them: "
                                 "a·u + b·v (video 1).", pos=DR, width=4.6)
        with self.voice(N["mach2"]) as v:
            self.wait_until(v, "linear")
            self.play(FadeIn(rem, shift=UP * 0.3), FadeIn(lab), run_time=0.7)
            self.wait_until(v, "take")
            self.play(Write(cv[:5]), run_time=1.4)
            self.wait_until(v, "One", 2)
            self.play(Write(cv[5:7]), run_time=0.8)
            self.wait_until(v, "Two", 4)
            self.play(Write(cv[7:9]), run_time=0.8)
        res = mat([[demo.Ax[0]], [demo.Ax[1]]], 1.0, YELLOW).move_to(q).align_to(q, LEFT)
        with self.voice(N["mach3"]) as v:
            self.play(FadeOut(rem), run_time=0.4)
            self.wait_until(v, "add")
            self.play(Write(cv[9:]), run_time=1)
            self.wait_until(v, "four")
            self.play(ReplacementTransform(q, res), run_time=0.8)
            self.wait_until(v, "turned")
            self.play(Indicate(x, color=PURPLE), run_time=0.7)
            self.play(Indicate(res, color=YELLOW), run_time=0.7)
        r1 = SurroundingRectangle(A.get_rows()[0], color=ORANGE, buff=0.1)
        r2 = SurroundingRectangle(A.get_rows()[1], color=ORANGE, buff=0.1)
        rv = VGroup(MathTex(r"\text{row 1: }", r"2\cdot1+1\cdot2", "=", "4", font_size=44),
                    MathTex(r"\text{row 2: }", r"1\cdot1+3\cdot2", "=", "7", font_size=44))
        for r in rv:
            r[0].set_color(ORANGE); r[3].set_color(YELLOW)
        rv.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(LEFT * 3.0 + DOWN * 2.6)
        with self.voice(N["mach4"]) as v:
            self.wait_until(v, "row", 2)
            self.play(Create(r1), Indicate(x, color=PURPLE), run_time=0.8)
            self.wait_until(v, "four")
            self.play(Write(rv[0]), run_time=1)
            self.wait_until(v, "second", 2)
            self.play(ReplacementTransform(r1, r2), run_time=0.6)
            self.wait_until(v, "seven")
            self.play(Write(rv[1]), run_time=1)
        inner = VGroup(T("row × vector: multiply, then add", 26, FG),
                       T("= inner product → video 5", 26, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        inner.move_to(RIGHT * 3.7 + DOWN * 2.6)
        shp = T("columns of A = components of x", 26, YELLOW).move_to(RIGHT * 3.7 + DOWN * 1.5)
        with self.voice(N["mach5"]) as v:
            self.play(FadeOut(r2), run_time=0.4)
            self.wait_until(v, "inner")
            self.play(FadeIn(inner), run_time=0.6)
            self.wait_until(v, "shapes")
            self.play(FadeIn(shp), run_time=0.6)
            self.play(Indicate(A, color=YELLOW), Indicate(x, color=PURPLE), run_time=1)


class C3_Grid(Studio):
    def construct(self):
        self.heading("3 · Watching a matrix move the plane")
        bg = grid_plane([-4, 4, 1], [-3, 3, 1], 6.8, 5.1).shift(LEFT * 3.0 + DOWN * 0.55)
        o = bg.c2p(0, 0)
        mov = NumberPlane(x_range=[-2, 2, 1], y_range=[-3, 3, 1], x_length=3.4, y_length=5.1,
                          background_line_style={"stroke_color": BLUE, "stroke_opacity": 0.55,
                                                 "stroke_width": 2},
                          axis_config={"stroke_color": FG}).move_to(o)
        e1 = arr(bg, (0, 0), (1, 0), BLUE)
        e2 = arr(bg, (0, 0), (0, 1), GREEN)
        p = arr(bg, (0, 0), (1, 2), YELLOW, 7)
        l1 = always_redraw(lambda: MathTex("e_1", color=BLUE, font_size=34)
                           .next_to(e1.get_end(), DR, buff=0.05))
        l2 = always_redraw(lambda: MathTex("e_2", color=GREEN, font_size=34)
                           .next_to(e2.get_end(), LEFT, buff=0.1))
        notes = VGroup(MathTex(r"A\,e_1", r"=\text{column 1}", font_size=40),
                       MathTex(r"A\,e_2", r"=\text{column 2}", font_size=40))
        notes[0][0].set_color(BLUE); notes[1][0].set_color(GREEN)
        notes.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 4.0 + UP * 2.0)
        pl = T("(1, 2)", 24, YELLOW).move_to(bg.c2p(1, 2) + UP * 0.3 + RIGHT * 0.3)
        with self.voice(N["grid1"]) as v:
            self.play(Create(bg), Create(mov), run_time=1.2)
            self.wait_until(v, "blue")
            self.play(GrowArrow(e1), run_time=0.6)
            self.add(l1)
            self.wait_until(v, "green")
            self.play(GrowArrow(e2), run_time=0.6)
            self.add(l2)
            self.wait_until(v, "yellow")
            self.play(GrowArrow(p), FadeIn(pl), run_time=0.7)
            self.wait_until(v, "first")
            self.play(Write(notes[0]), run_time=0.8)
            self.wait_until(v, "second")
            self.play(Write(notes[1]), run_time=0.8)
        S = colour_cols(mat(demo.S.tolist(), 0.9), [BLUE, GREEN])
        Sg = VGroup(MathTex("S", "=", font_size=48), S).arrange(RIGHT, buff=0.2)
        Sg.move_to(RIGHT * 3.6 + UP * 0.1)
        sl = T("stretch", 26, PURPLE).next_to(Sg, RIGHT, buff=0.3)
        with self.voice(N["grid2"]) as v:
            self.wait_until(v, "columns", 2)
            self.play(FadeIn(Sg), FadeIn(sl), run_time=0.7)
            self.wait_until(v, "sends")
            self.play(Indicate(S.get_columns()[0], color=BLUE), run_time=0.7)
            self.wait_until(v, "Watch")
            self.play(mov.animate.apply_matrix(demo.S, about_point=o),
                      Transform(e1, arr(bg, (0, 0), (2, 0), BLUE)),
                      Transform(p, arr(bg, (0, 0), tuple(demo.S @ demo.p), YELLOW, 7)),
                      pl.animate.move_to(bg.c2p(2, 2) + UP * 0.3 + RIGHT * 0.3),
                      run_time=2.5)
        chk = MathTex("1", r"\begin{bmatrix}2\\0\end{bmatrix}", "+", "2",
                      r"\begin{bmatrix}0\\1\end{bmatrix}", "=",
                      r"\begin{bmatrix}%d\\%d\end{bmatrix}" % tuple(demo.S @ demo.p), font_size=42)
        for i, c in [(0, PURPLE), (1, BLUE), (3, PURPLE), (4, GREEN), (6, YELLOW)]:
            chk[i].set_color(c)
        chk.move_to(RIGHT * 3.9 + DOWN * 1.9)
        pl2 = T("(2, 2)", 24, YELLOW).move_to(pl)
        with self.voice(N["grid3"]) as v:
            self.wait_until(v, "two", 3)
            self.play(Transform(pl, pl2), run_time=0.5)
            self.wait_until(v, "Check")
            self.play(Write(chk), run_time=1.5)
            self.wait_until(v, "agree")
            self.play(Indicate(p, color=YELLOW), Indicate(chk[6], color=YELLOW), run_time=1)
        mov2 = NumberPlane(x_range=[-3, 3, 1], y_range=[-3, 3, 1], x_length=5.1, y_length=5.1,
                           background_line_style={"stroke_color": BLUE, "stroke_opacity": 0.55,
                                                  "stroke_width": 2},
                           axis_config={"stroke_color": FG}).move_to(o)
        R = colour_cols(mat(demo.R.tolist(), 0.9), [BLUE, GREEN])
        Rg = VGroup(MathTex("R", "=", font_size=48), R).arrange(RIGHT, buff=0.2).move_to(Sg)
        rl = T("rotate 90°", 26, PURPLE).next_to(Rg, RIGHT, buff=0.3)
        pl3 = T("(1, 2)", 24, YELLOW).move_to(bg.c2p(1, 2) + UP * 0.3 + RIGHT * 0.3)
        with self.voice(N["grid4"]) as v:
            self.wait_until(v, "different")
            self.play(FadeOut(chk), FadeOut(mov), FadeIn(mov2),
                      Transform(e1, arr(bg, (0, 0), (1, 0), BLUE)),
                      Transform(p, arr(bg, (0, 0), (1, 2), YELLOW, 7)), Transform(pl, pl3),
                      FadeOut(Sg), FadeOut(sl), FadeIn(Rg), FadeIn(rl), run_time=1.2)
            self.wait_until(v, "sends")
            self.play(Indicate(R.get_columns()[0], color=BLUE), run_time=0.6)
            self.play(Indicate(R.get_columns()[1], color=GREEN), run_time=0.6)
            self.wait_until(v, "rotates")
            self.play(FadeOut(pl), Rotate(VGroup(mov2, e1, e2, p), angle=PI / 2, about_point=o),
                      run_time=2.5)
        pl4 = T("(−2, 1)", 24, YELLOW).move_to(bg.c2p(-2, 1) + UP * 0.35 + LEFT * 0.3)
        keep = VGroup(T("grid lines stay straight,", 24, FG),
                      T("parallel, evenly spaced", 24, FG),
                      T("the origin stays fixed", 24, FG)).arrange(DOWN, aligned_edge=LEFT,
                                                                   buff=0.2)
        keep.move_to(RIGHT * 3.9 + DOWN * 1.6)
        v4 = T("→ video 4: linear maps", 24, MUTED).next_to(keep, DOWN, buff=0.35)
        with self.voice(N["grid5"]) as v:
            self.wait_until(v, "minus")
            self.play(FadeIn(pl4), Indicate(p, color=YELLOW), run_time=0.8)
            self.wait_until(v, "straight")
            self.play(FadeIn(keep[:2]), run_time=0.6)
            self.wait_until(v, "origin")
            self.play(FadeIn(keep[2]), Flash(o, color=YELLOW), run_time=0.8)
            self.wait_until(v, "linear")
            self.play(FadeIn(v4), run_time=0.6)


class C4_Multiply(Studio):
    def construct(self):
        self.heading("4 · Multiplying matrices")
        xin = MathTex("x", color=BLUE, font_size=54)
        bB = VGroup(RoundedRectangle(corner_radius=0.15, width=1.4, height=1.1, color=GREEN),
                    MathTex("B", color=GREEN, font_size=56))
        bx = MathTex("Bx", font_size=50)
        bA = VGroup(RoundedRectangle(corner_radius=0.15, width=1.4, height=1.1, color=ORANGE),
                    MathTex("A", color=ORANGE, font_size=56))
        abx = MathTex("A(Bx)", color=YELLOW, font_size=50)
        flow = VGroup(xin, bB, bx, bA, abx).arrange(RIGHT, buff=0.9).move_to(UP * 1.2)
        for b in (bB, bA):
            b[1].move_to(b[0])
        arws = VGroup(*[Arrow(flow[i].get_right(), flow[i + 1].get_left(), buff=0.12, color=MUTED)
                        for i in range(4)])
        form = MathTex("A(Bx)", "=", "(AB)", "x", font_size=70).move_to(DOWN * 1.0)
        form[2].set_color(YELLOW); form[3].set_color(BLUE)
        rtl = T("B stands next to x, so B acts first", 24, MUTED).next_to(form, DOWN, buff=0.45)
        with self.voice(N["mult1"]) as v:
            self.play(FadeIn(xin), run_time=0.4)
            self.wait_until(v, "First")
            self.play(GrowArrow(arws[0]), FadeIn(bB), GrowArrow(arws[1]), FadeIn(bx), run_time=1)
            self.wait_until(v, "Then")
            self.play(GrowArrow(arws[2]), FadeIn(bA), GrowArrow(arws[3]), FadeIn(abx), run_time=1)
            self.wait_until(v, "product")
            self.play(Write(form), run_time=1.2)
            self.wait_until(v, "order")
            self.play(FadeIn(rtl), Indicate(bB, color=GREEN), run_time=0.9)
        self.clear_body()
        card = self.term_card("matrix multiplication", "Doing one matrix after another. Entry "
                              "(i, j) of AB: row i of A times column j of B, multiplied entry "
                              "by entry and added.", pos=DR, width=5.6)
        words = VGroup(T("entry (i, j) of AB  =", 28, YELLOW), T("row i of A", 28, BLUE),
                       T("with", 28, MUTED), T("column j of B", 28, GREEN)).arrange(RIGHT, buff=0.2)
        words.move_to(UP * 2.3)
        f = MathTex("(AB)_{ij}", "=", r"\sum_{k=1}^{n}", "a_{ik}", r"\,b_{kj}", font_size=66)
        f[0].set_color(YELLOW); f[2].set_color(ORANGE); f[3].set_color(BLUE); f[4].set_color(GREEN)
        f.move_to(LEFT * 3.0 + UP * 0.8)
        leg = VGroup(*[VGroup(MathTex(s, color=c, font_size=36), T(t, 22, MUTED)).arrange(RIGHT, buff=0.3)
                       for s, c, t in [("(AB)_{ij}", YELLOW, "entry in row i, column j of AB"),
                                       (r"\sum_{k=1}^{n}", ORANGE, "add up for k = 1, 2, …, n"),
                                       ("a_{ik}", BLUE, "entry k of row i of A"),
                                       ("b_{kj}", GREEN, "entry k of column j of B")]])
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.22).to_edge(LEFT, buff=0.8).shift(DOWN * 2.3)
        with self.voice(N["mult2"]) as v:
            self.wait_until(v, "multiplication")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "row")
            self.play(FadeIn(words), run_time=0.8)
            self.wait_until(v, "symbols")
            self.play(Write(f), run_time=1.2)
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in leg], lag_ratio=0.3),
                      run_time=1.4)
        self.clear_body()
        A = mat(demo.P.tolist(), 1.0)
        B = mat(demo.Q.tolist(), 1.0)
        C = mat(demo.PQ.tolist(), 1.0, YELLOW, h_buff=1.2)
        for e in C.get_entries():
            e.set_opacity(0)
        top = VGroup(MathTex("A", font_size=48), A, MathTex("B", font_size=48), B,
                     MathTex("=", font_size=56), C).arrange(RIGHT, buff=0.3).move_to(UP * 1.0)
        top[0].next_to(A, UP, buff=0.15); top[2].next_to(B, UP, buff=0.15)
        calc = MathTex(r"1\cdot5+2\cdot7=19", font_size=52).move_to(DOWN * 1.4)
        Ar, Bc, Ce = A.get_rows(), B.get_columns(), C.get_entries()
        rb = SurroundingRectangle(Ar[0], color=BLUE, buff=0.1)
        cb = SurroundingRectangle(Bc[0], color=GREEN, buff=0.1)
        sums = [(0, 0, r"1\cdot5+2\cdot7=19"), (0, 1, r"1\cdot6+2\cdot8=22"),
                (1, 0, r"3\cdot5+4\cdot7=43"), (1, 1, r"3\cdot6+4\cdot8=50")]
        for i, j, s in sums:     # check the on-screen sums against the demo
            assert s.endswith(str(demo.PQ[i, j]))

        def show(k):
            i, j, s = sums[k]
            new = MathTex(s, font_size=52).move_to(calc)
            return [Transform(rb, SurroundingRectangle(Ar[i], color=BLUE, buff=0.1)),
                    Transform(cb, SurroundingRectangle(Bc[j], color=GREEN, buff=0.1)),
                    Transform(calc, new)]
        with self.voice(N["mult3"]) as v:
            self.play(FadeIn(top), run_time=0.8)
            self.wait_until(v, "Top")
            self.play(Create(rb), Create(cb), run_time=0.6)
            self.wait_until(v, "nineteen")
            self.play(Write(calc), Ce[0].animate.set_opacity(1), run_time=0.8)
            self.wait_until(v, "Top", 2)
            self.play(*show(1), run_time=0.6)
            self.wait_until(v, "eight", 2)
            self.play(Ce[1].animate.set_opacity(1), run_time=0.6)
        with self.voice(N["mult4"]) as v:
            self.wait_until(v, "Bottom")
            self.play(*show(2), run_time=0.6)
            self.wait_until(v, "seven", 2)
            self.play(Ce[2].animate.set_opacity(1), run_time=0.6)
            self.wait_until(v, "Bottom", 2)
            self.play(*show(3), run_time=0.6)
            self.wait_until(v, "fifty")
            self.play(Ce[3].animate.set_opacity(1), run_time=0.6)
            self.wait_until(v, "So")
            self.play(FadeOut(rb), FadeOut(cb), Indicate(C, color=YELLOW), run_time=1)
        self.clear_body()
        rule = MathTex("(m", r"\times", "n", ")", "(", "n", r"\times", "p)", r"\;\to\;", "(m",
                       r"\times", "p)", font_size=72).move_to(UP * 1.4)
        rule[2].set_color(ORANGE); rule[5].set_color(ORANGE)
        rule[9:].set_color(YELLOW)
        ok = VGroup(MathTex(r"(2\times3)(3\times2)\to 2\times2", font_size=52, color=GREEN),
                    T("✓", 40, GREEN)).arrange(RIGHT, buff=0.4)
        no = VGroup(MathTex(r"(2\times3)(2\times3)", font_size=52, color=RED),
                    T("✗ not allowed", 32, RED)).arrange(RIGHT, buff=0.4)
        VGroup(ok, no).arrange(DOWN, buff=0.6, aligned_edge=LEFT).move_to(DOWN * 1.3)
        with self.voice(N["mult5"]) as v:
            self.wait_until(v, "shape")
            self.play(Write(rule), run_time=1.4)
            self.wait_until(v, "inner")
            self.play(Indicate(rule[2], color=ORANGE), Indicate(rule[5], color=ORANGE), run_time=1)
            self.wait_until(v, "gives")
            self.play(FadeIn(ok, shift=UP * 0.2), run_time=0.7)
            self.wait_until(v, "allowed")
            self.play(FadeIn(no, shift=UP * 0.2), run_time=0.7)
        self.clear_body()
        pls, finals = [], []
        for cx, title in [(-3.6, "S first, then R"), (2.9, "R first, then S")]:
            pl = grid_plane([-1, 3, 1], [-1, 3, 1], 3.2, 3.2).move_to([cx, 0.5, 0])
            pl.shift(pl.get_center() - pl.c2p(1, 1))
            tt = T(title, 26, FG).next_to(pl, UP, buff=0.2)
            pls.append((pl, tt))
        e_l = arr(pls[0][0], (0, 0), (1, 0), BLUE)
        e_r = arr(pls[1][0], (0, 0), (1, 0), BLUE)
        with self.voice(N["mult6"]) as v:
            self.play(*[Create(pl) for pl, _ in pls], *[FadeIn(t) for _, t in pls],
                      GrowArrow(e_l), GrowArrow(e_r), run_time=1.2)
            self.wait_until(v, "goes")
            self.play(Transform(e_l, arr(pls[0][0], (0, 0), (2, 0), PURPLE)), run_time=0.8)
            self.play(Transform(e_l, arr(pls[0][0], (0, 0), tuple(demo.RS[:, 0]), YELLOW, 7)),
                      run_time=0.8)
            self.wait_until(v, "goes", 2)
            self.play(Transform(e_r, arr(pls[1][0], (0, 0), (0, 1), PURPLE)), run_time=0.8)
            self.wait_until(v, "leaves")
            self.play(Transform(e_r, arr(pls[1][0], (0, 0), tuple(demo.SR[:, 0]), YELLOW, 7)),
                      run_time=0.8)
            self.wait_until(v, "different")
            self.play(Indicate(e_l, color=YELLOW), Indicate(e_r, color=YELLOW), run_time=1)
        RSm = mat(demo.RS.tolist(), 0.8)
        SRm = mat(demo.SR.tolist(), 0.8)
        g1 = VGroup(MathTex("RS", "=", font_size=44), RSm).arrange(RIGHT, buff=0.2)
        g2 = VGroup(MathTex("SR", "=", font_size=44), SRm).arrange(RIGHT, buff=0.2)
        g1.move_to([-3.6, -2.0, 0]); g2.move_to([2.9, -2.0, 0])
        c1 = SurroundingRectangle(RSm.get_columns()[0], color=YELLOW, buff=0.08)
        c2 = SurroundingRectangle(SRm.get_columns()[0], color=YELLOW, buff=0.08)
        neq = MathTex(r"RS \neq SR", r"\qquad", r"\text{in general } AB \neq BA", font_size=44,
                      color=RED).move_to(DOWN * 3.4)
        with self.voice(N["mult7"]) as v:
            self.wait_until(v, "first")
            self.play(FadeIn(g1), Create(c1), run_time=0.8)
            self.wait_until(v, "first", 2)
            self.play(FadeIn(g2), Create(c2), run_time=0.8)
            self.wait_until(v, "equal")
            self.play(Write(neq[0]), run_time=0.8)
            self.wait_until(v, "general")
            self.play(Write(neq[2]), run_time=1)


class C5_TransposeIdentity(Studio):
    def construct(self):
        self.heading("5 · Transpose and identity")
        M = mat(demo.M.tolist(), 1.0).move_to(LEFT * 4.2 + UP * 0.6)
        e = M.get_entries()
        diag = DashedLine(e[0].get_center() + UL * 0.45, e[4].get_center() + DR * 0.45,
                          color=ORANGE, stroke_width=2, stroke_opacity=0.6)
        note = VGroup(MathTex(r"A^{\top}", font_size=60, color=YELLOW),
                      T("“A transpose”", 24, MUTED)).arrange(RIGHT, buff=0.3)
        note.move_to(RIGHT * 3.6 + UP * 2.3)
        card = self.term_card("transpose", "Flip a matrix over its diagonal: its rows become "
                              "its columns.", pos=DR, width=5.0)
        with self.voice(N["tr1"]) as v:
            self.wait_until(v, "transpose")
            self.play(FadeIn(M), FadeIn(card, shift=UP * 0.3), run_time=0.8)
            self.wait_until(v, "diagonal")
            self.play(Create(diag), run_time=0.8)
            self.wait_until(v, "small")
            self.play(FadeIn(note), run_time=0.6)
        MT = mat(demo.M.T.tolist(), 1.0).move_to(RIGHT * 0.3 + UP * 0.6)
        ar = Arrow(M.get_right(), MT.get_left(), buff=0.25, color=MUTED)
        s1 = MathTex(r"2\times3", font_size=40, color=MUTED).next_to(M, DOWN, buff=0.35)
        s2 = MathTex(r"3\times2", font_size=40, color=MUTED).next_to(MT, DOWN, buff=0.35)
        with self.voice(N["tr2"]) as v:
            self.play(FadeOut(card), run_time=0.4)
            self.wait_until(v, "transpose")
            self.play(GrowArrow(ar), FadeIn(MT), run_time=0.8)
            self.wait_until(v, "first")
            self.play(M.get_rows()[0].animate.set_color(BLUE),
                      MT.get_columns()[0].animate.set_color(BLUE), run_time=0.7)
            self.wait_until(v, "second")
            self.play(M.get_rows()[1].animate.set_color(GREEN),
                      MT.get_columns()[1].animate.set_color(GREEN), run_time=0.7)
            self.wait_until(v, "flips")
            self.play(FadeIn(s1), FadeIn(s2), run_time=0.6)
        bad = mat(demo.M.reshape(3, 2).tolist(), 1.0, RED).move_to(RIGHT * 4.6 + UP * 0.4)
        badl = T("reshape: not the transpose", 22, RED).next_to(bad, DOWN, buff=0.3)
        cross = Cross(bad, stroke_color=RED, stroke_width=4)
        rule = MathTex(r"(A^{\top})_{ij}", "=", "a_{ji}", font_size=60)
        rule[0].set_color(YELLOW); rule[2].set_color(BLUE)
        rl = T("row i, column j of Aᵀ  ←  row j, column i of A", 24, MUTED)
        rg = VGroup(rule, rl).arrange(DOWN, buff=0.3).move_to(LEFT * 2.0 + DOWN * 2.6)
        with self.voice(N["tr3"]) as v:
            self.wait_until(v, "pouring")
            self.play(FadeIn(bad), FadeIn(badl), run_time=0.8)
            self.wait_until(v, "wrong")
            self.play(Create(cross), run_time=0.6)
            self.wait_until(v, "comes")
            self.play(Write(rule), FadeIn(rl), run_time=1.2)
        self.clear_body()
        I = mat([[1, 0], [0, 1]], 1.1)
        for k in (0, 3):
            I.get_entries()[k].set_color(YELLOW)
        Ig = VGroup(MathTex("I", "=", font_size=60), I).arrange(RIGHT, buff=0.25)
        Ig.move_to(LEFT * 3.8 + UP * 0.8)
        card = self.term_card("identity matrix", "Ones on the diagonal, zeros everywhere "
                              "else. It changes nothing: I x = x.", pos=DR, width=5.0)
        cols = VGroup(MathTex(r"\text{column 1} = e_1", color=BLUE, font_size=38),
                      MathTex(r"\text{column 2} = e_2", color=GREEN, font_size=38))
        cols.arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(Ig, DOWN, buff=0.5)
        ix = MathTex("I", "x", "=", "x", font_size=72).move_to(RIGHT * 2.2 + UP * 1.2)
        ix[1].set_color(BLUE); ix[3].set_color(BLUE)
        with self.voice(N["id1"]) as v:
            self.wait_until(v, "identity")
            self.play(FadeIn(Ig), FadeIn(card, shift=UP * 0.3), run_time=0.8)
            self.wait_until(v, "diagonal")
            self.play(Indicate(I.get_entries()[0], color=YELLOW),
                      Indicate(I.get_entries()[3], color=YELLOW), run_time=0.8)
            self.wait_until(v, "columns")
            self.play(FadeIn(cols), run_time=0.7)
            self.wait_until(v, "nothing")
            self.play(Write(ix), run_time=1)
        one = MathTex(r"1\cdot 5 = 5", font_size=52, color=MUTED)
        ai = MathTex(r"AI = A", r"\qquad", r"IA = A", font_size=60)
        ai[0].set_color(YELLOW); ai[2].set_color(YELLOW)
        VGroup(one, ai).arrange(DOWN, buff=0.5).move_to(RIGHT * 2.2 + DOWN * 1.0)
        with self.voice(N["id2"]) as v:
            self.play(FadeOut(card), run_time=0.4)
            self.wait_until(v, "one", 2)
            self.play(FadeIn(one), run_time=0.6)
            self.wait_until(v, "back")
            self.play(Write(ai), run_time=1)
        self.clear_body()
        ax_t = T("Axiom A · Linearity", 60, YELLOW, weight=BOLD).shift(UP * 0.9)
        l1 = MathTex(r"Ax", "=", r"x_1\,\text{col}_1", "+", r"x_2\,\text{col}_2", font_size=50)
        l1[0].set_color(YELLOW); l1[2].set_color(BLUE); l1[4].set_color(GREEN)
        l2 = T("AB = one machine after another", 30, FG)
        sub = VGroup(l1, l2).arrange(DOWN, buff=0.45).next_to(ax_t, DOWN, buff=0.6)
        with self.voice(N["axiom"]) as v:
            self.wait_until(v, "columns")
            self.play(Write(l1), run_time=1)
            self.wait_until(v, "after")
            self.play(FadeIn(l2), run_time=0.7)
            self.wait_until(v, "axiom")
            self.play(Write(ax_t), run_time=1)
        _ = sub


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · Matrices in machine learning")
        X = mat(demo.X.tolist(), 0.8, BLUE, h_buff=1.5)
        w = mat([[e] for e in demo.w], 0.8, PURPLE)
        Y = mat([[e] for e in demo.Xw], 0.8, YELLOW, h_buff=1.4)
        for e in Y.get_entries():
            e.set_opacity(0)
        q = MathTex("?", font_size=60, color=YELLOW)
        g = VGroup(MathTex("X", font_size=48), X, MathTex("w", font_size=48, color=PURPLE), w,
                   MathTex("=", font_size=56), Y).arrange(RIGHT, buff=0.3).move_to(UP * 0.8)
        g[0].next_to(X, UP, buff=0.15); g[2].next_to(w, UP, buff=0.15)
        q.move_to(Y)
        tw = T("toy weights", 20, PURPLE).next_to(w, DOWN, buff=0.3)
        ts = T("toy scores", 20, YELLOW).next_to(Y, DOWN, buff=0.3)
        with self.voice(N["ml1"]) as v:
            self.wait_until(v, "matrix")
            self.play(FadeIn(g[0]), FadeIn(X), run_time=0.8)
            self.wait_until(v, "weights")
            self.play(FadeIn(g[2]), FadeIn(w), FadeIn(tw), run_time=0.8)
            self.wait_until(v, "scores")
            self.play(FadeIn(g[4]), FadeIn(Y.get_brackets()), FadeIn(q), run_time=0.7)
        rb = SurroundingRectangle(X.get_rows()[0], color=ORANGE, buff=0.1)
        calc = MathTex(r"2\cdot120+10\cdot3-1\cdot15", "=", "240+30-15", "=", "255",
                       font_size=46).move_to(DOWN * 1.9)
        calc[4].set_color(YELLOW)
        assert demo.Xw[0] == 2 * 120 + 10 * 3 - 15 == 255
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "Row")
            self.play(Create(rb), Write(calc[0]), run_time=1)
            self.wait_until(v, "hundred", 2)
            self.play(Write(calc[1:3]), run_time=0.8)
            self.wait_until(v, "which")
            self.play(Write(calc[3:]), FadeOut(q), Y.get_entries()[0].animate.set_opacity(1),
                      run_time=0.8)
            self.wait_until(v, "other")
            self.play(Y.get_entries()[1].animate.set_opacity(1),
                      Y.get_entries()[2].animate.set_opacity(1), FadeIn(ts), FadeOut(rb), run_time=0.8)
            self.wait_until(v, "predictions")
            self.play(Indicate(Y, color=YELLOW), run_time=0.8)
        self.clear_body()
        ins = VGroup(*[Circle(0.35, color=BLUE) for _ in range(3)]).arrange(DOWN, buff=0.7)
        ins.move_to(LEFT * 5.2 + DOWN * 0.3)
        outs = VGroup(*[Circle(0.35, color=YELLOW) for _ in range(2)]).arrange(DOWN, buff=1.3)
        outs.move_to(LEFT * 1.6 + DOWN * 0.3)
        il = VGroup(*[MathTex(f"x_{i}", color=BLUE, font_size=34).move_to(c) for i, c in
                      zip((1, 2, 3), ins)])
        ol = VGroup(*[MathTex(f"y_{i}", color=YELLOW, font_size=34).move_to(c) for i, c in
                      zip((1, 2), outs)])
        edges = VGroup(*[Line(a.get_right(), b.get_left(), color=PURPLE, stroke_width=2.5)
                         for b in outs for a in ins])
        W = Matrix([[f"w_{{{i}{j}}}" for j in (1, 2, 3)] for i in (1, 2)], h_buff=1.2)
        W.set_color(PURPLE)
        yw = MathTex("y", "=", "W", "x", font_size=60)
        yw[0].set_color(YELLOW); yw[2].set_color(PURPLE); yw[3].set_color(BLUE)
        sh = MathTex(r"(2\times3)\ \text{weights}", font_size=36, color=MUTED)
        right = VGroup(yw, W, sh).arrange(DOWN, buff=0.45).move_to(RIGHT * 3.6 + DOWN * 0.3)
        with self.voice(N["ml3"]) as v:
            self.wait_until(v, "inputs")
            self.play(FadeIn(ins), FadeIn(il), run_time=0.6)
            self.wait_until(v, "outputs")
            self.play(FadeIn(outs), FadeIn(ol), run_time=0.6)
            self.wait_until(v, "weights")
            self.play(Create(edges), FadeIn(W), FadeIn(sh), run_time=1.2)
            self.wait_until(v, "multiplies")
            self.play(Write(yw), run_time=0.8)
            self.wait_until(v, "row")
            self.play(Indicate(W.get_rows()[0], color=YELLOW), Indicate(edges[:3], color=YELLOW),
                      Indicate(outs[0], color=YELLOW), run_time=1.2)
        self.clear_body()
        big = MathTex(r"2048\times 2048", font_size=80).shift(UP * 1.0)
        res = MathTex(r"=4{,}194{,}304\ \text{weights}", font_size=60, color=YELLOW)
        res.next_to(big, DOWN, buff=0.5)
        assert 2048 * 2048 == 4194304
        src = T("example layer in Raschka, Build a Large Language Model (From Scratch), p. 94",
                20, MUTED).to_edge(DOWN, buff=0.6)
        with self.voice(N["ml4"]) as v:
            self.wait_until(v, "two")
            self.play(Write(big), FadeIn(src), run_time=1.2)
            self.wait_until(v, "million")
            self.play(Write(res), run_time=1)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        q = MathTex(r"\begin{bmatrix}1&2\\0&1\end{bmatrix}", r"\begin{bmatrix}3\\1\end{bmatrix}",
                    "=", r"\;?", font_size=72)
        q[1].set_color(PURPLE); q[3].set_color(YELLOW)
        q.shift(UP * 1.5)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac1"]) as v:
            self.play(Write(q), run_time=1.5)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
            self.wait_until(v, "column")
            self.play(Indicate(q[0], color=BLUE), run_time=0.8)
        a1 = MathTex(r"\text{row 1: }", r"1\cdot3+2\cdot1=5", font_size=44)
        a2 = MathTex(r"\text{row 2: }", r"0\cdot3+1\cdot1=1", font_size=44)
        for a in (a1, a2):
            a[0].set_color(ORANGE)
        a3 = MathTex("3", r"\begin{bmatrix}1\\0\end{bmatrix}", "+", "1",
                     r"\begin{bmatrix}2\\1\end{bmatrix}", "=", r"\begin{bmatrix}5\\1\end{bmatrix}",
                     font_size=44)
        for i, c in [(0, PURPLE), (1, BLUE), (3, PURPLE), (4, GREEN), (6, YELLOW)]:
            a3[i].set_color(c)
        VGroup(a1, a2).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(LEFT * 3.2 + DOWN * 1.4)
        a3.move_to(RIGHT * 3.4 + DOWN * 1.4)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "Row")
            self.play(Write(a1), run_time=1)
            self.wait_until(v, "Row", 2)
            self.play(Write(a2), run_time=1)
            self.wait_until(v, "answer", 2)
            self.play(Indicate(VGroup(a1[1], a2[1]), color=YELLOW), run_time=0.8)
            self.wait_until(v, "column")
            self.play(Write(a3), run_time=1.2)
        self.clear_body()
        s = MathTex(r"A:\ 2\times3", r"\qquad", r"B:\ 3\times4", font_size=60).shift(UP * 1.6)
        s[0].set_color(BLUE); s[2].set_color(GREEN)
        qs = VGroup(T("AB allowed? shape?", 32, FG), T("BA allowed?", 32, FG))
        qs.arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(s, DOWN, buff=0.6)
        pause = T("⏸  pause and try", 30, MUTED).to_edge(DOWN, buff=0.6)
        with self.voice(N["prac3"]) as v:
            self.play(Write(s), run_time=1)
            self.wait_until(v, "allowed")
            self.play(FadeIn(qs[0]), run_time=0.6)
            self.wait_until(v, "allowed", 2)
            self.play(FadeIn(qs[1]), run_time=0.6)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
        ok = VGroup(MathTex(r"AB:\ (2\times", "3", r")(", "3", r"\times4)\to", r"2\times4",
                            font_size=50, color=GREEN), T("✓", 40, GREEN)).arrange(RIGHT, buff=0.4)
        ok[0][5].set_color(YELLOW)
        no = VGroup(MathTex(r"BA:\ (3\times", "4", r")(", "2", r"\times3)", font_size=50,
                            color=RED), T("✗  4 ≠ 2", 34, RED)).arrange(RIGHT, buff=0.4)
        VGroup(ok, no).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(DOWN * 0.9)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(qs), FadeOut(pause), run_time=0.5)
            self.wait_until(v, "allowed")
            self.play(FadeIn(ok[0][:5]), run_time=0.8)
            self.wait_until(v, "four")
            self.play(FadeIn(ok[0][5:]), FadeIn(ok[1]), run_time=0.6)
            self.wait_until(v, "not")
            self.play(FadeIn(no[0]), run_time=0.8)
            self.wait_until(v, "match")
            self.play(FadeIn(no[1]), Indicate(no[0][1], color=RED), Indicate(no[0][3], color=RED),
                      run_time=1)


CODE = '''import numpy as np
X = np.array([[120, 3, 15], [80, 2, 30], [150, 4, 5]])  # toy houses
A, x = np.array([[2, 1], [1, 3]]), np.array([1, 2])
print("A @ x =", A @ x)          # @ = matrix multiplication
print("columns:", 1 * A[:, 0] + 2 * A[:, 1])
S, R = np.array([[2, 0], [0, 1]]), np.array([[0, -1], [1, 0]])
print("S @ p =", S @ [1, 2], " R @ p =", R @ [1, 2])
print("R @ S =", R @ S, " S @ R =", S @ R)   # order matters
P, Q = np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])
print("P @ Q =", P @ Q, " Q @ P =", Q @ P)
M = np.array([[1, 2, 3], [4, 5, 6]])
print("M.T =", M.T, " I @ P == P:", (np.eye(2) @ P == P).all())
w = np.array([2, 10, -1])                    # toy weights
print("X @ w =", X @ w)'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=22)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[3], color=YELLOW, buff=0.06)
            self.wait_until(v, "at")
            self.play(Create(box), run_time=0.5)
            for i, w in [(6, "stretch"), (7, "orders"), (11, "transpose")]:
                self.wait_until(v, w)
                self.play(Transform(box, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                              buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=24, command="python demo.py").shift(DOWN * 0.6)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for idx, w in [((1, 2), "four"), ((3,), "stretch"), ((4,), "rotation"),
                           ((5, 6), "different"), ((11,), "scores")]:
                self.wait_until(v, w)
                self.play(*[Indicate(term[1][i], color=YELLOW) for i in idx], run_time=0.7)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["A matrix = a rectangular grid: m rows × n columns",
                            "As a table: each row is a data point",
                            "As a machine: Ax = linear combination of the columns",
                            "The columns show where e₁ and e₂ land",
                            "AB = one machine after another: (m×n)(n×p) → m×p",
                            "Order matters: AB ≠ BA · Aᵀ swaps rows and columns · I does nothing"],
                           size=27).shift(UP * 0.5)
        if pts.width > 13:
            pts.scale_to_fit_width(13)
        with self.voice(N["sum1"]) as v:
            for p in pts[:4]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 6)
        nxt = VGroup(T("Next · Video 3", 26, MUTED),
                     T("Basis & rank: how many directions are really there", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "order")
            self.play(FadeIn(pts[5], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Table", "C2_Machine", "C3_Grid", "C4_Multiply",
          "C5_TransposeIdentity", "C6_ML", "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Table": "A matrix is a table",
                  "C2_Machine": "A matrix is a machine", "C3_Grid": "Moving the plane",
                  "C4_Multiply": "Multiplying matrices", "C5_TransposeIdentity": "Transpose and identity",
                  "C6_ML": "Matrices in ML", "C7_Practice": "Try it yourself",
                  "C7_Code": "Python demo", "C8_Summary": "Summary", "C9_Credits": "Credits"}
