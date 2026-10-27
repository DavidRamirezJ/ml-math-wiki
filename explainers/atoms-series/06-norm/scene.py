"""Norm — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 6 of 16"
TITLE = "Norm — how long is a vector?"

L1C, L2C, LIC = ORANGE, BLUE, GREEN     # colour of each norm (taxi path = orange)


def arr(plane, a, b, color, width=6):
    return Arrow(plane.c2p(*a), plane.c2p(*b), buff=0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.18)


def grid_plane(xr, yr, xl, yl):
    return NumberPlane(x_range=xr, y_range=yr, x_length=xl, y_length=yl,
                       background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.35,
                                              "stroke_width": 1},
                       axis_config={"stroke_color": MUTED})


def fit(m, w):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def legend(rows, x_sym=-3.4, x_txt=-2.5, y0=0.2, dy=0.75, size=26):
    g = VGroup()
    for i, (sym, txt, color) in enumerate(rows):
        s = MathTex(sym, color=color, font_size=44).move_to([x_sym, y0 - i * dy, 0])
        t = T(txt, size, FG)
        t.move_to([x_txt, y0 - i * dy, 0], aligned_edge=LEFT)
        g.add(VGroup(s, t))
    return g


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        fit(title, 12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.2)
        qs = VGroup(T("How big is the error?", 32, RED),
                    T("How far apart are two words?", 32, GREEN),
                    T("Is this training step too large?", 32, PURPLE))
        qs.arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(LEFT * 2.6 + DOWN * 0.7)
        ans = MathTex(r"\|x\|", color=YELLOW, font_size=130).move_to(RIGHT * 4.3 + DOWN * 0.5)
        lab = T("one number: the size", 24, MUTED).next_to(ans, DOWN, buff=0.3)
        ar = Arrow(qs.get_right() + RIGHT * 0.2, ans.get_left() + LEFT * 0.2, buff=0,
                   color=MUTED)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            for q, w in zip(qs, ["error", "words", "step"]):
                self.wait_until(v, w)
                self.play(FadeIn(q, shift=RIGHT * 0.2), run_time=0.7)
            self.wait_until(v, "norm")
            self.play(GrowArrow(ar), Write(ans), FadeIn(lab), run_time=1.2)
        items = VGroup(
            VGroup(MathTex(r"\sqrt{x\cdot x}", color=YELLOW, font_size=56), T("length", 24, MUTED)),
            VGroup(MathTex(r"L_1\;\;L_2\;\;L_\infty", color=BLUE, font_size=50),
                   T("three ways", 24, MUTED)),
            VGroup(MathTex(r"\|u+v\|\le\|u\|+\|v\|", color=GREEN, font_size=42),
                   T("rules", 24, MUTED)),
            VGroup(MathTex(r"\|x-y\|", color=ORANGE, font_size=56), T("distance", 24, MUTED)))
        for it in items:
            it.arrange(DOWN, buff=0.3)
        items.arrange(RIGHT, buff=0.8).shift(DOWN * 0.5)
        fit(items, 13)
        pre = T("you need: video 1 (vector) · video 5 (inner product)", 26, MUTED)
        pre.to_edge(DOWN, buff=0.7)
        with self.voice(N["intro2"]) as v:
            self.play(FadeOut(qs), FadeOut(ar), FadeOut(ans), FadeOut(lab), run_time=0.6)
            for it, w in zip(items, ["length", "ways", "rules", "distance"]):
                self.wait_until(v, w)
                self.play(FadeIn(it, shift=UP * 0.2), run_time=0.6)
            self.wait_until(v, "need")
            self.play(FadeIn(pre), run_time=0.6)


class C1_Length(Studio):
    def construct(self):
        self.heading("1 · Length from the inner product")
        plane = grid_plane([-1, 5, 1], [-1, 5, 1], 5, 5).shift(LEFT * 3.6 + DOWN * 0.7)
        a = arr(plane, (0, 0), (3, 4), BLUE)
        tip = T("(3, 4)", 26, BLUE).next_to(plane.c2p(3, 4), UR, buff=0.08)
        hx = DashedLine(plane.c2p(0, 0), plane.c2p(3, 0), color=ORANGE, stroke_width=4)
        hy = DashedLine(plane.c2p(3, 0), plane.c2p(3, 4), color=ORANGE, stroke_width=4)
        lx = T("3", 26, ORANGE).next_to(hx, DOWN, buff=0.12)
        ly = T("4", 26, ORANGE).next_to(hy, RIGHT, buff=0.12)
        q = T("?", 34, YELLOW, weight=BOLD).move_to(plane.c2p(1.2, 2.4))
        with self.voice(N["len1"]) as v:
            self.play(Create(plane), run_time=1)
            self.wait_until(v, "blue")
            self.play(GrowArrow(a), FadeIn(tip), run_time=1)
            self.wait_until(v, "right")
            self.play(Create(hx), FadeIn(lx), run_time=0.7)
            self.wait_until(v, "up")
            self.play(Create(hy), FadeIn(ly), run_time=0.7)
            self.wait_until(v, "long", 2)
            self.play(Indicate(a, color=YELLOW), FadeIn(q), run_time=1)
        e1 = MathTex(r"\|x\|^2", "=", "3^2", "+", "4^2", font_size=52)
        e1[0].set_color(YELLOW); e1[2].set_color(ORANGE); e1[4].set_color(ORANGE)
        e2 = MathTex("=", "9", "+", "16", "=", "25", font_size=52)
        e3 = MathTex(r"\|x\|", "=", r"\sqrt{25}", "=", "5", font_size=52)
        e3[0].set_color(YELLOW); e3[4].set_color(YELLOW)
        eqs = VGroup(e1, e2, e3).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        eqs.move_to(RIGHT * 3.3 + UP * 0.2)
        five = T("5", 34, YELLOW, weight=BOLD).move_to(q)
        with self.voice(N["len2"]) as v:
            self.wait_until(v, "Pythagoras")
            self.play(Write(e1), run_time=1.2)
            self.wait_until(v, "nine")
            self.play(FadeIn(e2[:2]), run_time=0.5)
            self.wait_until(v, "sixteen")
            self.play(FadeIn(e2[2:4]), run_time=0.5)
            self.wait_until(v, "twenty")
            self.play(FadeIn(e2[4:]), run_time=0.5)
            self.wait_until(v, "root")
            self.play(Write(e3), run_time=1)
            self.wait_until(v, "length")
            self.play(Transform(q, five), Indicate(e3[4], color=YELLOW), run_time=0.8)
        f1 = MathTex(r"3\cdot 3", "+", r"4\cdot 4", "=", r"x\cdot x", font_size=52)
        f1[0].set_color(ORANGE); f1[2].set_color(ORANGE); f1[4].set_color(BLUE)
        f2 = MathTex(r"\|x\|", "=", r"\sqrt{x\cdot x}", font_size=66)
        f2[0].set_color(YELLOW); f2[2].set_color(BLUE)
        VGroup(f1, f2).arrange(DOWN, buff=0.55).move_to(RIGHT * 3.4 + DOWN * 0.1)
        rem = self.reminder_card("dot product", "Multiply matching components and add: "
                                 "x·y = x₁y₁ + x₂y₂ + …  (video 5)", pos=DR, width=5.4)
        card = self.term_card("norm", "A rule that gives the size (length) of a vector "
                              "as one number that is never negative.", pos=DR, width=5.4)
        with self.voice(N["len3"]) as v:
            self.play(FadeOut(eqs), run_time=0.5)
            self.play(Write(f1), run_time=1.2)
            self.wait_until(v, "dot")
            self.play(FadeIn(rem, shift=DOWN * 0.2), run_time=0.6)
            self.wait_until(v, "root")
            self.play(Write(f2), run_time=1)
            self.wait_until(v, "norm")
            self.play(FadeOut(rem), FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "bars")
            self.play(Indicate(f2[0], color=YELLOW), run_time=0.8)
        self.clear_body()
        big = MathTex(r"\|x\|", "=", r"\sqrt{x_1^2+x_2^2+\cdots+x_n^2}", "=",
                      r"\sqrt{\textstyle\sum_{i=1}^{n} x_i^2}", font_size=58)
        big[0].set_color(YELLOW); big[2].set_color(ORANGE); big[4].set_color(ORANGE)
        fit(big, 12.8).move_to(UP * 1.6)
        leg = legend([(r"\|x\|", "the norm (length) of x", YELLOW),
                      (r"\sqrt{\;\;}", "take the square root at the end", FG),
                      (r"x_i^2", "square each component", ORANGE),
                      (r"\textstyle\sum_{i=1}^{n}", "add them all", FG),
                      (r"i", "counts the components: 1, 2, …, n", MUTED)],
                     x_sym=-3.6, x_txt=-2.4, y0=0.1, dy=0.72)
        with self.voice(N["len4"]) as v:
            self.play(Write(big), run_time=1.5)
            for row, (w, n) in zip(leg, [("bars", 1), ("root", 1), ("square", 2),
                                         ("add", 1), ("counts", 1)]):
                self.wait_until(v, w, n)
                self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.5)
            self.wait_until(v, "Pythagoras")
            self.play(Indicate(big[2], color=YELLOW), run_time=0.8)
            self.wait_until(v, "any")
            self.play(Indicate(big[4], color=YELLOW), run_time=0.8)
        self.clear_body()
        vec = MathTex(r"x=\begin{bmatrix}2\\3\\6\end{bmatrix}", color=BLUE, font_size=60)
        vec.move_to(LEFT * 5.3 + DOWN * 0.6)
        eq = MathTex(r"\|x\|", "=", r"\sqrt{2^2+3^2+6^2}", "=", r"\sqrt{4+9+36}", "=",
                     r"\sqrt{%d}" % round(demo.n3 ** 2), "=", "%d" % round(demo.n3),
                     font_size=48)
        eq[0].set_color(YELLOW); eq[8].set_color(YELLOW)
        fit(eq, 12.8).move_to(UP * 1.9)
        s = 0.34
        sq = VGroup(*[Square(k * s, color=ORANGE, fill_color=ORANGE, fill_opacity=0.3)
                      for k in (2, 3, 6)]).arrange(RIGHT, buff=0.3, aligned_edge=DOWN)
        sq.move_to(LEFT * 1.4 + DOWN * 0.8)
        sql = VGroup(*[T(str(k * k), 22, FG).move_to(b) for k, b in zip((2, 3, 6), sq)])
        big7 = Square(7 * s, color=YELLOW, fill_color=YELLOW, fill_opacity=0.2)
        big7.next_to(sq, RIGHT, buff=1.0).align_to(sq, DOWN)
        b7 = T("49 = 7 × 7", 26, YELLOW).move_to(big7)
        note = T("a million components: the same steps", 24, MUTED).to_edge(DOWN, buff=0.45)
        with self.voice(N["len5"]) as v:
            self.wait_until(v, "two")
            self.play(Write(vec), run_time=0.8)
            self.wait_until(v, "Square")
            self.play(FadeIn(sq), FadeIn(sql), Write(eq[:3]), run_time=1.2)
            self.wait_until(v, "Add")
            self.play(Write(eq[3:5]), run_time=0.8)
            self.wait_until(v, "forty")
            self.play(TransformFromCopy(sq, big7), Write(eq[5:7]), run_time=1.2)
            self.play(FadeIn(b7), run_time=0.4)
            self.wait_until(v, "seven")
            self.play(Write(eq[7:]), run_time=0.6)
            self.wait_until(v, "million")
            self.play(FadeIn(note), run_time=0.6)
        self.clear_body()
        ax = T("Axiom B · Inner product", 56, YELLOW, weight=BOLD).shift(UP * 1.3)
        fm = MathTex(r"\langle x,x\rangle", r"\;\longrightarrow\;", r"\|x\|=\sqrt{\langle x,x\rangle}",
                     font_size=56).next_to(ax, DOWN, buff=0.6)
        fm[0].set_color(BLUE); fm[2].set_color(YELLOW)
        sub = T("inner product → lengths and angles · the norm is the length part", 26, MUTED)
        sub.next_to(fm, DOWN, buff=0.5)
        names = T("also called: the L2 norm · the Euclidean norm", 30, BLUE).next_to(sub, DOWN, buff=0.6)
        with self.voice(N["len6"]) as v:
            self.wait_until(v, "inner")
            self.play(Write(ax), run_time=1)
            self.wait_until(v, "lengths")
            self.play(Write(fm), FadeIn(sub), run_time=1.2)
            self.wait_until(v, "length")
            self.play(Indicate(fm[2], color=YELLOW), run_time=0.8)
            self.wait_until(v, "names")
            self.play(FadeIn(names, shift=UP * 0.2), run_time=0.6)


class C2_OtherNorms(Studio):
    def construct(self):
        self.heading("2 · Other ways to measure size")
        plane = grid_plane([-1, 5, 1], [-1, 5, 1], 5, 5).shift(LEFT * 3.6 + DOWN * 0.7)
        u = plane.c2p(1, 0)[0] - plane.c2p(0, 0)[0]
        blocks = VGroup(*[Square(u * 0.62, stroke_width=0, fill_color=MUTED, fill_opacity=0.28)
                          .move_to(plane.c2p(i + 0.5, j + 0.5))
                          for i in range(-1, 5) for j in range(-1, 5)])
        straight = arr(plane, (0, 0), (3, 4), L2C)
        sl = T("5", 28, L2C, weight=BOLD).move_to(plane.c2p(1.1, 2.5))
        path = VMobject(color=L1C, stroke_width=7).set_points_as_corners(
            [plane.c2p(0, 0), plane.c2p(3, 0), plane.c2p(3, 4)])
        car = Dot(plane.c2p(0, 0), color=L1C, radius=0.12)
        t1 = VGroup(T("taxi: 3 + 4 = 7 blocks", 30, L1C), T("straight line: 5", 30, L2C))
        t1.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.4 + DOWN * 2.6)
        with self.voice(N["oth1"]) as v:
            self.play(Create(plane), FadeIn(blocks), run_time=1)
            self.wait_until(v, "straight")
            self.play(GrowArrow(straight), FadeIn(sl), run_time=0.8)
            self.wait_until(v, "taxi")
            self.play(FadeIn(car, scale=0.5), run_time=0.4)
            self.wait_until(v, "drives")
            self.play(Create(path), MoveAlongPath(car, path.copy()), run_time=2.5)
            self.wait_until(v, "seven")
            self.play(FadeIn(t1), run_time=0.6)
        f1 = MathTex(r"\|x\|_1", "=", r"|x_1|+|x_2|+\cdots+|x_n|", font_size=46)
        f1[0].set_color(L1C)
        ab = VGroup(MathTex(r"|-3|=3", font_size=40), T("absolute value: drop the minus sign", 22, MUTED)
                    ).arrange(RIGHT, buff=0.3)
        ex1 = MathTex(r"\|(3,4)\|_1", "=", r"|3|+|4|", "=", "%g" % demo.l1, font_size=46)
        ex1[0].set_color(L1C); ex1[4].set_color(L1C)
        col = VGroup(f1, ab, ex1).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        fit(col, 6.8).move_to(RIGHT * 3.5 + UP * 1.5)
        with self.voice(N["oth2"]) as v:
            self.play(FadeOut(t1), run_time=0.4)
            self.wait_until(v, "Manhattan")
            self.play(Write(f1), run_time=1.2)
            self.wait_until(v, "absolute", 2)
            self.play(FadeIn(ab), run_time=0.6)
            self.wait_until(v, "minus")
            self.play(Indicate(ab[0], color=YELLOW), run_time=0.8)
            self.wait_until(v, "seven")
            self.play(Write(ex1), run_time=1)
        fi = MathTex(r"\|x\|_\infty", "=", r"\max_i |x_i|", font_size=46)
        fi[0].set_color(LIC)
        exi = MathTex(r"\|(3,4)\|_\infty", "=", r"\max(3,4)", "=", "%g" % demo.linf, font_size=46)
        exi[0].set_color(LIC); exi[4].set_color(LIC)
        col2 = VGroup(fi, exi).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        col2.next_to(col, DOWN, buff=0.45, aligned_edge=LEFT)
        summ = VGroup(MathTex(r"L_1 = %g" % demo.l1, color=L1C, font_size=44),
                      MathTex(r"L_2 = %g" % demo.l2, color=L2C, font_size=44),
                      MathTex(r"L_\infty = %g" % demo.linf, color=LIC, font_size=44)
                      ).arrange(RIGHT, buff=0.6)
        summ.next_to(col2, DOWN, buff=0.5).set_x(3.5)
        sbox = SurroundingRectangle(summ, color=YELLOW, buff=0.18, corner_radius=0.1)
        with self.voice(N["oth3"]) as v:
            self.wait_until(v, "infinity")
            self.play(Write(fi), run_time=1)
            self.wait_until(v, "four", 2)
            self.play(Write(exi), run_time=1)
            self.wait_until(v, "sizes")
            self.play(FadeIn(summ), Create(sbox), run_time=0.8)
            self.wait_until(v, "real")
            self.play(Indicate(summ, color=YELLOW), run_time=0.8)
        self.clear_body()
        pl = grid_plane([-1.5, 1.5, 0.5], [-1.5, 1.5, 0.5], 5.4, 5.4).shift(LEFT * 3.4 + DOWN * 0.6)
        r = pl.c2p(1, 0)[0] - pl.c2p(0, 0)[0]
        circ = Circle(radius=r, color=L2C, stroke_width=5).move_to(pl.c2p(0, 0))
        diam = Polygon(pl.c2p(1, 0), pl.c2p(0, 1), pl.c2p(-1, 0), pl.c2p(0, -1),
                       color=L1C, stroke_width=5)
        sqr = Polygon(pl.c2p(1, 1), pl.c2p(-1, 1), pl.c2p(-1, -1), pl.c2p(1, -1),
                      color=LIC, stroke_width=5)
        one = T("1", 22, MUTED).next_to(pl.c2p(1, 0), DR, buff=0.08)
        rows = VGroup(
            VGroup(MathTex(r"\|x\|_2=1", color=L2C, font_size=46), T("circle", 28, FG)),
            VGroup(MathTex(r"\|x\|_1=1", color=L1C, font_size=46), T("diamond", 28, FG)),
            VGroup(MathTex(r"\|x\|_\infty=1", color=LIC, font_size=46), T("square", 28, FG)))
        for rw in rows:
            rw.arrange(RIGHT, buff=0.4)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.6).move_to(RIGHT * 3.4 + DOWN * 0.2)
        with self.voice(N["oth4"]) as v:
            self.play(Create(pl), FadeIn(one), run_time=1)
            self.wait_until(v, "circle")
            self.play(Create(circ), FadeIn(rows[0]), run_time=1)
            self.wait_until(v, "diamond")
            self.play(TransformFromCopy(circ, diam), FadeIn(rows[1]), run_time=1.2)
            self.wait_until(v, "square")
            self.play(TransformFromCopy(diam, sqr), FadeIn(rows[2]), run_time=1.2)
            self.wait_until(v, "own")
            self.play(Indicate(VGroup(circ, diam, sqr), color=YELLOW), run_time=1)
        n1 = VGroup(MathTex(r"L_2:\ \sqrt{x\cdot x}\ \checkmark", color=L2C, font_size=42),
                    T("comes from an inner product", 24, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        n2 = VGroup(MathTex(r"L_1,\ L_\infty", color=RED, font_size=42),
                    T("real norms, but from no inner product", 24, FG)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        n3 = T("(test: the parallelogram law fails)", 22, MUTED)
        n4 = T("→ sizes, but no angles", 28, YELLOW)
        notes = VGroup(n1, n2, n3, n4).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        notes.move_to(RIGHT * 3.5 + DOWN * 0.2)
        with self.voice(N["oth5"]) as v:
            self.play(FadeOut(rows), run_time=0.5)
            self.wait_until(v, "Only")
            self.play(FadeIn(n1), Indicate(circ, color=L2C), run_time=1)
            self.wait_until(v, "infinity", 1)
            self.play(FadeIn(n2), Indicate(diam, color=RED), Indicate(sqr, color=RED), run_time=1)
            self.wait_until(v, "parallelogram")
            self.play(FadeIn(n3), run_time=0.6)
            self.wait_until(v, "angles")
            self.play(FadeIn(n4), run_time=0.6)


class C3_Rules(Studio):
    def construct(self):
        self.heading("3 · The three rules of every norm")
        plane = grid_plane([-4, 4, 1], [-3, 3, 1], 6.0, 4.5).shift(LEFT * 3.5 + DOWN * 0.8)
        arrows = VGroup(arr(plane, (0, 0), (3, 1), BLUE), arr(plane, (0, 0), (-2, 2), BLUE),
                        arr(plane, (0, 0), (-1, -2), BLUE))
        pos = VGroup(*[T("> 0", 22, GREEN).next_to(plane.c2p(*p), d, buff=0.08)
                       for p, d in [((3, 1), UR), ((-2, 2), UL), ((-1, -2), DL)]])
        zero = Dot(plane.c2p(0, 0), color=YELLOW, radius=0.1)
        r1 = MathTex(r"\|x\|\ge 0", font_size=56)
        r1b = MathTex(r"\|x\|=0", r"\iff", r"x=0", font_size=56)
        r1b[2].set_color(YELLOW)
        rr = VGroup(T("Rule 1 · positive", 30, YELLOW, weight=BOLD), r1, r1b)
        rr.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(RIGHT * 3.6 + UP * 0.3)
        with self.voice(N["rul1"]) as v:
            self.play(Create(plane), *[GrowArrow(a) for a in arrows], run_time=1.2)
            self.play(FadeIn(rr[0]), run_time=0.5)
            self.wait_until(v, "never")
            self.play(Write(r1), run_time=0.8)
            self.wait_until(v, "zero")
            self.play(Write(r1b), FadeIn(zero, scale=0.5), run_time=1)
            self.wait_until(v, "positive")
            self.play(FadeIn(pos), Indicate(arrows, color=GREEN), run_time=1)
        self.clear_body()
        pl = grid_plane([-8, 8, 2], [-10, 10, 2], 4.4, 5.5).shift(LEFT * 3.6 + DOWN * 0.75)
        c = ValueTracker(1.0)
        base = arr(pl, (0, 0), (3, 4), BLUE, 4).set_opacity(0.45)
        live = always_redraw(lambda: arr(pl, (0, 0), (3 * c.get_value(), 4 * c.get_value()),
                                         YELLOW, 7) if abs(c.get_value()) > 0.06 else
                             Dot(pl.c2p(0, 0), color=YELLOW, radius=0.1))
        rule = MathTex(r"\|c\,x\|", "=", r"|c|", r"\,\|x\|", font_size=60)
        rule[0].set_color(YELLOW); rule[2].set_color(PURPLE)
        rule.move_to(RIGHT * 3.4 + UP * 1.5)
        rlab = T("Rule 2 · scaling", 28, YELLOW, weight=BOLD).next_to(rule, UP, buff=0.3)
        read = always_redraw(lambda: VGroup(
            T(f"c = {c.get_value():+.1f}", 32, PURPLE, font=MONO),
            T(f"length = {abs(c.get_value()):.1f} × 5 = {5 * abs(c.get_value()):.1f}", 30,
              YELLOW, font=MONO)).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            .move_to(RIGHT * 3.4 + DOWN * 0.1))
        ex = VGroup(MathTex(r"-2\begin{bmatrix}3\\4\end{bmatrix}=\begin{bmatrix}-6\\-8\end{bmatrix}",
                            font_size=42),
                    MathTex(r"\sqrt{36+64}=10=|-2|\cdot 5", font_size=42))
        ex.arrange(DOWN, buff=0.3).move_to(RIGHT * 3.4 + DOWN * 2.2)
        with self.voice(N["rul2"]) as v:
            self.play(Create(pl), GrowArrow(base), run_time=1)
            self.add(live, read)
            self.wait_until(v, "absolute")
            self.play(FadeIn(rlab), Write(rule), run_time=1)
            self.wait_until(v, "doubles")
            self.play(c.animate.set_value(2), run_time=1.2)
            self.wait_until(v, "flips")
            self.play(c.animate.set_value(-2), run_time=1.8)
            self.wait_until(v, "six")
            self.play(FadeIn(ex[0]), run_time=0.8)
            self.wait_until(v, "ten")
            self.play(FadeIn(ex[1]), run_time=0.8)
        self.clear_body()
        p3 = grid_plane([-1, 8, 1], [-1, 5, 1], 6.3, 4.2).shift(LEFT * 3.5 + DOWN * 0.9)
        ua = arr(p3, (0, 0), (3, 4), BLUE)
        va = arr(p3, (3, 4), (7, 1), GREEN)
        sa = arr(p3, (0, 0), (7, 1), YELLOW, 7)
        ul = MathTex("u", color=BLUE, font_size=40).next_to(p3.c2p(1.5, 2), UL, buff=0.1)
        vl = MathTex("v", color=GREEN, font_size=40).next_to(p3.c2p(5, 2.5), UR, buff=0.1)
        sl = MathTex("u+v", color=YELLOW, font_size=36).next_to(p3.c2p(4.5, 0.6), DOWN, buff=0.15)
        tri = MathTex(r"\|u+v\|", r"\le", r"\|u\|", "+", r"\|v\|", font_size=58)
        tri[0].set_color(YELLOW); tri[2].set_color(BLUE); tri[4].set_color(GREEN)
        tri.move_to(RIGHT * 3.6 + UP * 2.0)
        tlab = T("straight route ≤ detour", 24, MUTED).next_to(tri, DOWN, buff=0.25)
        with self.voice(N["rul3"]) as v:
            self.play(Create(p3), run_time=0.8)
            self.wait_until(v, "along")
            self.play(GrowArrow(ua), FadeIn(ul), run_time=0.8)
            self.wait_until(v, "along", 2)
            self.play(GrowArrow(va), FadeIn(vl), run_time=0.8)
            self.wait_until(v, "straight")
            self.play(GrowArrow(sa), FadeIn(sl), run_time=0.8)
            self.wait_until(v, "never")
            self.play(FadeIn(tlab), run_time=0.5)
            self.wait_until(v, "symbols")
            self.play(Write(tri), run_time=1)
        lines = VGroup(MathTex(r"\|u\|=\|v\|=5", r"\;\Rightarrow\;", r"\text{detour}=10", font_size=38),
                       MathTex(r"u+v=(7,\,1)", font_size=38),
                       MathTex(r"\|u+v\|=\sqrt{49+1}=\sqrt{50}\approx %.2f" % demo.tri_left,
                               font_size=38),
                       MathTex(r"%.2f\le %g\ \checkmark" % (demo.tri_left, demo.tri_right),
                               color=GREEN, font_size=44))
        lines[1].set_color(YELLOW)
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(tlab, DOWN, buff=0.45)
        fit(lines, 6.6).set_x(3.6)
        with self.voice(N["rul4"]) as v:
            for ln, w in zip(lines, ["five", "sum", "fifty", "less"]):
                self.wait_until(v, w)
                self.play(FadeIn(ln, shift=RIGHT * 0.2), run_time=0.7)
            self.play(Indicate(sa, color=YELLOW), run_time=0.8)


class C4_Unit(Studio):
    def construct(self):
        self.heading("4 · Unit vectors and normalizing")
        pl = grid_plane([-1.5, 1.5, 0.5], [-1.5, 1.5, 0.5], 5, 5).shift(LEFT * 3.5 + DOWN * 0.6)
        r = pl.c2p(1, 0)[0] - pl.c2p(0, 0)[0]
        circ = DashedVMobject(Circle(radius=r, color=BLUE).move_to(pl.c2p(0, 0)), num_dashes=40)
        ang = ValueTracker(0.6)
        live = always_redraw(lambda: arr(pl, (0, 0), (np.cos(ang.get_value()),
                                                      np.sin(ang.get_value())), YELLOW, 7))
        card = self.term_card("unit vector", "A vector whose norm is exactly 1. "
                              "It keeps only a direction.", pos=DR, width=5.4)
        with self.voice(N["unit1"]) as v:
            self.play(Create(pl), run_time=0.8)
            self.wait_until(v, "direction")
            self.add(live)
            self.play(ang.animate.set_value(1.2), run_time=0.8)
            self.wait_until(v, "unit")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "circle")
            self.play(Create(circ), ang.animate.set_value(1.2 + 2 * PI), run_time=3)
        self.clear_body()
        p2 = grid_plane([-1, 5, 1], [-1, 5, 1], 5, 5).shift(LEFT * 3.6 + DOWN * 0.7)
        r2 = p2.c2p(1, 0)[0] - p2.c2p(0, 0)[0]
        c2 = DashedVMobject(Circle(radius=r2, color=MUTED).move_to(p2.c2p(0, 0)), num_dashes=24)
        xa = arr(p2, (0, 0), (3, 4), BLUE)
        ua = arr(p2, (0, 0), tuple(demo.unit), YELLOW, 8)
        ul = MathTex(r"\frac{x}{\|x\|}", color=YELLOW, font_size=40).move_to(p2.c2p(1.5, 0.45))
        xl = MathTex("x", color=BLUE, font_size=40).next_to(p2.c2p(3, 4), UR, buff=0.08)
        eq = MathTex(r"\frac{x}{\|x\|}", "=", r"\frac{1}{5}\begin{bmatrix}3\\4\end{bmatrix}", "=",
                     r"\begin{bmatrix}%.1f\\%.1f\end{bmatrix}" % tuple(demo.unit), font_size=54)
        eq[0].set_color(YELLOW); eq[4].set_color(YELLOW)
        eq.move_to(RIGHT * 3.4 + UP * 1.4)
        with self.voice(N["unit2"]) as v:
            self.play(Create(p2), Create(c2), GrowArrow(xa), FadeIn(xl), run_time=1.2)
            self.wait_until(v, "divide")
            self.play(Write(eq[0]), run_time=0.8)
            self.wait_until(v, "five")
            self.play(Write(eq[1:3]), run_time=0.8)
            self.wait_until(v, "zero")
            self.play(Write(eq[3:]), TransformFromCopy(xa, ua), FadeIn(ul), run_time=1.2)
            self.wait_until(v, "direction")
            self.play(Indicate(VGroup(xa, ua), color=YELLOW), run_time=0.8)
        chk = MathTex(r"0.6^2+0.8^2", "=", "0.36+0.64", "=", "1", font_size=46)
        chk[4].set_color(GREEN)
        ok = MathTex(r"\Big\|\tfrac{x}{\|x\|}\Big\|=\sqrt{1}=%g\ \checkmark" % round(np.linalg.norm(demo.unit), 6),
                     color=GREEN, font_size=46)
        why = T("rule 2: divide by 5 → length divided by 5", 24, MUTED)
        VGroup(chk, ok, why).arrange(DOWN, buff=0.4).move_to(RIGHT * 3.4 + DOWN * 1.4)
        with self.voice(N["unit3"]) as v:
            self.wait_until(v, "six")
            self.play(Write(chk[0]), run_time=0.8)
            self.wait_until(v, "sum")
            self.play(Write(chk[1:]), run_time=1)
            self.wait_until(v, "works")
            self.play(Write(ok), run_time=0.8)
            self.wait_until(v, "rule")
            self.play(FadeIn(why), run_time=0.6)
        self.clear_body()
        zr = MathTex(r"\frac{0}{\|0\|}", "=", r"\frac{0}{0}", font_size=80, color=RED)
        zr.move_to(LEFT * 3.6 + UP * 0.6)
        cr = Cross(zr[2], stroke_color=RED, stroke_width=6)
        nd = T("the zero vector has no direction", 28, RED).next_to(zr, DOWN, buff=0.6)
        rem = self.reminder_card("cosine similarity", "How much two vectors point the same "
                                 "way: x·y divided by both lengths (video 5).", pos=DR, width=5.4)
        cs = MathTex(r"\cos\theta", "=", r"\frac{x}{\|x\|}", r"\cdot", r"\frac{y}{\|y\|}",
                     font_size=58)
        cs[2].set_color(BLUE); cs[4].set_color(GREEN)
        cs.move_to(RIGHT * 3.6 + UP * 1.0)
        with self.voice(N["unit4"]) as v:
            self.wait_until(v, "zero")
            self.play(Write(zr), run_time=1)
            self.wait_until(v, "cannot")
            self.play(Create(cr), run_time=0.6)
            self.wait_until(v, "direction")
            self.play(FadeIn(nd), run_time=0.6)
            self.wait_until(v, "cosine")
            self.play(FadeIn(rem, shift=DOWN * 0.2), Write(cs), run_time=1.2)


class C5_Distance(Studio):
    def construct(self):
        self.heading("5 · Distance = norm of the difference")
        pl = grid_plane([-1, 6, 1], [-1, 6, 1], 5.2, 5.2).shift(LEFT * 3.6 + DOWN * 0.6)
        p = Dot(pl.c2p(1, 1), color=BLUE, radius=0.1)
        q = Dot(pl.c2p(4, 5), color=GREEN, radius=0.1)
        pll = MathTex("p", color=BLUE, font_size=40).next_to(p, DL, buff=0.08)
        qll = MathTex("q", color=GREEN, font_size=40).next_to(q, UR, buff=0.08)
        pq = arr(pl, (1, 1), (4, 5), YELLOW)
        hx = DashedLine(pl.c2p(1, 1), pl.c2p(4, 1), color=ORANGE, stroke_width=4)
        hy = DashedLine(pl.c2p(4, 1), pl.c2p(4, 5), color=ORANGE, stroke_width=4)
        lx = T("3", 24, ORANGE).next_to(hx, DOWN, buff=0.1)
        ly = T("4", 24, ORANGE).next_to(hy, RIGHT, buff=0.1)
        e1 = MathTex(r"q-p", "=", r"\begin{bmatrix}4-1\\5-1\end{bmatrix}", "=",
                     r"\begin{bmatrix}3\\4\end{bmatrix}", font_size=48)
        e1[0].set_color(YELLOW); e1[4].set_color(YELLOW)
        e2 = MathTex(r"d(p,q)", "=", r"\|q-p\|", "=", "5", font_size=52)
        e2[0].set_color(YELLOW); e2[4].set_color(YELLOW)
        VGroup(e1, e2).arrange(DOWN, buff=0.5).move_to(RIGHT * 3.4 + UP * 1.0)
        card = self.term_card("distance", "How far apart two vectors are: the norm of "
                              "their difference, x − y.", pos=DR, width=5.4)
        with self.voice(N["dist1"]) as v:
            self.play(Create(pl), run_time=0.8)
            self.wait_until(v, "p")
            self.play(FadeIn(p), FadeIn(pll), run_time=0.5)
            self.wait_until(v, "q")
            self.play(FadeIn(q), FadeIn(qll), run_time=0.5)
            self.wait_until(v, "arrow")
            self.play(GrowArrow(pq), Write(e1), run_time=1.2)
            self.wait_until(v, "length")
            self.play(Create(hx), Create(hy), FadeIn(lx), FadeIn(ly), Write(e2), run_time=1)
            self.wait_until(v, "distance", 2)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
        h = MathTex(r"d(x,y)", "=", r"\|x-y\|", font_size=58)
        h[0].set_color(YELLOW)
        w1 = MathTex(r"x-y", "=", r"\begin{bmatrix}1\\2\\3\end{bmatrix}-\begin{bmatrix}-1\\-1\\0\end{bmatrix}",
                     "=", r"\begin{bmatrix}2\\3\\3\end{bmatrix}", font_size=40)
        w1[4].set_color(ORANGE)
        w2 = MathTex(r"d", "=", r"\sqrt{4+9+9}", "=", r"\sqrt{22}\approx %.2f" % demo.dist,
                     font_size=44)
        w2[0].set_color(YELLOW); w2[4].set_color(YELLOW)
        col = VGroup(h, w1, w2).arrange(DOWN, buff=0.45)
        fit(col, 6.6).move_to(RIGHT * 3.5 + DOWN * 0.5)
        with self.voice(N["dist2"]) as v:
            self.play(FadeOut(card), FadeOut(e1), FadeOut(e2), run_time=0.5)
            self.wait_until(v, "formula")
            self.play(Write(h), run_time=1)
            self.wait_until(v, "three")
            self.play(Write(w1[:3]), run_time=1.2)
            self.wait_until(v, "difference")
            self.play(Write(w1[3:]), run_time=0.8)
            self.wait_until(v, "squares")
            self.play(Write(w2[:3]), run_time=0.8)
            self.wait_until(v, "twenty")
            self.play(Write(w2[3:]), run_time=0.8)
        self.clear_body()
        top = T("metric = a rule for distance", 32, YELLOW, weight=BOLD).move_to(LEFT * 3.0 + UP * 2.2)
        rules = VGroup(
            VGroup(MathTex(r"d(x,y)\ge 0,\quad d(x,y)=0\iff x=y", font_size=38),
                   T("never negative; zero only for the same point", 22, MUTED)),
            VGroup(MathTex(r"d(x,y)=d(y,x)", font_size=38), T("symmetric", 22, MUTED)),
            VGroup(MathTex(r"d(x,z)\le d(x,y)+d(y,z)", font_size=38),
                   T("a detour is never shorter", 22, MUTED)))
        for rw in rules:
            rw.arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        rules.arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(top, DOWN, buff=0.5)
        rules.align_to(top, LEFT)
        X, Y, Z = RIGHT * 2.4 + DOWN * 1.8, RIGHT * 4.6 + UP * 1.0, RIGHT * 6.3 + DOWN * 1.6
        dots = VGroup(Dot(X, color=BLUE), Dot(Y, color=GREEN), Dot(Z, color=YELLOW))
        labs = VGroup(MathTex("x", color=BLUE).next_to(X, DL, buff=0.1),
                      MathTex("y", color=GREEN).next_to(Y, UP, buff=0.1),
                      MathTex("z", color=YELLOW).next_to(Z, DR, buff=0.1))
        direct = Line(X, Z, color=YELLOW, stroke_width=5)
        detour = VGroup(Line(X, Y, color=MUTED, stroke_width=4), Line(Y, Z, color=MUTED, stroke_width=4))
        with self.voice(N["dist3"]) as v:
            self.wait_until(v, "metric")
            self.play(FadeIn(top), FadeIn(dots), FadeIn(labs), run_time=0.8)
            self.wait_until(v, "negative")
            self.play(FadeIn(rules[0]), run_time=0.7)
            self.wait_until(v, "equals")
            self.play(FadeIn(rules[1]), Create(Line(X, Y, color=GREEN, stroke_width=3)), run_time=0.7)
            self.wait_until(v, "detour")
            self.play(FadeIn(rules[2]), Create(direct), Create(detour), run_time=1)
        self.clear_body()
        p4 = grid_plane([-1, 5, 1], [-1, 5, 1], 4.8, 4.8).shift(LEFT * 3.8 + DOWN * 0.7)
        xa = arr(p4, (0, 0), tuple(demo.sim_x), BLUE)
        ya = arr(p4, (0, 0), tuple(demo.sim_y), GREEN)
        s1 = VGroup(MathTex(r"x\cdot y = %.1f" % demo.sim_dot, color=BLUE, font_size=50),
                    T("inner product: large", 26, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        s2 = VGroup(MathTex(r"\|x-y\|\approx %.2f" % demo.sim_dist, color=YELLOW, font_size=50),
                    T("distance: small", 26, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        toy = T("toy example: x = (4, 3), y = (4.2, 3.4)", 22, MUTED)
        grp = VGroup(s1, s2, toy).arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to(RIGHT * 3.2 + DOWN * 0.3)
        with self.voice(N["dist4"]) as v:
            self.play(Create(p4), GrowArrow(xa), GrowArrow(ya), FadeIn(toy), run_time=1)
            self.wait_until(v, "opposite")
            self.play(FadeIn(s1), FadeIn(s2), run_time=0.8)
            self.wait_until(v, "large")
            self.play(Indicate(s1, color=BLUE), run_time=0.8)
            self.wait_until(v, "small")
            self.play(Indicate(s2, color=YELLOW), run_time=0.8)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · Norms in machine learning")
        tag = T("① regularization", 30, YELLOW, weight=BOLD).move_to(UP * 2.3)
        f = MathTex(r"\text{loss}", "+", r"\lambda", r"\,\|w\|_2^2", font_size=80).move_to(UP * 0.9)
        f[2].set_color(PURPLE); f[3].set_color(PURPLE)
        leg = legend([(r"\text{loss}", "the error score (video 15)", FG),
                      (r"\lambda", "lambda: how strong the penalty is", PURPLE),
                      (r"\|w\|_2^2", "sum of the squared weights", PURPLE)],
                     x_sym=-3.0, x_txt=-1.8, y0=-0.7, dy=0.85, size=28)
        with self.voice(N["ml1"]) as v:
            self.wait_until(v, "regularization")
            self.play(FadeIn(tag), run_time=0.5)
            self.wait_until(v, "penalty")
            self.play(Write(f), run_time=1.2)
            self.wait_until(v, "loss", 1)
            self.play(FadeIn(leg[0]), run_time=0.5)
            self.wait_until(v, "lambda")
            self.play(FadeIn(leg[1]), run_time=0.5)
            self.wait_until(v, "squared")
            self.play(FadeIn(leg[2]), run_time=0.5)
            self.wait_until(v, "prefers")
            self.play(Indicate(f[3], color=YELLOW), run_time=0.8)
        self.clear_body()
        pl = grid_plane([-1.5, 1.5, 0.5], [-1.5, 1.5, 0.5], 4.8, 4.8).shift(LEFT * 3.8 + DOWN * 0.7)
        r = pl.c2p(1, 0)[0] - pl.c2p(0, 0)[0]
        w1 = MathTex("w_1", color=PURPLE, font_size=32).next_to(pl.c2p(1.5, 0), UP, buff=0.1)
        w2 = MathTex("w_2", color=PURPLE, font_size=32).next_to(pl.c2p(0, 1.5), RIGHT, buff=0.1)
        circ = Circle(radius=r, color=L2C, stroke_width=3, stroke_opacity=0.6).move_to(pl.c2p(0, 0))
        diam = Polygon(pl.c2p(1, 0), pl.c2p(0, 1), pl.c2p(-1, 0), pl.c2p(0, -1), color=L1C,
                       stroke_width=5)
        corners = VGroup(*[Dot(pl.c2p(*c), color=YELLOW, radius=0.09)
                           for c in [(1, 0), (0, 1), (-1, 0), (0, -1)]])
        f1 = MathTex(r"\text{loss}", "+", r"\lambda", r"\,\|w\|_1", font_size=64)
        f1[2].set_color(PURPLE); f1[3].set_color(L1C)
        t1 = T("→ many weights exactly 0", 28, YELLOW)
        t2 = T("sparsity: only a few inputs are used", 24, FG)
        t3 = T("corners of the diamond lie on the axes", 24, MUTED)
        t4 = T("ridge = L2 penalty · lasso = L1 penalty", 26, BLUE)
        colr = VGroup(f1, t1, t2, t3, t4).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        fit(colr, 6.6).move_to(RIGHT * 3.4 + DOWN * 0.3)
        with self.voice(N["ml2"]) as v:
            self.play(Create(pl), FadeIn(w1), FadeIn(w2), Create(circ), run_time=1)
            self.wait_until(v, "instead")
            self.play(Write(f1), run_time=1)
            self.wait_until(v, "zero")
            self.play(FadeIn(t1), run_time=0.5)
            self.wait_until(v, "sparsity")
            self.play(FadeIn(t2), run_time=0.5)
            self.wait_until(v, "diamond")
            self.play(Create(diam), run_time=0.8)
            self.wait_until(v, "corners")
            self.play(FadeIn(corners, scale=0.5), FadeIn(t3), run_time=0.7)
            self.wait_until(v, "Ridge")
            self.play(FadeIn(t4), run_time=0.6)
        self.clear_body()
        tag2 = T("② gradient clipping", 30, YELLOW, weight=BOLD).move_to(RIGHT * 3.4 + UP * 2.4)
        p3 = grid_plane([-1, 9, 1], [-1, 9, 1], 5.2, 5.2).shift(LEFT * 3.6 + DOWN * 0.7)
        r3 = p3.c2p(1, 0)[0] - p3.c2p(0, 0)[0]
        g = arr(p3, (0, 0), (6, 8), RED)
        gl = MathTex("g", color=RED, font_size=40).next_to(p3.c2p(6, 8), RIGHT, buff=0.1)
        ball = DashedVMobject(Arc(radius=5 * r3, start_angle=0, angle=PI / 2, color=ORANGE,
                                  arc_center=p3.c2p(0, 0)), num_dashes=20)
        bl = T("max norm = 5", 22, ORANGE).next_to(p3.c2p(5, 0), UR, buff=0.1)
        gn = MathTex(r"\|g\| = \sqrt{36+64} = 10", color=RED, font_size=40)
        rule = MathTex(r"\text{if } \|g\|>\text{max}:\quad g\leftarrow g\cdot\frac{\text{max}}{\|g\|}",
                       font_size=40)
        VGroup(gn, rule).arrange(DOWN, aligned_edge=LEFT, buff=0.4).next_to(tag2, DOWN, buff=0.4)
        with self.voice(N["ml3"]) as v:
            self.play(FadeIn(tag2), Create(p3), run_time=1)
            self.wait_until(v, "gradient")
            self.play(GrowArrow(g), FadeIn(gl), run_time=0.8)
            self.wait_until(v, "huge")
            self.play(Indicate(g, color=RED), run_time=0.8)
            self.wait_until(v, "measure")
            self.play(FadeIn(gn), run_time=0.6)
            self.wait_until(v, "maximum")
            self.play(Create(ball), FadeIn(bl), run_time=1)
            self.wait_until(v, "down")
            self.play(Write(rule), run_time=1)
        clip = arr(p3, (0, 0), tuple(demo.g_clipped), YELLOW, 8)
        cl = MathTex(r"g'", color=YELLOW, font_size=40).next_to(p3.c2p(*demo.g_clipped), RIGHT, buff=0.15)
        l1 = MathTex(r"\|g\|=10,\ \ \text{max}=5", font_size=40)
        l2 = MathTex(r"g\cdot\tfrac{5}{10}=\tfrac12\begin{bmatrix}6\\8\end{bmatrix}=\begin{bmatrix}%g\\%g\end{bmatrix}"
                     % tuple(demo.g_clipped), font_size=40)
        l2.set_color(YELLOW)
        l3 = T("norm 5, same direction", 24, GREEN)
        toy = T("toy example", 20, MUTED)
        low = VGroup(l1, l2, l3, toy).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        low.next_to(rule, DOWN, buff=0.45, aligned_edge=LEFT)
        with self.voice(N["ml4"]) as v:
            self.play(FadeOut(gn), run_time=0.4)
            self.wait_until(v, "toy")
            self.play(FadeIn(toy), run_time=0.4)
            self.wait_until(v, "ten")
            self.play(FadeIn(l1), run_time=0.6)
            self.wait_until(v, "half")
            self.play(Write(l2), run_time=1)
            self.wait_until(v, "three")
            self.play(TransformFromCopy(g, clip), FadeIn(cl), run_time=1)
            self.wait_until(v, "direction")
            self.play(FadeIn(l3), Indicate(clip, color=YELLOW), run_time=0.8)
        self.clear_body()
        tag3 = T("③ nearest neighbour", 30, YELLOW, weight=BOLD).move_to(RIGHT * 3.4 + UP * 2.4)
        p5 = grid_plane([-1, 6, 1], [-1, 6, 1], 5.2, 5.2).shift(LEFT * 3.6 + DOWN * 0.7)
        pts = VGroup(*[Dot(p5.c2p(*demo.points[k]), color=GREEN, radius=0.1) for k in "ABC"])
        pls = VGroup(*[T(k, 26, GREEN, weight=BOLD).next_to(d, UL, buff=0.08)
                       for k, d in zip("ABC", pts)])
        qd = Dot(p5.c2p(*demo.query), color=BLUE, radius=0.13)
        ql = T("query", 24, BLUE).next_to(qd, DR, buff=0.08)
        lines = VGroup(*[DashedLine(qd.get_center(), d.get_center(), color=MUTED) for d in pts])
        dl = VGroup(*[MathTex(r"d_%s \approx %.2f" % (k, demo.dists[k]), font_size=44)
                      for k in "ABC"]).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        dl.next_to(tag3, DOWN, buff=0.5)
        best = T("nearest: %s" % demo.nearest, 34, YELLOW, weight=BOLD).next_to(dl, DOWN, buff=0.5)
        best.align_to(dl, LEFT)
        many = T("search over word vectors: same idea, millions of points", 22, MUTED)
        many.to_edge(DOWN, buff=0.35).set_x(2.9)
        toy3 = T("toy points", 20, MUTED).next_to(p5, DOWN, buff=0.1)
        bi = "ABC".index(demo.nearest)
        with self.voice(N["ml5"]) as v:
            self.play(FadeIn(tag3), Create(p5), run_time=0.8)
            self.wait_until(v, "toy")
            self.play(FadeIn(pts), FadeIn(pls), FadeIn(toy3), run_time=0.6)
            self.wait_until(v, "query")
            self.play(FadeIn(qd), FadeIn(ql), run_time=0.5)
            self.wait_until(v, "Compute")
            self.play(Create(lines), run_time=0.8)
            for i, (w, n) in enumerate([("A", 2), ("B", 1), ("C", 1)]):
                self.wait_until(v, w, n)
                self.play(FadeIn(dl[i]), run_time=0.5)
            self.wait_until(v, "nearest")
            self.play(FadeIn(best), pts[bi].animate.set_color(YELLOW),
                      lines[bi].animate.set_color(YELLOW), run_time=0.8)
            self.wait_until(v, "millions")
            self.play(FadeIn(many), run_time=0.6)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        x = MathTex(r"x=\begin{bmatrix}-5\\12\end{bmatrix}", color=BLUE, font_size=70).shift(UP * 1.4)
        qs = MathTex(r"\|x\|_2=\;?", r"\qquad", r"\|x\|_1=\;?", font_size=60).next_to(x, DOWN, buff=0.5)
        qs[0].set_color(L2C); qs[2].set_color(L1C)
        pause = T("⏸  pause and try", 30, MUTED).next_to(qs, DOWN, buff=0.5)
        with self.voice(N["prac1"]) as v:
            self.wait_until(v, "vector")
            self.play(Write(x), run_time=1)
            self.wait_until(v, "two")
            self.play(Write(qs[0]), run_time=0.7)
            self.wait_until(v, "one")
            self.play(Write(qs[2]), run_time=0.7)
            self.wait_until(v, "Remember")
            self.play(FadeIn(pause), run_time=0.5)
        a1 = MathTex(r"\|x\|_2", "=", r"\sqrt{(-5)^2+12^2}", "=", r"\sqrt{25+144}", "=",
                     r"\sqrt{169}", "=", "13", font_size=46)
        a1[0].set_color(L2C); a1[8].set_color(YELLOW)
        a2 = MathTex(r"\|x\|_1", "=", r"|-5|+|12|", "=", "5+12", "=", "17", font_size=46)
        a2[0].set_color(L1C); a2[6].set_color(YELLOW)
        ans = VGroup(a1, a2).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        fit(ans, 12.8).next_to(qs, DOWN, buff=0.5)
        nt = T("17 ≥ 13: the taxi route is never shorter", 24, MUTED).next_to(ans, DOWN, buff=0.35)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "Minus")
            self.play(Write(a1[:3]), run_time=0.8)
            self.wait_until(v, "twenty")
            self.play(Write(a1[3:5]), run_time=0.6)
            self.wait_until(v, "sum")
            self.play(Write(a1[5:7]), run_time=0.6)
            self.wait_until(v, "thirteen")
            self.play(Write(a1[7:]), run_time=0.5)
            self.wait_until(v, "seventeen")
            self.play(Write(a2), run_time=1)
            self.wait_until(v, "taxi")
            self.play(FadeIn(nt), Indicate(a2[6], color=YELLOW), run_time=0.8)
        self.clear_body()
        pl = grid_plane([-1, 6, 1], [-1, 8, 1], 4.2, 5.4).shift(LEFT * 3.8 + DOWN * 0.7)
        A = Dot(pl.c2p(2, 7), color=BLUE, radius=0.1)
        B = Dot(pl.c2p(5, 3), color=GREEN, radius=0.1)
        Al = T("(2, 7)", 24, BLUE).next_to(A, LEFT, buff=0.12)
        Bl = T("(5, 3)", 24, GREEN).next_to(B, RIGHT, buff=0.12)
        qq = VGroup(MathTex(r"d\big((2,7),\,(5,3)\big)=\;?", font_size=48),
                    T("then normalize the difference", 28, FG),
                    T("⏸  pause and think", 28, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        qq.move_to(RIGHT * 3.3 + UP * 0.5)
        with self.voice(N["prac3"]) as v:
            self.play(Create(pl), run_time=0.8)
            self.wait_until(v, "distance")
            self.play(FadeIn(A), FadeIn(B), FadeIn(Al), FadeIn(Bl), Write(qq[0]), run_time=1)
            self.wait_until(v, "normalize")
            self.play(FadeIn(qq[1]), run_time=0.5)
            self.wait_until(v, "Pause")
            self.play(FadeIn(qq[2]), run_time=0.5)
        dv = arr(pl, (5, 3), (2, 7), YELLOW)
        s1 = MathTex(r"\begin{bmatrix}2\\7\end{bmatrix}-\begin{bmatrix}5\\3\end{bmatrix}=\begin{bmatrix}-3\\4\end{bmatrix}",
                     font_size=44)
        s2 = MathTex(r"\left\|\begin{bmatrix}-3\\4\end{bmatrix}\right\|=\sqrt{9+16}=5", font_size=44)
        s3 = MathTex(r"\tfrac15\begin{bmatrix}-3\\4\end{bmatrix}=\begin{bmatrix}-0.6\\0.8\end{bmatrix}",
                     color=YELLOW, font_size=44)
        sol = VGroup(s1, s2, s3).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        fit(sol, 6.6).move_to(RIGHT * 3.3 + DOWN * 0.5)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(qq), run_time=0.4)
            self.wait_until(v, "subtract")
            self.play(Write(s1), GrowArrow(dv), run_time=1.2)
            self.wait_until(v, "norm")
            self.play(Write(s2), run_time=1)
            self.wait_until(v, "normalize")
            self.play(Write(s3), run_time=1)
            self.wait_until(v, "one")
            self.play(Indicate(s3, color=YELLOW), run_time=0.8)


CODE = '''import numpy as np
from numpy.linalg import norm
x = np.array([3.0, 4.0])
print(norm(x, ord=1), norm(x), norm(x, ord=np.inf))
print(norm([2.0, 3.0, 6.0]))
print(x / norm(x))                       # normalize
a, b = np.array([1., 2, 3]), np.array([-1., -1, 0])
print(norm(a - b))                       # distance
g = np.array([6.0, 8.0])                 # toy gradient
print(g * min(1, 5 / norm(g)))           # clip to norm 5
query = np.array([2.0, 2.0])
pts = {"A": [1, 5], "B": [4, 3], "C": [0, 0]}
d = {k: norm(np.array(p) - query) for k, p in pts.items()}
print("nearest:", min(d, key=d.get))'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=24)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[3], color=YELLOW, buff=0.06)
            self.wait_until(v, "ord")
            self.play(Create(box), run_time=0.5)
            for i, w in [(5, "normalize"), (7, "distance"), (9, "clip"), (13, "nearest")]:
                self.wait_until(v, w)
                self.play(Transform(box, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                              buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=22, command="python demo.py").shift(DOWN * 0.3)
        fit(term, 13.4)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for i, w in [(0, "norms"), (1, "Two"), (2, "unit"), (3, "distance"),
                         (5, "clipped"), (7, "nearest")]:
                self.wait_until(v, w)
                self.play(Indicate(term[1][i], color=YELLOW), run_time=0.6)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["Norm = the size of a vector: one number, never negative",
                            "L2 norm: ‖x‖ = √(x·x) — Pythagoras in any dimension",
                            "L1 adds the |xᵢ|;  L∞ takes the largest |xᵢ|",
                            "Rules: positive · ‖c x‖ = |c| ‖x‖ · triangle inequality",
                            "x / ‖x‖ = unit vector;  ‖x − y‖ = distance",
                            "ML: small weights, clipped gradients, nearest neighbours"],
                           size=28).shift(UP * 0.5)
        with self.voice(N["sum1"]) as v:
            for p in pts[:4]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 6)
        nxt = VGroup(T("Next · Video 7", 26, MUTED),
                     T("Derivative — the slope right here", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "machine")
            self.play(FadeIn(pts[5], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Length", "C2_OtherNorms", "C3_Rules", "C4_Unit", "C5_Distance",
          "C6_ML", "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Length": "Length from the inner product",
                  "C2_OtherNorms": "L1, L2 and L-infinity norms", "C3_Rules": "The three norm rules",
                  "C4_Unit": "Unit vectors and normalizing", "C5_Distance": "Distance and metrics",
                  "C6_ML": "Norms in ML", "C7_Practice": "Try it yourself", "C7_Code": "Python demo",
                  "C8_Summary": "Summary", "C9_Credits": "Credits"}
