"""Basis & rank — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 3 of 16"
TITLE = "Basis & rank — how many directions are really there"


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


def fmt(x):
    return f"{x:g}"


def toy_table(scale=0.85):
    rows = [[f"{r[0]:.1f}", fmt(r[1]), fmt(r[2])] for r in demo.table]
    X = mat(rows, scale, FG, h_buff=2.5)
    heads = VGroup(*[T(s, 22, c).move_to([col.get_center()[0], X.get_top()[1] + 0.4, 0])
                     for s, c, col in zip(["length (m)", "length (cm)", "weight (kg)"],
                                          [BLUE, ORANGE, GREEN], X.get_columns())])
    return X, heads


# pseudo-3D drawing: x toward the viewer (down-left), y to the right, z up
X3, Y3, Z3 = np.array([-0.55, -0.45, 0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])


def axes3(O, s):
    P = lambda v: O + s * (v[0] * X3 + v[1] * Y3 + v[2] * Z3)  # noqa: E731
    ax = VGroup()
    for d, lab, ln in [((1, 0, 0), "x", 4), ((0, 1, 0), "y", 6.5), ((0, 0, 1), "z", 4)]:
        a = np.array(d, dtype=float)
        ax.add(Line(P(-0.6 * a), P(ln * a), color=MUTED, stroke_width=2))
        ax.add(MathTex(lab, color=MUTED, font_size=30).next_to(P(ln * a), UP if lab != "y" else RIGHT,
                                                                 buff=0.08))
    return ax, P


def arr3(P, v, color, width=6):
    return Arrow(P((0, 0, 0)), P(v), buff=0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.15)


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        if title.width > 12.5:
            title.scale_to_fit_width(12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.4)
        hundred = T("100 columns → 100 directions?", 30, FG).shift(UP * 0.9)
        X, heads = toy_table(0.8)
        tab = VGroup(X, heads).shift(DOWN * 1.4)
        box = SurroundingRectangle(VGroup(X.get_columns()[1], heads[1]), color=ORANGE, buff=0.1)
        same = T("= 100 × meters: the same information", 24, ORANGE).next_to(tab, DOWN, buff=0.3)
        rank = T("rank", 64, YELLOW, weight=BOLD).next_to(hundred, RIGHT, buff=0.8)
        VGroup(hundred, rank).move_to(UP * 0.9)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            self.wait_until(v, "columns")
            self.play(FadeIn(hundred), run_time=0.7)
            self.wait_until(v, "meters")
            self.play(FadeIn(tab), run_time=0.8)
            self.wait_until(v, "centimeters")
            self.play(Create(box), FadeIn(same), run_time=0.8)
            self.wait_until(v, "rank")
            self.play(FadeIn(rank, scale=1.3), run_time=0.8)
        plan = self.bullets(["when are vectors redundant?", "what is a basis?",
                             "one point, two bases", "the rank of a matrix"], size=28)
        plan.move_to(DOWN * 1.0)
        pre = T("You need: video 1 · Vector, video 2 · Matrix", 24, MUTED).to_edge(DOWN, buff=0.4)
        with self.voice(N["intro2"]) as v:
            self.play(FadeOut(tab), FadeOut(box), FadeOut(same), run_time=0.5)
            for p, w in zip(plan, ["redundant", "basis", "point", "rank"]):
                self.wait_until(v, w)
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
            self.wait_until(v, "need")
            self.play(FadeIn(pre), run_time=0.6)


class C1_Redundant(Studio):
    def construct(self):
        self.heading("1 · Redundant directions")
        plane = grid_plane([-1, 5, 1], [-1, 4, 1], 5.1, 4.25).move_to(LEFT * 3.4 + DOWN * 0.5)
        plane.shift(plane.get_center() - plane.c2p(2, 1.5))
        u = arr(plane, (0, 0), (2, 1), BLUE, 7)
        w = arr(plane, (0, 0), (4, 2), RED, 4)
        ln = DashedLine(plane.c2p(-1, -0.5), plane.c2p(5, 2.5), color=YELLOW, stroke_width=3)
        eq = MathTex(r"\begin{bmatrix}4\\2\end{bmatrix}", "=", "2", r"\begin{bmatrix}2\\1\end{bmatrix}",
                     font_size=54)
        eq[0].set_color(RED); eq[2].set_color(PURPLE); eq[3].set_color(BLUE)
        eq.move_to(RIGHT * 3.5 + UP * 1.5)
        red = T("redundant: adds nothing new", 28, RED).next_to(eq, DOWN, buff=0.4)
        rem = self.reminder_card("span", "All the vectors you can reach with linear "
                                 "combinations of a set (video 1).", pos=DR, width=4.6)
        with self.voice(N["red1"]) as v:
            self.play(Create(plane), run_time=1)
            self.wait_until(v, "two")
            self.play(GrowArrow(u), run_time=0.6)
            self.wait_until(v, "four")
            self.play(GrowArrow(w), run_time=0.6)
            self.wait_until(v, "Four", 2)
            self.play(Write(eq), run_time=1)
            self.wait_until(v, "reach")
            self.play(Create(ln), FadeIn(rem, shift=UP * 0.3), run_time=1)
            self.wait_until(v, "redundant")
            self.play(FadeIn(red), run_time=0.6)
        a = arr(plane, (0, 0), tuple(demo.a), BLUE)
        b = arr(plane, (0, 0), tuple(demo.b), GREEN)
        c = arr(plane, (0, 0), tuple(demo.c), YELLOW, 7)
        la = MathTex("a", color=BLUE, font_size=40).next_to(plane.c2p(2, 1), RIGHT, buff=0.1)
        lb = MathTex("b", color=GREEN, font_size=40).next_to(plane.c2p(1, 2), LEFT, buff=0.1)
        lc = MathTex("c", color=YELLOW, font_size=40).next_to(plane.c2p(3, 3), UR, buff=0.05)
        vals = VGroup(*[MathTex(s, "=", r"\begin{bmatrix}%d\\%d\end{bmatrix}" % tuple(vv), font_size=44)
                        for s, vv in [("a", demo.a), ("b", demo.b), ("c", demo.c)]])
        for m, col in zip(vals, [BLUE, GREEN, YELLOW]):
            m.set_color(col)
        vals.arrange(RIGHT, buff=0.6).move_to(RIGHT * 3.6 + UP * 1.6)
        q = T("is each one needed?", 28, FG).next_to(vals, DOWN, buff=0.4)
        with self.voice(N["red2"]) as v:
            self.play(FadeOut(VGroup(u, w, ln, eq, red, rem)), run_time=0.6)
            self.wait_until(v, "two")
            self.play(GrowArrow(a), FadeIn(la), FadeIn(vals[0]), run_time=0.7)
            self.wait_until(v, "one", 2)
            self.play(GrowArrow(b), FadeIn(lb), FadeIn(vals[1]), run_time=0.7)
            self.wait_until(v, "three", 2)
            self.play(GrowArrow(c), FadeIn(lc), FadeIn(vals[2]), run_time=0.7)
            self.wait_until(v, "needed")
            self.play(FadeIn(q), run_time=0.6)
        bm = arr(plane, tuple(demo.a), tuple(demo.a + demo.b), GREEN)
        ceq = MathTex("c", "=", "a", "+", "b", font_size=64)
        ceq[0].set_color(YELLOW); ceq[2].set_color(BLUE); ceq[4].set_color(GREEN)
        ceq.next_to(q, DOWN, buff=0.6)
        nonew = T("c brings no new direction", 26, RED).next_to(ceq, DOWN, buff=0.4)
        with self.voice(N["red3"]) as v:
            self.wait_until(v, "along")
            self.play(Indicate(a, color=BLUE), run_time=0.7)
            self.wait_until(v, "along", 2)
            self.play(TransformFromCopy(b, bm), run_time=1)
            self.wait_until(v, "exactly")
            self.play(Indicate(c, color=YELLOW), run_time=0.8)
            self.wait_until(v, "equals")
            self.play(Write(ceq), run_time=0.8)
            self.wait_until(v, "new")
            self.play(FadeIn(nonew), run_time=0.6)
        dots = VGroup(*[Dot(plane.c2p(x, y), radius=0.035, color=YELLOW)
                        for x in np.arange(-1, 5.01, 0.5) for y in np.arange(-1, 4.01, 0.5)])
        sp = MathTex(r"\mathrm{span}\{a,b,c\}", "=", r"\mathrm{span}\{a,b\}", "=",
                     r"\text{plane}", font_size=40)
        sp.move_to(RIGHT * 3.6 + DOWN * 2.3)
        cnt = T("3 vectors, 2 directions", 28, YELLOW).next_to(sp, DOWN, buff=0.3)
        with self.voice(N["red4"]) as v:
            self.wait_until(v, "span")
            self.play(Write(sp), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.005),
                      run_time=2)
            self.wait_until(v, "three")
            self.play(FadeIn(cnt), run_time=0.6)


class C2_Independence(Studio):
    def construct(self):
        self.heading("2 · Linear independence")
        s1 = T("independent: no vector is a combination of the others", 28, GREEN)
        s2 = T("dependent: at least one can be built from the others", 28, RED)
        st = VGroup(s1, s2).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(LEFT, buff=0.7)
        st.shift(UP * 1.9)
        card = self.term_card("linear independence", "No vector in the set is a combination "
                              "of the others. Only all-zero weights give the zero vector.",
                              pos=DR, width=5.6)
        with self.voice(N["ind1"]) as v:
            self.wait_until(v, "independent")
            self.play(FadeIn(card, shift=UP * 0.3), FadeIn(s1), run_time=0.8)
            self.wait_until(v, "dependent")
            self.play(FadeIn(s2), run_time=0.7)
        f = MathTex(r"\sum_{i=1}^{k}", r"\lambda_i", r"\,x_i", "=", r"\mathbf{0}",
                    r"\;\Longrightarrow\;", r"\lambda_1=\dots=\lambda_k=0", font_size=60)
        f[0].set_color(ORANGE); f[1].set_color(PURPLE); f[2].set_color(BLUE)
        f[4].set_color(YELLOW); f[6].set_color(PURPLE)
        f.move_to(UP * 0.2)
        always = T("(all-zero weights always work)", 22, MUTED).next_to(f, DOWN, buff=0.3)
        with self.voice(N["ind2"]) as v:
            self.play(FadeOut(card), run_time=0.4)
            self.wait_until(v, "weights")
            self.play(Write(f[:5]), run_time=1.2)
            self.wait_until(v, "always")
            self.play(FadeIn(always), run_time=0.6)
            self.wait_until(v, "only")
            self.play(Write(f[5:]), run_time=1)
        leg = VGroup(*[VGroup(MathTex(sy, color=co, font_size=40), T(tx, 24, MUTED)).arrange(RIGHT, buff=0.3)
                       for sy, co, tx in [(r"\lambda_i", PURPLE, "lambda i: a weight (a scalar)"),
                                          ("x_i", BLUE, "the i-th vector"),
                                          (r"\sum_{i=1}^{k}", ORANGE, "add up for i = 1, 2, …, k"),
                                          (r"\Longrightarrow", FG, "“if … then …”")]])
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(DOWN * 2.3)
        with self.voice(N["ind3"]) as v:
            self.play(FadeOut(always), run_time=0.3)
            self.wait_until(v, "Lambda")
            self.play(FadeIn(leg[0]), Indicate(f[1], color=PURPLE), run_time=0.8)
            self.wait_until(v, "vector")
            self.play(FadeIn(leg[1]), Indicate(f[2], color=BLUE), run_time=0.8)
            self.wait_until(v, "sum")
            self.play(FadeIn(leg[2]), Indicate(f[0], color=ORANGE), run_time=0.8)
            self.wait_until(v, "arrow")
            self.play(FadeIn(leg[3]), Indicate(f[5]), run_time=0.8)
        self.clear_body()
        plane = grid_plane([-1, 4, 1], [-1, 4, 1], 4.25, 4.25).move_to(LEFT * 4.3 + DOWN * 0.6)
        plane.shift(plane.get_center() - plane.c2p(1.5, 1.5))
        a = arr(plane, (0, 0), (2, 1), BLUE)
        b = arr(plane, (2, 1), (3, 3), GREEN)
        mc = arr(plane, (3, 3), (0, 0), RED)
        comb = MathTex("1", r"\begin{bmatrix}2\\1\end{bmatrix}", "+", "1", r"\begin{bmatrix}1\\2\end{bmatrix}",
                       "-", "1", r"\begin{bmatrix}3\\3\end{bmatrix}", "=", r"\begin{bmatrix}0\\0\end{bmatrix}",
                       font_size=44)
        for i, c in [(0, PURPLE), (1, BLUE), (3, PURPLE), (4, GREEN), (6, PURPLE), (7, YELLOW), (9, YELLOW)]:
            comb[i].set_color(c)
        comb.move_to(RIGHT * 2.4 + UP * 1.7)
        l1 = MathTex(r"\text{first: } 2+1-3=0", font_size=40)
        l2 = MathTex(r"\text{second: } 1+2-3=0", font_size=40)
        VGroup(l1, l2).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(comb, DOWN, buff=0.5)
        verdict = T("non-zero weights give 0 → dependent", 28, RED).next_to(VGroup(l1, l2), DOWN, buff=0.5)
        assert (demo.zero == 0).all()
        with self.voice(N["ind4"]) as v:
            self.play(Create(plane), run_time=0.8)
            self.wait_until(v, "weights")
            self.play(Write(comb), run_time=1.5)
            self.wait_until(v, "first")
            self.play(FadeIn(l1), run_time=0.6)
            self.wait_until(v, "second")
            self.play(FadeIn(l2), run_time=0.6)
            self.wait_until(v, "Walk")
            self.play(GrowArrow(a), run_time=0.6)
            self.play(GrowArrow(b), run_time=0.6)
            self.play(GrowArrow(mc), run_time=0.6)
            self.wait_until(v, "dependent")
            self.play(FadeIn(verdict), run_time=0.6)
        self.clear_body()
        e0 = MathTex(r"\lambda_1", r"\begin{bmatrix}2\\1\end{bmatrix}", "+", r"\lambda_2",
                     r"\begin{bmatrix}1\\2\end{bmatrix}", "=", r"\begin{bmatrix}0\\0\end{bmatrix}",
                     font_size=50)
        for i, c in [(0, PURPLE), (1, BLUE), (3, PURPLE), (4, GREEN), (6, YELLOW)]:
            e0[i].set_color(c)
        e1 = MathTex(r"2\lambda_1+\lambda_2=0", r"\;\Rightarrow\;", r"\lambda_2=-2\lambda_1", font_size=46)
        e2 = MathTex(r"\lambda_1+2\lambda_2=0", r"\;\Rightarrow\;", r"\lambda_1-4\lambda_1=-3\lambda_1=0",
                     font_size=46)
        e3 = MathTex(r"\lambda_1=0,\ \lambda_2=0", font_size=46, color=PURPLE)
        ok = T("only zero weights → independent", 28, GREEN)
        lines = VGroup(e0, e1, e2, VGroup(e3, ok).arrange(RIGHT, buff=0.6)).arrange(DOWN, buff=0.45)
        lines.move_to(DOWN * 0.3)
        tags = VGroup(T("first components", 20, MUTED).next_to(e1, LEFT, buff=0.4),
                      T("second components", 20, MUTED).next_to(e2, LEFT, buff=0.4))
        with self.voice(N["ind5"]) as v:
            self.wait_until(v, "Suppose")
            self.play(Write(e0), run_time=1.2)
            self.wait_until(v, "first")
            self.play(Write(e1[0]), FadeIn(tags[0]), run_time=0.8)
            self.wait_until(v, "So")
            self.play(Write(e1[1:]), run_time=0.8)
        with self.voice(N["ind6"]) as v:
            self.wait_until(v, "second")
            self.play(Write(e2[0]), FadeIn(tags[1]), run_time=0.8)
            self.wait_until(v, "becomes")
            self.play(Write(e2[1:]), run_time=1)
            self.wait_until(v, "So")
            self.play(Write(e3), run_time=0.8)
            self.wait_until(v, "independent")
            self.play(FadeIn(ok), run_time=0.6)
        self.clear_body()
        f1 = VGroup(MathTex(r"\{\,x,\ \mathbf{0}\,\}:", font_size=48),
                    MathTex(r"0\cdot x + 1\cdot\mathbf{0} = \mathbf{0}", font_size=48),
                    T("→ dependent", 30, RED)).arrange(RIGHT, buff=0.4)
        f2 = VGroup(MathTex(r"\left\{\begin{bmatrix}2\\1\end{bmatrix},\begin{bmatrix}4\\2\end{bmatrix}\right\}:",
                            font_size=48),
                    MathTex(r"2\begin{bmatrix}2\\1\end{bmatrix}-\begin{bmatrix}4\\2\end{bmatrix}=\mathbf{0}",
                            font_size=48),
                    T("→ dependent", 30, RED)).arrange(RIGHT, buff=0.4)
        VGroup(f1, f2).arrange(DOWN, buff=0.8, aligned_edge=LEFT).move_to(DOWN * 0.2)
        with self.voice(N["ind7"]) as v:
            self.wait_until(v, "zero")
            self.play(FadeIn(f1, shift=UP * 0.2), run_time=0.8)
            self.wait_until(v, "multiple")
            self.play(FadeIn(f2, shift=UP * 0.2), run_time=0.8)


class C3_Basis(Studio):
    def construct(self):
        self.heading("3 · Basis, coordinates, dimension")
        l1 = T("basis = linearly independent + spans the space", 32, YELLOW)
        l2 = T("a smallest spanning set: remove one vector → lose part of the space", 26, FG)
        VGroup(l1, l2).arrange(DOWN, aligned_edge=LEFT, buff=0.4).to_edge(LEFT, buff=0.7).shift(UP * 1.2)
        card = self.term_card("basis", "A linearly independent set of vectors that spans the "
                              "whole space.", pos=DR, width=5.0)
        with self.voice(N["basis1"]) as v:
            self.wait_until(v, "basis")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "independent")
            self.play(FadeIn(l1), run_time=0.8)
            self.wait_until(v, "smallest")
            self.play(FadeIn(l2), run_time=0.8)
        self.clear_body()
        plane = grid_plane([-3, 3, 1], [-1, 4, 1], 5.1, 4.25).move_to(LEFT * 3.6 + DOWN * 0.5)
        plane.shift(plane.get_center() - plane.c2p(0, 1.5))
        e1 = arr(plane, (0, 0), (1, 0), BLUE)
        e2 = arr(plane, (0, 0), (0, 1), GREEN)
        le = VGroup(MathTex("e_1", color=BLUE, font_size=34).next_to(plane.c2p(1, 0), DOWN, buff=0.1),
                    MathTex("e_2", color=GREEN, font_size=34).next_to(plane.c2p(0, 1), LEFT, buff=0.1))
        p = Dot(plane.c2p(*demo.p), color=YELLOW, radius=0.1)
        pl = MathTex("p", color=YELLOW, font_size=40).next_to(p, UR, buff=0.08)
        path = VGroup(DashedLine(plane.c2p(0, 0), plane.c2p(1, 0), color=ORANGE, stroke_width=4),
                      DashedLine(plane.c2p(1, 0), plane.c2p(1, 3), color=ORANGE, stroke_width=4))
        eq = MathTex(r"\begin{bmatrix}1\\3\end{bmatrix}", "=", "1", "e_1", "+", "3", "e_2", font_size=48)
        for i, c in [(0, YELLOW), (2, PURPLE), (3, BLUE), (5, PURPLE), (6, GREEN)]:
            eq[i].set_color(c)
        eq.move_to(RIGHT * 3.4 + UP * 1.4)
        cE = T("coordinates in e₁, e₂:  (1, 3)", 26, ORANGE).move_to(RIGHT * 3.4 + UP * 2.4)
        with self.voice(N["basis2"]) as v:
            self.play(Create(plane), run_time=0.8)
            self.wait_until(v, "familiar")
            self.play(GrowArrow(e1), GrowArrow(e2), FadeIn(le), run_time=0.8)
            self.wait_until(v, "point")
            self.play(FadeIn(p), FadeIn(pl), run_time=0.5)
            self.wait_until(v, "times")
            self.play(Create(path), Write(eq), run_time=1.5)
            self.wait_until(v, "coordinates")
            self.play(FadeIn(cE), run_time=0.6)
        b1 = arr(plane, (0, 0), (1, 1), BLUE, 7)
        b2 = arr(plane, (0, 0), (-1, 1), GREEN, 7)
        lb = VGroup(MathTex("b_1", color=BLUE, font_size=34).next_to(plane.c2p(1, 1), RIGHT, buff=0.1),
                    MathTex("b_2", color=GREEN, font_size=34).next_to(plane.c2p(-1, 1), LEFT, buff=0.1))
        grid = VGroup()
        for k in range(3):
            grid.add(Line(plane.c2p(k, k), plane.c2p(k - 2, k + 2), color=PURPLE, stroke_width=2))
            grid.add(Line(plane.c2p(-k, k), plane.c2p(2 - k, 2 + k), color=PURPLE, stroke_width=2))
        with self.voice(N["basis3"]) as v:
            self.play(FadeOut(path), FadeOut(eq), e1.animate.set_opacity(0.25),
                      e2.animate.set_opacity(0.25), FadeOut(le), run_time=0.6)
            self.wait_until(v, "Here")
            self.play(GrowArrow(b1), FadeIn(lb[0]), run_time=0.6)
            self.wait_until(v, "two")
            self.play(GrowArrow(b2), FadeIn(lb[1]), run_time=0.6)
            self.wait_until(v, "grid")
            self.play(Create(grid), run_time=1.5)
        q0 = MathTex("c_1", "b_1", "+", "c_2", "b_2", "=", r"\begin{bmatrix}1\\3\end{bmatrix}", font_size=46)
        for i, c in [(0, PURPLE), (1, BLUE), (3, PURPLE), (4, GREEN), (6, YELLOW)]:
            q0[i].set_color(c)
        q1 = MathTex(r"c_1-c_2=1", font_size=42)
        q2 = MathTex(r"c_1+c_2=3", font_size=42)
        q3 = MathTex(r"2c_1=4\ \Rightarrow\ c_1=2", font_size=42, color=PURPLE)
        q4 = MathTex(r"c_2=1", font_size=42, color=PURPLE)
        col = VGroup(q0, VGroup(q1, q2).arrange(RIGHT, buff=0.6), VGroup(q3, q4).arrange(RIGHT, buff=0.6))
        col.arrange(DOWN, buff=0.4).move_to(RIGHT * 3.6 + UP * 0.2)
        assert tuple(demo.coords) == (2.0, 1.0)
        npath = VGroup(arr(plane, (0, 0), (2, 2), BLUE, 5), arr(plane, (2, 2), (1, 3), GREEN, 5))
        cB = T("coordinates in b₁, b₂:  (2, 1)", 26, PURPLE).next_to(col, DOWN, buff=0.5)
        with self.voice(N["basis4"]) as v:
            self.wait_until(v, "weights")
            self.play(Write(q0), run_time=1)
            self.wait_until(v, "first")
            self.play(FadeIn(q1), run_time=0.6)
            self.wait_until(v, "second")
            self.play(FadeIn(q2), run_time=0.6)
        with self.voice(N["basis5"]) as v:
            self.wait_until(v, "Add")
            self.play(Write(q3), run_time=0.8)
            self.wait_until(v, "Then")
            self.play(Write(q4), run_time=0.6)
            self.wait_until(v, "Check")
            self.play(GrowArrow(npath[0]), run_time=0.8)
            self.wait_until(v, "Add", 2)
            self.play(GrowArrow(npath[1]), run_time=0.8)
            self.wait_until(v, "coordinates")
            self.play(FadeIn(cB), run_time=0.6)
        uniq = T("one point → one set of coordinates per basis", 22, FG).next_to(cB, DOWN, buff=0.3)
        with self.voice(N["basis6"]) as v:
            self.wait_until(v, "depend")
            self.play(Indicate(cE, color=ORANGE), Indicate(cB, color=PURPLE), run_time=1)
            self.wait_until(v, "exactly")
            self.play(FadeIn(uniq), run_time=0.6)
        self.clear_body()
        card = self.term_card("dimension", "The number of vectors in any basis of a space.",
                              pos=DR, width=5.0)
        top = T("every basis has the same number of vectors", 30, YELLOW).to_edge(LEFT, buff=0.7).shift(UP * 1.8)
        r1 = MathTex(r"\text{plane: } e_1, e_2", r"\;\to\;", r"\dim = 2", font_size=44)
        r2 = MathTex(r"\text{space: } e_1, e_2, e_3", r"\;\to\;", r"\dim = 3", font_size=44)
        r3 = T("3 vectors in the plane → always dependent", 28, RED)
        rows = VGroup(r1, r2, r3).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(top, DOWN, buff=0.6)
        rows.align_to(top, LEFT)
        with self.voice(N["dim1"]) as v:
            self.wait_until(v, "same")
            self.play(FadeIn(top), run_time=0.6)
            self.wait_until(v, "dimension")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "two")
            self.play(FadeIn(r1), run_time=0.6)
            self.wait_until(v, "three")
            self.play(FadeIn(r2), run_time=0.6)
            self.wait_until(v, "always")
            self.play(FadeIn(r3), run_time=0.6)
        self.clear_body()
        pl2 = grid_plane([-3, 3, 1], [-2, 2, 1], 5.1, 3.4).move_to(LEFT * 3.4 + DOWN * 0.4)
        yl = Line(pl2.c2p(0, -2), pl2.c2p(0, 2), color=YELLOW, stroke_width=6)
        e = arr(pl2, (0, 0), (0, 1), GREEN, 7)
        n1 = MathTex(r"\mathrm{span}\left\{\begin{bmatrix}0\\1\end{bmatrix}\right\}", r"=\text{a line}",
                     font_size=44)
        n2 = T("2 entries per vector, but dimension 1", 26, YELLOW)
        n3 = T("a plane through 0 in 3-D: dimension 2", 26, FG)
        VGroup(n1, n2, n3).arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to(RIGHT * 3.4 + DOWN * 0.3)
        warn = T("dimension ≠ number of entries", 28, RED).move_to(RIGHT * 3.4 + UP * 2.0)
        with self.voice(N["dim2"]) as v:
            self.play(FadeIn(warn), Create(pl2), run_time=0.8)
            self.wait_until(v, "line")
            self.play(GrowArrow(e), Create(yl), Write(n1), run_time=1.2)
            self.wait_until(v, "dimension", 2)
            self.play(FadeIn(n2), run_time=0.6)
            self.wait_until(v, "plane")
            self.play(FadeIn(n3), run_time=0.6)


class C4_Rank(Studio):
    def construct(self):
        self.heading("4 · The rank of a matrix")
        d1 = T("rank = number of linearly independent columns", 32, YELLOW)
        d2 = T("column space = all combinations of the columns", 26, FG)
        d3 = T("its dimension = the rank", 26, FG)
        VGroup(d1, d2, d3).arrange(DOWN, aligned_edge=LEFT, buff=0.35).to_edge(LEFT, buff=0.7).shift(UP * 1.2)
        card = self.term_card("rank", "The number of linearly independent columns of a matrix "
                              "(it equals the number of independent rows).", pos=DR, width=5.4)
        with self.voice(N["rank1"]) as v:
            self.wait_until(v, "rank")
            self.play(FadeIn(card, shift=UP * 0.3), FadeIn(d1), run_time=0.8)
            self.wait_until(v, "combinations")
            self.play(FadeIn(d2), run_time=0.6)
            self.wait_until(v, "dimension")
            self.play(FadeIn(d3), run_time=0.6)
        self.clear_body()
        O = np.array([1.6, -0.9, 0])
        cols_c = [BLUE, GREEN, YELLOW]

        def example(M, s=0.55):
            m = colour_cols(mat(np.asarray(M).astype(int).tolist(), 0.9), cols_c).move_to(LEFT * 4.6 + UP * 1.0)
            ax, P = axes3(O, s)
            arrows = [arr3(P, tuple(np.asarray(M)[:, j]), cols_c[j]) for j in range(3)]
            return m, ax, P, arrows
        m, ax, P, ar = example(demo.R1)
        n1 = MathTex(r"\text{col}_2 = -1\cdot\text{col}_1", font_size=36, color=GREEN)
        n2 = MathTex(r"\text{col}_3 = 3\cdot\text{col}_1", font_size=36, color=YELLOW)
        rk1 = MathTex(r"\text{rank} = 1", font_size=52, color=YELLOW)
        VGroup(n1, n2, rk1).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(m, DOWN, buff=0.5)
        line = Line(P(-1.2 * np.array([1, 2, 0])), P(3.4 * np.array([1, 2, 0])), color=YELLOW,
                    stroke_width=3)
        cs = T("column space: a line", 26, YELLOW).move_to(RIGHT * 4.0 + UP * 2.4)
        assert demo.rk(demo.R1) == 1
        with self.voice(N["rank2"]) as v:
            self.wait_until(v, "columns")
            self.play(FadeIn(m), Create(ax), run_time=1)
            self.play(GrowArrow(ar[0]), run_time=0.6)
            self.wait_until(v, "second")
            self.play(GrowArrow(ar[1]), FadeIn(n1), run_time=0.7)
            self.wait_until(v, "third")
            self.play(GrowArrow(ar[2]), FadeIn(n2), run_time=0.7)
            self.wait_until(v, "rank")
            self.play(Write(rk1), run_time=0.7)
            self.wait_until(v, "line")
            self.play(Create(line), FadeIn(cs), run_time=1)
        self.clear_body()
        m, ax, P, ar = example(demo.R2)
        c1, c2 = demo.R2[:, 0], demo.R2[:, 1]
        corners = [P(a_ * c1 + b_ * c2) for a_, b_ in [(-1.2, -1.2), (2.2, -1.2), (2.2, 2.2), (-1.2, 2.2)]]
        patch = Polygon(*corners, color=YELLOW, stroke_width=1.5, fill_color=YELLOW, fill_opacity=0.18)
        n3 = MathTex(r"\text{col}_3 = \text{col}_1 + \text{col}_2", font_size=36, color=YELLOW)
        rk2 = MathTex(r"\text{rank} = 2", font_size=52, color=YELLOW)
        VGroup(n3, rk2).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(m, DOWN, buff=0.5)
        cs2 = T("column space: a plane", 26, YELLOW).move_to(RIGHT * 4.0 + UP * 2.4)
        assert demo.rk(demo.R2) == 2
        with self.voice(N["rank3"]) as v:
            self.wait_until(v, "columns")
            self.play(FadeIn(m), Create(ax), run_time=1)
            self.wait_until(v, "first")
            self.play(GrowArrow(ar[0]), GrowArrow(ar[1]), run_time=0.8)
            self.wait_until(v, "third")
            self.play(GrowArrow(ar[2]), FadeIn(n3), run_time=0.7)
            self.wait_until(v, "rank")
            self.play(Write(rk2), run_time=0.7)
            self.wait_until(v, "plane")
            self.play(FadeIn(patch), FadeIn(cs2), run_time=1.2)
        self.clear_body()
        m, ax, P, ar = example(np.eye(3), 0.75)
        rk3 = MathTex(r"\text{rank} = 3", font_size=52, color=YELLOW).next_to(m, DOWN, buff=0.5)
        rule = MathTex(r"\text{rank} \le \min(\text{rows}, \text{columns})", font_size=36).next_to(rk3, DOWN, buff=0.4)
        rule.align_to(rk3, LEFT)
        full = T("as large as possible → full rank", 26, GREEN).next_to(rule, DOWN, buff=0.3).align_to(rk3, LEFT)
        with self.voice(N["rank4"]) as v:
            self.wait_until(v, "identity")
            self.play(FadeIn(m), Create(ax), *[GrowArrow(a_) for a_ in ar], run_time=1.2)
            self.wait_until(v, "three", 4)
            self.play(Write(rk3), run_time=0.7)
            self.wait_until(v, "larger")
            self.play(FadeIn(rule), run_time=0.7)
            self.wait_until(v, "full")
            self.play(FadeIn(full), run_time=0.6)
        self.clear_body()
        R = mat(demo.R1.tolist(), 1.0).move_to(LEFT * 3.5 + UP * 0.3)
        rws = R.get_rows()
        rb = [SurroundingRectangle(r, color=c, buff=0.08) for r, c in zip(rws, [BLUE, ORANGE, ORANGE])]
        rn = VGroup(MathTex(r"\text{row}_2 = 2\cdot\text{row}_1", font_size=40),
                    MathTex(r"\text{row}_3 = 0\cdot\text{row}_1", font_size=40),
                    MathTex(r"\text{independent rows} = 1 = \text{independent columns}", font_size=40,
                            color=YELLOW)).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        rn.move_to(RIGHT * 2.6 + UP * 0.3)
        if rn.get_right()[0] > 6.9:
            rn.scale_to_fit_width(13.8 - 2 * 6.9 + rn.width).move_to(RIGHT * 2.6 + UP * 0.3)
        tool = T("np.linalg.matrix_rank(A)", 28, GREEN, font=MONO).move_to(DOWN * 2.6)
        with self.voice(N["rank5"]) as v:
            self.play(FadeIn(R), run_time=0.6)
            self.wait_until(v, "rows")
            self.play(*[Create(b_) for b_ in rb], FadeIn(rn[:2]), run_time=1)
            self.wait_until(v, "columns")
            self.play(FadeIn(rn[2]), run_time=0.7)
            self.wait_until(v, "computer")
            self.play(FadeIn(tool), run_time=0.6)


class C5_ML(Studio):
    def construct(self):
        self.heading("5 · Rank in machine learning")
        X, heads = toy_table(0.9)
        tab = VGroup(X, heads).move_to(LEFT * 2.6 + DOWN * 0.1)
        toy = T("toy example", 20, MUTED).next_to(tab, DOWN, buff=0.35)
        box = SurroundingRectangle(VGroup(X.get_columns()[1], heads[1]), color=ORANGE, buff=0.1)
        note = MathTex(r"\text{col}_2 = 100\cdot\text{col}_1", font_size=44, color=ORANGE)
        note.move_to(RIGHT * 4.0 + UP * 1.4)
        with self.voice(N["ml1"]) as v:
            self.play(FadeIn(X), FadeIn(toy), run_time=0.8)
            for h, w in zip(heads, ["meters", "centimeters", "kilograms"]):
                self.wait_until(v, w)
                self.play(FadeIn(h), run_time=0.5)
            self.wait_until(v, "hundred")
            self.play(Create(box), Write(note), run_time=1)
        rk = MathTex(r"\text{rank} = %d" % demo.rk(demo.table), r"\ \ (\text{not } 3)", font_size=56)
        rk[0].set_color(YELLOW); rk[1].set_color(MUTED)
        rk.next_to(note, DOWN, buff=0.6)
        red = T("redundant feature: no new information", 26, RED).next_to(rk, DOWN, buff=0.5)
        if red.get_right()[0] > 6.9:
            red.shift(LEFT * (red.get_right()[0] - 6.9))
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "rank")
            self.play(Write(rk), run_time=0.8)
            self.wait_until(v, "redundant")
            self.play(FadeIn(red), run_time=0.6)
        self.clear_body()
        ttl = T("LoRA = low-rank adaptation (mention only)", 30, YELLOW).shift(UP * 2.2)
        W = Square(2.6, color=BLUE, fill_color=BLUE, fill_opacity=0.15).move_to(LEFT * 4.4 + DOWN * 0.4)
        Wl = MathTex(r"\Delta W", r"\ (768\times768)", font_size=32).next_to(W, DOWN, buff=0.2)
        eq = MathTex(r"\approx", font_size=60).next_to(W, RIGHT, buff=0.4)
        A_ = Rectangle(width=0.35, height=2.6, color=GREEN, fill_color=GREEN, fill_opacity=0.3)
        B_ = Rectangle(width=2.6, height=0.35, color=PURPLE, fill_color=PURPLE, fill_opacity=0.3)
        A_.next_to(eq, RIGHT, buff=0.4); B_.next_to(A_, RIGHT, buff=0.3).align_to(A_, UP)
        Al = MathTex(r"A\ (768\times16)", font_size=30, color=GREEN).next_to(A_, DOWN, buff=0.2)
        Bl = MathTex(r"B\ (16\times768)", font_size=30, color=PURPLE).next_to(B_, DOWN, buff=0.2)
        rk16 = MathTex(r"\text{rank}(AB)\le 16", font_size=44, color=YELLOW).move_to(RIGHT * 3.6 + DOWN * 1.0)
        nums = MathTex(r"768^2 = 589{,}824", r"\quad\text{vs}\quad", r"2\cdot768\cdot16 = 24{,}576",
                       font_size=36).move_to(DOWN * 2.9)
        assert 768 ** 2 == 589824 and 2 * 768 * 16 == 24576
        src = T("Raschka, Build a LLM (From Scratch), App. E, pp. 322–323", 18, MUTED).to_edge(DOWN, buff=0.2)
        with self.voice(N["ml3"]) as v:
            self.wait_until(v, "popular")
            self.play(FadeIn(ttl), run_time=0.6)
            self.wait_until(v, "matrix")
            self.play(FadeIn(W), FadeIn(Wl), run_time=0.7)
            self.wait_until(v, "sixteen")
            self.play(FadeIn(eq), FadeIn(A_), FadeIn(B_), FadeIn(Al), FadeIn(Bl), FadeIn(rk16), run_time=1)
            self.wait_until(v, "Fewer")
            self.play(FadeIn(nums), FadeIn(src), run_time=0.8)
        self.clear_body()
        ax_t = T("Axiom A · Linearity", 60, YELLOW, weight=BOLD).shift(UP * 0.6)
        sub = T("independence · basis · dimension · rank = questions about add and scale", 26, FG)
        sub.next_to(ax_t, DOWN, buff=0.5)
        with self.voice(N["axiom"]) as v:
            self.wait_until(v, "Independence")
            self.play(FadeIn(sub), run_time=0.8)
            self.wait_until(v, "axiom")
            self.play(Write(ax_t), run_time=1)


class C7_Practice(Studio):
    def construct(self):
        self.heading("6 · Try it yourself")
        q = MathTex(r"\begin{bmatrix}1\\2\end{bmatrix}", ",", r"\begin{bmatrix}3\\6\end{bmatrix}",
                    r"\quad\text{independent?}", font_size=64)
        q[0].set_color(BLUE); q[2].set_color(GREEN)
        q.shift(UP * 1.5)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac1"]) as v:
            self.play(Write(q), run_time=1.5)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
        a1 = MathTex(r"\begin{bmatrix}3\\6\end{bmatrix}", "=", "3", r"\begin{bmatrix}1\\2\end{bmatrix}",
                     r"\ \Rightarrow\ \text{dependent}", font_size=48)
        a1[0].set_color(GREEN); a1[2].set_color(PURPLE); a1[3].set_color(BLUE); a1[4].set_color(RED)
        a2 = MathTex("3", r"\begin{bmatrix}1\\2\end{bmatrix}", "-", "1", r"\begin{bmatrix}3\\6\end{bmatrix}",
                     "=", r"\begin{bmatrix}0\\0\end{bmatrix}", font_size=48)
        for i, c in [(0, PURPLE), (1, BLUE), (3, PURPLE), (4, GREEN), (6, YELLOW)]:
            a2[i].set_color(c)
        VGroup(a1, a2).arrange(DOWN, buff=0.5).move_to(DOWN * 1.3)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "dependent")
            self.play(Write(a1), run_time=1.2)
            self.wait_until(v, "weights")
            self.play(Write(a2), run_time=1.2)
            self.wait_until(v, "gives")
            self.play(Indicate(a2[6], color=YELLOW), run_time=0.8)
        self.clear_body()
        M = MathTex(r"\text{rank}", r"\begin{bmatrix}1&2&3\\2&4&6\end{bmatrix}", r"=\ ?", font_size=64)
        M[2].set_color(YELLOW)
        M.shift(UP * 1.4)
        pause = T("⏸  pause and try", 30, MUTED).next_to(M, DOWN, buff=0.5)
        with self.voice(N["prac3"]) as v:
            self.play(Write(M), run_time=1.2)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
        r1 = MathTex(r"\text{row}_2 = 2\cdot\text{row}_1", r"\ \Rightarrow\ \text{rank} = 1", font_size=46)
        r1[1].set_color(YELLOW)
        r2 = MathTex(r"\text{columns } \begin{bmatrix}1\\2\end{bmatrix},\begin{bmatrix}2\\4\end{bmatrix},"
                     r"\begin{bmatrix}3\\6\end{bmatrix}\ \text{all multiples of } \begin{bmatrix}1\\2\end{bmatrix}",
                     font_size=42)
        VGroup(r1, r2).arrange(DOWN, buff=0.5).move_to(DOWN * 1.4)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "second")
            self.play(Write(r1[0]), run_time=0.8)
            self.wait_until(v, "rank")
            self.play(Write(r1[1]), run_time=0.6)
            self.wait_until(v, "columns")
            self.play(Write(r2), run_time=1.2)


CODE = '''import numpy as np
rk = np.linalg.matrix_rank          # counts independent columns
a, b, c = np.array([2, 1]), np.array([1, 2]), np.array([3, 3])
print("a + b - c =", a + b - c)
print("rank [a b c] =", rk(np.column_stack([a, b, c])))
B = np.column_stack([[1, 1], [-1, 1]])          # basis b1, b2
print("coordinates of p =", np.linalg.solve(B, [1, 3]))
R1 = np.outer([1, 2, 0], [1, -1, 3])
R2 = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 2]])
print("ranks:", rk(R1), rk(R2), rk(np.eye(3)))
m = np.array([1.2, 0.8, 2.0])                   # toy table
table = np.column_stack([m, 100 * m, [5, 3, 9]])
print("rank of toy table =", rk(table))'''


class C7_Code(Studio):
    def construct(self):
        self.heading("7 · Check it in Python")
        code = self.code_panel(CODE, font_size=22)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[1], color=YELLOW, buff=0.06)
            self.wait_until(v, "function")
            self.play(Create(box), run_time=0.5)
            for i, w in [(4, "test"), (6, "solve"), (9, "examples"), (12, "table")]:
                self.wait_until(v, w)
                self.play(Transform(box, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                              buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=24, command="python demo.py").shift(DOWN * 0.6)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for idx, w in [((0,), "zero"), ((1,), "two"), ((2,), "coordinates"),
                           ((4, 5, 6), "examples"), ((11,), "table")]:
                self.wait_until(v, w)
                self.play(*[Indicate(term[1][i], color=YELLOW) for i in idx], run_time=0.7)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["Independent: only all-zero weights give the zero vector",
                            "Basis = independent set that spans the space",
                            "Coordinates are unique in a basis, and change with the basis",
                            "Every basis has the same size: the dimension",
                            "Rank = number of independent columns = independent rows"],
                           size=28).shift(UP * 0.5)
        with self.voice(N["sum1"]) as v:
            for p in pts[:3]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 5)
        nxt = VGroup(T("Next · Video 4", 26, MUTED),
                     T("Linear mapping: functions that respect add and scale", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[3], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "rank")
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Redundant", "C2_Independence", "C3_Basis", "C4_Rank", "C5_ML",
          "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Redundant": "Redundant directions",
                  "C2_Independence": "Linear independence", "C3_Basis": "Basis, coordinates, dimension",
                  "C4_Rank": "The rank of a matrix", "C5_ML": "Rank in ML",
                  "C7_Practice": "Try it yourself", "C7_Code": "Python demo",
                  "C8_Summary": "Summary", "C9_Credits": "Credits"}
