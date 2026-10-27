"""Linear mapping — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 4 of 16"
TITLE = "Linear mapping — functions that respect add and scale"


def arr(plane, a, b, color, width=6):
    return Arrow(plane.c2p(*a), plane.c2p(*b), buff=0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.18)


def grid_plane(xr, yr, xl, yl):
    return NumberPlane(x_range=xr, y_range=yr, x_length=xl, y_length=yl,
                       background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.35,
                                              "stroke_width": 1},
                       axis_config={"stroke_color": MUTED})


def moving_grid(r, unit, origin):
    g = NumberPlane(x_range=[-r, r, 1], y_range=[-r, r, 1], x_length=2 * r * unit,
                    y_length=2 * r * unit,
                    background_line_style={"stroke_color": BLUE, "stroke_opacity": 0.6,
                                           "stroke_width": 2},
                    axis_config={"stroke_color": FG})
    return g.move_to(origin)


def mat(rows, scale=1.0, color=FG, h_buff=1.0, v_buff=0.8):
    m = Matrix([[str(e) for e in r] for r in rows], h_buff=h_buff, v_buff=v_buff)
    m.set_color(color)
    m.scale(scale)
    return m


def colour_cols(m, colours):
    for c, k in zip(m.get_columns(), colours):
        c.set_color(k)
    return m


def colmath(parts, colours, size=46):
    m = MathTex(*parts, font_size=size)
    for i, c in colours:
        m[i].set_color(c)
    return m


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        if title.width > 12.5:
            title.scale_to_fit_width(12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.4)
        xin = MathTex("x", color=BLUE, font_size=60)
        box = VGroup(RoundedRectangle(corner_radius=0.15, width=2.4, height=1.2, color=PURPLE),
                     T("layer", 30, PURPLE))
        box[1].move_to(box[0])
        yout = MathTex("y", color=YELLOW, font_size=60)
        flow = VGroup(xin, box, yout).arrange(RIGHT, buff=1.3).shift(DOWN * 0.3)
        a1 = Arrow(xin.get_right(), box.get_left(), buff=0.15, color=MUTED)
        a2 = Arrow(box.get_right(), yout.get_left(), buff=0.15, color=MUTED)
        ok = T("respects add ✓  and scale ✓", 28, GREEN).next_to(flow, DOWN, buff=0.5)
        name = T("linear mapping", 44, YELLOW, weight=BOLD).next_to(ok, DOWN, buff=0.4)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            self.wait_until(v, "vector")
            self.play(FadeIn(xin), GrowArrow(a1), FadeIn(box), GrowArrow(a2), FadeIn(yout), run_time=1.2)
            self.wait_until(v, "respects")
            self.play(FadeIn(ok), run_time=0.6)
            self.wait_until(v, "linear")
            self.play(Write(name), run_time=0.8)
        plan = self.bullets(["the definition", "what it does to a grid", "maps that are not linear",
                             "the matrix of any linear map"], size=28).move_to(DOWN * 0.9)
        pre = T("You need: videos 1–3 · vector, matrix, basis", 24, MUTED).to_edge(DOWN, buff=0.4)
        with self.voice(N["intro2"]) as v:
            self.play(FadeOut(VGroup(xin, a1, box, a2, yout, ok, name)), run_time=0.5)
            for p, w in zip(plan, ["definition", "grid", "not", "matrix"]):
                self.wait_until(v, w)
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
            self.wait_until(v, "need")
            self.play(FadeIn(pre), run_time=0.6)


class C1_Definition(Studio):
    def construct(self):
        self.heading("1 · What makes a map linear")
        xin = MathTex("x", color=BLUE, font_size=56)
        box = VGroup(RoundedRectangle(corner_radius=0.15, width=1.4, height=1.0, color=PURPLE),
                     MathTex("f", color=PURPLE, font_size=56))
        box[1].move_to(box[0])
        fx = MathTex("f(x)", color=YELLOW, font_size=56)
        flow = VGroup(xin, box, fx).arrange(RIGHT, buff=1.0).move_to(UP * 1.7)
        a1 = Arrow(xin.get_right(), box.get_left(), buff=0.12, color=MUTED)
        a2 = Arrow(box.get_right(), fx.get_left(), buff=0.12, color=MUTED)
        lin = T("linear = respects adding and scaling", 30, GREEN).next_to(flow, DOWN, buff=0.5)
        with self.voice(N["def1"]) as v:
            self.wait_until(v, "input")
            self.play(FadeIn(xin), GrowArrow(a1), FadeIn(box), run_time=0.8)
            self.wait_until(v, "output")
            self.play(GrowArrow(a2), FadeIn(fx), run_time=0.6)
            self.wait_until(v, "linear")
            self.play(FadeIn(lin), run_time=0.7)
        f = colmath(["f(", "a", "u", "+", "b", "v", ")", "=", "a", r"\,f(u)", "+", "b", r"\,f(v)"],
                    [(1, PURPLE), (2, BLUE), (4, PURPLE), (5, GREEN), (8, PURPLE), (9, BLUE),
                     (11, PURPLE), (12, GREEN)], 66)
        f.move_to(UP * 0.2)
        card = self.term_card("linear mapping", "A function f with f(a·u + b·v) = a·f(u) + b·f(v): "
                              "it respects adding and scaling.", pos=DR, width=5.4)
        lb = SurroundingRectangle(f[:7], color=YELLOW, buff=0.1)
        rb = SurroundingRectangle(f[8:], color=YELLOW, buff=0.1)
        with self.voice(N["def2"]) as v:
            self.play(FadeOut(flow), FadeOut(a1), FadeOut(a2), lin.animate.move_to(UP * 2.2), run_time=0.6)
            self.wait_until(v, "definition")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "equals")
            self.play(Write(f), run_time=1.5)
            self.wait_until(v, "combine")
            self.play(Create(lb), run_time=0.6)
            self.wait_until(v, "apply", 2)
            self.play(ReplacementTransform(lb, rb), run_time=0.6)
        leg = VGroup(*[VGroup(MathTex(s, color=c, font_size=38), T(t, 24, MUTED)).arrange(RIGHT, buff=0.3)
                       for s, c, t in [("u,\\ v", BLUE, "input vectors"),
                                       ("a,\\ b", PURPLE, "scalars (the weights)"),
                                       (r"f(au+bv)", FG, "left: combine first, then apply f"),
                                       (r"af(u)+bf(v)", FG, "right: apply f first, then combine")]])
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(DOWN * 2.2)
        with self.voice(N["def3"]) as v:
            self.play(FadeOut(card), FadeOut(rb), run_time=0.4)
            for row, (w, n) in zip(leg, [("letters", 1), ("letters", 2), ("left", 1), ("right", 1)]):
                self.wait_until(v, w, n)
                self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.6)
        self.clear_body()
        r1 = colmath(["f(u+v)", "=", "f(u)+f(v)"], [(0, YELLOW), (2, YELLOW)], 52)
        r2 = colmath(["f(c\\,u)", "=", "c\\,f(u)"], [(0, PURPLE), (2, PURPLE)], 52)
        rules = VGroup(r1, r2).arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to(LEFT * 3.4 + UP * 0.9)
        ex = MathTex(r"f(x)=3x", font_size=52, color=BLUE).move_to(RIGHT * 3.4 + UP * 1.9)
        e1 = MathTex(r"f(2+5)=f(7)=21", font_size=44)
        e2 = MathTex(r"f(2)+f(5)=6+15=21\ \checkmark", font_size=44, color=GREEN)
        VGroup(e1, e2).arrange(DOWN, buff=0.35, aligned_edge=LEFT).next_to(ex, DOWN, buff=0.5)
        with self.voice(N["def4"]) as v:
            self.wait_until(v, "First")
            self.play(Write(r1), run_time=1)
            self.wait_until(v, "Second")
            self.play(Write(r2), run_time=1)
            self.wait_until(v, "three")
            self.play(FadeIn(ex), run_time=0.6)
            self.wait_until(v, "Then")
            self.play(Write(e1), run_time=0.9)
            self.wait_until(v, "fifteen")
            self.play(Write(e2), run_time=0.9)
        z1 = MathTex(r"f(0\cdot u)", "=", r"0\cdot f(u)", "=", r"\mathbf{0}", font_size=48)
        z1[4].set_color(YELLOW)
        z0 = MathTex(r"f(\mathbf{0})=\mathbf{0}\ \text{ always}", font_size=48, color=YELLOW)
        warn = T("moves the origin → not linear", 30, RED)
        VGroup(z0, z1, warn).arrange(DOWN, buff=0.35).move_to(DOWN * 2.1)
        with self.voice(N["def5"]) as v:
            self.play(VGroup(rules, ex, e1, e2).animate.shift(UP * 0.4), run_time=0.5)
            self.wait_until(v, "zero")
            self.play(Write(z0), run_time=0.8)
            self.wait_until(v, "Take")
            self.play(Write(z1), run_time=1.2)
            self.wait_until(v, "moves")
            self.play(FadeIn(warn), run_time=0.6)


class C2_Pictures(Studio):
    def construct(self):
        self.heading("2 · What a linear map does to a grid")
        bg = grid_plane([-3, 3, 1], [-2, 2, 1], 5.4, 3.6).move_to(LEFT * 3.3 + DOWN * 0.5)
        o = bg.c2p(0, 0)
        mv = moving_grid(2, 0.9, o)
        M = np.array([[1.0, 0.3], [0.2, 0.8]])
        lab = T("some linear map", 22, MUTED).next_to(bg, DOWN, buff=0.2)
        pts = VGroup(*[T(s, 28, FG) for s in ["lines stay straight", "parallel lines stay parallel",
                                              "spacing stays even", "the origin stays fixed"]])
        pts.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(RIGHT * 3.7 + DOWN * 0.3)
        dot = Dot(o, color=YELLOW, radius=0.09)
        with self.voice(N["pic1"]) as v:
            self.play(Create(bg), Create(mv), FadeIn(lab), run_time=1)
            self.wait_until(v, "After")
            self.play(mv.animate.apply_matrix(M, about_point=o), run_time=2)
            for p, w in zip(pts, ["straight", "Parallel", "spacing", "origin"]):
                self.wait_until(v, w)
                self.play(FadeIn(p, shift=LEFT * 0.2), run_time=0.5)
            self.play(FadeIn(dot), Flash(o, color=YELLOW), run_time=0.8)
        self.clear_body()
        names = ["stretch", "rotation 90°", "shear", "projection"]
        Ms = [demo.np.array([[2, 0], [0, 1]]), demo.np.array([[0, -1], [1, 0]]), demo.S, demo.P]
        panels = []
        for k, (nm, Mk) in enumerate(zip(names, Ms)):
            cx = -5.3 + 3.55 * k
            pb = grid_plane([-2, 2, 1], [-1.5, 1.5, 1], 2.8, 2.1).move_to([cx, 0.4, 0])
            ok = pb.c2p(0, 0)
            g = moving_grid(1, 0.7, ok)
            t = arr(pb, (0, 0), (1, 1), YELLOW, 5)
            tt = T(nm, 26, FG).next_to(pb, UP, buff=0.2)
            mm = mat(Mk.tolist(), 0.7).next_to(pb, DOWN, buff=0.35)
            panels.append((pb, ok, g, t, tt, mm, Mk))
        with self.voice(N["pic2"]) as v:
            self.play(*[AnimationGroup(Create(p[0]), Create(p[2]), GrowArrow(p[3]), FadeIn(p[4]),
                                       FadeIn(p[5])) for p in panels], run_time=1.5)
            for k, w in [(0, "stretch"), (1, "rotation"), (2, "shear")]:
                pb, ok, g, t, tt, mm, Mk = panels[k]
                self.wait_until(v, w)
                tip = tuple(int(c) for c in Mk @ [1, 1])
                if k == 1:
                    self.play(Rotate(VGroup(g, t), angle=PI / 2, about_point=ok), run_time=1.5)
                else:
                    self.play(g.animate.apply_matrix(Mk, about_point=ok),
                              Transform(t, arr(pb, (0, 0), tip, YELLOW, 5)), run_time=1.5)
            self.wait_until(v, "deck")
            self.play(Indicate(panels[2][2], color=BLUE), run_time=0.8)
        ok_all = T("all four: straight, parallel, evenly spaced, origin fixed → linear ✓", 24, GREEN)
        ok_all.to_edge(DOWN, buff=0.35)
        with self.voice(N["pic3"]) as v:
            pb, ok, g, t, tt, mm, Mk = panels[3]
            self.wait_until(v, "projection")
            self.play(g.animate.apply_matrix(Mk, about_point=ok),
                      Transform(t, arr(pb, (0, 0), (1, 0), YELLOW, 5)), run_time=1.8)
            self.wait_until(v, "collapses")
            self.play(Indicate(g, color=YELLOW), run_time=0.8)
            self.wait_until(v, "rules")
            self.play(FadeIn(ok_all), run_time=0.6)


class C3_NonExamples(Studio):
    def construct(self):
        self.heading("3 · Maps that are not linear")
        nl = NumberLine(x_range=[-1, 6, 1], length=8, include_numbers=True, color=MUTED,
                        font_size=28).move_to(LEFT * 1.8 + UP * 0.6)
        f = MathTex(r"f(x)=x+1", font_size=56, color=ORANGE).move_to(LEFT * 1.8 + UP * 2.2)
        dots = VGroup(*[Dot(nl.n2p(k), color=BLUE, radius=0.09) for k in range(5)])
        moved = VGroup(*[Dot(nl.n2p(k + 1), color=YELLOW, radius=0.09) for k in range(5)])
        o_bad = MathTex(r"f(0)=1\neq 0", font_size=48, color=RED).move_to(RIGHT * 4.6 + UP * 1.9)
        verdict = T("origin moved → not linear", 28, RED).next_to(o_bad, DOWN, buff=0.35)
        with self.voice(N["non1"]) as v:
            self.play(Create(nl), run_time=0.8)
            self.wait_until(v, "plus")
            self.play(Write(f), FadeIn(dots), run_time=0.8)
            self.wait_until(v, "shifts")
            self.play(*[TransformFromCopy(d, m) for d, m in zip(dots, moved)], run_time=1.2)
            self.wait_until(v, "origin")
            self.play(Indicate(moved[0], color=RED), Indicate(dots[0], color=RED), run_time=0.8)
            self.wait_until(v, "not")
            self.play(Write(o_bad), run_time=0.8)
            self.wait_until(v, "linear", 2)
            self.play(FadeIn(verdict), run_time=0.6)
        c1 = MathTex(r"f(2+3)=f(5)=6", font_size=44)
        c2 = MathTex(r"f(2)+f(3)=3+4=7", font_size=44)
        c3 = MathTex(r"6\neq 7", font_size=48, color=RED)
        chk = VGroup(c1, c2, c3).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(LEFT * 3.4 + DOWN * 2.0)
        aff = VGroup(T("affine map = linear map, then a move", 28, ORANGE),
                     T("lines stay straight, but the origin moves", 24, MUTED)).arrange(DOWN, buff=0.2)
        aff.move_to(RIGHT * 3.3 + DOWN * 2.0)
        with self.voice(N["non2"]) as v:
            self.wait_until(v, "Check")
            self.play(Write(c1), run_time=0.9)
            self.wait_until(v, "But")
            self.play(Write(c2), run_time=0.9)
            self.wait_until(v, "Six", 2)
            self.play(Write(c3), run_time=0.6)
            self.wait_until(v, "affine")
            self.play(FadeIn(aff), run_time=0.8)
        self.clear_body()
        g = MathTex(r"f(x)=x^2", font_size=56, color=RED).move_to(UP * 2.0)
        s1 = MathTex(r"f(1+1)=f(2)=4", font_size=46)
        s2 = MathTex(r"f(1)+f(1)=1+1=2", font_size=46)
        s3 = MathTex(r"4\neq 2", font_size=48, color=RED)
        t1 = MathTex(r"f(2\cdot3)=f(6)=36", font_size=46)
        t2 = MathTex(r"2\cdot f(3)=2\cdot9=18", font_size=46)
        t3 = MathTex(r"36\neq 18", font_size=48, color=RED)
        left = VGroup(s1, s2, s3).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(LEFT * 3.3 + DOWN * 0.4)
        right = VGroup(t1, t2, t3).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(RIGHT * 3.3 + DOWN * 0.4)
        tags = VGroup(T("adding fails", 24, MUTED).next_to(left, UP, buff=0.3),
                      T("scaling fails", 24, MUTED).next_to(right, UP, buff=0.3))
        with self.voice(N["non3"]) as v:
            self.wait_until(v, "squared")
            self.play(Write(g), run_time=0.8)
            self.play(Write(s1), FadeIn(tags[0]), run_time=0.9)
            self.wait_until(v, "But")
            self.play(Write(s2), run_time=0.9)
            self.wait_until(v, "Four", 2)
            self.play(Write(s3), run_time=0.5)
            self.wait_until(v, "Scaling")
            self.play(Write(t1), FadeIn(tags[1]), run_time=0.9)
            self.wait_until(v, "eighteen")
            self.play(Write(t2), Write(t3), run_time=0.9)
        self.clear_body()
        ax = Axes(x_range=[-2, 2, 1], y_range=[-2, 4, 1], x_length=6, y_length=5, tips=False,
                  axis_config={"stroke_color": MUTED}).move_to(LEFT * 2.6 + DOWN * 0.5)
        p1 = ax.plot(lambda t: 2 * t, x_range=[-1, 2], color=GREEN, stroke_width=5)
        p2 = ax.plot(lambda t: t + 1, x_range=[-2, 2], color=ORANGE, stroke_width=5)
        p3 = ax.plot(lambda t: t ** 2, x_range=[-2, 2], color=RED, stroke_width=5)
        leg = VGroup(T("2x: straight, through 0 → linear", 26, GREEN),
                     T("x + 1: straight, misses 0 → affine", 26, ORANGE),
                     T("x²: curved → not linear", 26, RED)).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        leg.move_to(RIGHT * 3.6 + DOWN * 0.3)
        with self.voice(N["non4"]) as v:
            self.play(Create(ax), run_time=0.8)
            self.wait_until(v, "two")
            self.play(Create(p1), FadeIn(leg[0]), run_time=0.9)
            self.wait_until(v, "shift")
            self.play(Create(p2), FadeIn(leg[1]), run_time=0.9)
            self.wait_until(v, "squared")
            self.play(Create(p3), FadeIn(leg[2]), run_time=0.9)
            self.wait_until(v, "first")
            self.play(Indicate(p1, color=GREEN), Indicate(leg[0], color=GREEN), run_time=0.9)


class C4_Matrix(Studio):
    def construct(self):
        self.heading("4 · Every linear map is a matrix")
        thm = T("every linear map  =  a matrix", 40, YELLOW, weight=BOLD).move_to(UP * 2.0)
        A = colmath(["A", "=", r"\Big[", r"\;f(e_1)", r"\quad", r"f(e_2)\;", r"\Big]"],
                    [(3, BLUE), (5, GREEN)], 64).move_to(UP * 0.6)
        cl = T("columns = where the basis vectors land", 26, MUTED).next_to(A, DOWN, buff=0.3)
        rem = self.reminder_card("basis", "An independent set of vectors that spans the space; "
                                 "here e₁, e₂ (video 3).", pos=DR, width=4.6)
        with self.voice(N["mat1"]) as v:
            self.wait_until(v, "theorem")
            self.play(Write(thm), run_time=1)
            self.wait_until(v, "columns")
            self.play(Write(A), FadeIn(cl), FadeIn(rem, shift=UP * 0.3), run_time=1.2)
            self.wait_until(v, "first")
            self.play(Indicate(A[3], color=BLUE), run_time=0.7)
            self.wait_until(v, "second")
            self.play(Indicate(A[5], color=GREEN), run_time=0.7)
        d1 = colmath(["x", "=", "x_1", "e_1", "+", "x_2", "e_2"], [(2, PURPLE), (3, BLUE), (5, PURPLE), (6, GREEN)], 46)
        d2 = colmath(["f(x)", "=", "x_1", "f(e_1)", "+", "x_2", "f(e_2)"],
                     [(0, YELLOW), (2, PURPLE), (3, BLUE), (5, PURPLE), (6, GREEN)], 46)
        d3 = colmath(["f(x)", "=", "A", "x"], [(0, YELLOW)], 46)
        der = VGroup(d1, d2, d3).arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_edge(LEFT, buff=0.8).shift(DOWN * 2.1)
        why = T("f is linear", 22, MUTED).next_to(d2, RIGHT, buff=0.4)
        with self.voice(N["mat2"]) as v:
            self.play(FadeOut(rem), run_time=0.4)
            self.wait_until(v, "Any")
            self.play(Write(d1), run_time=0.9)
            self.wait_until(v, "Because")
            self.play(Write(d2), FadeIn(why), run_time=1)
            self.wait_until(v, "matrix")
            self.play(Write(d3), run_time=0.7)
        self.clear_body()
        bg = grid_plane([-3, 6, 1], [-2, 5, 1], 4.95, 3.85).move_to(LEFT * 3.8 + DOWN * 0.6)
        bg.shift(bg.get_center() - bg.c2p(1.5, 1.5))
        o = bg.c2p(0, 0)
        g = moving_grid(1, 0.55, o)
        e1 = arr(bg, (0, 0), (1, 0), BLUE)
        e2 = arr(bg, (0, 0), (0, 1), GREEN)
        Am = colour_cols(mat(demo.A.tolist(), 0.9), [BLUE, GREEN])
        Ag = VGroup(MathTex("A", "=", font_size=48), Am).arrange(RIGHT, buff=0.2).move_to(RIGHT * 3.6 + UP * 1.9)
        sends = VGroup(colmath([r"e_1", r"\mapsto", r"\begin{bmatrix}2\\1\end{bmatrix}"], [(0, BLUE), (2, BLUE)], 40),
                       colmath([r"e_2", r"\mapsto", r"\begin{bmatrix}-1\\1\end{bmatrix}"], [(0, GREEN), (2, GREEN)], 40))
        sends.arrange(RIGHT, buff=0.6).move_to(RIGHT * 3.6 + UP * 0.4)
        with self.voice(N["mat3"]) as v:
            self.play(Create(bg), Create(g), GrowArrow(e1), GrowArrow(e2), run_time=1)
            self.wait_until(v, "sends")
            self.play(Transform(e1, arr(bg, (0, 0), tuple(demo.fe1), BLUE)), FadeIn(sends[0]), run_time=0.9)
            self.wait_until(v, "sends", 2)
            self.play(Transform(e2, arr(bg, (0, 0), tuple(demo.fe2), GREEN)), FadeIn(sends[1]), run_time=0.9)
            self.wait_until(v, "matrix")
            self.play(FadeIn(Ag), run_time=0.8)
            self.wait_until(v, "Two", 3)
            self.play(g.animate.apply_matrix(demo.A, about_point=o), run_time=1.5)
        p1 = DashedLine(bg.c2p(0, 0), bg.c2p(*(3 * demo.fe1)), color=BLUE, stroke_width=4)
        p2 = DashedLine(bg.c2p(*(3 * demo.fe1)), bg.c2p(*demo.Ax), color=GREEN, stroke_width=4)
        res = arr(bg, (0, 0), tuple(demo.Ax), YELLOW, 7)
        xl = MathTex(r"x=\begin{bmatrix}3\\2\end{bmatrix}", font_size=40, color=PURPLE)
        cv = colmath(["3", r"\begin{bmatrix}2\\1\end{bmatrix}", "+", "2", r"\begin{bmatrix}-1\\1\end{bmatrix}",
                      "=", r"\begin{bmatrix}6-2\\3+2\end{bmatrix}", "=",
                      r"\begin{bmatrix}%d\\%d\end{bmatrix}" % tuple(demo.Ax)],
                     [(0, PURPLE), (1, BLUE), (3, PURPLE), (4, GREEN), (8, YELLOW)], 38)
        rv = MathTex(r"\text{rows: } 2\cdot3-1\cdot2=4,\ \ 1\cdot3+1\cdot2=5", font_size=32, color=MUTED)
        grp = VGroup(xl, cv, rv).arrange(DOWN, buff=0.35).move_to(RIGHT * 3.6 + DOWN * 1.7)
        with self.voice(N["mat4"]) as v:
            self.play(FadeOut(sends), run_time=0.4)
            self.wait_until(v, "three")
            self.play(FadeIn(xl), run_time=0.6)
            self.wait_until(v, "Use")
            self.play(Create(p1), Write(cv[:3]), run_time=1)
            self.wait_until(v, "six")
            self.play(Create(p2), Write(cv[3:7]), run_time=1)
            self.wait_until(v, "five")
            self.play(GrowArrow(res), Write(cv[7:]), run_time=0.9)
            self.wait_until(v, "row")
            self.play(FadeIn(rv), run_time=0.6)
        self.clear_body()
        pl = grid_plane([-2, 2, 1], [-2, 2, 1], 4, 4).move_to(LEFT * 3.6 + DOWN * 0.5)
        r1 = arr(pl, (0, 0), (1, 0), BLUE)
        r2 = arr(pl, (0, 0), (0, 1), GREEN)
        Rm = colour_cols(mat(demo.np.array([[0, -1], [1, 0]]).tolist(), 1.0), [BLUE, GREEN])
        Rg = VGroup(MathTex("R", "=", font_size=56), Rm).arrange(RIGHT, buff=0.2).move_to(RIGHT * 3.4 + UP * 0.6)
        same = T("= the rotation matrix of video 2", 26, YELLOW).next_to(Rg, DOWN, buff=0.5)
        with self.voice(N["mat5"]) as v:
            self.wait_until(v, "rotation")
            self.play(Create(pl), GrowArrow(r1), GrowArrow(r2), run_time=1)
            self.wait_until(v, "sends")
            self.play(Rotate(VGroup(r1, r2), angle=PI / 2, about_point=pl.c2p(0, 0)), run_time=1.5)
            self.wait_until(v, "matrix")
            self.play(FadeIn(Rg), run_time=0.8)
            self.wait_until(v, "exactly")
            self.play(FadeIn(same), run_time=0.6)


class C5_Compose(Studio):
    def construct(self):
        self.heading("5 · Composition, kernel and image")
        gen = colmath(["g(f(x))", "=", "(B\\,A)", "\\,x"], [(0, YELLOW), (2, YELLOW)], 56).move_to(UP * 2.0)
        gl = T("f has matrix A, g has matrix B: first f, then g", 22, MUTED).next_to(gen, DOWN, buff=0.2)
        S = mat(demo.S.tolist(), 0.9)
        P = mat(demo.P.tolist(), 0.9)
        PS = mat(demo.PS.tolist(), 0.9, YELLOW)
        prod = VGroup(MathTex("P", font_size=44), P, MathTex("S", font_size=44), S, MathTex("=", font_size=50), PS)
        prod.arrange(RIGHT, buff=0.25).move_to(DOWN * 0.3)
        prod[0].next_to(P, UP, buff=0.12); prod[2].next_to(S, UP, buff=0.12)
        names = VGroup(T("projection", 22, MUTED).next_to(P, DOWN, buff=0.2),
                       T("shear", 22, MUTED).next_to(S, DOWN, buff=0.2))
        with self.voice(N["comp1"]) as v:
            self.wait_until(v, "product")
            self.play(Write(gen), FadeIn(gl), run_time=1.2)
            self.wait_until(v, "shear")
            self.play(FadeIn(prod[2:4]), FadeIn(names[1]), run_time=0.6)
            self.wait_until(v, "projection")
            self.play(FadeIn(prod[0:2]), FadeIn(names[0]), run_time=0.6)
            self.wait_until(v, "combined")
            self.play(FadeIn(prod[4:]), run_time=0.8)
        pl = grid_plane([-1, 4, 1], [-1, 3, 1], 4.5, 3.6).move_to(LEFT * 3.8 + DOWN * 1.2)
        pl.shift(pl.get_center() - pl.c2p(1.5, 1))
        y0 = arr(pl, (0, 0), tuple(demo.y), BLUE)
        y1 = arr(pl, (0, 0), tuple(demo.S @ demo.y), PURPLE)
        y2 = arr(pl, (0, 0), tuple(demo.PS @ demo.y), YELLOW, 7)
        steps = VGroup(MathTex(r"S\begin{bmatrix}1\\2\end{bmatrix}=\begin{bmatrix}3\\2\end{bmatrix}", font_size=40, color=PURPLE),
                       MathTex(r"P\begin{bmatrix}3\\2\end{bmatrix}=\begin{bmatrix}3\\0\end{bmatrix}", font_size=40, color=YELLOW),
                       MathTex(r"(PS)\begin{bmatrix}1\\2\end{bmatrix}=\begin{bmatrix}1+2\\0\end{bmatrix}=\begin{bmatrix}3\\0\end{bmatrix}",
                               font_size=40, color=YELLOW))
        steps.arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(RIGHT * 3.5 + DOWN * 1.6)
        with self.voice(N["comp2"]) as v:
            self.play(FadeOut(gen), FadeOut(gl), VGroup(prod, names).animate.scale(0.8).move_to(RIGHT * 3.5 + UP * 1.6),
                      run_time=0.8)
            self.play(Create(pl), run_time=0.6)
            self.wait_until(v, "one")
            self.play(GrowArrow(y0), run_time=0.6)
            self.wait_until(v, "three")
            self.play(TransformFromCopy(y0, y1), FadeIn(steps[0]), run_time=0.9)
            self.wait_until(v, "zero")
            self.play(TransformFromCopy(y1, y2), FadeIn(steps[1]), run_time=0.9)
            self.wait_until(v, "matrix")
            self.play(FadeIn(steps[2]), run_time=0.8)
        self.clear_body()
        kp = grid_plane([-3, 3, 1], [-1, 5, 1], 3.6, 3.6).move_to(LEFT * 3.8 + DOWN * 0.5)
        kp.shift(kp.get_center() - kp.c2p(0, 2))
        kd = VGroup(*[Dot(kp.c2p(0, k), color=ORANGE, radius=0.08) for k in range(1, 6)])
        kline = Line(kp.c2p(0, -1), kp.c2p(0, 5), color=ORANGE, stroke_width=6)
        kdef = MathTex(r"\ker f = \{\,x : f(x)=\mathbf{0}\,\}", font_size=44).move_to(RIGHT * 2.2 + UP * 1.9)
        kres = T("kernel of the projection: the vertical axis", 26, ORANGE).next_to(kdef, DOWN, buff=0.4)
        card = self.term_card("kernel (null space)", "All inputs that the map sends to the zero "
                              "vector.", pos=DR, width=5.0)
        with self.voice(N["ker1"]) as v:
            self.play(Create(kp), run_time=0.6)
            self.wait_until(v, "kernel")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "zero")
            self.play(Write(kdef), run_time=1)
            self.wait_until(v, "vertical")
            self.play(FadeIn(kd), run_time=0.6)
            self.wait_until(v, "drops")
            self.play(*[d.animate.move_to(kp.c2p(0, 0)) for d in kd], run_time=1.2)
            self.wait_until(v, "vertical", 2)
            self.play(Create(kline), FadeIn(kres), run_time=0.8)
        pts = [(-2, 3), (-1, 1), (1, 4), (2, 2), (2.5, 0.5), (-2.5, 1.5)]
        idots = VGroup(*[Dot(kp.c2p(*q), color=BLUE, radius=0.08) for q in pts])
        iline = Line(kp.c2p(-3, 0), kp.c2p(3, 0), color=YELLOW, stroke_width=6)
        idef = MathTex(r"\text{image} = \{\,f(x)\,\}", font_size=44).next_to(kres, DOWN, buff=0.6)
        ires = T("image of the projection: the horizontal axis", 26, YELLOW).next_to(idef, DOWN, buff=0.3)
        rk = T("dim image = rank = 1 (video 3)", 26, FG).next_to(ires, DOWN, buff=0.3)
        card2 = self.term_card("image (range)", "All the outputs that the map can produce.", pos=DR, width=4.6)
        with self.voice(N["ker2"]) as v:
            self.play(FadeOut(card), FadeOut(kd), run_time=0.4)
            self.wait_until(v, "image")
            self.play(FadeIn(card2, shift=UP * 0.3), FadeIn(idef), FadeIn(idots), run_time=0.8)
            self.wait_until(v, "produces")
            self.play(*[d.animate.move_to(kp.c2p(q[0], 0)).set_color(YELLOW) for d, q in zip(idots, pts)],
                      run_time=1.2)
            self.wait_until(v, "axis")
            self.play(Create(iline), FadeIn(ires), run_time=0.8)
            self.wait_until(v, "rank")
            self.play(FadeIn(rk), run_time=0.6)
        bal = colmath([r"\dim\ker", "+", r"\dim\text{image}", "=", r"\dim\text{input}"],
                      [(0, ORANGE), (2, YELLOW)], 44)
        num = colmath(["1", "+", "1", "=", "2"], [(0, ORANGE), (2, YELLOW)], 50)
        words = T("what it destroys + what it keeps = everything", 24, MUTED)
        bg_ = VGroup(bal, num, words).arrange(DOWN, buff=0.3).move_to(RIGHT * 2.6 + DOWN * 2.6)
        with self.voice(N["ker3"]) as v:
            self.play(FadeOut(card2), FadeOut(VGroup(kdef, idef)),
                      VGroup(kres, ires, rk).animate.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(RIGHT * 2.6 + UP * 1.6),
                      run_time=0.8)
            self.wait_until(v, "kernel")
            self.play(Write(bal), run_time=1)
            self.wait_until(v, "projection")
            self.play(Write(num), run_time=0.8)
            self.wait_until(v, "destroys")
            self.play(FadeIn(words), run_time=0.6)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · Linear maps in machine learning")
        f = colmath(["y", "=", "W", "x", "+", "b"], [(0, YELLOW), (2, PURPLE), (3, BLUE), (5, ORANGE)], 80)
        f.move_to(UP * 1.4)
        l1 = T("W x: a linear map", 28, PURPLE)
        l2 = T("+ b: a shift (the bias) → the layer is affine", 28, ORANGE)
        VGroup(l1, l2).arrange(DOWN, buff=0.35).next_to(f, DOWN, buff=0.6)
        src = T("PyTorch nn.Linear, written for column vectors", 20, MUTED).to_edge(DOWN, buff=0.4)
        with self.voice(N["ml1"]) as v:
            self.wait_until(v, "computes")
            self.play(Write(f), FadeIn(src), run_time=1)
            self.wait_until(v, "weight")
            self.play(Indicate(f[2], color=PURPLE), run_time=0.7)
            self.wait_until(v, "bias")
            self.play(Indicate(f[5], color=ORANGE), run_time=0.7)
            self.wait_until(v, "linear", 2)
            self.play(FadeIn(l1), run_time=0.6)
            self.wait_until(v, "affine")
            self.play(FadeIn(l2), run_time=0.6)
        self.clear_body()

        def chain(labels, y):
            items = [MathTex("x", color=BLUE, font_size=50)]
            for lab in labels:
                b = VGroup(RoundedRectangle(corner_radius=0.12, width=1.9, height=0.95, color=PURPLE),
                           MathTex(lab, color=PURPLE, font_size=40))
                b[1].move_to(b[0])
                items.append(b)
            items.append(MathTex("y", color=YELLOW, font_size=50))
            g = VGroup(*items).arrange(RIGHT, buff=0.8).move_to([-2.4, y, 0])
            arrows = VGroup(*[Arrow(g[i].get_right(), g[i + 1].get_left(), buff=0.1, color=MUTED)
                              for i in range(len(g) - 1)])
            return VGroup(g, arrows)
        c1 = chain(["W_1", "W_2"], 1.6)
        c2 = chain(["W_2W_1"], -0.6)
        WW = mat((demo.W2 @ demo.W1).tolist(), 0.8, YELLOW)
        WWg = VGroup(MathTex(r"W_2W_1=", font_size=40), WW).arrange(RIGHT, buff=0.2).move_to(RIGHT * 4.3 + DOWN * 0.6)
        toy = T("toy layers: 2 → 3 → 2", 22, MUTED).next_to(WWg, UP, buff=0.3)
        eq = colmath(["W_2(W_1x)", "=", "(W_2W_1)x"], [(2, YELLOW)], 46).move_to(RIGHT * 3.8 + UP * 1.6)
        ten = T("10 linear layers in a row = still one matrix", 28, RED).move_to(DOWN * 2.7)
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "Two")
            self.play(FadeIn(c1), run_time=1)
            self.wait_until(v, "equals")
            self.play(FadeIn(eq), FadeIn(c2), FadeIn(WWg), FadeIn(toy), run_time=1.2)
            self.wait_until(v, "ten")
            self.play(FadeIn(ten), run_time=0.6)
        self.clear_body()
        sizes = ["768", "3072", "768"]
        boxes = VGroup(*[VGroup(Rectangle(width=0.6, height=h, color=BLUE, fill_color=BLUE, fill_opacity=0.25),
                                T(s, 22, FG)) for s, h in zip(sizes, [1.2, 3.0, 1.2])])
        for b in boxes:
            b[1].next_to(b[0], DOWN, buff=0.15)
        boxes.arrange(RIGHT, buff=2.6, aligned_edge=DOWN).move_to(LEFT * 1.2 + DOWN * 0.3)
        lin = VGroup(*[Arrow(boxes[i][0].get_right(), boxes[i + 1][0].get_left(), buff=0.15, color=PURPLE)
                       for i in range(2)])
        lt = VGroup(*[T("linear", 20, PURPLE).next_to(a, UP, buff=0.1) for a in lin])
        gpt = T("GPT-2 feed-forward block (Raschka §4.3, p. 108)", 20, MUTED).to_edge(RIGHT, buff=0.4).set_y(2.4)
        col = T("without anything between: collapses to one map", 26, RED).move_to(DOWN * 2.9)
        act = VGroup(RoundedRectangle(corner_radius=0.12, width=1.9, height=0.7, color=GREEN),
                     T("activation", 22, GREEN))
        act[1].move_to(act[0])
        act.move_to(boxes[1][0].get_top() + UP * 0.55)
        v16 = T("→ video 16: activation function", 26, GREEN).move_to(RIGHT * 3.8 + UP * 1.6)
        with self.voice(N["ml3"]) as v:
            self.play(FadeIn(gpt), run_time=0.5)
            self.wait_until(v, "expands")
            self.play(FadeIn(boxes[0]), GrowArrow(lin[0]), FadeIn(lt[0]), FadeIn(boxes[1]), run_time=1)
            self.wait_until(v, "back")
            self.play(GrowArrow(lin[1]), FadeIn(lt[1]), FadeIn(boxes[2]), run_time=0.8)
            self.wait_until(v, "collapse")
            self.play(FadeIn(col), run_time=0.6)
            self.wait_until(v, "activation")
            self.play(FadeIn(act, shift=DOWN * 0.2), run_time=0.7)
            self.wait_until(v, "sixteen")
            self.play(FadeIn(v16), run_time=0.6)
        self.clear_body()
        ax_t = T("Axiom A · Linearity", 60, YELLOW, weight=BOLD).shift(UP * 0.9)
        sub = colmath(["f(au+bv)", "=", "af(u)+bf(v)"], [], 50).next_to(ax_t, DOWN, buff=0.5)
        fol = T("matrices · composition · kernel · image all follow", 28, MUTED).next_to(sub, DOWN, buff=0.45)
        with self.voice(N["axiom"]) as v:
            self.wait_until(v, "axiom")
            self.play(Write(ax_t), FadeIn(sub), run_time=1.2)
            self.wait_until(v, "Matrices")
            self.play(FadeIn(fol), run_time=0.7)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        q = MathTex(r"f\begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}2x\\x+y\end{bmatrix}", r"\quad\text{linear? matrix?}",
                    font_size=60)
        q[1].set_color(MUTED)
        q.shift(UP * 1.6)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac1"]) as v:
            self.play(Write(q), run_time=1.5)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
        a0 = T("linear ✓", 30, GREEN)
        a1 = colmath([r"f(e_1)=f\begin{bmatrix}1\\0\end{bmatrix}=\begin{bmatrix}2\\1\end{bmatrix}"], [(0, BLUE)], 40)
        a2 = colmath([r"f(e_2)=f\begin{bmatrix}0\\1\end{bmatrix}=\begin{bmatrix}0\\1\end{bmatrix}"], [(0, GREEN)], 40)
        a3 = MathTex(r"A=\begin{bmatrix}2&0\\1&1\end{bmatrix}", font_size=44, color=YELLOW)
        a4 = MathTex(r"A\begin{bmatrix}3\\1\end{bmatrix}=\begin{bmatrix}6\\4\end{bmatrix}=f\begin{bmatrix}3\\1\end{bmatrix}\ \checkmark",
                     font_size=40)
        left = VGroup(a0, a1, a2).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(LEFT * 3.2 + DOWN * 1.3)
        right = VGroup(a3, a4).arrange(DOWN, buff=0.4).move_to(RIGHT * 3.4 + DOWN * 1.3)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "linear")
            self.play(FadeIn(a0), run_time=0.5)
            self.wait_until(v, "zero")
            self.play(Write(a1), run_time=0.9)
            self.wait_until(v, "zero", 2)
            self.play(Write(a2), run_time=0.9)
            self.wait_until(v, "matrix")
            self.play(Write(a3), run_time=0.8)
            self.wait_until(v, "Check")
            self.play(Write(a4), run_time=1)
        self.clear_body()
        q2 = MathTex(r"g(x)=5x-3", r"\quad\text{linear?}", font_size=64)
        q2[1].set_color(MUTED)
        q2.shift(UP * 1.4)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q2, DOWN, buff=0.5)
        with self.voice(N["prac3"]) as v:
            self.play(Write(q2), run_time=1.2)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
        b1 = MathTex(r"g(0)=-3\neq 0", r"\ \Rightarrow\ \text{not linear}", font_size=50, color=RED)
        b2 = T("affine: the linear map 5x, then a shift of −3", 30, ORANGE)
        b3 = T("its graph is a straight line that misses the origin", 26, MUTED)
        VGroup(b1, b2, b3).arrange(DOWN, buff=0.4).move_to(DOWN * 1.2)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "origin")
            self.play(Write(b1), run_time=1)
            self.wait_until(v, "affine")
            self.play(FadeIn(b2), run_time=0.6)
            self.wait_until(v, "graph")
            self.play(FadeIn(b3), run_time=0.6)


CODE = '''import numpy as np
A = np.column_stack([[2, 1], [-1, 1]])   # f(e1), f(e2)
print("A @ [3 2] =", A @ [3, 2])
u, v = np.array([1, 2]), np.array([3, -1])
print(A @ (2*u + 3*v), 2*(A @ u) + 3*(A @ v))
f, g = (lambda t: t + 1), (lambda t: t ** 2)
print(f(2 + 3), f(2) + f(3), g(1 + 1), g(1) + g(1))
S = np.array([[1, 1], [0, 1]])           # shear
P = np.array([[1, 0], [0, 0]])           # projection
print(P @ S, P @ (S @ [1, 2]), (P @ S) @ [1, 2])
print("P @ [0 5] =", P @ [0, 5])         # kernel
W1 = np.array([[1, 2], [0, 1], [1, 0]])
W2 = np.array([[1, 0, 1], [2, 1, 0]])
print("W2 @ W1 =", W2 @ W1)              # one matrix'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=22)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[1], color=YELLOW, buff=0.06)
            self.wait_until(v, "build")
            self.play(Create(box), run_time=0.5)
            for i, w in [(4, "rule"), (6, "shift"), (9, "compose"), (13, "collapse")]:
                self.wait_until(v, w)
                self.play(Transform(box, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                              buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=24, command="python demo.py").shift(DOWN * 0.5)
        if term.width > 13.4:
            term.scale_to_fit_width(13.4)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for idx, w in [((1,), "four"), ((2,), "Both"), ((3,), "shift"), ((4,), "square"),
                           ((6,), "orders"), ((8, 9), "layers")]:
                self.wait_until(v, w)
                self.play(*[Indicate(term[1][i], color=YELLOW) for i in idx], run_time=0.7)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["Linear map: f(a·u + b·v) = a·f(u) + b·f(v)",
                            "Grid lines stay straight, parallel, evenly spaced; origin fixed",
                            "x + 1 is affine (moves the origin); x² is not linear",
                            "Every linear map is a matrix: columns = f(e₁), f(e₂)",
                            "Composition = matrix product; kernel → 0; image = all outputs"],
                           size=28).shift(UP * 0.5)
        with self.voice(N["sum1"]) as v:
            for p in pts[:3]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 5)
        nxt = VGroup(T("Next · Video 5", 26, MUTED),
                     T("Inner product: one number that measures similarity", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[3], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Composition")
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Definition", "C2_Pictures", "C3_NonExamples", "C4_Matrix", "C5_Compose",
          "C6_ML", "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Definition": "What makes a map linear",
                  "C2_Pictures": "What it does to a grid", "C3_NonExamples": "Maps that are not linear",
                  "C4_Matrix": "Every linear map is a matrix", "C5_Compose": "Composition, kernel, image",
                  "C6_ML": "Linear maps in ML", "C7_Practice": "Try it yourself",
                  "C7_Code": "Python demo", "C8_Summary": "Summary", "C9_Credits": "Credits"}
