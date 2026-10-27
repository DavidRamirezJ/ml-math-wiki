"""Vector — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 1 of 16"
TITLE = "Vector — the atom everything is built from"


def arr(plane, a, b, color, width=6):
    return Arrow(plane.c2p(*a), plane.c2p(*b), buff=0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.18)


def col(a, b, color=FG, size=48):
    return MathTex(r"\begin{bmatrix}%s\\%s\end{bmatrix}" % (a, b), color=color,
                   font_size=size)


def grid_plane(xr, yr, xl, yl):
    return NumberPlane(x_range=xr, y_range=yr, x_length=xl, y_length=yl,
                       background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.35,
                                              "stroke_width": 1},
                       axis_config={"stroke_color": MUTED})


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        if title.width > 12.5:
            title.scale_to_fit_width(12.5)
        g = VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.2)
        rows = VGroup()
        for name, nums in [("photo", "[0.12, 0.80, 0.33, 0.95, …]"),
                           ("word", "[0.41, -1.20, 0.07, …]"),
                           ("customer", "[34, 52000, 7]")]:
            a = T(name, 30, BLUE, weight=BOLD)
            ar = Arrow(LEFT * 0.6, RIGHT * 0.6, buff=0, color=MUTED)
            b = T(nums, 28, YELLOW, font=MONO)
            rows.add(VGroup(a, ar, b).arrange(RIGHT, buff=0.35))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.45).shift(DOWN * 0.9)
        toy = T("toy numbers", 18, MUTED).next_to(rows, DOWN, buff=0.3)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            for r, w in zip(rows, ["photo", "word", "customer"]):
                self.wait_until(v, w, 2 if w != "customer" else 1)
                self.play(FadeIn(r[0]), GrowArrow(r[1]), FadeIn(r[2], shift=RIGHT * 0.2),
                          run_time=0.9)
            self.play(FadeIn(toy), run_time=0.4)
            self.wait_until(v, "vector")
            self.play(*[Indicate(r[2], color=YELLOW, scale_factor=1.1) for r in rows], run_time=1.2)
        moves = VGroup(T("add", 54, GREEN, weight=BOLD), T("+", 54, MUTED),
                       T("scale", 54, PURPLE, weight=BOLD)).arrange(RIGHT, buff=0.5)
        moves.shift(DOWN * 0.8)
        with self.voice(N["intro2"]) as v:
            self.play(FadeOut(rows), FadeOut(toy), run_time=0.6)
            self.wait_until(v, "adding")
            self.play(FadeIn(moves[0], shift=UP * 0.2), run_time=0.5)
            self.wait_until(v, "scaling")
            self.play(FadeIn(moves[1]), FadeIn(moves[2], shift=UP * 0.2), run_time=0.5)
            self.wait_until(v, "built")
            self.play(Indicate(moves, color=YELLOW), run_time=1)


class C1_ThreeFaces(Studio):
    def construct(self):
        self.heading("1 · Three faces of a vector")
        plane = grid_plane([-1, 5, 1], [-1, 4, 1], 6, 5).shift(LEFT * 3.3 + DOWN * 0.6)
        o = Dot(plane.c2p(0, 0), color=FG)
        olab = T("origin", 20, MUTED).next_to(o, DL, buff=0.1)
        a = arr(plane, (0, 0), (3, 2), BLUE)
        tip = T("(3, 2)", 28, BLUE).next_to(plane.c2p(3, 2), UR, buff=0.1)
        with self.voice(N["faces1"]) as v:
            self.play(Create(plane), run_time=1.2)
            self.wait_until(v, "origin")
            self.play(FadeIn(o), FadeIn(olab), run_time=0.5)
            self.wait_until(v, "right")
            self.play(GrowArrow(a), run_time=1.5)
            self.play(FadeIn(tip), run_time=0.5)
        hx = DashedLine(plane.c2p(0, 0), plane.c2p(3, 0), color=ORANGE, stroke_width=4)
        hy = DashedLine(plane.c2p(3, 0), plane.c2p(3, 2), color=ORANGE, stroke_width=4)
        lx = T("3 right", 22, ORANGE).next_to(hx, DOWN, buff=0.12)
        ly = T("2 up", 22, ORANGE).next_to(hy, RIGHT, buff=0.12)
        c = col(3, 2, BLUE, 72).move_to(RIGHT * 2.2 + UP * 1.3)
        card = self.term_card("component", "One entry of a vector: the number in one "
                              "position of the list.", pos=DR, width=5.4)
        with self.voice(N["faces2"]) as v:
            self.wait_until(v, "list")
            self.play(TransformFromCopy(tip, c), run_time=1)
            self.wait_until(v, "component")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "right")
            self.play(Create(hx), FadeIn(lx), Indicate(c[0][1], color=ORANGE), run_time=0.8)
            self.wait_until(v, "up")
            self.play(Create(hy), FadeIn(ly), Indicate(c[0][2], color=ORANGE), run_time=0.8)
        self.clear_body()
        feats = VGroup(*[T(s, 30, FG) for s in ["area: 120 m²", "bedrooms: 3",
                                                 "age: 15 years"]])
        feats.arrange(DOWN, aligned_edge=LEFT, buff=0.4).shift(LEFT * 3.3)
        house = T("one house", 30, BLUE, weight=BOLD).next_to(feats, UP, buff=0.5)
        vec3 = MathTex(r"\begin{bmatrix}120\\3\\15\end{bmatrix}", color=YELLOW,
                       font_size=80).shift(RIGHT * 2.5)
        ar = Arrow(feats.get_right() + RIGHT * 0.3, vec3.get_left() + LEFT * 0.3,
                   color=MUTED, buff=0)
        dim = T("3 components → 3 dimensions", 24, MUTED).next_to(vec3, DOWN, buff=0.5)
        with self.voice(N["faces3"]) as v:
            self.play(FadeIn(house), run_time=0.5)
            self.play(LaggedStart(*[FadeIn(f, shift=RIGHT * 0.2) for f in feats],
                                  lag_ratio=0.4), run_time=2.5)
            self.wait_until(v, "vector")
            self.play(GrowArrow(ar), Write(vec3), run_time=1.2)
            self.play(FadeIn(dim), run_time=0.6)
        self.clear_body()
        icons = VGroup()
        ax = Axes(x_range=[0, 6.3, 1], y_range=[-1.2, 1.2, 1], x_length=2.4, y_length=1.2,
                  tips=False, axis_config={"stroke_color": MUTED, "include_ticks": False})
        wave = VGroup(ax, ax.plot(lambda x: np.sin(x), color=GREEN))
        mini = grid_plane([-1, 3, 1], [-1, 2, 1], 2.4, 1.8)
        miniv = VGroup(mini, arr(mini, (0, 0), (2, 1), BLUE, 5))
        lst = T("[120, 3, 15]", 30, YELLOW, font=MONO)
        for obj, name in [(miniv, "arrow"), (lst, "list"), (wave, "sound wave")]:
            box = VGroup(obj, T(name, 24, FG), T("add ✓   scale ✓", 22, GREEN))
            box.arrange(DOWN, buff=0.3)
            icons.add(box)
        icons.arrange(RIGHT, buff=1.2).shift(UP * 1.3)
        for b in icons:
            b[0].move_to(b[1].get_center() + UP * 1.1)
        card = self.term_card("vector", "Anything you can add to another of its kind and "
                              "scale by a number, and still get the same kind of thing.",
                              pos=DOWN, width=7.5)
        with self.voice(N["faces4"]) as v:
            self.wait_until(v, "anything")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.7)
            for b, w in zip(icons, ["Arrows", "Lists", "sound"]):
                self.wait_until(v, w)
                self.play(FadeIn(b, shift=UP * 0.2), run_time=0.7)
            self.wait_until(v, "add", 2)
            self.play(*[Indicate(b[2], color=YELLOW) for b in icons], run_time=1)


class C2_Add(Studio):
    def construct(self):
        self.heading("2 · Adding vectors")
        plane = grid_plane([-1, 6, 1], [-1, 4, 1], 6.3, 4.5).shift(LEFT * 3.2 + DOWN * 0.8)
        u = arr(plane, (0, 0), (3, 1), BLUE)
        v_ = arr(plane, (0, 0), (1, 2), GREEN)
        ul = MathTex(r"u", color=BLUE, font_size=40).next_to(u.get_center(), DOWN, buff=0.2)
        vl = MathTex(r"v", color=GREEN, font_size=40).next_to(v_.get_center(), LEFT, buff=0.2)
        uc = VGroup(MathTex("u=", color=BLUE), col(3, 1, BLUE)).arrange(RIGHT)
        vc = VGroup(MathTex("v=", color=GREEN), col(1, 2, GREEN)).arrange(RIGHT)
        VGroup(uc, vc).arrange(RIGHT, buff=0.8).move_to(RIGHT * 3.6 + UP * 1.9)
        with self.voice(N["add1"]) as v:
            self.play(Create(plane), run_time=1)
            self.wait_until(v, "blue")
            self.play(GrowArrow(u), FadeIn(ul), FadeIn(uc), run_time=1)
            self.wait_until(v, "green")
            self.play(GrowArrow(v_), FadeIn(vl), FadeIn(vc), run_time=1)
            self.wait_until(v, "walk")
            self.play(Indicate(u, color=BLUE), run_time=0.8)
            self.wait_until(v, "along", 2)
            self.play(Indicate(v_, color=GREEN), run_time=0.8)
        v_moved = arr(plane, (3, 1), (4, 3), GREEN)
        s = arr(plane, (0, 0), (4, 3), YELLOW, 7)
        sl = MathTex(r"u+v", color=YELLOW, font_size=40).next_to(plane.c2p(4, 3), UP, buff=0.15)
        with self.voice(N["add2"]) as v:
            self.wait_until(v, "tail")
            self.play(TransformFromCopy(v_, v_moved), run_time=1.2)
            self.wait_until(v, "yellow")
            self.play(GrowArrow(s), FadeIn(sl), run_time=1.2)
            self.wait_until(v, "four", 1)
            self.play(Indicate(sl, color=YELLOW), run_time=0.8)
        eq = MathTex(r"\begin{bmatrix}3\\1\end{bmatrix}", "+",
                     r"\begin{bmatrix}1\\2\end{bmatrix}", "=",
                     r"\begin{bmatrix}3+1\\1+2\end{bmatrix}", "=",
                     r"\begin{bmatrix}4\\3\end{bmatrix}", font_size=46)
        eq[0].set_color(BLUE); eq[2].set_color(GREEN); eq[6].set_color(YELLOW)
        eq.move_to(RIGHT * 3.6 + DOWN * 0.1)
        if eq.width > 6.4:
            eq.scale_to_fit_width(6.4).move_to(RIGHT * 3.6 + DOWN * 0.1)
        rule = T("add first with first, second with second", 22, MUTED).next_to(eq, DOWN, buff=0.4)
        with self.voice(N["add3"]) as v:
            self.play(Write(eq[:4]), run_time=1.2)
            self.wait_until(v, "component")
            self.play(Write(eq[4]), run_time=1)
            self.play(Write(eq[5:]), run_time=0.8)
            self.wait_until(v, "first", 1)
            self.play(FadeIn(rule), run_time=0.6)
        u_moved = arr(plane, (1, 2), (4, 3), BLUE)
        comm = MathTex(r"u+v", "=", r"v+u", font_size=54).next_to(rule, DOWN, buff=0.6)
        comm[0].set_color(YELLOW); comm[2].set_color(YELLOW)
        with self.voice(N["add4"]) as v:
            self.wait_until(v, "first")
            self.play(TransformFromCopy(u, u_moved), run_time=1.2)
            self.wait_until(v, "parallelogram")
            self.play(Indicate(VGroup(u, v_, u_moved, v_moved), color=YELLOW), run_time=1)
            self.wait_until(v, "equals")
            self.play(Write(comm), run_time=1)


class C3_Scale(Studio):
    def construct(self):
        self.heading("3 · Scaling a vector")
        plane = grid_plane([-5, 5, 1], [-3, 3, 1], 7.4, 4.6).shift(LEFT * 2.8 + DOWN * 0.6)
        c = ValueTracker(1.0)
        base = arr(plane, (0, 0), (2, 1), BLUE, 4).set_opacity(0.45)
        ulab = MathTex("u", color=BLUE, font_size=36).next_to(plane.c2p(2, 1), UP, buff=0.1)
        live = always_redraw(lambda: arr(plane, (0, 0), (2 * c.get_value(), c.get_value()),
                                         YELLOW, 7) if abs(c.get_value()) > 0.06 else
                             Dot(plane.c2p(0, 0), color=YELLOW, radius=0.1))
        read = always_redraw(lambda: VGroup(
            T(f"c = {c.get_value():+.1f}", 34, PURPLE, font=MONO),
            T(f"c·u = ({2 * c.get_value():.1f}, {c.get_value():.1f})", 30, YELLOW, font=MONO)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 4.3 + UP * 1.6))
        card = self.term_card("scalar", "An ordinary single number, used to stretch, "
                              "shrink or flip a vector.", pos=DR, width=5.0)
        with self.voice(N["scale1"]) as v:
            self.play(Create(plane), run_time=1)
            self.play(GrowArrow(base), FadeIn(ulab), run_time=0.8)
            self.add(live, read)
            self.wait_until(v, "scalar")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "Two", 2)
            self.play(c.animate.set_value(2), run_time=1.5)
        with self.voice(N["scale2"]) as v:
            self.play(FadeOut(card), run_time=0.5)
            self.wait_until(v, "stretches")
            self.play(c.animate.set_value(2.4), run_time=1)
            self.wait_until(v, "shrinks")
            self.play(c.animate.set_value(0.5), run_time=1.2)
            self.wait_until(v, "zero", 2)
            self.play(c.animate.set_value(0), run_time=1)
            self.wait_until(v, "negative")
            self.play(c.animate.set_value(-1.5), run_time=1.5)
        line = DashedLine(plane.c2p(-5, -2.5), plane.c2p(5, 2.5), color=PURPLE,
                          stroke_width=3)
        ex = VGroup(MathTex(r"-1\cdot", r"\begin{bmatrix}2\\1\end{bmatrix}", "=",
                            r"\begin{bmatrix}-2\\-1\end{bmatrix}", font_size=40),
                    MathTex(r"\tfrac12\cdot", r"\begin{bmatrix}2\\1\end{bmatrix}", "=",
                            r"\begin{bmatrix}1\\ \tfrac12\end{bmatrix}", font_size=40))
        for e in ex:
            e[0].set_color(PURPLE); e[1].set_color(BLUE); e[3].set_color(YELLOW)
        ex.arrange(DOWN, buff=0.4).move_to(RIGHT * 4.4 + DOWN * 1.3)
        with self.voice(N["scale3"]) as v:
            self.wait_until(v, "Minus")
            self.play(c.animate.set_value(-1), FadeIn(ex[0]), run_time=1)
            self.wait_until(v, "half", 1)
            self.play(c.animate.set_value(0.5), FadeIn(ex[1]), run_time=1)
            self.wait_until(v, "line")
            self.play(Create(line), run_time=1.2)


class C4_Combine(Studio):
    def construct(self):
        self.heading("4 · Linear combinations")
        plane = grid_plane([-1, 6, 1], [-1, 5, 1], 5.6, 4.8).shift(LEFT * 3.5 + DOWN * 0.7)
        u = arr(plane, (0, 0), (2, 1), BLUE)
        v_ = arr(plane, (0, 0), (1, 2), GREEN)
        f = MathTex("a", r"\,u", "+", "b", r"\,v", font_size=80).move_to(RIGHT * 3.3 + UP * 1.6)
        f[0].set_color(PURPLE); f[1].set_color(BLUE); f[3].set_color(PURPLE); f[4].set_color(GREEN)
        card = self.term_card("linear combination", "Scale some vectors, then add them: "
                              "a·u + b·v.", pos=DR, width=5.4)
        with self.voice(N["comb1"]) as v:
            self.play(Create(plane), GrowArrow(u), GrowArrow(v_), run_time=1.2)
            self.wait_until(v, "linear")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "times")
            self.play(Write(f), run_time=1.2)
            self.wait_until(v, "scalars")
            self.play(Indicate(f[0], color=PURPLE), Indicate(f[3], color=PURPLE), run_time=0.8)
        target = Dot(plane.c2p(5, 4), color=YELLOW, radius=0.11)
        tl = T("(5, 4)?", 26, YELLOW).next_to(target, UP, buff=0.12)
        two_u = arr(plane, (0, 0), (4, 2), BLUE)
        v_m = arr(plane, (4, 2), (5, 4), GREEN)
        res = arr(plane, (0, 0), (5, 4), YELLOW, 7)
        work = MathTex(r"2", r"\begin{bmatrix}2\\1\end{bmatrix}", "+", r"1",
                       r"\begin{bmatrix}1\\2\end{bmatrix}", "=",
                       r"\begin{bmatrix}4+1\\2+2\end{bmatrix}", "=",
                       r"\begin{bmatrix}%d\\%d\end{bmatrix}" % tuple(demo.comb), font_size=40)
        work[0].set_color(PURPLE); work[1].set_color(BLUE); work[3].set_color(PURPLE)
        work[4].set_color(GREEN); work[8].set_color(YELLOW)
        work.scale_to_fit_width(min(work.width, 6.4)).move_to(RIGHT * 3.4 + DOWN * 0.4)
        with self.voice(N["comb2"]) as v:
            self.play(FadeOut(card), run_time=0.4)
            self.wait_until(v, "yellow")
            self.play(FadeIn(target), FadeIn(tl), run_time=0.6)
            self.wait_until(v, "Two", 4)
            self.play(Transform(u.copy(), two_u), run_time=1)
            self.add(two_u)
            self.wait_until(v, "Add")
            self.play(TransformFromCopy(v_, v_m), run_time=1)
            self.play(Write(work), run_time=1.5)
            self.wait_until(v, "Yes")
            self.play(GrowArrow(res), run_time=0.8)
        dots = VGroup()
        for a in np.arange(-2, 4.01, 0.5):
            for b in np.arange(-2, 4.01, 0.5):
                x, y = 2 * a + b, a + 2 * b
                if -1 <= x <= 6 and -1 <= y <= 5:
                    dots.add(Dot(plane.c2p(x, y), radius=0.045, color=YELLOW))
        card2 = self.term_card("span", "All the vectors you can reach with linear "
                               "combinations of a set of vectors.", pos=DR, width=5.4)
        with self.voice(N["comb3"]) as v:
            self.play(FadeOut(work), run_time=0.4)
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.02),
                      run_time=v.duration * 0.45)
            self.wait_until(v, "span")
            self.play(FadeIn(card2, shift=UP * 0.3), run_time=0.6)
        self.clear_body()
        plane2 = grid_plane([-5, 5, 1], [-3, 3, 1], 7.4, 4.6).shift(LEFT * 2.8 + DOWN * 0.6)
        a1 = arr(plane2, (0, 0), (2, 1), BLUE)
        a2 = arr(plane2, (0, 0), (4, 2), RED, 4)
        ln = Line(plane2.c2p(-5, -2.5), plane2.c2p(5, 2.5), color=YELLOW, stroke_width=3)
        note = VGroup(MathTex(r"\begin{bmatrix}4\\2\end{bmatrix}=2\begin{bmatrix}2\\1\end{bmatrix}",
                              font_size=44),
                      T("span = only a line", 28, YELLOW),
                      T("→ video 3: basis & rank", 24, MUTED)).arrange(DOWN, buff=0.4)
        note.move_to(RIGHT * 4.4 + UP * 0.6)
        with self.voice(N["comb4"]) as v:
            self.play(Create(plane2), run_time=0.8)
            self.wait_until(v, "two", 1)
            self.play(GrowArrow(a1), run_time=0.6)
            self.wait_until(v, "four")
            self.play(GrowArrow(a2), FadeIn(note[0]), run_time=0.8)
            self.wait_until(v, "line", 2)
            self.play(Create(ln), FadeIn(note[1]), run_time=1)
            self.wait_until(v, "basis")
            self.play(FadeIn(note[2]), run_time=0.5)


class C5_Rules(Studio):
    def construct(self):
        self.heading("5 · What a vector really is")
        V = Ellipse(width=4.6, height=3.4, color=PURPLE, stroke_width=3).shift(LEFT * 4.3 + DOWN * 0.5)
        Vl = MathTex("V", color=PURPLE, font_size=50).next_to(V, UL, buff=-0.1)
        cl = VGroup(MathTex(r"u+v\in V", font_size=52), MathTex(r"c\,u\in V", font_size=52))
        cl.arrange(DOWN, buff=0.5).move_to(V.get_center())
        cl.scale(0.85)
        mem = T("∈  means  “is a member of”", 24, MUTED).next_to(V, DOWN, buff=0.25)
        card = self.term_card("vector space", "A collection where adding and scaling never "
                              "take you outside it, and the usual rules hold.", pos=UR, width=5.6)
        with self.voice(N["rules1"]) as v:
            self.wait_until(v, "space")
            self.play(Create(V), FadeIn(Vl), FadeIn(card, shift=DOWN * 0.3), run_time=1.2)
            self.wait_until(v, "adding")
            self.play(Write(cl[0]), FadeIn(mem), run_time=1)
            self.wait_until(v, "scaling")
            self.play(Write(cl[1]), run_time=1)
            self.wait_until(v, "fall")
            self.play(Indicate(V, color=PURPLE), run_time=1)
        rules = VGroup(MathTex(r"u+v=v+u", font_size=44), MathTex(r"u+0=u", font_size=44),
                       MathTex(r"u+(-u)=0", font_size=44),
                       MathTex(r"a(u+v)=au+av", font_size=44))
        names = ["order does not matter", "zero changes nothing", "every vector has an opposite",
                 "scaling spreads over a sum"]
        rows = VGroup(*[VGroup(r, T(n, 22, MUTED)).arrange(RIGHT, buff=0.4)
                        for r, n in zip(rules, names)])
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        rows.scale_to_fit_width(min(rows.width, 7.6)).move_to(RIGHT * 2.9 + DOWN * 0.5)
        with self.voice(N["rules2"]) as v:
            self.play(FadeOut(card), run_time=0.4)
            for r, w in zip(rows, ["Order", "zero", "opposite", "spreads"]):
                self.wait_until(v, w)
                self.play(FadeIn(r, shift=LEFT * 0.2), run_time=0.7)
        self.clear_body()
        ax = Axes(x_range=[0, 6.3, 1], y_range=[-2.4, 2.4, 1], x_length=10, y_length=4.4,
                  tips=False, axis_config={"stroke_color": MUTED, "include_ticks": False}
                  ).shift(DOWN * 0.6)
        f = ax.plot(lambda x: np.sin(x), color=BLUE)
        g = ax.plot(lambda x: 0.5 * np.sin(3 * x), color=GREEN)
        s = ax.plot(lambda x: np.sin(x) + 0.5 * np.sin(3 * x), color=YELLOW, stroke_width=5)
        f2 = ax.plot(lambda x: 2 * np.sin(x), color=PURPLE)
        leg = VGroup(T("f = sound 1", 24, BLUE), T("g = sound 2", 24, GREEN),
                     T("f + g", 24, YELLOW), T("2f (louder)", 24, PURPLE)
                     ).arrange(RIGHT, buff=0.6).next_to(ax, UP, buff=0.25)
        with self.voice(N["rules3"]) as v:
            self.play(Create(ax), run_time=0.6)
            self.wait_until(v, "sound")
            self.play(Create(f), FadeIn(leg[0]), Create(g), FadeIn(leg[1]), run_time=1.5)
            self.wait_until(v, "Add")
            self.play(Create(s), FadeIn(leg[2]), run_time=1.5)
            self.wait_until(v, "louder")
            self.play(FadeOut(s), Create(f2), FadeIn(leg[3]), run_time=1.5)
            self.wait_until(v, "vectors")
            self.play(Indicate(leg, color=YELLOW), run_time=0.8)
        self.clear_body()
        ax_t = T("Axiom A · Linearity", 60, YELLOW, weight=BOLD).shift(UP * 0.6)
        sub = MathTex(r"\text{add } u+v \quad\text{and}\quad \text{scale } c\,u",
                      font_size=50).next_to(ax_t, DOWN, buff=0.5)
        uses = T("arrows · data lists · sound · images", 28, MUTED).next_to(sub, DOWN, buff=0.5)
        with self.voice(N["rules4"]) as v:
            self.wait_until(v, "arrows")
            self.play(FadeIn(uses), run_time=0.8)
            self.wait_until(v, "axiom")
            self.play(Write(ax_t), run_time=1)
            self.play(FadeIn(sub), run_time=0.8)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · Vectors in machine learning")

        def image(px, label, color):
            sq = VGroup(*[Square(0.9, stroke_color=MUTED, stroke_width=2, fill_color=WHITE,
                                 fill_opacity=float(p)) for p in px])
            sq.arrange_in_grid(2, 2, buff=0)
            vec = T("[" + ", ".join(f"{p:g}" for p in px) + "]", 22, color, font=MONO)
            lab = T(label, 24, color, weight=BOLD)
            return VGroup(lab, sq, vec).arrange(DOWN, buff=0.3)

        i1 = image(demo.img1, "image 1", BLUE)
        i2 = image(demo.img2, "image 2", GREEN)
        i3 = image(demo.blend, "½·image 1 + ½·image 2", YELLOW)
        VGroup(i1, i2, i3).arrange(RIGHT, buff=1.4).shift(DOWN * 0.3)
        toy = T("toy 2×2 images · brightness 0 = black, 1 = white", 20, MUTED).to_edge(DOWN, buff=0.4)
        with self.voice(N["ml1"]) as v:
            self.play(FadeIn(i1[:2]), FadeIn(i2[:2]), FadeIn(toy), run_time=1)
            self.wait_until(v, "row")
            self.play(FadeIn(i1[2], shift=DOWN * 0.2), FadeIn(i2[2], shift=DOWN * 0.2), run_time=1)
        with self.voice(N["ml2"]) as v:
            self.play(FadeIn(i3[0]), run_time=0.8)
            self.wait_until(v, "average")
            self.play(TransformFromCopy(VGroup(i1[1], i2[1]), i3[1]), run_time=1.5)
            self.play(FadeIn(i3[2]), run_time=0.6)
        self.clear_body()
        xs = VGroup(*[MathTex(f"x_{i}", color=BLUE, font_size=44) for i in (1, 2, 3)])
        xs.arrange(DOWN, buff=1.0).shift(LEFT * 5.6 + UP * 0.1)
        node = Circle(0.6, color=YELLOW).shift(LEFT * 1.6 + UP * 0.1)
        sig = MathTex(r"\Sigma", color=YELLOW, font_size=56).move_to(node)
        arrows = VGroup(*[Arrow(x.get_right(), node.get_left(), buff=0.15, color=MUTED) for x in xs])
        ws = VGroup(*[MathTex(f"w_{i}", color=PURPLE, font_size=36).move_to(a.get_center() + UP * 0.32)
                      for i, a in zip((1, 2, 3), arrows)])
        out = Arrow(node.get_right(), node.get_right() + RIGHT * 1.2, buff=0.1, color=YELLOW)
        f = MathTex("w_1", "x_1", "+", "w_2", "x_2", "+", "w_3", "x_3", font_size=48)
        for i in (0, 3, 6):
            f[i].set_color(PURPLE); f[i + 1].set_color(BLUE)
        f.next_to(out, RIGHT, buff=0.3)
        if f.get_right()[0] > 6.9:
            f.scale_to_fit_width(6.9 - f.get_left()[0])
        lab = T("a linear combination of the inputs", 24, MUTED).to_edge(DOWN, buff=0.7)
        with self.voice(N["ml3"]) as v:
            self.wait_until(v, "input")
            self.play(FadeIn(xs), run_time=0.8)
            self.wait_until(v, "weight")
            self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2),
                      FadeIn(ws), run_time=1.2)
            self.wait_until(v, "adds")
            self.play(Create(node), Write(sig), GrowArrow(out), Write(f), run_time=1.5)
            self.wait_until(v, "linear")
            self.play(FadeIn(lab), run_time=0.6)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        q = MathTex("3", r"\begin{bmatrix}1\\-2\end{bmatrix}", "+",
                    r"\begin{bmatrix}2\\5\end{bmatrix}", "=", r"\;?", font_size=72)
        q[0].set_color(PURPLE); q[1].set_color(BLUE); q[3].set_color(GREEN); q[5].set_color(YELLOW)
        q.shift(UP * 1.2)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.6)
        with self.voice(N["prac1"]) as v:
            self.play(Write(q), run_time=1.5)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
            self.wait_until(v, "scale")
            self.play(Indicate(q[0], color=PURPLE), run_time=0.8)
            self.wait_until(v, "add")
            self.play(Indicate(q[2], color=YELLOW), run_time=0.8)
        s1 = MathTex(r"=\begin{bmatrix}3\\-6\end{bmatrix}", "+",
                     r"\begin{bmatrix}2\\5\end{bmatrix}", "=",
                     r"\begin{bmatrix}3+2\\-6+5\end{bmatrix}", "=",
                     r"\begin{bmatrix}5\\-1\end{bmatrix}", font_size=56)
        s1[2].set_color(GREEN); s1[6].set_color(YELLOW)
        s1.next_to(q, DOWN, buff=0.6)
        if s1.width > 13:
            s1.scale_to_fit_width(13)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "Three", 1)
            self.play(Write(s1[0]), run_time=1)
            self.wait_until(v, "add")
            self.play(Write(s1[1:5]), run_time=1.5)
            self.wait_until(v, "answer", 2)
            self.play(Write(s1[5:]), run_time=1)
            self.play(Circumscribe(s1[6], color=YELLOW), run_time=1)
        self.clear_body()
        plane = grid_plane([-1, 7, 1], [-1, 4, 1], 7.2, 4.5).shift(LEFT * 2.6 + DOWN * 0.7)
        u = arr(plane, (0, 0), (2, 1), BLUE)
        ln = Line(plane.c2p(-1, -0.5), plane.c2p(7, 3.5), color=PURPLE, stroke_width=3)
        p1 = Dot(plane.c2p(6, 3), color=GREEN, radius=0.1)
        p1l = T("(6, 3)", 24, GREEN).next_to(p1, UP, buff=0.12)
        p2 = Dot(plane.c2p(1, 1), color=RED, radius=0.1)
        p2l = T("(1, 1)", 24, RED).next_to(p2, UL, buff=0.1)
        ask = MathTex(r"c\begin{bmatrix}2\\1\end{bmatrix}", font_size=56).move_to(RIGHT * 4.6 + UP * 1.6)
        with self.voice(N["prac3"]) as v:
            self.play(Create(plane), GrowArrow(u), run_time=1)
            self.wait_until(v, "six")
            self.play(FadeIn(p1), FadeIn(p1l), run_time=0.6)
            self.wait_until(v, "one", 3)
            self.play(FadeIn(p2), FadeIn(p2l), run_time=0.6)
            self.wait_until(v, "times")
            self.play(Write(ask), run_time=0.8)
        a1 = MathTex(r"3\begin{bmatrix}2\\1\end{bmatrix}=\begin{bmatrix}6\\3\end{bmatrix}\ \checkmark",
                     color=GREEN, font_size=44)
        a2 = MathTex(r"c=\tfrac12 \Rightarrow \begin{bmatrix}1\\ \tfrac12\end{bmatrix}\neq\begin{bmatrix}1\\1\end{bmatrix}",
                     color=RED, font_size=44)
        VGroup(a1, a2).arrange(DOWN, buff=0.5, aligned_edge=LEFT).next_to(ask, DOWN, buff=0.5)
        grp = VGroup(ask, a1, a2)
        if grp.get_right()[0] > 6.9:
            grp.shift(LEFT * (grp.get_right()[0] - 6.9))
        with self.voice(N["prac4"]) as v:
            self.wait_until(v, "three", 2)
            self.play(Create(ln), Write(a1), run_time=1.2)
            self.wait_until(v, "half", 1)
            self.play(Write(a2), run_time=1.2)
            self.wait_until(v, "outside")
            self.play(Indicate(p2, color=RED), Indicate(ln, color=PURPLE), run_time=1)


CODE = '''import numpy as np
u, v = np.array([3, 1]), np.array([1, 2])
print("u + v =", u + v)              # component by component
u2, v2 = np.array([2, 1]), np.array([1, 2])
print("2 * u =", 2 * u2)
print("2*u + 1*v =", 2 * u2 + 1 * v2)   # linear combination
# which a, b give [5 4]?  solve  a*u2 + b*v2 = [5 4]
ab = np.linalg.solve(np.column_stack([u2, v2]), [5, 4])
img1 = np.array([1.0, 0.0, 0.0, 1.0])   # toy 2x2 image
img2 = np.array([0.0, 1.0, 0.5, 0.5])
print("blend =", 0.5 * img1 + 0.5 * img2)'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=24)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[2], color=YELLOW, buff=0.06)
            self.wait_until(v, "add")
            self.play(Create(box), run_time=0.5)
            for i, w in [(5, "combination"), (7, "find"), (10, "blend")]:
                self.wait_until(v, w)
                self.play(Transform(box, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                              buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=24, command="python demo.py").shift(DOWN * 0.3)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for i, w in [(0, "three"), (1, "Two"), (4, "found"), (7, "blended")]:
                self.wait_until(v, w)
                self.play(Indicate(term[1][i], color=YELLOW), run_time=0.7)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["A vector = anything you can add and scale",
                            "In practice: a list of numbers (an arrow in 2-D)",
                            "Add and scale component by component",
                            "Scale, then add → linear combination; all reachable → span",
                            "Data points are vectors; a neuron computes a linear combination"],
                           size=28).shift(UP * 0.4)
        with self.voice(N["sum1"]) as v:
            for p in pts[:4]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 6)
        nxt = VGroup(T("Next · Video 2", 26, MUTED),
                     T("Matrix: a grid of numbers that moves vectors", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.6)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_ThreeFaces", "C2_Add", "C3_Scale", "C4_Combine", "C5_Rules",
          "C6_ML", "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_ThreeFaces": "Three faces of a vector",
                  "C2_Add": "Adding vectors", "C3_Scale": "Scaling", "C4_Combine":
                  "Linear combinations & span", "C5_Rules": "What a vector really is",
                  "C6_ML": "Vectors in ML", "C7_Practice": "Try it yourself", "C7_Code": "Python demo", "C8_Summary": "Summary", "C9_Credits": "Credits"}
