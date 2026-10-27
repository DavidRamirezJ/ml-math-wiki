"""Inner product — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 5 of 16"
TITLE = "Inner product — one number that measures similarity"


def arr(plane, a, b, color, width=6):
    return Arrow(plane.c2p(*a), plane.c2p(*b), buff=0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.18)


def grid_plane(xr, yr, xl, yl):
    return NumberPlane(x_range=xr, y_range=yr, x_length=xl, y_length=yl,
                       background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.35,
                                              "stroke_width": 1},
                       axis_config={"stroke_color": MUTED})


def colmath(parts, colours, size=46):
    m = MathTex(*parts, font_size=size)
    for i, c in colours:
        m[i].set_color(c)
    return m


def colvec(vals, color, size=48):
    return MathTex(r"\begin{bmatrix}%s\end{bmatrix}" % r"\\".join(str(v) for v in vals),
                   color=color, font_size=size)


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        if title.width > 12.5:
            title.scale_to_fit_width(12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.4)
        pl = grid_plane([-1, 3, 1], [-0.5, 2.5, 1], 4, 3).move_to(LEFT * 3.2 + DOWN * 1.1)
        ws = [("cat", (2.2, 1.6), BLUE), ("dog", (1.8, 2.0), GREEN), ("car", (-0.6, 2.2), ORANGE)]
        arrows = [arr(pl, (0, 0), p, c) for _, p, c in ws]
        labs = [T(n, 24, c).next_to(pl.c2p(*p), UP, buff=0.08) for n, p, c in ws]
        toy = T("toy picture", 18, MUTED).next_to(pl, DOWN, buff=0.15)
        q = MathTex(r"\text{cat}\cdot\text{dog} = \;?", font_size=48, color=YELLOW).move_to(RIGHT * 3.0 + DOWN * 0.4)
        name = T("inner product", 48, YELLOW, weight=BOLD).next_to(q, DOWN, buff=0.6)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            self.play(Create(pl), FadeIn(toy), run_time=0.6)
            for a_, l_, w in zip(arrows, labs, ["cat", "dog", "car"]):
                self.wait_until(v, w)
                self.play(GrowArrow(a_), FadeIn(l_), run_time=0.5)
            self.wait_until(v, "number")
            self.play(Write(q), run_time=0.8)
            self.wait_until(v, "inner")
            self.play(Write(name), run_time=0.8)
        plan = self.bullets(["compute the dot product by hand", "see it as a picture",
                             "use its sign to find right angles", "measure similarity with the cosine"],
                            size=28).move_to(DOWN * 0.9)
        pre = T("You need: video 1 · Vector, video 2 · Matrix (transpose)", 24, MUTED).to_edge(DOWN, buff=0.4)
        with self.voice(N["intro2"]) as v:
            self.play(FadeOut(VGroup(pl, toy, q, name, *arrows, *labs)), run_time=0.5)
            for p, w in zip(plan, ["dot", "picture", "sign", "cosine"]):
                self.wait_until(v, w)
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
            self.wait_until(v, "need")
            self.play(FadeIn(pre), run_time=0.6)


class C1_Dot(Studio):
    def construct(self):
        self.heading("1 · The dot product")
        xs = VGroup(*[MathTex(f"x_{i}", color=BLUE, font_size=50) for i in (1, 2, 3)]).arrange(DOWN, buff=0.45)
        ys = VGroup(*[MathTex(f"y_{i}", color=GREEN, font_size=50) for i in (1, 2, 3)]).arrange(DOWN, buff=0.45)
        xb = VGroup(xs, MathTex(r"x", color=BLUE, font_size=44).next_to(xs, UP, buff=0.3))
        yb = VGroup(ys, MathTex(r"y", color=GREEN, font_size=44).next_to(ys, UP, buff=0.3))
        VGroup(xb, yb).arrange(RIGHT, buff=1.6).move_to(LEFT * 3.8 + DOWN * 0.2)
        links = VGroup(*[DashedLine(a.get_right(), b.get_left(), color=ORANGE, buff=0.15)
                         for a, b in zip(xs, ys)])
        s = colmath(["x_1y_1", "+", "x_2y_2", "+", "x_3y_3"], [(0, YELLOW), (2, YELLOW), (4, YELLOW)], 54)
        s.move_to(RIGHT * 2.6 + UP * 0.6)
        one = T("one number, not a vector", 26, MUTED).next_to(s, DOWN, buff=0.35)
        card = self.term_card("dot product", "Multiply matching components and add: "
                              "the result is one number.", pos=DR, width=5.2)
        with self.voice(N["dot1"]) as v:
            self.wait_until(v, "Take")
            self.play(FadeIn(xb), FadeIn(yb), run_time=0.8)
            self.wait_until(v, "Multiply")
            self.play(Create(links), run_time=1)
            self.wait_until(v, "add")
            self.play(Write(s), run_time=1.2)
            self.wait_until(v, "one")
            self.play(FadeIn(one), run_time=0.5)
            self.wait_until(v, "dot")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
        self.clear_body()
        f = colmath(["x", r"\cdot", "y", "=", r"\sum_{i=1}^{n}", "x_i", "\\,y_i", r"\;=\;", r"x^{\top}y"],
                    [(0, BLUE), (2, GREEN), (4, ORANGE), (5, BLUE), (6, GREEN), (8, YELLOW)], 64)
        f.move_to(UP * 1.6)
        leg = VGroup(*[VGroup(MathTex(sy, color=co, font_size=38), T(tx, 24, MUTED)).arrange(RIGHT, buff=0.3)
                       for sy, co, tx in [(r"\sum_{i=1}^{n}", ORANGE, "add up for i = 1, 2, …, n (each component)"),
                                          ("x_i\\,y_i", YELLOW, "the i-th component of x times the i-th of y"),
                                          (r"x^{\top}y", YELLOW, "row × column = a 1×1 matrix: one number")]])
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_edge(LEFT, buff=0.8).shift(DOWN * 1.4)
        rem = self.reminder_card("transpose", "Flip a matrix over its diagonal: rows become "
                                 "columns (video 2).", pos=DR, width=4.4)
        with self.voice(N["dot2"]) as v:
            self.wait_until(v, "symbols")
            self.play(Write(f[:7]), run_time=1.2)
            self.wait_until(v, "letter")
            self.play(FadeIn(leg[0]), FadeIn(leg[1]), run_time=0.8)
            self.wait_until(v, "transpose")
            self.play(Write(f[7:]), FadeIn(leg[2]), FadeIn(rem, shift=UP * 0.3), run_time=1)
        self.clear_body()
        X = colvec(demo.x, BLUE, 56)
        Y = colvec(demo.y, GREEN, 56)
        top = VGroup(MathTex("x=", font_size=50), X, MathTex("y=", font_size=50), Y).arrange(RIGHT, buff=0.25)
        top.move_to(LEFT * 3.6 + UP * 0.6)
        lines = VGroup(MathTex(r"1\cdot4=4", font_size=44), MathTex(r"2\cdot(-5)=-10", font_size=44),
                       MathTex(r"3\cdot6=18", font_size=44))
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 2.6 + UP * 1.0)
        tot = MathTex(r"x\cdot y = 4-10+18 = %d" % demo.xy, font_size=50, color=YELLOW).next_to(lines, DOWN, buff=0.5)
        with self.voice(N["dot3"]) as v:
            self.wait_until(v, "one")
            self.play(FadeIn(top), run_time=0.8)
            for ln, w in zip(lines, ["First", "Second", "Third"]):
                self.wait_until(v, w)
                self.play(FadeIn(ln, shift=LEFT * 0.2), run_time=0.6)
            self.wait_until(v, "Add")
            self.play(Write(tot), run_time=1)
            self.wait_until(v, "So")
            self.play(Indicate(tot, color=YELLOW), run_time=0.8)
        ex2 = MathTex(r"\begin{bmatrix}1\\1\end{bmatrix}\cdot\begin{bmatrix}1\\2\end{bmatrix} = 1\cdot1+1\cdot2 = %d" % demo.ab,
                      font_size=46).move_to(DOWN * 1.7)
        v2 = MathTex(r"\text{video 2: row}_1\text{ of }A \cdot x = 2\cdot1+1\cdot2 = 4", font_size=34, color=MUTED)
        v2.move_to(DOWN * 2.75)
        every = T("every entry of a matrix product is a dot product", 24, YELLOW).next_to(v2, DOWN, buff=0.3)
        with self.voice(N["dot4"]) as v:
            self.wait_until(v, "Take")
            self.play(Write(ex2), run_time=1.2)
            self.wait_until(v, "row")
            self.play(FadeIn(v2), run_time=0.6)
            self.wait_until(v, "Every")
            self.play(FadeIn(every), run_time=0.6)


class C2_Geometry(Studio):
    def construct(self):
        self.heading("2 · What the number means")
        pl = grid_plane([-1, 4, 1], [-1, 3, 1], 5.5, 4.4).move_to(LEFT * 3.6 + DOWN * 0.6)
        pl.shift(pl.get_center() - pl.c2p(1.5, 1))
        y = arr(pl, (0, 0), (3, 0), GREEN, 7)
        x = arr(pl, (0, 0), (2, 1), BLUE, 7)
        yl = MathTex("y", color=GREEN, font_size=40).next_to(pl.c2p(3, 0), DR, buff=0.08)
        xl = MathTex("x", color=BLUE, font_size=40).next_to(pl.c2p(2, 1), UR, buff=0.05)
        ray = DashedLine(pl.c2p(2, 1), pl.c2p(2, 0), color=ORANGE, stroke_width=4)
        light = T("light from above ↓", 22, ORANGE).next_to(pl.c2p(2, 2.6), UP, buff=0.05)
        shadow = Line(pl.c2p(0, 0), pl.c2p(2, 0), color=YELLOW, stroke_width=10)
        sl = T("shadow = 2", 24, YELLOW).next_to(shadow, DOWN, buff=0.15)
        with self.voice(N["geo1"]) as v:
            self.play(Create(pl), run_time=0.8)
            self.wait_until(v, "horizontal")
            self.play(GrowArrow(y), FadeIn(yl), run_time=0.6)
            self.wait_until(v, "two")
            self.play(GrowArrow(x), FadeIn(xl), run_time=0.6)
            self.wait_until(v, "light")
            self.play(FadeIn(light), Create(ray), run_time=0.8)
            self.wait_until(v, "shadow")
            self.play(Create(shadow), run_time=0.8)
            self.wait_until(v, "length")
            self.play(FadeIn(sl), run_time=0.5)
        l1 = MathTex(r"x\cdot y = 2\cdot3+1\cdot0 = 6", font_size=44)
        l2 = MathTex(r"\text{length of } y \times \text{shadow} = 3\times2 = 6", font_size=40)
        l3 = T("dot product = length of y × shadow of x on y", 26, YELLOW)
        rgt = VGroup(l1, l2, l3).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(RIGHT * 3.3 + UP * 1.0)
        if rgt.get_right()[0] > 6.9:
            rgt.scale_to_fit_width(6.6).move_to(RIGHT * 3.6 + UP * 1.0)
        proj = T("projection", 24, YELLOW).next_to(sl, DOWN, buff=0.1)
        with self.voice(N["geo2"]) as v:
            self.wait_until(v, "compute")
            self.play(Write(l1), run_time=1)
            self.wait_until(v, "also")
            self.play(Write(l2), run_time=1)
            self.wait_until(v, "meaning")
            self.play(FadeIn(l3), run_time=0.6)
            self.wait_until(v, "projection")
            self.play(FadeIn(proj), Indicate(shadow, color=YELLOW), run_time=0.8)
        f = colmath([r"x\cdot y", "=", r"\|x\|", r"\,\|y\|", r"\,\cos\theta"],
                    [(0, YELLOW), (2, BLUE), (3, GREEN), (4, PURPLE)], 60).move_to(RIGHT * 3.4 + UP * 1.6)
        leg = VGroup(*[VGroup(MathTex(sy, color=co, font_size=36), T(tx, 22, MUTED)).arrange(RIGHT, buff=0.25)
                       for sy, co, tx in [(r"\theta", PURPLE, "the angle between x and y"),
                                          (r"\cos\theta", PURPLE, "1 at 0°, 0 at 90°, −1 at 180°"),
                                          (r"\|x\|", BLUE, "length of x (made exact in video 6)")]])
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(f, DOWN, buff=0.5)
        ang = Angle(Line(pl.c2p(0, 0), pl.c2p(3, 0)), Line(pl.c2p(0, 0), pl.c2p(2, 1)), radius=0.8, color=PURPLE)
        th = MathTex(r"\theta", color=PURPLE, font_size=34).move_to(pl.c2p(1.2, 0.25))
        with self.voice(N["geo3"]) as v:
            self.play(FadeOut(rgt), run_time=0.4)
            self.wait_until(v, "famous")
            self.play(Write(f), Create(ang), FadeIn(th), run_time=1.2)
            self.wait_until(v, "cosine")
            self.play(FadeIn(leg[0]), FadeIn(leg[1]), run_time=0.8)
            self.wait_until(v, "Here")
            self.play(FadeIn(leg[2]), run_time=0.6)
        self.clear_body()
        p2 = grid_plane([-1, 3, 1], [-1, 3, 1], 4, 4).move_to(LEFT * 4.0 + DOWN * 0.5)
        a = arr(p2, (0, 0), tuple(demo.a), BLUE, 7)
        b = arr(p2, (0, 0), tuple(demo.b), GREEN, 7)
        ang2 = Angle(Line(p2.c2p(0, 0), p2.c2p(1, 1)), Line(p2.c2p(0, 0), p2.c2p(1, 2)), radius=1.0, color=PURPLE)
        s1 = MathTex(r"\begin{bmatrix}1\\1\end{bmatrix}\cdot\begin{bmatrix}1\\2\end{bmatrix} = 3", font_size=44)
        s2 = MathTex(r"\|(1,1)\| = \sqrt{1^2+1^2}=\sqrt2,\quad \|(1,2)\|=\sqrt5", font_size=40)
        s3 = MathTex(r"\cos\theta = \frac{3}{\sqrt2\,\sqrt5} = \frac{3}{\sqrt{10}} \approx %.2f" % demo.cos_ab,
                     font_size=44, color=PURPLE)
        s4 = MathTex(r"\theta \approx %.1f^\circ" % demo.deg_ab, font_size=56, color=YELLOW)
        col = VGroup(s1, s2, s3, s4).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(RIGHT * 2.4 + DOWN * 0.3)
        with self.voice(N["geo4"]) as v:
            self.play(Create(p2), GrowArrow(a), GrowArrow(b), Create(ang2), run_time=1)
            self.wait_until(v, "Their")
            self.play(Write(s1), run_time=0.8)
            self.wait_until(v, "Pythagoras")
            self.play(Write(s2), run_time=1.2)
            self.wait_until(v, "cosine")
            self.play(Write(s3), run_time=1.2)
        note = T("from the dot product: angles (and lengths, video 6)", 24, MUTED).to_edge(DOWN, buff=0.4)
        with self.voice(N["geo5"]) as v:
            self.wait_until(v, "eighteen")
            self.play(Write(s4), Indicate(ang2, color=YELLOW), run_time=1)
            self.wait_until(v, "angles")
            self.play(FadeIn(note), run_time=0.6)


class C3_Sign(Studio):
    def construct(self):
        self.heading("3 · The sign tells the direction")
        pl = grid_plane([-3, 3, 1], [-1, 2, 1], 6.3, 3.15).move_to(LEFT * 2.8 + DOWN * 0.2)
        th = ValueTracker(20 * DEGREES)
        u = arr(pl, (0, 0), (2, 0), BLUE, 7)
        vv = always_redraw(lambda: arr(pl, (0, 0), (1.5 * np.cos(th.get_value()), 1.5 * np.sin(th.get_value())),
                                       GREEN, 7))

        def val():
            d = 3 * np.cos(th.get_value())
            return 0.0 if abs(d) < 0.005 else d

        def colour():
            d = val()
            return YELLOW if d > 0.005 else (RED if d < -0.005 else FG)
        read = always_redraw(lambda: VGroup(
            T(f"angle = {np.degrees(th.get_value()):.0f}°", 30, PURPLE, font=MONO),
            T(f"u · v = {val():+.2f}", 36, colour(), font=MONO)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            .move_to(RIGHT * 4.2 + UP * 1.2))
        tag = always_redraw(lambda: T("same way" if val() > 0.005 else ("right angle" if val() == 0 else "opposite way"),
                                      28, colour()).move_to(RIGHT * 4.2 + DOWN * 0.3))
        info = T("u = (2, 0),  v has length 1.5", 22, MUTED).next_to(pl, DOWN, buff=0.25)
        with self.voice(N["sign1"]) as v:
            self.play(Create(pl), GrowArrow(u), FadeIn(info), run_time=1)
            self.wait_until(v, "Watch")
            self.add(vv, read, tag)
            self.wait_until(v, "positive")
            self.play(th.animate.set_value(50 * DEGREES), run_time=1.5)
            self.wait_until(v, "shrinks")
            self.play(th.animate.set_value(75 * DEGREES), run_time=1.2)
        with self.voice(N["sign2"]) as v:
            self.wait_until(v, "right")
            self.play(th.animate.set_value(90 * DEGREES), run_time=1)
            self.wait_until(v, "negative")
            self.play(th.animate.set_value(130 * DEGREES), run_time=1.5)
            self.wait_until(v, "opposite", 2)
            self.play(th.animate.set_value(180 * DEGREES), run_time=1.5)
        self.remove(vv, read, tag)
        self.clear_body()
        p2 = grid_plane([-2, 2, 1], [-1, 2, 1], 4.8, 3.6).move_to(LEFT * 3.8 + DOWN * 0.4)
        o = p2.c2p(0, 0)
        a = arr(p2, (0, 0), (1, 1), BLUE, 7)
        b = arr(p2, (0, 0), (-1, 1), GREEN, 7)
        ra = RightAngle(Line(o, p2.c2p(1, 1)), Line(o, p2.c2p(-1, 1)), length=0.3, color=YELLOW)
        card = self.term_card("orthogonal", "Two vectors are orthogonal when their inner product is "
                              "zero: they meet at a right angle.", pos=DR, width=5.2)
        e = MathTex(r"\begin{bmatrix}1\\1\end{bmatrix}\cdot\begin{bmatrix}-1\\1\end{bmatrix}", r"= -1+1",
                    r"= 0", font_size=46).move_to(RIGHT * 2.6 + UP * 1.6)
        e[2].set_color(YELLOW)
        z = MathTex(r"\mathbf{0}\cdot x = 0 \ \text{ for every } x", font_size=40).next_to(e, DOWN, buff=0.5)
        with self.voice(N["sign3"]) as v:
            self.play(Create(p2), run_time=0.6)
            self.wait_until(v, "orthogonal")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "Check")
            self.play(GrowArrow(a), GrowArrow(b), Write(e[:2]), run_time=1)
            self.wait_until(v, "zero", 2)
            self.play(Write(e[2]), Create(ra), run_time=0.7)
            self.wait_until(v, "zero", 3)
            self.play(FadeIn(z), run_time=0.6)
        self.clear_body()
        c = LEFT * 3.0 + DOWN * 0.6
        nA = Arrow(c, c + UP * 2.2, buff=0, color=GREEN, stroke_width=7)
        eA = Arrow(c, c + RIGHT * 2.2, buff=0, color=BLUE, stroke_width=7)
        nl = T("north", 26, GREEN).next_to(nA, UP, buff=0.1)
        el = T("east", 26, BLUE).next_to(eA, RIGHT, buff=0.1)
        ne = MathTex(r"\text{north}\cdot\text{east} = 0", font_size=48, color=YELLOW).move_to(RIGHT * 3.0 + UP * 1.0)
        feat = T("orthogonal features: separate information", 28, FG).next_to(ne, DOWN, buff=0.6)
        with self.voice(N["sign4"]) as v:
            self.wait_until(v, "north")
            self.play(GrowArrow(nA), FadeIn(nl), run_time=0.6)
            self.wait_until(v, "east")
            self.play(GrowArrow(eA), FadeIn(el), Write(ne), run_time=0.8)
            self.wait_until(v, "features")
            self.play(FadeIn(feat), run_time=0.6)


class C4_Cosine(Studio):
    def construct(self):
        self.heading("4 · Cosine similarity")
        pl = grid_plane([-0.5, 4, 1], [-0.5, 2.5, 1], 4.5, 3).move_to(LEFT * 4.0 + UP * 0.5)
        s = arr(pl, (0, 0), (1.2, 0.6), BLUE, 7)
        lg = arr(pl, (0, 0), (3.6, 1.8), GREEN, 5)
        same = T("same direction, different size", 22, MUTED).next_to(pl, DOWN, buff=0.2)
        f = colmath([r"\cos(x,y)", "=", r"\frac{x\cdot y}{\|x\|\,\|y\|}"], [(0, YELLOW)], 60).move_to(RIGHT * 2.6 + UP * 1.2)
        card = self.term_card("cosine similarity", "The dot product divided by the two lengths: "
                              "the cosine of the angle, from −1 to 1.", pos=DR, width=5.4)
        with self.voice(N["cos1"]) as v:
            self.wait_until(v, "direction")
            self.play(Create(pl), GrowArrow(s), GrowArrow(lg), FadeIn(same), run_time=1)
            self.wait_until(v, "divide")
            self.play(Write(f), run_time=1.2)
            self.wait_until(v, "cosine")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
        nl = NumberLine(x_range=[-1, 1, 0.5], length=8, include_numbers=True, color=MUTED,
                        font_size=36).move_to(DOWN * 2.2 + LEFT * 0.5)
        tags = VGroup(T("opposite", 24, RED).next_to(nl.n2p(-1), UP, buff=0.3),
                      T("orthogonal", 24, FG).next_to(nl.n2p(0), UP, buff=0.3),
                      T("same direction", 24, YELLOW).next_to(nl.n2p(1), UP, buff=0.3))
        with self.voice(N["cos2"]) as v:
            self.play(FadeOut(card), FadeOut(VGroup(pl, s, lg, same)), f.animate.move_to(UP * 1.4), run_time=0.6)
            self.wait_until(v, "between")
            self.play(Create(nl), run_time=0.8)
            self.wait_until(v, "same")
            self.play(FadeIn(tags[2]), run_time=0.5)
            self.wait_until(v, "Zero")
            self.play(FadeIn(tags[1]), run_time=0.5)
            self.wait_until(v, "opposite")
            self.play(FadeIn(tags[0]), run_time=0.5)
        self.clear_body()
        rows = VGroup()
        for name, colr in [("cat", BLUE), ("dog", GREEN), ("car", ORANGE)]:
            vec = demo.words[name]
            rows.add(VGroup(T(name, 32, colr, weight=BOLD),
                            T("[" + ", ".join(f"{e:.1f}" for e in vec) + "]", 30, colr, font=MONO)).arrange(RIGHT, buff=0.5))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(LEFT * 3.4 + UP * 0.6)
        toy = T("toy example: not real word vectors", 22, MUTED).next_to(rows, DOWN, buff=0.4)
        with self.voice(N["cos3"]) as v:
            self.wait_until(v, "toy")
            self.play(FadeIn(toy), run_time=0.5)
            for r, w in zip(rows, ["Cat", "Dog", "Car"]):
                self.wait_until(v, w)
                self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.6)
        base = RIGHT * 0.8 + UP * 1.2
        bars = VGroup()
        for k, (lab, val, colr) in enumerate([("cos(cat, dog)", demo.cat_dog, GREEN),
                                              ("cos(cat, car)", demo.cat_car, ORANGE)]):
            y0 = base + DOWN * 1.2 * k
            t = T(lab, 26, colr).move_to(y0, aligned_edge=LEFT)
            bar = Rectangle(width=4.0 * val, height=0.4, fill_color=colr, fill_opacity=0.8, stroke_width=0)
            bar.next_to(t, DOWN, buff=0.15, aligned_edge=LEFT)
            num = T(f"≈ {val:.2f}", 26, colr).next_to(bar, RIGHT, buff=0.2)
            bars.add(VGroup(t, bar, num))
        closer = T("cat is much closer to dog than to car", 26, YELLOW).move_to(RIGHT * 3.0 + DOWN * 1.6)
        real = T("real embeddings: hundreds of components", 22, MUTED).next_to(closer, DOWN, buff=0.3)
        with self.voice(N["cos4"]) as v:
            self.wait_until(v, "dog")
            self.play(FadeIn(bars[0][0]), GrowFromEdge(bars[0][1], LEFT), FadeIn(bars[0][2]), run_time=0.9)
            self.wait_until(v, "car")
            self.play(FadeIn(bars[1][0]), GrowFromEdge(bars[1][1], LEFT), FadeIn(bars[1][2]), run_time=0.9)
            self.wait_until(v, "closer")
            self.play(FadeIn(closer), run_time=0.6)
            self.wait_until(v, "Real")
            self.play(FadeIn(real), run_time=0.6)


class C5_AxiomB(Studio):
    def construct(self):
        self.heading("5 · Axiom B: the inner product")
        br = colmath([r"\langle x,\,y\rangle"], [(0, YELLOW)], 80).move_to(UP * 1.6)
        three = T("any rule that follows 3 rules", 28, MUTED).next_to(br, DOWN, buff=0.4)
        card = self.term_card("inner product", "A rule ⟨x, y⟩ that turns two vectors into one "
                              "number. It is symmetric, linear in each input, and positive.", pos=DR, width=5.6)
        with self.voice(N["ax1"]) as v:
            self.wait_until(v, "brackets")
            self.play(Write(br), FadeIn(card, shift=UP * 0.3), run_time=1)
            self.wait_until(v, "three")
            self.play(FadeIn(three), run_time=0.6)
        rules = VGroup(
            VGroup(T("1 symmetric", 28, BLUE), MathTex(r"\langle x,y\rangle=\langle y,x\rangle", font_size=44)),
            VGroup(T("2 linear", 28, GREEN), MathTex(r"\langle a x+b z,\ y\rangle=a\langle x,y\rangle+b\langle z,y\rangle",
                                                    font_size=44)),
            VGroup(T("3 positive", 28, YELLOW), MathTex(r"\langle x,x\rangle>0\ \text{ for } x\neq\mathbf{0}", font_size=44)))
        for r in rules:
            r.arrange(RIGHT, buff=0.5)
        rules.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to(DOWN * 1.0 + LEFT * 0.3)
        with self.voice(N["ax2"]) as v:
            self.play(FadeOut(card), FadeOut(three), br.animate.scale(0.7).move_to(UP * 2.2), run_time=0.6)
            for r, w in zip(rules, ["one", "two", "three"]):
                self.wait_until(v, w)
                self.play(FadeIn(r, shift=RIGHT * 0.2), run_time=0.7)
        self.clear_body()
        c1 = MathTex(r"\begin{bmatrix}1\\2\\3\end{bmatrix}\cdot\begin{bmatrix}1\\2\\3\end{bmatrix}=1+4+9=%d>0" % (demo.x @ demo.x),
                     font_size=48).move_to(UP * 1.0)
        c2 = T("squares are never negative → positive for every x ≠ 0", 28, FG).next_to(c1, DOWN, buff=0.5)
        c3 = T("the dot product follows all three rules ✓", 30, GREEN).next_to(c2, DOWN, buff=0.45)
        with self.voice(N["ax3"]) as v:
            self.wait_until(v, "Check")
            self.play(Write(c1), run_time=1.2)
            self.wait_until(v, "Squares")
            self.play(FadeIn(c2), run_time=0.6)
            self.wait_until(v, "all")
            self.play(FadeIn(c3), run_time=0.6)
        self.clear_body()
        l1 = T("vector space: add and scale — no lengths, no angles", 28, MUTED).move_to(UP * 2.0)
        l2 = T("+ inner product → lengths and angles", 30, FG).next_to(l1, DOWN, buff=0.35)
        ax_t = T("Axiom B · Inner product", 56, YELLOW, weight=BOLD).next_to(l2, DOWN, buff=0.6)
        oth = T("other inner products exist, and they give different angles", 22, MUTED).next_to(ax_t, DOWN, buff=0.5)
        with self.voice(N["ax4"]) as v:
            self.wait_until(v, "add")
            self.play(FadeIn(l1), run_time=0.6)
            self.wait_until(v, "adds")
            self.play(FadeIn(l2), run_time=0.6)
            self.wait_until(v, "axiom")
            self.play(Write(ax_t), run_time=1)
            self.wait_until(v, "other")
            self.play(FadeIn(oth), run_time=0.6)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · Inner products in machine learning")
        t1 = T("attention score = dot product", 32, YELLOW).move_to(UP * 2.1)
        yv = VGroup(T("Your", 30, BLUE, weight=BOLD), T("[0.43, 0.15, 0.89]", 28, BLUE, font=MONO)).arrange(RIGHT, buff=0.4)
        jv = VGroup(T("journey", 30, GREEN, weight=BOLD), T("[0.55, 0.87, 0.66]", 28, GREEN, font=MONO)).arrange(RIGHT, buff=0.4)
        vecs = VGroup(yv, jv).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(UP * 0.6)
        src = T("Raschka, Build a Large Language Model (From Scratch), §3.3.1, p. 58", 20, MUTED).to_edge(DOWN, buff=0.4)
        with self.voice(N["ml1"]) as v:
            self.wait_until(v, "attention")
            self.play(FadeIn(t1), run_time=0.6)
            self.wait_until(v, "Your")
            self.play(FadeIn(yv), FadeIn(src), run_time=0.7)
        calc = MathTex(r"0.43\cdot0.55+0.15\cdot0.87+0.89\cdot0.66", r"\approx %.4f" % demo.score, font_size=44)
        calc[1].set_color(YELLOW)
        calc.next_to(vecs, DOWN, buff=0.6)
        big = T("bigger score → more related → attention mixes in more of that word", 24, FG).next_to(calc, DOWN, buff=0.45)
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "journey")
            self.play(FadeIn(jv), run_time=0.6)
            self.wait_until(v, "product")
            self.play(Write(calc), run_time=1.5)
            self.wait_until(v, "bigger")
            self.play(FadeIn(big), run_time=0.6)
        self.clear_body()
        pl = grid_plane([-2, 3, 1], [-1, 3, 1], 4.5, 3.6).move_to(LEFT * 3.8 + DOWN * 0.4)
        q = arr(pl, (0, 0), (2, 1.5), YELLOW, 7)
        ql = T("query", 22, YELLOW).next_to(pl.c2p(2, 1.5), UR, buff=0.05)
        stored = [(-1.5, 1.5), (1.0, 2.6), (2.6, 1.6), (2.5, -0.6), (-1.8, -0.5)]
        st = VGroup(*[arr(pl, (0, 0), s, MUTED, 4) for s in stored])
        best = arr(pl, (0, 0), stored[2], GREEN, 6)
        bl = T("largest cosine", 22, GREEN).next_to(pl, DOWN, buff=0.15)
        neu = colmath([r"w\cdot x", "=", "w_1x_1+w_2x_2+w_3x_3"], [(0, YELLOW)], 46).move_to(RIGHT * 3.2 + UP * 0.8)
        neul = T("a neuron's weighted sum is a dot product", 24, MUTED).next_to(neu, DOWN, buff=0.35)
        with self.voice(N["ml3"]) as v:
            self.wait_until(v, "Similarity")
            self.play(Create(pl), GrowArrow(q), FadeIn(ql), *[GrowArrow(s) for s in st], run_time=1.2)
            self.wait_until(v, "largest")
            self.play(ReplacementTransform(st[2], best), FadeIn(bl), run_time=0.8)
            self.wait_until(v, "neuron")
            self.play(Write(neu), FadeIn(neul), run_time=1)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        q = MathTex(r"\begin{bmatrix}2\\-1\\3\end{bmatrix}\cdot\begin{bmatrix}1\\4\\2\end{bmatrix}", r"=\ ?", font_size=64)
        q[1].set_color(YELLOW)
        q.shift(UP * 1.3)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.4)
        with self.voice(N["prac1"]) as v:
            self.play(Write(q), run_time=1.5)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
        ans = MathTex(r"2\cdot1", r"+(-1)\cdot4", r"+3\cdot2", r"=2-4+6", r"=4", font_size=50)
        ans[4].set_color(YELLOW)
        ans.move_to(DOWN * 1.6)
        pos = T("positive → roughly the same way", 26, GREEN).next_to(ans, DOWN, buff=0.4)
        assert 2 * 1 + (-1) * 4 + 3 * 2 == 4
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.play(Write(ans[0]), run_time=0.6)
            self.wait_until(v, "Minus")
            self.play(Write(ans[1]), run_time=0.6)
            self.wait_until(v, "Three")
            self.play(Write(ans[2]), run_time=0.6)
            self.wait_until(v, "Add")
            self.play(Write(ans[3:]), run_time=0.8)
            self.wait_until(v, "positive")
            self.play(FadeIn(pos), run_time=0.6)
        self.clear_body()
        q2 = MathTex(r"\begin{bmatrix}3\\6\end{bmatrix},\ \begin{bmatrix}-2\\1\end{bmatrix}", r"\quad\text{orthogonal?}",
                     font_size=60)
        q2[1].set_color(MUTED)
        q2.shift(UP * 1.4)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q2, DOWN, buff=0.4)
        with self.voice(N["prac3"]) as v:
            self.play(Write(q2), run_time=1.2)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
        a2 = MathTex(r"3\cdot(-2)+6\cdot1=-6+6=0", font_size=52).move_to(DOWN * 1.2)
        yes = T("dot product 0 → orthogonal ✓ (right angle)", 30, GREEN).next_to(a2, DOWN, buff=0.4)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "Three")
            self.play(Write(a2), run_time=1.5)
            self.wait_until(v, "yes")
            self.play(FadeIn(yes), run_time=0.6)


CODE = '''import numpy as np
x, y = np.array([1, 2, 3]), np.array([4, -5, 6])
print("x . y =", x @ y)                     # sum of x_i * y_i
a, b = np.array([1, 1]), np.array([1, 2])
cos = a @ b / (np.sqrt(a @ a) * np.sqrt(b @ b))
print(a @ b, cos, np.degrees(np.arccos(cos)))   # angle
print(np.array([1, 1]) @ np.array([-1, 1]))     # orthogonal
print(x @ y == y @ x, x @ x)                    # rule checks
cat, dog = np.array([.9, .8, .1]), np.array([.8, .9, .2])
car = np.array([.1, .2, .9])                    # toy vectors
cs = lambda p, q: p @ q / (np.sqrt(p @ p) * np.sqrt(q @ q))
print(cs(cat, dog), cs(cat, car))
your, journey = np.array([.43, .15, .89]), np.array([.55, .87, .66])
print("score =", your @ journey)                # book, p. 58'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=22)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[2], color=YELLOW, buff=0.06)
            self.wait_until(v, "at")
            self.play(Create(box), run_time=0.5)
            for i, w in [(5, "angle"), (6, "orthogonal"), (7, "rule"), (11, "cosine"), (13, "attention")]:
                self.wait_until(v, w)
                self.play(Transform(box, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                              buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=26, command="python demo.py").shift(DOWN * 0.4)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for idx, w in [((0,), "twelve"), ((2,), "eighteen"), ((3,), "zero"), ((5, 6), "toy"),
                           ((7,), "attention")]:
                self.wait_until(v, w)
                self.play(*[Indicate(term[1][i], color=YELLOW) for i in idx], run_time=0.7)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["Dot product: multiply matching components, add → one number",
                            "x·y = ‖x‖ ‖y‖ cos θ: length × length × cosine",
                            "Sign: same way (+), right angle (0), opposite (−); 0 = orthogonal",
                            "Cosine similarity keeps only direction: from −1 to 1",
                            "Inner product: symmetric, linear, positive = axiom B (geometry)"],
                           size=28).shift(UP * 0.5)
        with self.voice(N["sum1"]) as v:
            for p in pts[:3]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 5)
        nxt = VGroup(T("Next · Video 6", 26, MUTED),
                     T("Norm: measuring length and distance", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[3], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "symmetric")
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Dot", "C2_Geometry", "C3_Sign", "C4_Cosine", "C5_AxiomB", "C6_ML",
          "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Dot": "The dot product", "C2_Geometry": "What the number means",
                  "C3_Sign": "The sign and orthogonality", "C4_Cosine": "Cosine similarity",
                  "C5_AxiomB": "Axiom B: the inner product", "C6_ML": "Inner products in ML",
                  "C7_Practice": "Try it yourself", "C7_Code": "Python demo", "C8_Summary": "Summary",
                  "C9_Credits": "Credits"}
