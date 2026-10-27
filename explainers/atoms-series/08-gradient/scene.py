"""Gradient — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 8 of 16"
TITLE = "Gradient — the slope in many directions"
LEVELS = [5, 10, 15, 20, 25]


def axes(xr, yr, xl, yl, ticks=True):
    return Axes(x_range=xr, y_range=yr, x_length=xl, y_length=yl, tips=False,
                axis_config={"stroke_color": MUTED, "include_ticks": ticks})


def fit(m, w):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def unit(ax):
    return ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]


def bowl_map(center, size=5.6, labels=True):
    """Contour map of f(x, y) = x^2 + y^2: circles of radius sqrt(level)."""
    ax = axes([-6, 6, 2], [-6, 6, 2], size, size).move_to(center)
    u = unit(ax)
    rings = VGroup(*[Circle(radius=np.sqrt(c) * u, color=BLUE, stroke_width=2.5,
                            stroke_opacity=0.35 + 0.12 * i).move_to(ax.c2p(0, 0))
                     for i, c in enumerate(LEVELS)])
    labs = VGroup()
    if labels:
        for i, c in enumerate(LEVELS):
            r, a = np.sqrt(c), -0.25 - 0.32 * i
            labs.add(T(str(c), 18, MUTED).move_to(ax.c2p(r * np.cos(a), r * np.sin(a)) +
                                                  0.2 * np.array([np.cos(a), np.sin(a), 0])))
    return ax, rings, labs


def arr(ax, a, b, color, width=6):
    return Arrow(ax.c2p(*a), ax.c2p(*b), buff=0, color=color, stroke_width=width,
                 max_tip_length_to_length_ratio=0.2)


def legend(rows, x_sym=-3.4, x_txt=-2.4, y0=0.2, dy=0.75, size=26, fs=44):
    g = VGroup()
    for i, (sym, txt, color) in enumerate(rows):
        s = MathTex(sym, color=color, font_size=fs).move_to([x_sym, y0 - i * dy, 0])
        t = T(txt, size, FG)
        t.move_to([x_txt, y0 - i * dy, 0], aligned_edge=LEFT)
        g.add(VGroup(s, t))
    return g


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 56, FG, weight=BOLD)
        fit(title, 12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.2)
        ws = VGroup(*[MathTex(r"w_%d" % i, color=PURPLE, font_size=44) for i in (1, 2, 3)],
                    MathTex(r"\vdots", color=PURPLE, font_size=44))
        ws.arrange(DOWN, buff=0.75).move_to(LEFT * 5 + DOWN * 0.8)
        sl = VGroup(*[MathTex(r"\frac{\partial L}{\partial w_%d}" % i, color=ORANGE, font_size=40)
                      for i in (1, 2, 3)], MathTex(r"\vdots", color=ORANGE, font_size=44))
        for s, w in zip(sl, ws):
            s.move_to([-2.0, w.get_y(), 0])
        arrows = VGroup(*[Arrow(w.get_right(), s.get_left(), buff=0.2, color=MUTED, stroke_width=3)
                          for w, s in zip(ws[:3], sl[:3])])
        vec = MathTex(r"\nabla L", color=YELLOW, font_size=96).move_to(RIGHT * 2.4 + DOWN * 0.8)
        up = Arrow(RIGHT * 4.3 + DOWN * 1.8, RIGHT * 5.6 + UP * 0.2, buff=0, color=YELLOW)
        ul = T("uphill", 26, YELLOW).next_to(up, RIGHT, buff=0.1)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            self.wait_until(v, "millions")
            self.play(FadeIn(ws), run_time=0.7)
            self.wait_until(v, "each")
            self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2), FadeIn(sl), run_time=1)
            self.wait_until(v, "gradient")
            self.play(TransformFromCopy(sl, vec), run_time=1)
            self.wait_until(v, "uphill")
            self.play(GrowArrow(up), FadeIn(ul), run_time=0.7)
        items = VGroup(*[VGroup(MathTex(s, color=c, font_size=50), T(t, 24, MUTED)).arrange(DOWN, buff=0.3)
                         for s, c, t in [(r"f(x,y)", BLUE, "a map of heights"),
                                         (r"\frac{\partial f}{\partial x}", ORANGE, "partial derivative"),
                                         (r"\nabla f", YELLOW, "gradient"),
                                         (r"\uparrow", GREEN, "steepest uphill")]])
        items.arrange(RIGHT, buff=1.0).shift(DOWN * 0.5)
        fit(items, 13)
        pre = T("you need: video 5 (inner product) · video 6 (norm) · video 7 (derivative)", 24, MUTED)
        pre.to_edge(DOWN, buff=0.7)
        with self.voice(N["intro2"]) as v:
            self.play(*[FadeOut(m) for m in (ws, sl, arrows, vec, up, ul)], run_time=0.6)
            for it, w in zip(items, ["heights", "partial", "build", "uphill"]):
                self.wait_until(v, w)
                self.play(FadeIn(it, shift=UP * 0.2), run_time=0.5)
            self.wait_until(v, "need")
            self.play(FadeIn(pre), run_time=0.6)


class C1_Map(Studio):
    def construct(self):
        self.heading("1 · A function of two inputs = a map of heights")
        inp = MathTex(r"(x,\,y)", color=BLUE, font_size=60)
        box = VGroup(RoundedRectangle(width=2.0, height=1.3, corner_radius=0.15, color=FG),
                     MathTex("f", font_size=60))
        out = MathTex(r"\text{height}", color=YELLOW, font_size=50)
        row = VGroup(inp, box, out).arrange(RIGHT, buff=1.2).shift(UP * 1.0)
        a1 = Arrow(inp.get_right(), box.get_left(), buff=0.15, color=MUTED)
        a2 = Arrow(box.get_right(), out.get_left(), buff=0.15, color=MUTED)
        f = MathTex(r"f(x,y)", "=", r"x^2+y^2", font_size=56).shift(DOWN * 0.6)
        f[0].set_color(YELLOW)
        ex = MathTex(r"f(3,4)", "=", "9+16", "=", "%g" % demo.f_before, font_size=56).next_to(f, DOWN, buff=0.5)
        ex[4].set_color(YELLOW)
        with self.voice(N["map1"]) as v:
            self.wait_until(v, "curve")
            self.play(FadeIn(box), run_time=0.5)
            self.wait_until(v, "two")
            self.play(FadeIn(inp), GrowArrow(a1), run_time=0.6)
            self.wait_until(v, "output", 2)
            self.play(GrowArrow(a2), FadeIn(out), run_time=0.6)
            self.wait_until(v, "example")
            self.play(Write(f), run_time=1)
            self.wait_until(v, "nine")
            self.play(Write(ex), run_time=1)
        self.clear_body()
        ax = axes([-5.5, 5.5, 1], [0, 31, 5], 7.0, 4.4).shift(LEFT * 1.8 + DOWN * 0.9)
        side = ax.plot(lambda x: x * x, x_range=[-5.5, 5.5], color=BLUE, stroke_width=5)
        grd = Line(ax.c2p(-5.5, 0), ax.c2p(5.5, 0), color=GREEN, stroke_width=5)
        gl = T("the ground: the x y plane", 22, GREEN).next_to(grd, DOWN, buff=0.15)
        bl = T("the bowl, seen from the side", 26, BLUE).next_to(ax, UP, buff=0.1)
        low = Dot(ax.c2p(0, 0), color=YELLOW, radius=0.1)
        ll = T("lowest point: (0, 0), height 0", 26, YELLOW).move_to(RIGHT * 4.6 + DOWN * 0.2)
        fit(ll, 4.4)
        with self.voice(N["map2"]) as v:
            self.wait_until(v, "ground")
            self.play(Create(ax), Create(grd), FadeIn(gl), run_time=1)
            self.wait_until(v, "surface")
            self.play(Create(side), run_time=1.2)
            self.wait_until(v, "bowl")
            self.play(FadeIn(bl), run_time=0.5)
            self.wait_until(v, "lowest")
            self.play(FadeIn(low), FadeIn(ll), run_time=0.6)
        self.clear_body()
        ax, rings, labs = bowl_map(LEFT * 3.2 + DOWN * 0.6)
        p = Dot(ax.c2p(3, 4), color=YELLOW, radius=0.09)
        pl = MathTex("(3,4)", color=YELLOW, font_size=32).next_to(p, UR, buff=0.05)
        t1 = VGroup(T("contour line:", 28, BLUE, weight=BOLD),
                    T("all points with the same height", 26, FG),
                    MathTex(r"x^2+y^2=25\ \Rightarrow\ \text{radius } 5", font_size=40),
                    T("heights drawn: 5, 10, 15, 20, 25", 24, MUTED)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.5 + UP * 0.6)
        with self.voice(N["map3"]) as v:
            self.play(Create(ax), run_time=0.6)
            self.wait_until(v, "contour")
            self.play(FadeIn(t1[:2]), run_time=0.6)
            self.wait_until(v, "circle")
            self.play(Create(rings[-1]), FadeIn(p), FadeIn(pl), Write(t1[2]), run_time=1)
            self.wait_until(v, "Lower")
            self.play(LaggedStart(*[Create(r) for r in rings[:-1][::-1]], lag_ratio=0.3), run_time=1.5)
            self.wait_until(v, "draw")
            self.play(FadeIn(labs), FadeIn(t1[3]), run_time=0.6)
        close = T("close together → steep", 26, RED)
        far = T("far apart → gentle", 26, GREEN)
        cf = VGroup(close, far).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.5 + DOWN * 2.1)
        outer = Line(ax.c2p(np.sqrt(20), 0), ax.c2p(5, 0), color=RED, stroke_width=8)
        inner = Line(ax.c2p(np.sqrt(5), 0), ax.c2p(np.sqrt(10), 0), color=GREEN, stroke_width=8)
        with self.voice(N["map4"]) as v:
            self.wait_until(v, "close")
            self.play(FadeIn(close), Create(outer), run_time=0.7)
            self.wait_until(v, "far")
            self.play(FadeIn(far), Create(inner), run_time=0.7)
            self.wait_until(v, "arrows")
            self.play(Indicate(p, color=YELLOW), run_time=0.8)


class C2_Partial(Studio):
    def construct(self):
        self.heading("2 · Partial derivatives: freeze the others")
        ax = axes([-3, 4, 1], [-3, 6, 1], 4.2, 5.4).shift(LEFT * 4.4 + DOWN * 0.7)
        cont = VGroup(*[ax.plot(lambda x, c=c: (c - x * x) / 3, x_range=[-3, 3], color=BLUE,
                                stroke_width=2.5, stroke_opacity=0.6) for c in (1, 4, 7, 10, 13)])
        p = Dot(ax.c2p(1, 2), color=YELLOW, radius=0.09)
        pl = MathTex("(1,2)", color=YELLOW, font_size=30).next_to(p, DL, buff=0.05)
        ax_x = arr(ax, (1, 2), (2.3, 2), ORANGE, 5)
        ax_y = arr(ax, (1, 2), (1, 3.3), GREEN, 5)
        q = T("how steep here?", 28, YELLOW).move_to(RIGHT * 2.0 + UP * 1.8)
        d1 = T("walk in x only, y frozen", 26, ORANGE).move_to(RIGHT * 2.0 + UP * 0.8)
        d2 = T("walk in y only, x frozen", 26, GREEN).next_to(d1, DOWN, buff=0.3)
        with self.voice(N["par1"]) as v:
            self.play(Create(ax), run_time=0.6)
            self.play(Create(cont), FadeIn(p), FadeIn(pl), run_time=1)
            self.wait_until(v, "steep")
            self.play(FadeIn(q), run_time=0.5)
            self.wait_until(v, "x")
            self.play(GrowArrow(ax_x), FadeIn(d1), run_time=0.6)
            self.wait_until(v, "y", 2)
            self.play(GrowArrow(ax_y), FadeIn(d2), run_time=0.6)
        f = MathTex(r"f(x,y)=x^2+3y", color=BLUE, font_size=46).move_to(RIGHT * 2.2 + UP * 2.2)
        fy = MathTex(r"y=2:\ \ f=x^2+6", font_size=42)
        dx = MathTex(r"\tfrac{d}{dx}(x^2+6)=2x", r"\;\Rightarrow\;", r"2", font_size=42)
        dx[2].set_color(ORANGE)
        g1 = VGroup(fy, dx).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(f, DOWN, buff=0.35)
        g1.align_to(f, LEFT)
        hline = DashedLine(ax.c2p(-3, 2), ax.c2p(4, 2), color=ORANGE)
        sx = axes([-0.5, 2.5, 1], [4, 14, 2], 3.6, 2.4).move_to(RIGHT * 2.4 + DOWN * 2.2)
        sc = sx.plot(lambda x: x * x + 6, x_range=[-0.5, 2.5], color=ORANGE)
        stan = Line(sx.c2p(0.2, 5.4), sx.c2p(1.8, 8.6), color=YELLOW, stroke_width=4)
        sxl = T("slice at y = 2", 20, ORANGE).next_to(sx, RIGHT, buff=0.15)
        with self.voice(N["par2"]) as v:
            self.play(FadeOut(q), FadeOut(d1), FadeOut(d2), run_time=0.4)
            self.wait_until(v, "example")
            self.play(Write(f), run_time=0.8)
            self.wait_until(v, "Freeze")
            self.play(Create(hline), Write(fy), run_time=0.8)
            self.play(Create(sx), Create(sc), FadeIn(sxl), run_time=0.8)
            self.wait_until(v, "derivative")
            self.play(Write(dx[0]), run_time=0.8)
            self.wait_until(v, "one", 2)
            self.play(Write(dx[1:]), Create(stan), run_time=0.8)
        fx = MathTex(r"x=1:\ \ f=1+3y", font_size=42)
        dy = MathTex(r"\tfrac{d}{dy}(1+3y)=", r"3", font_size=42)
        dy[1].set_color(GREEN)
        g2 = VGroup(fx, dy).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(g1, DOWN, buff=0.35)
        g2.align_to(f, LEFT)
        vline = DashedLine(ax.c2p(1, -3), ax.c2p(1, 6), color=GREEN)
        card = self.term_card("partial derivative", "The slope of a function when only one input "
                              "moves and all the other inputs are frozen.", pos=DR, width=5.4)
        with self.voice(N["par3"]) as v:
            self.play(FadeOut(VGroup(sx, sc, stan, sxl)), run_time=0.4)
            self.wait_until(v, "freeze")
            self.play(Create(vline), Write(fx), run_time=0.8)
            self.wait_until(v, "derivative")
            self.play(Write(dy), run_time=0.8)
            self.wait_until(v, "called")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
        self.clear_body()
        big = MathTex(r"\frac{\partial f}{\partial x}", "=", "2x", r"\qquad",
                      r"\frac{\partial f}{\partial y}", "=", "3", font_size=72).move_to(UP * 1.3)
        big[0].set_color(ORANGE); big[4].set_color(GREEN)
        leg = legend([(r"\partial", "“curly d”: read it as “partial”", FG),
                      (r"\frac{\partial f}{\partial x}", "slope of f when only x moves", ORANGE),
                      (r"y\to\text{const.}", "freeze the other inputs", MUTED)],
                     x_sym=-5.5, x_txt=-3.9, y0=-0.6, dy=1.0, size=26, fs=40)
        rem = self.reminder_card("derivative", "The slope at one point: the limit of rise over "
                                 "run (video 7). Its rules still work.", pos=DR, width=5.0)
        with self.voice(N["par4"]) as v:
            self.wait_until(v, "curly")
            self.play(Write(big), FadeIn(leg[0]), run_time=1.2)
            self.wait_until(v, "partial", 2)
            self.play(FadeIn(leg[1]), run_time=0.5)
            self.wait_until(v, "constant")
            self.play(FadeIn(leg[2]), run_time=0.5)
            self.wait_until(v, "normal")
            self.play(FadeIn(rem, shift=UP * 0.3), run_time=0.6)


class C3_Gradient(Studio):
    def construct(self):
        self.heading("3 · The gradient bundles the slopes")
        ax = axes([-3, 4, 1], [-3, 6, 1], 4.2, 5.4).shift(LEFT * 4.4 + DOWN * 0.7)
        cont = VGroup(*[ax.plot(lambda x, c=c: (c - x * x) / 3, x_range=[-3, 3], color=BLUE,
                                stroke_width=2.5, stroke_opacity=0.6) for c in (1, 4, 7, 10, 13)])
        p = Dot(ax.c2p(1, 2), color=FG, radius=0.08)
        ga = arr(ax, (1, 2), (1 + demo.g1_exact[0], 2 + demo.g1_exact[1]), YELLOW, 7)
        gl = MathTex(r"\nabla f", color=YELLOW, font_size=36).next_to(ga.get_end(), RIGHT, buff=0.1)
        g = MathTex(r"\nabla f", "=", r"\begin{bmatrix}\partial f/\partial x\\ \partial f/\partial y\end{bmatrix}",
                    "=", r"\begin{bmatrix}2x\\3\end{bmatrix}", "=",
                    r"\begin{bmatrix}%g\\%g\end{bmatrix}" % tuple(demo.g1_exact), font_size=48)
        g[0].set_color(YELLOW); g[6].set_color(YELLOW)
        fit(g, 8.2).move_to(RIGHT * 2.6 + UP * 1.4)
        at = T("at (1, 2)", 24, MUTED).next_to(g, DOWN, buff=0.2)
        card = self.term_card("gradient", "The vector of all partial derivatives. It points "
                              "in the direction of steepest increase.", pos=DR, width=5.4)
        with self.voice(N["grad1"]) as v:
            self.play(Create(ax), Create(cont), FadeIn(p), run_time=1)
            self.wait_until(v, "bundle")
            self.play(Write(g[:3]), run_time=1)
            self.wait_until(v, "gradient")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "nabla")
            self.play(Indicate(g[0], color=YELLOW), run_time=0.7)
            self.wait_until(v, "two", 2)
            self.play(Write(g[3:]), FadeIn(at), run_time=1)
            self.wait_until(v, "arrow")
            self.play(GrowArrow(ga), FadeIn(gl), run_time=0.8)
        self.clear_body()
        bx, rings, labs = bowl_map(LEFT * 3.4 + DOWN * 0.6)
        p2 = Dot(bx.c2p(3, 4), color=FG, radius=0.08)
        s = 0.25
        g2a = arr(bx, (3, 4), (3 + s * demo.g2_exact[0], 4 + s * demo.g2_exact[1]), YELLOW, 7)
        rows = VGroup(MathTex(r"f=x^2+y^2", color=BLUE, font_size=46),
                      MathTex(r"\tfrac{\partial f}{\partial x}=2x,\quad \tfrac{\partial f}{\partial y}=2y", font_size=44),
                      MathTex(r"\nabla f=\begin{bmatrix}2x\\2y\end{bmatrix}", font_size=44),
                      MathTex(r"\nabla f(3,4)=\begin{bmatrix}%g\\%g\end{bmatrix},\ \ \|\nabla f\|=%g"
                              % (demo.g2_exact[0], demo.g2_exact[1], np.linalg.norm(demo.g2_exact)),
                              color=YELLOW, font_size=44))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.3 + UP * 0.2)
        fit(rows, 6.6)
        note = T("arrows drawn at ¼ length", 20, MUTED).next_to(bx, DOWN, buff=0.1)
        with self.voice(N["grad2"]) as v:
            self.play(Create(bx), Create(rings), FadeIn(labs), FadeIn(p2), FadeIn(rows[0]), run_time=1)
            self.wait_until(v, "partial")
            self.play(Write(rows[1]), run_time=1)
            self.wait_until(v, "gradient")
            self.play(Write(rows[2]), run_time=0.8)
            self.wait_until(v, "six")
            self.play(Write(rows[3]), GrowArrow(g2a), FadeIn(note), run_time=1)
        field = VGroup()
        for x in (-4, -2, 0, 2, 4):
            for y in (-4, -2, 0, 2, 4):
                if (x, y) != (0, 0):
                    field.add(arr(bx, (x, y), (x + s * 2 * x, y + s * 2 * y), ORANGE, 3))
        tip = VGroup(T("long where steep (far out)", 24, RED), T("short near the bottom", 24, GREEN)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(RIGHT * 3.3 + DOWN * 2.6)
        with self.voice(N["grad3"]) as v:
            self.wait_until(v, "many")
            self.play(LaggedStart(*[GrowArrow(a) for a in field], lag_ratio=0.05), run_time=2)
            self.wait_until(v, "long")
            self.play(FadeIn(tip[0]), run_time=0.5)
            self.wait_until(v, "short")
            self.play(FadeIn(tip[1]), run_time=0.5)
        self.clear_body()
        a = VGroup(T("MML book: a row", 28, BLUE),
                   MathTex(r"\nabla f=\big[\,6\ \ \ 8\,\big]", font_size=56)).arrange(DOWN, buff=0.3)
        b = VGroup(T("here: an arrow", 28, YELLOW),
                   Arrow(LEFT * 0.9 + DOWN * 0.6, RIGHT * 0.3 + UP * 1.0, buff=0, color=YELLOW)
                   ).arrange(DOWN, buff=0.3)
        c = VGroup(T("in code: same shape as the weights", 28, PURPLE),
                   T("W.shape == W.grad.shape", 26, PURPLE, font=MONO)).arrange(DOWN, buff=0.3)
        VGroup(a, b).arrange(RIGHT, buff=2.0).shift(UP * 0.8)
        same = T("same information", 26, MUTED).next_to(VGroup(a, b), DOWN, buff=0.4)
        c.next_to(same, DOWN, buff=0.6)
        with self.voice(N["grad4"]) as v:
            self.wait_until(v, "row")
            self.play(FadeIn(a), run_time=0.7)
            self.wait_until(v, "arrow")
            self.play(FadeIn(b), run_time=0.7)
            self.wait_until(v, "same")
            self.play(FadeIn(same), run_time=0.5)
            self.wait_until(v, "code")
            self.play(FadeIn(c), run_time=0.7)


class C4_Linear(Studio):
    def construct(self):
        self.heading("4 · Local linearity with many inputs")
        one = MathTex(r"f(x+h)", r"\approx", r"f(x)", "+", r"f'(x)", r"\,h", font_size=58).move_to(UP * 1.6)
        one[0].set_color(YELLOW); one[4].set_color(PURPLE); one[5].set_color(BLUE)
        many = MathTex(r"f(x+h)", r"\approx", r"f(x)", "+", r"\nabla f", r"\cdot", r"h", font_size=66).move_to(UP * 1.6)
        many[0].set_color(YELLOW); many[4].set_color(PURPLE); many[6].set_color(BLUE)
        lab1 = T("one input (video 7)", 24, MUTED).next_to(one, DOWN, buff=0.3)
        lab2 = T("h is now a vector: a step on the map", 24, BLUE).next_to(many, DOWN, buff=0.3)
        rem1 = self.reminder_card("local linear approximation", "Near x, a function is almost its "
                                  "tangent: f(x+h) ≈ f(x) + f′(x)·h  (video 7).", pos=DR, width=5.4)
        rem2 = self.reminder_card("dot product", "Multiply matching components and add: "
                                  "a·b = a₁b₁ + a₂b₂ + …  (video 5).", pos=DR, width=5.4)
        with self.voice(N["lin1"]) as v:
            self.wait_until(v, "linear")
            self.play(Write(one), FadeIn(lab1), FadeIn(rem1, shift=UP * 0.3), run_time=1.2)
            self.wait_until(v, "vector")
            self.play(ReplacementTransform(one, many), FadeOut(lab1), FadeIn(lab2), run_time=1.2)
            self.wait_until(v, "dot")
            self.play(FadeOut(rem1), FadeIn(rem2, shift=UP * 0.3), Indicate(many[5], color=YELLOW), run_time=0.8)
        ex = MathTex(r"\nabla f\cdot h", "=", r"\frac{\partial f}{\partial x}\,h_1", "+",
                     r"\frac{\partial f}{\partial y}\,h_2", font_size=56).move_to(DOWN * 0.4 + LEFT * 1.2)
        ex[0].set_color(PURPLE)
        axc = T("Axiom C · local linearity, written with the inner product (axiom B)", 26, YELLOW)
        axc.to_edge(DOWN, buff=0.6)
        fit(axc, 13)
        with self.voice(N["lin2"]) as v:
            self.play(FadeOut(rem2), run_time=0.4)
            self.wait_until(v, "multiplies")
            self.play(Write(ex), run_time=1.2)
            self.wait_until(v, "axiom")
            self.play(FadeIn(axc), run_time=0.6)
        self.clear_body()
        ax = axes([-1, 4, 1], [-1, 5, 1], 4.2, 5.04).shift(LEFT * 4.5 + DOWN * 0.7)
        cont = VGroup(*[ax.plot(lambda x, c=c: (c - x * x) / 3, x_range=[-1, 3.6], color=BLUE,
                                stroke_width=2.5, stroke_opacity=0.5)
                        for c in (1, 4, 7, 10, 13) if True])
        p = Dot(ax.c2p(1, 2), color=FG, radius=0.08)
        hstep = arr(ax, (1, 2), (1.5, 2.5), BLUE, 6)
        hl = T("h (drawn 5× longer)", 20, BLUE).next_to(hstep.get_end(), UR, buff=0.05)
        rows = VGroup(MathTex(r"f(1,2)=1+6=7", font_size=42),
                      MathTex(r"h=(0.1,\,0.1)", color=BLUE, font_size=42),
                      MathTex(r"\nabla f\cdot h=2(0.1)+3(0.1)=0.5", color=PURPLE, font_size=42),
                      MathTex(r"\text{guess: } 7+0.5=%.1f" % demo.lin_guess, color=YELLOW, font_size=42),
                      MathTex(r"\text{true: } 1.1^2+3(2.1)=%.2f" % demo.lin_true, color=GREEN, font_size=42))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 2.2 + DOWN * 0.3)
        fit(rows, 8)
        with self.voice(N["lin3"]) as v:
            self.play(Create(ax), Create(cont), FadeIn(p), run_time=0.8)
            self.wait_until(v, "seven")
            self.play(FadeIn(rows[0]), run_time=0.5)
            self.wait_until(v, "Step")
            self.play(FadeIn(rows[1]), GrowArrow(hstep), FadeIn(hl), run_time=0.7)
            self.wait_until(v, "gradient")
            self.play(FadeIn(rows[2]), run_time=0.7)
            self.wait_until(v, "guess")
            self.play(FadeIn(rows[3]), run_time=0.6)
            self.wait_until(v, "true")
            self.play(FadeIn(rows[4]), run_time=0.6)
        rows2 = VGroup(MathTex(r"f=x^2+y^2 \text{ at } (3,4):\ 25", color=BLUE, font_size=42),
                       MathTex(r"h=(0.1,\,-0.1)", color=BLUE, font_size=42),
                       MathTex(r"\nabla f\cdot h=6(0.1)-8(0.1)=-0.2", color=PURPLE, font_size=42),
                       MathTex(r"\text{guess: } %.1f" % demo.lin2_guess, color=YELLOW, font_size=42),
                       MathTex(r"\text{true: } 3.1^2+3.9^2=%.2f" % demo.lin2_true, color=GREEN, font_size=42))
        rows2.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 2.2 + DOWN * 0.3)
        fit(rows2, 8)
        with self.voice(N["lin4"]) as v:
            self.play(FadeOut(rows), FadeOut(ax), FadeOut(cont), FadeOut(p), FadeOut(hstep), FadeOut(hl),
                      run_time=0.5)
            self.play(FadeIn(rows2[0]), run_time=0.5)
            self.wait_until(v, "Step")
            self.play(FadeIn(rows2[1]), run_time=0.5)
            self.wait_until(v, "gradient")
            self.play(FadeIn(rows2[2]), run_time=0.6)
            self.wait_until(v, "guess")
            self.play(FadeIn(rows2[3]), run_time=0.5)
            self.wait_until(v, "true")
            self.play(FadeIn(rows2[4]), run_time=0.5)


class C5_Steepest(Studio):
    def construct(self):
        self.heading("5 · Why the gradient points uphill")
        bx, rings, labs = bowl_map(LEFT * 3.4 + DOWN * 0.6)
        P = (3.0, 4.0)
        gdir = demo.g2_exact / np.linalg.norm(demo.g2_exact)
        g0 = np.arctan2(gdir[1], gdir[0])
        ga = arr(bx, P, (P[0] + 0.25 * demo.g2_exact[0], P[1] + 0.25 * demo.g2_exact[1]), YELLOW, 7)
        gl = MathTex(r"\nabla f", color=YELLOW, font_size=34).next_to(ga.get_end(), UP, buff=0.05)
        th = ValueTracker(70.0)

        def u_arrow():
            a = g0 + np.radians(th.get_value())
            return arr(bx, P, (P[0] + 1.6 * np.cos(a), P[1] + 1.6 * np.sin(a)), GREEN, 6)
        ua = always_redraw(u_arrow)
        f = MathTex(r"\text{change}", r"\approx", r"\nabla f\cdot u", "=", r"\|\nabla f\|\,\|u\|\cos\theta",
                    font_size=44)
        f[2].set_color(GREEN); f[4].set_color(YELLOW)
        f.move_to(RIGHT * 3.4 + UP * 2.2)
        fit(f, 6.8)
        rem = self.reminder_card("norm", "The length of a vector: ‖x‖ = √(x·x)  (video 6).",
                                 pos=DR, width=5.0)
        read = always_redraw(lambda: VGroup(
            T(f"θ = {th.get_value():5.1f}°", 30, GREEN, font=MONO),
            T(f"slope = 10 cos θ = {10 * np.cos(np.radians(th.get_value())):+.2f}", 28, YELLOW, font=MONO)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(RIGHT * 3.4 + UP * 0.6))
        with self.voice(N["st1"]) as v:
            self.play(Create(bx), Create(rings), FadeIn(labs), run_time=1)
            self.wait_until(v, "direction")
            self.play(GrowArrow(ga), FadeIn(gl), run_time=0.7)
            self.wait_until(v, "step")
            self.add(ua)
            self.wait_until(v, "change")
            self.play(Write(f[:3]), run_time=0.8)
            self.wait_until(v, "lengths")
            self.play(Write(f[3:]), FadeIn(rem, shift=UP * 0.3), run_time=1)
        with self.voice(N["st2"]) as v:
            self.play(FadeOut(rem), run_time=0.4)
            self.add(read)
            self.wait_until(v, "turn")
            self.play(th.animate.set_value(140), run_time=1.5)
            self.wait_until(v, "along")
            self.play(th.animate.set_value(0), run_time=2)
            self.wait_until(v, "steepest")
            self.play(Indicate(ga, color=YELLOW), run_time=0.8)
        circ5 = Circle(radius=5 * unit(bx), color=PURPLE, stroke_width=5).move_to(bx.c2p(0, 0))
        perp = T("along the contour: no change", 26, PURPLE).move_to(RIGHT * 3.4 + DOWN * 0.8)
        perp2 = T("∇f ⟂ contour lines", 30, PURPLE, weight=BOLD).next_to(perp, DOWN, buff=0.25)
        down = T("−∇f: steepest downhill", 28, RED).next_to(perp2, DOWN, buff=0.35)
        with self.voice(N["st3"]) as v:
            self.wait_until(v, "ninety")
            self.play(th.animate.set_value(90), run_time=1.5)
            self.wait_until(v, "contour")
            self.play(Create(circ5), FadeIn(perp), run_time=1)
            self.wait_until(v, "perpendicular")
            self.play(FadeIn(perp2), run_time=0.6)
            self.wait_until(v, "hundred")
            self.play(th.animate.set_value(180), run_time=1.5)
            self.wait_until(v, "Minus", 2)
            self.play(FadeIn(down), run_time=0.6)
        pts = [np.array(P)]
        for _ in range(5):
            pts.append(pts[-1] - 0.1 * 2 * pts[-1])
        path = VGroup(*[Arrow(bx.c2p(*a), bx.c2p(*b), buff=0, color=RED, stroke_width=4,
                              max_tip_length_to_length_ratio=0.3) for a, b in zip(pts[:-1], pts[1:])])
        gd = VGroup(MathTex(r"x\ \leftarrow\ x-\eta\,\nabla f", font_size=48),
                    T("gradient descent (a molecule): preview only", 22, MUTED),
                    T("η = step size (here 0.1)", 22, PURPLE)).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        gd.move_to(RIGHT * 3.4 + DOWN * 1.2)
        with self.voice(N["st4"]) as v:
            self.play(FadeOut(VGroup(perp, perp2, down, circ5, read, f)), run_time=0.5)
            self.remove(ua)
            self.wait_until(v, "descent")
            self.play(Write(gd[0]), FadeIn(gd[1]), run_time=1)
            self.wait_until(v, "again")
            self.play(LaggedStart(*[GrowArrow(a) for a in path], lag_ratio=0.5), FadeIn(gd[2]), run_time=2)
            self.wait_until(v, "preview")
            self.play(Indicate(gd[1], color=YELLOW), run_time=0.8)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · The gradient in machine learning")
        L = MathTex(r"L(w_1,\,w_2,\,\dots,\,w_n)", font_size=60).move_to(UP * 1.8)
        L[0].set_color(RED)
        G = MathTex(r"\nabla L=\Big[\tfrac{\partial L}{\partial w_1},\ \tfrac{\partial L}{\partial w_2},\ \dots,\ "
                    r"\tfrac{\partial L}{\partial w_n}\Big]", color=YELLOW, font_size=56).next_to(L, DOWN, buff=0.6)
        gpt = VGroup(T("small GPT-2: about 124 million weights", 30, PURPLE),
                     T("→ about 124 million partial derivatives", 30, YELLOW)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(G, DOWN, buff=0.6)
        with self.voice(N["ml1"]) as v:
            self.wait_until(v, "loss")
            self.play(Write(L), run_time=1)
            self.wait_until(v, "GPT")
            self.play(FadeIn(gpt[0]), run_time=0.6)
            self.wait_until(v, "gradient")
            self.play(Write(G), run_time=1.2)
            self.wait_until(v, "partial")
            self.play(FadeIn(gpt[1]), run_time=0.6)
        self.clear_body()

        def grid(color, label):
            sqs = VGroup(*[Square(0.7, color=color, fill_color=color, fill_opacity=0.25) for _ in range(9)])
            sqs.arrange_in_grid(3, 3, buff=0.08)
            return VGroup(T(label, 26, color, font=MONO), sqs).arrange(DOWN, buff=0.3)
        W = grid(PURPLE, "W  (3×3 weights)")
        Wg = grid(YELLOW, "W.grad  (3×3)")
        VGroup(W, Wg).arrange(RIGHT, buff=2.5).shift(UP * 0.3)
        ar = Arrow(W.get_right(), Wg.get_left(), buff=0.3, color=MUTED)
        q = T("“nudge only this weight: how fast does the loss change?”", 26, FG).to_edge(DOWN, buff=0.9)
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "nudge")
            self.play(FadeIn(W), FadeIn(q), run_time=0.8)
            self.wait_until(v, "collects")
            self.play(GrowArrow(ar), FadeIn(Wg), run_time=1)
            self.wait_until(v, "shape")
            self.play(Indicate(W[1], color=PURPLE), Indicate(Wg[1], color=YELLOW), run_time=0.8)
        self.clear_body()
        bx, rings, labs = bowl_map(LEFT * 3.4 + DOWN * 0.6)
        a = Dot(bx.c2p(3, 4), color=FG, radius=0.09)
        b = Dot(bx.c2p(*demo.step), color=GREEN, radius=0.1)
        st = Arrow(bx.c2p(3, 4), bx.c2p(*demo.step), buff=0, color=RED, stroke_width=6)
        rows = VGroup(MathTex(r"x=(3,4),\ \ f=%g" % demo.f_before, font_size=44),
                      MathTex(r"\nabla f=(6,8)", color=YELLOW, font_size=44),
                      MathTex(r"x-0.1\,\nabla f=(%.1f,\,%.1f)" % tuple(demo.step), color=RED, font_size=44),
                      MathTex(r"f:\ %g\ \to\ %g" % (demo.f_before, demo.f_after), color=GREEN, font_size=50),
                      T("toy example: the bowl x² + y²", 22, MUTED))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.3 + DOWN * 0.3)
        with self.voice(N["ml3"]) as v:
            self.play(Create(bx), Create(rings), FadeIn(labs), FadeIn(rows[4]), run_time=0.8)
            self.wait_until(v, "Start")
            self.play(FadeIn(a), FadeIn(rows[0]), run_time=0.5)
            self.wait_until(v, "six")
            self.play(FadeIn(rows[1]), run_time=0.5)
            self.wait_until(v, "against")
            self.play(GrowArrow(st), FadeIn(b), FadeIn(rows[2]), run_time=0.8)
            self.wait_until(v, "drops")
            self.play(FadeIn(rows[3]), run_time=0.6)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        q = VGroup(MathTex(r"f(x,y)=3x+y^2", color=BLUE, font_size=64),
                   MathTex(r"\frac{\partial f}{\partial x}=?\qquad\frac{\partial f}{\partial y}=?\qquad"
                           r"\nabla f(2,1)=?", font_size=50)).arrange(DOWN, buff=0.5).shift(UP * 1.3)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac1"]) as v:
            self.wait_until(v, "three")
            self.play(Write(q[0]), run_time=1)
            self.wait_until(v, "partial")
            self.play(Write(q[1]), run_time=1)
            self.wait_until(v, "Remember")
            self.play(FadeIn(pause), run_time=0.5)
        a = VGroup(MathTex(r"\frac{\partial f}{\partial x}=3", color=ORANGE, font_size=48),
                   MathTex(r"\frac{\partial f}{\partial y}=2y", color=GREEN, font_size=48),
                   MathTex(r"\nabla f(2,1)=\begin{bmatrix}3\\2\end{bmatrix}", color=YELLOW, font_size=48)
                   ).arrange(RIGHT, buff=1.0).next_to(q, DOWN, buff=0.6)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "three")
            self.play(Write(a[0]), run_time=0.7)
            self.wait_until(v, "two")
            self.play(Write(a[1]), run_time=0.7)
            self.wait_until(v, "gradient")
            self.play(Write(a[2]), run_time=0.8)
        self.clear_body()
        bx, rings, labs = bowl_map(LEFT * 3.4 + DOWN * 0.6)
        p = Dot(bx.c2p(0, 5), color=YELLOW, radius=0.09)
        pl = MathTex("(0,5)", color=YELLOW, font_size=30).next_to(p, UL, buff=0.05)
        qq = VGroup(T("bowl  f = x² + y²  at (0, 5):", 28, BLUE),
                    T("1) gradient?", 28, FG), T("2) steepest way up?", 28, FG),
                    T("3) direction with no change?", 28, FG), T("⏸  pause and think", 26, MUTED)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.3 + UP * 0.6)
        with self.voice(N["prac3"]) as v:
            self.play(Create(bx), Create(rings), FadeIn(labs), FadeIn(p), FadeIn(pl), FadeIn(qq[0]), run_time=1)
            self.wait_until(v, "gradient")
            self.play(FadeIn(qq[1]), run_time=0.4)
            self.wait_until(v, "steepest")
            self.play(FadeIn(qq[2]), run_time=0.4)
            self.wait_until(v, "change")
            self.play(FadeIn(qq[3]), run_time=0.4)
            self.wait_until(v, "Pause")
            self.play(FadeIn(qq[4]), run_time=0.4)
        up = arr(bx, (0, 5), (0, 6.3), YELLOW, 7)
        side = arr(bx, (0, 5), (1.6, 5), PURPLE, 6)
        ans = VGroup(MathTex(r"\nabla f=(2x,\,2y)=(0,\,10)", color=YELLOW, font_size=42),
                     T("up: the y direction, slope 10", 26, YELLOW),
                     T("no change: the x direction (along the circle)", 24, PURPLE),
                     T("⟂ to the gradient", 24, PURPLE)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        ans.move_to(RIGHT * 3.3 + DOWN * 1.9)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(qq[1:]), run_time=0.4)
            self.wait_until(v, "ten")
            self.play(Write(ans[0]), run_time=0.8)
            self.wait_until(v, "steepest")
            self.play(GrowArrow(up), FadeIn(ans[1]), run_time=0.7)
            self.wait_until(v, "x", 2)
            self.play(GrowArrow(side), FadeIn(ans[2]), run_time=0.7)
            self.wait_until(v, "perpendicular")
            self.play(FadeIn(ans[3]), run_time=0.5)


CODE = '''import numpy as np
def num_grad(f, p, h=1e-6):
    g = np.zeros_like(p)
    for i in range(len(p)):                  # one input at a time
        step = np.zeros_like(p); step[i] = h  # nudge it, freeze the others
        g[i] = (f(p + step) - f(p - step)) / (2 * h)
    return g
f1 = lambda p: p[0]**2 + 3 * p[1]
f2 = lambda p: p[0]**2 + p[1]**2
print(num_grad(f1, np.array([1.0, 2.0])))    # exact: [2 3]
print(num_grad(f2, np.array([3.0, 4.0])))    # exact: [6 8]
p = np.array([3.0, 4.0])
print(f2(p), "->", f2(p - 0.1 * num_grad(f2, p)))   # one step downhill'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=24)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[4], color=YELLOW, buff=0.06)
            self.wait_until(v, "nudge")
            self.play(Create(box), run_time=0.5)
            for i, w in [(5, "divide"), (9, "exact"), (12, "downhill")]:
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
            for i, w in [(0, "gradients"), (2, "guesses"), (4, "ten"), (6, "ninety"), (9, "sixteen")]:
                self.wait_until(v, w)
                self.play(Indicate(term[1][i], color=YELLOW), run_time=0.6)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["Two inputs → a landscape; a contour map shows it flat",
                            "Partial derivative: slope when one input moves, others frozen",
                            "Gradient ∇f = all partial derivatives in one vector",
                            "Close up: f(x + h) ≈ f(x) + ∇f · h  (axiom C + inner product)",
                            "∇f points steepest uphill, ⟂ to contour lines; −∇f goes downhill",
                            "In ML: one partial derivative per weight"],
                           size=28).shift(UP * 0.5)
        fit(pts, 13)
        with self.voice(N["sum1"]) as v:
            for p in pts[:4]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 6)
        nxt = VGroup(T("Next · Video 9", 26, MUTED),
                     T("Chain rule — slopes multiply", 32, BLUE, weight=BOLD)
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


SCENES = ["C0_Intro", "C1_Map", "C2_Partial", "C3_Gradient", "C4_Linear", "C5_Steepest",
          "C6_ML", "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Map": "A map of heights",
                  "C2_Partial": "Partial derivatives", "C3_Gradient": "The gradient",
                  "C4_Linear": "Local linearity with many inputs",
                  "C5_Steepest": "Why the gradient points uphill",
                  "C6_ML": "The gradient in ML", "C7_Practice": "Try it yourself",
                  "C7_Code": "Python demo", "C8_Summary": "Summary", "C9_Credits": "Credits"}
