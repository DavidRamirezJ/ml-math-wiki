"""Derivative — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 7 of 16"
TITLE = "Derivative — the slope right here"


def axes(xr, yr, xl, yl, ticks=True):
    return Axes(x_range=xr, y_range=yr, x_length=xl, y_length=yl, tips=False,
                axis_config={"stroke_color": MUTED, "include_ticks": ticks})


def fit(m, w):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def legend(rows, x_sym=-3.4, x_txt=-2.4, y0=0.2, dy=0.75, size=26, fs=44):
    g = VGroup()
    for i, (sym, txt, color) in enumerate(rows):
        s = MathTex(sym, color=color, font_size=fs).move_to([x_sym, y0 - i * dy, 0])
        t = T(txt, size, FG)
        t.move_to([x_txt, y0 - i * dy, 0], aligned_edge=LEFT)
        g.add(VGroup(s, t))
    return g


def line_through(ax, x0, y0, m, xa, xb, color, width=4):
    return Line(ax.c2p(xa, y0 + m * (xa - x0)), ax.c2p(xb, y0 + m * (xb - x0)),
                color=color, stroke_width=width)


def sq(x):
    return x * x


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        fit(title, 12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.2)
        w = VGroup(T("weight", 30, PURPLE, weight=BOLD), T("w → w + 0.01", 32, PURPLE, font=MONO)
                   ).arrange(DOWN, buff=0.25).move_to(LEFT * 4.6 + DOWN * 0.7)
        e = VGroup(T("error", 30, RED, weight=BOLD), T("changes by ?", 32, RED, font=MONO)
                   ).arrange(DOWN, buff=0.25).move_to(RIGHT * 0.2 + DOWN * 0.7)
        a1 = Arrow(w.get_right(), e.get_left(), buff=0.3, color=MUTED)
        d = MathTex(r"\frac{df}{dx}", color=YELLOW, font_size=110).move_to(RIGHT * 4.9 + DOWN * 0.7)
        a2 = Arrow(e.get_right(), d.get_left(), buff=0.3, color=MUTED)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            self.wait_until(v, "nudge")
            self.play(FadeIn(w, shift=RIGHT * 0.2), run_time=0.7)
            self.wait_until(v, "error")
            self.play(GrowArrow(a1), FadeIn(e, shift=RIGHT * 0.2), run_time=0.8)
            self.wait_until(v, "derivative")
            self.play(GrowArrow(a2), Write(d), run_time=1)
        items = VGroup(*[VGroup(MathTex(s, color=c, font_size=46), T(t, 24, MUTED)).arrange(DOWN, buff=0.3)
                         for s, c, t in [(r"\text{slope}", BLUE, "at one point"),
                                         (r"\lim_{h\to0}", ORANGE, "limit"),
                                         (r"\approx\text{line}", YELLOW, "close up"),
                                         (r"n\,x^{n-1}", PURPLE, "rules"),
                                         (r"f'=0", GREEN, "lowest point")]])
        items.arrange(RIGHT, buff=0.7).shift(DOWN * 0.5)
        fit(items, 13)
        pre = T("you need: school algebra (functions, graphs, x²)", 26, MUTED).to_edge(DOWN, buff=0.7)
        with self.voice(N["intro2"]) as v:
            self.play(*[FadeOut(m) for m in (w, e, a1, a2, d)], run_time=0.6)
            for it, (wd, n) in zip(items, [("slope", 1), ("limit", 1), ("straight", 1),
                                         ("rules", 1), ("lowest", 1)]):
                self.wait_until(v, wd, n)
                self.play(FadeIn(it, shift=UP * 0.2), run_time=0.5)
            self.wait_until(v, "need")
            self.play(FadeIn(pre), run_time=0.6)


class C1_Average(Studio):
    def construct(self):
        self.heading("1 · From average slope to the tangent")
        ax = axes([0, 4.2, 1], [0, 8.5, 1], 5.5, 5).shift(LEFT * 3.3 + DOWN * 0.7)
        ln = ax.plot(lambda x: 2 * x, x_range=[0, 4.1], color=BLUE)
        run = Line(ax.c2p(1, 2), ax.c2p(2, 2), color=ORANGE, stroke_width=5)
        rise = Line(ax.c2p(2, 2), ax.c2p(2, 4), color=ORANGE, stroke_width=5)
        rl = T("run 1", 22, ORANGE).next_to(run, DOWN, buff=0.1)
        sl = T("rise 2", 22, ORANGE).next_to(rise, RIGHT, buff=0.1)
        f = MathTex(r"\text{slope}", "=", r"\frac{\text{rise}}{\text{run}}", "=", r"\frac{2}{1}", "=", "2",
                    font_size=52).move_to(RIGHT * 3.4 + UP * 0.6)
        f[6].set_color(YELLOW)
        same = T("same slope everywhere", 26, MUTED).next_to(f, DOWN, buff=0.5)
        run2 = Line(ax.c2p(3, 6), ax.c2p(3.5, 6), color=ORANGE, stroke_width=5)
        rise2 = Line(ax.c2p(3.5, 6), ax.c2p(3.5, 7), color=ORANGE, stroke_width=5)
        with self.voice(N["avg1"]) as v:
            self.play(Create(ax), Create(ln), run_time=1.2)
            self.wait_until(v, "rise")
            self.play(Write(f[:3]), run_time=0.8)
            self.wait_until(v, "two")
            self.play(Create(run), Create(rise), FadeIn(rl), FadeIn(sl), run_time=0.8)
            self.wait_until(v, "slope", 3)
            self.play(Write(f[3:]), run_time=0.8)
            self.wait_until(v, "everywhere")
            self.play(Create(run2), Create(rise2), FadeIn(same), run_time=0.8)
        self.clear_body()
        ax = axes([-0.5, 4.6, 1], [-1, 21, 5], 6.2, 5.2).shift(LEFT * 3.2 + DOWN * 0.6)
        cv = ax.plot(sq, x_range=[-0.5, 4.5], color=BLUE, stroke_width=5)
        cl = MathTex("f(x)=x^2", color=BLUE, font_size=44).next_to(ax.c2p(4.5, 20.25), LEFT, buff=0.3)
        ticks = VGroup(*[T(str(k), 20, MUTED).next_to(ax.c2p(k, 0), DOWN, buff=0.12) for k in (1, 2, 3, 4)])
        p = Dot(ax.c2p(3, 9), color=YELLOW, radius=0.1)
        pl = MathTex("x=3", color=YELLOW, font_size=36).next_to(p, LEFT, buff=0.15)
        flat = T("flat", 24, GREEN).next_to(ax.c2p(0, 0), UP, buff=0.35)
        steep = T("steep", 24, RED).next_to(ax.c2p(4.3, 18.5), RIGHT, buff=0.15)
        qn = T("slope at one point = ?", 30, YELLOW).move_to(RIGHT * 3.6 + UP * 1.2)
        with self.voice(N["avg2"]) as v:
            self.play(Create(ax), FadeIn(ticks), run_time=0.8)
            self.wait_until(v, "squared")
            self.play(Create(cv), FadeIn(cl), run_time=1.2)
            self.wait_until(v, "flat")
            self.play(FadeIn(flat), run_time=0.5)
            self.wait_until(v, "steep")
            self.play(FadeIn(steep), run_time=0.5)
            self.wait_until(v, "point")
            self.play(FadeIn(p), FadeIn(pl), run_time=0.6)
            self.wait_until(v, "sense")
            self.play(FadeIn(qn), run_time=0.6)
        q = Dot(ax.c2p(4, 16), color=GREEN, radius=0.1)
        sec = line_through(ax, 3, 9, 7, 1.8, 4.4, ORANGE)
        run = DashedLine(ax.c2p(3, 9), ax.c2p(4, 9), color=ORANGE)
        rise = DashedLine(ax.c2p(4, 9), ax.c2p(4, 16), color=ORANGE)
        hl = MathTex("h", color=ORANGE, font_size=34).next_to(run, DOWN, buff=0.08)
        fm = MathTex(r"\frac{16-9}{1}", "=", "%g" % demo.ratios[0], font_size=56)
        fm[2].set_color(ORANGE)
        fl = T("average slope, h = 1", 26, ORANGE)
        g = VGroup(fl, fm).arrange(DOWN, buff=0.35).move_to(RIGHT * 3.6 + DOWN * 0.4)
        with self.voice(N["avg3"]) as v:
            self.play(FadeOut(qn), run_time=0.4)
            self.wait_until(v, "second")
            self.play(FadeIn(q), Create(run), FadeIn(hl), run_time=0.8)
            self.wait_until(v, "straight")
            self.play(Create(sec), run_time=0.8)
            self.wait_until(v, "average")
            self.play(FadeIn(fl), run_time=0.5)
            self.wait_until(v, "rise")
            self.play(Create(rise), Write(fm[0]), run_time=1)
            self.wait_until(v, "seven", 2)
            self.play(Write(fm[1:]), run_time=0.6)
        h = ValueTracker(1.0)

        def m():
            return (sq(3 + h.get_value()) - 9) / h.get_value()
        sec_l = always_redraw(lambda: line_through(ax, 3, 9, m(), 1.8, 4.4, ORANGE))
        q_l = always_redraw(lambda: Dot(ax.c2p(3 + h.get_value(), sq(3 + h.get_value())),
                                        color=GREEN, radius=0.1))
        read = always_redraw(lambda: VGroup(
            T(f"h = {h.get_value():.2f}", 32, ORANGE, font=MONO),
            T(f"slope = {m():.2f}", 32, ORANGE, font=MONO)).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            .move_to(RIGHT * 3.6 + DOWN * 0.4))
        tan = line_through(ax, 3, 9, 6, 1.8, 4.4, YELLOW, 6)
        tl = T("tangent line: slope 6", 28, YELLOW).move_to(RIGHT * 3.6 + DOWN * 2.2)
        with self.voice(N["avg4"]) as v:
            self.remove(sec, q, run, rise, hl)
            self.play(FadeOut(g), run_time=0.3)
            self.add(sec_l, q_l, read)
            self.wait_until(v, "shrink")
            self.play(h.animate.set_value(0.01), run_time=v.duration * 0.4)
            self.wait_until(v, "tangent")
            self.play(Create(tan), FadeIn(tl), run_time=1)
            self.wait_until(v, "here")
            self.play(Indicate(p, color=YELLOW), run_time=0.8)


class C2_Limit(Studio):
    def construct(self):
        self.heading("2 · The derivative as a limit")
        r = MathTex(r"\frac{f(x+h)-f(x)}{h}", font_size=80).move_to(UP * 1.4)
        leg = legend([(r"f(x+h)-f(x)", "the rise: change in the output", YELLOW),
                      (r"h", "the run: change in the input", BLUE)],
                     x_sym=-3.0, x_txt=-1.2, y0=-0.4, dy=0.9, size=26)
        h0 = MathTex(r"h \to 0", color=ORANGE, font_size=60).next_to(r, RIGHT, buff=0.8)
        card = self.term_card("limit", "The value that an expression gets closer and closer "
                              "to as h approaches a number (here 0).", pos=DR, width=5.4)
        with self.voice(N["lim1"]) as v:
            self.wait_until(v, "formula")
            self.play(Write(r), run_time=1.2)
            self.wait_until(v, "rise")
            self.play(FadeIn(leg[0]), run_time=0.5)
            self.wait_until(v, "bottom")
            self.play(FadeIn(leg[1]), run_time=0.5)
            self.wait_until(v, "closer")
            self.play(Write(h0), run_time=0.7)
            self.wait_until(v, "limit")
            self.play(FadeOut(leg), FadeIn(card, shift=UP * 0.3), run_time=0.7)
        d = MathTex(r"f'(x)", "=", r"\lim_{h\to 0}", r"\frac{f(x+h)-f(x)}{h}", font_size=70)
        d[0].set_color(YELLOW); d[2].set_color(ORANGE)
        d.move_to(UP * 1.4)
        alt = MathTex(r"f'(x)", "=", r"\frac{df}{dx}", font_size=56).next_to(d, DOWN, buff=0.5)
        alt[0].set_color(YELLOW); alt[2].set_color(YELLOW)
        alt.shift(LEFT * 3.6)
        card2 = self.term_card("derivative", "The slope of a function at one point: the limit "
                               "of rise over run as the run shrinks to 0.", pos=DR, width=5.4)
        words = T("change in output ÷ change in input,\nfor a tiny change", 22, MUTED)
        words.next_to(alt, DOWN, buff=0.5).align_to(alt, LEFT)
        with self.voice(N["lim2"]) as v:
            self.play(FadeOut(card), ReplacementTransform(VGroup(r, h0), d), run_time=1)
            self.wait_until(v, "derivative")
            self.play(FadeIn(card2, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "prime")
            self.play(Indicate(d[0], color=YELLOW), run_time=0.6)
            self.wait_until(v, "d")
            self.play(Write(alt), run_time=0.8)
            self.wait_until(v, "change")
            self.play(FadeIn(words), run_time=0.6)
        self.clear_body()
        top = MathTex(r"f(x)=x^2,\quad x=3", color=BLUE, font_size=48).move_to(UP * 2.3)
        rows = VGroup()
        for hv, rv, num in [(1, demo.ratios[0], r"\frac{16-9}{1}"),
                            (0.1, demo.ratios[1], r"\frac{9.61-9}{0.1}"),
                            (0.01, demo.ratios[2], r"\frac{9.0601-9}{0.01}")]:
            row = MathTex(r"h=%g" % hv, r"\quad", num, "=", "%g" % round(rv, 4), font_size=46)
            row[0].set_color(ORANGE); row[4].set_color(YELLOW)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(LEFT * 1.5 + DOWN * 0.3)
        with self.voice(N["lim3"]) as v:
            self.play(Write(top), run_time=0.8)
            self.wait_until(v, "seven")
            self.play(FadeIn(rows[0]), run_time=0.7)
            self.wait_until(v, "Subtract")
            self.play(FadeIn(rows[1]), run_time=0.7)
            self.wait_until(v, "zero", 3)
            self.play(FadeIn(rows[2]), run_time=0.7)
        to6 = VGroup(MathTex(r"\longrightarrow", color=YELLOW, font_size=70),
                     MathTex("6", color=YELLOW, font_size=96)).arrange(RIGHT, buff=0.3)
        to6.next_to(rows, RIGHT, buff=0.8)
        bad = MathTex(r"h=0:\ \ \frac{9-9}{0}=\frac{0}{0}\ \ ", r"\text{(meaningless)}", color=RED,
                      font_size=42).to_edge(DOWN, buff=0.6)
        with self.voice(N["lim4"]) as v:
            self.wait_until(v, "walking")
            self.play(FadeIn(to6, shift=RIGHT * 0.3), run_time=0.8)
            self.wait_until(v, "mistake")
            self.play(Write(bad), run_time=1)
            self.wait_until(v, "heading")
            self.play(Indicate(to6, color=YELLOW), run_time=0.8)
        self.clear_body()
        steps = VGroup(
            MathTex(r"\frac{(x+h)^2-x^2}{h}", font_size=50),
            MathTex(r"=\frac{x^2+2xh+h^2-x^2}{h}", font_size=50),
            MathTex(r"=\frac{2xh+h^2}{h}", "=", r"2x+h", font_size=50),
            MathTex(r"\xrightarrow{\;h\to0\;}", r"2x", font_size=56))
        steps[2][2].set_color(ORANGE); steps[3][1].set_color(YELLOW)
        steps.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(LEFT * 1.8 + DOWN * 0.3)
        res = VGroup(MathTex(r"(x^2)'=2x", color=YELLOW, font_size=64),
                     MathTex(r"x=3:\ \ 2\cdot3=%g" % demo.exact, font_size=48)
                     ).arrange(DOWN, buff=0.4).move_to(RIGHT * 4.3 + DOWN * 0.3)
        with self.voice(N["lim5"]) as v:
            self.play(Write(steps[0]), run_time=0.8)
            self.wait_until(v, "Expand")
            self.play(Write(steps[1]), run_time=1)
            self.wait_until(v, "left")
            self.play(Write(steps[2]), run_time=1)
            self.wait_until(v, "becomes")
            self.play(Write(steps[3]), run_time=0.8)
            self.wait_until(v, "six")
            self.play(Write(res[1]), run_time=0.8)
            self.wait_until(v, "derivative")
            self.play(Write(res[0]), run_time=0.8)


class C3_Local(Studio):
    def construct(self):
        self.heading("3 · Close up, every curve is straight")
        w = ValueTracker(2.0)
        box = Rectangle(width=6.4, height=5.0, color=MUTED, stroke_width=1.5).move_to(LEFT * 2.8 + DOWN * 0.6)

        def view():
            ww = w.get_value()
            ax = Axes(x_range=[3 - ww, 3 + ww, ww], y_range=[9 - 8 * ww, 9 + 8 * ww, 4 * ww],
                      x_length=6.0, y_length=4.6, tips=False,
                      axis_config={"stroke_color": MUTED, "stroke_opacity": 0.0, "include_ticks": False})
            ax.move_to(box)
            cv = ax.plot(sq, x_range=[3 - ww, 3 + ww], color=BLUE, stroke_width=6)
            dot = Dot(ax.c2p(3, 9), color=YELLOW, radius=0.08)
            return VGroup(cv, dot)
        live = always_redraw(view)
        read = always_redraw(lambda: T(f"window: x from {3 - w.get_value():.2f} to {3 + w.get_value():.2f}",
                                       24, MUTED, font=MONO).next_to(box, DOWN, buff=0.15))
        zl = T("zoom in at x = 3", 30, YELLOW).move_to(RIGHT * 4.0 + UP * 1.4)
        st = T("→ looks straight", 30, GREEN).move_to(RIGHT * 4.0 + DOWN * 0.2)
        with self.voice(N["loc1"]) as v:
            self.play(Create(box), run_time=0.5)
            self.add(live, read)
            self.wait_until(v, "Zoom")
            self.play(FadeIn(zl), w.animate.set_value(0.6), run_time=2)
            self.wait_until(v, "more")
            self.play(w.animate.set_value(0.15), run_time=2)
            self.wait_until(v, "less")
            self.play(w.animate.set_value(0.03), run_time=2)
            self.wait_until(v, "straight")
            self.play(FadeIn(st), run_time=0.6)
        self.clear_body()
        ax = axes([1.5, 4.6, 1], [0, 21, 5], 5.6, 4.8).shift(LEFT * 3.6 + DOWN * 0.7)
        cv = ax.plot(sq, x_range=[1.5, 4.5], color=BLUE, stroke_width=5)
        tan = line_through(ax, 3, 9, 6, 1.6, 4.5, YELLOW)
        p = Dot(ax.c2p(3, 9), color=FG, radius=0.08)
        hstep = Line(ax.c2p(3, 9), ax.c2p(4, 9), color=BLUE, stroke_width=4)
        hlab = MathTex("h", color=BLUE, font_size=34).next_to(hstep, DOWN, buff=0.08)
        rise = Line(ax.c2p(4, 9), ax.c2p(4, 15), color=PURPLE, stroke_width=5)
        rlab = MathTex(r"f'(x)\,h", color=PURPLE, font_size=32).next_to(rise, RIGHT, buff=0.1)
        gap = Line(ax.c2p(4, 15), ax.c2p(4, 16), color=RED, stroke_width=5)
        glab = T("small error", 20, RED).next_to(gap, RIGHT, buff=0.1)
        f = MathTex(r"f(x+h)", r"\approx", r"f(x)", "+", r"f'(x)", r"\,h", font_size=58)
        f[0].set_color(YELLOW); f[4].set_color(PURPLE); f[5].set_color(BLUE)
        f.move_to(RIGHT * 3.3 + UP * 1.8)
        fit(f, 6.6)
        card = self.term_card("local linear approximation", "Near x, replace the curve by its "
                              "tangent line: f(x+h) ≈ f(x) + f′(x)·h.", pos=DR, width=5.6)
        leg = legend([(r"f(x+h)", "new value", YELLOW), (r"f(x)", "old value", FG),
                      (r"f'(x)", "slope", PURPLE), (r"h", "step", BLUE)],
                     x_sym=1.6, x_txt=2.7, y0=0.6, dy=0.68, size=26, fs=40)
        axc = T("Axiom C · Local linearity", 32, YELLOW, weight=BOLD).move_to(RIGHT * 3.4 + DOWN * 2.6)
        with self.voice(N["loc2"]) as v:
            self.play(Create(ax), Create(cv), FadeIn(p), run_time=1)
            self.wait_until(v, "tangent")
            self.play(Create(tan), run_time=0.7)
            self.wait_until(v, "formula")
            self.play(Write(f), Create(hstep), FadeIn(hlab), run_time=1.2)
            self.play(Create(rise), FadeIn(rlab), Create(gap), FadeIn(glab), run_time=0.8)
            self.wait_until(v, "called")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "new")
            self.play(FadeOut(card), FadeIn(leg), run_time=0.7)
            self.wait_until(v, "axiom")
            self.play(FadeIn(axc), run_time=0.6)
        c1 = VGroup(MathTex(r"f(3.1)\approx 9+6\cdot0.1=%.1f" % demo.approx_31, font_size=44),
                    MathTex(r"\text{true: }3.1^2=%.2f" % demo.true_31, font_size=44),
                    MathTex(r"\text{error: }%.2f" % (demo.true_31 - demo.approx_31), color=GREEN, font_size=44))
        c1[0].set_color(YELLOW)
        c1.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(RIGHT * 3.4 + DOWN * 0.5)
        with self.voice(N["loc3"]) as v:
            self.play(FadeOut(leg), FadeOut(axc), run_time=0.5)
            self.wait_until(v, "formula")
            self.play(Write(c1[0]), run_time=1)
            self.wait_until(v, "true")
            self.play(Write(c1[1]), run_time=0.8)
            self.wait_until(v, "wrong")
            self.play(Write(c1[2]), run_time=0.6)
        c2 = VGroup(MathTex(r"x^3 \text{ at } x=2:\ \ f'(2)=%g" % demo.cube_exact, font_size=44),
                    MathTex(r"2.1^3\approx 8+12\cdot0.1=%.1f" % demo.cube_approx, color=YELLOW, font_size=44),
                    MathTex(r"\text{true: }%.3f" % demo.cube_true, font_size=44),
                    T("smaller steps → even better", 24, GREEN))
        c2.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.4 + DOWN * 0.6)
        with self.voice(N["loc4"]) as v:
            self.play(FadeOut(c1), run_time=0.4)
            self.wait_until(v, "cubed")
            self.play(Write(c2[0]), run_time=0.8)
            self.wait_until(v, "guesses")
            self.play(Write(c2[1]), run_time=0.8)
            self.wait_until(v, "true")
            self.play(Write(c2[2]), run_time=0.6)
            self.wait_until(v, "smaller")
            self.play(FadeIn(c2[3]), run_time=0.5)
        lin = MathTex(r"h", r"\;\longmapsto\;", r"f'(x)\,h", font_size=60)
        lin[0].set_color(BLUE); lin[2].set_color(PURPLE)
        ex = VGroup(MathTex(r"h=0.1\ \mapsto\ 0.6", font_size=42),
                    MathTex(r"h=0.2\ \mapsto\ 1.2", font_size=42),
                    T("double the step → double the change", 24, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        g = VGroup(lin, ex).arrange(DOWN, buff=0.5).move_to(RIGHT * 3.4 + UP * 1.0)
        rem = self.reminder_card("linear mapping", "A function that keeps adding and scaling: "
                                 "f(a·u + b·v) = a·f(u) + b·f(v)  (video 4).", pos=DR, width=5.4)
        with self.voice(N["loc5"]) as v:
            self.play(FadeOut(c2), FadeOut(f), run_time=0.4)
            self.wait_until(v, "multiplied")
            self.play(Write(lin), run_time=0.8)
            self.wait_until(v, "Double")
            self.play(FadeIn(ex), run_time=0.8)
            self.wait_until(v, "linear")
            self.play(FadeIn(rem, shift=UP * 0.3), run_time=0.6)


class C4_Rules(Studio):
    def construct(self):
        self.heading("4 · Three quick rules")
        pr = VGroup(T("power rule", 28, YELLOW, weight=BOLD),
                    MathTex(r"\frac{d}{dx}\,x^n", "=", r"n\,x^{n-1}", font_size=60))
        pr[1][2].set_color(YELLOW)
        pr.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(LEFT * 3.4 + UP * 1.2)
        pe = VGroup(MathTex(r"x^2\ \to\ 2x", font_size=48), MathTex(r"x^3\ \to\ 3x^2", font_size=48)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(pr, DOWN, buff=0.5).align_to(pr, LEFT)
        with self.voice(N["rul1"]) as v:
            self.wait_until(v, "power")
            self.play(FadeIn(pr[0]), Write(pr[1]), run_time=1.2)
            self.wait_until(v, "squared", 1)
            self.play(FadeIn(pe[0]), run_time=0.6)
            self.wait_until(v, "cubed")
            self.play(FadeIn(pe[1]), run_time=0.6)
        cr = VGroup(T("constant factor", 28, PURPLE, weight=BOLD),
                    MathTex(r"(c\,f)'", "=", r"c\,f'", font_size=56))
        cr.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.0 + UP * 1.2)
        sr = VGroup(T("sum", 28, GREEN, weight=BOLD), MathTex(r"(f+g)'", "=", r"f'+g'", font_size=56))
        sr.arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(cr, DOWN, buff=0.6).align_to(cr, LEFT)
        with self.voice(N["rul2"]) as v:
            self.wait_until(v, "constant")
            self.play(FadeIn(cr), run_time=0.8)
            self.wait_until(v, "sums")
            self.play(FadeIn(sr), run_time=0.8)
            self.wait_until(v, "piece")
            self.play(Indicate(sr[1][2], color=GREEN), run_time=0.8)
        self.clear_body()
        w = VGroup(MathTex(r"f(x)=3x^2+5x", color=BLUE, font_size=50),
                   MathTex(r"f'(x)=3\cdot 2x+5", font_size=50),
                   MathTex(r"f'(x)=6x+5", color=YELLOW, font_size=50),
                   MathTex(r"f'(1)=6+5=11", color=YELLOW, font_size=50))
        w.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(LEFT * 3.4 + DOWN * 0.3)
        ax = axes([-0.5, 2.1, 1], [-2, 24, 5], 5, 4.8).move_to(RIGHT * 3.6 + DOWN * 0.6)
        cv = ax.plot(lambda x: 3 * x * x + 5 * x, x_range=[-0.5, 2], color=BLUE)
        tan = line_through(ax, 1, 8, 11, 0.2, 2.0, YELLOW)
        dot = Dot(ax.c2p(1, 8), color=YELLOW, radius=0.08)
        tl = T("slope 11 at x = 1", 24, YELLOW).next_to(ax.c2p(0.2, 20), RIGHT, buff=0)
        with self.voice(N["rul3"]) as v:
            self.play(Write(w[0]), Create(ax), Create(cv), run_time=1.2)
            self.wait_until(v, "gives")
            self.play(Write(w[1]), run_time=0.8)
            self.wait_until(v, "prime")
            self.play(Write(w[2]), run_time=0.8)
            self.wait_until(v, "eleven")
            self.play(Write(w[3]), Create(tan), FadeIn(dot), FadeIn(tl), run_time=1)
        self.clear_body()
        inner = VGroup(MathTex(r"g\big(f(x)\big)", color=ORANGE, font_size=80),
                       T("a function inside a function", 28, FG),
                       T("→ video 9", 30, BLUE)).arrange(DOWN, buff=0.4)
        with self.voice(N["rul4"]) as v:
            self.wait_until(v, "inside")
            self.play(Write(inner[0]), FadeIn(inner[1]), run_time=1)
            self.wait_until(v, "nine")
            self.play(FadeIn(inner[2]), run_time=0.6)


class C5_Flat(Studio):
    def construct(self):
        self.heading("5 · Flat at the bottom")
        ax = axes([-0.5, 4.5, 1], [0, 8, 1], 6.4, 4.8).shift(LEFT * 3.0 + DOWN * 0.7)
        fn = lambda x: (x - 2) ** 2 + 1
        cv = ax.plot(fn, x_range=[-0.5, 4.5], color=BLUE, stroke_width=5)
        lab = MathTex(r"f(x)=(x-2)^2+1", color=BLUE, font_size=40).next_to(ax.c2p(4.4, 7.5), LEFT, buff=0.3)
        t = ValueTracker(0.4)

        def tangent():
            x = t.get_value()
            m = 2 * (x - 2)
            c = RED if m < -0.05 else (GREEN if m > 0.05 else YELLOW)
            dx = 0.8 / np.sqrt(1 + m * m)
            return VGroup(line_through(ax, x, fn(x), m, x - dx, x + dx, c, 5),
                          Dot(ax.c2p(x, fn(x)), color=c, radius=0.08))
        tg = always_redraw(tangent)
        read = always_redraw(lambda: T(f"slope = {2 * (t.get_value() - 2):+.2f}", 32,
                                       RED if t.get_value() < 1.975 else (GREEN if t.get_value() > 2.025 else YELLOW),
                                       font=MONO).move_to(RIGHT * 4.3 + UP * 1.4))
        with self.voice(N["flat1"]) as v:
            self.play(Create(ax), Create(cv), FadeIn(lab), run_time=1)
            self.wait_until(v, "tangent")
            self.add(tg, read)
            self.wait_until(v, "left")
            self.play(t.animate.set_value(1.2), run_time=1.5)
            self.wait_until(v, "right")
            self.play(t.animate.set_value(3.6), run_time=2)
            self.wait_until(v, "bottom")
            self.play(t.animate.set_value(2.0), run_time=1.5)
        d = VGroup(MathTex(r"f'(x)=2(x-2)", font_size=46), MathTex(r"=0\iff x=2", color=YELLOW, font_size=46)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 4.3 + UP * 0.1)
        ax2 = axes([-1, 1, 1], [-1.2, 1.2, 1], 2.6, 1.6, ticks=False).move_to(RIGHT * 4.3 + DOWN * 2.2)
        hill = ax2.plot(lambda x: 1 - x * x, x_range=[-1, 1], color=PURPLE)
        top = Line(ax2.c2p(-0.5, 1), ax2.c2p(0.5, 1), color=YELLOW, stroke_width=4)
        hl = T("maximum: also flat", 22, PURPLE).next_to(ax2, UP, buff=0.1)
        with self.voice(N["flat2"]) as v:
            self.wait_until(v, "derivative", 2)
            self.play(Write(d[0]), run_time=0.8)
            self.wait_until(v, "zero", 2)
            self.play(Write(d[1]), run_time=0.8)
            self.wait_until(v, "maximum")
            self.play(Create(ax2), Create(hill), FadeIn(hl), run_time=1)
            self.play(Create(top), run_time=0.5)
        self.clear_body()
        ax3 = axes([-1.6, 1.6, 1], [-4, 4, 2], 5.6, 4.8).shift(LEFT * 3.2 + DOWN * 0.7)
        c3 = ax3.plot(lambda x: x ** 3, x_range=[-1.58, 1.58], color=BLUE, stroke_width=5)
        l3 = MathTex("x^3", color=BLUE, font_size=44).next_to(ax3.c2p(1.5, 3.4), LEFT, buff=0.2)
        fl = Line(ax3.c2p(-0.8, 0), ax3.c2p(0.8, 0), color=YELLOW, stroke_width=6)
        w1 = VGroup(T("slope 0 at x = 0", 30, YELLOW), T("but no minimum or maximum", 30, RED),
                    T("f′ = 0 only tells you where to look", 26, MUTED)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.5 + DOWN * 0.2)
        with self.voice(N["flat3"]) as v:
            self.play(Create(ax3), run_time=0.6)
            self.wait_until(v, "cubed")
            self.play(Create(c3), FadeIn(l3), run_time=1)
            self.wait_until(v, "slope")
            self.play(Create(fl), FadeIn(w1[0]), run_time=0.8)
            self.wait_until(v, "keeps")
            self.play(FadeIn(w1[1]), run_time=0.6)
            self.wait_until(v, "look")
            self.play(FadeIn(w1[2]), run_time=0.6)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · Derivatives in machine learning")
        loss = lambda w: (2 * w - 6) ** 2
        ax = axes([0, 5.5, 1], [0, 40, 10], 6.0, 4.8).shift(LEFT * 3.2 + DOWN * 0.7)
        cv = ax.plot(loss, x_range=[0, 5.5], color=BLUE, stroke_width=5)
        xl = MathTex("w", color=PURPLE, font_size=40).next_to(ax.c2p(5.5, 0), DOWN, buff=0.15)
        yl = T("loss", 24, RED).next_to(ax.c2p(0, 40), RIGHT, buff=0.15)
        ticks = VGroup(*[T(str(k), 20, MUTED).next_to(ax.c2p(k, 0), DOWN, buff=0.12) for k in (1, 2, 3, 4, 5)])
        f = VGroup(MathTex(r"L(w)=(2w-6)^2", font_size=46),
                   MathTex(r"=4w^2-24w+36", font_size=46),
                   T("toy example · one weight", 22, MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        f.move_to(RIGHT * 3.6 + UP * 1.3)
        with self.voice(N["ml1"]) as v:
            self.play(Create(ax), FadeIn(ticks), FadeIn(yl), run_time=0.8)
            self.wait_until(v, "weight")
            self.play(FadeIn(xl), FadeIn(f[2]), run_time=0.5)
            self.wait_until(v, "squared")
            self.play(Create(cv), Write(f[0]), run_time=1.2)
            self.wait_until(v, "Expanded")
            self.play(Write(f[1]), run_time=1)
        tan = line_through(ax, 1, 16, -16, 0.2, 2.0, RED, 5)
        dot = Dot(ax.c2p(1, 16), color=RED, radius=0.09)
        g = VGroup(MathTex(r"L'(w)=8w-24", font_size=46),
                   MathTex(r"L'(1)=%g" % round(demo.loss_slope, 4), color=RED, font_size=46),
                   MathTex(r"\Delta L\approx -16\cdot 0.01=-0.16", font_size=44),
                   MathTex(r"\text{true: }%.4f" % demo.loss_change, color=GREEN, font_size=44))
        g.arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(f, DOWN, buff=0.4).align_to(f, LEFT)
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "eight")
            self.play(Write(g[0]), run_time=0.8)
            self.wait_until(v, "minus", 2)
            self.play(Write(g[1]), Create(tan), FadeIn(dot), run_time=0.8)
            self.wait_until(v, "nudge")
            self.play(Write(g[2]), run_time=1)
            self.wait_until(v, "true")
            self.play(Write(g[3]), run_time=0.8)
        self.play(FadeOut(f), FadeOut(g), run_time=0.4)
        arrow = Arrow(ax.c2p(3.3, 31), ax.c2p(4.6, 31), buff=0, color=GREEN)
        al = T("increase w → loss goes down", 22, GREEN).next_to(arrow, UP, buff=0.1)
        best = Dot(ax.c2p(3, 0), color=YELLOW, radius=0.11)
        bl = Line(ax.c2p(2.3, 0), ax.c2p(3.7, 0), color=YELLOW, stroke_width=6)
        txt = VGroup(T("sign → direction", 28, FG), T("size → how fast", 28, FG),
                     T("slope 0 at w = 3: the best weight", 28, YELLOW),
                     T("millions of weights → video 8", 26, BLUE)).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        txt.move_to(RIGHT * 3.6 + DOWN * 0.3)
        with self.voice(N["ml3"]) as v:
            self.wait_until(v, "sign")
            self.play(FadeIn(txt[0]), run_time=0.5)
            self.wait_until(v, "increase")
            self.play(GrowArrow(arrow), FadeIn(al), run_time=0.8)
            self.wait_until(v, "size")
            self.play(FadeIn(txt[1]), run_time=0.5)
            self.wait_until(v, "zero")
            self.play(FadeIn(best), Create(bl), FadeIn(txt[2]), run_time=0.8)
            self.wait_until(v, "millions")
            self.play(FadeIn(txt[3]), run_time=0.6)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        q1 = MathTex(r"1)\ \ (x^2)' \text{ at } x=5 \;=\; ?", font_size=56)
        q2 = MathTex(r"2)\ \ 5.1^2 \approx\ ?\quad", r"\text{(use } f(x+h)\approx f(x)+f'(x)\,h)", font_size=46)
        q2[1].set_color(MUTED)
        qs = VGroup(q1, q2).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(UP * 1.3)
        fit(qs, 13)
        pause = T("⏸  pause and try", 30, MUTED).next_to(qs, DOWN, buff=0.5)
        with self.voice(N["prac1"]) as v:
            self.wait_until(v, "First")
            self.play(Write(q1), run_time=1)
            self.wait_until(v, "Second")
            self.play(Write(q2), run_time=1.2)
            self.wait_until(v, "calculator")
            self.play(FadeIn(pause), run_time=0.5)
        a = VGroup(MathTex(r"(x^2)'=2x\ \Rightarrow\ 2\cdot5=10", color=YELLOW, font_size=48),
                   MathTex(r"5.1^2\approx 25+10\cdot0.1=26", color=YELLOW, font_size=48),
                   MathTex(r"\text{true: }26.01", color=GREEN, font_size=44)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(qs, DOWN, buff=0.5)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "ten")
            self.play(Write(a[0]), run_time=1)
            self.wait_until(v, "guess")
            self.play(Write(a[1]), run_time=1)
            self.wait_until(v, "true")
            self.play(Write(a[2]), run_time=0.6)
        self.clear_body()
        q3 = MathTex(r"f(x)=4x^3-2x", color=BLUE, font_size=64).move_to(UP * 1.6)
        q4 = MathTex(r"f'(x)=\;?\qquad f'(1)=\;?", font_size=56).next_to(q3, DOWN, buff=0.5)
        p2 = T("⏸  pause and try", 30, MUTED).next_to(q4, DOWN, buff=0.5)
        with self.voice(N["prac3"]) as v:
            self.wait_until(v, "four")
            self.play(Write(q3), run_time=1)
            self.wait_until(v, "prime")
            self.play(Write(q4), run_time=0.8)
            self.wait_until(v, "Pause")
            self.play(FadeIn(p2), run_time=0.5)
        s = VGroup(MathTex(r"x^3\to 3x^2,\quad 4\cdot3x^2=12x^2", font_size=46),
                   MathTex(r"-2x\to -2", font_size=46),
                   MathTex(r"f'(x)=12x^2-2", color=YELLOW, font_size=50),
                   MathTex(r"f'(1)=12-2=10", color=YELLOW, font_size=50)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(q4, DOWN, buff=0.4)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(p2), run_time=0.4)
            self.wait_until(v, "power")
            self.play(Write(s[0]), run_time=1)
            self.wait_until(v, "minus")
            self.play(Write(s[1]), run_time=0.6)
            self.wait_until(v, "So")
            self.play(Write(s[2]), run_time=0.8)
            self.wait_until(v, "ten")
            self.play(Write(s[3]), run_time=0.6)


CODE = '''import numpy as np
def f(x):
    return x ** 2
def ratio(fun, x, h):                 # average slope: rise / run
    return (fun(x + h) - fun(x)) / h
for h in np.array([1.0, 0.1, 0.01, 0.001]):
    print(h, ratio(f, 3.0, h))        # smaller and smaller steps
print("exact:", 2 * 3.0)
print("guess f(3.1):", f(3.0) + 6 * 0.1, "true:", f(3.1))
cube = lambda x: x ** 3
print(ratio(cube, 2.0, 0.01), "vs exact", 3 * 2.0 ** 2)
loss = lambda w: (2 * w - 6) ** 2     # toy loss, one weight
print("loss slope at w=1:", ratio(loss, 1.0, 1e-6))'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=24)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[3], color=YELLOW, buff=0.06)
            self.wait_until(v, "ratio")
            self.play(Create(box), run_time=0.5)
            for i, w in [(6, "smaller"), (8, "guess"), (10, "cubed"), (12, "loss")]:
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
            for i, w in [(0, "seven"), (4, "exact"), (5, "guess"), (8, "toy")]:
                self.wait_until(v, w)
                self.play(Indicate(term[1][i], color=YELLOW), run_time=0.6)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["Derivative = the slope of a curve at one point",
                            "f′(x) = limit of (f(x+h) − f(x)) / h as h → 0",
                            "Example: the derivative of x² is 2x",
                            "Close up, curves look straight: f(x+h) ≈ f(x) + f′(x)·h",
                            "Rules: power n·xⁿ⁻¹, constant factor, sum",
                            "f′ = 0 at a minimum or maximum; in ML: loss change per weight"],
                           size=28).shift(UP * 0.5)
        fit(pts, 13)
        with self.voice(N["sum1"]) as v:
            for p in pts[:4]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 6)
        nxt = VGroup(T("Next · Video 8", 26, MUTED),
                     T("Gradient — the slope in many directions", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "minimum")
            self.play(FadeIn(pts[5], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Average", "C2_Limit", "C3_Local", "C4_Rules", "C5_Flat",
          "C6_ML", "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Average": "From average slope to the tangent",
                  "C2_Limit": "The derivative as a limit", "C3_Local": "Local linearity",
                  "C4_Rules": "Three quick rules", "C5_Flat": "Flat at the bottom",
                  "C6_ML": "Derivatives in ML", "C7_Practice": "Try it yourself",
                  "C7_Code": "Python demo", "C8_Summary": "Summary", "C9_Credits": "Credits"}
