"""Loss function — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 15 of 16"
TITLE = "Loss function — one number for how wrong you are"


def fmt(v):
    return f"{v:g}"


def axes(xr, yr, xl, yl, nums=True):
    return Axes(x_range=xr, y_range=yr, x_length=xl, y_length=yl, tips=False,
                axis_config={"stroke_color": MUTED, "include_numbers": nums, "font_size": 22})


def mse(w, x=demo.x, y=demo.y):
    return float(np.mean((y - w * x) ** 2))


def row(label, vals, color, lw=1.9, cw=0.95, size=36):
    lab = MathTex(label, color=color, font_size=size)
    cells = VGroup(*[MathTex(v, color=color, font_size=size) for v in vals])
    for i, c in enumerate(cells):
        c.move_to(RIGHT * (lw + i * cw))
    lab.align_to(ORIGIN, RIGHT).shift(RIGHT * (lw - 0.5))
    return VGroup(lab, cells)


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        if title.width > 12.5:
            title.scale_to_fit_width(12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.4)
        model = RoundedRectangle(corner_radius=0.15, width=2.6, height=1.3, color=BLUE).shift(LEFT * 4 + DOWN * 0.8)
        ml = T("model", 30, BLUE, weight=BOLD).move_to(model)
        ar = Arrow(model.get_right(), model.get_right() + RIGHT * 2.2, buff=0.1, color=MUTED)
        lt = T("loss", 30, RED, weight=BOLD)
        val = DecimalNumber(10.99, num_decimal_places=2, font_size=72, color=RED)
        box = VGroup(lt, val).arrange(DOWN, buff=0.2).next_to(ar, RIGHT, buff=0.4)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            self.wait_until(v, "score")
            self.play(Create(model), FadeIn(ml), GrowArrow(ar), run_time=1)
            self.wait_until(v, "single")
            self.play(FadeIn(box), run_time=0.8)
            self.wait_until(v, "smaller")
            self.play(Indicate(val, color=YELLOW), run_time=1)
        src = T("GPT on its training text · numbers from the book (Raschka, Build a LLM)", 20,
                MUTED).to_edge(DOWN, buff=0.4)
        ep = T("start", 24, MUTED).next_to(box, RIGHT, buff=0.5)
        plan = VGroup(T("you will build a loss by hand", 28, YELLOW),
                      T("you need: video 13 (mean)", 24, MUTED)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(RIGHT * 0.5 + DOWN * 2.6)
        with self.voice(N["intro2"]) as v:
            self.wait_until(v, "eleven")
            self.play(FadeIn(src), FadeIn(ep), run_time=0.6)
            self.wait_until(v, "passes")
            self.play(ChangeDecimalToValue(val, 0.391, run_time=2.5),
                      Transform(ep, T("after 10 passes", 24, MUTED).next_to(box, RIGHT, buff=0.5)))
            self.wait_until(v, "hand")
            self.play(FadeOut(src), FadeIn(plan[0]), run_time=0.6)
            self.wait_until(v, "thirteen")
            self.play(FadeIn(plan[1]), run_time=0.5)


class C1_OnePrediction(Studio):
    def construct(self):
        self.heading("1 · Scoring one guess")
        nl = NumberLine(x_range=[160, 190, 5], length=10, color=MUTED, include_numbers=True,
                        font_size=26).shift(DOWN * 0.3)
        unit = T("height in cm · toy example", 20, MUTED).next_to(nl, DOWN, buff=0.5)
        yh = Dot(nl.n2p(170), color=YELLOW, radius=0.14)
        yt = Dot(nl.n2p(180), color=GREEN, radius=0.14)
        yhl = MathTex(r"\hat y = 170", color=YELLOW, font_size=40).next_to(yh, UP, buff=0.3)
        ytl = MathTex(r"y = 180", color=GREEN, font_size=40).next_to(yt, UP, buff=0.3)
        pl = T("prediction", 22, YELLOW).next_to(yhl, UP, buff=0.15)
        tl = T("label (true value)", 22, GREEN).next_to(ytl, UP, buff=0.15)
        gap = DoubleArrow(nl.n2p(170) + DOWN * 0.0, nl.n2p(180), buff=0, color=ORANGE,
                          stroke_width=5).shift(UP * 0.0)
        q = T("how bad?  → one number", 30, RED).move_to(UP * 2.4 + RIGHT * 2.5)
        with self.voice(N["one1"]) as v:
            self.play(Create(nl), FadeIn(unit), run_time=1)
            self.wait_until(v, "guess")
            self.play(FadeIn(yh), FadeIn(yhl), run_time=0.6)
            self.wait_until(v, "fact")
            self.play(FadeIn(yt), FadeIn(ytl), run_time=0.6)
            self.wait_until(v, "label")
            self.play(FadeIn(tl), run_time=0.5)
            self.wait_until(v, "prediction")
            self.play(FadeIn(pl), run_time=0.5)
            self.wait_until(v, "rule")
            self.play(GrowFromCenter(gap), FadeIn(q), run_time=0.8)
        f = MathTex(r"\ell", "(", "y", ",", r"\hat y", ")", r"\ \ge 0", font_size=60)
        f[0].set_color(RED); f[2].set_color(GREEN); f[4].set_color(YELLOW)
        f.move_to(UP * 2.4 + RIGHT * 2.5)
        card = self.term_card("loss function", "A rule that takes the label and the prediction "
                              "and returns one number, never negative: how wrong the prediction is.",
                              pos=DR, width=5.6)
        zero = T("0 = perfect · bigger = worse", 24, MUTED).next_to(f, DOWN, buff=0.2)
        with self.voice(N["one2"]) as v:
            self.wait_until(v, "loss")
            self.play(FadeOut(q), FadeOut(VGroup(unit)), FadeIn(card, shift=UP * 0.3), run_time=0.8)
            self.wait_until(v, "never")
            self.play(Write(f), run_time=1)
            self.wait_until(v, "Zero")
            self.play(FadeIn(zero), run_time=0.6)
            self.wait_until(v, "write")
            self.play(Indicate(f, color=RED), run_time=0.8)
        sq = MathTex(r"(y-\hat y)^2", "=", r"(180-170)^2", "=", "100", font_size=44)
        sq[0].set_color(RED); sq[4].set_color(RED)
        ab = MathTex(r"|y-\hat y|", "=", r"|180-170|", "=", "10", font_size=44)
        ab[0].set_color(ORANGE); ab[4].set_color(ORANGE)
        VGroup(sq, ab).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to(DOWN * 2.6)
        sqn = T("squared error", 22, RED).next_to(sq, LEFT, buff=0.4)
        abn = T("absolute error", 22, ORANGE).next_to(ab, LEFT, buff=0.4)
        with self.voice(N["one3"]) as v:
            self.play(FadeOut(card), run_time=0.5)
            self.wait_until(v, "squared")
            self.play(Write(sq[:3]), FadeIn(sqn), run_time=1.2)
            self.wait_until(v, "hundred", 3)
            self.play(Write(sq[3:]), run_time=0.8)
            self.wait_until(v, "absolute")
            self.play(Write(ab[:3]), FadeIn(abn), run_time=1.2)
            self.wait_until(v, "just")
            self.play(Write(ab[3:]), run_time=0.6)
        self.clear_body()
        ax = axes([-3.2, 3.2, 1], [0, 9.5, 2], 5.6, 4.6).move_to(LEFT * 3.4 + DOWN * 0.5)
        gl = T("gap  y − ŷ", 20, MUTED).next_to(ax, DOWN, buff=0.15)
        csq = ax.plot(lambda t: t * t, x_range=[-3.05, 3.05], color=RED, stroke_width=5)
        cab = ax.plot(lambda t: abs(t), x_range=[-3.2, 3.2], color=ORANGE, stroke_width=5)
        tb = VGroup(row(r"\text{gap}", ["1", "2", "10"], MUTED, lw=2.4, cw=1.1),
                    row(r"\text{squared}", ["1", "4", "100"], RED, lw=2.4, cw=1.1),
                    row(r"\text{absolute}", ["1", "2", "10"], ORANGE, lw=2.4, cw=1.1)
                    ).arrange(DOWN, buff=0.35, aligned_edge=RIGHT).move_to(RIGHT * 3.2 + UP * 0.6)
        with self.voice(N["one4"]) as v:
            self.play(Create(ax), FadeIn(gl), Create(csq), FadeIn(tb[0]), run_time=1.2)
            self.wait_until(v, "costs")
            self.play(FadeIn(tb[1]), run_time=0.8)
            self.wait_until(v, "evenly")
            self.play(Create(cab), FadeIn(tb[2]), run_time=1)
            self.wait_until(v, "punishes")
            self.play(Indicate(tb[1][1][2], color=RED, scale_factor=1.4), run_time=1)
        self.clear_body()
        rows = VGroup(
            VGroup(T("true: cat", 30, GREEN), T("predicted: cat", 30, YELLOW), MathTex(r"\ell = 0", font_size=48, color=GREEN)),
            VGroup(T("true: cat", 30, GREEN), T("predicted: dog", 30, YELLOW), MathTex(r"\ell = 1", font_size=48, color=RED)))
        for r in rows:
            r.arrange(RIGHT, buff=0.9)
        rows.arrange(DOWN, buff=0.7, aligned_edge=LEFT).shift(UP * 0.4)
        name = T("0–1 loss: counts mistakes", 32, RED, weight=BOLD).next_to(rows, DOWN, buff=0.7)
        hard = T("honest, but hard to learn from", 24, MUTED).next_to(name, DOWN, buff=0.25)
        with self.voice(N["one5"]) as v:
            self.wait_until(v, "cat")
            self.play(FadeIn(rows[0][:2]), run_time=0.6)
            self.wait_until(v, "zero", 2)
            self.play(FadeIn(rows[0][2]), run_time=0.5)
            self.wait_until(v, "wrong")
            self.play(FadeIn(rows[1]), run_time=0.7)
            self.wait_until(v, "counts")
            self.play(FadeIn(name), run_time=0.6)
            self.wait_until(v, "hard")
            self.play(FadeIn(hard), run_time=0.5)


class C2_Risk(Studio):
    def construct(self):
        self.heading("2 · Average loss over a data set")
        ax = axes([0, 3.6, 1], [0, 9, 2], 4.8, 4.4).move_to(LEFT * 4.0 + DOWN * 0.6)
        nl = T("example n", 20, MUTED).next_to(ax, DOWN, buff=0.15)
        ys = VGroup(*[Dot(ax.c2p(i + 1, v), color=GREEN, radius=0.11) for i, v in enumerate(demo.y_true)])
        ps = VGroup(*[Dot(ax.c2p(i + 1, v), color=YELLOW, radius=0.11) for i, v in enumerate(demo.y_pred)])
        gaps = VGroup(*[Line(ax.c2p(i + 1, a), ax.c2p(i + 1, b), color=ORANGE, stroke_width=5)
                        for i, (a, b) in enumerate(zip(demo.y_true, demo.y_pred)) if a != b])
        leg = VGroup(T("● label y", 22, GREEN), T("● prediction ŷ", 22, YELLOW)
                     ).arrange(RIGHT, buff=0.4).next_to(ax, UP, buff=0.15)
        tb = VGroup(row("y", [fmt(a) for a in demo.y_true], GREEN),
                    row(r"\hat y", [fmt(a) for a in demo.y_pred], YELLOW),
                    row(r"(y-\hat y)^2", [fmt(a) for a in demo.sq], RED),
                    row(r"|y-\hat y|", [fmt(a) for a in demo.ab], ORANGE)
                    ).arrange(DOWN, buff=0.3, aligned_edge=RIGHT).move_to(RIGHT * 2.6 + UP * 1.2)
        toy = T("toy data set · 3 examples", 20, MUTED).next_to(nl, DOWN, buff=0.1)
        with self.voice(N["risk1"]) as v:
            self.play(Create(ax), FadeIn(nl), FadeIn(leg), run_time=1)
            self.wait_until(v, "toy")
            self.play(FadeIn(toy), run_time=0.5)
            self.wait_until(v, "labels")
            self.play(FadeIn(ys), FadeIn(tb[0]), run_time=0.8)
            self.wait_until(v, "predicts")
            self.play(FadeIn(ps), FadeIn(tb[1]), run_time=0.8)
            self.play(Create(gaps), run_time=0.6)
        with self.voice(N["risk2"]) as v:
            self.wait_until(v, "squared")
            self.play(FadeIn(tb[2][0]), FadeIn(tb[2][1][0]), run_time=0.6)
            self.wait_until(v, "zero")
            self.play(FadeIn(tb[2][1][1]), run_time=0.5)
            self.wait_until(v, "four", 3)
            self.play(FadeIn(tb[2][1][2]), run_time=0.5)
            self.wait_until(v, "disappears")
            self.play(Indicate(tb[2][1][2], color=RED), run_time=0.8)
        avg = MathTex(r"\frac{1+0+4}{3}", "=", r"\frac53", r"\approx", "%.2f" % demo.sq.mean(), font_size=44)
        avg[4].set_color(RED)
        avg.next_to(tb, DOWN, buff=0.5).set_x(2.8)
        er = T("empirical risk = mean loss on our data", 24, RED).next_to(avg, DOWN, buff=0.3)
        rem = self.reminder_card("mean", "Add the values and divide by how many there are.",
                                 pos=DR, width=4.2)
        with self.voice(N["risk3"]) as v:
            self.wait_until(v, "mean")
            self.play(FadeIn(rem, shift=UP * 0.2), Write(avg[0]), run_time=1)
            self.wait_until(v, "about")
            self.play(Write(avg[1:]), run_time=1)
            self.wait_until(v, "empirical")
            self.play(FadeOut(rem), FadeIn(er), run_time=0.6)
        self.play(FadeOut(VGroup(avg, er, toy)), tb.animate.shift(UP * 0.3), run_time=0.5)
        R = MathTex("R", "=", r"\frac{1}{N}", r"\sum_{n=1}^{N}", r"\ell(y_n,\hat y_n)", font_size=52)
        R[0].set_color(RED); R[2].set_color(PURPLE); R[4].set_color(RED)
        R.move_to(RIGHT * 2.8 + DOWN * 0.9)
        legend = VGroup(T("N = number of examples", 22, MUTED),
                        T("Σ = add up all the losses", 22, MUTED),
                        T("1/N turns the sum into a mean", 22, PURPLE)
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(R, DOWN, buff=0.3)
        abm = MathTex(r"\text{absolute: } \tfrac{1+0+2}{3} = %s" % fmt(demo.ab.mean()), color=ORANGE,
                      font_size=36).next_to(legend, DOWN, buff=0.3)
        with self.voice(N["risk4"]) as v:
            self.wait_until(v, "formula")
            self.play(Write(R), run_time=1.2)
            self.wait_until(v, "number")
            self.play(FadeIn(legend[0]), run_time=0.5)
            self.wait_until(v, "sum", 2)
            self.play(FadeIn(legend[1]), run_time=0.5)
            self.wait_until(v, "mean")
            self.play(FadeIn(legend[2]), run_time=0.5)
            self.wait_until(v, "absolute")
            self.play(FadeIn(tb[3]), Write(abm), run_time=1)


class C3_Fit(Studio):
    def construct(self):
        self.heading("3 · Parameters: the knobs of a model")
        ax = axes([0, 5, 1], [0, 10, 2], 5.4, 4.6).move_to(LEFT * 3.6 + DOWN * 0.6)
        pts = VGroup(*[Dot(ax.c2p(a, b), color=GREEN, radius=0.1) for a, b in zip(demo.x, demo.y)])
        w = ValueTracker(1.0)
        line = always_redraw(lambda: ax.plot(lambda t: w.get_value() * t,
                                             x_range=[0, min(5, 10 / max(w.get_value(), 0.01))],
                                             color=YELLOW, stroke_width=5))
        f = MathTex(r"\hat y", "=", "w", r"\cdot", "x", font_size=56).move_to(RIGHT * 3.2 + UP * 2.1)
        f[0].set_color(YELLOW); f[2].set_color(PURPLE); f[4].set_color(BLUE)
        toy = T("toy points", 20, MUTED).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        with self.voice(N["fit1"]) as v:
            self.play(Create(ax), FadeIn(toy), run_time=1)
            self.wait_until(v, "four")
            self.play(FadeIn(pts), run_time=0.6)
            self.wait_until(v, "line")
            self.play(Write(f), run_time=1)
            self.wait_until(v, "slope")
            self.add(line)
            self.play(Indicate(f[2], color=PURPLE), run_time=0.8)
            self.wait_until(v, "Different")
            self.play(w.animate.set_value(3), run_time=1.2)
            self.play(w.animate.set_value(1), run_time=1.2)
        card = self.term_card("parameter", "A number inside a model that we can change, "
                              "like a knob. Learning means choosing its value.", pos=DR, width=5.4)
        knob = T("w = one knob · GPT: millions or billions of knobs", 22, PURPLE).next_to(f, DOWN, buff=0.35)
        with self.voice(N["fit2"]) as v:
            self.wait_until(v, "parameter")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "billions")
            self.play(FadeIn(knob), run_time=0.6)
            self.wait_until(v, "Learning")
            self.play(Indicate(f[2], color=PURPLE), run_time=0.8)
        vals = VGroup(*[MathTex(r"w=%d:\ \ L = %s" % (k, fmt(round(demo.L_at[k], 2))), font_size=40,
                                color=YELLOW if k == 2 else FG) for k in (1, 3, 2)]
                      ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(RIGHT * 3.0 + DOWN * 0.6)
        tags = VGroup(T("too flat", 22, MUTED), T("too steep", 22, MUTED), T("close to all", 22, YELLOW))
        for t_, vv in zip(tags, vals):
            t_.next_to(vv, RIGHT, buff=0.3)
        with self.voice(N["fit3"]) as v:
            self.play(FadeOut(card), run_time=0.4)
            self.wait_until(v, "one")
            self.play(FadeIn(vals[0]), FadeIn(tags[0]), run_time=0.6)
            self.wait_until(v, "three")
            self.play(w.animate.set_value(3), run_time=1)
            self.play(FadeIn(vals[1]), FadeIn(tags[1]), run_time=0.6)
            self.wait_until(v, "two")
            self.play(w.animate.set_value(2), run_time=1)
            self.play(FadeIn(vals[2]), FadeIn(tags[2]), run_time=0.6)
        lax = axes([0, 4, 1], [0, 35, 10], 5.4, 4.2).move_to(RIGHT * 3.5 + DOWN * 0.8)
        lxl = MathTex("w", color=PURPLE, font_size=32).next_to(lax, DOWN, buff=0.1).align_to(lax, RIGHT)
        lyl = MathTex("L(w)", color=RED, font_size=32).next_to(lax, UP, buff=0.1).align_to(lax, LEFT)
        curve = lax.plot_line_graph(demo.ws, demo.L, add_vertex_dots=False, line_color=RED, stroke_width=4)
        dot = always_redraw(lambda: Dot(lax.c2p(w.get_value(), mse(w.get_value())), color=YELLOW, radius=0.1))
        with self.voice(N["fit4"]) as v:
            self.play(FadeOut(VGroup(vals, tags, knob)), f.animate.scale(0.8).move_to(LEFT * 1.6 + UP * 2.4),
                      run_time=0.6)
            self.wait_until(v, "right")
            self.play(Create(lax), FadeIn(lxl), FadeIn(lyl), run_time=0.8)
            self.wait_until(v, "curve")
            self.play(Create(curve), run_time=1.5)
            self.wait_until(v, "Watch")
            self.play(w.animate.set_value(0.2), run_time=0.8)
            self.add(dot)
            self.play(w.animate.set_value(3.8), run_time=4)
        wb = float(demo.w_best)
        best = MathTex(r"w^\ast = %.2f,\ \ L = %.2f" % (wb, demo.L.min()), color=YELLOW,
                       font_size=36).move_to(lax.c2p(2.0, 21))
        bowl = T("a bowl", 24, RED).move_to(lax.c2p(2.0, 32))
        with self.voice(N["fit5"]) as v:
            self.wait_until(v, "bowl")
            self.play(FadeIn(bowl), run_time=0.5)
            self.wait_until(v, "bottom")
            self.play(w.animate.set_value(wb), run_time=2)
            self.wait_until(v, "lowest")
            self.play(FadeIn(best), run_time=0.6)
            self.wait_until(v, "fitted")
            self.play(Indicate(pts, color=YELLOW), run_time=0.8)


class C4_Smooth(Studio):
    def construct(self):
        self.heading("4 · Smooth losses are easier")
        lax = axes([0, 4, 1], [0, 35, 10], 6.0, 4.6).move_to(LEFT * 3.0 + DOWN * 0.6)
        curve = lax.plot_line_graph(demo.ws, demo.L, add_vertex_dots=False, line_color=RED, stroke_width=4)
        w = ValueTracker(0.8)

        def dL(t):
            return float(np.mean(-2 * demo.x * (demo.y - t * demo.x)))

        def tangent():
            t = w.get_value()
            s = dL(t)
            a, b = t - 0.4, t + 0.4
            return Line(lax.c2p(a, mse(t) + s * (a - t)), lax.c2p(b, mse(t) + s * (b - t)),
                        color=YELLOW, stroke_width=4)
        tan = always_redraw(tangent)
        dot = always_redraw(lambda: Dot(lax.c2p(w.get_value(), mse(w.get_value())), color=YELLOW, radius=0.1))
        rd = always_redraw(lambda: T(f"slope = {dL(w.get_value()):+.1f}", 30,
                                     GREEN if dL(w.get_value()) < 0 else RED, font=MONO
                                     ).move_to(RIGHT * 3.8 + UP * 1.4))
        rule = VGroup(T("slope > 0 → make w smaller", 26, RED),
                      T("slope < 0 → make w bigger", 26, GREEN)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.8 + DOWN * 0.2)
        rem = self.reminder_card("derivative", "The slope of a function at a point (video 7).",
                                 pos=DR, width=4.4)
        with self.voice(N["smooth1"]) as v:
            self.play(Create(lax), Create(curve), run_time=1.2)
            self.wait_until(v, "smooth")
            self.add(tan, dot, rd)
            self.wait_until(v, "derivative")
            self.play(FadeIn(rem, shift=UP * 0.2), run_time=0.5)
            self.wait_until(v, "positive")
            self.play(FadeOut(rem), w.animate.set_value(3.2), FadeIn(rule[0]), run_time=1.2)
            self.wait_until(v, "negative")
            self.play(w.animate.set_value(1.0), FadeIn(rule[1]), run_time=1.2)
        self.clear_body()
        nl = NumberLine(x_range=[0, 7, 1], length=9, color=MUTED, include_numbers=True, font_size=26).shift(UP * 0.4)
        cols = {0: BLUE, 1: GREEN}
        pts = VGroup(*[Dot(nl.n2p(a), radius=0.16, color=cols[int(c)]) for a, c in zip(demo.xc, demo.lab)])
        cl = VGroup(*[MathTex(str(int(c)), color=cols[int(c)], font_size=34).next_to(p, UP, buff=0.2)
                      for p, c in zip(pts, demo.lab)])
        b = ValueTracker(2.5)
        th = always_redraw(lambda: DashedLine(nl.n2p(b.get_value()) + DOWN * 0.8,
                                              nl.n2p(b.get_value()) + UP * 1.3, color=PURPLE))
        bl = always_redraw(lambda: MathTex("b", color=PURPLE, font_size=36).next_to(
            nl.n2p(b.get_value()) + UP * 1.3, UP, buff=0.1))
        rr = MathTex(r"\text{predict } 1 \text{ if } x > b", font_size=40).move_to(DOWN * 1.6)
        mis = always_redraw(lambda: T(
            "mistakes: %d of 6" % int(np.sum((demo.xc > b.get_value()).astype(int) != demo.lab)),
            28, RED).move_to(DOWN * 2.5))
        lg = T("class 0 = blue · class 1 = green · toy data", 20, MUTED).to_edge(DOWN, buff=0.3)
        with self.voice(N["smooth2"]) as v:
            self.wait_until(v, "six")
            self.play(Create(nl), FadeIn(pts), FadeIn(lg), run_time=1)
            self.wait_until(v, "class")
            self.play(FadeIn(cl), run_time=0.6)
            self.wait_until(v, "threshold")
            self.add(th, bl)
            self.play(Write(rr), run_time=0.8)
            self.wait_until(v, "count")
            self.add(mis)
            self.play(b.animate.set_value(3.5), run_time=1)
            self.play(b.animate.set_value(4.5), run_time=1)
        self.clear_body()
        sax = axes([0, 7, 1], [0, 0.6, 0.1], 7.0, 4.4).move_to(LEFT * 2.2 + DOWN * 0.6)
        pts01 = [sax.c2p(bb, ll) for bb, ll in zip(demo.bs, demo.L01)]
        stair = VMobject(color=RED, stroke_width=5).set_points_as_corners(pts01)
        sxl = MathTex("b", color=PURPLE, font_size=32).next_to(sax, DOWN, buff=0.1).align_to(sax, RIGHT)
        syl = T("0–1 loss (fraction wrong)", 20, RED).next_to(sax, UP, buff=0.1).align_to(sax, LEFT)
        notes = VGroup(T("701 tries → only 3 values", 26, YELLOW),
                       T("flat → slope 0 → no direction", 26, MUTED),
                       T("jumps → no slope at all", 26, MUTED)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 4.6 + UP * 0.2)
        if notes.get_right()[0] > 7.0:
            notes.shift(LEFT * (notes.get_right()[0] - 7.0))
        with self.voice(N["smooth3"]) as v:
            self.play(Create(sax), FadeIn(sxl), FadeIn(syl), run_time=0.8)
            self.wait_until(v, "staircase")
            self.play(Create(stair), run_time=1.5)
            self.wait_until(v, "three")
            self.play(FadeIn(notes[0]), run_time=0.5)
            self.wait_until(v, "flat")
            self.play(FadeIn(notes[1]), run_time=0.5)
            self.wait_until(v, "steps", 2)
            self.play(FadeIn(notes[2]), run_time=0.5)
        self.clear_body()
        pl = VGroup(VGroup(T("train with", 26, MUTED), T("a smooth loss", 34, GREEN, weight=BOLD)).arrange(DOWN, buff=0.2),
                    VGroup(T("report", 26, MUTED), T("the count of mistakes", 34, RED, weight=BOLD)).arrange(DOWN, buff=0.2)
                    ).arrange(RIGHT, buff=1.6).shift(UP * 0.6)
        ce = T("classification → cross-entropy loss (a molecule, later)", 26, YELLOW).next_to(pl, DOWN, buff=0.9)
        with self.voice(N["smooth4"]) as v:
            self.wait_until(v, "smooth")
            self.play(FadeIn(pl[0]), run_time=0.6)
            self.wait_until(v, "report")
            self.play(FadeIn(pl[1]), run_time=0.6)
            self.wait_until(v, "classification")
            self.play(FadeIn(ce), run_time=0.6)


class C5_Outlier(Studio):
    def construct(self):
        self.heading("5 · Outliers: squared vs absolute")
        ax = axes([0, 5.5, 1], [0, 32, 5], 6.0, 4.8).move_to(LEFT * 3.2 + DOWN * 0.6)
        pts = VGroup(*[Dot(ax.c2p(a, b), color=GREEN, radius=0.1) for a, b in zip(demo.xo, demo.yo)])
        pts[4].set_color(RED)
        ol = T("outlier", 22, RED).next_to(pts[4], LEFT, buff=0.2)
        should = DashedLine(ax.c2p(5, 10), ax.c2p(5, 30), color=MUTED)
        sl = T("should be 10", 20, MUTED).next_to(ax.c2p(5, 18), RIGHT, buff=0.15)
        l2 = ax.plot(lambda t: 2 * t, x_range=[0, 5.5], color=MUTED, stroke_width=3)
        with self.voice(N["out1"]) as v:
            self.play(Create(ax), run_time=0.8)
            self.wait_until(v, "five")
            self.play(FadeIn(pts[:4]), run_time=0.6)
            self.wait_until(v, "exactly")
            self.play(Create(l2), run_time=0.8)
            self.wait_until(v, "thirty")
            self.play(FadeIn(pts[4]), Create(should), FadeIn(sl), run_time=0.8)
            self.wait_until(v, "outlier")
            self.play(FadeIn(ol), Indicate(pts[4], color=RED), run_time=0.8)
        cost = VGroup(MathTex(r"\text{error} = 30 - 10 = 20", font_size=40),
                      MathTex(r"|20| = 20", color=ORANGE, font_size=44),
                      MathTex(r"20^2 = 400", color=RED, font_size=44)
                      ).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to(RIGHT * 3.6 + UP * 1.0)
        with self.voice(N["out2"]) as v:
            self.wait_until(v, "twenty")
            self.play(FadeIn(cost[0]), Indicate(should, color=ORANGE), run_time=0.8)
            self.wait_until(v, "charges")
            self.play(FadeIn(cost[1]), run_time=0.6)
            self.wait_until(v, "charges", 2)
            self.play(FadeIn(cost[2]), run_time=0.6)
            self.wait_until(v, "bend")
            self.play(Indicate(cost[2], color=RED), run_time=0.8)
        lsq = ax.plot(lambda t: demo.w_sq * t, x_range=[0, 5.5], color=RED, stroke_width=5)
        lab_ = ax.plot(lambda t: demo.w_ab * t, x_range=[0, 5.5], color=ORANGE, stroke_width=5)
        res = VGroup(MathTex(r"\text{squared: } w = %.2f" % demo.w_sq, color=RED, font_size=40),
                     MathTex(r"\text{absolute: } w = %.2f" % demo.w_ab, color=ORANGE, font_size=40)
                     ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(RIGHT * 3.6 + DOWN * 1.5)
        with self.voice(N["out3"]) as v:
            self.play(FadeOut(l2), run_time=0.4)
            self.wait_until(v, "squared")
            self.play(Create(lsq), FadeIn(res[0]), run_time=1)
            self.wait_until(v, "pulled")
            self.play(Indicate(lsq, color=RED), run_time=0.8)
            self.wait_until(v, "absolute")
            self.play(Create(lab_), FadeIn(res[1]), run_time=1)
        self.clear_body()
        hdr = VGroup(Dot(radius=0.02).set_opacity(0), T("squared", 30, RED, weight=BOLD), T("absolute", 30, ORANGE, weight=BOLD))
        r1 = VGroup(T("smooth, easy to minimize", 26, FG), T("✓", 34, GREEN), T("corner at 0", 26, MUTED))
        r2 = VGroup(T("robust to outliers", 26, FG), T("✗", 34, RED), T("✓", 34, GREEN))
        grid = VGroup(hdr, r1, r2)
        for r in grid:
            for i, c in enumerate(r):
                c.move_to(RIGHT * [-3.4, 0.8, 4.0][i])
        grid.arrange(DOWN, buff=0.6)
        for r in grid:
            for i, c in enumerate(r):
                c.set_x([-3.4, 0.8, 4.0][i])
        grid.shift(UP * 0.4)
        ch = T("choosing the loss = choosing which mistakes you care about", 26, YELLOW).to_edge(DOWN, buff=0.8)
        with self.voice(N["out4"]) as v:
            self.wait_until(v, "neither")
            self.play(FadeIn(hdr), run_time=0.6)
            self.wait_until(v, "Squared")
            self.play(FadeIn(r1[:2]), FadeIn(r2[:2]), run_time=0.8)
            self.wait_until(v, "robust")
            self.play(FadeIn(r1[2]), FadeIn(r2[2]), run_time=0.8)
            self.wait_until(v, "Choosing")
            self.play(FadeIn(ch), run_time=0.6)


class C6_Optimize(Studio):
    def construct(self):
        self.heading("6 · Learning = optimization")
        f = MathTex(r"w^\ast", "=", r"\arg\min_{w}", r"\ L(w)", font_size=64).shift(UP * 1.4)
        f[0].set_color(PURPLE); f[3].set_color(RED)
        rd = T("w* = the parameter value with the smallest loss", 24, MUTED).next_to(f, DOWN, buff=0.3)
        card = self.term_card("optimization", "Finding the input that makes a function as small "
                              "(or as large) as possible.", pos=DR, width=5.2)
        ax_t = T("Axiom E · Learning = optimization", 44, YELLOW, weight=BOLD).move_to(DOWN * 1.2 + LEFT * 1.6)
        with self.voice(N["opt1"]) as v:
            self.wait_until(v, "Learning")
            self.play(Write(f), run_time=1.2)
            self.play(FadeIn(rd), run_time=0.5)
            self.wait_until(v, "optimization")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "axiom")
            self.play(FadeOut(card), Write(ax_t), run_time=1)
        ch = VGroup(T("a choice, not a theorem", 28, FG),
                    T("straight line · logistic regression · GPT → all minimize a loss", 24, MUTED)
                    ).arrange(DOWN, buff=0.3).next_to(ax_t, DOWN, buff=0.5)
        ch.set_x(0)
        with self.voice(N["opt2"]) as v:
            self.play(ax_t.animate.set_x(0), run_time=0.5)
            self.wait_until(v, "choice")
            self.play(FadeIn(ch[0]), run_time=0.6)
            self.wait_until(v, "GPT")
            self.play(FadeIn(ch[1]), run_time=0.6)
        self.clear_body()
        lax = axes([0, 4, 1], [0, 35, 10], 6.0, 4.6).move_to(LEFT * 3.0 + DOWN * 0.6)
        curve = lax.plot_line_graph(demo.ws, demo.L, add_vertex_dots=False, line_color=RED, stroke_width=4)
        steps, wv = [0.5], 0.5
        for _ in range(4):
            wv = wv - 0.05 * float(np.mean(-2 * demo.x * (demo.y - wv * demo.x)))
            steps.append(wv)
        dots = VGroup(*[Dot(lax.c2p(s, mse(s)), color=YELLOW, radius=0.09) for s in steps])
        arrs = VGroup(*[Arrow(lax.c2p(a, mse(a)), lax.c2p(b, mse(b)), buff=0.08, color=YELLOW,
                              stroke_width=3, max_tip_length_to_length_ratio=0.2)
                        for a, b in zip(steps, steps[1:])])
        tx = VGroup(T("1 knob: 401 tries is fine", 24, FG),
                    T("1 billion knobs: a grid is impossible", 24, RED),
                    T("follow the slope downhill", 26, YELLOW),
                    T("→ gradient descent (molecule)", 26, YELLOW, weight=BOLD)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.7 + UP * 0.3)
        il = T("illustration: real steps of size 0.05 × slope", 20, MUTED).next_to(lax, DOWN, buff=0.2)
        with self.voice(N["opt3"]) as v:
            self.play(Create(lax), Create(curve), run_time=1)
            self.wait_until(v, "minimum")
            self.play(FadeIn(tx[0]), run_time=0.5)
            self.wait_until(v, "billion")
            self.play(FadeIn(tx[1]), run_time=0.5)
            self.wait_until(v, "downhill")
            self.play(FadeIn(tx[2]), FadeIn(dots[0]), FadeIn(il), run_time=0.5)
            self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(d)) for a, d in zip(arrs, dots[1:])],
                                  lag_ratio=0.6), run_time=2)
            self.wait_until(v, "descent")
            self.play(FadeIn(tx[3]), run_time=0.5)
        self.clear_body()
        ce = MathTex(r"\mathcal{L}", "=", r"-\frac1N\sum_{n}", r"\log p(\text{correct next token})", font_size=48)
        ce[0].set_color(RED); ce[3].set_color(YELLOW)
        ce.shift(UP * 1.2)
        ml = VGroup(T("molecules built from this atom", 30, YELLOW, weight=BOLD),
                    T("gradient descent = gradient + loss", 26, FG),
                    T("cross-entropy = log + probability + loss", 26, FG),
                    T("→ the molecules map", 24, MUTED)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).shift(DOWN * 1.4)
        with self.voice(N["opt4"]) as v:
            self.wait_until(v, "logarithm")
            self.play(Write(ce), run_time=1.5)
            self.wait_until(v, "molecules")
            self.play(FadeIn(ml[:3]), run_time=0.8)
            self.wait_until(v, "map")
            self.play(FadeIn(ml[3]), run_time=0.5)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        q = VGroup(MathTex(r"y = (3,\ 5)", color=GREEN, font_size=52),
                   MathTex(r"\hat y = (1,\ 6)", color=YELLOW, font_size=52),
                   MathTex(r"\text{mean squared error} = ?\qquad \text{mean absolute error} = ?", font_size=42)
                   ).arrange(DOWN, buff=0.35).shift(UP * 1.4)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.4)
        with self.voice(N["prac1"]) as v:
            self.play(FadeIn(q[0]), run_time=0.6)
            self.wait_until(v, "predictions")
            self.play(FadeIn(q[1]), run_time=0.6)
            self.play(Write(q[2]), run_time=1)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
        a = VGroup(MathTex(r"(3-1)^2 = 4,\quad (5-6)^2 = 1", color=RED, font_size=42),
                   MathTex(r"\text{MSE} = \tfrac{4+1}{2} = 2.5", color=RED, font_size=44),
                   MathTex(r"\text{MAE} = \tfrac{2+1}{2} = 1.5", color=ORANGE, font_size=44)
                   ).arrange(DOWN, buff=0.3).next_to(q, DOWN, buff=0.4)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "squared")
            self.play(Write(a[0]), run_time=1.2)
            self.wait_until(v, "mean")
            self.play(Write(a[1]), run_time=1)
            self.wait_until(v, "absolute")
            self.play(Write(a[2]), run_time=1)
        self.clear_body()
        ax = axes([0, 3, 1], [0, 7, 1], 4.4, 4.4).move_to(LEFT * 4.0 + DOWN * 0.6)
        P = [(1, 3), (2, 6)]
        pts = VGroup(*[Dot(ax.c2p(*p), radius=0.1, color=GREEN) for p in P])
        q2 = VGroup(MathTex(r"(1,3),\ (2,6)", color=GREEN, font_size=44),
                    MathTex(r"\hat y = w\cdot x:\quad w = 2 \text{ or } w = 3\,?", font_size=42),
                    T("⏸  pause and try", 28, MUTED)).arrange(DOWN, buff=0.4).move_to(RIGHT * 2.8 + UP * 1.4)
        with self.voice(N["prac3"]) as v:
            self.play(Create(ax), run_time=0.8)
            self.wait_until(v, "points")
            self.play(FadeIn(pts), FadeIn(q2[0]), run_time=0.8)
            self.wait_until(v, "Which")
            self.play(FadeIn(q2[1]), run_time=0.6)
            self.wait_until(v, "Pause")
            self.play(FadeIn(q2[2]), run_time=0.5)
        l2 = ax.plot(lambda t: 2 * t, x_range=[0, 3], color=MUTED)
        l3 = ax.plot(lambda t: 3 * t, x_range=[0, 7 / 3], color=YELLOW, stroke_width=5)
        ans = VGroup(MathTex(r"w=2:\ \hat y = (2,4),\ L = \tfrac{1+4}{2} = 2.5", font_size=40),
                     MathTex(r"w=3:\ \hat y = (3,6),\ L = 0", color=YELLOW, font_size=40),
                     T("w = 3 wins", 30, YELLOW, weight=BOLD)
                     ).arrange(DOWN, buff=0.35).move_to(RIGHT * 2.8 + DOWN * 1.2)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(q2[2]), run_time=0.4)
            self.wait_until(v, "two")
            self.play(Create(l2), FadeIn(ans[0]), run_time=1)
            self.wait_until(v, "three")
            self.play(Create(l3), FadeIn(ans[1]), run_time=1)
            self.wait_until(v, "wins")
            self.play(FadeIn(ans[2]), run_time=0.5)


CODE = '''import numpy as np
yt, yp = np.array([2, 4, 6]), np.array([1, 4, 8])
print((yt - yp)**2, np.abs(yt - yp))       # per prediction
x, y = np.array([1, 2, 3, 4]), np.array([2, 5, 6, 8])
ws = np.linspace(0, 4, 401)                # 401 slopes
L = [np.mean((y - w*x)**2) for w in ws]    # loss curve
print(ws[np.argmin(L)])                    # smallest loss
xc, lab = np.arange(1, 7), np.array([0, 0, 1, 0, 1, 1])
L01 = [np.mean((xc > b) != lab) for b in np.linspace(0, 7, 701)]
xo, yo = np.arange(1, 6), np.array([2, 4, 6, 8, 30])   # outlier
Lsq = [np.mean((yo - w*xo)**2) for w in ws]
Lab = [np.mean(np.abs(yo - w*xo)) for w in ws]
print(ws[np.argmin(Lsq)], ws[np.argmin(Lab)])'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=22)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[2], color=YELLOW, buff=0.06)
            self.wait_until(v, "squared")
            self.play(Create(box), run_time=0.5)
            for i, w in [(4, "slopes"), (6, "smallest"), (8, "zero"), (9, "outlier")]:
                self.wait_until(v, w)
                self.play(Transform(box, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                              buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=24, command="python demo.py").shift(DOWN * 0.3)
        if term.width > 13.4:
            term.scale_to_fit_width(13.4)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for i, (w, n) in enumerate([("squared", 1), ("absolute", 1), ("best", 1), ("three", 1),
                                        ("outlier", 1), ("gives", 2)]):
                line = [0, 1, 3, 5, 6, 7][i]
                self.wait_until(v, w, n)
                self.play(Indicate(term[1][line], color=YELLOW), run_time=0.6)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["Loss function: how wrong one prediction is, as one number ≥ 0",
                            "Squared error, absolute error, 0–1 loss",
                            "Empirical risk = mean loss over the data set",
                            "Parameters = the knobs of a model",
                            "Learning = optimization: choose parameters that make the loss small",
                            "Smooth losses are easier; squared error punishes outliers hard"],
                           size=26).shift(UP * 0.5)
        if pts.width > 13:
            pts.scale_to_fit_width(13)
        with self.voice(N["sum1"]) as v:
            for p in pts[:4]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 6)
        nxt = VGroup(T("Next · Video 16", 26, MUTED),
                     T("Activation function: the bend that makes networks deep", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Smooth")
            self.play(FadeIn(pts[5], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_OnePrediction", "C2_Risk", "C3_Fit", "C4_Smooth", "C5_Outlier",
          "C6_Optimize", "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_OnePrediction": "Scoring one guess",
                  "C2_Risk": "Average loss over a data set", "C3_Fit": "Parameters & fitting a line",
                  "C4_Smooth": "Smooth losses are easier", "C5_Outlier": "Outliers",
                  "C6_Optimize": "Learning = optimization", "C7_Practice": "Try it yourself",
                  "C7_Code": "Python demo", "C8_Summary": "Summary", "C9_Credits": "Credits"}
