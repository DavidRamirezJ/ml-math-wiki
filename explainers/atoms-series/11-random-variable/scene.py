"""Random variable — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 11 of 16"
TITLE = "Random variable — turning chance into numbers"


def mt(*parts, size=48, colors=None):
    m = MathTex(*parts, font_size=size)
    for i, c in (colors or {}).items():
        m[i].set_color(c)
    return m


def dice_grid(cs=0.58, center=LEFT * 3.4 + DOWN * 0.55):
    """6x6 outcome grid. Rows = blue die 1..6 (top to bottom), columns = green die 1..6."""
    cells = VGroup(*[Square(cs, stroke_color=MUTED, stroke_width=1.5, fill_color=BG,
                            fill_opacity=1) for _ in range(36)])
    cells.arrange_in_grid(6, 6, buff=0)
    cells.move_to(center)
    rows = VGroup(*[T(str(b), 22, BLUE, weight=BOLD).next_to(cells[(b - 1) * 6], LEFT, buff=0.15)
                    for b in range(1, 7)])
    cols = VGroup(*[T(str(g), 22, GREEN, weight=BOLD).next_to(cells[g - 1], UP, buff=0.12)
                    for g in range(1, 7)])
    rl = T("blue die", 20, BLUE).next_to(rows, LEFT, buff=0.15).rotate(PI / 2)
    cl = T("green die", 20, GREEN).next_to(cols, UP, buff=0.12)
    return cells, VGroup(rows, rl), VGroup(cols, cl)


def cell(cells, b, g):
    return cells[(b - 1) * 6 + (g - 1)]


def sums_in(cells, size=20):
    return VGroup(*[T(str(b + g), size, YELLOW).move_to(cell(cells, b, g))
                    for b in range(1, 7) for g in range(1, 7)]).set_z_index(3)


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        if title.width > 12.5:
            title.scale_to_fit_width(12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.2)
        qs = VGroup()
        for q, a in [("Will the customer click?", "1 or 0"), ("Is this email spam?", "1 or 0"),
                     ("Which word comes next?", "one word of many")]:
            qs.add(VGroup(T(q, 30, FG), Arrow(LEFT * 0.5, RIGHT * 0.5, buff=0, color=MUTED),
                          T(a, 28, YELLOW, font=MONO)).arrange(RIGHT, buff=0.35))
        qs.arrange(DOWN, aligned_edge=LEFT, buff=0.45).shift(DOWN * 0.9)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            for k, w in enumerate(["click", "spam", "next"]):
                self.wait_until(v, w)
                self.play(FadeIn(qs[k][0], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "numbers")
            self.play(*[AnimationGroup(GrowArrow(r[1]), FadeIn(r[2])) for r in qs], run_time=1)
            self.wait_until(v, "random")
            self.play(Indicate(title, color=YELLOW, scale_factor=1.05), run_time=1)
        pipe = VGroup(T("outcome", 36, BLUE, weight=BOLD), T("rule", 36, PURPLE, weight=BOLD),
                      T("number", 36, YELLOW, weight=BOLD)).arrange(RIGHT, buff=1.6).shift(DOWN * 0.6)
        arrs = VGroup(*[Arrow(pipe[i].get_right(), pipe[i + 1].get_left(), buff=0.2, color=MUTED)
                        for i in range(2)])
        ax = T("axiom D · probability rules", 28, MUTED).next_to(pipe, DOWN, buff=0.9)
        with self.voice(N["intro2"]) as v:
            self.play(FadeOut(qs), run_time=0.6)
            self.wait_until(v, "rule")
            self.play(FadeIn(pipe[0]), GrowArrow(arrs[0]), FadeIn(pipe[1]), GrowArrow(arrs[1]),
                      FadeIn(pipe[2]), run_time=1.2)
            self.wait_until(v, "simulation")
            self.play(Indicate(pipe[1], color=PURPLE), run_time=0.8)
            self.wait_until(v, "axiom")
            self.play(FadeIn(ax), run_time=0.6)


class C1_Outcomes(Studio):
    def construct(self):
        self.heading("1 · Experiments and outcomes")
        coins = VGroup()
        for s, c in [("H", BLUE), ("T", GREEN)]:
            coins.add(VGroup(Circle(0.75, color=c, fill_color=c, fill_opacity=0.15),
                             T(s, 44, c, weight=BOLD)))
        coins.arrange(RIGHT, buff=1.2).shift(LEFT * 2.5 + UP * 0.2)
        names = VGroup(T("heads", 24, BLUE).next_to(coins[0], DOWN, buff=0.25),
                       T("tails", 24, GREEN).next_to(coins[1], DOWN, buff=0.25))
        exl = T("experiment: flip a coin", 28, MUTED).next_to(coins, UP, buff=0.6)
        card = self.term_card("outcome", "One possible result of an experiment, like heads "
                              "or a roll of blue 3, green 4.", pos=DR, width=5.4)
        with self.voice(N["exp1"]) as v:
            self.wait_until(v, "flipping")
            self.play(FadeIn(exl), run_time=0.6)
            self.wait_until(v, "outcome")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "heads")
            self.play(FadeIn(coins), FadeIn(names), run_time=0.8)
        self.clear_body()
        cells, rows, cols = dice_grid()
        ex = cell(cells, 3, 4)
        exlab = T("blue 3, green 4", 26, YELLOW).move_to(RIGHT * 3.2 + UP * 0.5)
        ar = Arrow(exlab.get_left(), ex.get_right(), buff=0.15, color=YELLOW)
        cnt = mt(r"6\times 6=36", r"\text{ outcomes}", size=46).move_to(RIGHT * 3.2 + DOWN * 1.0)
        with self.voice(N["exp2"]) as v:
            self.wait_until(v, "blue")
            self.play(FadeIn(rows), run_time=0.6)
            self.wait_until(v, "green", 2)
            self.play(FadeIn(cols), run_time=0.6)
            self.wait_until(v, "thirty")
            self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.02), Write(cnt),
                      run_time=1.5)
            self.wait_until(v, "example")
            self.play(ex.animate.set_fill(YELLOW, opacity=0.5), FadeIn(exlab), GrowArrow(ar),
                      run_time=0.8)
        items = VGroup(*[VGroup(T(a, 28, FG), T("→ ?", 28, RED)).arrange(RIGHT, buff=0.3)
                         for a in ["heads", "an email", "a photo", "(3, 4)"]])
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(RIGHT * 3.4 + DOWN * 0.3)
        want = T("we want numbers", 28, YELLOW).next_to(items, UP, buff=0.5)
        with self.voice(N["exp3"]) as v:
            self.play(FadeOut(exlab), FadeOut(ar), FadeOut(cnt), run_time=0.5)
            for k, w in enumerate(["Heads", "email", "photo", "pair"]):
                self.wait_until(v, w)
                self.play(FadeIn(items[k], shift=LEFT * 0.2), run_time=0.5)
            self.wait_until(v, "numbers")
            self.play(FadeIn(want), run_time=0.6)


class C2_Rule(Studio):
    def construct(self):
        self.heading("2 · A random variable is a rule")
        card = self.term_card("random variable", "A rule (a function) that gives each outcome "
                              "of an experiment a number.", pos=DR, width=5.6)
        H = VGroup(Circle(0.6, color=BLUE, fill_color=BLUE, fill_opacity=0.15), T("H", 36, BLUE))
        Tt = VGroup(Circle(0.6, color=GREEN, fill_color=GREEN, fill_opacity=0.15), T("T", 36, GREEN))
        VGroup(H, Tt).arrange(DOWN, buff=1.0).move_to(LEFT * 4.6 + DOWN * 0.3)
        one = T("1", 48, YELLOW, weight=BOLD).move_to(LEFT * 1.0 + H.get_center()[1] * UP)
        zero = T("0", 48, YELLOW, weight=BOLD).move_to(LEFT * 1.0 + Tt.get_center()[1] * UP)
        a1 = Arrow(H.get_right(), one.get_left(), buff=0.2, color=PURPLE)
        a2 = Arrow(Tt.get_right(), zero.get_left(), buff=0.2, color=PURPLE)
        yl = VGroup(MathTex("Y", color=PURPLE, font_size=40).next_to(a1, UP, buff=0.1),
                    MathTex("Y", color=PURPLE, font_size=40).next_to(a2, UP, buff=0.1))
        f = mt(r"Y(\text{H})=1,\quad Y(\text{T})=0", size=44).move_to(LEFT * 2.8 + UP * 2.1)
        with self.voice(N["rv1"]) as v:
            self.wait_until(v, "rule")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "coin")
            self.play(FadeIn(H), FadeIn(Tt), run_time=0.6)
            self.wait_until(v, "one")
            self.play(GrowArrow(a1), FadeIn(yl[0]), FadeIn(one), run_time=0.7)
            self.wait_until(v, "zero")
            self.play(GrowArrow(a2), FadeIn(yl[1]), FadeIn(zero), run_time=0.7)
            self.wait_until(v, "Each", 2)
            self.play(Write(f), run_time=1)
            self.wait_until(v, "random", 2)
            self.play(Indicate(VGroup(a1, a2), color=PURPLE), run_time=0.8)
        self.clear_body()
        cells, rows, cols = dice_grid()
        sums = sums_in(cells)
        rule = mt("X", "=", r"\text{blue}+\text{green}", size=48,
                  colors={0: PURPLE}).move_to(RIGHT * 3.3 + UP * 1.9)
        e1 = mt(r"3+4=7", size=44, colors={0: YELLOW}).move_to(RIGHT * 3.3 + UP * 0.6)
        e2 = mt(r"6+6=12", size=44, colors={0: YELLOW}).next_to(e1, DOWN, buff=0.4)
        with self.voice(N["rv2"]) as v:
            self.play(FadeIn(cells), FadeIn(rows), FadeIn(cols), run_time=0.8)
            self.wait_until(v, "sum")
            self.play(Write(rule), run_time=0.8)
            self.wait_until(v, "seven")
            self.play(FadeIn(sums[2 * 6 + 3]), Write(e1), cell(cells, 3, 4).animate.set_fill(YELLOW, 0.3),
                      run_time=0.7)
            self.wait_until(v, "twelve")
            self.play(FadeIn(sums[35]), Write(e2), cell(cells, 6, 6).animate.set_fill(YELLOW, 0.3),
                      run_time=0.7)
            self.wait_until(v, "whole")
            self.play(LaggedStart(*[FadeIn(s) for s in sums], lag_ratio=0.03), run_time=2)
        n1 = VGroup(mt("X", size=48, colors={0: PURPLE}), T("= the rule", 26, MUTED)).arrange(RIGHT, buff=0.25)
        n2 = VGroup(mt("x", size=48, colors={0: YELLOW}), T("= one value", 26, MUTED)).arrange(RIGHT, buff=0.25)
        n3 = mt(r"X(", r"3", ",", r"4", r")=", "7", size=50,
                colors={0: PURPLE, 1: BLUE, 3: GREEN, 4: PURPLE, 5: YELLOW})
        n4 = T("not random, not a variable:\na function", 26, YELLOW)
        side = VGroup(n1, n2, n3, n4).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to(RIGHT * 3.3 + DOWN * 0.2)
        with self.voice(N["rv3"]) as v:
            self.play(FadeOut(rule), FadeOut(e1), FadeOut(e2), run_time=0.5)
            self.wait_until(v, "capital")
            self.play(FadeIn(n1), run_time=0.5)
            self.wait_until(v, "small")
            self.play(FadeIn(n2), run_time=0.5)
            self.wait_until(v, "So")
            self.play(Write(n3), run_time=1)
            self.wait_until(v, "function")
            self.play(FadeIn(n4), run_time=0.6)
        oc = cell(cells, 2, 5)
        r = VGroup(mt(r"\text{sum}=2+5=7", size=40, colors={0: YELLOW}),
                   mt(r"\text{largest}=5", size=40, colors={0: ORANGE}),
                   mt(r"\text{green}-\text{blue}=5-2=3", size=40, colors={0: PURPLE}))
        r.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to(RIGHT * 3.4 + DOWN * 0.3)
        olab = T("outcome: blue 2, green 5", 26, FG).next_to(r, UP, buff=0.5)
        with self.voice(N["rv4"]) as v:
            self.play(FadeOut(side), cell(cells, 3, 4).animate.set_fill(BG, 1),
                      cell(cells, 6, 6).animate.set_fill(BG, 1), run_time=0.5)
            self.wait_until(v, "two")
            self.play(oc.animate.set_fill(ORANGE, 0.45), FadeIn(olab), run_time=0.6)
            self.wait_until(v, "sum")
            self.play(FadeIn(r[0]), run_time=0.5)
            self.wait_until(v, "largest")
            self.play(FadeIn(r[1]), run_time=0.5)
            self.wait_until(v, "difference")
            self.play(FadeIn(r[2]), run_time=0.5)
            self.wait_until(v, "choose")
            self.play(Indicate(r, color=YELLOW), run_time=0.8)


class C3_Events(Studio):
    def construct(self):
        self.heading("3 · Events: questions about X")
        cells, rows, cols = dice_grid()
        sums = sums_in(cells)
        self.add(cells, rows, cols, sums)
        hi = [cell(cells, b, g) for b in range(1, 7) for g in range(1, 7) if b + g >= 10]
        t1 = mt(r"\text{event: } X\ge 10", size=48, colors={0: YELLOW}).move_to(RIGHT * 3.3 + UP * 1.6)
        c1 = T("10: 3 cells · 11: 2 · 12: 1", 26, FG).next_to(t1, DOWN, buff=0.45)
        c2 = mt(r"6\text{ outcomes of }36", size=44, colors={0: YELLOW}).next_to(c1, DOWN, buff=0.45)
        with self.voice(N["ev1"]) as v:
            self.wait_until(v, "event")
            self.play(Write(t1), run_time=0.8)
            self.wait_until(v, "Which")
            self.play(*[c.animate.set_fill(YELLOW, 0.35) for c in hi], run_time=0.8)
            self.wait_until(v, "three")
            self.play(FadeIn(c1), run_time=0.6)
            self.wait_until(v, "six")
            self.play(Write(c2), run_time=0.8)
        diag = [cell(cells, b, 7 - b) for b in range(1, 7)]
        t2 = mt(r"\text{event: } X=7", size=48, colors={0: ORANGE}).move_to(RIGHT * 3.3 + UP * 1.6)
        d1 = mt(r"(1,6),(2,5),\dots,(6,1)", size=40).next_to(t2, DOWN, buff=0.45)
        d2 = mt(r"6\text{ outcomes: the most}", size=44, colors={0: ORANGE}).next_to(d1, DOWN, buff=0.45)
        with self.voice(N["ev2"]) as v:
            self.play(*[c.animate.set_fill(BG, 1) for c in hi], FadeOut(VGroup(t1, c1, c2)),
                      run_time=0.6)
            self.wait_until(v, "seven")
            self.play(Write(t2), run_time=0.7)
            self.wait_until(v, "diagonal")
            self.play(*[c.animate.set_fill(ORANGE, 0.4) for c in diag], FadeIn(d1), run_time=1)
            self.wait_until(v, "too")
            self.play(Write(d2), run_time=0.7)
            self.wait_until(v, "common")
            self.play(Indicate(VGroup(*diag), color=ORANGE), run_time=0.8)
        e1 = VGroup(mt(r"X=1", size=46, colors={0: RED}), T("empty: no outcomes", 26, RED)
                    ).arrange(DOWN, buff=0.2).move_to(RIGHT * 3.3 + UP * 1.2)
        e2 = VGroup(mt(r"X\le 12", size=46, colors={0: GREEN}), T("all 36 outcomes", 26, GREEN)
                    ).arrange(DOWN, buff=0.2).next_to(e1, DOWN, buff=0.6)
        with self.voice(N["ev3"]) as v:
            self.play(*[c.animate.set_fill(BG, 1) for c in diag], FadeOut(VGroup(t2, d1, d2)),
                      run_time=0.6)
            self.wait_until(v, "impossible")
            self.play(FadeIn(e1), run_time=0.7)
            self.wait_until(v, "always")
            self.play(FadeIn(e2), *[c.animate.set_fill(GREEN, 0.2) for c in cells], run_time=0.9)
            self.wait_until(v, "likely")
            self.play(*[c.animate.set_fill(BG, 1) for c in cells], run_time=0.6)
        f = mt(r"\text{chance}(X\ge 10)", "=", r"\frac{6}{36}", r"\approx", "0.167", size=46,
               colors={0: YELLOW, 4: YELLOW}).move_to(RIGHT * 3.3 + DOWN * 0.3)
        if f.width > 6.6:
            f.scale_to_fit_width(6.6)
        each = mt(r"\text{each outcome: }\tfrac{1}{36}", size=40).next_to(f, UP, buff=0.6)
        nxt = T("full table of chances → video 12", 24, MUTED).next_to(f, DOWN, buff=0.6)
        with self.voice(N["ev4"]) as v:
            self.play(FadeOut(VGroup(e1, e2)), run_time=0.5)
            self.wait_until(v, "same")
            self.play(FadeIn(each), run_time=0.6)
            self.wait_until(v, "divided")
            self.play(*[c.animate.set_fill(YELLOW, 0.35) for c in hi], run_time=0.6)
            self.wait_until(v, "least")
            self.play(Write(f), run_time=1.2)
            self.wait_until(v, "twelve")
            self.play(FadeIn(nxt), run_time=0.6)


class C4_Kinds(Studio):
    def construct(self):
        self.heading("4 · Discrete and continuous")
        nl = NumberLine(x_range=[0, 12, 1], length=10, color=MUTED, include_numbers=True,
                        font_size=24).shift(UP * 1.0 + RIGHT * 0.8)
        dl = T("discrete", 30, BLUE, weight=BOLD).next_to(nl, LEFT, buff=0.4)
        coin = VGroup(*[Dot(nl.n2p(k), color=BLUE, radius=0.11) for k in (0, 1)])
        dice = VGroup(*[Dot(nl.n2p(k), color=YELLOW, radius=0.09) for k in range(2, 13)])
        lab = VGroup(T("coin: 0 or 1", 22, BLUE), T("dice sum: 2 … 12", 22, YELLOW),
                     T("emails per day: 0, 1, 2, …", 22, MUTED)).arrange(RIGHT, buff=0.6)
        lab.next_to(nl, UP, buff=0.45)
        with self.voice(N["dc1"]) as v:
            self.wait_until(v, "discrete")
            self.play(Create(nl), FadeIn(dl), run_time=1)
            self.wait_until(v, "coin")
            self.play(FadeIn(coin), FadeIn(lab[0]), run_time=0.6)
            self.wait_until(v, "dice")
            self.play(LaggedStart(*[FadeIn(d) for d in dice], lag_ratio=0.1), FadeIn(lab[1]), run_time=1)
            self.wait_until(v, "emails")
            self.play(FadeIn(lab[2]), run_time=0.6)
            self.wait_until(v, "count")
            self.play(Indicate(dice, color=YELLOW), run_time=0.8)
        cl = NumberLine(x_range=[0, 10, 1], length=10, color=MUTED, include_numbers=True,
                        font_size=24).shift(DOWN * 1.9 + RIGHT * 0.8)
        cn = T("continuous", 30, GREEN, weight=BOLD).next_to(cl, LEFT, buff=0.4)
        if cn.get_left()[0] < -7.0:
            cn.scale_to_fit_width(cn.width - (-7.0 - cn.get_left()[0])).next_to(cl, LEFT, buff=0.3)
        band = Line(cl.n2p(0), cl.n2p(10), color=GREEN, stroke_width=10)
        unit = T("waiting time (minutes)", 22, MUTED).next_to(cl, DOWN, buff=0.5)
        w = ValueTracker(2.0)
        dot = always_redraw(lambda: Dot(cl.n2p(w.get_value()), color=YELLOW, radius=0.11))
        read = always_redraw(lambda: T(f"{w.get_value():.2f} min", 28, YELLOW, font=MONO)
                             .next_to(cl.n2p(w.get_value()), UP, buff=0.3))
        h = T("height: 172.3 cm, …", 22, GREEN).next_to(cl, UP, buff=1.0).align_to(cl, LEFT)
        with self.voice(N["dc2"]) as v:
            self.wait_until(v, "continuous")
            self.play(Create(cl), FadeIn(cn), run_time=1)
            self.wait_until(v, "height")
            self.play(FadeIn(h), run_time=0.5)
            self.wait_until(v, "wait")
            self.play(Create(band), FadeIn(unit), run_time=0.8)
            self.add(dot, read)
            self.wait_until(v, "five")
            self.play(w.animate.set_value(2.5), run_time=0.8)
            self.wait_until(v, "one", 2)
            self.play(w.animate.set_value(2.51), run_time=0.5)
            self.wait_until(v, "Between")
            self.play(w.animate.set_value(6.37), run_time=1.5)
        rng_ = Line(cl.n2p(2), cl.n2p(5), color=YELLOW, stroke_width=14)
        q = mt(r"\text{chance}(2\le \text{time}\le 5)\ ?", size=40, colors={0: YELLOW})
        q.next_to(cl, UP, buff=1.0).align_to(cl, RIGHT)
        pipe = T("outcome → rule → number (same idea)", 24, MUTED).to_edge(DOWN, buff=0.25)
        with self.voice(N["dc3"]) as v:
            self.play(FadeOut(dot), FadeOut(read), run_time=0.4)
            self.wait_until(v, "ranges")
            self.play(Create(rng_), run_time=0.8)
            self.wait_until(v, "between")
            self.play(Write(q), run_time=1)
            self.wait_until(v, "rule")
            self.play(FadeIn(pipe), run_time=0.6)


class C5_Samples(Studio):
    def construct(self):
        self.heading("5 · Samples: running it many times")
        rows = VGroup()
        for k, ((b, g), x) in enumerate(zip(demo.dice[:6].tolist(), demo.X[:6].tolist())):
            rows.add(VGroup(T(f"run {k + 1}", 24, MUTED), T(f"({b}, {g})", 30, FG, font=MONO),
                            T("→", 30, PURPLE), T(str(x), 34, YELLOW, font=MONO, weight=BOLD)
                            ).arrange(RIGHT, buff=0.45))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.28).shift(LEFT * 2.0 + DOWN * 0.4)
        hdr = VGroup(T("outcome", 24, BLUE), T("sample of X", 24, YELLOW)).arrange(RIGHT, buff=1.0)
        hdr.next_to(rows, UP, buff=0.35).align_to(rows[0][1], LEFT)
        note = T("seeded computer simulation", 22, MUTED).move_to(RIGHT * 4.0 + DOWN * 0.4)
        with self.voice(N["sm1"]) as v:
            self.wait_until(v, "outcome")
            self.play(FadeIn(hdr), run_time=0.6)
            self.wait_until(v, "sample")
            self.play(Indicate(hdr[1], color=YELLOW), run_time=0.6)
            self.wait_until(v, "first")
            self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows], lag_ratio=0.25),
                      FadeIn(note), run_time=2)
        self.clear_body()
        chart = BarChart(values=[int(c) for c in demo.counts], bar_names=[str(k) for k in range(2, 13)],
                         y_range=[0, 1800, 400], x_length=8.6, y_length=4.6, bar_colors=[BLUE, YELLOW],
                         bar_width=0.7, axis_config={"font_size": 22, "color": MUTED})
        chart.move_to(LEFT * 1.6 + DOWN * 0.5)
        top7 = T(str(int(demo.counts[5])), 24, YELLOW, font=MONO).next_to(chart.bars[5], UP, buff=0.1)
        xl = T("sum X", 22, MUTED).next_to(chart, DOWN, buff=0.1)
        lab = T("10,000 runs", 26, FG).move_to(RIGHT * 4.9 + UP * 1.8)
        with self.voice(N["sm2"]) as v:
            self.wait_until(v, "ten")
            self.play(Create(chart), FadeIn(xl), FadeIn(lab), run_time=2)
            self.wait_until(v, "histogram")
            self.play(Indicate(chart.bars, color=YELLOW), run_time=0.8)
            self.wait_until(v, "seven")
            self.play(FadeIn(top7), Indicate(chart.bars[5], color=YELLOW), run_time=0.8)
            self.wait_until(v, "rare")
            self.play(Indicate(chart.bars[0], color=RED), Indicate(chart.bars[10], color=RED),
                      run_time=0.8)
        s1 = mt(r"\text{simulation: } %.4f" % demo.frac_ge10, size=40, colors={0: YELLOW})
        s2 = mt(r"\text{counting: } \tfrac{6}{36}\approx 0.1667", size=40, colors={0: GREEN})
        s3 = T("close, not equal", 24, MUTED)
        side = VGroup(T("X ≥ 10", 30, YELLOW, weight=BOLD), s1, s2, s3).arrange(DOWN, buff=0.35)
        side.move_to(RIGHT * 4.7 + DOWN * 0.3)
        if side.width > 4.4:
            side.scale_to_fit_width(4.4).move_to(RIGHT * 4.75 + DOWN * 0.3)
        with self.voice(N["sm3"]) as v:
            self.play(FadeOut(lab), *[chart.bars[i].animate.set_fill(ORANGE, 0.9) for i in (8, 9, 10)],
                      FadeIn(side[0]), run_time=0.7)
            self.wait_until(v, "zero")
            self.play(FadeIn(s1), run_time=0.6)
            self.wait_until(v, "Counting", 2)
            self.play(FadeIn(s2), run_time=0.6)
            self.wait_until(v, "close")
            self.play(FadeIn(s3), run_time=0.5)
        win = T("data = samples\n→ our window into X", 24, YELLOW).next_to(side, DOWN, buff=0.6).set_x(5.1)
        with self.voice(N["sm4"]) as v:
            self.wait_until(v, "samples")
            self.play(FadeIn(win, shift=UP * 0.2), run_time=0.6)
            self.wait_until(v, "histogram")
            self.play(Indicate(chart.bars, color=YELLOW), run_time=1)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · Random variables in ML")
        mail = VGroup(Rectangle(width=2.6, height=1.7, color=BLUE),
                      Line(LEFT * 1.3 + UP * 0.85, ORIGIN + DOWN * 0.1, color=BLUE),
                      Line(RIGHT * 1.3 + UP * 0.85, ORIGIN + DOWN * 0.1, color=BLUE)).move_to(LEFT * 4 + UP * 0.2)
        ml = T("a random email", 24, BLUE).next_to(mail, DOWN, buff=0.3)
        ar = Arrow(mail.get_right(), mail.get_right() + RIGHT * 2.2, buff=0.2, color=PURPLE)
        yl = MathTex("Y", color=PURPLE, font_size=40).next_to(ar, UP, buff=0.1)
        out = VGroup(T("1 = spam", 30, YELLOW, font=MONO), T("0 = not spam", 30, YELLOW, font=MONO)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(ar, RIGHT, buff=0.3)
        cl = T("classifier: predict Y from the email", 26, MUTED).to_edge(DOWN, buff=0.9)
        with self.voice(N["ml1"]) as v:
            self.wait_until(v, "email")
            self.play(Create(mail), FadeIn(ml), run_time=0.8)
            self.wait_until(v, "label")
            self.play(GrowArrow(ar), FadeIn(yl), FadeIn(out), run_time=0.8)
            self.wait_until(v, "coin")
            self.play(Indicate(out, color=YELLOW), run_time=0.8)
            self.wait_until(v, "classifier")
            self.play(FadeIn(cl), run_time=0.6)
        self.clear_body()
        nl = NumberLine(x_range=[19, 21, 0.5], length=10, color=MUTED, include_numbers=True, font_size=24,
                        decimal_number_config={"num_decimal_places": 1}).shift(DOWN * 0.6)
        true = VGroup(Line(nl.n2p(20) + DOWN * 0.3, nl.n2p(20) + UP * 0.6, color=GREEN, stroke_width=5),
                      T("true: 20.0", 24, GREEN))
        true[1].next_to(true[0], UP, buff=0.15)
        r1 = Dot(nl.n2p(20.3), color=YELLOW, radius=0.11)
        r2 = Dot(nl.n2p(19.8), color=YELLOW, radius=0.11)
        rl = VGroup(T("20.3", 22, YELLOW).next_to(r1, DOWN, buff=0.45),
                    T("19.8", 22, YELLOW).next_to(r2, DOWN, buff=0.45))
        f = mt(r"\text{reading}", "=", r"20", "+", r"\text{noise}", size=50,
               colors={0: YELLOW, 2: GREEN, 4: RED}).shift(UP * 2.0)
        toy = T("toy example", 20, MUTED).to_edge(DOWN, buff=0.3)
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "thermometer")
            self.play(Create(nl), FadeIn(toy), run_time=0.8)
            self.wait_until(v, "noise")
            self.play(Write(f), run_time=1)
            self.wait_until(v, "twenty")
            self.play(Create(true[0]), FadeIn(true[1]), run_time=0.6)
            self.wait_until(v, "twenty", 2)
            self.play(FadeIn(r1), FadeIn(rl[0]), run_time=0.5)
            self.wait_until(v, "nineteen")
            self.play(FadeIn(r2), FadeIn(rl[1]), run_time=0.5)
            self.wait_until(v, "continuous")
            self.play(Indicate(f[4], color=RED), run_time=0.8)
        self.clear_body()
        data = [("34", "52 000", "1"), ("51", "38 000", "0"), ("27", "61 000", "1")]
        tab = VGroup(*[T(h, 24, MUTED) for h in ("age", "income", "label")])
        for row in data:
            tab.add(*[T(c, 26, FG, font=MONO) for c in row])
        tab.arrange_in_grid(4, 3, buff=(0.8, 0.35)).move_to(LEFT * 3.6 + UP * 0.4)
        boxes = VGroup(*[SurroundingRectangle(VGroup(*tab[3 + 3 * i: 6 + 3 * i]), color=BLUE, buff=0.1)
                         for i in range(3)])
        tl = T("toy data set: 3 samples", 22, MUTED).next_to(tab, DOWN, buff=0.4)
        rem = self.reminder_card("vector", "A list of numbers you can add and scale (video 01).",
                                 pos=DR, width=4.6)
        lm = VGroup(T("every effort moves you …", 26, FG), T("next word = a random variable", 26, YELLOW)
                    ).arrange(DOWN, buff=0.25).move_to(RIGHT * 3.4 + UP * 1.6)
        with self.voice(N["ml3"]) as v:
            self.wait_until(v, "collection")
            self.play(FadeIn(tab), FadeIn(tl), run_time=0.8)
            self.wait_until(v, "row")
            self.play(LaggedStart(*[Create(b) for b in boxes], lag_ratio=0.3), run_time=1)
            self.wait_until(v, "vector")
            self.play(FadeIn(rem, shift=UP * 0.2), run_time=0.6)
            self.wait_until(v, "language")
            self.play(FadeIn(lm), run_time=0.8)
            self.wait_until(v, "sample", 2)
            self.play(Indicate(lm[1], color=YELLOW), run_time=0.8)
        self.clear_body()
        words = [("forward", 0.57, 582), ("toward", 0.36, 343), ("closer", 0.06, 73)]
        ctx = T("every effort moves you  ___", 32, FG).shift(UP * 2.1)
        bars = VGroup()
        for k, (w, pr, n) in enumerate(words):
            r = Rectangle(width=pr * 9, height=0.55, color=YELLOW, fill_color=YELLOW, fill_opacity=0.5)
            lab = T(w, 28, BLUE).next_to(r, LEFT, buff=0.3)
            val = T(f"≈ {pr:.2f}", 26, YELLOW, font=MONO).next_to(r, RIGHT, buff=0.3)
            bars.add(VGroup(r, lab, val))
        for k, b_ in enumerate(bars):
            b_[0].move_to(LEFT * 1.6 + UP * (0.7 - 1.0 * k), aligned_edge=LEFT)
            b_[1].next_to(b_[0], LEFT, buff=0.3)
            b_[2].next_to(b_[0], RIGHT, buff=0.3)
        smp = T("1000 samples: forward 582 times", 28, GREEN).shift(DOWN * 2.3)
        src = T("source: Raschka, Build a Large Language Model (From Scratch), §5.3", 20, MUTED
                ).to_edge(DOWN, buff=0.3)
        with self.voice(N["ml4"]) as v:
            self.play(FadeIn(src), run_time=0.5)
            self.wait_until(v, "every")
            self.play(FadeIn(ctx), run_time=0.6)
            self.wait_until(v, "forward")
            self.play(GrowFromEdge(bars[0], LEFT), run_time=0.6)
            self.wait_until(v, "toward")
            self.play(GrowFromEdge(bars[1], LEFT), run_time=0.6)
            self.wait_until(v, "closer")
            self.play(GrowFromEdge(bars[2], LEFT), run_time=0.6)
            self.wait_until(v, "thousand")
            self.play(FadeIn(smp), run_time=0.6)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        q = VGroup(T("Flip two coins.  Z = number of heads.", 34, FG),
                   T("Which outcomes give Z = 1?", 34, YELLOW)).arrange(DOWN, buff=0.4).shift(UP * 1.5)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac1"]) as v:
            self.play(FadeIn(q[0]), run_time=0.8)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
            self.wait_until(v, "Which")
            self.play(FadeIn(q[1]), run_time=0.6)
        rows = VGroup(*[VGroup(T(o, 32, FG, font=MONO), T("→", 30, PURPLE), T(z, 32, YELLOW, font=MONO))
                        .arrange(RIGHT, buff=0.5) for o, z in [("HH", "2"), ("HT", "1"), ("TH", "1"), ("TT", "0")]])
        rows.arrange(DOWN, buff=0.25).next_to(q, DOWN, buff=0.5)
        hl = VGroup(SurroundingRectangle(rows[1], color=YELLOW, buff=0.08),
                    SurroundingRectangle(rows[2], color=YELLOW, buff=0.08))
        ok = T("two outcomes, same number: allowed", 26, GREEN).next_to(rows, DOWN, buff=0.4)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "four")
            self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.5), run_time=v.duration * 0.4)
            self.wait_until(v, "So")
            self.play(Create(hl), run_time=0.8)
            self.wait_until(v, "Notice")
            self.play(FadeIn(ok), run_time=0.6)
        self.clear_body()
        q2 = VGroup(T("Two fair dice.  X = sum.", 34, FG),
                    mt(r"\text{How many outcomes in } X\le 3\,?\quad \text{Its chance?}", size=44,
                       colors={0: YELLOW})).arrange(DOWN, buff=0.4).shift(UP * 1.6)
        pause2 = T("⏸  pause and try", 30, MUTED).next_to(q2, DOWN, buff=0.5)
        with self.voice(N["prac3"]) as v:
            self.wait_until(v, "Roll")
            self.play(FadeIn(q2[0]), run_time=0.6)
            self.wait_until(v, "many")
            self.play(Write(q2[1]), run_time=1)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause2), run_time=0.5)
        a = VGroup(mt(r"X=2:\ (1,1)", size=44),
                   mt(r"X=3:\ (1,2),\ (2,1)", size=44),
                   mt(r"3\text{ outcomes}\ \Rightarrow\ \tfrac{3}{36}=\tfrac{1}{12}\approx 0.083", size=44,
                      colors={0: YELLOW})).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(q2, DOWN, buff=0.5)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(pause2), run_time=0.4)
            self.wait_until(v, "two")
            self.play(FadeIn(a[0]), run_time=0.6)
            self.wait_until(v, "three")
            self.play(FadeIn(a[1]), run_time=0.6)
            self.wait_until(v, "chance")
            self.play(Write(a[2]), run_time=1)
            self.play(Indicate(a[2], color=YELLOW), run_time=0.8)


CODE = '''import numpy as np
rng = np.random.default_rng(7)               # fixed seed: repeats exactly
dice = rng.integers(1, 7, size=(10000, 2))   # 10,000 rolls of two dice
X = dice.sum(axis=1)                         # random variable X = sum
print("first 6 rolls :", dice[:6].tolist())
print("their sums X  :", X[:6])
counts = np.bincount(X, minlength=13)[2:]    # how often each sum 2..12
print(counts)
print("share X >= 10 :", (X >= 10).mean())
print("share X == 7  :", (X == 7).mean())
coin = rng.integers(0, 2, size=10)           # Y = 1 if heads else 0
print("ten coin flips Y:", coin)'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=22)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[1], color=YELLOW, buff=0.06)
            self.wait_until(v, "seed")
            self.play(Create(box), run_time=0.5)
            for i, w in [(2, "roll"), (3, "sum"), (6, "count"), (8, "share")]:
                self.wait_until(v, w)
                self.play(Transform(box, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                              buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=24, command="python demo.py").shift(DOWN * 0.3)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for i, w in [(1, "sums"), (3, "Seven"), (5, "share"), (7, "coin")]:
                self.wait_until(v, w)
                self.play(Indicate(term[1][i], color=YELLOW), run_time=0.7)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["An experiment has outcomes",
                            "Random variable = a rule: outcome → number",
                            "One experiment, many random variables (sum, max, …)",
                            "Event = a set of outcomes; equally likely → count",
                            "Discrete or continuous; seen through samples; labels, noise, data"],
                           size=28).shift(UP * 0.4)
        with self.voice(N["sum1"]) as v:
            for p in pts[:4]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 6)
        nxt = VGroup(T("Next · Video 12", 26, MUTED),
                     T("Probability distribution: the rules of chance", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.6)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Outcomes", "C2_Rule", "C3_Events", "C4_Kinds", "C5_Samples", "C6_ML",
          "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Outcomes": "Experiments and outcomes",
                  "C2_Rule": "A random variable is a rule", "C3_Events": "Events",
                  "C4_Kinds": "Discrete and continuous", "C5_Samples": "Samples",
                  "C6_ML": "Random variables in ML", "C7_Practice": "Try it yourself",
                  "C7_Code": "Python demo", "C8_Summary": "Summary", "C9_Credits": "Credits"}
