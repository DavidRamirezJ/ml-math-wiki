"""Expected value — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 13 of 16"
TITLE = "Expected value — the long-run average"


def mt(*parts, size=48, colors=None):
    m = MathTex(*parts, font_size=size)
    for i, c in (colors or {}).items():
        m[i].set_color(c)
    return m


def vbars(vals, labels, origin, unit, color, width=0.7, gap=1.0, fmt=None, lab_size=24):
    g = VGroup()
    for k, (v, l) in enumerate(zip(vals, labels)):
        r = Rectangle(width=width, height=max(v * unit, 0.02), color=color, fill_color=color,
                      fill_opacity=0.55, stroke_width=2)
        r.move_to(origin + RIGHT * gap * k, aligned_edge=DOWN)
        lab = T(l, lab_size, MUTED).next_to(r, DOWN, buff=0.12)
        parts = [r, lab]
        if fmt:
            parts.append(T(fmt(v), 22, color, font=MONO).next_to(r, UP, buff=0.1))
        g.add(VGroup(*parts))
    return g


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        if title.width > 12.5:
            title.scale_to_fit_width(12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.2)
        rows = VGroup(VGroup(T("one game", 30, BLUE), T("→  unknown", 30, RED)).arrange(RIGHT, buff=0.4),
                      VGroup(T("a million games", 30, BLUE), T("→  predictable average", 30, GREEN)).arrange(RIGHT, buff=0.4),
                      VGroup(T("next example", 30, PURPLE), T("→  unknown error", 30, RED)).arrange(RIGHT, buff=0.4),
                      VGroup(T("all examples", 30, PURPLE), T("→  average error", 30, GREEN)).arrange(RIGHT, buff=0.4))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(DOWN * 1.0)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            for k, w in enumerate(["casino", "million", "error", "average"]):
                self.wait_until(v, w)
                self.play(FadeIn(rows[k], shift=RIGHT * 0.2), run_time=0.5)
            self.wait_until(v, "expected")
            self.play(Indicate(title, color=YELLOW, scale_factor=1.05), run_time=0.8)
        plan = VGroup(mt(r"E[X]=\sum_x p(x)\,x", size=44, colors={0: YELLOW}), T("die → 3.5 · lottery", 28, FG),
                      T("linearity: a shortcut", 28, FG), T("simulation: the average settles", 28, FG))
        plan.arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(DOWN * 0.9)
        ax = T("axiom A (linearity) + axiom D (probability rules)", 26, MUTED).to_edge(DOWN, buff=0.4)
        with self.voice(N["intro2"]) as v:
            self.play(FadeOut(rows), run_time=0.6)
            for k, w in enumerate(["compute", "die", "linearity", "settle"]):
                self.wait_until(v, w)
                self.play(FadeIn(plan[k], shift=RIGHT * 0.2), run_time=0.5)
            self.wait_until(v, "axiom")
            self.play(FadeIn(ax), run_time=0.6)


class C1_Weighted(Studio):
    def construct(self):
        self.heading("1 · A weighted average")
        bars = vbars([80, 90], ["homework", "exam"], LEFT * 4.6 + DOWN * 2.6, 0.03, BLUE, width=1.2,
                     gap=2.2, fmt=lambda v: f"{v:g}", lab_size=26)
        wts = VGroup(T("weight 30%", 26, PURPLE).next_to(bars[0], UP, buff=0.55),
                     T("weight 70%", 26, PURPLE).next_to(bars[1], UP, buff=0.55))
        plain = mt(r"\text{plain average: }\tfrac{80+90}{2}=85", size=40, colors={0: RED}).move_to(RIGHT * 3.0 + UP * 1.6)
        with self.voice(N["avg1"]) as v:
            self.wait_until(v, "grade")
            self.play(FadeIn(bars[0][1]), FadeIn(bars[1][1]), run_time=0.5)
            self.wait_until(v, "eighty")
            self.play(GrowFromEdge(bars[0][0], DOWN), FadeIn(bars[0][2]), run_time=0.6)
            self.wait_until(v, "ninety")
            self.play(GrowFromEdge(bars[1][0], DOWN), FadeIn(bars[1][2]), run_time=0.6)
            self.wait_until(v, "thirty")
            self.play(FadeIn(wts), run_time=0.6)
            self.wait_until(v, "plain")
            self.play(Write(plain), run_time=1)
        w = mt(r"0.3\cdot 80", "+", r"0.7\cdot 90", size=48, colors={0: PURPLE, 2: PURPLE})
        w2 = mt("=", "24", "+", "63", "=", "87", size=48, colors={5: YELLOW})
        VGroup(w, w2).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to(RIGHT * 3.0 + DOWN * 0.1)
        note = T("weights add up to 1", 26, MUTED).next_to(w2, DOWN, buff=0.5)
        with self.voice(N["avg2"]) as v:
            self.wait_until(v, "multiply")
            self.play(Write(w), run_time=1)
            self.wait_until(v, "twenty")
            self.play(Write(w2[:2]), run_time=0.5)
            self.wait_until(v, "sixty")
            self.play(Write(w2[2:4]), run_time=0.5)
            self.wait_until(v, "eighty", 2)
            self.play(Write(w2[4:]), run_time=0.5)
            self.wait_until(v, "weighted")
            self.play(Indicate(w2[5], color=YELLOW), FadeIn(note), run_time=0.8)
            self.wait_until(v, "pull")
            self.play(Indicate(bars[1][0], color=BLUE), run_time=0.8)
        self.clear_body()
        a = VGroup(T("weights", 34, PURPLE, weight=BOLD), T("→", 34, MUTED),
                   T("probabilities p(x)", 34, YELLOW, weight=BOLD)).arrange(RIGHT, buff=0.4).shift(UP * 1.2)
        f = mt(r"\text{expected value}", "=", r"\sum_x", r"p(x)", r"\cdot", "x", size=50,
               colors={0: YELLOW, 3: PURPLE, 5: BLUE}).next_to(a, DOWN, buff=0.8)
        rem = self.reminder_card("probability distribution", "Every value of a random variable with its "
                                 "probability; they add up to 1 (video 12).", pos=DR, width=5.0)
        with self.voice(N["avg3"]) as v:
            self.wait_until(v, "distribution")
            self.play(FadeIn(rem, shift=UP * 0.2), FadeIn(a[0]), run_time=0.7)
            self.wait_until(v, "weights", 2)
            self.play(FadeIn(a[1:]), run_time=0.7)
            self.wait_until(v, "weighted")
            self.play(Write(f), run_time=1.2)


class C2_Definition(Studio):
    def construct(self):
        self.heading("2 · The expected value")
        f = mt(r"E[X]", "=", r"\sum_{x}", r"p(x)", r"\cdot", "x", size=72,
               colors={0: YELLOW, 2: ORANGE, 3: PURPLE, 5: BLUE}).shift(UP * 1.4 + LEFT * 2.0)
        leg = VGroup(*[VGroup(MathTex(s, font_size=36, color=c), T(t, 24, FG)).arrange(RIGHT, buff=0.3)
                       for s, c, t in [(r"E[X]", YELLOW, "expected value of X"),
                                       (r"\sum_x", ORANGE, "add over all values x"),
                                       (r"x", BLUE, "a value that can happen"),
                                       (r"p(x)", PURPLE, "its probability: the weight")]])
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(f, DOWN, buff=0.6).align_to(f, LEFT)
        card = self.term_card("expected value", "The probability-weighted average of the values: "
                              "E[X] = Σ p(x)·x. The long-run average.", pos=DR, width=5.4)
        with self.voice(N["def1"]) as v:
            self.wait_until(v, "formula")
            self.play(Write(f), FadeIn(card, shift=UP * 0.3), run_time=1.5)
            self.wait_until(v, "sum")
            self.play(FadeIn(leg[0]), FadeIn(leg[1]), run_time=0.6)
            self.wait_until(v, "values", 2)
            self.play(FadeIn(leg[2]), run_time=0.5)
            self.wait_until(v, "weight")
            self.play(FadeIn(leg[3]), run_time=0.5)
            self.wait_until(v, "Multiply")
            self.play(Indicate(f[3:], color=YELLOW), run_time=0.8)
        self.clear_body()
        bars = vbars([1 / 6] * 6, [str(k) for k in range(1, 7)], LEFT * 6.0 + DOWN * 1.8, 8.0, BLUE,
                     width=0.6, gap=0.85)
        pl = T("p = 1/6 each", 24, PURPLE).next_to(bars, UP, buff=0.3)
        c1 = mt(r"E[X]=\tfrac16\cdot1+\tfrac16\cdot2+\dots+\tfrac16\cdot6", size=42).move_to(RIGHT * 2.6 + UP * 1.0)
        c2 = mt(r"=\tfrac{1+2+3+4+5+6}{6}=\tfrac{21}{6}", size=42).next_to(c1, DOWN, buff=0.4).align_to(c1, LEFT)
        c3 = mt(r"=%.1f" % demo.e_die, size=56, colors={0: YELLOW}).next_to(c2, DOWN, buff=0.4).align_to(c1, LEFT)
        for c in (c1, c2, c3):
            if c.get_right()[0] > 6.9:
                c.shift(LEFT * (c.get_right()[0] - 6.9))
        with self.voice(N["def2"]) as v:
            self.wait_until(v, "die")
            self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.1), run_time=1)
            self.wait_until(v, "sixth")
            self.play(FadeIn(pl), run_time=0.5)
            self.wait_until(v, "plus")
            self.play(Write(c1), run_time=1.2)
            self.wait_until(v, "twenty")
            self.play(Write(c2), run_time=1)
            self.wait_until(v, "Twenty")
            self.play(Write(c3), run_time=0.8)
        self.clear_body()
        nl = NumberLine(x_range=[0.5, 6.5, 1], length=9, color=MUTED, include_numbers=False).shift(DOWN * 1.2)
        nums = VGroup(*[T(str(k), 26, MUTED).next_to(nl.n2p(k), DOWN, buff=0.45) for k in range(1, 7)])
        bl = VGroup(*[Rectangle(width=0.8, height=1.4, color=BLUE, fill_color=BLUE, fill_opacity=0.55)
                      .move_to(nl.n2p(k), aligned_edge=DOWN) for k in range(1, 7)])
        ful = Triangle(color=YELLOW, fill_color=YELLOW, fill_opacity=1).scale(0.25).next_to(nl.n2p(3.5), DOWN, buff=0.0)
        fl = T("balance point 3.5", 28, YELLOW).next_to(ful, DOWN, buff=0.75)
        nv = T("you can never roll 3.5", 30, RED).shift(UP * 2.0)
        with self.voice(N["def3"]) as v:
            self.wait_until(v, "never")
            self.play(FadeIn(nv), run_time=0.6)
            self.wait_until(v, "balance")
            self.play(Create(nl), FadeIn(nums), FadeIn(bl), run_time=1)
            self.wait_until(v, "ruler")
            self.play(FadeIn(ful, shift=UP * 0.2), FadeIn(fl), run_time=0.8)
        self.clear_body()
        hdr = VGroup(T("result", 24, MUTED), T("net win", 24, MUTED), T("probability", 24, MUTED))
        r1 = VGroup(T("win 10 €", 28, GREEN), T("+8 €", 28, GREEN, font=MONO), T("0.1", 28, PURPLE, font=MONO))
        r2 = VGroup(T("win nothing", 28, RED), T("−2 €", 28, RED, font=MONO), T("0.9", 28, PURPLE, font=MONO))
        tab = VGroup(*hdr, *r1, *r2).arrange_in_grid(3, 3, buff=(0.9, 0.4)).move_to(LEFT * 3.2 + UP * 0.6)
        cost = T("toy lottery · ticket costs 2 €", 26, BLUE).next_to(tab, UP, buff=0.5)
        e = mt(r"E[\text{win}]", "=", r"0.1\cdot 8", "+", r"0.9\cdot(-2)", size=46,
               colors={0: YELLOW, 2: GREEN, 4: RED}).move_to(DOWN * 1.6)
        with self.voice(N["def4"]) as v:
            self.wait_until(v, "ticket")
            self.play(FadeIn(cost), FadeIn(VGroup(*hdr)), run_time=0.6)
            self.wait_until(v, "eight")
            self.play(FadeIn(r1), run_time=0.6)
            self.wait_until(v, "nothing")
            self.play(FadeIn(r2), run_time=0.6)
            self.wait_until(v, "win", 5)
            self.play(Write(e), run_time=1.2)
        e2 = mt("=", r"0.8-1.8", "=", r"%.1f\text{ €}" % demo.e_lottery, size=46, colors={3: RED}).next_to(e, DOWN, buff=0.35)
        e2.align_to(e[1], LEFT)
        avg = T("on average: lose 1 € per ticket", 28, RED).move_to(RIGHT * 3.6 + UP * 0.6)
        with self.voice(N["def5"]) as v:
            self.wait_until(v, "minus")
            self.play(Write(e2), run_time=1)
            self.wait_until(v, "average")
            self.play(FadeIn(avg), run_time=0.6)
            self.wait_until(v, "single")
            self.play(Indicate(r1, color=GREEN), run_time=0.8)
            self.wait_until(v, "truth")
            self.play(Indicate(e2[3], color=RED), run_time=0.8)


class C3_Linearity(Studio):
    def construct(self):
        self.heading("3 · Linearity: the shortcut")
        f = mt(r"E[", "a", "X", "+", "b", "Y", "]", "=", "a", r"\,E[X]", "+", "b", r"\,E[Y]", size=64,
               colors={1: PURPLE, 2: BLUE, 4: PURPLE, 5: GREEN, 8: PURPLE, 9: BLUE, 11: PURPLE, 12: GREEN}).shift(UP * 1.5)
        leg = T("a, b = ordinary numbers", 26, PURPLE).next_to(f, DOWN, buff=0.4)
        axa = T("axiom A: respects adding and scaling", 30, YELLOW).next_to(leg, DOWN, buff=0.7)
        with self.voice(N["lin1"]) as v:
            self.wait_until(v, "linear")
            self.play(Write(f[:7]), run_time=1)
            self.wait_until(v, "plus", 1)
            self.play(Write(f[7:]), run_time=1.2)
            self.wait_until(v, "numbers")
            self.play(FadeIn(leg), run_time=0.5)
            self.wait_until(v, "axiom")
            self.play(FadeIn(axa), run_time=0.6)
        self.play(FadeOut(VGroup(leg, axa)), f.animate.scale(0.7).shift(UP * 0.3), run_time=0.6)
        s1 = mt(r"E[aX]=\sum_x p(x)\,(a\,x)=a\sum_x p(x)\,x=a\,E[X]", size=40).next_to(f, DOWN, buff=0.6)
        s2 = mt(r"E[X+Y]=\sum_{x,y}p(x,y)\,(x+y)=\sum_x p(x)\,x+\sum_y p(y)\,y", size=40).next_to(s1, DOWN, buff=0.45)
        for s in (s1, s2):
            if s.width > 13:
                s.scale_to_fit_width(13)
        s3 = T("sum rule (video 12) · works even if X, Y are not independent", 24, MUTED).next_to(s2, DOWN, buff=0.45)
        with self.voice(N["lin2"]) as v:
            self.wait_until(v, "Scaling")
            self.play(Write(s1), run_time=1.5)
            self.wait_until(v, "splits")
            self.play(Write(s2), run_time=1.5)
            self.wait_until(v, "independent")
            self.play(FadeIn(s3), run_time=0.6)
        self.clear_body()
        tab = T("36 products …", 32, MUTED).move_to(LEFT * 4.8 + UP * 0.8)
        cross = Cross(tab, stroke_color=RED, stroke_width=4)
        l1 = mt(r"E[X+Y]", "=", r"E[X]+E[Y]", size=54, colors={0: YELLOW}).move_to(RIGHT * 2.0 + UP * 0.8)
        l2 = mt("=", r"3.5+3.5", "=", r"%g" % demo.e_two, size=54, colors={3: YELLOW}).next_to(l1, DOWN, buff=0.4)
        l2.align_to(l1[1], LEFT)
        with self.voice(N["lin3"]) as v:
            self.wait_until(v, "Roll")
            self.play(FadeIn(tab), run_time=0.5)
            self.wait_until(v, "linearity")
            self.play(Create(cross), Write(l1), run_time=1.2)
            self.wait_until(v, "three")
            self.play(Write(l2[:2]), run_time=0.8)
            self.wait_until(v, "seven")
            self.play(Write(l2[2:]), run_time=0.6)
        self.clear_body()
        k1 = mt(r"E[2X+1]", "=", r"2\,E[X]+1", "=", r"2\cdot 3.5+1", "=", "8", size=50,
                colors={0: YELLOW, 6: YELLOW}).shift(UP * 1.3)
        if k1.width > 13:
            k1.scale_to_fit_width(13)
        warn = T("careful: only for adding and scaling", 28, RED).next_to(k1, DOWN, buff=0.7)
        k2 = mt(r"E[X^2]=\tfrac{1+4+9+16+25+36}{6}=\tfrac{91}{6}\approx 15.17", size=42).next_to(warn, DOWN, buff=0.45)
        k3 = mt(r"\neq\ (E[X])^2=3.5^2=12.25", size=42, colors={0: RED}).next_to(k2, DOWN, buff=0.35)
        with self.voice(N["lin4"]) as v:
            self.wait_until(v, "score")
            self.play(Write(k1[:3]), run_time=1)
            self.wait_until(v, "eight")
            self.play(Write(k1[3:]), run_time=1)
            self.wait_until(v, "careful")
            self.play(FadeIn(warn), run_time=0.5)
            self.wait_until(v, "curved")
            self.play(Write(k2), run_time=1.2)
            self.wait_until(v, "not")
            self.play(Write(k3), run_time=1)


class C4_LongRun(Studio):
    def construct(self):
        self.heading("4 · The long run: law of large numbers")
        rl = VGroup(*[T(str(r), 34, BLUE, font=MONO) for r in demo.rolls[:10]]).arrange(RIGHT, buff=0.45).shift(UP * 1.2)
        avg = mt(r"\frac{29}{10}=%.1f" % demo.running[9], size=56, colors={0: YELLOW}).next_to(rl, DOWN, buff=0.8)
        seed = T("seeded simulation", 22, MUTED).next_to(rl, UP, buff=0.4)
        with self.voice(N["lln1"]) as v:
            self.wait_until(v, "seed")
            self.play(FadeIn(seed), run_time=0.5)
            self.wait_until(v, "five")
            self.play(LaggedStart(*[FadeIn(r, shift=DOWN * 0.2) for r in rl], lag_ratio=0.3), run_time=2.5)
            self.wait_until(v, "only")
            self.play(Write(avg), run_time=1)
        self.clear_body()
        ax = Axes(x_range=[0, 4, 1], y_range=[1, 6, 1], x_length=9.0, y_length=4.4, tips=False,
                  axis_config={"stroke_color": MUTED}, y_axis_config={"include_numbers": True, "font_size": 22}
                  ).shift(LEFT * 1.6 + DOWN * 0.6)
        xt = VGroup(*[T(s, 20, MUTED).next_to(ax.c2p(k, 1), DOWN, buff=0.15)
                      for k, s in enumerate(["1", "10", "100", "1000", "10000"])])
        xl = T("number of rolls (log scale)", 20, MUTED).next_to(xt, DOWN, buff=0.1)
        ref = DashedLine(ax.c2p(0, 3.5), ax.c2p(4, 3.5), color=YELLOW, stroke_width=3)
        refl = T("3.5", 24, YELLOW).next_to(ax.c2p(4, 3.5), RIGHT, buff=0.1)
        ns = np.unique(np.round(np.logspace(0, 4, 400)).astype(int))
        lt = ValueTracker(0.0)

        def curve():
            m = ns[np.log10(ns) <= lt.get_value() + 1e-9]
            pts = [ax.c2p(np.log10(n), demo.running[n - 1]) for n in m]
            if len(pts) < 2:
                pts = [ax.c2p(0, demo.running[0])] * 2
            return VMobject(color=BLUE, stroke_width=3).set_points_as_corners(pts)
        live = always_redraw(curve)

        def readout():
            n = int(round(10 ** lt.get_value()))
            n = max(1, min(n, 10000))
            return VGroup(T(f"rolls: {n}", 26, BLUE, font=MONO),
                          T(f"average: {demo.running[n - 1]:.4f}", 26, YELLOW, font=MONO)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(RIGHT * 4.6 + UP * 2.2)
        read = always_redraw(readout)
        with self.voice(N["lln2"]) as v:
            self.play(Create(ax), FadeIn(xt), FadeIn(xl), Create(ref), FadeIn(refl), run_time=1)
            self.add(live, read)
            self.wait_until(v, "Watch")
            self.play(lt.animate.set_value(1), run_time=1.0, rate_func=linear)
            self.wait_until(v, "hundred")
            self.play(lt.animate.set_value(2), run_time=1.0, rate_func=linear)
            self.wait_until(v, "thousand")
            self.play(lt.animate.set_value(3), run_time=1.0, rate_func=linear)
            self.wait_until(v, "ten")
            self.play(lt.animate.set_value(4), run_time=1.2, rate_func=linear)
            self.wait_until(v, "calm")
            self.play(Indicate(ref, color=YELLOW), run_time=0.8)
        self.clear_body()
        law = T("Law of large numbers", 40, YELLOW, weight=BOLD).shift(UP * 1.6)
        f = mt(r"\bar{x}_N=\frac{1}{N}\sum_{n=1}^{N}x_n", r"\ \longrightarrow\ ", r"E[X]", size=56,
               colors={0: BLUE, 2: YELLOW}).next_to(law, DOWN, buff=0.6)
        sub = T("as N grows · close, never promised exact", 26, MUTED).next_to(f, DOWN, buff=0.5)
        use = T("why casinos can plan · why averages over data are reliable", 26, GREEN).next_to(sub, DOWN, buff=0.5)
        with self.voice(N["lln3"]) as v:
            self.wait_until(v, "law")
            self.play(Write(law), run_time=0.8)
            self.wait_until(v, "close")
            self.play(Write(f), run_time=1.2)
            self.wait_until(v, "exact")
            self.play(FadeIn(sub), run_time=0.5)
            self.wait_until(v, "casino")
            self.play(FadeIn(use), run_time=0.5)


class C5_DataMean(Studio):
    def construct(self):
        self.heading("5 · The mean of a data set")
        nl = NumberLine(x_range=[0, 11, 1], length=10, color=MUTED, include_numbers=True, font_size=24).shift(DOWN * 0.4 + LEFT * 0.5)
        pts = VGroup()
        seen = {}
        for x in demo.data:
            k = seen.get(x, 0)
            seen[x] = k + 1
            pts.add(Dot(nl.n2p(x) + UP * (0.35 + 0.4 * k), color=BLUE, radius=0.13))
        f = mt(r"\bar{x}", "=", r"\frac{2+4+4+5+10}{5}", "=", r"\frac{25}{5}", "=", "%g" % demo.data.mean(), size=46,
               colors={0: YELLOW, 6: YELLOW}).shift(UP * 2.0)
        mk = VGroup(Triangle(color=YELLOW, fill_color=YELLOW, fill_opacity=1).scale(0.2).next_to(nl.n2p(5), DOWN, buff=0.45),
                    T("mean", 22, YELLOW))
        mk[1].next_to(mk[0], DOWN, buff=0.1)
        toy = T("toy data set", 20, MUTED).to_edge(DOWN, buff=0.3)
        card = self.term_card("mean", "The average of values; for data: sum ÷ count. "
                              "The expected value is the mean of a distribution.", pos=DR, width=5.4)
        with self.voice(N["mean1"]) as v:
            self.wait_until(v, "data")
            self.play(Create(nl), LaggedStart(*[FadeIn(p, shift=DOWN * 0.3) for p in pts], lag_ratio=0.2), FadeIn(toy),
                      run_time=1.5)
            self.wait_until(v, "sum")
            self.play(Write(f[:3]), run_time=1)
            self.wait_until(v, "Twenty")
            self.play(Write(f[3:]), FadeIn(mk), run_time=1)
            self.wait_until(v, "everyday")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
        pr = VGroup(*[T("1/5", 18, PURPLE).next_to(p, RIGHT, buff=0.06) for p in pts])
        e = mt(r"E[X]=\tfrac15\cdot2+\tfrac15\cdot4+\tfrac15\cdot4+\tfrac15\cdot5+\tfrac15\cdot10=5", size=40,
               colors={0: YELLOW}).move_to(UP * 2.0)
        if e.width > 13:
            e.scale_to_fit_width(13)
        lab = T("pick a row at random", 26, PURPLE).next_to(e, DOWN, buff=0.3)
        with self.voice(N["mean2"]) as v:
            self.play(FadeOut(card), run_time=0.4)
            self.wait_until(v, "Pick")
            self.play(FadeIn(pr), FadeOut(f), FadeIn(lab), run_time=0.8)
            self.wait_until(v, "fifth", 2)
            self.play(Write(e), run_time=1.5)
            self.wait_until(v, "again")
            self.play(Indicate(mk, color=YELLOW), run_time=0.8)
        two = T("value 4: weight 2/5", 24, ORANGE).move_to(LEFT * 4.6 + DOWN * 1.6)
        nt = VGroup(mt(r"E[X]", size=40, colors={0: YELLOW}), T("true long-run average", 24, FG),
                    mt(r"\bar{x}", size=40, colors={0: BLUE}), T("mean of our samples (estimate)", 24, FG))
        nt.arrange_in_grid(2, 2, buff=(0.4, 0.3), col_alignments="rl").to_edge(DOWN, buff=0.7).shift(RIGHT * 2.5)
        with self.voice(N["mean3"]) as v:
            self.play(FadeOut(toy), run_time=0.3)
            self.wait_until(v, "twice")
            self.play(FadeIn(two), Indicate(VGroup(pts[1], pts[2]), color=ORANGE), run_time=0.8)
            self.wait_until(v, "ten")
            self.play(Indicate(pts[4], color=BLUE), run_time=0.8)
            self.wait_until(v, "bar")
            self.play(FadeIn(nt), run_time=0.8)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · Expected value in machine learning")
        flow = VGroup(T("example", 30, BLUE), T("model", 30, PURPLE), T("error = loss", 30, RED)).arrange(RIGHT, buff=1.3).shift(UP * 1.2)
        ars = VGroup(*[Arrow(flow[i].get_right(), flow[i + 1].get_left(), buff=0.15, color=MUTED) for i in range(2)])
        goal = mt(r"\text{goal: make the average loss small}", size=44, colors={0: YELLOW}).next_to(flow, DOWN, buff=1.0)
        v15 = T("loss functions → video 15", 24, MUTED).next_to(goal, DOWN, buff=0.4)
        with self.voice(N["ml1"]) as v:
            self.play(FadeIn(flow[0]), run_time=0.5)
            self.wait_until(v, "error")
            self.play(GrowArrow(ars[0]), FadeIn(flow[1]), GrowArrow(ars[1]), FadeIn(flow[2]), run_time=1)
            self.wait_until(v, "fifteen")
            self.play(FadeIn(v15), run_time=0.5)
            self.wait_until(v, "average")
            self.play(Write(goal), run_time=1)
        self.clear_body()
        losses = [0.2, 0.5, 0.1, 1.2]
        bars = vbars(losses, ["ex 1", "ex 2", "ex 3", "ex 4"], LEFT * 5.4 + DOWN * 2.0, 2.6, RED, width=0.8, gap=1.3,
                     fmt=lambda v: f"{v:g}")
        mline = DashedLine(LEFT * 6.0 + DOWN * 2.0 + UP * 0.5 * 2.6, LEFT * 0.9 + DOWN * 2.0 + UP * 0.5 * 2.6, color=YELLOW)
        ml = T("average 0.5", 24, YELLOW).next_to(mline, RIGHT, buff=0.15)
        f = mt(r"\frac{0.2+0.5+0.1+1.2}{4}=\frac{2}{4}=0.5", size=44, colors={0: YELLOW}).move_to(RIGHT * 3.6 + UP * 1.4)
        if f.get_right()[0] > 6.9:
            f.scale_to_fit_width(5.8).move_to(RIGHT * 3.8 + UP * 1.4)
        est = T("≈ expected loss on new data", 26, GREEN).next_to(f, DOWN, buff=0.5)
        toy = T("toy example", 20, MUTED).to_edge(DOWN, buff=0.2)
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "four")
            self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.2), FadeIn(toy), run_time=1.5)
            self.wait_until(v, "average")
            self.play(Write(f), Create(mline), FadeIn(ml), run_time=1.2)
            self.wait_until(v, "estimate")
            self.play(FadeIn(est), run_time=0.6)
        self.clear_body()
        g = mt(r"\frac{1}{N}\sum_{n=1}^{N}\ell_n", r"\ \approx\ ", r"E[\ell]", size=60,
               colors={0: RED, 2: YELLOW}).shift(UP * 1.4)
        items = VGroup(T("split into pieces (linearity)", 28, FG), T("estimate from small random batches", 28, FG),
                       T("take derivatives inside the sum", 28, FG)).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        items.next_to(g, DOWN, buff=0.6)
        with self.voice(N["ml3"]) as v:
            self.play(Write(g), run_time=1.2)
            for k, w in enumerate(["split", "batches", "derivatives"]):
                self.wait_until(v, w)
                self.play(FadeIn(items[k], shift=RIGHT * 0.2), run_time=0.5)
            self.wait_until(v, "smaller")
            self.play(Indicate(g[2], color=YELLOW), run_time=0.8)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        hdr = VGroup(mt("x", size=40, colors={0: BLUE}), mt("0", size=40), mt("2", size=40), mt("10", size=40))
        prow = VGroup(mt("p(x)", size=40, colors={0: PURPLE}), mt("0.5", size=40), mt("0.3", size=40), mt("0.2", size=40))
        tab = VGroup(*hdr, *prow).arrange_in_grid(2, 4, buff=(1.0, 0.4)).shift(UP * 1.5)
        q = mt(r"E[X]=\ ?", size=56, colors={0: YELLOW}).next_to(tab, DOWN, buff=0.5)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.4)
        with self.voice(N["prac1"]) as v:
            self.wait_until(v, "value")
            self.play(FadeIn(tab), run_time=1)
            self.wait_until(v, "expected")
            self.play(Write(q), run_time=0.6)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
        a = mt(r"0.5\cdot0+0.3\cdot2+0.2\cdot10", "=", r"0+0.6+2", "=", "2.6", size=48,
               colors={4: YELLOW}).next_to(q, DOWN, buff=0.4)
        if a.width > 13:
            a.scale_to_fit_width(13)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "Multiply")
            self.play(Write(a[0]), run_time=1)
            self.wait_until(v, "two")
            self.play(Write(a[1:3]), run_time=1)
            self.wait_until(v, "sum")
            self.play(Write(a[3:]), run_time=0.6)
            self.wait_until(v, "rare")
            self.play(Indicate(hdr[3], color=BLUE), run_time=0.8)
        self.clear_body()
        q2 = VGroup(mt(r"\text{three dice: } E[X_1+X_2+X_3]=\ ?", size=46, colors={0: YELLOW}),
                    mt(r"\text{one die: } E[3X-2]=\ ?", size=46, colors={0: YELLOW})).arrange(DOWN, buff=0.4).shift(UP * 1.5)
        pause2 = T("⏸  pause and try", 30, MUTED).next_to(q2, DOWN, buff=0.4)
        with self.voice(N["prac3"]) as v:
            self.wait_until(v, "three")
            self.play(Write(q2[0]), run_time=1)
            self.wait_until(v, "times")
            self.play(Write(q2[1]), run_time=1)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause2), run_time=0.5)
        b1 = mt(r"3.5+3.5+3.5=10.5", size=46, colors={0: GREEN})
        b2 = mt(r"3\cdot 3.5-2=10.5-2=8.5", size=46, colors={0: GREEN})
        VGroup(b1, b2).arrange(DOWN, buff=0.4).next_to(q2, DOWN, buff=0.5)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(pause2), run_time=0.4)
            self.wait_until(v, "add")
            self.play(Write(b1), run_time=1)
            self.wait_until(v, "multiply")
            self.play(Write(b2), run_time=1)
            self.wait_until(v, "eight")
            self.play(Indicate(b2, color=YELLOW), run_time=0.8)


CODE = '''import numpy as np
faces, p = np.arange(1, 7), np.full(6, 1 / 6)
print("E[die] =", np.sum(faces * p))          # sum of p(x) * x
win, p_win = np.array([8, -2]), np.array([0.1, 0.9])
print("E[lottery win] =", np.sum(win * p_win))
sums = np.add.outer(faces, faces).ravel()   # 36 outcomes
print("E[two dice] =", sums.mean())
rng = np.random.default_rng(3)              # fixed seed
rolls = rng.integers(1, 7, size=10000)
running = np.cumsum(rolls) / np.arange(1, 10001)
print(running[[9, 99, 999, 9999]])          # running averages
print("mean of data =", np.mean([2, 4, 4, 5, 10]))'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=22)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[2], color=YELLOW, buff=0.06)
            self.wait_until(v, "probability")
            self.play(Create(box), run_time=0.5)
            for i, w in [(4, "lottery"), (6, "thirty"), (8, "ten"), (9, "cumulative")]:
                self.wait_until(v, w)
                self.play(Transform(box, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                              buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=24, command="python demo.py").shift(DOWN * 0.3)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for i, w in [(0, "die"), (1, "lottery"), (3, "first"), (7, "Then"), (8, "data")]:
                self.wait_until(v, w)
                self.play(Indicate(term[1][i], color=YELLOW), run_time=0.7)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["Expected value = probability-weighted average: E[X] = Σ p(x)·x",
                            "Fair die: 3.5, a value you can never roll",
                            "Linear: E[aX + bY] = a E[X] + b E[Y]  (two dice → 7)",
                            "Law of large numbers: sample averages approach E[X]",
                            "Data mean = expected value; training minimizes an average loss"],
                           size=28).shift(UP * 0.4)
        if pts.width > 13:
            pts.scale_to_fit_width(13)
        with self.voice(N["sum1"]) as v:
            for p in pts[:3]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 5)
        nxt = VGroup(T("Next · Video 14", 26, MUTED),
                     T("Variance & covariance: how values spread", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.6)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[3], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "mean")
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Weighted", "C2_Definition", "C3_Linearity", "C4_LongRun", "C5_DataMean", "C6_ML",
          "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Weighted": "A weighted average",
                  "C2_Definition": "The expected value", "C3_Linearity": "Linearity: the shortcut",
                  "C4_LongRun": "Law of large numbers", "C5_DataMean": "The mean of a data set",
                  "C6_ML": "Expected value in ML", "C7_Practice": "Try it yourself",
                  "C7_Code": "Python demo", "C8_Summary": "Summary", "C9_Credits": "Credits"}
