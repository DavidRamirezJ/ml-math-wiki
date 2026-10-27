"""Probability distribution — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 12 of 16"
TITLE = "Probability distribution — the rules of chance"
J = demo.joint


def mt(*parts, size=48, colors=None):
    m = MathTex(*parts, font_size=size)
    for i, c in (colors or {}).items():
        m[i].set_color(c)
    return m


def jtable(origin=LEFT * 3.0 + UP * 0.6, cw=1.9, ch=0.85, size=30):
    """Joint table. rows: rain / no rain (BLUE); cols: umbrella / no umbrella (GREEN).
    Returns dict of parts. Margin cells (row sums at col 2, col sums at row 2) are created but not shown."""
    def pos(i, j):
        return origin + RIGHT * (j - 0.5) * cw + DOWN * (i - 0.5) * ch
    cells = [[Rectangle(width=cw, height=ch, stroke_color=MUTED, stroke_width=1.5).move_to(pos(i, j))
              for j in range(2)] for i in range(2)]
    vals = [[T(f"{J[i, j]:.2f}", size, YELLOW, font=MONO).move_to(pos(i, j)) for j in range(2)]
            for i in range(2)]
    rowh = [T(s, 26, BLUE, weight=BOLD).next_to(cells[i][0], LEFT, buff=0.25)
            for i, s in enumerate(["rain", "no rain"])]
    colh = [T(s, 24 if cw >= 2.2 else 19, GREEN, weight=BOLD).next_to(cells[0][j], UP, buff=0.2)
            for j, s in enumerate(["umbrella", "no umbrella"])]
    rsum = [T(f"{demo.p_w[i]:.2f}", size, ORANGE, font=MONO).move_to(pos(i, 2) + RIGHT * 0.1)
            for i in range(2)]
    csum = [T(f"{demo.p_u[j]:.2f}", size, ORANGE, font=MONO).move_to(pos(2, j) + DOWN * 0.05)
            for j in range(2)]
    grid = VGroup(*[c for r in cells for c in r])
    return dict(cells=cells, vals=vals, rowh=rowh, colh=colh, rsum=rsum, csum=csum, grid=grid,
                frame=VGroup(grid, *rowh, *colh), allvals=VGroup(*[v for r in vals for v in r]))


def bars_h(items, origin, scale=6.0, color=YELLOW):
    g = VGroup()
    for k, (name, p) in enumerate(items):
        r = Rectangle(width=max(p * scale, 0.03), height=0.5, color=color, fill_color=color,
                      fill_opacity=0.55)
        r.move_to(origin + DOWN * 0.75 * k, aligned_edge=LEFT)
        lab = T(name, 28, BLUE).next_to(r, LEFT, buff=0.3)
        lab.align_to(origin + LEFT * 0.3, RIGHT)
        val = T(f"{p:g}", 26, color, font=MONO).next_to(r, RIGHT, buff=0.25)
        g.add(VGroup(r, lab, val))
    return g


CLS = [("cat", 0.9), ("dog", 0.08), ("fox", 0.02)]


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        if title.width > 12.5:
            title.scale_to_fit_width(12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.2)
        bars = bars_h(CLS, LEFT * 1.0 + DOWN * 0.2)
        toy = T("toy example", 20, MUTED).next_to(bars, DOWN, buff=0.4)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            for k, w in enumerate([("cat", 2), ("dog", 1), ("fox", 1)]):
                self.wait_until(v, *w)
                self.play(FadeIn(bars[k][1]), GrowFromEdge(bars[k][0], LEFT), FadeIn(bars[k][2]),
                          run_time=0.6)
            self.play(FadeIn(toy), run_time=0.4)
            self.wait_until(v, "distributions")
            self.play(Indicate(VGroup(*[b[2] for b in bars]), color=YELLOW), run_time=1)
        plan = VGroup(T("count → a table of chances", 30, FG), T("continuous: chance = area", 30, FG),
                      T("sum rule · product rule", 30, YELLOW), T("→ Bayes' theorem in 3 lines", 30, GREEN))
        plan.arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(DOWN * 0.8)
        with self.voice(N["intro2"]) as v:
            self.play(FadeOut(bars), FadeOut(toy), run_time=0.6)
            for k, w in enumerate(["counting", "areas", "sum", "Bayes"]):
                self.wait_until(v, w)
                self.play(FadeIn(plan[k], shift=RIGHT * 0.2), run_time=0.5)


class C1_Table(Studio):
    def construct(self):
        self.heading("1 · A table of chances")
        xs = list(range(2, 13))
        cw = 1.02
        x0 = -4.5

        def row(texts, y, color, size=28):
            return VGroup(*[T(t, size, color, font=MONO).move_to(RIGHT * (x0 + cw * k) + UP * y)
                            for k, t in enumerate(texts)])
        rx = row([str(x) for x in xs], 2.0, BLUE)
        rc = row([str(c) for c in demo.counts], 1.2, FG)
        lx = T("sum x", 24, BLUE).next_to(rx, LEFT, buff=0.35)
        lc = T("outcomes", 24, FG).next_to(rc, LEFT, buff=0.35)
        for m in (lx, lc):
            m.align_to(LEFT * 6.9, LEFT)
        with self.voice(N["tab1"]) as v:
            self.play(FadeIn(rx), FadeIn(lx), FadeIn(lc), run_time=0.8)
            self.wait_until(v, "Count")
            self.play(Indicate(lc), run_time=0.5)
            self.wait_until(v, "two", 2)
            self.play(FadeIn(rc[0]), run_time=0.4)
            self.wait_until(v, "three")
            self.play(FadeIn(rc[1]), run_time=0.4)
            self.wait_until(v, "seven")
            self.play(LaggedStart(*[FadeIn(r) for r in rc[2:6]], lag_ratio=0.3), run_time=1)
            self.wait_until(v, "twelve")
            self.play(LaggedStart(*[FadeIn(r) for r in rc[6:]], lag_ratio=0.3), run_time=1)
        rp = VGroup(*[MathTex(r"\tfrac{%d}{36}" % c, font_size=40, color=YELLOW)
                      .move_to(RIGHT * (x0 + cw * k) + UP * 0.3) for k, c in enumerate(demo.counts)])
        lp = MathTex(r"P(X{=}x)", font_size=34, color=YELLOW).next_to(rp, LEFT, buff=0.35).align_to(LEFT * 6.9, LEFT)
        e1 = mt(r"P(X=2)=\tfrac{1}{36}\approx 0.028", size=40).move_to(LEFT * 3.4 + DOWN * 1.2)
        e2 = mt(r"P(X=7)=\tfrac{6}{36}\approx %.3f" % demo.p_sum[5], size=40,
                colors={0: YELLOW}).next_to(e1, DOWN, buff=0.4).align_to(e1, LEFT)
        card = self.term_card("probability distribution", "The complete list of the values of a "
                              "random variable with their probabilities (or a density).", pos=DR, width=5.6)
        with self.voice(N["tab2"]) as v:
            self.wait_until(v, "Divide")
            self.play(LaggedStart(*[FadeIn(r) for r in rp], lag_ratio=0.1), FadeIn(lp), run_time=1.5)
            self.wait_until(v, "two")
            self.play(Write(e1), Indicate(rp[0], color=YELLOW), run_time=1)
            self.wait_until(v, "seven")
            self.play(Write(e2), Indicate(rp[5], color=YELLOW), run_time=1)
            self.wait_until(v, "distribution")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
        self.clear_body()
        chart = BarChart(values=[float(p) for p in demo.p_sum], bar_names=[str(x) for x in xs],
                         y_range=[0, 0.2, 0.05], x_length=7.6, y_length=4.4, bar_colors=[BLUE, YELLOW],
                         bar_width=0.7, axis_config={"font_size": 22, "color": MUTED})
        chart.move_to(LEFT * 2.6 + DOWN * 0.5)
        xl = T("sum x", 22, MUTED).next_to(chart, DOWN, buff=0.1)
        r1 = mt(r"P(X=x)\ge 0", size=44, colors={0: GREEN})
        r2 = mt(r"\textstyle\sum_x P(X=x)=1", size=44, colors={0: GREEN})
        r3 = mt(r"\tfrac{1+2+\dots+6+\dots+1}{36}=\tfrac{36}{36}=1", size=36)
        rules = VGroup(r1, r2, r3).arrange(DOWN, buff=0.5).move_to(RIGHT * 4.3 + UP * 0.4)
        if rules.width > 5.4:
            rules.scale_to_fit_width(5.4).move_to(RIGHT * 4.3 + UP * 0.4)
        with self.voice(N["tab3"]) as v:
            self.play(Create(chart), FadeIn(xl), run_time=1.5)
            self.wait_until(v, "negative")
            self.play(FadeIn(r1), run_time=0.6)
            self.wait_until(v, "add")
            self.play(FadeIn(r2), run_time=0.6)
            self.wait_until(v, "Check")
            self.play(Write(r3), run_time=1.2)
            self.wait_until(v, "Something")
            self.play(Indicate(r2, color=YELLOW), run_time=0.8)
        c1 = mt(r"\text{simulation (video 11): }\tfrac{1680}{10000}=0.168", size=36, colors={0: BLUE})
        c2 = mt(r"\text{distribution: }\tfrac{6}{36}\approx 0.1667", size=36, colors={0: YELLOW})
        c3 = T("samples estimate · distribution is exact", 22, MUTED)
        cmp_ = VGroup(c1, c2, c3).arrange(DOWN, buff=0.4).move_to(RIGHT * 4.2 + DOWN * 0.3)
        if cmp_.width > 5.6:
            cmp_.scale_to_fit_width(5.6).move_to(RIGHT * 4.2 + DOWN * 0.3)
        with self.voice(N["tab4"]) as v:
            self.play(FadeOut(rules), run_time=0.5)
            self.wait_until(v, "simulation")
            self.play(FadeIn(c1), Indicate(chart.bars[5], color=YELLOW), run_time=0.8)
            self.wait_until(v, "distribution")
            self.play(FadeIn(c2), run_time=0.6)
            self.wait_until(v, "estimate")
            self.play(FadeIn(c3), run_time=0.6)


class C2_Density(Studio):
    def construct(self):
        self.heading("2 · Continuous: probability is area")
        ax = Axes(x_range=[0, 11, 1], y_range=[0, 0.16, 0.05], x_length=7.6, y_length=4.2, tips=False,
                  axis_config={"stroke_color": MUTED, "include_numbers": True, "font_size": 22},
                  y_axis_config={"decimal_number_config": {"num_decimal_places": 2}},
                  ).shift(LEFT * 2.6 + DOWN * 0.6)
        xl = T("waiting time (minutes)", 22, MUTED).next_to(ax, DOWN, buff=0.15)
        dens = Line(ax.c2p(0, 0.1), ax.c2p(10, 0.1), color=GREEN, stroke_width=6)
        dl = mt(r"\text{density} = 0.1", size=36, colors={0: GREEN}).next_to(ax.c2p(5, 0.1), UP, buff=0.3)
        toy = T("toy example: bus every 10 minutes", 20, MUTED).to_edge(DOWN, buff=0.2)
        with self.voice(N["den1"]) as v:
            self.wait_until(v, "waiting")
            self.play(Create(ax), FadeIn(xl), FadeIn(toy), run_time=1.2)
            self.wait_until(v, "curve")
            self.play(Create(dens), run_time=1)
            self.wait_until(v, "flat")
            self.play(FadeIn(dl), run_time=0.6)
        slab = Polygon(ax.c2p(2, 0), ax.c2p(5, 0), ax.c2p(5, 0.1), ax.c2p(2, 0.1), color=YELLOW,
                       fill_color=YELLOW, fill_opacity=0.45, stroke_width=2)
        full = Polygon(ax.c2p(0, 0), ax.c2p(10, 0), ax.c2p(10, 0.1), ax.c2p(0, 0.1), color=GREEN,
                       fill_color=GREEN, fill_opacity=0.15, stroke_width=0)
        a1 = mt(r"P(2\le T\le 5)", r"=3\times 0.1", r"=0.3", size=40, colors={0: YELLOW, 2: YELLOW})
        a2 = mt(r"\text{total}=10\times 0.1=1", size=40, colors={0: GREEN})
        ag = VGroup(a1, a2).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        ag.scale_to_fit_width(min(ag.width, 3.9)).move_to(RIGHT * 4.6 + UP * 1.0)
        with self.voice(N["den2"]) as v:
            self.wait_until(v, "area")
            self.play(FadeIn(full), run_time=0.6)
            self.wait_until(v, "two")
            self.play(FadeIn(slab), run_time=0.8)
            self.wait_until(v, "rectangle")
            self.play(Write(a1), run_time=1.2)
            self.wait_until(v, "whole")
            self.play(Write(a2), Indicate(full, color=GREEN), run_time=1)
        self.clear_body()
        ax2 = Axes(x_range=[145, 196, 10], y_range=[0, 0.065, 0.02], x_length=9, y_length=4.0, tips=False,
                   axis_config={"stroke_color": MUTED, "include_numbers": True, "font_size": 22},
                   y_axis_config={"include_numbers": False}).shift(LEFT * 1.5 + DOWN * 0.7)
        f = lambda x: np.exp(-0.5 * ((x - 170) / 7) ** 2) / (7 * np.sqrt(2 * np.pi))  # noqa: E731
        bell = ax2.plot(f, x_range=[145, 195], color=GREEN, stroke_width=5)
        sl = ax2.get_area(bell, x_range=[165, 175], color=YELLOW, opacity=0.45)
        lab = T("height (cm) · illustrative bell shape", 22, MUTED).next_to(ax2, DOWN, buff=0.15)
        sll = T("slice area = P(165 ≤ height ≤ 175)", 24, YELLOW).next_to(ax2, UP, buff=0.1).shift(RIGHT * 1.2)
        line = Line(ax2.c2p(170, 0), ax2.c2p(170, f(170)), color=RED, stroke_width=4)
        z = T("one exact value: width 0 → probability 0", 24, RED).move_to(RIGHT * 4.3 + UP * 0.4)
        if z.get_right()[0] > 6.9:
            z = T("exact value:\nwidth 0 → prob. 0", 24, RED).move_to(RIGHT * 5.3 + UP * 0.6)
        with self.voice(N["den3"]) as v:
            self.wait_until(v, "height")
            self.play(Create(ax2), Create(bell), FadeIn(lab), run_time=1.5)
            self.wait_until(v, "slice")
            self.play(FadeIn(sl), FadeIn(sll), run_time=0.8)
            self.wait_until(v, "exactly")
            self.play(FadeOut(sl), Create(line), run_time=0.8)
            self.wait_until(v, "zero")
            self.play(FadeIn(z), run_time=0.6)


class C3_Joint(Studio):
    def construct(self):
        self.heading("3 · Two variables: a joint table")
        t = jtable(origin=LEFT * 0.6 + UP * 0.4, cw=2.4, ch=1.0, size=34)
        toy = T("toy example", 20, MUTED).to_edge(DOWN, buff=0.3)
        title = mt(r"p(\text{weather},\ \text{umbrella})", size=40).next_to(t["frame"], UP, buff=0.4)
        with self.voice(N["joint1"]) as v:
            self.wait_until(v, "two")
            self.play(FadeIn(toy), run_time=0.5)
            self.wait_until(v, "weather")
            self.play(FadeIn(t["rowh"][0]), FadeIn(t["rowh"][1]), run_time=0.6)
            self.wait_until(v, "umbrella")
            self.play(FadeIn(t["colh"][0]), FadeIn(t["colh"][1]), run_time=0.6)
            self.wait_until(v, "joint")
            self.play(Create(t["grid"]), Write(title), run_time=1)
        tot = mt(r"0.25+0.05+0.10+0.60=1", size=40, colors={0: GREEN}).next_to(t["grid"], DOWN, buff=0.6)
        with self.voice(N["joint2"]) as v:
            for k, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                self.wait_until(v, "rain", k + 1)
                self.play(FadeIn(t["vals"][i][j], scale=1.3), run_time=0.5)
            self.wait_until(v, "cover")
            self.play(Write(tot), run_time=1)


class C4_Rules(Studio):
    def construct(self):
        self.heading("4 · Sum rule and product rule")
        t = jtable(origin=LEFT * 3.4 + UP * 0.9, cw=1.8, ch=0.9, size=30)
        self.add(t["frame"], t["allvals"])
        rowbox = SurroundingRectangle(VGroup(t["cells"][0][0], t["cells"][0][1]), color=ORANGE, buff=0.05)
        sh = T("sum", 22, ORANGE).next_to(t["rsum"][0], UP, buff=0.25)
        f = mt(r"p(x)", "=", r"\sum_y p(x,y)", size=46, colors={0: ORANGE}).move_to(RIGHT * 3.9 + UP * 1.7)
        w = mt(r"0.25+0.05=0.30", size=40, colors={0: ORANGE}).next_to(f, DOWN, buff=0.4)
        card = self.term_card("sum rule", "To forget a variable, add over all its values. The result "
                              "is a marginal, like p(rain).", pos=DR, width=5.4)
        with self.voice(N["rule1"]) as v:
            self.wait_until(v, "Add")
            self.play(Create(rowbox), run_time=0.6)
            self.wait_until(v, "gives")
            self.play(FadeIn(t["rsum"][0]), FadeIn(sh), Write(w), run_time=0.8)
            self.wait_until(v, "sum")
            self.play(Write(f), FadeIn(card, shift=UP * 0.3), run_time=1)
            self.wait_until(v, "marginal")
            self.play(Indicate(t["rsum"][0], color=ORANGE), run_time=0.8)
        chk = mt(r"0.30+0.70=1", size=40, colors={0: GREEN}).next_to(w, DOWN, buff=0.4)
        with self.voice(N["rule2"]) as v:
            self.play(FadeOut(card), run_time=0.4)
            self.wait_until(v, "other")
            self.play(rowbox.animate.become(SurroundingRectangle(VGroup(t["cells"][1][0], t["cells"][1][1]),
                                                                  color=ORANGE, buff=0.05)),
                      FadeIn(t["rsum"][1]), run_time=0.7)
            self.wait_until(v, "columns")
            self.play(FadeOut(rowbox), FadeIn(t["csum"][0]), FadeIn(t["csum"][1]), run_time=0.8)
            self.wait_until(v, "Each")
            self.play(Write(chk), run_time=0.8)
        self.play(FadeOut(VGroup(f, w, chk, sh)), run_time=0.4)
        dim = VGroup(t["vals"][1][0], t["vals"][1][1], t["rsum"][1], t["csum"][0], t["csum"][1])
        rowbox2 = SurroundingRectangle(VGroup(t["cells"][0][0], t["cells"][0][1]), color=PURPLE, buff=0.05)
        c = mt(r"p(\text{umbrella}\mid\text{rain})", "=", r"\frac{0.25}{0.30}", r"\approx", r"%.3f" % demo.p_u_given_rain[0],
               size=42, colors={0: PURPLE, 4: YELLOW}).move_to(RIGHT * 3.4 + UP * 1.7)
        if c.get_right()[0] > 6.9:
            c.scale_to_fit_width(6.6).move_to(RIGHT * 3.5 + UP * 1.7)
        bar = T("|  means “given”", 24, MUTED).next_to(c, DOWN, buff=0.35)
        card2 = self.term_card("conditional probability", "The probability of y when we already know x: "
                               "p(y | x) = p(x, y) / p(x).", pos=DR, width=5.6)
        with self.voice(N["rule3"]) as v:
            self.wait_until(v, "know")
            self.play(dim.animate.set_opacity(0.25), Create(rowbox2), run_time=0.8)
            self.wait_until(v, "divided")
            self.play(Write(c), run_time=1.2)
            self.wait_until(v, "conditional")
            self.play(FadeIn(card2, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "bar")
            self.play(FadeIn(bar), run_time=0.5)
        pr = mt(r"p(x,y)", "=", r"p(y\mid x)", r"\,p(x)", size=52,
                colors={0: YELLOW, 2: PURPLE, 3: ORANGE}).move_to(RIGHT * 3.4 + UP * 0.3)
        num = mt(r"0.833\times 0.30=%.2f" % demo.product, size=42, colors={0: YELLOW}).next_to(pr, DOWN, buff=0.4)
        card3 = self.term_card("product rule", "Joint = conditional × marginal: p(x, y) = p(y | x) p(x).",
                               pos=DR, width=5.4)
        with self.voice(N["rule4"]) as v:
            self.play(FadeOut(card2), FadeOut(bar), run_time=0.4)
            self.wait_until(v, "equals")
            self.play(Write(pr), run_time=1.2)
            self.wait_until(v, "Check")
            self.play(Write(num), Indicate(t["vals"][0][0], color=YELLOW), run_time=1)
            self.wait_until(v, "product")
            self.play(FadeIn(card3, shift=UP * 0.3), run_time=0.6)


class C5_IndepBayes(Studio):
    def construct(self):
        self.heading("5 · Independence and Bayes")
        f = mt(r"p(x,y)", "=", r"p(x)", r"\,p(y)", size=64, colors={0: YELLOW, 2: BLUE, 3: GREEN}).shift(UP * 1.2)
        pre = mt(r"p(y\mid x)=p(y)", size=44, colors={0: PURPLE}).next_to(f, UP, buff=0.35)
        ex = mt(r"P(\text{heads},\,6)", "=", r"\tfrac12\cdot\tfrac16", "=", r"\tfrac{1}{12}", size=50,
                colors={0: YELLOW, 4: YELLOW}).next_to(f, DOWN, buff=0.7)
        exl = T("coin and die: independent", 24, MUTED).next_to(ex, DOWN, buff=0.3)
        card = self.term_card("independence", "Knowing one variable tells nothing about the other: "
                              "p(x, y) = p(x) p(y).", pos=DR, width=5.4)
        with self.voice(N["ind1"]) as v:
            self.wait_until(v, "nothing")
            self.play(FadeIn(pre), run_time=0.6)
            self.wait_until(v, "becomes")
            self.play(Write(f), run_time=1)
            self.wait_until(v, "independence")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "heads")
            self.play(Write(ex[:3]), FadeIn(exl), run_time=1)
            self.wait_until(v, "twelfth")
            self.play(Write(ex[3:]), run_time=0.6)
        self.clear_body()
        t = jtable(origin=LEFT * 3.4 + UP * 0.9, cw=1.8, ch=0.9, size=30)
        self.play(FadeIn(t["frame"]), FadeIn(t["allvals"]), *[FadeIn(m) for m in t["rsum"] + t["csum"]],
                  run_time=0.6)
        m1 = mt(r"p(\text{rain})\,p(\text{umbrella})", size=40).move_to(RIGHT * 3.5 + UP * 1.8)
        m2 = mt(r"=0.30\times 0.35=%.3f" % demo.indep[0, 0], size=40, colors={0: BLUE}).next_to(m1, DOWN, buff=0.3)
        m3 = mt(r"%.3f\ \neq\ 0.25" % demo.indep[0, 0], size=48, colors={0: RED}).next_to(m2, DOWN, buff=0.5)
        no = T("NOT independent", 30, RED, weight=BOLD).next_to(m3, DOWN, buff=0.4)
        with self.voice(N["ind2"]) as v:
            self.wait_until(v, "Multiply")
            self.play(Write(m1), Indicate(t["rsum"][0], color=ORANGE), Indicate(t["csum"][0], color=ORANGE),
                      run_time=1)
            self.play(Write(m2), run_time=0.8)
            self.wait_until(v, "table")
            self.play(Write(m3), Indicate(t["vals"][0][0], color=YELLOW), run_time=1)
            self.wait_until(v, "not")
            self.play(FadeIn(no), run_time=0.6)
        self.clear_body()
        l1 = mt(r"p(x,y)", "=", r"p(y\mid x)\,p(x)", size=50, colors={0: YELLOW, 2: PURPLE})
        l2 = mt(r"p(x,y)", "=", r"p(x\mid y)\,p(y)", size=50, colors={0: YELLOW, 2: GREEN})
        l3 = mt(r"p(x\mid y)", "=", r"\frac{p(y\mid x)\,p(x)}{p(y)}", size=58, colors={0: GREEN, 2: PURPLE})
        lines = VGroup(l1, l2, l3).arrange(DOWN, buff=0.5).shift(UP * 0.2)
        for k, l in enumerate(lines):
            l.shift(RIGHT * (lines[0][1].get_center()[0] - l[1].get_center()[0]))
        nums = VGroup(T("1", 26, MUTED), T("2", 26, MUTED), T("3", 26, MUTED))
        for n_, l in zip(nums, lines):
            n_.next_to(l, LEFT, buff=0.6).set_x(-4.5)
        box = SurroundingRectangle(l3, color=YELLOW, buff=0.15)
        with self.voice(N["bay1"]) as v:
            self.wait_until(v, "both")
            self.play(Write(l1), FadeIn(nums[0]), run_time=1)
            self.wait_until(v, "also")
            self.play(Write(l2), FadeIn(nums[1]), run_time=1)
            self.wait_until(v, "Divide")
            self.play(Write(l3), FadeIn(nums[2]), run_time=1.2)
        name = T("Bayes' theorem", 32, YELLOW, weight=BOLD).next_to(box, RIGHT, buff=0.4)
        b = mt(r"p(\text{rain}\mid\text{umbrella})", "=", r"\frac{0.833\times 0.30}{0.35}", r"\approx",
               r"%.3f" % demo.bayes, size=42, colors={0: GREEN, 4: YELLOW})
        b.next_to(lines, DOWN, buff=0.6)
        if b.width > 13:
            b.scale_to_fit_width(13)
        mol = T("a molecule built from this atom", 22, MUTED).to_edge(DOWN, buff=0.25)
        with self.voice(N["bay2"]) as v:
            self.wait_until(v, "theorem")
            self.play(Create(box), FadeIn(name), run_time=0.8)
            self.wait_until(v, "gives")
            self.play(Write(b[:3]), run_time=1.2)
            self.wait_until(v, "seven")
            self.play(Write(b[3:]), run_time=0.6)
            self.wait_until(v, "molecule")
            self.play(FadeIn(mol), run_time=0.6)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · Distributions in machine learning")
        photo = VGroup(RoundedRectangle(corner_radius=0.1, width=2.2, height=1.8, color=BLUE),
                       T("photo", 26, BLUE)).move_to(LEFT * 5.2 + UP * 0.2)
        ar = Arrow(photo.get_right(), photo.get_right() + RIGHT * 1.4, buff=0.15, color=MUTED)
        bars = bars_h(CLS, LEFT * 0.6 + UP * 0.95)
        cond = mt(r"p(\text{class}\mid\text{photo})", size=40, colors={0: PURPLE}).move_to(RIGHT * 1.6 + DOWN * 1.8)
        s1 = mt(r"\ge 0,\ \ \text{sum}=1", size=36, colors={0: GREEN}).next_to(cond, RIGHT, buff=0.6)
        toy = T("toy example", 20, MUTED).to_edge(DOWN, buff=0.3)
        with self.voice(N["ml1"]) as v:
            self.wait_until(v, "classifier")
            self.play(FadeIn(photo), GrowArrow(ar), FadeIn(toy), run_time=0.8)
            self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.3), run_time=1.2)
            self.wait_until(v, "negative")
            self.play(FadeIn(s1), run_time=0.6)
            self.wait_until(v, "conditional")
            self.play(Write(cond), run_time=1)
        self.clear_body()
        ch = mt(r"p(w_1,w_2,w_3)", "=", r"p(w_1)", r"\,p(w_2\mid w_1)", r"\,p(w_3\mid w_1,w_2)", size=46,
                colors={0: YELLOW, 2: BLUE, 3: PURPLE, 4: PURPLE}).shift(UP * 1.2)
        if ch.width > 13:
            ch.scale_to_fit_width(13)
        nt = T("next-token distribution, given the words so far", 26, MUTED).next_to(ch, UP, buff=0.5)
        lg = T("a product of many probabilities → use logs (video 10)", 26, GREEN).next_to(ch, DOWN, buff=0.8)
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "language")
            self.play(FadeIn(nt), run_time=0.6)
            self.wait_until(v, "chains")
            self.play(Write(ch), run_time=1.5)
            self.wait_until(v, "sentence")
            self.play(Indicate(ch[0], color=YELLOW), run_time=0.8)
            self.wait_until(v, "logs")
            self.play(FadeIn(lg), run_time=0.6)
        self.clear_body()
        d = mt(r"p(\text{data})", "=", r"\prod_{n=1}^{N} p(x_n)", size=60, colors={0: YELLOW, 2: BLUE}).shift(UP * 0.6)
        lab = VGroup(T("independent examples", 26, GREEN), T("same distribution", 26, GREEN)
                     ).arrange(RIGHT, buff=1.0).next_to(d, UP, buff=0.6)
        iid = T("i.i.d. = independent and identically distributed", 24, MUTED).next_to(d, DOWN, buff=0.7)
        with self.voice(N["ml3"]) as v:
            self.wait_until(v, "independence")
            self.play(FadeIn(lab[0]), run_time=0.5)
            self.wait_until(v, "same")
            self.play(FadeIn(lab[1]), run_time=0.5)
            self.wait_until(v, "product")
            self.play(Write(d), FadeIn(iid), run_time=1.2)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        t = jtable(origin=LEFT * 3.4 + UP * 0.6, cw=1.8, ch=0.9, size=30)
        q = mt(r"p(\text{no umbrella}\mid\text{rain})=\ ?", size=44, colors={0: YELLOW}).move_to(RIGHT * 3.6 + UP * 1.6)
        if q.get_right()[0] > 6.9:
            q.scale_to_fit_width(6.2).move_to(RIGHT * 3.7 + UP * 1.6)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.5)
        hint = T("hint: keep only the rain row", 24, MUTED).next_to(pause, DOWN, buff=0.4)
        with self.voice(N["prac1"]) as v:
            self.wait_until(v, "table")
            self.play(FadeIn(t["frame"]), FadeIn(t["allvals"]), run_time=0.8)
            self.wait_until(v, "probability")
            self.play(Write(q), run_time=1)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
            self.wait_until(v, "Hint")
            self.play(FadeIn(hint), run_time=0.5)
        box = SurroundingRectangle(VGroup(t["cells"][0][0], t["cells"][0][1]), color=PURPLE, buff=0.05)
        a1 = mt(r"\frac{0.05}{0.30}\approx %.3f" % demo.p_u_given_rain[1], size=46, colors={0: YELLOW})
        a2 = mt(r"0.833+0.167=1\ \checkmark", size=40, colors={0: GREEN})
        VGroup(a1, a2).arrange(DOWN, buff=0.4).next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), FadeOut(hint), run_time=0.4)
            self.wait_until(v, "rain")
            self.play(Create(box), FadeIn(t["rsum"][0]), run_time=0.6)
            self.wait_until(v, "Divide")
            self.play(Write(a1), run_time=1)
            self.wait_until(v, "Check")
            self.play(Write(a2), run_time=1)
        self.clear_body()
        q2 = mt(r"\text{Two fair dice: }P(X\ge 11)=\ ?", size=50, colors={0: YELLOW}).shift(UP * 1.6)
        pause2 = T("⏸  pause and try", 30, MUTED).next_to(q2, DOWN, buff=0.5)
        with self.voice(N["prac3"]) as v:
            self.wait_until(v, "dice")
            self.play(Write(q2), run_time=1)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause2), run_time=0.5)
        s = VGroup(mt(r"P(X=11)=\tfrac{2}{36}", size=44), mt(r"P(X=12)=\tfrac{1}{36}", size=44),
                   mt(r"P(X\ge 11)=\tfrac{2}{36}+\tfrac{1}{36}=\tfrac{3}{36}=\tfrac{1}{12}\approx 0.083", size=44,
                      colors={0: YELLOW})).arrange(DOWN, buff=0.35).next_to(q2, DOWN, buff=0.5)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(pause2), run_time=0.4)
            self.wait_until(v, "eleven")
            self.play(FadeIn(s[0]), run_time=0.5)
            self.wait_until(v, "twelve")
            self.play(FadeIn(s[1]), run_time=0.5)
            self.wait_until(v, "add")
            self.play(Write(s[2]), run_time=1.2)
            self.wait_until(v, "twelfth")
            self.play(Indicate(s[2], color=YELLOW), run_time=0.8)


CODE = '''import numpy as np
faces = np.arange(1, 7)
sums = np.add.outer(faces, faces)       # 36 outcomes
counts = np.bincount(sums.ravel())[2:]  # sums 2..12
print(counts, (counts / 36).sum())
joint = np.array([[0.25, 0.05],         # rain
                  [0.10, 0.60]])        # no rain
p_w = joint.sum(axis=1)                 # sum rule
p_u = joint.sum(axis=0)
p_u_rain = joint[0] / p_w[0]            # conditional
print(p_u_rain[0] * p_w[0])             # product rule
print(np.outer(p_w, p_u)[0, 0])         # if independent
print(p_u_rain[0] * p_w[0] / p_u[0])    # Bayes'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=22)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[2], color=YELLOW, buff=0.06)
            self.wait_until(v, "outer")
            self.play(Create(box), run_time=0.5)
            for i, w in [(3, "count"), (5, "array"), (7, "row"), (9, "conditional"), (11, "independence")]:
                self.wait_until(v, w)
                self.play(Transform(box, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                              buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=24, command="python demo.py").shift(DOWN * 0.3)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for i, w in [(0, "counts"), (5, "marginals"), (7, "product"), (8, "Independence"), (9, "Bayes")]:
                self.wait_until(v, w)
                self.play(Indicate(term[1][i], color=YELLOW), run_time=0.7)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["Distribution = every value with its probability",
                            "Probabilities ≥ 0 and add up to 1",
                            "Continuous: density, probability = area",
                            "Sum rule: p(x) = Σ_y p(x, y)  ·  product rule: p(x, y) = p(y | x) p(x)",
                            "Independence: p(x, y) = p(x) p(y)  ·  Bayes follows in 3 lines"],
                           size=28).shift(UP * 0.4)
        if pts.width > 13:
            pts.scale_to_fit_width(13)
        with self.voice(N["sum1"]) as v:
            for p in pts[:3]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 5)
        nxt = VGroup(T("Next · Video 13", 26, MUTED),
                     T("Expected value: the long-run average", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.6)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[3], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "independence")
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Table", "C2_Density", "C3_Joint", "C4_Rules", "C5_IndepBayes", "C6_ML",
          "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Table": "A table of chances",
                  "C2_Density": "Continuous: probability is area", "C3_Joint": "Two variables: a joint table",
                  "C4_Rules": "Sum rule and product rule", "C5_IndepBayes": "Independence and Bayes",
                  "C6_ML": "Distributions in ML", "C7_Practice": "Try it yourself",
                  "C7_Code": "Python demo", "C8_Summary": "Summary", "C9_Credits": "Credits"}
