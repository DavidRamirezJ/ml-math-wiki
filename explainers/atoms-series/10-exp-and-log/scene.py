"""exp & log — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 10 of 16"
TITLE = "exp & log — the pair that tames tiny numbers"


def axes(xr, yr, xl, yl):
    return Axes(x_range=xr, y_range=yr, x_length=xl, y_length=yl, tips=False,
                axis_config={"stroke_color": MUTED, "include_numbers": True,
                             "font_size": 22, "decimal_number_config": {"color": MUTED,
                                                                        "num_decimal_places": 0}})


def mt(*parts, size=48, colors=None):
    m = MathTex(*parts, font_size=size)
    for i, c in (colors or {}).items():
        m[i].set_color(c)
    return m


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        if title.width > 12.5:
            title.scale_to_fit_width(12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.2)
        prod = mt(r"0.01", r"\times", r"0.01", r"\times", r"0.01", r"\times\cdots\times",
                  r"0.01", size=52, colors={0: BLUE, 2: BLUE, 4: BLUE, 6: BLUE})
        brace = Brace(prod, DOWN, color=MUTED)
        cnt = T("1000 factors (toy example)", 24, MUTED).next_to(brace, DOWN, buff=0.15)
        res = mt(r"=\ 0.0", size=60, colors={0: RED}).next_to(prod, RIGHT, buff=0.4)
        grp = VGroup(prod, brace, cnt, res).move_to(DOWN * 0.6)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            self.wait_until(v, "multiply")
            self.play(Write(prod), GrowFromCenter(brace), FadeIn(cnt), run_time=1.5)
            self.wait_until(v, "zero")
            self.play(FadeIn(res, shift=LEFT * 0.2), run_time=0.6)
            self.wait_until(v, "Exactly")
            self.play(Indicate(res, color=RED, scale_factor=1.2), run_time=1)
        ex = T("exp", 60, GREEN, weight=BOLD)
        lg = T("log", 60, PURPLE, weight=BOLD)
        VGroup(ex, lg).arrange(RIGHT, buff=3.2).shift(DOWN * 0.7)
        a1 = CurvedArrow(ex.get_top() + UP * 0.1, lg.get_top() + UP * 0.1, angle=-PI / 3,
                         color=MUTED)
        a2 = CurvedArrow(lg.get_bottom() + DOWN * 0.1, ex.get_bottom() + DOWN * 0.1,
                         angle=-PI / 3, color=MUTED)
        und = T("one undoes the other", 26, YELLOW).next_to(a2, DOWN, buff=0.25)
        with self.voice(N["intro2"]) as v:
            self.play(FadeOut(grp), run_time=0.6)
            self.wait_until(v, "exponential")
            self.play(FadeIn(ex, shift=UP * 0.2), run_time=0.5)
            self.wait_until(v, "logarithm")
            self.play(FadeIn(lg, shift=UP * 0.2), run_time=0.5)
            self.wait_until(v, "undoes")
            self.play(Create(a1), Create(a2), FadeIn(und), run_time=1.2)
            self.wait_until(v, "logs")
            self.play(Indicate(lg, color=YELLOW), run_time=0.8)


class C1_Exp(Studio):
    def construct(self):
        self.heading("1 · Repeated growth: e to the x")
        cols = VGroup()
        for h, n in enumerate([1, 2, 4, 8]):
            dots = VGroup(*[Dot(radius=0.13, color=GREEN) for _ in range(n)])
            dots.arrange_in_grid(rows=max(1, n // 2) if n > 1 else 1, buff=0.12)
            lab = T(f"hour {h}", 24, MUTED)
            num = T(str(n), 34, YELLOW, weight=BOLD)
            c = VGroup(dots, num, lab).arrange(DOWN, buff=0.3)
            cols.add(c)
        cols.arrange(RIGHT, buff=1.3, aligned_edge=DOWN).shift(UP * 0.5)
        f = mt(r"2^3", "=", r"2\cdot 2\cdot 2", "=", "8", size=60,
               colors={0: PURPLE, 4: YELLOW}).next_to(cols, DOWN, buff=0.7)
        with self.voice(N["exp1"]) as v:
            self.play(FadeIn(cols[0]), run_time=0.6)
            for k in (1, 2, 3):
                self.wait_until(v, "After", k)
                self.play(TransformFromCopy(cols[k - 1][0], cols[k][0]), FadeIn(cols[k][1:]),
                          run_time=0.8)
            self.wait_until(v, "written")
            self.play(Write(f), run_time=1.2)
            self.wait_until(v, "counts")
            self.play(Indicate(f[0], color=PURPLE), run_time=0.8)
        self.clear_body()
        form = mt(r"\left(1+\tfrac{1}{n}\right)^{n}", size=64).shift(UP * 1.6)
        rows = [("1 payment", "2"), ("12 payments", "2.613"), ("365 payments", "2.715"),
                ("all the time", "e ≈ 2.718")]
        tab = VGroup(*[VGroup(T(a, 28, PURPLE), T(b, 32, YELLOW if i == 3 else FG, font=MONO))
                       .arrange(RIGHT, buff=0.6) for i, (a, b) in enumerate(rows)])
        for r in tab:
            r[1].shift(RIGHT * (3.0 - (r[1].get_left()[0] - r[0].get_left()[0])))
        tab.arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(form, DOWN, buff=0.6)
        note = T("growth of 100% per year, paid in n pieces", 22, MUTED).next_to(form, RIGHT,
                                                                                 buff=0.5)
        if note.get_right()[0] > 6.9:
            note.next_to(form, DOWN, buff=0.15)
            tab.next_to(note, DOWN, buff=0.4)
        with self.voice(N["exp2"]) as v:
            self.play(Write(form), FadeIn(note), run_time=1.2)
            for k, w in enumerate(["once", "twelve", "day", "e"]):
                self.wait_until(v, w)
                self.play(FadeIn(tab[k], shift=RIGHT * 0.2), run_time=0.6)
            self.play(Indicate(tab[3], color=YELLOW), run_time=0.8)
        self.clear_body()
        ax = axes([-3, 2.5, 1], [0, 8, 2], 6.2, 5.2).shift(LEFT * 3.0 + DOWN * 0.6)
        curve = ax.plot(lambda x: np.exp(x), x_range=[-3, np.log(8)], color=GREEN,
                        stroke_width=5)
        clab = MathTex("y=e^x", color=GREEN, font_size=40).next_to(ax.c2p(1.2, 7), LEFT,
                                                                   buff=0.3)
        pts = [(0, 1, "(0,\\,1)"), (1, np.e, "(1,\\,2.718)"), (2, np.exp(2), "(2,\\,7.389)")]
        dots = VGroup(*[Dot(ax.c2p(x, y), color=YELLOW, radius=0.08) for x, y, _ in pts])
        labs = VGroup(*[MathTex(s, color=YELLOW, font_size=30).next_to(d, RIGHT, buff=0.15)
                        for d, (_, _, s) in zip(dots, pts)])
        pos = T("always above 0", 26, GREEN).next_to(ax.c2p(-2.6, 0.4), UP, buff=0.15)
        card = self.term_card("exponential", "The function e to the power x, with e ≈ 2.718: "
                              "repeated growth that is always positive.", pos=DR, width=5.4)
        with self.voice(N["exp3"]) as v:
            self.play(Create(ax), run_time=1)
            self.wait_until(v, "exponential")
            self.play(Create(curve), FadeIn(clab), FadeIn(card, shift=UP * 0.3), run_time=1.5)
            self.wait_until(v, "zero")
            self.play(FadeIn(dots[0]), FadeIn(labs[0]), run_time=0.5)
            self.wait_until(v, "one", 2)
            self.play(FadeIn(dots[1]), FadeIn(labs[1]), run_time=0.5)
            self.wait_until(v, "two")
            self.play(FadeIn(dots[2]), FadeIn(labs[2]), run_time=0.5)
            self.wait_until(v, "left")
            self.play(Indicate(curve, color=GREEN), run_time=1)
            self.wait_until(v, "positive")
            self.play(FadeIn(pos), run_time=0.5)
        self.clear_body()
        rule = mt(r"e^{a+b}", "=", r"e^{a}", r"\cdot", r"e^{b}", size=72,
                  colors={0: YELLOW, 2: BLUE, 4: GREEN}).shift(UP * 1.6)
        copies = mt(r"\underbrace{e\cdot e}_{e^2}", r"\cdot", r"\underbrace{e\cdot e\cdot e}_{e^3}",
                    "=", r"e^{5}", size=56, colors={0: BLUE, 2: GREEN, 4: YELLOW})
        copies.next_to(rule, DOWN, buff=0.7)
        num = mt(r"7.389", r"\times", r"20.086", "=", r"148.41", "=", r"e^5", size=52,
                 colors={0: BLUE, 2: GREEN, 4: YELLOW, 6: YELLOW}).next_to(copies, DOWN, buff=0.6)
        leg = T("add in the exponent = multiply outside", 26, MUTED).next_to(num, DOWN, buff=0.45)
        with self.voice(N["exp4"]) as v:
            self.wait_until(v, "rule")
            self.play(Write(rule), run_time=1.5)
            self.wait_until(v, "squared")
            self.play(FadeIn(copies[0]), run_time=0.7)
            self.wait_until(v, "cubed")
            self.play(FadeIn(copies[1:3]), run_time=0.7)
            self.wait_until(v, "five")
            self.play(FadeIn(copies[3:]), FadeIn(leg), run_time=0.7)
            self.wait_until(v, "Check")
            self.play(Write(num[:3]), run_time=1)
            self.wait_until(v, "hundred")
            self.play(Write(num[3:]), run_time=1)
            self.play(Indicate(num[4], color=YELLOW), run_time=0.8)


class C2_Log(Studio):
    def construct(self):
        self.heading("2 · The logarithm: the way back")
        q = mt(r"e^{\,?}", "=", r"7.389", size=80, colors={0: PURPLE, 2: GREEN}).shift(UP * 1.4)
        ask = T("what power gives this number?", 30, YELLOW).next_to(q, DOWN, buff=0.5)
        ans = mt(r"\ln", r"(7.389)", "=", "2", size=72,
                 colors={0: PURPLE, 3: YELLOW}).next_to(ask, DOWN, buff=0.6)
        card = self.term_card("logarithm", "The power you must put on e to get a number: "
                              "ln y = x means e^x = y.", pos=DR, width=5.4)
        with self.voice(N["log1"]) as v:
            self.play(Write(q), run_time=1.2)
            self.wait_until(v, "two")
            self.play(Transform(q[0], mt(r"e^{2}", size=80, colors={0: PURPLE}).move_to(q[0])),
                      run_time=0.8)
            self.wait_until(v, "logarithm")
            self.play(FadeIn(ask), FadeIn(card, shift=UP * 0.3), run_time=0.8)
            self.wait_until(v, "natural")
            self.play(Write(ans[0]), run_time=0.6)
            self.wait_until(v, "seven", 2)
            self.play(Write(ans[1:]), run_time=1)
        self.clear_body()
        x1 = mt("x", size=56, colors={0: BLUE})
        b1 = VGroup(RoundedRectangle(corner_radius=0.15, width=1.8, height=1.0, color=GREEN),
                    T("exp", 30, GREEN))
        mid = mt(r"e^{x}", size=56, colors={0: GREEN})
        b2 = VGroup(RoundedRectangle(corner_radius=0.15, width=1.8, height=1.0, color=PURPLE),
                    T("ln", 30, PURPLE))
        x2 = mt("x", size=56, colors={0: YELLOW})
        chain = VGroup(x1, b1, mid, b2, x2).arrange(RIGHT, buff=0.9).shift(UP * 0.8)
        arrows = VGroup(*[Arrow(chain[i].get_right(), chain[i + 1].get_left(), buff=0.12,
                                color=MUTED) for i in range(4)])
        inv = mt(r"\ln\!\left(e^{x}\right)", "=", "x", size=60,
                 colors={0: PURPLE, 2: YELLOW}).next_to(chain, DOWN, buff=1.0)
        with self.voice(N["log2"]) as v:
            self.play(FadeIn(x1), run_time=0.5)
            self.wait_until(v, "machine")
            self.play(GrowArrow(arrows[0]), FadeIn(b1), GrowArrow(arrows[1]), FadeIn(mid),
                      run_time=1)
            self.wait_until(v, "machine", 2)
            self.play(GrowArrow(arrows[2]), FadeIn(b2), GrowArrow(arrows[3]), run_time=1)
            self.wait_until(v, "back")
            self.play(FadeIn(x2, shift=RIGHT * 0.2), run_time=0.5)
            self.wait_until(v, "So")
            self.play(Write(inv), run_time=1)
            self.wait_until(v, "graph")
            self.play(FadeOut(VGroup(chain, arrows, inv)), run_time=0.6)
            ax = axes([-3, 8, 1], [-3, 8, 1], 5.6, 5.6).shift(LEFT * 3.2 + DOWN * 0.75)
            ex = ax.plot(lambda x: np.exp(x), x_range=[-3, np.log(8)], color=GREEN)
            ln = ax.plot(lambda x: np.log(x), x_range=[np.exp(-3), 8], color=PURPLE,
                         stroke_width=5)
            diag = DashedLine(ax.c2p(-3, -3), ax.c2p(8, 8), color=MUTED, stroke_width=2)
            el = MathTex("e^x", color=GREEN, font_size=36).next_to(ax.c2p(1.6, 6.5), LEFT, buff=0.2)
            ll = MathTex(r"\ln x", color=PURPLE, font_size=36).next_to(ax.c2p(6.5, 1.9), UP, buff=0.2)
            dl = MathTex("y=x", color=MUTED, font_size=30).next_to(ax.c2p(7, 7), RIGHT, buff=0.1)
            self.play(Create(ax), Create(ex), FadeIn(el), run_time=1.2)
            self.wait_until(v, "mirror")
            self.play(Create(ln), FadeIn(ll), run_time=1.2)
            self.wait_until(v, "line")
            self.play(Create(diag), FadeIn(dl), run_time=0.8)
        vals = [("1", "0", 1.0), ("e", "1", np.e), ("0.5", "-0.69", 0.5)]
        tab = VGroup(*[mt(r"\ln %s" % a, "=", b, size=48, colors={0: PURPLE, 2: YELLOW})
                       for a, b, _ in vals])
        bad = mt(r"\ln 0,\ \ln(-1)", r"\ \text{do not exist}", size=40, colors={0: RED, 1: RED})
        tab.add(bad)
        tab.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to(RIGHT * 3.6 + DOWN * 0.3)
        dots = VGroup(*[Dot(ax.c2p(x, np.log(x)), color=YELLOW, radius=0.08) for _, _, x in vals])
        with self.voice(N["log3"]) as v:
            self.wait_until(v, "one")
            self.play(FadeIn(tab[0]), FadeIn(dots[0]), run_time=0.6)
            self.wait_until(v, "e", 2)
            self.play(FadeIn(tab[1]), FadeIn(dots[1]), run_time=0.6)
            self.wait_until(v, "between")
            self.play(FadeIn(tab[2]), FadeIn(dots[2]), run_time=0.6)
            self.wait_until(v, "exist")
            self.play(FadeIn(tab[3]), run_time=0.6)
            self.wait_until(v, "positive")
            self.play(Indicate(ex, color=GREEN), run_time=0.8)


class C3_Products(Studio):
    def construct(self):
        self.heading("3 · Products become sums")
        rule = mt(r"\ln(a\cdot b)", "=", r"\ln a", "+", r"\ln b", size=72,
                  colors={0: YELLOW, 2: BLUE, 4: GREEN}).shift(UP * 1.7)
        steps = VGroup(mt(r"a=e^{p},\quad b=e^{q}", size=46, colors={0: FG}),
                       mt(r"a\cdot b=e^{p}\cdot e^{q}=e^{p+q}", size=46),
                       mt(r"\ln(a\cdot b)=p+q=\ln a+\ln b", size=46, colors={0: YELLOW}))
        steps.arrange(DOWN, buff=0.4).next_to(rule, DOWN, buff=0.7)
        with self.voice(N["prod1"]) as v:
            self.play(Write(rule[0]), run_time=1)
            self.wait_until(v, "plus")
            self.play(Write(rule[1:]), run_time=1)
            self.wait_until(v, "sums")
            self.play(Indicate(rule[3], color=YELLOW), run_time=0.8)
            for k, w in enumerate(["Write", "Then", "Its"]):
                self.wait_until(v, w)
                self.play(FadeIn(steps[k], shift=UP * 0.2), run_time=0.7)
        self.clear_body()
        a, b, s = demo.log_rule[0], demo.log_rule[1], demo.log_rule[0]
        l1 = mt(r"\ln 2", r"+", r"\ln 3", "=", r"0.693", "+", r"1.099", "=", r"%.3f" % s,
                size=56, colors={0: BLUE, 2: GREEN, 4: BLUE, 6: GREEN, 8: YELLOW})
        l2 = mt(r"\ln 6", "=", r"%.3f" % b, r"\ \checkmark", size=56,
                colors={0: YELLOW, 2: YELLOW, 3: GREEN})
        VGroup(l1, l2).arrange(DOWN, buff=0.8).shift(UP * 0.3)
        if l1.width > 13:
            l1.scale_to_fit_width(13)
        note = T("6 = 2 × 3", 28, MUTED).next_to(l2, DOWN, buff=0.6)
        with self.voice(N["prod2"]) as v:
            self.wait_until(v, "two")
            self.play(Write(l1[0]), run_time=0.6)
            self.wait_until(v, "three")
            self.play(Write(l1[1:3]), run_time=0.6)
            self.play(Write(l1[3:7]), run_time=1)
            self.wait_until(v, "Add")
            self.play(Write(l1[7:]), run_time=0.8)
            self.wait_until(v, "six", 2)
            self.play(Write(l2[:3]), FadeIn(note), run_time=1)
            self.wait_until(v, "works")
            self.play(FadeIn(l2[3]), run_time=0.5)
        self.clear_body()
        r1 = mt(r"\ln(a^{n})", "=", r"n\,\ln a", size=60, colors={0: YELLOW, 2: PURPLE})
        r2 = mt(r"\ln\!\left(\tfrac{a}{b}\right)", "=", r"\ln a-\ln b", size=60,
                colors={0: YELLOW, 2: BLUE})
        n1 = T("a power is repeated multiplication", 24, MUTED)
        n2 = T("divide → subtract", 24, MUTED)
        g = VGroup(VGroup(r1, n1).arrange(DOWN, buff=0.25),
                   VGroup(r2, n2).arrange(DOWN, buff=0.25)).arrange(DOWN, buff=0.8)
        g.shift(DOWN * 0.1)
        summ = VGroup(T("× → +", 40, YELLOW, weight=BOLD), T("÷ → −", 40, YELLOW, weight=BOLD)
                      ).arrange(DOWN, buff=0.4).move_to(RIGHT * 5 + DOWN * 0.1)
        with self.voice(N["prod3"]) as v:
            self.wait_until(v, "power")
            self.play(Write(r1), run_time=1)
            self.play(FadeIn(n1), run_time=0.5)
            self.wait_until(v, "division")
            self.play(Write(r2), FadeIn(n2), run_time=1)
            self.wait_until(v, "Multiply")
            self.play(FadeIn(summ), run_time=0.8)
        self.clear_body()
        boxes = [mt(r"2\times 3", size=56), mt(r"0.693\ +\ 1.099", size=56, colors={0: PURPLE}),
                 mt(r"1.792", size=56, colors={0: PURPLE}), mt(r"6", size=64, colors={0: YELLOW})]
        flow = VGroup(*boxes).arrange(RIGHT, buff=1.2).shift(UP * 0.4)
        if flow.width > 13:
            flow.scale_to_fit_width(13)
        arrs = VGroup(*[Arrow(flow[i].get_right(), flow[i + 1].get_left(), buff=0.15,
                              color=MUTED) for i in range(3)])
        tags = VGroup(*[T(s, 22, MUTED).next_to(a, UP, buff=0.15) for s, a in
                        zip(["look up logs", "add", "look up back"], arrs)])
        hist = T("before computers: printed log tables", 28, BLUE).to_edge(DOWN, buff=1.2)
        with self.voice(N["prod4"]) as v:
            self.wait_until(v, "tables")
            self.play(FadeIn(hist), FadeIn(flow[0]), run_time=0.8)
            self.wait_until(v, "looked")
            self.play(GrowArrow(arrs[0]), FadeIn(tags[0]), FadeIn(flow[1]), run_time=0.8)
            self.wait_until(v, "added")
            self.play(GrowArrow(arrs[1]), FadeIn(tags[1]), FadeIn(flow[2]), run_time=0.8)
            self.wait_until(v, "looked", 2)
            self.play(GrowArrow(arrs[2]), FadeIn(tags[2]), FadeIn(flow[3]), run_time=0.8)
            self.wait_until(v, "today")
            self.play(Indicate(flow[3], color=YELLOW), run_time=0.8)


class C4_Tiny(Studio):
    def construct(self):
        self.heading("4 · Taming tiny probabilities")
        line = NumberLine(x_range=[-10, 0, 1], length=5.6, rotation=PI / 2, color=MUTED,
                          include_numbers=True, font_size=22, label_direction=RIGHT)
        line.move_to(RIGHT * 2.6 + DOWN * 0.55)
        ltitle = MathTex(r"\ln p", color=PURPLE, font_size=40).next_to(line, UP, buff=0.2)
        ps = [("1", 0.0), ("0.5", np.log(0.5)), ("0.01", np.log(0.01)), ("0.0001", demo.tiny_log)]
        ptitle = MathTex("p", color=BLUE, font_size=40).move_to(LEFT * 3.5 + UP * 2.35)
        rows = VGroup()
        for i, (s, lv) in enumerate(ps):
            lab = T(s, 30, BLUE, font=MONO).move_to(LEFT * 3.5 + UP * (1.5 - 1.25 * i))
            d = Dot(line.n2p(lv), color=YELLOW, radius=0.09)
            val = T(f"{lv:.2f}" if lv != 0 else "0", 26, YELLOW, font=MONO).next_to(d, LEFT, buff=0.2)
            ar = Arrow(lab.get_right(), val.get_left(), buff=0.15, color=MUTED, stroke_width=3)
            rows.add(VGroup(lab, ar, d, val))
        with self.voice(N["tiny1"]) as v:
            self.play(Create(line), FadeIn(ltitle), FadeIn(ptitle), run_time=1)
            self.wait_until(v, "Probability", 2)
            self.play(FadeIn(rows[0]), run_time=0.6)
            self.wait_until(v, "half")
            self.play(FadeIn(rows[1]), run_time=0.6)
            self.wait_until(v, "hundred")
            self.play(FadeIn(rows[2]), run_time=0.6)
            self.wait_until(v, "And", 2)
            self.play(FadeIn(rows[3]), run_time=0.6)
            self.wait_until(v, "friendly")
            self.play(Indicate(rows[3][3], color=YELLOW), run_time=0.8)
        steps = VGroup()
        for k in range(4):
            y0, y1 = line.n2p(-np.log(10) * k), line.n2p(-np.log(10) * (k + 1))
            br = BraceBetweenPoints(y0 + RIGHT * 0.9, y1 + RIGHT * 0.9, RIGHT, color=ORANGE)
            t = T("÷ 10 → − 2.30", 22, ORANGE).next_to(br, RIGHT, buff=0.1)
            steps.add(VGroup(br, t))
        with self.voice(N["tiny2"]) as v:
            self.wait_until(v, "ten")
            self.play(LaggedStart(*[FadeIn(s) for s in steps], lag_ratio=0.3), run_time=2)
            self.wait_until(v, "ruler")
            self.play(Indicate(line, color=PURPLE), run_time=1)
        self.clear_body()
        lim = mt(r"\text{float64: smallest positive} \approx 10^{-324}", size=44,
                 colors={0: FG}).shift(UP * 2.0)
        n = ValueTracker(1)

        def readout():
            k = int(round(n.get_value()))
            val = demo.run[k - 1]
            s = "0.0" if val == 0 else f"{val:.0e}"
            col = RED if val == 0 else YELLOW
            return VGroup(T(f"factors: {k}", 34, PURPLE, font=MONO),
                          T(f"product = {s}", 40, col, font=MONO)
                          ).arrange(DOWN, buff=0.35).move_to(DOWN * 0.3)
        live = always_redraw(readout)
        uf = T("underflow: rounded to zero", 30, RED).to_edge(DOWN, buff=0.9)
        with self.voice(N["tiny3"]) as v:
            self.wait_until(v, "float")
            self.play(Write(lim), run_time=1.2)
            self.wait_until(v, "underflow")
            self.play(FadeIn(uf), run_time=0.6)
            self.wait_until(v, "Multiply")
            self.add(live)
            self.play(n.animate.set_value(160), run_time=2.5, rate_func=linear)
            self.wait_until(v, "sixty", 2)
            self.play(n.animate.set_value(demo.first_zero), run_time=0.8, rate_func=linear)
            self.play(Indicate(live, color=RED), run_time=0.6)
        self.clear_body()
        e1 = mt(r"\ln\!\left(0.01^{1000}\right)", "=", r"1000\cdot\ln 0.01", size=56,
                colors={0: BLUE, 2: PURPLE}).shift(UP * 1.6)
        e2 = mt("=", r"1000\times(-4.605)", "=", r"%.2f" % demo.sum_logs, size=56,
                colors={1: PURPLE, 3: YELLOW}).next_to(e1, DOWN, buff=0.5)
        e2.align_to(e1[1], LEFT)
        VGroup(e1, e2).set_x(0)
        ok = T("a normal number ✓", 28, GREEN).next_to(e2, DOWN, buff=0.4)
        order = mt(r"p_1>p_2", r"\iff", r"\ln p_1>\ln p_2", size=50,
                   colors={0: BLUE, 2: PURPLE}).next_to(ok, DOWN, buff=0.7)
        olab = T("same order: the best answer stays the best", 24, MUTED).next_to(order, DOWN, buff=0.3)
        with self.voice(N["tiny4"]) as v:
            self.wait_until(v, "sum")
            self.play(Write(e1), run_time=1.2)
            self.wait_until(v, "thousand")
            self.play(Write(e2), run_time=1.2)
            self.wait_until(v, "normal")
            self.play(FadeIn(ok), run_time=0.5)
            self.wait_until(v, "order")
            self.play(Write(order), FadeIn(olab), run_time=1.2)


class C5_Slopes(Studio):
    def construct(self):
        self.heading("5 · The slopes of exp and log")
        ax = axes([-2, 2.2, 1], [0, 9, 2], 6.0, 5.2).shift(LEFT * 3.1 + DOWN * 0.6)
        curve = ax.plot(lambda x: np.exp(x), x_range=[-2, np.log(9)], color=GREEN, stroke_width=5)
        t = ValueTracker(0.0)

        def tangent():
            a = t.get_value()
            y = np.exp(a)
            x0, x1 = a - 1.0, a + 1.0
            x1 = min(x1, a + (9 - y) / y)
            return Line(ax.c2p(x0, y + y * (x0 - a)), ax.c2p(x1, y + y * (x1 - a)),
                        color=YELLOW, stroke_width=4)
        tan = always_redraw(tangent)
        dot = always_redraw(lambda: Dot(ax.c2p(t.get_value(), np.exp(t.get_value())),
                                        color=YELLOW, radius=0.09))
        read = always_redraw(lambda: VGroup(
            T(f"x = {t.get_value():.2f}", 28, BLUE, font=MONO),
            T(f"height = {np.exp(t.get_value()):.3f}", 28, GREEN, font=MONO),
            T(f"slope  = {np.exp(t.get_value()):.3f}", 28, YELLOW, font=MONO)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(RIGHT * 3.6 + DOWN * 0.6))
        der = mt(r"\frac{d}{dx}\,e^{x}", "=", r"e^{x}", size=60,
                 colors={0: YELLOW, 2: GREEN}).move_to(RIGHT * 3.6 + UP * 1.6)
        rem = self.reminder_card("derivative", "The slope of a curve at a point (video 07).",
                                 pos=DR, width=4.6)
        with self.voice(N["slope1"]) as v:
            self.play(Create(ax), Create(curve), run_time=1.2)
            self.wait_until(v, "seven")
            self.play(FadeIn(rem, shift=UP * 0.2), run_time=0.6)
            self.wait_until(v, "itself")
            self.play(Write(der), run_time=1)
            self.wait_until(v, "Watch")
            self.play(FadeOut(rem), run_time=0.4)
            self.add(tan, dot, read)
            self.wait_until(v, "one", 3)
            self.play(t.animate.set_value(1.0), run_time=1.5)
        self.clear_body()
        ax2 = axes([0, 5, 1], [-3, 2, 1], 6.0, 5.0).shift(LEFT * 3.1 + DOWN * 0.6)
        lc = ax2.plot(lambda x: np.log(x), x_range=[np.exp(-3), 5], color=PURPLE, stroke_width=5)
        s = ValueTracker(1.0)

        def tangent2():
            a = s.get_value()
            y, m = np.log(a), 1 / a
            h = min(1.0, 0.6 / m)
            return Line(ax2.c2p(max(a - h, 0.0), y - m * (a - max(a - h, 0.0))),
                        ax2.c2p(a + h, y + m * h), color=YELLOW, stroke_width=4)
        tan2 = always_redraw(tangent2)
        dot2 = always_redraw(lambda: Dot(ax2.c2p(s.get_value(), np.log(s.get_value())),
                                         color=YELLOW, radius=0.09))
        read2 = always_redraw(lambda: VGroup(
            T(f"x = {s.get_value():.2f}", 28, BLUE, font=MONO),
            T(f"slope = 1/x = {1 / s.get_value():.2f}", 28, YELLOW, font=MONO)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(RIGHT * 3.6 + DOWN * 0.4))
        der2 = mt(r"\frac{d}{dx}\,\ln x", "=", r"\frac{1}{x}", size=60,
                  colors={0: PURPLE, 2: YELLOW}).move_to(RIGHT * 3.6 + UP * 1.6)
        big = mt(r"x=0.01\ \Rightarrow\ \text{slope}=100", size=40,
                 colors={0: RED}).move_to(RIGHT * 3.6 + DOWN * 2.2)
        with self.voice(N["slope2"]) as v:
            self.play(Create(ax2), Create(lc), Write(der2), run_time=1.5)
            self.add(tan2, dot2, read2)
            self.wait_until(v, "four")
            self.play(s.animate.set_value(4.0), run_time=1.2)
            self.wait_until(v, "Near")
            self.play(s.animate.set_value(0.1), run_time=1.5)
            self.wait_until(v, "hundred")
            self.play(FadeIn(big), run_time=0.6)
            self.wait_until(v, "pushed")
            self.play(Indicate(big, color=RED), run_time=0.8)
        self.clear_body()
        letters = [("A", "linearity"), ("B", "inner product"), ("C", "local linearity"),
                   ("D", "probability"), ("E", "optimization")]
        pillars = VGroup()
        for L, name in letters:
            r = RoundedRectangle(corner_radius=0.1, width=2.2, height=2.0, color=BLUE)
            g = VGroup(r, VGroup(T(L, 40, BLUE, weight=BOLD), T(name, 20, FG)).arrange(DOWN, buff=0.2))
            pillars.add(g)
        pillars.arrange(RIGHT, buff=0.25).shift(UP * 0.6)
        floor = VGroup(Rectangle(width=pillars.width, height=0.9, color=YELLOW, fill_color=YELLOW,
                                 fill_opacity=0.15),
                       T("floor: school math · powers · exp · log", 26, YELLOW))
        floor[1].move_to(floor[0])
        floor.next_to(pillars, DOWN, buff=0.1)
        chain = T("+ chain rule (video 9) → slopes of most loss functions", 24, MUTED
                  ).next_to(floor, DOWN, buff=0.5)
        with self.voice(N["slope3"]) as v:
            self.wait_until(v, "axioms")
            self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.2) for p in pillars], lag_ratio=0.15),
                      run_time=1.5)
            self.wait_until(v, "floor")
            self.play(FadeIn(floor, shift=UP * 0.2), run_time=0.8)
            self.wait_until(v, "chain")
            self.play(FadeIn(chain), run_time=0.6)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · exp and log in machine learning")
        base = Line(LEFT * 6.6, RIGHT * 6.6, color=MUTED).shift(DOWN * 1.0)
        unit = 0.36
        xs = [-1.1, 0.0, 1.1]
        scores = demo.scores
        exps = np.exp(scores)

        def bars(vals, cx, color, fmt):
            g = VGroup()
            for x, val in zip(xs, vals):
                h = abs(val) * unit
                r = Rectangle(width=0.8, height=max(h, 0.02), color=color, fill_color=color,
                              fill_opacity=0.6, stroke_width=2)
                r.move_to(base.get_center() + RIGHT * (cx + x) + UP * (np.sign(val) * h / 2))
                lab = T(fmt(val), 22, color, font=MONO)
                lab.next_to(r, UP if val >= 0 else DOWN, buff=0.1)
                g.add(VGroup(r, lab))
            return g
        sb = bars(scores, -4.5, BLUE, lambda z: f"{z:g}")
        st = T("scores", 26, BLUE).move_to(LEFT * 4.5 + UP * 2.5)
        eb = bars(exps, 0.0, GREEN, lambda z: f"{z:.2f}")
        et = MathTex(r"e^{\text{score}}", color=GREEN, font_size=38).move_to(UP * 2.5)
        arr_ = Arrow(LEFT * 2.9 + UP * 0.6, LEFT * 1.8 + UP * 0.6, color=MUTED, buff=0)
        toy = T("toy example", 20, MUTED).to_edge(DOWN, buff=0.3)
        with self.voice(N["ml1"]) as v:
            self.play(Create(base), FadeIn(toy), run_time=0.6)
            self.wait_until(v, "scores")
            self.play(FadeIn(st), LaggedStart(*[GrowFromEdge(b, DOWN) for b in sb], lag_ratio=0.2),
                      run_time=1.2)
            self.wait_until(v, "negative")
            self.play(Indicate(sb[2], color=RED), run_time=0.8)
            self.wait_until(v, "exponential")
            self.play(GrowArrow(arr_), FadeIn(et), run_time=0.6)
            self.wait_until(v, "become")
            self.play(*[TransformFromCopy(a, b) for a, b in zip(sb, eb)], run_time=1.5)
            self.wait_until(v, "positive", 3)
            self.play(Indicate(eb, color=GREEN), run_time=0.8)
        pb = bars(demo.probs * 10, 4.5, YELLOW, lambda z: f"{z / 10:.3f}")
        pt = T("÷ total", 26, YELLOW).move_to(RIGHT * 4.5 + UP * 2.5)
        arr2 = Arrow(RIGHT * 1.8 + UP * 0.6, RIGHT * 2.9 + UP * 0.6, color=MUTED, buff=0)
        tot = T(f"total = {exps.sum():.2f}", 24, GREEN, font=MONO).move_to(DOWN * 1.5)
        sm = mt(r"p_i", "=", r"\frac{e^{z_i}}{\sum_j e^{z_j}}", size=44,
                colors={0: YELLOW, 2: GREEN}).move_to(RIGHT * 4.5 + DOWN * 2.2)
        one = T("sum = 1", 24, YELLOW).next_to(pb, DOWN, buff=0.1)
        name = T("softmax · a molecule (later)", 24, MUTED).next_to(toy, UP, buff=0.15)
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "total")
            self.play(FadeIn(tot), FadeIn(pt), GrowArrow(arr2), run_time=0.6)
            self.play(*[TransformFromCopy(a, b) for a, b in zip(eb, pb)], run_time=1.5)
            self.wait_until(v, "add")
            self.play(FadeIn(one), run_time=0.5)
            self.wait_until(v, "softmax")
            self.play(FadeOut(one), Write(sm), run_time=1)
            self.wait_until(v, "molecule")
            self.play(FadeIn(name), run_time=0.6)
        self.clear_body()
        pr = mt("0.9", r"\times", "0.8", r"\times", "0.5", "=", "0.36", size=56,
                colors={0: BLUE, 2: BLUE, 4: BLUE, 6: YELLOW}).shift(UP * 1.5)
        lg = mt("(-0.105)", "+", "(-0.223)", "+", "(-0.693)", "=", "-1.022", size=52,
                colors={0: PURPLE, 2: PURPLE, 4: PURPLE, 6: YELLOW}).next_to(pr, DOWN, buff=0.9)
        chk = mt(r"\ln 0.36", "=", "-1.022", r"\ \checkmark", size=52,
                 colors={0: YELLOW, 2: YELLOW, 3: GREEN}).next_to(lg, DOWN, buff=0.7)
        tl = T("product of probabilities", 24, MUTED).next_to(pr, UP, buff=0.3)
        ll = T("sum of logs = log-likelihood", 24, MUTED).next_to(lg, UP, buff=0.3)
        toy2 = T("toy example: 3 examples", 20, MUTED).to_edge(DOWN, buff=0.3)
        with self.voice(N["ml3"]) as v:
            self.wait_until(v, "multiply")
            self.play(FadeIn(tl), FadeIn(toy2), run_time=0.5)
            self.wait_until(v, "Take")
            self.play(Write(pr[:5]), run_time=1)
            self.wait_until(v, "product")
            self.play(Write(pr[5:]), run_time=0.6)
            self.wait_until(v, "sum")
            self.play(FadeIn(ll), Write(lg), run_time=1.5)
            self.wait_until(v, "exactly")
            self.play(Write(chk), run_time=1)
        self.clear_body()
        f1 = VGroup(T("maximize", 28, GREEN), mt(r"\sum_n \ln p_n", size=50))
        f2 = VGroup(T("minimize", 28, RED), mt(r"-\frac{1}{N}\sum_n \ln p_n", size=50))
        for f in (f1, f2):
            f.arrange(RIGHT, buff=0.4)
        VGroup(f1, f2).arrange(DOWN, buff=0.9, aligned_edge=LEFT).shift(LEFT * 2.2 + UP * 0.6)
        ar = Arrow(f1.get_bottom(), f2.get_top(), buff=0.12, color=MUTED)
        ce = T("= cross-entropy loss (video 15)", 26, YELLOW).next_to(f2, RIGHT, buff=0.4)
        if ce.get_right()[0] > 6.9:
            ce.next_to(f2, DOWN, buff=0.3).align_to(f2, LEFT)
        stab = T("stable code: log of softmax computed straight from scores", 24, BLUE
                 ).to_edge(DOWN, buff=0.8)
        with self.voice(N["ml4"]) as v:
            self.wait_until(v, "maximize")
            self.play(FadeIn(f1), run_time=0.8)
            self.wait_until(v, "minimize")
            self.play(GrowArrow(ar), FadeIn(f2), run_time=0.8)
            self.wait_until(v, "fifteen")
            self.play(FadeIn(ce), run_time=0.6)
            self.wait_until(v, "stability")
            self.play(FadeIn(stab), run_time=0.6)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        given = mt(r"\ln 2\approx 0.693,\qquad \ln 5\approx 1.609", size=52).shift(UP * 1.6)
        q = mt(r"\ln 10", "=", r"\;?", size=72, colors={0: YELLOW, 2: YELLOW}).next_to(given, DOWN, buff=0.7)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac1"]) as v:
            self.wait_until(v, "know")
            self.play(Write(given), run_time=1.5)
            self.wait_until(v, "Without")
            self.play(Write(q), FadeIn(pause), run_time=1)
        a1 = mt(r"\ln 10", "=", r"\ln(2\cdot 5)", "=", r"0.693+1.609", r"\approx", "2.30",
                size=52, colors={0: YELLOW, 6: YELLOW})
        a1.next_to(q, DOWN, buff=0.5)
        if a1.width > 13:
            a1.scale_to_fit_width(13)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "Ten")
            self.play(Write(a1[:3]), run_time=1)
            self.wait_until(v, "plus")
            self.play(Write(a1[3:5]), run_time=1)
            self.play(Write(a1[5:]), run_time=0.6)
            self.wait_until(v, "step")
            self.play(Indicate(a1[6], color=YELLOW), run_time=0.8)
        self.clear_body()
        q2 = mt(r"p_1=p_2=p_3=p_4=\tfrac12", size=56, colors={0: BLUE}).shift(UP * 1.8)
        qa = mt(r"p_1 p_2 p_3 p_4", "=", r"\;?\qquad", r"\sum \ln p_n", "=", r"\;?",
                size=52, colors={0: BLUE, 3: PURPLE}).next_to(q2, DOWN, buff=0.6)
        hint = mt(r"\text{hint: } \ln\tfrac12\approx -0.693", size=40, colors={0: MUTED}
                  ).next_to(qa, DOWN, buff=0.5)
        pause2 = T("⏸  pause and try", 30, MUTED).next_to(hint, DOWN, buff=0.5)
        with self.voice(N["prac3"]) as v:
            self.wait_until(v, "four")
            self.play(Write(q2), run_time=1)
            self.wait_until(v, "product")
            self.play(Write(qa[:3]), run_time=0.8)
            self.wait_until(v, "sum")
            self.play(Write(qa[3:]), run_time=0.8)
            self.wait_until(v, "Use")
            self.play(FadeIn(hint), run_time=0.5)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause2), run_time=0.5)
        s1 = mt(r"\left(\tfrac12\right)^4=\tfrac1{16}=0.0625", size=48, colors={0: YELLOW})
        s2 = mt(r"4\times(-0.693)\approx -2.77", size=48, colors={0: PURPLE})
        s3 = mt(r"\ln 0.0625\approx -2.77\ \checkmark", size=48, colors={0: GREEN})
        VGroup(s1, s2, s3).arrange(DOWN, buff=0.35).next_to(qa, DOWN, buff=0.5)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(hint), FadeOut(pause2), run_time=0.4)
            self.wait_until(v, "product")
            self.play(Write(s1), run_time=1)
            self.wait_until(v, "sum")
            self.play(Write(s2), run_time=1)
            self.wait_until(v, "indeed")
            self.play(Write(s3), run_time=1)
            self.play(Indicate(s3, color=GREEN), run_time=0.8)


CODE = '''import numpy as np
print("e^2 * e^3 =", np.exp(2) * np.exp(3))
print("e^5       =", np.exp(5))
print("ln 2 + ln 3 =", np.log(2) + np.log(3))
print("ln 6        =", np.log(6))
print("ln 0.0001 =", np.log(0.0001))
p = np.full(1000, 0.01)        # 1000 probabilities of 0.01
print("product =", np.prod(p))              # underflow!
print("sum of logs =", np.sum(np.log(p)))   # safe
run = np.cumprod(p)            # running product
print("0.0 after", np.argmax(run == 0) + 1, "factors")
s = np.array([2.0, 1.0, -1.0])            # toy scores
print("softmax =", np.exp(s) / np.exp(s).sum())'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=22)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[1], color=YELLOW, buff=0.06)
            self.wait_until(v, "exponential")
            self.play(Create(box), run_time=0.5)
            for i, w in [(3, "log"), (6, "thousand"), (7, "prod"), (8, "sum"), (10, "first")]:
                self.wait_until(v, w)
                self.play(Transform(box, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                              buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=24, command="python demo.py").shift(DOWN * 0.3)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for i, w in [(1, "squared"), (4, "six"), (6, "product"), (8, "after"), (7, "sum")]:
                self.wait_until(v, w)
                self.play(Indicate(term[1][i], color=YELLOW), run_time=0.7)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["exp: e^x, repeated growth, e ≈ 2.718, always positive",
                            "exp turns sums into products: e^(a+b) = e^a · e^b",
                            "log = what power gives this number? It undoes exp",
                            "log turns products into sums → no underflow",
                            "slopes: (e^x)' = e^x  and  (ln x)' = 1/x"],
                           size=28).shift(UP * 0.4)
        with self.voice(N["sum1"]) as v:
            for p in pts[:3]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 5)
        nxt = VGroup(T("Next · Video 11", 26, MUTED),
                     T("Random variable: turning chance into numbers", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.6)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[3], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "slope")
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Exp", "C2_Log", "C3_Products", "C4_Tiny", "C5_Slopes", "C6_ML",
          "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Exp": "Repeated growth: e to the x",
                  "C2_Log": "The logarithm", "C3_Products": "Products become sums",
                  "C4_Tiny": "Taming tiny probabilities", "C5_Slopes": "Slopes of exp and log",
                  "C6_ML": "exp and log in ML", "C7_Practice": "Try it yourself",
                  "C7_Code": "Python demo", "C8_Summary": "Summary", "C9_Credits": "Credits"}
