"""Variance & covariance — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 14 of 16"
TITLE = "Variance & covariance — spread and moving together"


def fmt(v):
    return f"{v:g}"


def number_line(xr, length):
    return NumberLine(x_range=[xr[0], xr[1], 1], length=length, color=MUTED,
                      include_numbers=True, font_size=24)


def dots_on(nl, vals, color, r=0.12, step=0.3):
    seen, g = {}, VGroup()
    for v in vals:
        k = seen.get(v, 0)
        seen[v] = k + 1
        g.add(Dot(nl.n2p(v) + UP * (0.28 + step * k), radius=r, color=color))
    return g


def mean_line(nl, m, h=1.3):
    return DashedLine(nl.n2p(m) + DOWN * 0.15, nl.n2p(m) + UP * h, color=YELLOW, stroke_width=4)


def row(label, vals, color, lw=1.7, cw=0.85, size=34):
    lab = MathTex(label, color=color, font_size=size)
    cells = VGroup(*[MathTex(v, color=color, font_size=size) for v in vals])
    for i, c in enumerate(cells):
        c.move_to(RIGHT * (lw + i * cw))
    lab.move_to(RIGHT * 0).align_to(ORIGIN, RIGHT).shift(RIGHT * (lw - 0.5))
    return VGroup(lab, cells)


def scatter(xs, ys, color, xl=3.4, yl=2.8, rng=3):
    ax = Axes(x_range=[-rng, rng, 1], y_range=[-rng, rng, 1], x_length=xl, y_length=yl,
              tips=False, axis_config={"stroke_color": MUTED, "include_ticks": False})
    pts = VGroup(*[Dot(ax.c2p(a, b), radius=0.045, color=color) for a, b in zip(xs, ys)])
    return VGroup(ax, pts)


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        if title.width > 12.5:
            title.scale_to_fit_width(12.5)
        g = VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.4)
        nla = number_line((0, 10), 9).shift(UP * 0.0)
        nlb = number_line((0, 10), 9).shift(DOWN * 2.3)
        da = dots_on(nla, demo.A, BLUE)
        db = dots_on(nlb, demo.B, GREEN)
        la = T("class A", 26, BLUE, weight=BOLD).next_to(nla, LEFT, buff=0.3).shift(UP * 0.3)
        lb = T("class B", 26, GREEN, weight=BOLD).next_to(nlb, LEFT, buff=0.3).shift(UP * 0.3)
        ma, mb = mean_line(nla, 5), mean_line(nlb, 5)
        mt = T("both means = 5", 26, YELLOW).move_to(RIGHT * 4.6 + DOWN * 1.0)
        toy = T("toy quiz scores out of 10", 20, MUTED).to_edge(DOWN, buff=0.15)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            self.play(Create(nla), Create(nlb), FadeIn(la), FadeIn(lb), FadeIn(toy), run_time=1)
            self.play(Create(ma), Create(mb), FadeIn(mt), run_time=0.8)
            self.wait_until(v, "almost")
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in da], lag_ratio=0.2), run_time=1)
            self.wait_until(v, "other")
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in db], lag_ratio=0.2), run_time=1)
            self.wait_until(v, "spread")
            self.play(Indicate(db, color=YELLOW), run_time=1)
        self.play(*[FadeOut(m) for m in [nla, nlb, da, db, la, lb, ma, mb, mt, toy]], run_time=0.6)
        items = VGroup(
            T("rescale every input → spread 1", 30, FG),
            T("GPT: layer normalization in every block", 30, FG),
            T("you will compute: variance · standard deviation · covariance", 28, YELLOW),
            T("you need: video 13 (expected value)", 26, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.45).shift(DOWN * 0.7)
        with self.voice(N["intro2"]) as v:
            self.wait_until(v, "rescale")
            self.play(FadeIn(items[0], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "layer")
            self.play(FadeIn(items[1], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "compute")
            self.play(FadeIn(items[2], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "thirteen")
            self.play(FadeIn(items[3], shift=RIGHT * 0.2), run_time=0.6)


class C1_Spread(Studio):
    def construct(self):
        self.heading("1 · Same mean, different spread")
        nla = number_line((0, 10), 9).shift(RIGHT * 0.6 + UP * 0.2)
        nlb = number_line((0, 10), 9).shift(RIGHT * 0.6 + DOWN * 2.6)
        da = dots_on(nla, demo.A, BLUE)
        db = dots_on(nlb, demo.B, GREEN)
        la = T("class A", 26, BLUE, weight=BOLD).next_to(nla, LEFT, buff=0.3).shift(UP * 0.3)
        lb = T("class B", 26, GREEN, weight=BOLD).next_to(nlb, LEFT, buff=0.3).shift(UP * 0.3)
        ma, mb = mean_line(nla, 5), mean_line(nlb, 5)
        mul = MathTex(r"\mu = 5", color=YELLOW, font_size=36).next_to(ma, UP, buff=0.1)
        mub = MathTex(r"\mu = 5", color=YELLOW, font_size=36).next_to(mb, UP, buff=0.1)
        toy = T("toy example · 5 students per class", 22, MUTED).move_to(LEFT * 2.5 + UP * 2.4)
        with self.voice(N["spread1"]) as v:
            self.play(Create(nla), Create(nlb), FadeIn(la), FadeIn(lb), run_time=1.2)
            self.wait_until(v, "toy")
            self.play(FadeIn(toy), run_time=0.5)
            self.wait_until(v, "four")
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in da], lag_ratio=0.2), run_time=1.2)
            self.wait_until(v, "two", 2)
            self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in db], lag_ratio=0.2), run_time=1.2)
            self.wait_until(v, "means")
            self.play(Create(ma), Create(mb), FadeIn(mul), FadeIn(mub), run_time=1)
        card = self.reminder_card("expected value", "Each value times its probability, "
                                  "all added up. For data: the mean.", pos=UR, width=4.6)
        p = MathTex(r"p = \tfrac15 \text{ each}", color=PURPLE, font_size=34).move_to(LEFT * 2.5 + UP * 1.7)
        with self.voice(N["spread2"]) as v:
            self.play(FadeIn(card, shift=DOWN * 0.2), run_time=0.6)
            self.wait_until(v, "fifth")
            self.play(FadeIn(p), run_time=0.6)
            self.wait_until(v, "balance")
            self.play(Indicate(VGroup(ma, mb, mul, mub), color=YELLOW), run_time=1)
            self.wait_until(v, "not")
            self.play(Indicate(db, color=GREEN), run_time=1)
        devs = VGroup()
        labs = VGroup()
        y0 = nlb.n2p(5)[1] + 1.25
        for i, x in enumerate(demo.B):
            d = demo.dev[i]
            yy = y0 + 0.0
            if d != 0:
                devs.add(Arrow(nlb.n2p(5) + UP * 1.25 * 0 + UP * (0.28 + 0.0) + UP * 0,
                               nlb.n2p(x) + UP * 0.28, buff=0, color=ORANGE, stroke_width=4,
                               max_tip_length_to_length_ratio=0.15).shift(UP * (0.0)))
            lab = MathTex(f"{d:+g}" if d else "0", color=ORANGE, font_size=30)
            lab.move_to(nlb.n2p(x) + UP * 1.35)
            if d == 0:
                lab.set_opacity(0)
            labs.add(lab)
        # stack arrows at different heights so they don't overlap
        for k, a in enumerate(devs):
            a.shift(UP * [0.67, 0.34, 0.34, 0.67][k])
        devl = T("deviation = value − mean", 24, ORANGE).move_to(LEFT * 2.5 + UP * 1.2)
        with self.voice(N["spread3"]) as v:
            self.play(FadeOut(card), run_time=0.4)
            self.wait_until(v, "deviation")
            self.play(FadeIn(devl), run_time=0.5)
            self.wait_until(v, "minus")
            self.play(LaggedStart(*[GrowArrow(a) for a in devs], lag_ratio=0.2),
                      LaggedStart(*[FadeIn(l) for l in labs], lag_ratio=0.2), run_time=2)
            self.wait_until(v, "left")
            self.play(Indicate(VGroup(labs[0], labs[1]), color=RED), run_time=0.8)
            self.wait_until(v, "right")
            self.play(Indicate(VGroup(labs[3], labs[4]), color=GREEN), run_time=0.8)
        s = MathTex(r"\tfrac15\big(", "-3", "-1", "+0", "+1", "+3", r"\big)", "=", "0",
                    font_size=36)
        s[1:6].set_color(ORANGE)
        s[8].set_color(RED)
        s.move_to(LEFT * 3.7 + UP * 1.8)
        warn = T("always 0 → useless", 26, RED).next_to(s, DOWN, buff=0.2).align_to(s, LEFT)
        with self.voice(N["spread4"]) as v:
            self.play(FadeOut(p), FadeOut(devl), Write(s[:7]), run_time=1.2)
            self.wait_until(v, "zero", 2)
            self.play(Write(s[7:]), run_time=0.8)
            self.wait_until(v, "cancel")
            self.play(Indicate(devs, color=RED), run_time=1)
            self.wait_until(v, "nothing")
            self.play(FadeIn(warn), run_time=0.6)


class C2_Variance(Studio):
    def construct(self):
        self.heading("2 · Variance")
        nl = number_line((1, 9), 6).move_to(LEFT * 3.6 + DOWN * 2.6)
        db = dots_on(nl, demo.B, GREEN, r=0.1)
        ml = mean_line(nl, 5, h=2.9)
        u = nl.n2p(1)[0] - nl.n2p(0)[0] if False else (nl.n2p(2)[0] - nl.n2p(1)[0])
        sq = VGroup()
        sql = VGroup()
        for x, d in zip(demo.B, demo.dev):
            if d == 0:
                continue
            side = abs(d) * u
            r = Square(side, color=ORANGE, stroke_width=3, fill_color=ORANGE, fill_opacity=0.15)
            cx = (nl.n2p(x)[0] + nl.n2p(5)[0]) / 2
            r.move_to([cx, nl.n2p(5)[1] + side / 2 + 0.05, 0])
            sq.add(r)
            t = MathTex(fmt(d * d), color=ORANGE, font_size=32 if abs(d) > 1 else 24)
            t.move_to(r.get_center() if abs(d) > 1 else r.get_top() + UP * 0.2)
            sql.add(t)
        with self.voice(N["var1"]) as v:
            self.play(Create(nl), FadeIn(db), Create(ml), run_time=1.2)
            self.wait_until(v, "square", 2)
            self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in sq], lag_ratio=0.2), run_time=1.6)
            self.wait_until(v, "nine")
            self.play(FadeIn(sql[0]), run_time=0.5)
            self.wait_until(v, "nine", 2)
            self.play(FadeIn(sql[3]), run_time=0.5)
            self.wait_until(v, "Big")
            self.play(FadeIn(sql[1]), FadeIn(sql[2]), run_time=0.6)
        f = MathTex(r"\mathrm{Var}(X)", "=", r"\mathbb{E}", r"\big[(", "X", "-", r"\mu", r")^2", r"\big]",
                    font_size=56)
        f[0].set_color(YELLOW); f[2].set_color(PURPLE); f[4].set_color(BLUE); f[6].set_color(YELLOW)
        f[7].set_color(ORANGE)
        f.move_to(RIGHT * 3.4 + UP * 2.0)
        legend = VGroup(
            T("X  =  the random value", 24, BLUE),
            T("μ  =  its mean", 24, YELLOW),
            T("X − μ  =  the deviation", 24, ORANGE),
            T("( )²  =  squared", 24, ORANGE),
            T("E[ ]  =  average with probabilities", 24, PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(f, DOWN, buff=0.35)
        legend.set_x(3.4)
        card = self.term_card("variance", "The expected squared distance from the mean: "
                              "a number for how spread out the values are.", pos=DR, width=5.4)
        with self.voice(N["var2"]) as v:
            self.wait_until(v, "definition")
            self.play(Write(f), run_time=1.5)
            for i, (w, n) in enumerate([("random", 1), ("mu", 2), ("deviation", 1),
                                        ("Then", 1), ("averages", 1)]):
                self.wait_until(v, w, n)
                self.play(FadeIn(legend[i], shift=RIGHT * 0.2), run_time=0.5)
            self.wait_until(v, "probabilities")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
        tb = VGroup(row(r"x", [fmt(a) for a in demo.B], GREEN),
                    row(r"x-\mu", [fmt(a) for a in demo.dev], ORANGE),
                    row(r"(x-\mu)^2", [fmt(a) for a in demo.dev ** 2], ORANGE))
        tb.arrange(DOWN, buff=0.3, aligned_edge=RIGHT)
        tb.move_to(RIGHT * 3.4 + UP * 0.3)
        calc = MathTex(r"\mathrm{Var}_B", "=", r"\frac{9+1+0+1+9}{5}", "=", r"\frac{20}{5}", "=",
                       fmt(demo.var_B), font_size=40)
        calc[0].set_color(YELLOW); calc[6].set_color(YELLOW)
        calc.next_to(tb, DOWN, buff=0.4).set_x(3.4)
        with self.voice(N["var3"]) as v:
            self.play(FadeOut(legend), FadeOut(card), FadeIn(tb[0]), FadeIn(tb[1]), run_time=0.8)
            self.wait_until(v, "squares")
            self.play(FadeIn(tb[2]), run_time=0.6)
            self.wait_until(v, "twenty")
            self.play(Write(calc[:3]), run_time=1)
            self.wait_until(v, "divide")
            self.play(Write(calc[3:5]), run_time=0.8)
            self.wait_until(v, "variance")
            self.play(Write(calc[5:]), run_time=0.6)
            self.play(Indicate(calc[6], color=YELLOW), run_time=0.8)
        calcA = MathTex(r"\mathrm{Var}_A", "=", r"\frac{1+0+0+0+1}{5}", "=", r"\frac{2}{5}", "=",
                        fmt(round(demo.var_A, 2)), font_size=40)
        calcA[0].set_color(BLUE); calcA[6].set_color(BLUE)
        calcA.next_to(calc, DOWN, buff=0.35).set_x(3.4)
        ten = T("10× more spread", 26, YELLOW).next_to(calcA, DOWN, buff=0.25).set_x(3.4)
        with self.voice(N["var4"]) as v:
            self.wait_until(v, "squares")
            self.play(Write(calcA[:3]), run_time=1)
            self.wait_until(v, "two", 2)
            self.play(Write(calcA[3:]), run_time=1)
            self.wait_until(v, "Ten")
            self.play(FadeIn(ten), run_time=0.6)
        self.clear_body()
        facts = VGroup(
            MathTex(r"\mathrm{Var}(X) \ge 0", font_size=48),
            MathTex(r"\mathrm{Var}(X) = 0 \iff \text{every } x = \mu", font_size=44),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(LEFT * 2.4 + UP * 1.0)
        fl = VGroup(T("an average of squares", 24, MUTED).next_to(facts[0], RIGHT, buff=0.5),
                    T("no spread at all", 24, MUTED).next_to(facts[1], RIGHT, buff=0.5))
        nl0 = number_line((1, 9), 6).move_to(LEFT * 2.6 + DOWN * 2.4)
        d0 = dots_on(nl0, [5] * 5, YELLOW, r=0.1, step=0.24)
        mis = VGroup(T("common mistake", 26, RED, weight=BOLD),
                     MathTex(r"\mathbb{E}[X-\mu] = 0 \text{ always}", color=RED, font_size=40)
                     ).arrange(DOWN, buff=0.2).move_to(RIGHT * 3.8 + DOWN * 1.9)
        with self.voice(N["var5"]) as v:
            self.wait_until(v, "never")
            self.play(Write(facts[0]), FadeIn(fl[0]), run_time=1)
            self.wait_until(v, "zero")
            self.play(Write(facts[1]), FadeIn(fl[1]), Create(nl0), FadeIn(d0), run_time=1.4)
            self.wait_until(v, "mistake")
            self.play(FadeIn(mis, shift=UP * 0.2), run_time=0.8)


class C3_StdDev(Studio):
    def construct(self):
        self.heading("3 · Standard deviation and a shortcut")
        sq = Square(3.0, color=ORANGE, fill_color=ORANGE, fill_opacity=0.15).move_to(LEFT * 3.6 + DOWN * 0.6)
        area = MathTex(r"4 \ \text{points}^2", color=ORANGE, font_size=44).move_to(sq)
        side = MathTex(r"2 \ \text{points}", color=YELLOW, font_size=40).next_to(sq, DOWN, buff=0.25)
        bl = Line(sq.get_corner(DL), sq.get_corner(DR), color=YELLOW, stroke_width=8)
        v1 = MathTex(r"\mathrm{Var}_B = 4", r"\ \text{points}^2", font_size=48).move_to(RIGHT * 3.0 + UP * 2.0)
        v1[1].set_color(RED)
        with self.voice(N["sd1"]) as v:
            self.wait_until(v, "units")
            self.play(Write(v1), run_time=1)
            self.wait_until(v, "squared", 2)
            self.play(Indicate(v1[1], color=RED), run_time=0.8)
            self.play(Create(sq), FadeIn(area), run_time=1)
            self.wait_until(v, "root")
            self.play(Create(bl), FadeIn(side), run_time=1)
        f = MathTex(r"\sigma", "=", r"\sqrt{\mathrm{Var}(X)}", font_size=56).move_to(RIGHT * 3.0 + UP * 0.8)
        f[0].set_color(YELLOW)
        sb = MathTex(r"\sigma_B = \sqrt{4} = %s \ \text{points}" % fmt(demo.sd_B), color=GREEN,
                     font_size=40).next_to(f, DOWN, buff=0.35)
        sa = MathTex(r"\sigma_A = \sqrt{0.4} \approx %.2f" % np.sqrt(demo.var_A), color=BLUE,
                     font_size=40).next_to(sb, DOWN, buff=0.3)
        typ = T("≈ a typical distance from the mean", 22, MUTED).next_to(sb, DOWN, buff=0.15)
        card = self.term_card("standard deviation", "The square root of the variance. It is "
                              "in the original units: a typical distance from the mean.",
                              pos=DR, width=5.2)
        with self.voice(N["sd2"]) as v:
            self.wait_until(v, "called")
            self.play(Write(f), run_time=1)
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "sigma")
            self.play(Indicate(f[0], color=YELLOW), run_time=0.6)
            self.wait_until(v, "two")
            self.play(FadeOut(card), Write(sb), run_time=1)
            self.wait_until(v, "typical")
            self.play(FadeIn(typ), run_time=0.5)
            self.wait_until(v, "zero")
            self.play(typ.animate.next_to(sa, DOWN, buff=0.2), Write(sa), run_time=1)
        self.clear_body()
        sc = MathTex(r"\mathrm{Var}(X)", "=", r"\mathbb{E}[X^2]", "-", r"\mu^2", font_size=60)
        sc[0].set_color(YELLOW); sc[2].set_color(GREEN); sc[4].set_color(PURPLE)
        sc.move_to(UP * 1.9)
        sw = VGroup(T("mean of the squares", 24, GREEN), T("minus", 24, MUTED),
                    T("square of the mean", 24, PURPLE)).arrange(RIGHT, buff=0.4).next_to(sc, DOWN, buff=0.3)
        with self.voice(N["sd3"]) as v:
            self.wait_until(v, "shortcut")
            self.play(Write(sc), run_time=1.5)
            self.wait_until(v, "words")
            self.play(FadeIn(sw), run_time=0.8)
            self.wait_until(v, "check")
            self.play(Indicate(sc[0], color=YELLOW), run_time=0.8)
        sq_row = row(r"x^2", [fmt(a) for a in demo.B ** 2], GREEN, cw=1.0, size=38)
        sq_row.move_to(DOWN * 0.15)
        l1 = MathTex(r"\mathbb{E}[X^2] = \frac{4+16+25+36+64}{5} = \frac{145}{5} = %s" %
                     fmt(np.mean(demo.B ** 2)), color=GREEN, font_size=40).next_to(sq_row, DOWN, buff=0.4)
        l2 = MathTex(r"%s - 5^2 = %s - 25 = %s" % (fmt(np.mean(demo.B ** 2)), fmt(np.mean(demo.B ** 2)),
                                                   fmt(demo.shortcut)), font_size=44).next_to(l1, DOWN, buff=0.35)
        ok = T("same as before ✓", 26, YELLOW).next_to(l2, RIGHT, buff=0.4)
        with self.voice(N["sd4"]) as v:
            self.wait_until(v, "squares")
            self.play(FadeIn(sq_row), run_time=0.8)
            self.wait_until(v, "hundred")
            self.play(Write(l1), run_time=1.5)
            self.wait_until(v, "mean")
            self.play(Write(l2), run_time=1.2)
            self.wait_until(v, "same")
            self.play(FadeIn(ok), run_time=0.5)


class C4_Covariance(Studio):
    def construct(self):
        self.heading("4 · Covariance: moving together")
        cl = VGroup(scatter(demo.t, demo.y_pos, GREEN), scatter(demo.t, demo.y_neg, RED),
                    scatter(demo.t, demo.y_none, MUTED)).arrange(RIGHT, buff=0.7).shift(DOWN * 0.3)
        names = VGroup(T("move together", 24, GREEN), T("opposite", 24, RED),
                       T("no pattern", 24, MUTED))
        for n, c in zip(names, cl):
            n.next_to(c, UP, buff=0.25)
        xy = T("each dot = one (x, y) pair · toy data", 22, MUTED).to_edge(DOWN, buff=0.5)
        with self.voice(N["cov1"]) as v:
            self.wait_until(v, "scatter")
            self.play(FadeIn(xy), *[Create(c[0]) for c in cl], run_time=1)
            for i, w in enumerate(["first", "second", "third"]):
                self.wait_until(v, w)
                self.play(LaggedStart(*[FadeIn(d) for d in cl[i][1]], lag_ratio=0.02),
                          FadeIn(names[i]), run_time=1)
        self.clear_body()
        big = scatter(demo.t, demo.y_pos, GREEN, xl=5.4, yl=4.8).move_to(LEFT * 3.6 + DOWN * 0.6)
        ax = big[0]
        mx, my = float(demo.t.mean()), float(demo.y_pos.mean())
        vline = DashedLine(ax.c2p(mx, -3), ax.c2p(mx, 3), color=YELLOW)
        hline = DashedLine(ax.c2p(-3, my), ax.c2p(3, my), color=YELLOW)
        q = VGroup(MathTex(r"(+)(+)=+", color=GREEN, font_size=30).move_to(ax.c2p(1.8, 2.5)),
                   MathTex(r"(-)(-)=+", color=GREEN, font_size=30).move_to(ax.c2p(-1.8, -2.5)),
                   MathTex(r"(-)(+)=-", color=RED, font_size=30).move_to(ax.c2p(-1.8, 2.5)),
                   MathTex(r"(+)(-)=-", color=RED, font_size=30).move_to(ax.c2p(1.8, -2.5)))
        with self.voice(N["cov2"]) as v:
            self.play(FadeIn(big), run_time=0.8)
            self.wait_until(v, "multiply")
            self.play(Create(vline), Create(hline), run_time=0.8)
            self.wait_until(v, "right")
            self.play(FadeIn(q[0]), run_time=0.5)
            self.wait_until(v, "Left")
            self.play(FadeIn(q[1]), run_time=0.5)
            self.wait_until(v, "other")
            self.play(FadeIn(q[2]), FadeIn(q[3]), run_time=0.6)
        f = MathTex(r"\mathrm{Cov}(X,Y)", "=", r"\mathbb{E}\big[", r"(X-\mu_X)", r"(Y-\mu_Y)", r"\big]",
                    font_size=40)
        f[0].set_color(YELLOW); f[3].set_color(BLUE); f[4].set_color(GREEN)
        f.move_to(RIGHT * 3.5 + UP * 2.0)
        signs = VGroup(
            T(f"> 0  move together      {demo.covs[0]:.2f}", 24, GREEN, font=MONO),
            T(f"< 0  opposite          {demo.covs[1]:.2f}", 24, RED, font=MONO),
            T(f"≈ 0  no straight line   {demo.covs[2]:.2f}", 24, MUTED, font=MONO),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(f, DOWN, buff=0.45).set_x(3.5)
        card = self.term_card("covariance", "The average product of the deviations of two "
                              "variables. Its sign says if they move together.", pos=DR, width=5.2)
        with self.voice(N["cov3"]) as v:
            self.wait_until(v, "Average")
            self.play(Write(f), run_time=1.2)
            self.wait_until(v, "covariance")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "positive")
            self.play(FadeIn(signs[0]), run_time=0.5)
            self.wait_until(v, "negative")
            self.play(FadeIn(signs[1]), run_time=0.5)
            self.wait_until(v, "near")
            self.play(FadeIn(signs[2]), run_time=0.5)
            self.wait_until(v, "clouds")
            self.play(FadeOut(card), Indicate(signs, color=YELLOW), run_time=1)
        self.clear_body()
        ax2 = Axes(x_range=[0, 8, 1], y_range=[0, 7, 1], x_length=5.2, y_length=4.4, tips=False,
                   axis_config={"stroke_color": MUTED, "include_numbers": True, "font_size": 22}
                   ).move_to(LEFT * 3.7 + DOWN * 0.6)
        xl = T("hours studied x", 20, BLUE).next_to(ax2, DOWN, buff=0.2)
        yl = T("score y", 20, GREEN).next_to(ax2, UP, buff=0.1).align_to(ax2, LEFT)
        pts = VGroup(*[Dot(ax2.c2p(a, b), radius=0.1, color=YELLOW) for a, b in zip(demo.x, demo.y)])
        mlx = DashedLine(ax2.c2p(4, 0), ax2.c2p(4, 7), color=BLUE)
        mly = DashedLine(ax2.c2p(0, 4), ax2.c2p(8, 4), color=GREEN)
        tb = VGroup(row("x", [fmt(a) for a in demo.x], BLUE, lw=2.0, cw=0.9),
                    row("y", [fmt(a) for a in demo.y], GREEN, lw=2.0, cw=0.9),
                    row(r"x-\mu_x", [fmt(a) for a in demo.x - demo.x.mean()], BLUE, lw=2.0, cw=0.9),
                    row(r"y-\mu_y", [fmt(a) for a in demo.y - demo.y.mean()], GREEN, lw=2.0, cw=0.9),
                    row(r"\text{product}", [fmt(a) for a in (demo.x - 4) * (demo.y - 4)], YELLOW,
                        lw=2.0, cw=0.9))
        tb.arrange(DOWN, buff=0.3, aligned_edge=RIGHT).scale(1.1).move_to(RIGHT * 3.1 + UP * 0.5)
        means = MathTex(r"\mu_x = 4,\ \ \mu_y = 4", font_size=36).next_to(tb, UP, buff=0.3)
        with self.voice(N["cov4"]) as v:
            self.play(Create(ax2), FadeIn(xl), FadeIn(yl), run_time=1)
            self.wait_until(v, "Four")
            self.play(FadeIn(tb[0]), FadeIn(pts), run_time=0.8)
            self.wait_until(v, "scores")
            self.play(FadeIn(tb[1]), run_time=0.6)
            self.wait_until(v, "mean")
            self.play(FadeIn(means), Create(mlx), Create(mly), run_time=1)
            self.wait_until(v, "deviations")
            self.play(FadeIn(tb[2]), run_time=0.6)
            self.wait_until(v, "deviations", 2)
            self.play(FadeIn(tb[3]), run_time=0.6)
        res = MathTex(r"\mathrm{Cov}(x,y)", "=", r"\frac{6+1+2+3}{4}", "=", r"\frac{12}{4}", "=",
                      fmt(demo.cov_xy), font_size=40)
        res[0].set_color(YELLOW); res[6].set_color(YELLOW)
        res.next_to(tb, DOWN, buff=0.4).set_x(3.1)
        with self.voice(N["cov5"]) as v:
            self.wait_until(v, "Multiply")
            self.play(FadeIn(tb[4]), run_time=1)
            self.wait_until(v, "twelve")
            self.play(Write(res[:5]), run_time=1)
            self.wait_until(v, "four")
            self.play(Write(res[5:]), run_time=0.6)
            self.wait_until(v, "positive")
            self.play(Indicate(res[6], color=YELLOW), Indicate(pts, color=YELLOW), run_time=1)
        self.clear_body()
        ax3 = Axes(x_range=[-2.5, 2.5, 1], y_range=[0, 4.5, 1], x_length=5.4, y_length=4.2, tips=False,
                   axis_config={"stroke_color": MUTED, "include_numbers": True, "font_size": 22}
                   ).move_to(LEFT * 3.4 + DOWN * 0.6)
        para = ax3.plot(lambda t: t * t, x_range=[-2.1, 2.1], color=MUTED)
        pp = VGroup(*[Dot(ax3.c2p(a, a * a), radius=0.1, color=YELLOW) for a in [-2, -1, 0, 1, 2]])
        w = VGroup(T("warning", 30, RED, weight=BOLD),
                   MathTex(r"x = -2,-1,0,1,2", font_size=40),
                   MathTex(r"y = x^2", font_size=40),
                   MathTex(r"\mathrm{Cov}(x,y) = 0", color=RED, font_size=44),
                   T("zero covariance ≠ unrelated", 28, YELLOW)
                   ).arrange(DOWN, buff=0.35).move_to(RIGHT * 3.4 + DOWN * 0.3)
        with self.voice(N["cov6"]) as v:
            self.wait_until(v, "warning")
            self.play(FadeIn(w[0]), Create(ax3), run_time=1)
            self.wait_until(v, "squared")
            self.play(Create(para), FadeIn(pp), FadeIn(w[1]), FadeIn(w[2]), run_time=1.2)
            self.wait_until(v, "zero")
            self.play(Write(w[3]), run_time=0.8)
            self.wait_until(v, "unrelated")
            self.play(FadeIn(w[4]), run_time=0.6)


class C5_Matrix(Studio):
    def construct(self):
        self.heading("5 · The covariance matrix")
        tmpl = MathTex(r"\Sigma", "=", r"\begin{bmatrix} \mathrm{Var}(x) & \mathrm{Cov}(x,y) \\"
                       r" \mathrm{Cov}(y,x) & \mathrm{Var}(y) \end{bmatrix}", font_size=52)
        tmpl[0].set_color(YELLOW)
        tmpl.move_to(UP * 1.2)
        dg = T("diagonal: variances", 26, ORANGE)
        od = T("off the diagonal: covariance", 26, PURPLE)
        lg = VGroup(dg, od).arrange(RIGHT, buff=1.0).next_to(tmpl, DOWN, buff=0.5)
        with self.voice(N["mat1"]) as v:
            self.wait_until(v, "three")
            self.play(Write(tmpl), run_time=1.6)
            self.wait_until(v, "diagonal")
            self.play(FadeIn(dg), run_time=0.6)
            self.wait_until(v, "both")
            self.play(FadeIn(od), run_time=0.6)
        C = demo.C_n
        num = MathTex(r"\Sigma", "=", r"\begin{bmatrix} %s & %s \\ %s & %s \end{bmatrix}" %
                      (fmt(C[0, 0]), fmt(C[0, 1]), fmt(C[1, 0]), fmt(C[1, 1])), font_size=60)
        num[0].set_color(YELLOW)
        num.next_to(lg, DOWN, buff=0.5)
        sym = T("symmetric: Cov(x, y) = Cov(y, x)", 24, MUTED).next_to(num, DOWN, buff=0.3)
        ten = T("10 features → 10 × 10 matrix", 24, MUTED).next_to(sym, DOWN, buff=0.2)
        with self.voice(N["mat2"]) as v:
            self.wait_until(v, "five")
            self.play(Write(num), run_time=1.5)
            self.wait_until(v, "symmetric")
            self.play(FadeIn(sym), run_time=0.6)
            self.wait_until(v, "ten")
            self.play(FadeIn(ten), run_time=0.6)
        self.clear_body()
        dx = demo.x - demo.x.mean()
        dy = demo.y - demo.y.mean()
        mA = MathTex(r"\mu = \textstyle\sum_x p(x)\,x", font_size=40).move_to(LEFT * 3.6 + UP * 2.0)
        tags = VGroup(T("weighted sum → axiom A", 22, BLUE), T("weights p(x) → axiom D", 22, PURPLE)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(mA, RIGHT, buff=0.5)
        dvx = MathTex(r"d_x = \begin{bmatrix}%s\end{bmatrix}" % r"\\".join(fmt(a) for a in dx),
                      color=BLUE, font_size=40)
        dvy = MathTex(r"d_y = \begin{bmatrix}%s\end{bmatrix}" % r"\\".join(fmt(a) for a in dy),
                      color=GREEN, font_size=40)
        VGroup(dvx, dvy).arrange(RIGHT, buff=0.8).move_to(LEFT * 3.6 + DOWN * 1.2)
        dot = MathTex(r"d_x \cdot d_y", "=", r"(-3)(-2)+(-1)(-1)+(1)(2)+(3)(1)", font_size=32)
        dot2 = MathTex("=", "6+1+2+3", "=", fmt(float(dx @ dy)), font_size=40)
        dot3 = MathTex(r"\tfrac14\, d_x\cdot d_y", "=", fmt(float(dx @ dy) / 4), "=",
                       r"\mathrm{Cov}(x,y)", font_size=44)
        dot3[2].set_color(YELLOW); dot3[4].set_color(YELLOW)
        VGroup(dot, dot2, dot3).arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to(RIGHT * 3.0 + DOWN * 1.0)
        for m in (dot, dot2, dot3):
            if m.get_right()[0] > 7.0:
                m.shift(LEFT * (m.get_right()[0] - 7.0))
        with self.voice(N["mat3"]) as v:
            self.wait_until(v, "weighted")
            self.play(Write(mA), run_time=1)
            self.wait_until(v, "axiom")
            self.play(FadeIn(tags[0]), run_time=0.5)
            self.wait_until(v, "axiom", 2)
            self.play(FadeIn(tags[1]), run_time=0.5)
            self.wait_until(v, "vector")
            self.play(FadeIn(dvx), FadeIn(dvy), run_time=0.8)
            self.wait_until(v, "dot")
            self.play(Write(dot), run_time=1.2)
            self.wait_until(v, "twelve")
            self.play(Write(dot2), run_time=0.8)
            self.wait_until(v, "four")
            self.play(Write(dot3), run_time=0.8)
        rem = self.reminder_card("dot product", "Multiply matching components and add: "
                                 "the inner product of two lists.", pos=UR, width=4.6)
        varl = MathTex(r"\mathrm{Var}(x) = \tfrac14\, d_x\cdot d_x = \tfrac{20}{4} = %s" % fmt(C[0, 0]),
                       color=BLUE, font_size=40).next_to(dot3, DOWN, buff=0.35).align_to(dot3, LEFT)
        badge = T("Axioms A + B + D", 40, YELLOW, weight=BOLD).move_to(LEFT * 3.6 + UP * 0.6)
        with self.voice(N["mat4"]) as v:
            self.wait_until(v, "inner")
            self.play(FadeIn(rem, shift=DOWN * 0.2), run_time=0.6)
            self.wait_until(v, "itself")
            self.play(Write(varl), run_time=1.2)
            self.wait_until(v, "built")
            self.play(FadeOut(rem), Write(badge), run_time=1)
        self.clear_body()
        Cn1 = demo.C_n1
        m1 = VGroup(T("÷ n", 30, BLUE, weight=BOLD),
                    MathTex(r"\begin{bmatrix} %s & %s \\ %s & %s \end{bmatrix}" %
                            (fmt(C[0, 0]), fmt(C[0, 1]), fmt(C[1, 0]), fmt(C[1, 1])), font_size=52)
                    ).arrange(DOWN, buff=0.3)
        m2 = VGroup(T("÷ (n − 1)  · Bessel's correction", 30, PURPLE, weight=BOLD),
                    MathTex(r"\begin{bmatrix} %.3f & %s \\ %s & %.3f \end{bmatrix}" %
                            (Cn1[0, 0], fmt(Cn1[0, 1]), fmt(Cn1[1, 0]), Cn1[1, 1]), font_size=52)
                    ).arrange(DOWN, buff=0.3)
        VGroup(m1, m2).arrange(RIGHT, buff=1.4).shift(UP * 0.5)
        npn = VGroup(T("np.var(...)  divides by n   (default)", 24, MUTED, font=MONO),
                     T("np.cov(...)  divides by n − 1 (default)", 24, MUTED, font=MONO)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(DOWN, buff=0.7)
        with self.voice(N["mat5"]) as v:
            self.wait_until(v, "divided")
            self.play(FadeIn(m1), run_time=0.8)
            self.wait_until(v, "sample")
            self.play(FadeIn(m2[0]), run_time=0.6)
            self.wait_until(v, "correction")
            self.play(FadeIn(m2[1]), run_time=0.8)
            self.wait_until(v, "var")
            self.play(FadeIn(npn[0]), run_time=0.5)
            self.wait_until(v, "cov")
            self.play(FadeIn(npn[1]), run_time=0.5)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · Variance in machine learning")
        zf = MathTex("z", "=", r"\frac{x-\mu}{\sigma}", font_size=60).move_to(UP * 1.9)
        zf[0].set_color(YELLOW)
        nl1 = number_line((-2, 9), 10).move_to(UP * 0.0)
        d1 = dots_on(nl1, demo.B, GREEN, r=0.1)
        l1 = T("scores", 22, GREEN).next_to(nl1, LEFT, buff=0.2)
        nl2 = number_line((-2, 9), 10).move_to(DOWN * 2.3)
        d2 = dots_on(nl2, demo.z, YELLOW, r=0.1)
        l2 = T("z-scores", 22, YELLOW).next_to(nl2, LEFT, buff=0.2)
        zr = MathTex(r"[2,4,5,6,8] \;\to\; [%s]" % ",".join(fmt(a) for a in demo.z),
                     font_size=36).move_to(RIGHT * 3.0 + UP * 1.9)
        with self.voice(N["ml1"]) as v:
            self.wait_until(v, "subtract")
            self.play(Write(zf), run_time=1)
            self.play(zf.animate.shift(LEFT * 3.8), run_time=0.6)
            self.play(Create(nl1), FadeIn(d1), FadeIn(l1), run_time=0.8)
            self.wait_until(v, "become")
            self.play(Create(nl2), TransformFromCopy(d1, d2), FadeIn(l2), run_time=1.5)
            self.play(Write(zr), run_time=1)
        res = VGroup(MathTex(r"\text{mean} = 0", color=YELLOW, font_size=40),
                     MathTex(r"\text{variance} = 1", color=YELLOW, font_size=40)
                     ).arrange(RIGHT, buff=0.8).move_to(DOWN * 3.3)
        same = T("meters · dollars · years → one common scale", 24, MUTED)
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "zero")
            self.play(FadeIn(res), run_time=0.8)
            self.wait_until(v, "meters")
            self.play(FadeOut(VGroup(nl1, d1, l1)), run_time=0.5)
            same.move_to(UP * 0.3)
            self.play(FadeIn(same), run_time=0.6)
        self.clear_body()
        tok = MathTex(r"\text{token} \;\to\; [x_1, x_2, \dots, x_{768}]", font_size=44).move_to(UP * 2.0)
        tok.set_color(BLUE)
        st = MathTex(r"\mu = \tfrac1n\sum_i x_i", r"\qquad", r"\sigma^2 = \tfrac1n\sum_i (x_i-\mu)^2",
                     font_size=40).move_to(UP * 0.6)
        st[0].set_color(YELLOW); st[2].set_color(ORANGE)
        ln = MathTex(r"\frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}}", font_size=52).move_to(DOWN * 1.1)
        lab = T("layer normalization · GPT-2 small: n = 768 · ε = 10⁻⁵", 24, MUTED).to_edge(DOWN, buff=0.5)
        with self.voice(N["ml3"]) as v:
            self.wait_until(v, "token")
            self.play(Write(tok), run_time=1)
            self.wait_until(v, "mean")
            self.play(Write(st), run_time=1.4)
            self.wait_until(v, "standardizes")
            self.play(Write(ln), run_time=1)
            self.wait_until(v, "seven")
            self.play(FadeIn(lab), run_time=0.6)
            self.wait_until(v, "epsilon")
            self.play(Indicate(ln, color=YELLOW), run_time=0.8)
        self.clear_body()
        big = scatter(demo.t, demo.y_pos, GREEN, xl=5.4, yl=4.6).move_to(LEFT * 3.4 + DOWN * 0.5)
        ax = big[0]
        S = np.cov(demo.t, demo.y_pos, bias=True)
        w, V = np.linalg.eigh(S)
        e = V[:, -1] * np.sign(V[0, -1])
        c0 = np.array([demo.t.mean(), demo.y_pos.mean()])
        L = 2.0 * np.sqrt(w[-1])
        pa = Arrow(ax.c2p(*(c0 - L * e)), ax.c2p(*(c0 + L * e)), buff=0, color=YELLOW, stroke_width=6)
        pl = T("most spread", 22, YELLOW).next_to(pa.get_end(), RIGHT, buff=0.15)
        mol = VGroup(T("molecules (preview)", 28, YELLOW, weight=BOLD),
                     T("PCA: directions from the covariance matrix", 24, FG),
                     T("layer normalization: mean + variance", 24, FG),
                     T("→ the molecules map", 24, MUTED)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.4 + UP * 0.2)
        with self.voice(N["ml4"]) as v:
            self.wait_until(v, "PCA")
            self.play(FadeIn(big), FadeIn(mol[1]), run_time=1)
            self.wait_until(v, "directions")
            self.play(GrowArrow(pa), FadeIn(pl), run_time=1)
            self.wait_until(v, "molecules")
            self.play(FadeIn(mol[0]), FadeIn(mol[2]), FadeIn(mol[3]), run_time=0.8)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        q = MathTex(r"\text{data: } 1,\ 1,\ 5,\ 5", r"\qquad \mathrm{Var} = ?", r"\quad \sigma = ?",
                    font_size=56).shift(UP * 1.6)
        q[1].set_color(YELLOW); q[2].set_color(YELLOW)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac1"]) as v:
            self.play(Write(q), run_time=1.5)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
            self.wait_until(v, "mean")
            self.play(Indicate(q[0], color=BLUE), run_time=0.8)
        a = VGroup(MathTex(r"\mu = 3", font_size=44),
                   MathTex(r"x-\mu:\ -2,\ -2,\ 2,\ 2", color=ORANGE, font_size=44),
                   MathTex(r"\mathrm{Var} = \frac{4+4+4+4}{4} = \frac{16}{4} = 4", color=YELLOW, font_size=44),
                   MathTex(r"\sigma = \sqrt{4} = 2", color=YELLOW, font_size=44)
                   ).arrange(DOWN, buff=0.3).next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "mean")
            self.play(FadeIn(a[0]), run_time=0.6)
            self.wait_until(v, "deviations")
            self.play(FadeIn(a[1]), run_time=0.6)
            self.wait_until(v, "sixteen")
            self.play(Write(a[2]), run_time=1.2)
            self.wait_until(v, "root")
            self.play(Write(a[3]), run_time=0.8)
            self.play(Indicate(a[3], color=YELLOW), run_time=0.8)
        self.clear_body()
        ax = Axes(x_range=[0, 4, 1], y_range=[0, 7, 1], x_length=4.6, y_length=4.4, tips=False,
                  axis_config={"stroke_color": MUTED, "include_numbers": True, "font_size": 22}
                  ).move_to(LEFT * 3.8 + DOWN * 0.6)
        P = [(1, 6), (2, 4), (3, 2)]
        pts = VGroup(*[Dot(ax.c2p(*p), radius=0.1, color=YELLOW) for p in P])
        q2 = VGroup(MathTex(r"(1,6),\ (2,4),\ (3,2)", font_size=44),
                    MathTex(r"\mathrm{Cov}(x,y):\ +\ \text{or}\ -\ ?", color=YELLOW, font_size=44),
                    T("⏸  pause and try", 28, MUTED)).arrange(DOWN, buff=0.4).move_to(RIGHT * 3.0 + UP * 1.2)
        with self.voice(N["prac3"]) as v:
            self.play(Create(ax), run_time=0.8)
            self.wait_until(v, "points")
            self.play(FadeIn(pts), FadeIn(q2[0]), run_time=0.8)
            self.wait_until(v, "positive")
            self.play(FadeIn(q2[1]), run_time=0.6)
            self.wait_until(v, "Pause")
            self.play(FadeIn(q2[2]), run_time=0.5)
        ans = VGroup(MathTex(r"\mu_x = 2,\ \ \mu_y = 4", font_size=40),
                     MathTex(r"(-1)(2) + (0)(0) + (1)(-2)", font_size=40),
                     MathTex(r"= -2 + 0 - 2 = -4", font_size=40),
                     MathTex(r"\mathrm{Cov} = -\tfrac43 \approx -1.33", color=RED, font_size=44)
                     ).arrange(DOWN, buff=0.3).move_to(RIGHT * 3.0 + DOWN * 1.0)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(q2[2]), q2[:2].animate.shift(UP * 0.6), run_time=0.5)
            self.wait_until(v, "mean")
            self.play(FadeIn(ans[0]), run_time=0.6)
            self.wait_until(v, "pairs")
            self.play(FadeIn(ans[1]), run_time=0.8)
            self.wait_until(v, "products")
            self.play(FadeIn(ans[2]), run_time=0.8)
            self.wait_until(v, "negative")
            self.play(Write(ans[3]), run_time=0.8)


CODE = '''import numpy as np
A = np.array([4, 5, 5, 5, 6]); B = np.array([2, 4, 5, 6, 8])
mu = B.mean(); dev = B - mu
print(dev.mean())                  # deviations cancel: 0
print(np.var(A), np.mean(dev**2))  # variance
print(np.std(B))                   # standard deviation
print(np.mean(B**2) - mu**2)       # shortcut
x = np.array([1, 3, 5, 7]); y = np.array([2, 3, 6, 5])
print(np.mean((x - x.mean()) * (y - y.mean())))  # covariance
print(np.cov(x, y, bias=True))     # matrix, divide by n
print(np.cov(x, y))                # default: n - 1
print((B - mu) / np.std(B))        # z-scores'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=22)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[2], color=YELLOW, buff=0.06)
            self.wait_until(v, "deviations")
            self.play(Create(box), run_time=0.5)
            for i, w in [(4, "squares"), (6, "shortcut"), (9, "matrix"), (11, "standardize")]:
                self.wait_until(v, w)
                self.play(Transform(box, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                              buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=22, command="python demo.py").shift(DOWN * 0.3)
        if term.width > 13.4:
            term.scale_to_fit_width(13.4)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for i, w in [(0, "means"), (1, "zero"), (3, "variances"), (5, "shortcut"),
                         (6, "covariance"), (8, "matrix"), (10, "variance")]:
                self.wait_until(v, w)
                self.play(Indicate(term[1][i], color=YELLOW), run_time=0.6)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["Variance = expected squared distance from the mean",
                            "Square, because plain deviations always cancel to 0",
                            "Standard deviation = √variance, in the original units",
                            "Shortcut: Var = E[X²] − μ²",
                            "Covariance = average product of deviations; sign = together / opposite",
                            "Covariance matrix: variances on the diagonal, covariances off it"],
                           size=26).shift(UP * 0.5)
        if pts.width > 13:
            pts.scale_to_fit_width(13)
        with self.voice(N["sum1"]) as v:
            for p in pts[:4]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 6)
        nxt = VGroup(T("Next · Video 15", 26, MUTED),
                     T("Loss function: one number for how wrong you are", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "matrix")
            self.play(FadeIn(pts[5], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Spread", "C2_Variance", "C3_StdDev", "C4_Covariance", "C5_Matrix",
          "C6_ML", "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Spread": "Same mean, different spread",
                  "C2_Variance": "Variance", "C3_StdDev": "Standard deviation & shortcut",
                  "C4_Covariance": "Covariance", "C5_Matrix": "The covariance matrix",
                  "C6_ML": "Variance in ML", "C7_Practice": "Try it yourself",
                  "C7_Code": "Python demo", "C8_Summary": "Summary", "C9_Credits": "Credits"}
