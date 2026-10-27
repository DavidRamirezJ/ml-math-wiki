"""Activation function — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
import contextlib
import io
import math
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

SERIES = "ML Math Atoms · Video 16 of 16"
TITLE = "Activation function — the bend that makes networks deep"


def relu(z):
    return max(0.0, z)


def gelu(z):
    return z * 0.5 * (1 + math.erf(z / math.sqrt(2)))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def axes(xr, yr, xl, yl, nums=True, fs=22):
    return Axes(x_range=xr, y_range=yr, x_length=xl, y_length=yl, tips=False,
                axis_config={"stroke_color": MUTED, "include_numbers": nums, "font_size": fs})


def mat(rows, color=PURPLE, size=44):
    body = r"\\".join(" & ".join(str(v) for v in r) for r in rows)
    return MathTex(r"\begin{bmatrix}%s\end{bmatrix}" % body, color=color, font_size=size)


def colv(vals, color=BLUE, size=44):
    return MathTex(r"\begin{bmatrix}%s\end{bmatrix}" % r"\\".join(str(v) for v in vals),
                   color=color, font_size=size)


def mts(rows):
    return r"\begin{bmatrix}%s\end{bmatrix}" % r"\\".join(" & ".join(str(v) for v in r) for r in rows)


def cvs(vals):
    return r"\begin{bmatrix}%s\end{bmatrix}" % r"\\".join(str(v) for v in vals)


def box(label, color, w=2.0, h=0.9, size=24):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h, color=color)
    t = T(label, size, color, weight=BOLD).move_to(r)
    if t.width > w - 0.15:
        t.scale_to_fit_width(w - 0.15)
    return VGroup(r, t)


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        if title.width > 12.5:
            title.scale_to_fit_width(12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.4)
        layers = VGroup(*[box(f"layer {i}", PURPLE, w=1.9) for i in (1, 2, 3)], T("…", 40, MUTED),
                        box("layer 100", PURPLE, w=1.9)).arrange(RIGHT, buff=0.35).shift(DOWN * 0.4)
        one = box("one layer", PURPLE, w=2.6).move_to(DOWN * 0.4)
        eq = T("= no smarter than", 28, RED).next_to(layers, DOWN, buff=0.5)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            self.wait_until(v, "stack")
            self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in layers], lag_ratio=0.2), run_time=1.5)
            self.wait_until(v, "hundred")
            self.play(FadeIn(eq), run_time=0.5)
            self.wait_until(v, "wasted")
            self.play(ReplacementTransform(layers, one), eq.animate.set_opacity(0), run_time=1.2)
            self.wait_until(v, "bend")
            self.play(Indicate(one, color=YELLOW), run_time=0.8)
        plan = VGroup(T("two linear layers → one", 28, FG), T("ReLU: max(0, z)", 28, YELLOW),
                      T("a few bends draw a curve", 28, FG),
                      T("you need: video 2 (matrix multiplication), video 7 (derivative)", 24, MUTED)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(DOWN * 0.9)
        with self.voice(N["intro2"]) as v:
            self.play(FadeOut(one), run_time=0.5)
            for i, w in enumerate(["collapse", "ReLU", "curve", "seven"]):
                self.wait_until(v, w)
                self.play(FadeIn(plan[i], shift=RIGHT * 0.2), run_time=0.5)


class C1_Collapse(Studio):
    def construct(self):
        self.heading("1 · Two linear layers = one")
        e1 = MathTex(r"W_1 x", "=", mts([[1, 2], [0, 1]]), cvs([3, -1]), "=",
                     r"\begin{bmatrix}1\cdot3+2\cdot(-1)\\0\cdot3+1\cdot(-1)\end{bmatrix}", "=",
                     cvs(demo.h.tolist()), font_size=40)
        e1[2].set_color(PURPLE); e1[3].set_color(BLUE); e1[7].set_color(YELLOW)
        e1.move_to(UP * 1.7)
        e2 = MathTex(r"W_2 (W_1 x)", "=", mts([[2, 0], [1, 1]]), cvs(demo.h.tolist()),
                     "=", cvs(demo.two.tolist()), font_size=40)
        e2[2].set_color(PURPLE); e2[3].set_color(YELLOW); e2[5].set_color(YELLOW)
        e2.move_to(UP * 0.0)
        e3 = MathTex(r"(W_2 W_1)\, x", "=", mts(demo.W.tolist()), cvs([3, -1]),
                     "=", cvs(demo.one.tolist()), font_size=40)
        e3[2].set_color(PURPLE); e3[3].set_color(BLUE); e3[5].set_color(YELLOW)
        e3.move_to(DOWN * 1.8)
        toy = T("toy numbers", 20, MUTED).to_corner(UR, buff=0.5)
        with self.voice(N["col1"]) as v:
            self.wait_until(v, "toy")
            self.play(FadeIn(toy), FadeIn(e1[3]), run_time=0.6)
            self.wait_until(v, "first")
            self.play(FadeIn(e1[:3]), run_time=0.8)
            self.wait_until(v, "Multiply")
            self.play(Write(e1[4:6]), run_time=1.5)
            self.wait_until(v, "gives")
            self.play(Write(e1[6:]), run_time=0.8)
        with self.voice(N["col2"]) as v:
            self.wait_until(v, "second")
            self.play(FadeIn(e2[:3]), TransformFromCopy(e1[7], e2[3]), run_time=1)
            self.wait_until(v, "Multiply")
            self.play(Write(e2[4:]), run_time=1)
            self.wait_until(v, "give")
            self.play(Indicate(e2[5], color=YELLOW), run_time=0.8)
        with self.voice(N["col3"]) as v:
            self.wait_until(v, "matrices")
            self.play(Write(e3[:3]), run_time=1.2)
            self.wait_until(v, "Apply")
            self.play(FadeIn(e3[3]), Write(e3[4:]), run_time=1.2)
            self.wait_until(v, "Again")
            self.play(Indicate(e2[5], color=YELLOW), Indicate(e3[5], color=YELLOW), run_time=1)
        self.clear_body()
        assoc = MathTex(r"W_2\,(W_1\,x)", "=", r"(W_2 W_1)\,x", font_size=64).shift(UP * 1.0)
        assoc[0].set_color(BLUE); assoc[2].set_color(PURPLE)
        lines = VGroup(T("linear layer ∘ linear layer = one linear layer", 28, FG),
                       T("stretch · turn · flip · squash — never a curve", 26, MUTED)
                       ).arrange(DOWN, buff=0.3).next_to(assoc, DOWN, buff=0.6)
        rem = self.reminder_card("matrix multiplication", "Rows times columns; it is associative: "
                                 "A(BC) = (AB)C.", pos=DR, width=4.6)
        with self.voice(N["col4"]) as v:
            self.wait_until(v, "associative")
            self.play(Write(assoc), FadeIn(rem, shift=UP * 0.2), run_time=1.2)
            self.wait_until(v, "disguise")
            self.play(FadeIn(lines[0]), run_time=0.6)
            self.wait_until(v, "curve")
            self.play(FadeIn(lines[1]), run_time=0.6)


class C2_ReLU(Studio):
    def construct(self):
        self.heading("2 · The bend: ReLU")
        chain = VGroup(box("linear", PURPLE), Arrow(LEFT * 0.4, RIGHT * 0.4, buff=0, color=MUTED),
                       box("bend", YELLOW, w=1.4), Arrow(LEFT * 0.4, RIGHT * 0.4, buff=0, color=MUTED),
                       box("linear", PURPLE)).arrange(RIGHT, buff=0.2).shift(UP * 1.3)
        each = T("applied to every number separately", 24, MUTED).next_to(chain, DOWN, buff=0.4)
        c1 = self.term_card("nonlinearity", "A function that is not linear: it does not keep "
                            "adding and scaling.", pos=DL, width=5.0)
        c2 = self.term_card("activation function", "The nonlinearity applied to every output "
                            "number of a layer, before the next layer.", pos=DR, width=5.4)
        with self.voice(N["relu1"]) as v:
            self.play(FadeIn(chain[0]), FadeIn(chain[4]), run_time=0.8)
            self.wait_until(v, "nonlinearity")
            self.play(GrowArrow(chain[1]), GrowArrow(chain[3]), FadeIn(chain[2]), FadeIn(c1, shift=UP * 0.3), run_time=1)
            self.wait_until(v, "activation")
            self.play(FadeIn(c2, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "separately")
            self.play(FadeIn(each), run_time=0.5)
        self.clear_body()
        ax = axes([-3, 3.2, 1], [-1, 3.2, 1], 6.4, 4.6).move_to(LEFT * 3.0 + DOWN * 0.6)
        zl = MathTex("z", color=BLUE, font_size=32).next_to(ax, DOWN, buff=0.05).align_to(ax, RIGHT)
        g = ax.plot(relu, x_range=[-3, 3.1], color=YELLOW, stroke_width=6, use_smoothing=False)
        f = MathTex(r"\mathrm{ReLU}(z)", "=", r"\max(0,\ z)", font_size=52).move_to(RIGHT * 3.6 + UP * 2.0)
        f[0].set_color(YELLOW); f[2].set_color(YELLOW)
        cases = VGroup(T("z > 0 → keep z", 26, GREEN), T("z < 0 → 0", 26, RED)
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(f, DOWN, buff=0.4)
        card = self.term_card("ReLU", "Rectified linear unit: ReLU(z) = max(0, z). Keeps positive "
                              "numbers, turns negative numbers into 0.", pos=DR, width=5.2)
        with self.voice(N["relu2"]) as v:
            self.play(Create(ax), FadeIn(zl), run_time=0.8)
            self.wait_until(v, "ReLU")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "maximum")
            self.play(Write(f), run_time=1)
            self.wait_until(v, "positive")
            self.play(FadeIn(cases[0]), run_time=0.5)
            self.wait_until(v, "negative")
            self.play(FadeIn(cases[1]), run_time=0.5)
            self.wait_until(v, "graph")
            self.play(FadeOut(card), Create(g), run_time=1.2)
        dots, labs = VGroup(), VGroup()
        for i, (zz, rr) in enumerate(zip(demo.z, demo.rz)):
            d = Dot(ax.c2p(zz, rr), color=BLUE if rr == 0 else GREEN, radius=0.1)
            dots.add(d)
        tbl = VGroup(*[MathTex(r"\mathrm{ReLU}(%s) = %s" % (f"{zz:g}", f"{rr:g}"), font_size=34,
                               color=GREEN if rr > 0 else MUTED) for zz, rr in zip(demo.z, demo.rz)]
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(RIGHT * 3.6 + DOWN * 1.4)
        with self.voice(N["relu3"]) as v:
            for i in range(5):
                self.wait_until(v, "ReLU", i + 1)
                self.play(FadeIn(dots[i], scale=0.5), FadeIn(tbl[i]), run_time=0.5)
            self.wait_until(v, "switched")
            self.play(Indicate(VGroup(*dots[:3]), color=RED), run_time=0.8)
        self.clear_body()
        h = colv(demo.h.tolist(), YELLOW)
        rh = colv(demo.relu(demo.h).tolist(), YELLOW)
        out = colv(demo.bent.tolist(), GREEN)
        a1 = VGroup(Arrow(LEFT * 0.6, RIGHT * 0.6, buff=0, color=MUTED), T("ReLU", 24, YELLOW, weight=BOLD))
        a1[1].next_to(a1[0], UP, buff=0.1)
        a2 = VGroup(Arrow(LEFT * 0.6, RIGHT * 0.6, buff=0, color=MUTED), MathTex("W_2", color=PURPLE, font_size=36))
        a2[1].next_to(a2[0], UP, buff=0.1)
        hl = MathTex(r"W_1 x =", font_size=40)
        chain2 = VGroup(hl, h, a1, rh, a2, out).arrange(RIGHT, buff=0.35).shift(UP * 0.8)
        cmp = VGroup(MathTex(r"\text{with ReLU: }", cvs(demo.bent.tolist()), font_size=40),
                     MathTex(r"\text{without: }", cvs(demo.two.tolist()), font_size=40)
                     ).arrange(RIGHT, buff=1.2).shift(DOWN * 1.6)
        cmp[0][1].set_color(GREEN); cmp[1][1].set_color(MUTED)
        neq = MathTex(r"\neq", color=RED, font_size=56).move_to(cmp.get_center())
        with self.voice(N["relu4"]) as v:
            self.wait_until(v, "first")
            self.play(FadeIn(hl), FadeIn(h), run_time=0.6)
            self.wait_until(v, "turns")
            self.play(GrowArrow(a1[0]), FadeIn(a1[1]), FadeIn(rh), run_time=0.8)
            self.wait_until(v, "gives")
            self.play(GrowArrow(a2[0]), FadeIn(a2[1]), FadeIn(out), run_time=0.8)
            self.wait_until(v, "Without")
            self.play(FadeIn(cmp), FadeIn(neq), run_time=0.8)
            self.wait_until(v, "broke")
            self.play(Indicate(neq, color=RED), run_time=0.8)


class C3_Shapes(Studio):
    def construct(self):
        self.heading("3 · Other bends: sigmoid, tanh, GELU")
        axl = axes([-4, 4, 2], [-1.2, 1.2, 1], 6.0, 3.6).move_to(LEFT * 3.4 + UP * 0.1)
        sg = axl.plot(sigmoid, x_range=[-4, 4], color=GREEN, stroke_width=5)
        th = axl.plot(math.tanh, x_range=[-4, 4], color=ORANGE, stroke_width=5)
        one = DashedLine(axl.c2p(-4, 1), axl.c2p(4, 1), color=MUTED, stroke_width=2)
        mone = DashedLine(axl.c2p(-4, -1), axl.c2p(4, -1), color=MUTED, stroke_width=2)
        s0 = Dot(axl.c2p(0, demo.sig0), color=GREEN, radius=0.09)
        t0 = Dot(axl.c2p(0, demo.tanh0), color=ORANGE, radius=0.09)
        sf = MathTex(r"\sigma(z) = \frac{1}{1+e^{-z}}", color=GREEN, font_size=34).move_to(axl.c2p(-2.3, 0.7))
        sl = VGroup(T("sigmoid: range (0, 1) · smooth", 22, GREEN), MathTex(r"\sigma(0) = 0.5", color=GREEN, font_size=32)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(axl, DOWN, buff=0.25).align_to(axl, LEFT)
        tl = VGroup(T("tanh: range (−1, 1) · smooth", 22, ORANGE), MathTex(r"\tanh(0) = 0", color=ORANGE, font_size=32)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(sl, DOWN, buff=0.2).align_to(sl, LEFT)
        with self.voice(N["shape1"]) as v:
            self.play(Create(axl), Create(one), run_time=0.8)
            self.wait_until(v, "sigmoid")
            self.play(Create(sg), Write(sf), run_time=1.2)
            self.wait_until(v, "half")
            self.play(FadeIn(s0), FadeIn(sl), run_time=0.6)
            self.wait_until(v, "probability")
            self.play(Indicate(one, color=GREEN), run_time=0.8)
        with self.voice(N["shape2"]) as v:
            self.wait_until(v, "tanh")
            self.play(Create(th), Create(mone), run_time=1.2)
            self.wait_until(v, "Tanh", 2)
            self.play(FadeIn(t0), FadeIn(tl), run_time=0.6)
            self.wait_until(v, "flat")
            self.play(Indicate(sg, color=GREEN), Indicate(th, color=ORANGE), run_time=1)
        axr = axes([-3, 3, 1], [-0.5, 3, 1], 5.4, 4.0).move_to(RIGHT * 3.6 + DOWN * 0.4)
        rl = axr.plot(relu, x_range=[-3, 3], color=MUTED, stroke_width=3, use_smoothing=False)
        ge = axr.plot(gelu, x_range=[-3, 3], color=BLUE, stroke_width=5)
        gf = MathTex(r"\mathrm{GELU}(z) = z\,\Phi(z)", color=BLUE, font_size=38).next_to(axr, UP, buff=0.25)
        gl = T("Φ(z) = chance a standard normal value is below z", 20, MUTED).next_to(axr, DOWN, buff=0.25)
        d1 = Dot(axr.c2p(1, demo.gelu[1.0]), color=BLUE, radius=0.09)
        d1l = MathTex(r"%.2f" % demo.gelu[1.0], color=BLUE, font_size=30).next_to(d1, RIGHT, buff=0.15)
        with self.voice(N["shape3"]) as v:
            self.play(FadeOut(sf), run_time=0.3)
            self.wait_until(v, "GELU")
            self.play(Create(axr), Create(rl), run_time=0.8)
            self.wait_until(v, "probability")
            self.play(Create(ge), Write(gf), FadeIn(gl), run_time=1.5)
            self.wait_until(v, "eight")
            self.play(FadeIn(d1), FadeIn(d1l), run_time=0.6)
        dm = Dot(axr.c2p(-1, demo.gelu[-1.0]), color=RED, radius=0.09)
        dml = MathTex(r"\mathrm{GELU}(-1) \approx %.2f" % demo.gelu[-1.0], color=RED, font_size=30
                      ).scale(0.9).move_to(axr.c2p(-1.9, 1.35))
        gpt = T("GPT-2 uses GELU", 26, YELLOW, weight=BOLD).next_to(gl, DOWN, buff=0.15)
        with self.voice(N["shape4"]) as v:
            self.wait_until(v, "dips")
            self.play(FadeIn(dm), Indicate(dm, color=RED), run_time=0.8)
            self.wait_until(v, "minus")
            self.play(FadeIn(dml), run_time=0.6)
            self.wait_until(v, "transformer")
            self.play(FadeIn(gpt), run_time=0.6)


class C4_Bends(Studio):
    def construct(self):
        self.heading("4 · How bends build curves")
        ax = axes([-2.2, 2.2, 1], [-0.5, 4.5, 1], 6.4, 5.0).move_to(LEFT * 3.2 + DOWN * 0.5)
        par = ax.plot(lambda t: t * t, x_range=[-2.1, 2.1], color=MUTED, stroke_width=4)
        pl = MathTex("y = x^2", color=MUTED, font_size=34).move_to(ax.c2p(1.5, 4.3))
        knots = [-2, -1, 0, 1]
        a = demo.a4.tolist()

        def partial(k):
            def g(t):
                return 4 + sum(a[i] * relu(t - knots[i]) for i in range(k))
            xs = np.linspace(-2, 2, 401)
            ok = np.array([g(t) >= -0.45 for t in xs])
            xmax = float(xs[np.argmin(ok)] if not ok.all() else 2.0)
            return ax.plot(g, x_range=[-2, xmax, 0.01], color=YELLOW, stroke_width=6, use_smoothing=False)
        gs = [partial(k) for k in range(5)]
        terms = VGroup(MathTex("4", font_size=38),
                       *[MathTex(r"%s\,\mathrm{ReLU}(x%s)" % (f"{a[i]:+g}", f"{-knots[i]:+g}" if knots[i] else ""),
                                 font_size=38) for i in range(4)])
        for tm in terms[1:]:
            tm[0][:2].set_color(PURPLE)
        terms.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.4 + UP * 0.6)
        sumt = T("y ≈ the sum of these pieces", 22, MUTED).next_to(terms, UP, buff=0.35).align_to(terms, LEFT)
        with self.voice(N["bend1"]) as v:
            self.play(Create(ax), run_time=0.8)
            self.wait_until(v, "squared")
            self.play(Create(par), FadeIn(pl), run_time=1)
            self.wait_until(v, "pieces")
            self.play(FadeIn(sumt), run_time=0.5)
            self.wait_until(v, "corner")
            self.play(Indicate(par, color=YELLOW), run_time=0.8)
        cur = gs[0]
        with self.voice(N["bend2"]) as v:
            self.wait_until(v, "four")
            self.play(Create(cur), FadeIn(terms[0]), run_time=0.8)
            self.wait_until(v, "Add")
            self.play(Transform(cur, gs[1]), FadeIn(terms[1]), run_time=1)
            self.wait_until(v, "Add", 2)
            self.play(Transform(cur, gs[2]), FadeIn(terms[2]), run_time=1)
        dots = VGroup(*[Dot(ax.c2p(k, k * k), color=GREEN, radius=0.09) for k in (-2, -1, 0, 1, 2)])
        with self.voice(N["bend3"]) as v:
            self.wait_until(v, "Add")
            self.play(Transform(cur, gs[3]), FadeIn(terms[3]), run_time=1)
            self.wait_until(v, "Add", 2)
            self.play(Transform(cur, gs[4]), FadeIn(terms[4]), run_time=1)
            self.wait_until(v, "touches")
            self.play(FadeIn(dots), run_time=0.6)
        k8 = np.linspace(-2, 2, 9)
        a8 = demo.a8.tolist()
        g8 = ax.plot(lambda t: 4 + sum(a8[i] * relu(t - k8[i]) for i in range(8)), x_range=[-2, 2],
                     color=GREEN, stroke_width=4, use_smoothing=False)
        e = VGroup(MathTex(r"4 \text{ pieces: max gap } %g" % round(demo.err4, 4), color=YELLOW, font_size=34),
                   MathTex(r"8 \text{ pieces: max gap } %g" % round(demo.err8, 4), color=GREEN, font_size=34)
                   ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(terms, DOWN, buff=0.5).align_to(terms, LEFT)
        with self.voice(N["bend4"]) as v:
            self.wait_until(v, "gap")
            self.play(FadeIn(e[0]), run_time=0.6)
            self.wait_until(v, "eight")
            self.play(FadeOut(cur), Create(g8), FadeIn(e[1]), run_time=1.2)
            self.wait_until(v, "continuous")
            self.play(Indicate(g8, color=GREEN), run_time=0.8)


class C5_Slope(Studio):
    def construct(self):
        self.heading("5 · The slope of ReLU — and dead units")
        ax1 = axes([-3, 3, 1], [-0.5, 3, 1], 5.2, 2.6, fs=20).move_to(LEFT * 3.6 + UP * 0.9)
        r = ax1.plot(relu, x_range=[-3, 3], color=YELLOW, stroke_width=5, use_smoothing=False)
        ax2 = axes([-3, 3, 1], [-0.3, 1.4, 1], 5.2, 1.7, fs=20).move_to(LEFT * 3.6 + DOWN * 2.1)
        s_l = Line(ax2.c2p(-3, 0), ax2.c2p(0, 0), color=RED, stroke_width=6)
        s_r = Line(ax2.c2p(0, 1), ax2.c2p(3, 1), color=GREEN, stroke_width=6)
        l1 = T("ReLU(z)", 20, YELLOW).next_to(ax1, UP, buff=0.05).align_to(ax1, LEFT)
        l2 = T("slope of ReLU", 20, MUTED).next_to(ax2, UP, buff=0.05).align_to(ax2, LEFT)
        lab_r = T("slope 1", 24, GREEN).move_to(ax1.c2p(1.6, 2.6))
        lab_l = T("slope 0", 24, RED).move_to(ax1.c2p(-1.6, 0.6))
        corner = Dot(ax1.c2p(0, 0), color=RED, radius=0.1)
        cl = T("corner at 0 · PyTorch uses slope 0 there", 22, MUTED).move_to(RIGHT * 3.4 + UP * 2.2)
        rem = self.reminder_card("derivative", "The slope of a function at a point (video 7).", pos=DR, width=4.4)
        with self.voice(N["slope1"]) as v:
            self.play(Create(ax1), Create(r), FadeIn(l1), run_time=1)
            self.wait_until(v, "derivative")
            self.play(FadeIn(rem, shift=UP * 0.2), Create(ax2), FadeIn(l2), run_time=0.8)
            self.wait_until(v, "right")
            self.play(FadeIn(lab_r), Create(s_r), run_time=0.6)
            self.wait_until(v, "left")
            self.play(FadeIn(lab_l), Create(s_l), run_time=0.6)
            self.wait_until(v, "corner")
            self.play(FadeIn(corner), FadeIn(cl), run_time=0.6)
        nums = VGroup(MathTex(r"\text{slope at } -2 = %g" % demo.slopes[0], color=RED, font_size=36),
                      MathTex(r"\text{slope at } 3 = %g" % demo.slopes[1], color=GREEN, font_size=36),
                      T("slope 1 → the signal passes back unshrunk", 24, FG),
                      T("sigmoid, tanh: tiny slopes in the flat tails", 24, MUTED)
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.4 + UP * 0.4)
        with self.voice(N["slope2"]) as v:
            self.play(FadeOut(rem), run_time=0.4)
            self.wait_until(v, "minus")
            self.play(FadeIn(nums[0]), run_time=0.5)
            self.wait_until(v, "three")
            self.play(FadeIn(nums[1]), run_time=0.5)
            self.wait_until(v, "good")
            self.play(FadeIn(nums[2]), run_time=0.5)
            self.wait_until(v, "replaced")
            self.play(FadeIn(nums[3]), run_time=0.5)
        self.clear_body()
        nl1 = NumberLine(x_range=[0, 5, 1], length=5, color=MUTED, include_numbers=True, font_size=24).move_to(LEFT * 3.5 + UP * 0.8)
        nl2 = NumberLine(x_range=[-10, 0, 1], length=6, color=MUTED, include_numbers=True, font_size=22).move_to(LEFT * 3.0 + DOWN * 1.6)
        d_in = VGroup(*[Dot(nl1.n2p(v), color=BLUE, radius=0.07) for v in demo.xs])
        d_pre = VGroup(*[Dot(nl2.n2p(v), color=RED, radius=0.07) for v in demo.pre])
        li = T("inputs x", 22, BLUE).next_to(nl1, UP, buff=0.3)
        lp = MathTex(r"z = 1\cdot x - 10", color=RED, font_size=36).next_to(nl2, UP, buff=0.3)
        arr = Arrow(nl1.get_bottom() + DOWN * 0.4, lp.get_top() + UP * 0.1, buff=0, color=MUTED)
        res = VGroup(MathTex(r"z < 0 \text{ always}", color=RED, font_size=38),
                     MathTex(r"\mathrm{ReLU}(z) = 0,\ \ \text{slope} = 0", color=RED, font_size=38),
                     MathTex(r"\text{inputs with slope 1: } %d \text{ of } %d" % (demo.alive, len(demo.xs)),
                             font_size=36)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.6 + UP * 0.4)
        with self.voice(N["slope3"]) as v:
            self.wait_until(v, "Imagine")
            self.play(Create(nl1), FadeIn(li), FadeIn(d_in), run_time=1)
            self.wait_until(v, "negative")
            self.play(GrowArrow(arr), FadeIn(lp), Create(nl2), TransformFromCopy(d_in, d_pre), FadeIn(res[0]), run_time=1.5)
            self.wait_until(v, "always", 2)
            self.play(FadeIn(res[1]), run_time=0.6)
            self.wait_until(v, "eleven")
            self.play(FadeIn(res[2]), run_time=0.6)
        dead = T("a dead ReLU", 36, RED, weight=BOLD).move_to(RIGHT * 3.6 + DOWN * 1.6)
        fix = T("GELU: small but non-zero slope for z < 0", 24, BLUE).next_to(dead, DOWN, buff=0.3)
        if fix.get_right()[0] > 7.0:
            fix.shift(LEFT * (fix.get_right()[0] - 7.0))
        with self.voice(N["slope4"]) as v:
            self.wait_until(v, "dead")
            self.play(FadeIn(dead), Indicate(d_pre, color=RED), run_time=0.8)
            self.wait_until(v, "GELU")
            self.play(FadeIn(fix), run_time=0.6)
            self.wait_until(v, "learn")
            self.play(Indicate(fix, color=BLUE), run_time=0.8)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · Activations in real networks")
        parts = [("Linear(in, 30)", PURPLE, 2.3), ("ReLU", YELLOW, 1.2), ("Linear(30, 20)", PURPLE, 2.3),
                 ("ReLU", YELLOW, 1.2), ("Linear(20, out)", PURPLE, 2.3)]
        mlp = VGroup()
        for i, (s, c, w) in enumerate(parts):
            mlp.add(box(s, c, w=w, size=22))
        mlp.arrange(RIGHT, buff=0.35).shift(UP * 1.4)
        if mlp.width > 13.4:
            mlp.scale_to_fit_width(13.4)
        arrows = VGroup(*[Arrow(mlp[i].get_right(), mlp[i + 1].get_left(), buff=0.03, color=MUTED,
                                stroke_width=3, max_tip_length_to_length_ratio=0.4) for i in range(4)])
        ml = T("MLP from the book (Raschka, §A.5) · activation after every linear layer except the last",
               20, MUTED).next_to(mlp, DOWN, buff=0.3)
        with self.voice(N["ml1"]) as v:
            self.wait_until(v, "perceptron")
            self.play(LaggedStart(*[FadeIn(b) for b in mlp], lag_ratio=0.2), Create(arrows), run_time=1.5)
            self.wait_until(v, "thirty")
            self.play(Indicate(mlp[0], color=PURPLE), Indicate(mlp[2], color=PURPLE), run_time=0.8)
            self.wait_until(v, "last")
            self.play(FadeIn(ml), run_time=0.6)
        ffn = VGroup(box("Linear", PURPLE, w=2.0), box("GELU", BLUE, w=1.6), box("Linear", PURPLE, w=2.0)
                     ).arrange(RIGHT, buff=0.6).shift(DOWN * 1.4)
        fa = VGroup(*[Arrow(ffn[i].get_right(), ffn[i + 1].get_left(), buff=0.05, color=MUTED) for i in range(2)])
        fl = T("GPT-2 transformer block: feed-forward network", 24, BLUE).next_to(ffn, UP, buff=0.3)
        zp = T("0 parameters", 22, YELLOW).next_to(ffn[1], DOWN, buff=0.25)
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "transformer")
            self.play(FadeIn(fl), FadeIn(ffn[0]), FadeIn(ffn[2]), run_time=0.8)
            self.wait_until(v, "GELU")
            self.play(FadeIn(ffn[1]), GrowArrow(fa[0]), GrowArrow(fa[1]), run_time=0.8)
            self.wait_until(v, "parameters")
            self.play(FadeIn(zp), run_time=0.5)
            self.wait_until(v, "depth")
            self.play(Indicate(ffn[1], color=YELLOW), run_time=0.8)
        self.clear_body()
        ax_t = T("Axiom A · linearity  +  one deliberate bend", 40, YELLOW, weight=BOLD).shift(UP * 0.8)
        if ax_t.width > 12.8:
            ax_t.scale_to_fit_width(12.8)
        sub = VGroup(T("linearity alone → one matrix", 30, MUTED),
                     T("linearity + a bend → deep learning", 30, GREEN)
                     ).arrange(DOWN, buff=0.35).next_to(ax_t, DOWN, buff=0.6)
        with self.voice(N["ml3"]) as v:
            self.wait_until(v, "axiom")
            self.play(Write(ax_t), run_time=1)
            self.wait_until(v, "alone")
            self.play(FadeIn(sub[0]), run_time=0.6)
            self.wait_until(v, "bend")
            self.play(FadeIn(sub[1]), run_time=0.6)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        q = MathTex(r"\mathrm{ReLU}\left(\begin{bmatrix}-3\\0.5\\2\end{bmatrix}\right) = \;?", r"\qquad \text{slopes} = \;?",
                    font_size=56).shift(UP * 1.2)
        q[1].set_color(YELLOW)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac1"]) as v:
            self.play(Write(q[0]), run_time=1.5)
            self.wait_until(v, "Pause")
            self.play(FadeIn(pause), run_time=0.5)
            self.wait_until(v, "slope")
            self.play(FadeIn(q[1]), run_time=0.6)
        a = VGroup(MathTex(r"\mathrm{ReLU} = \begin{bmatrix}0\\0.5\\2\end{bmatrix}", color=GREEN, font_size=48),
                   MathTex(r"\text{slopes} = \begin{bmatrix}0\\1\\1\end{bmatrix}", color=YELLOW, font_size=48)
                   ).arrange(RIGHT, buff=1.5).next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "result")
            self.play(Write(a[0]), run_time=1)
            self.wait_until(v, "slopes")
            self.play(Write(a[1]), run_time=1)
        self.clear_body()
        q2 = VGroup(MathTex(r"g(x) = \mathrm{ReLU}(x) + \mathrm{ReLU}(-x)", font_size=48),
                    MathTex(r"g(-2) = \;?\qquad g(3) = \;?", color=YELLOW, font_size=44),
                    T("⏸  pause and try", 28, MUTED)).arrange(DOWN, buff=0.4).move_to(RIGHT * 2.6 + UP * 1.5)
        with self.voice(N["prac3"]) as v:
            self.wait_until(v, "Define")
            self.play(Write(q2[0]), run_time=1.2)
            self.wait_until(v, "three")
            self.play(FadeIn(q2[1]), run_time=0.6)
            self.wait_until(v, "Pause")
            self.play(FadeIn(q2[2]), run_time=0.5)
        ax = axes([-3, 3, 1], [-0.5, 3.2, 1], 4.8, 3.8).move_to(LEFT * 4.2 + DOWN * 0.8)
        gl = ax.plot(lambda t: abs(t), x_range=[-3, 3], color=YELLOW, stroke_width=5, use_smoothing=False)
        ans = VGroup(MathTex(r"g(-2) = 0 + 2 = 2", font_size=40),
                     MathTex(r"g(3) = 3 + 0 = 3", font_size=40),
                     MathTex(r"g(x) = |x|", color=YELLOW, font_size=48)
                     ).arrange(DOWN, buff=0.3).move_to(RIGHT * 2.6 + DOWN * 1.3)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(q2[2]), Create(ax), run_time=0.6)
            self.wait_until(v, "For")
            self.play(FadeIn(ans[0]), run_time=0.6)
            self.wait_until(v, "For", 2)
            self.play(FadeIn(ans[1]), run_time=0.6)
            self.wait_until(v, "absolute")
            self.play(Write(ans[2]), Create(gl), run_time=1)


CODE = '''import numpy as np
relu = lambda z: np.maximum(0, z)
W1, W2 = np.array([[1, 2], [0, 1]]), np.array([[2, 0], [1, 1]])
x = np.array([3, -1])
print(W2 @ (W1 @ x), (W2 @ W1) @ x)    # two layers = one matrix
print(W2 @ relu(W1 @ x))               # ReLU in between
t = np.linspace(-2, 2, 4001)
g = 4 - 3*relu(t+2) + 2*relu(t+1) + 2*relu(t) + 2*relu(t-1)
print(np.max(np.abs(g - t**2)))        # largest error, 4 pieces
e = 1e-6
print([(relu(v+e) - relu(v-e)) / (2*e) for v in (-2.0, 3.0)])'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=22)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box_ = SurroundingRectangle(code.code_lines[1], color=YELLOW, buff=0.06)
            self.wait_until(v, "maximum")
            self.play(Create(box_), run_time=0.5)
            for i, w in [(4, "combined"), (5, "between"), (7, "squared"), (8, "error"), (10, "slopes")]:
                self.wait_until(v, w)
                self.play(Transform(box_, SurroundingRectangle(code.code_lines[i], color=YELLOW,
                                                               buff=0.06)), run_time=0.5)
        self.clear_body()
        out = run_python(HERE / "demo.py")
        term = self.terminal(out, font_size=22, command="python demo.py").shift(DOWN * 0.3)
        if term.width > 13.4:
            term.scale_to_fit_width(13.4)
        with self.voice(N["code2"]) as v:
            self.play(FadeIn(term[0]), FadeIn(term[2]), run_time=0.5)
            self.play(AddTextLetterByLetter(term[1], run_time=2.5))
            for i, w in [(1, "both"), (2, "between"), (4, "weights"), (6, "slopes"), (7, "dead")]:
                self.wait_until(v, w)
                self.play(Indicate(term[1][i], color=YELLOW), run_time=0.6)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["Linear layers stacked = one linear layer (associativity)",
                            "Activation function = a nonlinearity between layers",
                            "ReLU(z) = max(0, z): keep positives, zero out negatives",
                            "Sigmoid (0, 1), tanh (−1, 1), GELU: smooth bends",
                            "Sums of shifted ReLUs draw curves",
                            "ReLU slope is 0 or 1; always-negative units die",
                            "MLPs use ReLU; GPT-2 uses GELU"],
                           size=24, buff=0.26).shift(UP * 0.15)
        if pts.width > 13:
            pts.scale_to_fit_width(13)
        with self.voice(N["sum1"]) as v:
            for p in pts[:4]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 6)
        nxt = VGroup(T("That completes the atoms — next: the molecules map", 32, BLUE, weight=BOLD),
                     T("how atoms combine into gradient descent, attention and more", 24, MUTED)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.45)
        with self.voice(N["sum2"]) as v:
            for p, w in zip(pts[4:], ["Sums", "slope", "MLPs"]):
                self.wait_until(v, w)
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
            self.wait_until(v, "completes")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Collapse", "C2_ReLU", "C3_Shapes", "C4_Bends", "C5_Slope",
          "C6_ML", "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Collapse": "Two linear layers = one",
                  "C2_ReLU": "The bend: ReLU", "C3_Shapes": "Sigmoid, tanh, GELU",
                  "C4_Bends": "How bends build curves", "C5_Slope": "Slope of ReLU & dead units",
                  "C6_ML": "Activations in real networks", "C7_Practice": "Try it yourself",
                  "C7_Code": "Python demo", "C8_Summary": "Summary", "C9_Credits": "Credits"}
