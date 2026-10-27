"""Chain rule — scenes. Build: python <skill>/scripts/build.py <this_dir> [--draft]"""
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

SERIES = "ML Math Atoms · Video 9 of 16"
TITLE = "Chain rule — slopes multiply"


def fit(m, w):
    if m.width > w:
        m.scale_to_fit_width(w)
    return m


def machine(label, color, w=1.6, h=1.1, size=48):
    box = RoundedRectangle(width=w, height=h, corner_radius=0.15, color=color, stroke_width=3,
                           fill_color=color, fill_opacity=0.08)
    return VGroup(box, MathTex(label, color=color, font_size=size).move_to(box))


def pipeline(nodes, boxes, buff=0.45):
    """nodes: list of mobjects (values), boxes: machines between them. Returns group + arrows."""
    items = []
    for i, b in enumerate(boxes):
        items += [nodes[i], b]
    items.append(nodes[-1])
    row = VGroup(*items).arrange(RIGHT, buff=buff)
    arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.08, color=MUTED, stroke_width=3,
                            max_tip_length_to_length_ratio=0.35)
                      for a, b in zip(items[:-1], items[1:])])
    return row, arrows


def legend(rows, x_sym=-3.4, x_txt=-2.4, y0=0.2, dy=0.75, size=26, fs=44):
    g = VGroup()
    for i, (sym, txt, color) in enumerate(rows):
        s = MathTex(sym, color=color, font_size=fs).move_to([x_sym, y0 - i * dy, 0])
        t = T(txt, size, FG)
        t.move_to([x_txt, y0 - i * dy, 0], aligned_edge=LEFT)
        g.add(VGroup(s, t))
    return g


def gear(r, color, teeth=10):
    c = Circle(radius=r, color=color, stroke_width=4, fill_color=color, fill_opacity=0.12)
    sp = VGroup(*[Line(ORIGIN, r * np.array([np.cos(a), np.sin(a), 0]), color=color, stroke_width=3)
                  for a in np.linspace(0, 2 * PI, 4, endpoint=False)])
    tt = VGroup(*[Square(0.16, color=color, fill_color=color, fill_opacity=1, stroke_width=0)
                  .move_to((r + 0.06) * np.array([np.cos(a), np.sin(a), 0])).rotate(a)
                  for a in np.linspace(0, 2 * PI, teeth, endpoint=False)])
    mark = Dot(0.65 * r * RIGHT, color=YELLOW, radius=0.07)
    return VGroup(c, sp, tt, mark)


class C0_Intro(Studio):
    def construct(self):
        top = T(SERIES, 26, MUTED)
        title = T(TITLE, 60, FG, weight=BOLD)
        fit(title, 12.5)
        VGroup(top, title).arrange(DOWN, buff=0.35).shift(UP * 2.2)
        nodes = [MathTex("x", color=BLUE, font_size=48)] + \
                [MathTex(r"\cdot", color=MUTED, font_size=20) for _ in range(3)] + \
                [MathTex(r"\text{loss}", color=RED, font_size=44)]
        boxes = [machine(r"\text{layer %d}" % i, PURPLE, 1.9, 0.9, 34) for i in (1, 2, 3)]
        boxes.append(machine(r"L", RED, 1.0, 0.9, 40))
        row, ars = pipeline(nodes, boxes, buff=0.3)
        fit(VGroup(row, ars), 13.4)
        VGroup(row, ars).shift(DOWN * 0.3)
        w = MathTex("w", color=PURPLE, font_size=44).next_to(boxes[0], DOWN, buff=0.35)
        q = MathTex(r"\frac{\partial\,\text{loss}}{\partial w}=\ ?", font_size=48).next_to(row, DOWN, buff=1.0)
        mult = T("slopes multiply", 44, YELLOW, weight=BOLD).next_to(q, DOWN, buff=0.35)
        with self.voice(N["intro1"]) as v:
            self.play(FadeIn(top), Write(title), run_time=2)
            self.wait_until(v, "first")
            self.play(FadeIn(nodes[0]), GrowArrow(ars[0]), FadeIn(boxes[0]), run_time=0.7)
            self.wait_until(v, "second")
            self.play(LaggedStart(*[FadeIn(m) for m in list(row[2:]) + list(ars[1:])], lag_ratio=0.1),
                      run_time=1.5)
            self.wait_until(v, "weight")
            self.play(FadeIn(w), Write(q), run_time=0.8)
            self.wait_until(v, "multiply")
            self.play(FadeIn(mult, shift=UP * 0.2), run_time=0.6)
        items = VGroup(*[VGroup(MathTex(s, color=c, font_size=46), T(t, 24, MUTED)).arrange(DOWN, buff=0.3)
                         for s, c, t in [(r"g(f(x))", BLUE, "pipeline"),
                                         (r"\approx\ \text{twice}", ORANGE, "why"),
                                         (r"8\times3=24", YELLOW, "examples"),
                                         (r"J", GREEN, "Jacobian"),
                                         (r"\leftarrow", RED, "backprop")]])
        items.arrange(RIGHT, buff=0.8).shift(DOWN * 0.5)
        fit(items, 13)
        pre = T("you need: video 7 (derivative) · video 8 (gradient)", 26, MUTED).to_edge(DOWN, buff=0.7)
        with self.voice(N["intro2"]) as v:
            self.play(*[FadeOut(m) for m in (row, ars, w, q, mult)], run_time=0.6)
            for it, wd in zip(items, ["pipeline", "twice", "examples", "Jacobian", "backpropagation"]):
                self.wait_until(v, wd)
                self.play(FadeIn(it, shift=UP * 0.2), run_time=0.5)
            self.wait_until(v, "need")
            self.play(FadeIn(pre), run_time=0.6)


class C1_Composition(Studio):
    def construct(self):
        self.heading("1 · Functions inside functions")
        nx = MathTex("x", color=BLUE, font_size=56)
        nu = MathTex("u", color=ORANGE, font_size=56)
        ny = MathTex("y", color=YELLOW, font_size=56)
        F = machine("f", ORANGE)
        G = machine("g", GREEN)
        row, ars = pipeline([nx, nu, ny], [F, G], buff=0.6)
        VGroup(row, ars).move_to(UP * 1.8 + LEFT * 1.6)
        comp = MathTex(r"y", "=", r"g\big(f(x)\big)", r"=(g\circ f)(x)", font_size=54)
        comp[0].set_color(YELLOW)
        comp.move_to(LEFT * 1.6 + DOWN * 0.3)
        card = self.term_card("composition", "Putting one function after another: the output "
                              "of f becomes the input of g, written g(f(x)).", pos=DR, width=5.4)
        with self.voice(N["comp1"]) as v:
            self.wait_until(v, "first")
            self.play(FadeIn(nx), GrowArrow(ars[0]), FadeIn(F), GrowArrow(ars[1]), FadeIn(nu), run_time=1)
            self.wait_until(v, "second")
            self.play(GrowArrow(ars[2]), FadeIn(G), GrowArrow(ars[3]), FadeIn(ny), run_time=1)
            self.wait_until(v, "composition")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "write")
            self.play(Write(comp), run_time=1.2)
        fl = MathTex(r"f(x)=3x+1", color=ORANGE, font_size=40).next_to(F, DOWN, buff=0.3)
        gl = MathTex(r"g(u)=u^2", color=GREEN, font_size=40).next_to(G, DOWN, buff=0.3)
        v1 = MathTex("1", color=BLUE, font_size=56).move_to(nx)
        v2 = MathTex("4", color=ORANGE, font_size=56).move_to(nu)
        v3 = MathTex("16", color=YELLOW, font_size=56).move_to(ny)
        res = MathTex(r"g(f(1))", "=", r"(3\cdot1+1)^2", "=", "16", font_size=50).move_to(LEFT * 1.6 + DOWN * 1.6)
        res[4].set_color(YELLOW)
        with self.voice(N["comp2"]) as v:
            self.play(FadeOut(card), run_time=0.4)
            self.wait_until(v, "inner")
            self.play(FadeIn(fl), run_time=0.6)
            self.wait_until(v, "outer")
            self.play(FadeIn(gl), run_time=0.6)
            self.wait_until(v, "one", 2)
            self.play(Transform(nx, v1), run_time=0.6)
            self.wait_until(v, "four")
            self.play(Indicate(F), Transform(nu, v2), run_time=0.8)
            self.wait_until(v, "sixteen")
            self.play(Indicate(G), Transform(ny, v3), run_time=0.8)
            self.wait_until(v, "So")
            self.play(Write(res), run_time=1)
        rev = MathTex(r"f(g(1))", "=", r"3\cdot1^2+1", "=", "4", r"\ \neq\ 16", font_size=50, color=RED)
        rev.next_to(res, DOWN, buff=0.4)
        inside = T("read from the inside out", 26, MUTED).next_to(rev, DOWN, buff=0.3)
        with self.voice(N["comp3"]) as v:
            self.wait_until(v, "first")
            self.play(Write(rev[:3]), run_time=1)
            self.wait_until(v, "four")
            self.play(Write(rev[3:]), run_time=0.8)
            self.wait_until(v, "inside")
            self.play(FadeIn(inside), run_time=0.5)

        self.clear_body()
        self.heading("1 · Gears: rates multiply")
        A, B, Cg = gear(1.3, BLUE, 12), gear(0.95, ORANGE, 10), gear(0.75, YELLOW, 8)
        VGroup(A, B, Cg).arrange(RIGHT, buff=1.3).move_to(UP * 0.6)
        la = VGroup(MathTex("x", color=BLUE, font_size=44), T("gear 1", 22, MUTED)).arrange(DOWN, buff=0.1)
        lb = VGroup(MathTex("u", color=ORANGE, font_size=44), T("gear 2", 22, MUTED)).arrange(DOWN, buff=0.1)
        lc = VGroup(MathTex("y", color=YELLOW, font_size=44), T("gear 3", 22, MUTED)).arrange(DOWN, buff=0.1)
        for l, g in zip((la, lb, lc), (A, B, Cg)):
            l.next_to(g, DOWN, buff=0.35)
        r1 = MathTex(r"\times 3", color=ORANGE, font_size=48).move_to((A.get_right() + B.get_left()) / 2 + UP * 0.6)
        r2 = MathTex(r"\times 2", color=YELLOW, font_size=48).move_to((B.get_right() + Cg.get_left()) / 2 + UP * 0.6)
        with self.voice(N["gear1"]) as v:
            self.wait_until(v, "three")
            self.play(FadeIn(A), FadeIn(B), FadeIn(Cg), FadeIn(la), FadeIn(lb), FadeIn(lc), run_time=1)
            self.wait_until(v, "turns", 2)
            self.play(FadeIn(r1), run_time=0.5)
            self.wait_until(v, "turns", 4)
            self.play(FadeIn(r2), run_time=0.5)
        six = MathTex(r"3\times 2=6", color=YELLOW, font_size=64).to_edge(DOWN, buff=0.6)
        not_add = T("multiply, not add (3 + 2 = 5 would be wrong)", 24, RED).next_to(six, UP, buff=0.25)
        with self.voice(N["gear2"]) as v:
            self.play(Rotate(A, -PI / 2, about_point=A.get_center()),
                      Rotate(B, -3 * PI / 2, about_point=B.get_center()),
                      Rotate(Cg, -3 * PI, about_point=Cg.get_center()),
                      run_time=v.duration * 0.45, rate_func=linear)
            self.wait_until(v, "six")
            self.play(Write(six), run_time=0.8)
            self.wait_until(v, "add")
            self.play(FadeIn(not_add), run_time=0.5)
        self.clear_body()
        f = MathTex(r"\frac{dy}{dx}", "=", r"\frac{dy}{du}", r"\cdot", r"\frac{du}{dx}", font_size=90).shift(UP * 1.0)
        f[0].set_color(YELLOW); f[2].set_color(GREEN); f[4].set_color(ORANGE)
        leg = VGroup(T("y vs x: the whole pipeline", 26, YELLOW),
                     T("y vs u: the outer slope", 26, GREEN),
                     T("u vs x: the inner slope", 26, ORANGE)).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        leg.next_to(f, DOWN, buff=0.5).to_edge(LEFT, buff=1.0)
        card = self.term_card("chain rule", "The slope of a composition is the product of the "
                              "slopes: dy/dx = dy/du · du/dx.", pos=DR, width=5.4)
        canc = T("like fractions cancelling du", 24, MUTED).next_to(f, UP, buff=0.3)
        with self.voice(N["gear3"]) as v:
            self.wait_until(v, "slope")
            self.play(Write(f[0]), FadeIn(leg[0]), run_time=0.8)
            self.wait_until(v, "slope", 2)
            self.play(Write(f[1:3]), FadeIn(leg[1]), run_time=0.8)
            self.wait_until(v, "times")
            self.play(Write(f[3:]), FadeIn(leg[2]), run_time=0.8)
            self.wait_until(v, "chain")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "fractions")
            self.play(FadeIn(canc), run_time=0.5)


class C2_Derive(Studio):
    def construct(self):
        self.heading("2 · Why slopes multiply")
        nh = MathTex("h", color=BLUE, font_size=50)
        nk = MathTex("k", color=ORANGE, font_size=50)
        ny = MathTex(r"\Delta y", color=YELLOW, font_size=50)
        F = machine(r"\times f'(x)", ORANGE, 2.3, 1.0, 40)
        G = machine(r"\times g'(u)", GREEN, 2.3, 1.0, 40)
        row, ars = pipeline([nh, nk, ny], [F, G], buff=0.5)
        VGroup(row, ars).move_to(UP * 1.9)
        s1 = MathTex(r"k", r"\approx", r"f'(x)\,h", font_size=50)
        s2 = MathTex(r"\Delta y", r"\approx", r"g'(u)\,k", font_size=50)
        s3 = MathTex(r"\Delta y", r"\approx", r"g'(u)\,f'(x)\,h", font_size=50)
        s4 = MathTex(r"\big(g(f(x))\big)'", "=", r"g'\big(f(x)\big)\cdot f'(x)", font_size=56)
        for s in (s1, s2, s3):
            s[0].set_color(ORANGE if s is s1 else YELLOW)
        s4.set_color(YELLOW)
        steps = VGroup(s1, s2, s3).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(LEFT * 2.6 + DOWN * 0.3)
        lab = VGroup(T("step 1: inner, close up", 22, MUTED), T("step 2: outer, close up", 22, MUTED),
                     T("put k in", 22, MUTED))
        for l, s in zip(lab, steps):
            l.next_to(s, RIGHT, buff=0.5)
        rem = self.reminder_card("local linear approximation", "Close up, a change in the input "
                                 "is multiplied by the slope: f(x+h) ≈ f(x) + f′(x)·h  (video 7).",
                                 pos=DR, width=5.4)
        with self.voice(N["der1"]) as v:
            self.wait_until(v, "local")
            self.play(FadeIn(rem, shift=UP * 0.3), run_time=0.6)
            self.wait_until(v, "nudge")
            self.play(FadeIn(nh), GrowArrow(ars[0]), FadeIn(F), run_time=0.8)
            self.wait_until(v, "prime")
            self.play(Write(s1), FadeIn(lab[0]), run_time=0.8)
            self.wait_until(v, "k")
            self.play(GrowArrow(ars[1]), FadeIn(nk), run_time=0.6)
        with self.voice(N["der2"]) as v:
            self.play(FadeOut(rem), run_time=0.4)
            self.wait_until(v, "outer")
            self.play(FadeIn(G), GrowArrow(ars[2]), run_time=0.6)
            self.wait_until(v, "prime")
            self.play(Write(s2), FadeIn(lab[1]), GrowArrow(ars[3]), FadeIn(ny), run_time=1)
            self.wait_until(v, "put")
            self.play(Write(s3), FadeIn(lab[2]), run_time=1)
        s4.next_to(steps, DOWN, buff=0.45).set_x(0)
        box = SurroundingRectangle(s4, color=YELLOW, buff=0.2, corner_radius=0.1)
        note = T("two multiplications in a row = one multiplication by the product", 24, MUTED)
        note.to_edge(DOWN, buff=0.35)
        with self.voice(N["der3"]) as v:
            self.wait_until(v, "slope")
            self.play(Write(s4), run_time=1.2)
            self.wait_until(v, "chain")
            self.play(Create(box), run_time=0.6)
            self.wait_until(v, "Two")
            self.play(FadeIn(note), run_time=0.6)


class C3_Example(Studio):
    def construct(self):
        self.heading("3 · Worked example: (3x + 1)² at x = 1")
        nx = MathTex("x=1", color=BLUE, font_size=44)
        nu = MathTex("u=4", color=ORANGE, font_size=44)
        ny = MathTex("y=16", color=YELLOW, font_size=44)
        F = machine(r"3x+1", ORANGE, 2.0, 1.0, 40)
        G = machine(r"u^2", GREEN, 1.6, 1.0, 40)
        row, ars = pipeline([nx, nu, ny], [F, G], buff=0.5)
        VGroup(row, ars).move_to(UP * 1.8)
        sf = MathTex(r"\text{slope } 3", color=ORANGE, font_size=36).next_to(F, DOWN, buff=0.25)
        sg = MathTex(r"\text{slope } 2u=%g" % demo.outer, color=GREEN, font_size=36).next_to(G, DOWN, buff=0.25)
        tot = MathTex(r"%g\times%g=%g" % (demo.outer, demo.inner, demo.chain), color=YELLOW, font_size=64)
        tot.move_to(DOWN * 0.6)
        with self.voice(N["ex1"]) as v:
            self.play(FadeIn(row), FadeIn(ars), run_time=1)
            self.wait_until(v, "inner")
            self.play(FadeIn(sf), Indicate(F, color=ORANGE), run_time=0.8)
            self.wait_until(v, "outer")
            self.play(Indicate(G, color=GREEN), run_time=0.6)
            self.wait_until(v, "inner", 2)
            self.play(Indicate(nu, color=ORANGE), run_time=0.6)
            self.wait_until(v, "eight")
            self.play(FadeIn(sg), run_time=0.6)
        bad = MathTex(r"\text{mistake: } 2u \text{ at } x=1 \Rightarrow 2\times3=6\ \ \times", color=RED, font_size=42)
        bad.next_to(tot, DOWN, buff=0.6)
        with self.voice(N["ex2"]) as v:
            self.wait_until(v, "Multiply")
            self.play(Write(tot), run_time=1)
            self.wait_until(v, "mistake")
            self.play(Write(bad), run_time=1)
        self.clear_body()
        ax = Axes(x_range=[0, 1.6, 0.5], y_range=[0, 36, 10], x_length=5.0, y_length=4.6, tips=False,
                  axis_config={"stroke_color": MUTED}).shift(LEFT * 3.8 + DOWN * 0.6)
        cv = ax.plot(lambda x: (3 * x + 1) ** 2, x_range=[0, 1.6], color=BLUE, stroke_width=5)
        tan = Line(ax.c2p(0.6, 16 - 24 * 0.4), ax.c2p(1.4, 16 + 24 * 0.4), color=YELLOW, stroke_width=4)
        dot = Dot(ax.c2p(1, 16), color=YELLOW, radius=0.08)
        tl = T("slope 24 at x = 1", 24, YELLOW).next_to(ax.c2p(0.25, 32), RIGHT, buff=0)
        c = VGroup(MathTex(r"(3x+1)^2=9x^2+6x+1", font_size=44),
                   MathTex(r"\frac{d}{dx}:\ 18x+6 \ \Rightarrow\ 24", color=YELLOW, font_size=44),
                   MathTex(r"\text{numeric: } %.4f" % demo.num, color=GREEN, font_size=44),
                   T("three answers agree", 26, GREEN)).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        c.move_to(RIGHT * 3.0 + DOWN * 0.3)
        with self.voice(N["ex3"]) as v:
            self.play(Create(ax), Create(cv), FadeIn(dot), run_time=1)
            self.wait_until(v, "expand")
            self.play(Write(c[0]), run_time=0.8)
            self.wait_until(v, "eighteen")
            self.play(Write(c[1]), Create(tan), FadeIn(tl), run_time=1)
            self.wait_until(v, "numeric")
            self.play(Write(c[2]), run_time=0.8)
            self.wait_until(v, "agree")
            self.play(FadeIn(c[3]), run_time=0.5)


class C4_Long(Studio):
    def construct(self):
        self.heading("4 · Longer chains")
        nodes = [MathTex(s, color=c, font_size=42) for s, c in
                 [("x=1", BLUE), ("a=2", ORANGE), ("b=4", GREEN), ("c=20", YELLOW)]]
        boxes = [machine(r"2x", ORANGE, 1.4, 0.9, 38), machine(r"a^2", GREEN, 1.4, 0.9, 38),
                 machine(r"5b", YELLOW, 1.4, 0.9, 38)]
        row, ars = pipeline(nodes, boxes, buff=0.35)
        fit(VGroup(row, ars), 13.2)
        VGroup(row, ars).move_to(UP * 1.7)
        sl = VGroup(*[MathTex(s, color=c, font_size=36).next_to(b, DOWN, buff=0.25) for s, c, b in
                      [(r"\text{slope } 2", ORANGE, boxes[0]), (r"\text{slope } 2a=4", GREEN, boxes[1]),
                       (r"\text{slope } 5", YELLOW, boxes[2])]])
        with self.voice(N["long1"]) as v:
            self.wait_until(v, "First")
            self.play(FadeIn(row[:3]), FadeIn(ars[:2]), run_time=0.8)
            self.wait_until(v, "Then")
            self.play(FadeIn(row[3:5]), FadeIn(ars[2:4]), run_time=0.8)
            self.wait_until(v, "Then", 2)
            self.play(FadeIn(row[5:]), FadeIn(ars[4:]), run_time=0.8)
            self.wait_until(v, "slopes")
            self.play(FadeIn(sl[0]), run_time=0.5)
            self.wait_until(v, "four")
            self.play(FadeIn(sl[1]), run_time=0.5)
            self.wait_until(v, "five", 2)
            self.play(FadeIn(sl[2]), run_time=0.5)
        prod = MathTex(r"\frac{dc}{dx}", "=", r"2\times4\times5", "=", "%g" % demo.back[-1], font_size=60)
        prod[0].set_color(YELLOW); prod[4].set_color(YELLOW)
        prod.move_to(DOWN * 0.4)
        chk = MathTex(r"c=5(2x)^2=20x^2\ \Rightarrow\ 40x \ \Rightarrow\ 40\ \checkmark", color=GREEN, font_size=44)
        chk.next_to(prod, DOWN, buff=0.5)
        with self.voice(N["long2"]) as v:
            self.wait_until(v, "multiplies")
            self.play(Write(prod), run_time=1.2)
            self.wait_until(v, "Check")
            self.play(Write(chk), run_time=1.2)
        self.clear_body()
        many = VGroup(*[machine(r"\times s_{%d}" % i, [ORANGE, GREEN, YELLOW, PURPLE][i % 4], 1.7, 1.0, 40)
                        for i in range(1, 6)]).arrange(RIGHT, buff=0.4)
        fit(many, 13).move_to(UP * 0.6)
        dots = MathTex(r"\cdots", font_size=60).next_to(many, RIGHT, buff=0.2)
        fit(VGroup(many, dots), 13.4)
        tot = MathTex(r"\text{slope}=s_1\times s_2\times s_3\times\cdots", color=YELLOW, font_size=52)
        tot.next_to(many, DOWN, buff=0.8)
        deep = T("a deep network: many layers, many factors", 26, MUTED).next_to(tot, DOWN, buff=0.4)
        with self.voice(N["long3"]) as v:
            self.wait_until(v, "long")
            self.play(LaggedStart(*[FadeIn(m, shift=RIGHT * 0.2) for m in many], lag_ratio=0.15),
                      FadeIn(dots), run_time=2)
            self.wait_until(v, "multiply")
            self.play(Write(tot), run_time=1)
            self.wait_until(v, "deep")
            self.play(FadeIn(deep), run_time=0.5)


class C5_Jacobian(Studio):
    def construct(self):
        self.heading("5 · Many inputs and outputs: the Jacobian")
        J = MathTex(r"J", "=", r"\begin{bmatrix}\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n}\\"
                    r"\vdots & & \vdots\\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}\end{bmatrix}",
                    font_size=54)
        J[0].set_color(GREEN)
        J.move_to(LEFT * 2.6 + UP * 0.3)
        rows = T("one row per output", 24, ORANGE).next_to(J, UP, buff=0.3)
        cols = T("one column per input", 24, BLUE).next_to(J, DOWN, buff=0.3)
        rem = self.reminder_card("partial derivative", "The slope when only one input moves and "
                                 "the others are frozen  (video 8).", pos=UR, width=4.6)
        rem.shift(DOWN * 0.9)
        card = self.term_card("Jacobian", "The matrix of all partial derivatives of a function "
                              "with many outputs: row = output, column = input.", pos=DR, width=5.0)
        with self.voice(N["jac1"]) as v:
            self.wait_until(v, "partial")
            self.play(FadeIn(rem, shift=DOWN * 0.2), run_time=0.6)
            self.wait_until(v, "grid")
            self.play(Write(J), run_time=1.5)
            self.wait_until(v, "row")
            self.play(FadeIn(rows), run_time=0.5)
            self.wait_until(v, "column")
            self.play(FadeIn(cols), run_time=0.5)
            self.wait_until(v, "Jacobian")
            self.play(FadeIn(card, shift=UP * 0.3), run_time=0.6)
        self.clear_body()
        Fd = MathTex(r"F(x_1,x_2)=\begin{bmatrix}x_1^2\\ x_1x_2\end{bmatrix}", color=BLUE, font_size=52)
        Jf = MathTex(r"J_F=\begin{bmatrix}2x_1 & 0\\ x_2 & x_1\end{bmatrix}", font_size=52)
        Jv = MathTex(r"J_F(1,2)=\begin{bmatrix}%g & %g\\ %g & %g\end{bmatrix}" % tuple(demo.J_F.ravel()),
                     color=YELLOW, font_size=52)
        VGroup(Fd, Jf, Jv).arrange(RIGHT, buff=0.6).move_to(UP * 1.2)
        fit(VGroup(Fd, Jf, Jv), 13.4)
        r1 = SurroundingRectangle(Jf[0][4:9], color=ORANGE, buff=0.08)
        r2 = SurroundingRectangle(Jf[0][9:13], color=ORANGE, buff=0.08)
        with self.voice(N["jac2"]) as v:
            self.wait_until(v, "Example")
            self.play(Write(Fd), run_time=1)
            self.wait_until(v, "first")
            self.play(Write(Jf), run_time=1)
            self.play(Create(r1), run_time=0.4)
            self.wait_until(v, "second")
            self.play(ReplacementTransform(r1, r2), run_time=0.5)
            self.wait_until(v, "point")
            self.play(FadeOut(r2), Write(Jv), run_time=1)
        lin = MathTex(r"f(x)=A\,x", r"\quad\Rightarrow\quad", r"J=A", font_size=58).move_to(DOWN * 1.0)
        lin[2].set_color(YELLOW)
        ln = T("Jacobian = the local linear mapping (a matrix)", 26, MUTED).next_to(lin, DOWN, buff=0.35)
        with self.voice(N["jac3"]) as v:
            self.wait_until(v, "linear")
            self.play(Write(lin), run_time=1.2)
            self.wait_until(v, "local")
            self.play(FadeIn(ln), run_time=0.6)
            self.wait_until(v, "matrix", 2)
            self.play(Indicate(lin[2], color=YELLOW), run_time=0.8)
        self.clear_body()
        Gd = MathTex(r"G(u_1,u_2)=u_1+u_2^2", color=GREEN, font_size=48)
        Jg = MathTex(r"J_G=\begin{bmatrix}1 & 2u_2\end{bmatrix}", font_size=48)
        Jgv = MathTex(r"F(1,2)=(%g,%g)\ \Rightarrow\ J_G=\begin{bmatrix}%g & %g\end{bmatrix}"
                      % (demo.u0[0], demo.u0[1], demo.J_G[0, 0], demo.J_G[0, 1]), color=YELLOW, font_size=48)
        top = VGroup(Gd, Jg).arrange(RIGHT, buff=0.8).move_to(UP * 1.8)
        Jgv.next_to(top, DOWN, buff=0.4)
        rule = MathTex(r"J_{G\circ F}", "=", r"J_G\,J_F", font_size=60).next_to(Jgv, DOWN, buff=0.5)
        rule[0].set_color(YELLOW)
        rem2 = self.reminder_card("matrix multiplication", "Row times column: multiply matching "
                                  "entries and add  (video 2).", pos=DR, width=4.8)
        with self.voice(N["jac4"]) as v:
            self.wait_until(v, "chained")
            self.play(Write(rule), FadeIn(rem2, shift=UP * 0.3), run_time=1.2)
            self.wait_until(v, "Let")
            self.play(Write(Gd), run_time=0.8)
            self.wait_until(v, "Jacobian")
            self.play(Write(Jg), run_time=0.8)
            self.wait_until(v, "row")
            self.play(Write(Jgv), run_time=1)
        prod = MathTex(r"\begin{bmatrix}1 & 4\end{bmatrix}\begin{bmatrix}2 & 0\\ 2 & 1\end{bmatrix}", "=",
                       r"\begin{bmatrix}1\cdot2+4\cdot2 & 1\cdot0+4\cdot1\end{bmatrix}", "=",
                       r"\begin{bmatrix}%g & %g\end{bmatrix}" % tuple(demo.J_chain[0]), font_size=44)
        prod[4].set_color(YELLOW)
        fit(prod, 13).next_to(rule, DOWN, buff=0.45)
        dchk = MathTex(r"G(F(x))=x_1^2+x_1^2x_2^2:\ \ \tfrac{\partial}{\partial x_1}=2x_1+2x_1x_2^2=10,\ \ "
                       r"\tfrac{\partial}{\partial x_2}=2x_1^2x_2=4\ \checkmark", color=GREEN, font_size=36)
        fit(dchk, 13.2).to_edge(DOWN, buff=0.35)
        with self.voice(N["jac5"]) as v:
            self.play(FadeOut(rem2), run_time=0.4)
            self.wait_until(v, "Multiply")
            self.play(Write(prod[:3]), run_time=1.2)
            self.wait_until(v, "ten")
            self.play(Write(prod[3:]), run_time=0.8)
            self.wait_until(v, "directly")
            self.play(Write(dchk), run_time=1.5)


class C6_ML(Studio):
    def construct(self):
        self.heading("6 · In ML: backpropagation (preview)")
        nodes = [MathTex(s, color=c, font_size=40) for s, c in
                 [("x", BLUE), ("a", ORANGE), ("b", GREEN), ("c", YELLOW)]]
        boxes = [machine(r"2x", ORANGE, 1.4, 0.9, 36), machine(r"a^2", GREEN, 1.4, 0.9, 36),
                 machine(r"5b", YELLOW, 1.4, 0.9, 36)]
        row, ars = pipeline(nodes, boxes, buff=0.55)
        VGroup(row, ars).move_to(UP * 1.2)
        net = T("input → layer 1 → layer 2 → … → loss", 28, MUTED).next_to(row, UP, buff=0.35)
        quote = T("LLM book (Raschka): backpropagation is an implementation of the chain rule for neural networks",
                  22, FG)
        fit(quote, 13).to_edge(DOWN, buff=0.4)
        mol = T("backpropagation = a molecule; here only a preview", 22, MUTED).next_to(quote, UP, buff=0.2)
        with self.voice(N["ml1"]) as v:
            self.wait_until(v, "chain")
            self.play(FadeIn(net), FadeIn(row), FadeIn(ars), run_time=1)
            self.wait_until(v, "backpropagation")
            self.play(FadeIn(quote), run_time=0.6)
            self.wait_until(v, "molecule")
            self.play(FadeIn(mol), run_time=0.5)
        vals = ["1", "%g" % demo.back[0], "%g" % demo.back[1], "%g" % demo.back[2]]
        targets = [nodes[3], nodes[2], nodes[1], nodes[0]]
        bl = VGroup(*[MathTex(val, color=RED, font_size=40).next_to(t, DOWN, buff=0.9) for val, t in zip(vals, targets)])
        barrs = VGroup(*[Arrow(bl[i].get_left(), bl[i + 1].get_right(), buff=0.15, color=RED, stroke_width=4)
                         for i in range(3)])
        mlab = VGroup(*[MathTex(s, color=RED, font_size=32).next_to(barrs[i], DOWN, buff=0.1)
                        for i, s in enumerate([r"\times5", r"\times4", r"\times2"])])
        bt = T("backward: start at the end with 1", 24, RED).next_to(bl, DOWN, buff=0.9)
        with self.voice(N["ml2"]) as v:
            self.wait_until(v, "end")
            self.play(FadeIn(bl[0]), FadeIn(bt), run_time=0.6)
            for i, w in enumerate(["five", "twenty", "forty"]):
                self.wait_until(v, w)
                self.play(GrowArrow(barrs[i]), FadeIn(mlab[i]), FadeIn(bl[i + 1]), run_time=0.6)
        reuse = T("each product computed once, reused further back", 26, YELLOW).next_to(bt, DOWN, buff=0.3)
        with self.voice(N["ml3"]) as v:
            self.play(FadeOut(mol), run_time=0.3)
            self.wait_until(v, "reused")
            self.play(FadeIn(reuse), Indicate(bl[1:3], color=YELLOW), run_time=0.8)
            self.wait_until(v, "backward")
            self.play(Indicate(barrs, color=RED), run_time=0.8)


class C7_Practice(Studio):
    def construct(self):
        self.heading("7 · Try it yourself")
        q = MathTex(r"\frac{d}{dx}(2x-1)^3 \ \text{ at } x=1 \;=\; ?", font_size=64).shift(UP * 1.4)
        pause = T("⏸  pause and try", 30, MUTED).next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac1"]) as v:
            self.wait_until(v, "slope")
            self.play(Write(q), run_time=1.2)
            self.wait_until(v, "inner")
            self.play(FadeIn(pause), run_time=0.5)
        a = VGroup(MathTex(r"\text{inner } 2x-1:\ \text{slope } 2,\ \ u=1", color=ORANGE, font_size=46),
                   MathTex(r"\text{outer } u^3:\ \text{slope } 3u^2=3", color=GREEN, font_size=46),
                   MathTex(r"3\times2=6", color=YELLOW, font_size=60)).arrange(DOWN, buff=0.35)
        a.next_to(q, DOWN, buff=0.5)
        with self.voice(N["prac2"]) as v:
            self.play(FadeOut(pause), run_time=0.4)
            self.wait_until(v, "inner")
            self.play(Write(a[0]), run_time=1)
            self.wait_until(v, "outer")
            self.play(Write(a[1]), run_time=1)
            self.wait_until(v, "six")
            self.play(Write(a[2]), run_time=0.6)
        self.clear_body()
        q2 = VGroup(MathTex(r"a=x^2,\qquad b=3a+1", color=BLUE, font_size=60),
                    MathTex(r"\frac{db}{dx}\ \text{ at } x=2 \;=\; ?", font_size=56)).arrange(DOWN, buff=0.3).shift(UP * 1.45)
        p2 = T("⏸  pause and try", 30, MUTED).next_to(q2, DOWN, buff=0.4)
        with self.voice(N["prac3"]) as v:
            self.wait_until(v, "Let")
            self.play(Write(q2[0]), run_time=1)
            self.wait_until(v, "slope")
            self.play(Write(q2[1]), run_time=0.8)
            self.wait_until(v, "Pause")
            self.play(FadeIn(p2), run_time=0.5)
        s = VGroup(MathTex(r"\frac{da}{dx}=2x=4,\qquad \frac{db}{da}=3", font_size=42),
                   MathTex(r"\frac{db}{dx}=3\times4=12", color=YELLOW, font_size=48),
                   MathTex(r"\text{check: } b=3x^2+1,\ 6x=12\ \checkmark", color=GREEN, font_size=44)
                   ).arrange(DOWN, buff=0.25).next_to(q2, DOWN, buff=0.3)
        with self.voice(N["prac4"]) as v:
            self.play(FadeOut(p2), run_time=0.4)
            self.wait_until(v, "four")
            self.play(Write(s[0]), run_time=1)
            self.wait_until(v, "Multiply")
            self.play(Write(s[1]), run_time=0.8)
            self.wait_until(v, "Check")
            self.play(Write(s[2]), run_time=1)


CODE = '''import numpy as np
def slope(fun, x, h=1e-6):                   # numeric derivative
    return (fun(x + h) - fun(x - h)) / (2 * h)
f = lambda x: 3 * x + 1                      # inner
g = lambda u: u ** 2                         # outer
print(2 * f(1.0) * 3, slope(lambda x: g(f(x)), 1.0))   # chain rule vs numeric
local = np.array([2.0, 4.0, 5.0])            # slopes of a=2x, b=a^2, c=5b at x=1
print(np.cumprod(local[::-1]))               # multiply from the end backward
J_F = np.array([[2.0, 0.0], [2.0, 1.0]])     # Jacobian of F at (1, 2)
J_G = np.array([[1.0, 4.0]])                 # Jacobian of G at F(1, 2)
print(J_G @ J_F)                             # chained Jacobians multiply'''


class C7_Code(Studio):
    def construct(self):
        self.heading("8 · Check it in Python")
        code = self.code_panel(CODE, font_size=24)
        code.set_x(0).set_y(-0.4)
        with self.voice(N["code1"]) as v:
            self.play(FadeIn(code), run_time=1)
            box = SurroundingRectangle(code.code_lines[1], color=YELLOW, buff=0.06)
            self.wait_until(v, "slope")
            self.play(Create(box), run_time=0.5)
            for i, w in [(5, "compare"), (7, "backward"), (10, "Jacobians")]:
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
            for i, w in [(0, "Eight"), (1, "numeric"), (3, "backward"), (6, "product")]:
                self.wait_until(v, w)
                self.play(Indicate(term[1][i], color=YELLOW), run_time=0.6)


class C8_Summary(Studio):
    def construct(self):
        self.heading("In summary", YELLOW)
        pts = self.bullets(["Composition: one function after another, g(f(x))",
                            "Chain rule: slopes multiply — outer slope (at the inner value) × inner slope",
                            "It comes from local linearity used twice",
                            "Longer chains: multiply all the local slopes",
                            "Many inputs/outputs: Jacobians, and they multiply as matrices",
                            "Backpropagation: the chain rule from the end backward"],
                           size=28).shift(UP * 0.5)
        fit(pts, 13)
        with self.voice(N["sum1"]) as v:
            for p in pts[:4]:
                self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.5)
                self.wait(v.duration / 6)
        nxt = VGroup(T("Next · Video 10", 26, MUTED),
                     T("exp & log — turning products into sums", 32, BLUE, weight=BOLD)
                     ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)
        with self.voice(N["sum2"]) as v:
            self.play(FadeIn(pts[4], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Backpropagation")
            self.play(FadeIn(pts[5], shift=RIGHT * 0.2), run_time=0.6)
            self.wait_until(v, "Next")
            self.play(FadeIn(nxt, shift=UP * 0.3), run_time=0.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)


class C9_Credits(Credits):
    pass


SCENES = ["C0_Intro", "C1_Composition", "C2_Derive", "C3_Example", "C4_Long",
          "C5_Jacobian", "C6_ML", "C7_Practice", "C7_Code", "C8_Summary", "C9_Credits"]
CHAPTER_TITLES = {"C0_Intro": "Intro", "C1_Composition": "Functions inside functions; gears",
                  "C2_Derive": "Why slopes multiply",
                  "C3_Example": "Worked example", "C4_Long": "Longer chains",
                  "C5_Jacobian": "The Jacobian", "C6_ML": "Backpropagation (preview)",
                  "C7_Practice": "Try it yourself", "C7_Code": "Python demo",
                  "C8_Summary": "Summary", "C9_Credits": "Credits"}
