"""Shared pieces for the ML Math Atoms series. Import, do not edit per video.

Usage in scene.py (after `from studio import *`):
    sys.path.insert(0, str(HERE.parent))
    from common import CREDIT_TEXT, Credits
    ...
    class C9_Credits(Credits): pass        # always the LAST scene
    SCENES = [..., "C9_Credits"];  CHAPTER_TITLES["C9_Credits"] = "Credits"
"""
from studio import *  # noqa: F403

# Spoken text (the TTS reads the site name clearly this way). Shown on screen as the real URL.
CREDIT_TEXT = """This video was made by David R. You can find more lessons at
David Ramirez J dot com. Thanks for watching."""


class Credits(Studio):
    def construct(self):
        made = T("Made by", 30, MUTED)
        name = T("David R.", 64, FG, weight=BOLD)
        url = T("davidramirezj.com", 40, BLUE, font=MONO)
        series = T("ML Math Atoms", 24, MUTED)
        g = VGroup(made, name, url).arrange(DOWN, buff=0.35).shift(UP * 0.3)
        line = Line(LEFT * 3, RIGHT * 3, color=YELLOW, stroke_width=3).next_to(g, DOWN, buff=0.5)
        series.next_to(line, DOWN, buff=0.35)
        with self.voice(CREDIT_TEXT) as v:
            self.play(FadeIn(made, shift=UP * 0.2), Write(name), run_time=1.2)
            self.wait_until(v, "lessons")
            self.play(FadeIn(url, shift=UP * 0.2), Create(line), FadeIn(series), run_time=1.0)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
