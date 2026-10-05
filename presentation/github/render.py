#!/usr/bin/env python3
"""Render the artwork on the repository's GitHub README.

Reads the brand sources in src/ and the release facts in the root README.md (its
current release line and its RigMark table), and writes the SVGs in assets/.
GitHub shows a README image as an <img>, which cannot load web fonts, so every
word is drawn as outlines of the JSpark3 site's fonts: Geist and Geist Mono
(SIL Open Font License 1.1) and Sentient (Indian Type Foundry). The fonts are not
shipped in this repository; pass the directory that holds them.

    python3 presentation/github/render.py --fonts <dir>           # write assets/
    python3 presentation/github/render.py --fonts <dir> --check   # fail if assets/ differs
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import xml.dom.minidom
from pathlib import Path
from xml.sax.saxutils import quoteattr

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "src"
ASSETS = HERE / "assets"

# The fonts the artwork was drawn with, pinned so a re-render is byte-identical.
FONTS = {
    "sans": ("Geist-Medium.ttf", "edca18c9c39f33f9d7a66cf4b336117b543795afa1d1149383209ab4a23872c0"),
    "semi": ("Geist-SemiBold.ttf", "8a096dff8c92054201f22475447a0ec02800d3f5eb391f3c936b96af3df94851"),
    "mono": ("GeistMono-Medium.ttf", "d26dc0db22cd7d3e37c2fef00b48c1492c9cc73222dded178aea698f089e4f04"),
    "serif": ("Sentient-Bold.ttf", "bdd530f159b42f18f2066fc2f1a6a15bdfbbd29759164cfe733f60f3ff08157b"),
}
SOURCES = {
    "jspark3-mark.svg": "9837244666243476c054e7c275d24efc0106494e148896870cbb519e267bac41",
    "three-sparks.svg": "d0f4a1e0d1eeddd381633fd83eac333c86439408153356ff47c6c00ad8311937",
}

# The GLM release page's palette (jakeportfolio, app/(site)/jspark3/v2/glm.css):
# graphite and the Spark's chassis gold, with its paper colours for light surfaces.
BG, PANEL, TEXT, MUTED, LINE = "#141619", "#1c1f23", "#ecebe6", "#a9aba8", "#373b40"
GOLD, GOLD_INK, GOLD_DEEP = "#d4b87c", "#221c0f", "#80611d"
CARD, PAPER_INK, PAPER_MUTED, TRACK, BEFORE = "#f1f3f5", "#15171a", "#4d535a", "#dde1e5", "#9aa2ab"
# The same page recolours the illustration's fabric from Tempo olive to gold.
FABRIC = ("#0d0e10", "#8d7646", "#e3cc98")
TOKEN, GRID_DOT = "#f2dfb3", "#6b6f75"

THEMES = {
    "light": {"card": CARD, "ink": PAPER_INK, "muted": PAPER_MUTED, "track": TRACK,
              "before": BEFORE, "now": GOLD_DEEP},
    "dark": {"card": PANEL, "ink": TEXT, "muted": MUTED, "track": "#2c3036",
             "before": "#8b939c", "now": GOLD},
}

# The RigMark rows the README's headline charts show, in the README's own words.
# The figures themselves are read from the README's RigMark table.
CHARTS = [
    ("code", "Code, decode estimate", None),
    ("prose", "Prose, decode estimate", None),
    ("prefill", "Cold prefill, 64K prompt", None),
    ("four", "Four at once, end to end", "short code, end-to-end, 256-token cap per agent"),
]

# Badges: file stem, label, value (None for the current release), value colour, value ink. Release,
# hardware, engine, recipe license and draft model are the Hugging Face card's badges, in its colours.
BADGES = [
    ("release", "Release", None, "#0a7c3f", "#ffffff"),
    ("hardware", "Hardware", "3× DGX Spark", "#76b900", "#0f1a00"),
    ("engine", "Engine", "fork of TensorFold 0.3.6.2 (MIT)", "#8250df", "#ffffff"),
    ("api", "API", "OpenAI-compatible", "#4e6e8e", "#ffffff"),
    ("recipe-license", "Recipe license", "Apache-2.0", "#d73a49", "#ffffff"),
    ("base-weights", "Base weights", "MIT", GOLD, GOLD_INK),
    ("draft-model", "Draft model", "DFlash2 non-commercial", "#6e7781", "#ffffff"),
]
# Status tags for the README's release list, all one width so the versions after them line up.
STATUSES = [
    ("current", "Current", "#0a7c3f", "#ffffff"),
    ("rollback", "Rollback", GOLD, GOLD_INK),
    ("not-published", "Not published", "#6e7781", "#ffffff"),
    ("do-not-install", "Do not install", "#d73a49", "#ffffff"),
    ("historical", "Historical", "#4e6e8e", "#ffffff"),
]

# What the badges say, as the release's own guides say it.
BADGE_FACTS = (("README.md", "three NVIDIA DGX Sparks"), ("README.md", "a fork of TensorFold 0.3.6.2 (MIT)"),
               ("INSTALL.md", "The API is OpenAI-compatible"), ("README.md", "Recipe files are Apache-2.0"),
               ("README.md", "The base weights are MIT"), ("README.md", "CC BY-NC-ND 4.0 (non-commercial)"))


def num(value: float, places: int = 2) -> str:
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


class Face:
    """One font: glyph outlines in font units and GPOS pair kerning."""

    def __init__(self, path: Path, key: str) -> None:
        self.key = key
        self.font = TTFont(path)
        self.upm = self.font["head"].unitsPerEm
        self.cmap = self.font.getBestCmap()
        self.advance = {name: width for name, (width, _) in self.font["hmtx"].metrics.items()}
        self.glyphs = self.font.getGlyphSet()
        self.order = {name: index for index, name in enumerate(self.font.getGlyphOrder())}
        self.lookups = self._kerning()

    def _kerning(self) -> list[list]:
        lookups: list[list] = []
        if "GPOS" not in self.font:
            return lookups
        table = self.font["GPOS"].table
        indices = sorted({index for record in table.FeatureList.FeatureRecord
                          if record.FeatureTag == "kern" for index in record.Feature.LookupListIndex})
        for index in indices:
            lookup = table.LookupList.Lookup[index]
            subtables = []
            for sub in lookup.SubTable:
                if lookup.LookupType == 9:
                    if sub.ExtensionLookupType != 2:
                        continue
                    sub = sub.ExtSubTable
                elif lookup.LookupType != 2:
                    continue
                subtables.append((sub, set(sub.Coverage.glyphs)))
            lookups.append(subtables)
        return lookups

    def kern(self, left: str, right: str) -> int:
        total = 0
        for subtables in self.lookups:
            for sub, covered in subtables:
                if left not in covered:
                    continue
                if sub.Format == 1:
                    pairs = sub.PairSet[sub.Coverage.glyphs.index(left)].PairValueRecord
                    record = next((r for r in pairs if r.SecondGlyph == right), None)
                    if record is None:
                        continue
                    value = record.Value1
                else:
                    first = sub.ClassDef1.classDefs.get(left, 0)
                    second = sub.ClassDef2.classDefs.get(right, 0)
                    value = sub.Class1Record[first].Class2Record[second].Value1
                total += (getattr(value, "XAdvance", 0) or 0) if value else 0
                break
        return total

    def shape(self, text: str) -> list[str]:
        missing = [ch for ch in text if ord(ch) not in self.cmap]
        if missing:
            raise SystemExit(f"{self.key}: no glyph for {''.join(missing)!r}")
        return [self.cmap[ord(ch)] for ch in text]

    def layout(self, text: str, size: float, tracking: float) -> tuple[list[tuple[str, int]], float]:
        """Glyphs with their pen positions in font units, and the run's width in user units."""
        names = self.shape(text)
        track = round(tracking * self.upm / size)
        pen, placed = 0, []
        for index, name in enumerate(names):
            placed.append((name, pen))
            pen += self.advance[name]
            if index + 1 < len(names):
                pen += self.kern(name, names[index + 1]) + track
        return placed, pen * size / self.upm

    def outline(self, name: str) -> str:
        pen = SVGPathPen(self.glyphs)
        self.glyphs[name].draw(pen)
        return pen.getCommands()


class Svg:
    """One SVG file: shared glyph outlines in <defs>, then the drawing."""

    def __init__(self, width: float, height: float, label: str) -> None:
        self.width, self.height, self.label = width, height, label
        self.glyphs: dict[str, str] = {}
        self.defs: list[str] = []
        self.style = ""
        self.body: list[str] = []

    def add(self, markup: str) -> None:
        self.body.append(markup)

    def text(self, face: Face, text: str, x: float, y: float, size: float, fill: str,
             tracking: float = 0.0, anchor: str = "start") -> float:
        placed, width = face.layout(text, size, tracking)
        left = x - width / 2 if anchor == "middle" else x - width if anchor == "end" else x
        uses = []
        for name, pen in placed:
            ident = f"{face.key[0]}{face.order[name]}"
            if ident not in self.glyphs:
                self.glyphs[ident] = face.outline(name)
            if self.glyphs[ident]:
                uses.append(f'<use href="#{ident}" x="{pen}"/>' if pen else f'<use href="#{ident}"/>')
        scale = num(size / face.upm, 5)
        self.add(f'<g transform="translate({num(left)} {num(y)}) scale({scale} -{scale})" fill="{fill}">'
                 + "".join(uses) + "</g>")
        return width

    def render(self) -> str:
        defs = self.defs + [f'<path id="{ident}" d="{d}"/>' for ident, d in self.glyphs.items() if d]
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{num(self.width)}" height="{num(self.height)}" '
                f'viewBox="0 0 {num(self.width)} {num(self.height)}" role="img" aria-label={quoteattr(self.label)}>')
        style = f"<style>{self.style}</style>" if self.style else ""
        return head + style + "<defs>" + "".join(defs) + "</defs>" + "".join(self.body) + "</svg>\n"


def width_of(face: Face, text: str, size: float, tracking: float = 0.0) -> float:
    return face.layout(text, size, tracking)[1]


# ---------------------------------------------------------------- brand sources

def read_source(name: str) -> str:
    path = SRC / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCES[name]:
        raise SystemExit(f"src/{name}: sha256 {digest} differs from the pinned {SOURCES[name]}")
    return path.read_text(encoding="utf-8")


# The tumbling mark's frame 137 of 421, about 9.1 s into its 28 s turn: the triangle stands
# upright, apex over a level base. (Frame 0, where the animation starts, is seen almost edge-on.)
MARK_FRAME = 137


def mark(x: float, y: float, size: float) -> str:
    """The JSpark3 mark, held at MARK_FRAME of its own animation and nested at (x, y)."""
    doc = xml.dom.minidom.parseString(read_source("jspark3-mark.svg"))
    root = doc.documentElement
    for element in list(root.getElementsByTagName("line")) + list(root.getElementsByTagName("g")):
        moves = [child for child in element.childNodes
                 if child.nodeType == child.ELEMENT_NODE and child.tagName in ("animate", "animateTransform")]
        frame = {}
        for move in moves:
            value = move.getAttribute("values").split(";")[MARK_FRAME]
            frame[move.getAttribute("type") or move.getAttribute("attributeName")] = value
            element.removeChild(move)
        if element.tagName == "line":
            for name, value in frame.items():
                element.setAttribute(name, value)
        elif frame:
            element.setAttribute("transform", f"translate({frame['translate']}) scale({frame['scale']})")
    for gradient in root.getElementsByTagName("radialGradient"):
        gradient.setAttribute("id", "mark-shine")
    inner = "".join(child.toxml() for child in root.childNodes).replace("url(#g)", "url(#mark-shine)")
    inner = re.sub(r">\s+<", "><", inner).strip()
    view = root.getAttribute("viewBox")
    return f'<svg x="{num(x)}" y="{num(y)}" width="{num(size)}" height="{num(size)}" viewBox="{view}">{inner}</svg>'


def sparks(x: float, y: float, width: float, crop: tuple[float, float, float, float]) -> str:
    """The site's three-Spark illustration with the GLM page's gold fabric, nested at (x, y).

    The rank labels and captions inside the drawing are dropped, as the site hides them;
    the hero sets its own caption at a readable size.
    """
    doc = xml.dom.minidom.parseString(read_source("three-sparks.svg"))
    root = doc.documentElement
    for node in list(root.getElementsByTagName("title")) + list(root.getElementsByTagName("desc")):
        node.parentNode.removeChild(node)
    for node in list(root.getElementsByTagName("*")):
        if node.getAttribute("class") in ("jsv-svg-rank", "jsv-svg-cap") and node.parentNode:
            node.parentNode.removeChild(node)
    for pattern in root.getElementsByTagName("pattern"):
        if pattern.getAttribute("id") == "jsv-grid":
            for dot in pattern.getElementsByTagName("circle"):
                dot.setAttribute("fill", GRID_DOT)
    for group in root.getElementsByTagName("g"):
        if "jsv-fabric-link" not in group.getAttribute("class").split():
            continue
        for use, stroke in zip(group.getElementsByTagName("use"), FABRIC):
            use.setAttribute("stroke", stroke)
        for path in group.getElementsByTagName("path"):
            path.setAttribute("stroke", GOLD if "jsv-token-halo" in path.getAttribute("class") else TOKEN)
    inner = "".join(child.toxml() for child in root.childNodes)
    height = width * crop[3] / crop[2]
    return (f'<svg x="{num(x)}" y="{num(y)}" width="{num(width)}" height="{num(height)}" '
            f'viewBox="{" ".join(num(v) for v in crop)}">{inner}</svg>')


# The site's token motion along the three cables, stopped for reduced motion.
TOKEN_MOTION = (
    ".jsv-token{animation:flow 4.8s linear infinite;animation-delay:-1.2s}"
    ".jsv-fabric-link-02 .jsv-token{animation-duration:5.6s;animation-delay:-2.6s;animation-direction:reverse}"
    ".jsv-fabric-link-12 .jsv-token{animation-duration:6.4s;animation-delay:-.6s}"
    "@keyframes flow{from{stroke-dashoffset:0}to{stroke-dashoffset:-100}}"
    "@media (prefers-reduced-motion:reduce){.jsv-token{animation:none}}"
)
CROP = (30.0, 50.0, 660.0, 560.0)


# ---------------------------------------------------------------- README facts

def readme_facts() -> dict:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    release = re.search(r"\*\*Current release: JSpark3 (v\d+\.\d+\.\d+) \(GLM-5\.3 Flash\)\.\*\*", text)
    if not release:
        raise SystemExit("README.md: no 'Current release: JSpark3 vX.Y.Z (GLM-5.3 Flash).' line")
    section = text.split("\n## RigMark\n", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| ")]
    figures = {}
    for key, title, _ in CHARTS:
        row = next((r for r in rows if r.startswith(f"| {title} (tok/s, higher is better)")), None)
        if row is None:
            raise SystemExit(f"README.md RigMark table: no '{title} (tok/s, higher is better)' row")
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        figures[key] = (cells[1], cells[2])  # v1.8.4, then base weights + draft model
    for name, fact in BADGE_FACTS:
        if fact not in (ROOT / name).read_text(encoding="utf-8"):
            raise SystemExit(f"{name} no longer says {fact!r}; update the badges to match")
    return {"release": release.group(1), "figures": figures}


# ---------------------------------------------------------------- artwork

def hero(faces: dict, release: str, narrow: bool) -> str:
    sans, semi, mono, serif = faces["sans"], faces["semi"], faces["mono"], faces["serif"]
    label = (f"JSpark3 {release}: GLM-5.3 Flash on three DGX Sparks. "
             "Three DGX Sparks wired in a ring serve one endpoint.")
    if narrow:
        width, height, pad = 720, 1196, 48
        svg = Svg(width, height, label)
        svg.style = TOKEN_MOTION
        svg.add(f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="26" '
                f'fill="{BG}" stroke="{LINE}" stroke-width="2"/>')
        svg.add(mark(pad - 4, 46, 58))
        svg.text(semi, "JSPARK3", pad + 64, 90, 34, TEXT, tracking=-1.5)
        svg.text(mono, "JSPARK3 · Current release", pad, 186, 23, MUTED, tracking=0.6)
        svg.text(serif, release, pad - 6, 352, 176, GOLD, tracking=-8.8)
        svg.text(sans, "GLM-5.3 Flash", pad, 442, 48, TEXT, tracking=-1.0)
        svg.text(sans, "on three DGX Sparks.", pad, 502, 48, TEXT, tracking=-1.0)
        art = 640
        svg.add(sparks((width - art) / 2, 530, art, CROP))
        rule = 530 + art * CROP[3] / CROP[2] + 14
        svg.add(f'<path d="M{pad} {num(rule)}H{width - pad}" stroke="{LINE}" stroke-width="2"/>')
        svg.text(sans, "Three DGX Sparks · one endpoint", width / 2, rule + 44, 26, TEXT, anchor="middle")
        svg.text(sans, "One Spark answers requests; all three run the model.", width / 2, rule + 80, 21,
                 MUTED, anchor="middle")
        return svg.render()
    width, height, pad = 1280, 620, 64
    svg = Svg(width, height, label)
    svg.style = TOKEN_MOTION
    svg.add(f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="22" '
            f'fill="{BG}" stroke="{LINE}" stroke-width="2"/>')
    svg.add(mark(pad - 4, 50, 52))
    svg.text(semi, "JSPARK3", pad + 56, 88, 30, TEXT, tracking=-1.3)
    svg.text(mono, "JSPARK3 · Current release", pad, 196, 19, MUTED, tracking=0.5)
    svg.text(serif, release, pad - 6, 352, 156, GOLD, tracking=-7.8)
    svg.text(sans, "GLM-5.3 Flash", pad, 444, 42, TEXT, tracking=-0.9)
    svg.text(sans, "on three DGX Sparks.", pad, 498, 42, TEXT, tracking=-0.9)
    svg.text(mono, "jakejh.com/jspark3/glm", pad, 578, 16, MUTED, tracking=0.3)
    art, left = 600, 640
    svg.add(sparks(left, 22, art, CROP))
    rule = 22 + art * CROP[3] / CROP[2] + 10
    centre = left + art / 2
    svg.add(f'<path d="M{left + 24} {num(rule)}H{left + art - 24}" stroke="{LINE}" stroke-width="2"/>')
    svg.text(sans, "Three DGX Sparks · one endpoint", centre, rule + 32, 18, TEXT, anchor="middle")
    svg.text(sans, "One Spark answers requests; all three run the model.", centre, rule + 58, 15.5, MUTED,
             anchor="middle")
    return svg.render()


def badge(faces: dict, label: str, value: str, fill: str, ink: str, icon: bool) -> str:
    semi = faces["semi"]
    size, track, height, pad = 10, 0.7, 28, 8.5
    left_text, right_text = label.upper(), value.upper()
    icon_width = 16 + 6 if icon else 0
    left = pad + icon_width + width_of(semi, left_text, size, track) + pad
    right = pad + width_of(semi, right_text, size, track) + pad
    total = left + right
    svg = Svg(round(total), height, f"{label}: {value}")
    svg.defs.append(f'<clipPath id="round"><rect width="{num(total)}" height="{height}" rx="4"/></clipPath>')
    svg.add(f'<g clip-path="url(#round)"><rect width="{num(left)}" height="{height}" fill="#24272c"/>'
            f'<rect x="{num(left)}" width="{num(right)}" height="{height}" fill="{fill}"/></g>')
    if icon:
        svg.add(mark(pad - 2, 5, 18))
    baseline = 18
    svg.text(semi, left_text, pad + icon_width, baseline, size, "#ecebe6", tracking=track)
    svg.text(semi, right_text, left + pad, baseline, size, ink, tracking=track)
    return svg.render()


def status(faces: dict, label: str, fill: str, ink: str) -> str:
    semi = faces["semi"]
    size, track, height = 9.5, 0.7, 20
    width = max(width_of(semi, text.upper(), size, track) for _, text, _, _ in STATUSES) + 20
    svg = Svg(round(width), height, label)
    svg.add(f'<rect width="{round(width)}" height="{height}" rx="10" fill="{fill}"/>')
    svg.text(semi, label.upper(), round(width) / 2, 13.5, size, ink, tracking=track, anchor="middle")
    return svg.render()


def nice_ceiling(value: float) -> float:
    target = value * 1.05
    magnitude = 10 ** (len(str(int(target))) - 1)
    for step in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if step * magnitude >= target:
            return step * magnitude
    return 10 * magnitude


def chart(faces: dict, theme: str, title: str, subtitle: str | None, before: str, now: str) -> str:
    """One RigMark row as the GLM page draws it: v1.8.4 in grey, v2.0.1 in gold, from zero."""
    sans, mono = faces["sans"], faces["mono"]
    colours = THEMES[theme]
    width, height, pad = 404, 132, 18
    values = [float(before.replace(",", "")), float(now.replace(",", ""))]
    label = (f"{title}, tok/s, higher is better"
             + (f" ({subtitle})" if subtitle else "")
             + f": v1.8.4 {before}; v2.0.1 with base weights + draft model {now}.")
    svg = Svg(width, height, label)
    svg.add(f'<rect width="{width}" height="{height}" rx="10" fill="{colours["card"]}"/>')
    svg.text(sans, title, pad, 31, 15.5, colours["ink"], tracking=-0.2)
    svg.text(sans, "tok/s, higher is better", width - pad, 31, 12, colours["muted"], anchor="end")
    if subtitle:
        svg.text(sans, subtitle, pad, 50, 12, colours["muted"])
    ceiling = nice_ceiling(max(values))
    track_left, track_right = 112, 288
    span = track_right - track_left
    unit = width_of(mono, "tok/s", 11.5)
    for row, (name, shown, value, series) in enumerate((
            ("v1.8.4", before, values[0], colours["before"]),
            ("v2.0.1 base", now, values[1], colours["now"]))):
        middle = 82 + row * 29
        svg.text(mono, name, pad, middle + 4.5, 12.5, colours["ink"] if row else colours["muted"])
        svg.add(f'<rect x="{track_left}" y="{middle - 6}" width="{span}" height="12" fill="{colours["track"]}"/>')
        svg.add(f'<rect x="{track_left}" y="{middle - 6}" width="{num(span * value / ceiling)}" height="12" '
                f'fill="{series}"/>')
        svg.text(mono, "tok/s", width - pad, middle + 4.5, 11.5, colours["muted"], anchor="end")
        svg.text(mono, shown, width - pad - unit - 5, middle + 5, 15, colours["ink"], anchor="end")
    return svg.render()


def artwork(faces: dict) -> dict[str, str]:
    facts = readme_facts()
    still = Svg(44, 44, "JSpark3")
    still.add(mark(0, 0, 44))
    files = {
        "mark-still.svg": still.render(),
        "hero.svg": hero(faces, facts["release"], narrow=False),
        "hero-narrow.svg": hero(faces, facts["release"], narrow=True),
    }
    for stem, label, value, fill, ink in BADGES:
        files[f"badge-{stem}.svg"] = badge(faces, label, value or facts["release"], fill, ink, stem == "release")
    for stem, label, fill, ink in STATUSES:
        files[f"status-{stem}.svg"] = status(faces, label, fill, ink)
    for key, title, subtitle in CHARTS:
        before, now = facts["figures"][key]
        for theme in THEMES:
            files[f"rigmark-{key}-{theme}.svg"] = chart(faces, theme, title, subtitle, before, now)
    return files


def load_faces(directory: Path) -> dict:
    faces = {}
    for key, (name, pinned) in FONTS.items():
        path = directory / name
        if not path.is_file():
            raise SystemExit(f"missing font {path}; see presentation/README.md for where to get it")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != pinned:
            raise SystemExit(f"{name}: sha256 {digest} differs from the pinned {pinned}; "
                             "update FONTS only if you mean to change the artwork's fonts")
        faces[key] = Face(path, key)
    return faces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fonts", type=Path, required=True, help="directory holding the pinned font files")
    parser.add_argument("--check", action="store_true", help="compare with assets/ instead of writing it")
    args = parser.parse_args()
    files = artwork(load_faces(args.fonts))
    for name, content in files.items():
        xml.dom.minidom.parseString(content)  # every file must be well-formed XML
    if args.check:
        stale = sorted(name for name, content in files.items()
                       if not (ASSETS / name).is_file() or (ASSETS / name).read_text(encoding="utf-8") != content)
        extra = sorted(p.name for p in ASSETS.glob("*.svg") if p.name not in files)
        if stale or extra:
            print("STALE " + " ".join(stale) + (" EXTRA " + " ".join(extra) if extra else ""))
            return 1
        print(f"PASS {len(files)} README artwork files match")
        return 0
    ASSETS.mkdir(exist_ok=True)
    for name, content in files.items():
        (ASSETS / name).write_text(content, encoding="utf-8")
    print(f"wrote {len(files)} files, {sum(len(c.encode()) for c in files.values())} bytes, to {ASSETS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
