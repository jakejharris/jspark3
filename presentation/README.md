# README artwork

The images on this repository's front page: the hero, the badges, the RigMark charts and the closing mark. They are
presentation only. They change no recipe, weights, results, tag or release asset.

`github/render.py` draws `github/assets/` from the brand sources in `github/src/` and from the root `README.md`
itself: its current release line and its RigMark table. A chart therefore shows exactly the figures in that table,
and the hero and the release badge show the current release. Text is drawn as outlines, because GitHub shows a
README image as an `<img>`, which cannot load web fonts.

```bash
python3 presentation/github/render.py --fonts <dir>           # redraw github/assets/
python3 presentation/github/render.py --fonts <dir> --check   # fail if github/assets/ is out of date
```

Redraw after any change to the README's release line or RigMark table, then refresh `SHA256SUMS` from a clean
checkout with `python3 tools/validate_release.py . --landing --write-sums`. Check the page on github.com in the
light and dark themes and at phone width: the README picks the light or dark chart by theme, the narrow hero below
600 px, and the still mark for reduced motion.

`render.py` needs `fonttools` and these fonts in `<dir>`, pinned by SHA-256 in the script. The fonts are not part of
this repository.

| Files | Font | License | Source |
|---|---|---|---|
| `Geist-Medium.ttf`, `Geist-SemiBold.ttf`, `GeistMono-Medium.ttf` | Geist, Geist Mono | SIL Open Font License 1.1 | [vercel/geist-font](https://github.com/vercel/geist-font) |
| `Sentient-Bold.ttf` | Sentient | ITF Free Font License | [Fontshare](https://www.fontshare.com/fonts/sentient) |

## Sources

[`github/src/SOURCE.json`](github/src/SOURCE.json) records where each source came from and its SHA-256:

- `jspark3-mark.svg`: the animated JSpark3 mark, byte for byte the pinned asset that the JSpark3 site and the Tempo
  card also carry. The README closes with it. The hero and the release badge hold it still at one frame of its own
  animation.
- `jspark3-mark-static.svg`: the site's reduced-motion copy of the mark.
- `three-sparks.svg`: the site's three-Spark illustration. `render.py` gives it the GLM release page's gold fabric
  and token motion; the motion stops for reduced motion.

The palette is the GLM release page's graphite and chassis gold, with its paper colours for the light charts.

## Credits

Geist and Geist Mono are by Vercel, in collaboration with basement.studio, under the SIL Open Font License 1.1. The
release number in the hero is set in Sentient, a trademark of the Indian Type Foundry; Sentient is copyright
2019-2021 Indian Type Foundry, designed by Noopur Choksi and Barbara Bigosinska. The mark, the three-Spark
illustration and the palette come from the JSpark3 site.
