# 0003. Embed interactive plots with elm-charts

Goal: Render a per-task bar histogram grouped by backbone family.

Why: Let visitors understand the "ImageNet cliff" and model family spread without reading the paper.

Done when:

* Charts appear under "Interactive results"
* Hover tooltips show precise values.
* Zero JS errors in browser console.

Start here:

Use `elm-charts` (`elm install terezka/elm-charts`, https://www.elm-charts.org/).

Avoid:

* Don’t embed matplotlib PNGs; must be vector/interactive.
* No network calls; charts reuse initial JSON.
