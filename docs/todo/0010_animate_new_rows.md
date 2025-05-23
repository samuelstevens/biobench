# 0010. Animate New Table Rows

**Context**
BioBench table rows appear/disappear when the user toggles `selectedFamilies`. Newly re-added rows should flash for \~1 s to draw attention without disturbing layout or performance.

# 1 Goals

1. When a row transitions from *filtered-out* → *visible*, apply a background flash that fades to transparent over \~1000 ms.
2. Flash must trigger **only** for genuinely new rows, not for rows merely moved by resorting.
3. No noticeable regressions in scroll or sort performance (keep `Html.Keyed`).
4. Solution must remain declarative Elm—no direct DOM or JS interop.

# 2 Non-Goals

* No animation for rows removed from view.
* No support for per-cell flashes.
* No persistent animation queue (simple fire-and-forget fade).
