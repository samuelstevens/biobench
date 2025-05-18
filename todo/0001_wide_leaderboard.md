# 0001. Pivot leaderboard to wide format + bold best per task

Goal: Replace long table with wide grid: one row per model, one column per task, boldface the per-task maximum.
Why: Humans scan faster across models than down 1000-row lists; bolding highlights winners instantly.

Done when

* Elm view renders the new grid.
* Column order = ImageNet-1K, NeWT, then tasks in benchmark tasks sorted alphabetically.
* Bold values match BenchmarkBest.ties per task.

Start here:

`web/src/Leaderboard.elm :: viewTable`.
Use `Html.table` with `thead` from task order.

Avoid:

* Don’t transpose in JS; do it in Elm.
* Don’t fetch another file; reuse existing JSON.
