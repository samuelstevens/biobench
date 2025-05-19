# 0005. Client-side sorting + filtering

Goal: Enable clicking column headers to sort; add text box to filter by model substring.

Why: Wide table grows quickly; user needs basic navigation.

Done when

* Sort icon toggles asc/desc/none.
* Filter input reduces visible rows in real time.

Start here

Introduce Model fields `sort` and `filter`; update `update` function.

Avoid 

* No external JS libraries.

