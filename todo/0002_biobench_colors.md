# 0002. Apply BioBench color palette

Goal: Use the project's brand colors on the website.
Why: Visual identity and making it more fun!

Done when:

* css extended with `biobench-*` colors from `biobench/reporting.py`.
* There is at least some color in the webpage.

Start here:

Extract RGB hex codes from `reporting.py`.
Add to main.css via `@theme` directive:
Apply via `class` in Elm.

Avoid:

* No inline styles.
* No external CSS frameworks—Tailwind only.

References:

From https://tailwindcss.com/docs/theme:
```
@theme {
  --color-mint-500: oklch(0.72 0.11 178);
}
```
> Now you can use utility classes like bg-mint-500, text-mint-500, or fill-mint-500 in your HTML:

