# 0008. Website Polish

## Branding & Metadata
- [ ] Favicon: 16x16, 32x32, 48x48, 180x180; `<link rel="icon">`
- [ ] Apple touch icon
- [x] Open Graph / Twitter / iMessage tags: `og:title`, `og:description`, `og:image`, `twitter:card`
- [x] `<title>` <= 60 chars, meta description ≤ 155 chars

## Visual Polish

- [ ] Tailwind color palette finalized
- [ ] ImageNet1K and Newt columns visible and styled
- [ ] Column highlight / hover consistent with design
- [ ] Rotated headers (90 deg) on screens < 640 px
- [x] Monospace, tabular-nums for numeric cells

## Responsiveness
- [ ] Test at 320 px, 375 px, 768 px, 1024 px, 1280 px
- [ ] Touch targets >= 44 px
- [x] `<meta name="viewport" content="width=device-width,initial-scale=1">`

## Accessibility
- [ ] Semantic table (`<thead>`, `<th scope="col">`)
- [ ] `aria-sort` on sortable headers
- [ ] Contrast ratio >=  4.5 : 1
- [ ] Keyboard navigation and visible focus ring
- [ ] `title="best score"` or similar on bolded cells for screen readers

## Performance
- [ ] `NODE_ENV=production`; Tailwind purge runs
- [ ] HTML, CSS, JS minified
- [x] Gzip / Brotli enabled
- [ ] Lazy-load heavy images
- [ ] Lighthouse performance >=  90

## SEO
- [ ] Canonical `<link>`
- [ ] `robots.txt` allows crawl
- [ ] `sitemap.xml` generated

## Security Headers
- [ ] HTTPS only, HSTS 1 y
- [ ] `Content-Security-Policy: default-src 'self'`
- [ ] `X-Content-Type-Options: nosniff`
- [ ] `X-Frame-Options: SAMEORIGIN`

## Build & Deploy
- [ ] CI/CD pipeline green
- [ ] Version/tag bumped
- [ ] Rollback procedure documented
- [ ] Custom 404 page
- [ ] Service-worker cache versioned

## Documentation
- [ ] CHANGELOG updated
- [ ] README quick-start accurate
- [ ] LICENSE present

## Post-Launch Smoke Test
- [ ] Share link in iMessage/Slack/Twitter; preview card correct
- [ ] Core pages load without console errors
- [ ] No 404/500 in server logs

