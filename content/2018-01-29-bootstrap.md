---
title: "Bootstrap"
date: 2018-01-29T22:19:05+08:00
slug: bootstrap
---

## Quick notes

- `.d-none`, hide elements
    - `.d-{sm,md,lg,xl}-none`
    - Tips: use `d-none` first, and specify screen display mode to cover it.
- `container-fluid`, full width container

---

### Responsive breakpoints

[Github issue: Bootstrap default breakpoints](https://github.com/twbs/bootstrap/issues/14894)

Viewport dimensions.

- `xs`, extra small
    - \\( xs \le 575.98px \\)
- `sm`, small
    - \\( 576px \le sm \le 767.98px \\)
- `md`, medium
    - \\( 768px \le md \le 991.98px \\)
- `lg`, large
    - \\( 992px \le lg \le 1199.98px \\)
- `xl`, extra large
    - \\( xl \ge 1200px  \\)

Take control over the viewport:

``` html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```
<!--more-->
