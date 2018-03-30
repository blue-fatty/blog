{{- $title := replace (replaceRE "(\\d{4}-\\d{2}-\\d{2}-?)?(.*)" "$2" .TranslationBaseName) "-" " " | title -}}
{{- $slug := index (split $title " ") 0 | lower -}}
---
title: "{{ $title }}"
date: {{ .Date }}
draft: true
slug: {{ $slug }}
---

