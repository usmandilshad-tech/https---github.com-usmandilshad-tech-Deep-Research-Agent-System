# report_writer.py
from jinja2 import Template

DEFAULT_MD_TEMPLATE = """---
title: "{{ title }}"
date: "{{ date }}"
author: "{{ author }}"
---

# {{ title }}

**Research window:** {{ window }}

## Executive Summary
{{ executive_summary }}

## Key Findings
{{ key_findings }}

## Analysis & Discussion
{{ analysis }}

## Limitations
{{ limitations }}

## Sources
{% for s in sources %}
- {{ s }}
{% endfor %}
"""

def render_markdown(payload: dict, template: str | None = None) -> str:
    tmpl = Template(template or DEFAULT_MD_TEMPLATE)
    return tmpl.render(**payload)
