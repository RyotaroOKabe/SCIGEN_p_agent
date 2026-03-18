# Command: Save Critical Prompt

Save a critical prompt (e.g., for NotebookLM, ChatGPT, or other AI tools) to `docs/prompt/`.

## When to Write

- When a carefully crafted prompt is used for NotebookLM slide decks, deep research, or paper drafting
- When a prompt captures a key question or design specification worth reusing
- When the user explicitly asks to save a prompt for later reference

## File Naming

```
docs/prompt/YYYY-MM-DD_<slug>.md
```

- Date = today (`date +%Y-%m-%d`)
- Slug = short snake_case topic, e.g.:
  - `notebooklm_masaki_feedback`
  - `chatgpt_dpo_loss_design`
  - `overleaf_writing_prompt`

## Template

```markdown
# <Title> — YYYY-MM-DD

## Tool
Which AI tool this prompt is for (NotebookLM, ChatGPT, Claude, etc.)

## Context
What documents/files are provided alongside this prompt.

## Prompt

<the full prompt text>

## Notes
Any additional context on how to use this prompt or what it produced.
```

## How to Write (Claude instructions)

1. Capture the full prompt text.
2. Note which tool it's for and what source documents accompany it.
3. Save to `docs/prompt/YYYY-MM-DD_<slug>.md`.
4. Stage and commit: `git add docs/prompt/ && git commit`.
