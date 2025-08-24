PROMPT_TEMPLATES = {
    "headnote": (
        "SYSTEM: You are a careful legal analyst providing educational summaries, not legal advice."
        " Maintain a neutral tone, cite sources by doc_id, and avoid speculation.\n"
        "USER: Produce a concise legal summary with Facts, Issue, Holding, Reasoning.\n"
        "INPUT: {text}\n"
        "OUTPUT:"
    ),
    "summarization": (
        "SYSTEM: You are a careful legal analyst providing educational summaries, not legal advice."
        " Maintain a neutral tone, cite sources by doc_id, and avoid speculation.\n"
        "USER: Summarize the following legal text.\n"
        "INPUT: {text}\n"
        "OUTPUT:"
    ),
    "qa": (
        "SYSTEM: You are a careful legal analyst providing educational answers, not legal advice."
        " Maintain a neutral tone, cite sources by doc_id, and avoid speculation.\n"
        "USER: {question}\n"
        "INPUT: {text}\n"
        "OUTPUT:"
    ),
}


def build_prompt(text: str, style: str = "headnote", **kwargs) -> str:
    """Return a formatted prompt for ``text`` according to ``style``."""
    template = PROMPT_TEMPLATES.get(style)
    if template is None:
        raise KeyError(f"Unknown style: {style}")
    return template.format(text=text, **kwargs)
