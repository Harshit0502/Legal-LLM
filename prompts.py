PROMPT_TEMPLATES = {
    "headnote": (
        "SYSTEM: You are a careful legal analyst. Summarize faithfully without hallucinating.\n"
        "USER: Produce a concise legal summary with Facts, Issue, Holding, Reasoning.\n"
        "INPUT: {text}\n"
        "OUTPUT:"
    ),
    "summarization": (
        "SYSTEM: You are a careful legal analyst. Summarize the document accurately and concisely.\n"
        "USER: Summarize the following legal text.\n"
        "INPUT: {text}\n"
        "OUTPUT:"
    ),
    "qa": (
        "SYSTEM: You are a careful legal analyst. Answer the question using only the provided document.\n"
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
