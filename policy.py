import re
from typing import List

ADVICE_PATTERNS = [
    r"legal advice",
    r"should I",
    r"should we",
    r"can I",
    r"what should",
    r"am I allowed",
]

DISCLAIMER = (
    "This response is for educational purposes only and does not "
    "constitute legal advice."
)

def is_request_for_legal_advice(text: str) -> bool:
    """Return True if ``text`` looks like a request for legal advice."""
    t = text.lower()
    return any(re.search(p, t) for p in ADVICE_PATTERNS)

def policy_check(output: str, citations: List[str]) -> List[str]:
    """Return a list of policy flags for ``output`` and ``citations``."""
    flags: List[str] = []
    lowered = output.lower()
    if not citations:
        flags.append("missing_citation")
    if any(w in lowered for w in ["should", "must", "recommend", "advise"]):
        flags.append("advice_like_language")
    if any(w in lowered for w in ["maybe", "probably", "perhaps", "uncertain"]):
        flags.append("speculation")
    return flags
