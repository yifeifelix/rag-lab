"""Conservative query/egress privacy filters; source documents are never rewritten."""
import re

_PAN = re.compile(r'(?<!\w)\d(?:[ -]?\d){12,18}(?!\w)')
_OTHER_PII = [
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[REDACTED_SSN]'),
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'), '[REDACTED_EMAIL]'),
    (re.compile(r'\bBearer\s+[A-Za-z0-9\-._~+/]+=*', re.I), 'Bearer [REDACTED_TOKEN]'),
]
_INJECTION = [re.compile(pattern, re.I) for pattern in (
    r'ignore\s+(?:all\s+)?(?:previous\s+)?instructions',
    r'disregard\s+(?:all\s+)?(?:system|previous)\s+(?:rules|instructions)',
    r'override\s+(?:all\s+)?previous\s+instructions',
    r'(?:reveal|dump)\s+(?:your\s+)?(?:hidden\s+)?system\s+prompt',
    r'exfiltrate\s+(?:secrets|keys|credentials)',
    r'print\s+secret\s+(?:kms|api)\s+(?:credentials|keys)',
    r'<\|(?:system|im_start|endoftext)\|>',
    r'\[INST\]|<<SYS>>',
)]
_CITATION = re.compile(r'\[([A-Za-z0-9][A-Za-z0-9_.-]{0,79})\]')


def luhn_valid(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit() and c.isascii()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, bool]:
    original = text
    def redact_pan(match):
        # An international telephone number is not a card number.
        if match.start() and text[match.start() - 1] == '+':
            return match.group()
        return '[REDACTED_PAN]' if luhn_valid(match.group()) else match.group()
    text = _PAN.sub(redact_pan, text)
    for pattern, replacement in _OTHER_PII:
        text = pattern.sub(replacement, text)
    return text, text != original


def detects_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION)


def citation_ids(text: str) -> list[str]:
    return list(dict.fromkeys(_CITATION.findall(text)))


def strip_known_citations(text: str, known_ids: set[str]) -> str:
    return _CITATION.sub(lambda match: '' if match[1] in known_ids else match[0], text)
