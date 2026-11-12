"""Text cleaning shared by training and the API.

Using the *same* function in both places matters: if the model is trained on
cleaned text but served raw text, predictions quietly get worse
("training/serving skew").
"""

from __future__ import annotations

import re
import unicodedata

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
TICKET_REF_RE = re.compile(r"#\s?\d{3,}")
LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")  # order numbers, phone numbers, card fragments
WHITESPACE_RE = re.compile(r"\s+")
REPEATED_PUNCT_RE = re.compile(r"([!?.])\1{1,}")


def clean_text(text: str) -> str:
    """Normalise a raw support ticket into model-ready text.

    - normalises unicode (curly quotes, odd spaces)
    - masks personal / noisy tokens: emails, URLs, ticket refs, long numbers
    - collapses repeated punctuation ("!!!" -> "!") and whitespace
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFKC", text)
    text = EMAIL_RE.sub(" [EMAIL] ", text)
    text = URL_RE.sub(" [URL] ", text)
    text = TICKET_REF_RE.sub(" [REF] ", text)
    text = LONG_NUMBER_RE.sub(" [NUM] ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()
