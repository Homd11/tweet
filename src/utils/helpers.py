import re
import hashlib
from typing import List, Optional
from datetime import datetime
import emoji as emoji_lib


def sanitize_text(text: str) -> str:
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"http\S+", "[URL]", text)
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
    return text


def get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def extract_hashtags(text: str) -> List[str]:
    return re.findall(r"#(\w+)", text)


def extract_mentions(text: str) -> List[str]:
    return re.findall(r"@(\w+)", text)


def is_arabic_text(text: str) -> bool:
    arabic_pattern = re.compile(r"[\u0600-\u06FF]")
    return bool(arabic_pattern.search(text))


def get_text_length_category(length: int) -> str:
    if length < 50:
        return "short"
    elif length < 120:
        return "medium"
    else:
        return "long"


def format_timestamp(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = datetime.utcnow()
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def demojize_text(text: str) -> str:
    return emoji_lib.demojize(text, delimiters=(" ", " "))