"""Spoken face commands: "remember my face as ...", "forget ...".

These are answered directly, without the LLM, so enrolment is instant and
never depends on the model following instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARTICLES = (
    "นะครับ", "นะคะ", "นะค่ะ", "ครับผม", "ครับ", "ค่ะ", "คะ", "นะ", "จ้า", "จ้ะ", "จ๊ะ",
    "หน่อย", "ด้วย", "เลย", "ที", "ไว้", "ออก", "please",
)
_NAME_STOP = re.compile(r"\s+(?:และ|แล้ว|ด้วย|ไว้|นะ|ครับ|ค่ะ|คะ|จ้า)(?:\s|$)")

_ENROLL = [
    re.compile(r"จำหน้า(?:ฉัน|ผม|หนู|เรา|ดิฉัน|กู)?(?:ไว้)?(?:ด้วย|หน่อย)?(?:นะ)?\s*(?:ว่า|ว่าชื่อ|ชื่อ|เป็น)\s*(?P<name>.+)"),
    re.compile(r"(?:ฉัน|ผม|หนู|เรา|ดิฉัน)ชื่อ\s*(?P<name>.+?)\s*(?:นะ)?\s*จำหน้า"),
    re.compile(r"remember (?:my face|me) as\s+(?P<name>.+)", re.IGNORECASE),
]
_FORGET = [
    # Anchored at the start so "อย่าลืมหน้าที่..." (don't forget your duty) is not a command.
    re.compile(r"^(?:ช่วย|ให้)?\s*(?:ลืม|ลบ)(?:ใบ)?หน้า(?!ที่|ต่าง|จอ)(?:ของ)?\s*(?:คุณ)?\s*(?P<name>.+)"),
    re.compile(r"^forget (?:the face of )?(?P<name>(?!it\b|that\b|this\b|about\b)[^\s].{0,39})$", re.IGNORECASE),
]


@dataclass(frozen=True)
class FaceCommand:
    action: str  # "enroll" | "forget"
    name: str


def parse_face_command(text: str) -> FaceCommand | None:
    t = text.strip()
    for pattern in _ENROLL:
        m = pattern.search(t)
        if m and (name := clean_name(m.group("name"))):
            return FaceCommand("enroll", name)
    for pattern in _FORGET:
        m = pattern.search(t)
        if m and (name := clean_name(m.group("name"))):
            return FaceCommand("forget", name)
    return None


def clean_name(raw: str) -> str:
    name = raw.strip()
    name = _NAME_STOP.split(name, maxsplit=1)[0]
    name = re.sub(r"[\"'“”‘’.,!?…]", "", name).strip()
    changed = True
    while changed and name:
        changed = False
        for particle in _PARTICLES:
            if name.lower().endswith(particle):
                name = name[: -len(particle)].strip()
                changed = True
    for prefix in ("คุณ", "ชื่อ"):
        if name.startswith(prefix) and len(name) > len(prefix):
            name = name[len(prefix):].strip()
    return name[:40]
