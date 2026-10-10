"""Decide whether a turn needs real reasoning.

Calculations, plans and comparisons go to the thinking path; everything else
stays on the fast path. A rule set costs nothing per turn, where asking the LLM
first would add a full round trip before the answer could even begin.
"""

from __future__ import annotations

import re

_THAI_NUM = r"(?:หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ|ยี่สิบ|ร้อย|พัน|หมื่น|แสน|ล้าน)"
_NUMBER = rf"(?:\d|{_THAI_NUM})"

_CALC_WORDS = re.compile(
    r"คำนวณ|คิดเลข|สมการ|โจทย์|เปอร์เซ็นต์|เปอเซ็น|ร้อยละ|ดอกเบี้ย|ผ่อนเดือนละ|ค่าเฉลี่ย|เฉลี่ย"
    r"|ยกกำลัง|รากที่|ส่วนลด|กำไร|ขาดทุน|อัตราแลกเปลี่ยน|แปลงหน่วย"
    r"|\bcalculate\b|\bcompute\b|\bsolve\b|\bequation\b|\bpercent(?:age)?\b|\binterest rate\b",
    re.IGNORECASE,
)
# Operator words only mean arithmetic when numbers are around ("คิดบวก" is not maths).
_OPERATOR = re.compile(r"บวก|ลบ|คูณ|หาร|[+\-*/×÷^=%]|\bplus\b|\bminus\b|\btimes\b|divided by", re.IGNORECASE)
_HOW_MUCH = re.compile(r"เท่าไหร่|เท่าไร|กี่บาท|กี่เท่า|กี่วัน|กี่ชั่วโมง|กี่นาที|กี่ปี|กี่เดือน|\bhow (?:much|many|long)\b", re.IGNORECASE)

_PLAN_WORDS = re.compile(
    r"วางแผน|แผน(?!ที่|ก)|จัดตาราง|ตารางเวลา|ทริป|กำหนดการ|โรดแมป|roadmap|itinerary|\bplan(?:ning)?\b|\bschedule\b",
    re.IGNORECASE,
)
_COMPARE_WORDS = re.compile(
    r"เปรียบเทียบ|เทียบ|ต่างกัน|แตกต่าง|ข้อดีข้อเสีย|ข้อดี.*ข้อเสีย|ดีกว่ากัน|อันไหนดีกว่า|ตัวไหนดีกว่า|ควรเลือก"
    r"|\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b|pros and cons|which is better",
    re.IGNORECASE,
)


def needs_reasoning(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if _CALC_WORDS.search(t) or _PLAN_WORDS.search(t) or _COMPARE_WORDS.search(t):
        return True
    has_number = re.search(_NUMBER, t) is not None
    if has_number and (_OPERATOR.search(t) or _HOW_MUCH.search(t)):
        return True
    return False
