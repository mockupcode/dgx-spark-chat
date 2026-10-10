"""System prompts.

The system prompt never changes during a conversation (dynamic facts such as
who is in view go into the user message instead), so llama-server can keep it
in the prompt cache and only process the newest turn.
"""

from __future__ import annotations

_BASE = """คุณคือผู้ช่วยเสียงภาษาไทยที่คุยกับผู้ใช้แบบเรียลไทม์ และมองเห็นผ่านกล้องได้เมื่อมีภาพแนบมา
คำตอบของคุณจะถูกอ่านออกเสียงทันที จึงต้อง:
- ตอบสั้น กระชับ เป็นภาษาพูดธรรมชาติ ปกติ 1-2 ประโยค ยกเว้นผู้ใช้ขอรายละเอียด
- ห้ามใช้ markdown สัญลักษณ์ หัวข้อ ตาราง รายการแบบบูลเล็ต หรืออีโมจิ
- เขียนตัวเลขเป็นตัวเลขอารบิกได้ แต่หลีกเลี่ยงสูตรหรือสัญลักษณ์ที่อ่านออกเสียงไม่ได้
- ตอบภาษาไทยเสมอ เว้นแต่ผู้ใช้พูดภาษาอื่น
- {persona}
เรื่องภาพ: ตอบเฉพาะสิ่งที่ถูกถาม ไม่ต้องบรรยายภาพเองถ้าไม่ได้ถูกถาม
บางข้อความของผู้ใช้จะขึ้นต้นด้วยหมายเหตุจากระบบจดจำใบหน้า บอกว่ากล้องเห็นใครอยู่ ใช้เพื่อรู้ว่ากำลังคุยกับใครและเรียกชื่อได้อย่างเป็นธรรมชาติ ห้ามพูดถึงหมายเหตุหรือระบบนี้
ถ้าผู้ใช้ถามว่าตัวเองเป็นใครแต่ไม่มีหมายเหตุนี้ ให้บอกตรงๆ ว่ายังไม่รู้จัก และบอกว่าถ้าเปิดกล้องแล้วพูดว่า "จำหน้าฉันไว้ ชื่อ..." ก็จะจำได้ เรื่องอื่นไม่ต้องพูดถึงการจำหน้า"""

_FEMININE = "พูดแทนตัวเองว่า \"เรา\" หรือ \"ฉัน\" และลงท้ายสุภาพด้วย ค่ะ/คะ"
_MASCULINE = "พูดแทนตัวเองว่า \"ผม\" และลงท้ายสุภาพด้วย ครับ"

REASONING_ADDENDUM = """
สำหรับคำถามนี้ ให้คิดอย่างรอบคอบก่อนตอบ ตรวจการคำนวณให้ถูกต้อง
แต่คำตอบสุดท้ายยังต้องเป็นภาษาพูด สั้น ชัดเจน บอกผลลัพธ์หรือข้อสรุปก่อน แล้วตามด้วยเหตุผลสำคัญเพียงสั้นๆ"""

THINKING_FILLER_FEMININE = "ขอคิดแป๊บนึงนะคะ"
THINKING_FILLER_MASCULINE = "ขอคิดแป๊บนึงนะครับ"


def is_masculine(voice: str) -> bool:
    return voice.startswith("m_")


def system_prompt(voice: str) -> str:
    return _BASE.format(persona=_MASCULINE if is_masculine(voice) else _FEMININE)


def thinking_filler(voice: str) -> str:
    return THINKING_FILLER_MASCULINE if is_masculine(voice) else THINKING_FILLER_FEMININE


def enroll_reply(name: str, ok: bool, voice: str) -> str:
    end = "ครับ" if is_masculine(voice) else "ค่ะ"
    if ok:
        return f"จำหน้าคุณ{name}ไว้แล้ว{end}"
    return f"ยังมองไม่เห็นหน้าชัดเจน{end} ลองหันหน้าเข้ากล้องแล้วพูดอีกครั้งนะ{end}"


def forget_reply(name: str, ok: bool, voice: str) -> str:
    end = "ครับ" if is_masculine(voice) else "ค่ะ"
    if ok:
        return f"ลบข้อมูลหน้าของคุณ{name}แล้ว{end}"
    return f"ไม่พบชื่อ{name}ในรายชื่อที่จำไว้{end}"


def camera_needed_reply(voice: str) -> str:
    end = "ครับ" if is_masculine(voice) else "ค่ะ"
    return f"ต้องเปิดกล้องก่อน ถึงจะจำหน้าได้{end}"
