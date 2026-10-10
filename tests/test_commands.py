import pytest

from app.commands import FaceCommand, parse_face_command


@pytest.mark.parametrize("text, name", [
    ("จำหน้าฉันไว้ ชื่อสมชาย", "สมชาย"),
    ("จำหน้าผมไว้ด้วยนะ ชื่อ ต้น ครับ", "ต้น"),
    ("จำหน้าหนูไว้ว่า มายด์ ค่ะ", "มายด์"),
    ("ฉันชื่อน้ำฝน จำหน้าไว้ด้วยนะ", "น้ำฝน"),
    ("remember my face as Alex", "Alex"),
    ("Remember me as Bee please", "Bee"),
])
def test_enroll(text, name):
    assert parse_face_command(text) == FaceCommand("enroll", name)


@pytest.mark.parametrize("text, name", [
    ("ลืมหน้าสมชาย", "สมชาย"),
    ("ลบหน้าของคุณต้นออกด้วยครับ", "ต้น"),
    ("ช่วยลบหน้าสมชายออก", "สมชาย"),
    ("forget Alex", "Alex"),
])
def test_forget(text, name):
    assert parse_face_command(text) == FaceCommand("forget", name)


@pytest.mark.parametrize("text", [
    "สวัสดีค่ะ",
    "จำได้ไหมว่าฉันเป็นใคร",
    "ฉันชื่ออะไร",
    "ช่วยลบข้อความนี้",
    "จำหน้าฉันได้ไหม",
    "อย่าลืมหน้าที่ของตัวเองนะ",
    "don't forget to buy milk",
    "forget it",
])
def test_not_a_command(text):
    assert parse_face_command(text) is None
