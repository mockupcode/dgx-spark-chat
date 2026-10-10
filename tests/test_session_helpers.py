import numpy as np

from app.faces import SeenFace, best_match, describe
from app.prompts import system_prompt, thinking_filler
from app.session import build_user_content, trim_history


def test_text_only_content():
    content, kept = build_user_content("สวัสดี", "", None, reasoning=False)
    assert content == kept == "สวัสดี"


def test_image_goes_first_and_history_keeps_text_only():
    content, kept = build_user_content("นี่อะไร", "[คนที่จำได้ในภาพ: เอ]", b"\xff\xd8jpeg", reasoning=False)
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "[คนที่จำได้ในภาพ: เอ]\nนี่อะไร"}
    assert kept == "[คนที่จำได้ในภาพ: เอ]\nนี่อะไร"


def test_reasoning_instruction_goes_into_the_user_turn():
    content, _ = build_user_content("125 คูณ 48", "", None, reasoning=True)
    assert content.startswith("125 คูณ 48\n") and "คิด" in content


def test_system_prompt_is_stable_per_voice():
    assert system_prompt("f_teen_bright") == system_prompt("f_young_clear")
    assert "ค่ะ" in system_prompt("f_teen_bright")
    assert "ครับ" in system_prompt("m_young_clear")
    assert thinking_filler("m_mid_warm").endswith("ครับ")


def _history(n):
    out = []
    for i in range(n):
        out += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    return out


def test_history_is_untouched_below_the_limit():
    h = _history(5)
    assert trim_history(h, 24) is h


def test_history_is_trimmed_in_one_block_and_starts_with_user():
    h = _history(13)
    trimmed = trim_history(h, 24)
    assert len(trimmed) == 12
    assert trimmed[0]["role"] == "user"
    assert trimmed[-1] == h[-1]


def test_best_match_threshold():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    db = {"เอ": [a], "บี": [c]}
    assert best_match(b, db, 0.4)[0] == "เอ"
    assert best_match(np.array([0.0, 0.0, 1.0], dtype=np.float32), db, 0.4)[0] is None


def test_describe_faces():
    assert describe([]) == ""
    note = describe([SeenFace("เอ", 0.8, 100), SeenFace(None, 0.1, 50), SeenFace(None, 0.1, 40)])
    assert note == "(หมายเหตุจากระบบจดจำใบหน้า: เห็น คุณเอ และคนที่ยังไม่รู้จัก 2 คน)"
