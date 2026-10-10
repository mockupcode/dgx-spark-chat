from app.chunker import SpeechChunker, clean_for_speech


def stream(chunker: SpeechChunker, text: str, step: int = 3) -> list[str]:
    pieces = []
    for i in range(0, len(text), step):
        pieces += chunker.feed(text[i:i + step])
    return pieces + chunker.flush()


def test_first_piece_is_cut_early_at_a_thai_space():
    text = "สวัสดีค่ะ วันนี้อากาศดีมาก เหมาะกับการออกไปเดินเล่นที่สวนสาธารณะใกล้บ้าน แล้วก็แวะกินกาแฟด้วยกันนะคะ"
    pieces = stream(SpeechChunker(first_min=8, next_min=30), text)
    assert pieces[0] == "สวัสดีค่ะ"
    assert "".join(p.replace(" ", "") for p in pieces) == text.replace(" ", "")


def test_nothing_is_lost_or_duplicated():
    text = "หนึ่ง สอง สาม. สี่ ห้า หก! เจ็ด แปด เก้า สิบ? จบแล้วค่ะ"
    pieces = stream(SpeechChunker(first_min=5, next_min=10), text, step=1)
    assert " ".join(pieces).split() == text.split()


def test_decimal_point_is_not_a_sentence_end():
    chunker = SpeechChunker(first_min=50, next_min=50)
    assert chunker.feed("ราคา 3.") == []
    assert chunker.feed("5 บาท") == []
    assert chunker.flush() == ["ราคา 3.5 บาท"]


def test_word_still_streaming_is_not_split():
    chunker = SpeechChunker(first_min=5)
    assert chunker.feed("สวัสดีครั") == []
    assert chunker.feed("บ ") == ["สวัสดีครับ"]


def test_markdown_and_emoji_are_not_spoken():
    assert clean_for_speech("**สวัสดี** 😀 ดู [ที่นี่](https://example.com)") .split() == ["สวัสดี", "ดู", "ที่นี่"]


def test_bullets_are_stripped():
    assert clean_for_speech("- ข้อแรก\n- ข้อสอง").split() == ["ข้อแรก", "ข้อสอง"]


def test_unspeakable_pieces_are_dropped():
    chunker = SpeechChunker(first_min=2)
    assert chunker.feed("*** ") == []
    assert chunker.flush() == []


def test_very_long_text_without_spaces_is_forced_out():
    chunker = SpeechChunker(first_min=10, hard_max=50)
    pieces = chunker.feed("ก" * 120)
    assert pieces and all(len(p) <= 50 for p in pieces)


WORDS = ["เรา", "ยัง", "ไม่", "ทราบ", "ชื่อ", "ของ", "คุณ", "เลย", "ค่ะ"]


def fake_segment(text):
    out, i = [], 0
    while i < len(text):
        word = next((w for w in WORDS if text.startswith(w, i)), text[i])
        out.append(word)
        i += len(word)
    return out


def test_long_first_phrase_without_spaces_is_cut_at_a_word_boundary():
    chunker = SpeechChunker(segment=fake_segment)
    pieces = []
    for ch in "เรายังไม่ทราบชื่อของคุณเลยค่ะ ถ้าสะดวกบอกได้ไหมคะ":
        pieces += chunker.feed(ch)
    assert pieces[0] == "เรายังไม่ทราบ"


def test_word_cut_never_splits_the_word_still_arriving():
    chunker = SpeechChunker(segment=fake_segment, first_wait=10, first_word_cut=5)
    assert chunker.feed("เรายังไม่ทร") == ["เรายัง"]


def test_without_segmenter_the_first_phrase_waits_for_a_space():
    chunker = SpeechChunker()
    assert chunker.feed("เรายังไม่ทราบชื่อของคุณเลยค่ะ") == []
