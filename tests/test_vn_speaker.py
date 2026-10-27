from pathlib import Path
from PIL import Image
from desktop_pet.vision.ocr import OCRLine, _result_from_lines
from desktop_pet.vision.vn_speaker import NAME_BOX, correct_camus_speaker


def _frame(fixture, width=1400):
    frame = Image.new('RGB', (width, round(width * 9 / 16)), '#234452')
    box = tuple(round(v * s) for v, s in zip(NAME_BOX, (frame.width, frame.height, frame.width, frame.height)))
    with Image.open(Path(__file__).parent / 'fixtures/camus_names' / fixture) as crop:
        frame.paste(crop.resize((box[2]-box[0], box[3]-box[1])), box[:2])
    return frame


def _result():
    return _result_from_lines([OCRLine('焕', .9, (.218, .50, .24, .60)),
                              OCRLine('「焕这个字不应该在正文中被替换。」', .99, (.2, .7, .9, .8))])


def _correct(frame, result=None, **kwargs):
    options = dict(title='青空下的加缪', process_name='青い空のカミュ.exe', text_ratio=.58, bottom_margin=.07)
    options.update(kwargs)
    return correct_camus_speaker(result or _result(), frame, **options)


def test_rin_name_matches_held_out_backgrounds_and_keeps_body():
    for fixture in ['8.png', '14.png', 'large_rin.png']:
        corrected, note = _correct(_frame(fixture))
        assert corrected.lines[0].text == '燐'
        assert corrected.lines[1].text == _result().lines[1].text
        assert 'raw_name=焕' in note


def test_name_box_scales_with_window():
    for width in [1400, 2560, 3840]:
        assert _correct(_frame('8.png', width))[0].lines[0].text == '燐'


def test_other_character_and_blank_box_are_unchanged():
    for frame in [_frame('15.png'), _frame('large_other.png'), Image.new('RGB', (1400, 788), '#234452')]:
        raw = _result()
        result, note = _correct(frame, raw)
        assert result is raw and note == ''


def test_unverified_game_layout_and_crop_are_unchanged():
    for options in [dict(title='其他游戏'), dict(process_name='other.exe'), dict(text_ratio=.85)]:
        raw = _result()
        assert _correct(_frame('8.png'), raw, **options)[0] is raw
    raw = _result()
    assert _correct(Image.new('RGB', (1000, 1000)), raw)[0] is raw


def test_matching_name_without_dialogue_does_not_create_a_line():
    raw = _result_from_lines([OCRLine('焕', .9, (.218, .50, .24, .60))])
    assert _correct(_frame('8.png'), raw)[0] is raw


def test_missing_name_can_be_recovered_without_touching_dialogue():
    raw = _result_from_lines([_result().lines[1]])
    result, note = _correct(_frame('8.png'), raw)
    assert result.lines[0].text == '燐'
    assert result.lines[1] == raw.lines[0]
    assert 'raw_name=<empty>' in note
