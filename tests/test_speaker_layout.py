import base64
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from desktop_pet.vision.ocr import OCRLine, _result_from_lines
from desktop_pet.vision.speaker_layout import SpeakerLayout, load_profiles, save_profile


FRAME = Image.new('RGB', (1000, 600), '#234452')
CROP = (0, .5, 1, .95)


def reading(name, turn, x=.12):
    return _result_from_lines([
        OCRLine(name, .95, (x, .15, x+.04, .22)),
        OCRLine(f'这是第{turn}次完整对白，包含足够的文字。', .99, (x, .30, x+.55, .43)),
    ])


def observe(learner, result, title='游戏甲', frame=FRAME):
    return learner.observe(result, frame, CROP, title=title, process='engine.exe')


def learn(learner, title='游戏甲', x=.12):
    for i, name in enumerate(['小明', '小红', '小明']):
        raw = reading(name, i, x)
        result, info = observe(learner, raw, title)
        assert result is raw  # No inferred name edits before confirmation.
    return info


def test_candidate_requires_distinct_dialogue_and_changed_names():
    with TemporaryDirectory() as tmp:
        learner = SpeakerLayout(Path(tmp)/'profiles.json')
        for _ in range(6):
            assert observe(learner, reading('小明', 1))[1]['status'] == 'learning'
        for i in range(6):
            assert observe(learner, reading('小明', i))[1]['status'] == 'learning'
        assert observe(learner, reading('小红', 7))[1]['status'] == 'candidate'
        assert not (Path(tmp)/'profiles.json').exists()


def test_confirmed_region_survives_restart_and_scaling_without_stale_speaker():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'profiles.json'
        candidate = learn(SpeakerLayout(path))
        assert candidate['status'] == 'candidate'
        assert base64.b64decode(candidate['preview']).startswith(b'\x89PNG')
        save_profile(candidate, path=path)
        learner = SpeakerLayout(path)
        result, info = observe(learner, reading('小红', 8), frame=Image.new('RGB', (2000,1200)))
        assert info['status'] == 'confirmed' and info['speaker'] == '小红'
        assert result.lines[0].text == '小红'
        absent = _result_from_lines([reading('小红', 9).lines[1]])
        assert observe(learner, absent)[1]['speaker'] == ''


def test_shared_engine_different_games_never_share_name_boxes():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'profiles.json'
        learner = SpeakerLayout(path)
        a = learn(learner)
        save_profile(a, path=path)
        _, info = observe(learner, reading('小红', 10), title='游戏乙')
        assert info['key'] != a['key'] and info['status'] == 'learning'
        b = learn(learner, '游戏乙', x=.35)
        save_profile(b, path=path)
        assert len(load_profiles(path)) == 2
        save_profile(b, remove=True, path=path)
        assert set(load_profiles(path)) == {a['key']}


def test_changed_layout_requires_reconfirmation():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'profiles.json'
        learner = SpeakerLayout(path)
        save_profile(learn(learner), path=path)
        for i, name in enumerate(['小红','小明','小红'], 10):
            _, info = observe(learner, reading(name,i,x=.45))
            assert info['speaker'] == ''
        assert info['status'] == 'candidate' and info['box'][0] > .4
        _, info = observe(learner, reading('小明', 14), frame=Image.new('RGB',(1000,1000)))
        assert info['status'] == 'layout_changed' and info['speaker'] == ''


def test_background_names_without_dialogue_never_propose():
    with TemporaryDirectory() as tmp:
        learner = SpeakerLayout(Path(tmp)/'profiles.json')
        for name in ['小红', '小明', '小红', '小明']:
            result = _result_from_lines([OCRLine(name, .99, (.12,.15,.16,.22))])
            assert observe(learner, result)[1]['status'] == 'learning'


def test_dialogue_in_wrong_lane_does_not_validate_a_short_label():
    with TemporaryDirectory() as tmp:
        learner = SpeakerLayout(Path(tmp)/'profiles.json')
        for i, name in enumerate(['HP', 'MP', 'HP', 'MP']):
            result = _result_from_lines([OCRLine(name,.99,(.8,.05,.85,.10)), reading('小明',i).lines[1]])
            assert observe(learner,result)[1]['status'] == 'learning'


def test_wider_name_still_matches_left_aligned_saved_region():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'profiles.json'
        learner = SpeakerLayout(path)
        save_profile(learn(learner), path=path)
        result = _result_from_lines([OCRLine('Alexander',.95,(.12,.15,.30,.22)),reading('小明',20).lines[1]])
        assert observe(learner,result)[1]['speaker'] == 'Alexander'


def test_name_beside_short_dialogue_is_also_a_candidate():
    with TemporaryDirectory() as tmp:
        learner = SpeakerLayout(Path(tmp)/'profiles.json')
        for name, text in [('七七','「你好吗？」'), ('姬织','「唔，唔嗯」'), ('七七','「因为……」')]:
            result = _result_from_lines([OCRLine(name,.95,(.11,.62,.15,.74)),
                                        OCRLine(text,.9,(.21,.63,.34,.74))])
            _, info = observe(learner,result)
        assert info['status'] == 'candidate'
        assert set(info['examples']) == {'七七','姬织'}


def test_low_confidence_candidate_does_not_change_text_or_save_without_confirmation():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'profiles.json'
        learner = SpeakerLayout(path)
        for i, name in enumerate(['萤','燐','萤']):
            lines = list(reading(name,i).lines)
            lines[0] = OCRLine(name,.5,lines[0].box)
            raw = _result_from_lines(lines)
            result, info = observe(learner,raw)
            assert result is raw
        assert info['status'] == 'candidate' and not path.exists()


def test_confirmed_name_does_not_require_recognized_dialogue():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'profiles.json'
        save_profile(learn(SpeakerLayout(path)), path=path)
        learner = SpeakerLayout(path)
        name = reading('小红', 20).lines[0]
        for body in ['.......', '…', '']:
            lines = [name]
            if body:
                lines.append(OCRLine(body, .9, (.12, .30, .55, .43)))
            result, info = observe(learner, _result_from_lines(lines))
            assert info['speaker'] == '小红'
            assert result.lines[0] == name
        assert observe(learner, _result_from_lines([]))[1]['speaker'] == ''


def test_confirmed_name_rejects_ambiguous_and_outside_readings():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'profiles.json'
        save_profile(learn(SpeakerLayout(path)), path=path)
        learner = SpeakerLayout(path)
        duplicate = [reading(name, 20).lines[0] for name in ['小红', '小明']]
        assert observe(learner, _result_from_lines(duplicate))[1]['speaker'] == ''
        outside = _result_from_lines([reading('小红', 21, x=.6).lines[0]])
        assert observe(learner, outside)[1]['speaker'] == ''


def test_ui_chrome_never_teaches_or_matches_a_name_box():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'profiles.json'
        learner = SpeakerLayout(path)
        for i, label in enumerate(['Auto', 'SAVE', 'LOAD', 'Auto']):
            raw = reading(label, i)
            result, info = observe(learner, raw)
            assert info['status'] == 'learning'
            assert result.lines == (raw.lines[1],)
        save_profile(learn(learner), path=path)
        learner = SpeakerLayout(path)
        for i, label in enumerate(['Auto', 'SAVE', 'LOAD']):
            result, info = observe(learner, reading(label, i+20))
            assert info['speaker'] == ''
            assert label not in result.text
        assert all(text not in {'Auto', 'SAVE', 'LOAD'}
                   for _, items in learner.samples for text, _ in items)


def test_ui_chrome_filter_keeps_real_words_and_current_name():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'profiles.json'
        learner = SpeakerLayout(path)
        save_profile(learn(learner), path=path)
        body = OCRLine('他走进了 DIALOGUE 场景。 SAVELOADCONFIG', .99, (.12,.30,.7,.43))
        name = reading('Alexander', 20).lines[0]
        result, info = observe(learner, _result_from_lines([body, name]))
        assert info['speaker'] == 'Alexander'
        assert result.lines[0] == name
        assert result.lines[1].text == '他走进了 DIALOGUE 场景。'
        assert result.lines[1].box == body.box


def test_bracketed_single_character_names_survive_filter_and_learn():
    from desktop_pet.vision.ocr import filter_visual_novel_ocr_result
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'profiles.json'
        learner = SpeakerLayout(path)
        tracking = {}
        for i, name in enumerate(['【麦】','【肇】','【麦】','【肇】','【麦】','【肇】']):
            raw = reading(name,i)
            filtered, _ = filter_visual_novel_ocr_result(raw,tracking)
            result, info = observe(learner,filtered)
            assert result.lines[0].text == name
        assert info['status'] == 'candidate'
        assert set(info['examples']) == {'麦','肇'}
        save_profile(info,path=path)
        only_name = _result_from_lines([reading('【麦】',20).lines[0]])
        assert observe(SpeakerLayout(path),only_name)[1]['speaker'] == '麦'
