from pathlib import Path
from tempfile import TemporaryDirectory

from desktop_pet.llm.visual_novel import VisualNovelTracker
from desktop_pet.llm.dialog_manager import DialogManager


def test_old_dialogue_survives_cache_trim_and_restart():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / 'story.json'
        tracker = VisualNovelTracker(path, max_lines=60)
        tracker.observe('阿白初次在车站遇见小林。')
        for i in range(70):
            tracker._is_duplicate = lambda text: False
            tracker.observe(f'后来在第{i}座城镇购买不同的物品。')
        tracker.flush()
        assert all('车站' not in row['text'] for row in tracker._lines)
        tracker = VisualNovelTracker(path, max_lines=60,
            similarities_fn=lambda q, ts: [.9 if '车站' in t else .1 for t in ts])
        assert '车站' in tracker.retrieve_for_query('阿白最初怎么认识小林？')


def test_earlier_character_experience_is_a_retrieval_candidate():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / 'story.json',
            similarities_fn=lambda q, ts: [.9 if '车站' in t else .55 for t in ts])
        tracker.apply_evaluation({'character_updates': [{'name': '阿白', 'confidence': .9,
            'known_so_far': ['最初在车站遇见小林', '后来去学校', '后来去港口', '后来回家', '现在睡觉']} ]})
        assert '车站' in tracker.retrieve_for_query('阿白最初怎么认识小林？')


def test_comments_remain_retrievable_after_restart():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / 'memory.json'
        manager = DialogManager(None, path)
        manager.record_session_message('桌宠(视觉小说)', '寄信人原来是阿白，我误会他了。')
        manager = DialogManager(None, path)
        assert '寄信人' in manager._build_session_recall_block('寄信人是谁')


def test_conflicting_update_cannot_pollute_confirmed_character():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / 'story.json')
        tracker.apply_evaluation({'character_updates': [{'name':'阿白','role':'小狗','confidence':.95}]})
        cid = next(iter(tracker.characters))
        tracker.apply_evaluation({'character_updates': [{'id':cid,'name':'阿白','role':'人类',
            'known_so_far':['解开上衣'], 'confidence':.95, 'conflict':'与小狗身份冲突', 'evidence_quote':'解开上衣'}]})
        card = tracker.characters[cid]
        assert card['role'] == '小狗'
        assert '解开上衣' not in card['known_so_far']
        assert card['claims'][-1]['status'] == 'conflict'


def test_visual_identity_requires_fresh_object_and_matching_quote():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / 'story.json')
        tracker.observe('小林指着左边的白衣女孩说：这就是阿白。')
        tracker.apply_evaluation({'character_updates': [{'name':'阿白','role':'同行者','confidence':.9}]})
        cid = next(iter(tracker.characters))
        update = {'id':cid, 'name':'阿白', 'source':'text+visual','visual_object':'A',
                  'evidence_quote':'这就是阿白', 'appearance':['白衣'], 'confidence':.9}
        tracker.apply_evaluation({'character_updates':[update], 'evidence_line_ids':[1],
                                 'visual_evidence':'较早画面观察：对象A（左侧）：白衣女孩'})
        assert '白衣' not in tracker.characters[cid]['appearance']
        tracker.apply_evaluation({'character_updates':[update], 'evidence_line_ids':[1],
                                 'visual_evidence':'对象A（左侧）：白衣女孩'})
        assert '白衣' in tracker.characters[cid]['appearance']
        assert tracker.characters[cid]['claims'][-1]['status'] == 'supported'


def test_story_archive_does_not_leak_when_switching_game():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / 'one.json')
        tracker.observe('阿白在车站遇见小林。')
        tracker.switch_story(Path(tmp) / 'two.json')
        assert '车站' not in tracker.retrieve_for_query('阿白在车站发生什么')


def test_live_update_missing_evidence_cannot_overwrite_identity():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / 'story.json')
        tracker.apply_evaluation({'character_updates':[{'name':'阿白','role':'小狗','confidence':.95}]})
        tracker.apply_evaluation({'require_evidence':True,
            'character_updates':[{'name':'阿白','role':'人类','confidence':.99}]})
        assert next(iter(tracker.characters.values()))['role'] == '小狗'


def test_reload_creates_evidence_boundary_without_erasing_history():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / 'story.json')
        tracker.observe('第一条路线走向桥边。')
        tracker.mark_reload('我又重新读档了')
        tracker.observe('第二条路线走向镇上。')
        rows = [r for r in tracker._archive.rows() if r['kind'] == '原始台词']
        assert len(rows) == 2
        assert rows[0]['payload']['branch'] != rows[1]['payload']['branch']


def test_character_claims_validate_each_attribute_independently():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / 'story.json')
        tracker.observe('阿白回到了车站。')
        tracker.apply_evaluation({'character_updates':[{'name':'阿白','role':'同伴','confidence':.9}]})
        tracker.apply_evaluation({'require_evidence':True, 'evidence_line_ids':[1],
            'character_updates':[{'name':'阿白','confidence':.9,'claims':[
                {'field':'known_so_far','value':'回到车站','source':'text','evidence_quote':'阿白回到了车站'},
                {'field':'appearance','value':'红色头发','source':'visual','visual_object':'A'}]}]})
        card = next(iter(tracker.characters.values()))
        assert '回到车站' in card['known_so_far']
        assert '红色头发' not in card['appearance']
        assert {c['status'] for c in card['claims']} >= {'supported','provisional'}


def test_explicit_character_query_does_not_inherit_wrong_old_reply():
    with TemporaryDirectory() as tmp:
        queries = []
        def similarities(query, texts):
            queries.append(query)
            return [.6] * len(texts)
        tracker = VisualNovelTracker(Path(tmp) / 'story.json', similarities_fn=similarities)
        tracker.apply_evaluation({'character_updates':[
            {'name':'阿白','role':'小狗','confidence':.9}, {'name':'小林','role':'旅人','confidence':.9}]})
        queries.clear()
        text = tracker.retrieve_for_query('阿白是谁？\n最近对话中的指代线索（仅供检索）：\n小林是谁，小林是人。')
        assert queries[:2] == ['阿白是谁？', '阿白是谁？']
        assert text.startswith('当前作品: 《story》\n人物身份记录（历史整理，须与原文核对）: 阿白')


def test_embedding_cache_survives_archive_reopen():
    from desktop_pet.llm.history_archive import HistoryArchive
    import numpy as np
    class Router:
        cache_identity = 'test-model'
        def __init__(self): self.encoded = []
        def similarities(self, query, texts): raise AssertionError('should use persistent cache')
        def embeddings(self, texts):
            self.encoded.extend(texts)
            return np.array([[1., 0.] for _ in texts])
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / 'history.sqlite3'
        router = Router()
        HistoryArchive(path).similarities('问题', ['旧记录'], router.similarities)
        router.encoded.clear()
        assert HistoryArchive(path).similarities('新问题', ['旧记录'], router.similarities) == [1.]
        assert router.encoded == ['新问题']


def test_same_chat_prompt_for_paraphrases_without_extra_call():
    class Client:
        def __init__(self): self.calls = []
        def chat(self, user_text, system_prompt):
            self.calls.append((user_text, system_prompt))
            return '没有找到初遇原文。'
    with TemporaryDirectory() as tmp:
        client = Client()
        manager = DialogManager(client, Path(tmp) / 'memory.json',
            visual_novel_context_provider=lambda query: '原始台词: 阿白后来跟着她们走。')
        manager.record_session_message('桌宠', '我猜他们是在山洞初遇的。')
        questions = ['阿白最开始怎么遇到她们的？', '阿白加入同行的那一幕发生了什么？', '它和她们认识的经过呢？']
        for question in questions:
            manager.reply(question)
        assert len(client.calls) == len(questions)
        user_prompt, system_prompt = client.calls[0]
        assert '山洞' in user_prompt
        assert '阿白后来跟着她们走' in user_prompt
        assert len({prompt for _, prompt in client.calls}) == 1
        assert '剧情问答事实规则' not in system_prompt
        assert '查不到初遇' not in user_prompt


def test_other_story_chat_and_comments_excluded_before_recall():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / 'one.json')
        manager = DialogManager(None, Path(tmp) / 'memory.json',
            visual_novel_context_provider=tracker.retrieve_for_query)
        for role in ['你', '桌宠', '桌宠(视觉小说)']:
            manager.record_session_message(role, '阿白在旧作品的废弃建筑出现。')
        tracker.switch_story(Path(tmp) / 'two.json')
        assert '废弃建筑' not in manager._build_recent_session_block()
        assert '废弃建筑' not in manager.build_recent_session_hint()
        assert '废弃建筑' not in manager._build_session_recall_block('阿白在废弃建筑出现')
        manager.record_session_message('桌宠(视觉小说)', '阿白正在车站等车。')
        manager = DialogManager(None, Path(tmp) / 'memory.json',
            visual_novel_context_provider=tracker.retrieve_for_query)
        assert '车站' in manager._build_session_recall_block('阿白在车站等车')
        assert '废弃建筑' not in manager._build_session_recall_block('阿白在废弃建筑出现')


def test_unscoped_archive_does_not_supply_named_story_history():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / 'memory.json'
        manager = DialogManager(None, path)
        manager.record_session_message('桌宠', '阿白在废弃建筑出现。')
        tracker = VisualNovelTracker(Path(tmp) / 'one.json')
        manager = DialogManager(None, path, visual_novel_context_provider=tracker.retrieve_for_query)
        assert not manager._build_session_recall_block('阿白在废弃建筑出现')
        assert manager._session_archive.rows()


def test_same_name_visual_entities_remain_separate_and_ambiguous_update_waits():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / 'story.json')
        tracker.observe('阿白是一只白狗。阿白也是她哥哥的名字。')
        for kind, quote, role, obj in [('animal', '阿白是一只白狗', '白狗', 'A'),
                                      ('human', '阿白也是她哥哥的名字', '哥哥', 'B')]:
            tracker.apply_evaluation({'evidence_line_ids':[1],
                'visual_evidence':'对象A（左侧）：[type=animal]白狗；对象B（右侧）：[type=human]青年',
                'character_updates':[{'name':'阿白','entity_type':kind,'identity_action':'new','confidence':.9,
                    'claims':[{'field':'role','value':role,'source':'text+visual','visual_object':obj,'evidence_quote':quote},
                              {'field':'known_so_far','value':quote,'source':'text','evidence_quote':quote}]}]})
        assert len(tracker.characters) == 2
        assert {c['entity_type'] for c in tracker.characters.values()} == {'human','animal'}
        assert len(tracker._select_characters_for_context('阿白')) == 2
        tracker.observe('阿白走远了。')
        tracker.apply_evaluation({'evidence_line_ids':[2], 'character_updates':[
            {'name':'阿白','known_so_far':['走远了'],'source':'text','evidence_quote':'阿白走远了'}]})
        assert all('走远了' not in c['known_so_far'] for c in tracker.characters.values())
        assert tracker._archive.rows()[-1]['kind'] == '待归属人物信息'


def test_visual_type_mismatch_cannot_confirm_wrong_character():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / 'story.json')
        tracker.observe('阿白在这里。')
        tracker.apply_evaluation({'evidence_line_ids':[1],
            'visual_evidence':'对象A（中间）：[type=animal]白狗',
            'character_updates':[{'name':'阿白','entity_type':'human','role':'哥哥',
                'source':'text+visual','visual_object':'A','evidence_quote':'阿白在这里','confidence':.95}]})
        assert not tracker.characters


def test_ocr_name_variant_preserves_canonical_identity_after_restart():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / 'story.json'
        tracker = VisualNovelTracker(path)
        tracker.apply_evaluation({'character_updates':[{'name':'萤','entity_type':'human','role':'女孩','confidence':.9}]})
        cid = next(iter(tracker.characters))
        tracker.observe('荧穿着白衣。')
        tracker.apply_evaluation({'evidence_line_ids':[1], 'visual_evidence':'对象A（左侧）：[type=human]白衣女孩',
            'character_updates':[{'id':cid,'name':'荧','identity_action':'existing','entity_type':'human',
                'appearance':['白衣'],'source':'text+visual','visual_object':'A','evidence_quote':'荧穿着白衣','confidence':.95}]})
        tracker.flush()
        tracker = VisualNovelTracker(path)
        assert len(tracker.characters) == 1
        assert tracker.characters[cid]['name'] == '萤'
        assert tracker.characters[cid]['ocr_names'] == ['荧']
        assert '荧' not in tracker.characters[cid]['aliases']


def test_wrong_existing_id_cannot_mix_animal_and_human():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / 'story.json')
        tracker.apply_evaluation({'character_updates':[{'name':'阿白','entity_type':'animal','role':'白狗','confidence':.9}]})
        cid = next(iter(tracker.characters))
        tracker.observe('阿白是她哥哥。')
        tracker.apply_evaluation({'evidence_line_ids':[1], 'character_updates':[
            {'id':cid,'name':'阿白','entity_type':'human','identity_action':'existing','role':'哥哥',
             'source':'text','evidence_quote':'阿白是她哥哥','confidence':.99}]})
        assert tracker.characters[cid]['role'] == '白狗'


def test_visual_context_selects_character_when_ocr_omits_name():
    with TemporaryDirectory() as tmp:
        def similarity(query, texts):
            return [.95 if '金发' in query and '金发' in text else .1 for text in texts]
        tracker = VisualNovelTracker(Path(tmp)/'story.json', similarities_fn=similarity)
        tracker.apply_evaluation({'character_updates':[
            {'name':'小林','appearance':['金发、格纹发箍'],'role':'同行女孩','confidence':.9},
            {'name':'阿白','role':'白狗','confidence':.9}]})
        cards = tracker._select_characters_for_context('哥哥来过这里。', visual_context='右边是金发、格纹发箍女孩', limit=1)
        assert cards[0]['name'] == '小林'


def test_nameless_text_reference_does_not_create_another_character():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp)/'story.json')
        tracker.observe('这是哥哥的字。')
        tracker.apply_evaluation({'evidence_line_ids':[1], 'character_updates':[
            {'name':'','aliases':['同伴','哥哥'],'identity_action':'new','entity_type':'human',
             'known_so_far':['读笔记'],'source':'text','evidence_quote':'这是哥哥的字','confidence':.9}]})
        assert not tracker.characters
        assert tracker._archive.rows()[-1]['kind'] == '待核对人物信息'


def test_explicit_identity_revelation_merges_forms_and_redirects_after_restart():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'story.json'
        tracker = VisualNovelTracker(path)
        for kind, role in [('animal', '白狗'), ('human', '哥哥')]:
            tracker.apply_evaluation({'character_updates':[{'name':'阿白','entity_type':kind,
                'identity_action':'new','role':role,'known_so_far':[role+'的经历'],'confidence':.9}]})
        ids = list(tracker.characters)
        assert len(ids) == 2
        tracker.observe('这只白狗原来就是她的哥哥。')
        link = {'relation':'same_entity','character_ids':ids,'evidence_kind':'explicit_identity',
                'evidence_quote':'这只白狗原来就是她的哥哥','identity_summary':'哥哥变成了白狗，是同一个体。'}
        tracker.apply_evaluation({'evidence_line_ids':[1], 'identity_links':[link]})
        tracker.flush()
        tracker = VisualNovelTracker(path, similarities_fn=lambda query, texts: [.8] * len(texts))
        assert len(tracker.characters) == 1
        card = tracker.characters[ids[0]]
        assert {f['entity_type'] for f in card['forms']} == {'animal', 'human'}
        assert all(f['known_so_far'] for f in card['forms'])
        assert tracker._character_redirects[ids[1]] == ids[0]
        assert '同一个体' in tracker.retrieve_for_query('阿白是谁')
        tracker.observe('哥哥记得从前的事情。')
        tracker.apply_evaluation({'evidence_line_ids':[2], 'character_updates':[{'id':ids[1],
            'entity_type':'human','known_so_far':['记得从前的事情'],'source':'text',
            'evidence_quote':'哥哥记得从前的事情','confidence':.9}]})
        assert len(tracker.characters) == 1
        assert '记得从前的事情' in tracker.characters[ids[0]]['known_so_far']


def test_identity_link_requires_current_explicit_evidence():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp)/'story.json')
        for kind in ['animal', 'human']:
            tracker.apply_evaluation({'character_updates':[{'name':'阿白','entity_type':kind,
                'identity_action':'new','role':kind,'confidence':.9}]})
        tracker.observe('他们都叫阿白。')
        for evidence_kind, quote in [('inference', '他们都叫阿白'), ('explicit_identity', '狗就是哥哥')]:
            tracker.apply_evaluation({'evidence_line_ids':[1], 'identity_links':[{
                'relation':'same_entity','character_ids':list(tracker.characters),
                'evidence_kind':evidence_kind,'evidence_quote':quote,'identity_summary':'同一个体'}]})
            assert len(tracker.characters) == 2


def test_reaction_window_includes_whole_unread_batch():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp)/'story.json')
        tracker.observe('之前已知的事情。')
        tracker._summarized_line_count = 1
        for line in ['原来白狗就是哥哥。', '风从窗外吹进来。', '她收起了笔记。', '另一人走进屋里。', '她们转过身。']:
            tracker.observe(line)
        tracker._pending_candidate = True
        payload = tracker.build_evaluation_payload()
        assert '原来白狗就是哥哥' in payload['reaction_dialogue']
        assert '之前已知' not in payload['reaction_dialogue']


def test_homonymous_candidates_are_retrieved_together_for_identity_comparison():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp)/'story.json',
            similarities_fn=lambda query, texts: [.9 if '神秘' in text else .5 if '白狗' in text else .1 for text in texts])
        for name, kind, role in [('阿白','animal','白狗'), ('阿白','human','哥哥'), ('小林','human','神秘旅人')]:
            tracker.apply_evaluation({'character_updates':[{'name':name,'entity_type':kind,
                'identity_action':'new','role':role,'confidence':.9}]})
        cards = tracker._select_characters_for_context('神秘旅人说阿白就是她哥哥', limit=2)
        assert len(cards) == 2
        assert {card['entity_type'] for card in cards} == {'human', 'animal'}
        assert all(card['name'] == '阿白' for card in cards)


def test_inflight_backlog_does_not_drop_early_unread_lines():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp)/'story.json')
        tracker.observe('旧剧情。')
        tracker._summarized_line_count = tracker._last_requested_line_count = 1
        tracker.observe('哥哥被扭曲吞噬，变成了白狗。')
        for i in range(25):
            tracker.observe(f'后来第{i}个路人走过广场。')
        snapshot, _ = tracker.reserve_evaluation_snapshot()
        payload = snapshot.build_evaluation_payload()
        assert '哥哥被扭曲吞噬' in payload['new_dialogue']
        assert 2 in payload['evidence_line_ids']
        assert payload['line_count'] == tracker.last_requested_line_count


def test_evaluation_budget_leaves_unsent_lines_pending():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp)/'story.json', summary_batch_size=8)
        for i in range(40):
            tracker.observe(f'{i}：'+chr(0x4e00+i)*120)
        snapshot, _ = tracker.reserve_evaluation_snapshot()
        first = snapshot.build_evaluation_payload()
        assert len(first['new_dialogue']) <= 3600
        assert first['line_count'] < 40
        tracker.apply_evaluation({})
        second = tracker.build_evaluation_payload()
        assert second is not None
        assert str(first['line_count'])+'：' in second['new_dialogue']
        assert second['line_count'] == 40


def test_completed_ocr_revision_survives_json_trim_and_restart():
    with TemporaryDirectory() as tmp:
        path = Path(tmp)/'story.json'
        tracker = VisualNovelTracker(path, max_lines=60)
        short = '【小林】我一直想告诉你那件事情'
        complete = short + '，其实那封信是我写的。'
        tracker._commit_ocr(short, .95)
        line_id = tracker._lines[-1]['id']
        branch = tracker._lines[-1]['branch']
        assert tracker._commit_ocr(complete, .98).reason == 'extended_previous'
        assert tracker.total_lines == 1
        for i in range(65):
            tracker._is_duplicate = lambda text: False
            tracker.observe(f'第{i}个不同地点出现了新的剧情线索。')
        tracker.flush()
        restored = VisualNovelTracker(path, max_lines=60)
        assert all(row['id'] != line_id for row in restored._lines)
        versions = [row for row in restored._archive.rows()
                    if row['kind']=='原始台词' and row['payload'].get('id')==line_id
                    and row['payload'].get('branch')==branch]
        assert versions[-1]['text'] == complete
        assert versions[-1]['payload']['confidence'] == .98
        assert versions[0]['text'] == short  # Original evidence remains append-only.
