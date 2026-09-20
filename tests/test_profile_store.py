import json
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.services.profile_store import GroupMemoryStore


def message(key="1", user="10", nickname="阿明", text="今天又摸鱼", **extra):
    return {
        "message_id": key,
        "user_id": user,
        "nickname": nickname,
        "text": text,
        "time": int(key) if key.isdigit() else 1,
        "mentions": [],
        "reply_to": None,
        "emoji": [],
        "is_bot": False,
        **extra,
    }


@pytest.fixture
def store(tmp_path):
    return GroupMemoryStore(tmp_path / "memory" / "groups.sqlite3")


def analysis(user="10", evidence="1", alias="摸鱼王"):
    return {
        "members": [
            {
                "user_id": user,
                "summary": "喜欢用摸鱼开玩笑",
                "facts": [
                    {
                        "category": "catchphrase",
                        "text": "常说今天又摸鱼",
                        "confidence": "observed",
                        "evidence_ids": [evidence],
                    }
                ],
            }
        ],
        "aliases": [
            {
                "alias": alias,
                "user_ids": [user],
                "status": "likely",
                "evidence_ids": [evidence],
                "explanation": "聊天中被这样称呼",
            }
        ],
        "memes": [
            {
                "name": "摸鱼",
                "meaning": "拿假装忙碌自嘲",
                "user_ids": [user],
                "evidence_ids": [evidence],
            }
        ],
    }


def test_dedup_reimports_and_preserves_observed_names_and_emoji(store):
    assert store.ingest("group", [message(emoji=["😂", "😂", "[face:1]"])]) == 1
    assert store.ingest("group", [message(emoji=["😂"])], source="history") == 0
    assert store.ingest("group", [message("2", nickname="老明", emoji=["😂"])]) == 1
    overview = store.overview("group")
    member = overview["members"][0]
    assert member["names"] == ["阿明", "老明"]
    assert member["display_name"] == "老明"
    assert member["message_count"] == 2
    assert member["emoji_habits"] == [
        {"emoji": "😂", "count": 3},
        {"emoji": "[face:1]", "count": 1},
    ]
    assert member["profile"] is None
    assert {alias["status"] for alias in overview["aliases"]} == {"confirmed"}


def test_groups_are_isolated_even_with_same_message_and_user_ids(store):
    store.ingest("a", [message(text="a only")])
    store.ingest("b", [message(text="b only", nickname="另一人")])
    store.save_analysis("a", analysis())
    assert store.messages("a")[0]["text"] == "a only"
    assert store.messages("b")[0]["text"] == "b only"
    assert store.overview("a")["stats"]["profiled_members"] == 1
    assert store.overview("b")["stats"]["profiled_members"] == 0
    assert "摸鱼" not in store.prompt_context("b", ["10"], "")
    assert store.overview("unknown")["members"] == []


def test_semantic_tool_source_update_preserves_evidence_and_group_isolation(store):
    store.ingest("a", [message()])
    store.ingest("b", [message()])
    before = store.messages("a")[0]
    store.mark_message_source("a", "1", "jev_tool")
    store.mark_message_source("a", "missing", "jev_tool")
    assert store.messages("a") == [{**before, "source": "jev_tool"}]
    assert store.messages("b")[0]["source"] == "live"
    assert store.overview("a")["members"][0]["message_count"] == 1


def test_legacy_internal_jev_requests_remain_stored_but_cannot_be_retrieved(store):
    store.ingest("a", [message(text="测试接话机会摸鱼")], source="jev_test")
    store.ingest("a", [message("2", text="今天摸鱼被发现了")])
    assert len(store.messages("a")) == 2
    assert {row["message_id"] for row in store.search("a", "摸鱼")} == {"2"}


def test_persistence_and_permissions(tmp_path):
    path = tmp_path / "private" / "memory.sqlite3"
    first = GroupMemoryStore(path)
    first.ingest("a", [message()])
    first.save_analysis("a", analysis())
    first.close()
    second = GroupMemoryStore(path)
    assert second.messages("a")[0]["text"] == "今天又摸鱼"
    assert second.overview("a")["members"][0]["profile"]["summary"] == "喜欢用摸鱼开玩笑"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_roster_adds_card_aliases_without_inventing_profiles(store):
    store.ingest_members("a", [{"user_id": 10, "nickname": "阿明", "card": "老明"}])
    member = store.overview("a")["members"][0]
    assert member["display_name"] == "老明"
    assert member["names"] == ["阿明", "老明"]
    assert member["message_count"] == 0
    assert member["profile"] is None


def test_manual_alias_overrides_model_and_observation(store):
    store.ingest("a", [message(), message("2", "20", "小红")])
    store.save_analysis("a", analysis())
    store.set_alias("a", "摸鱼王", ["20"])
    store.save_analysis("a", analysis())
    store.ingest_members("a", [{"user_id": "10", "card": "摸鱼王"}])
    alias = next(item for item in store.overview("a")["aliases"] if item["alias"] == "摸鱼王")
    assert alias["user_ids"] == ["20"]
    assert alias["status"] == "confirmed"
    assert alias["source"] == "manual"
    with pytest.raises(ValueError, match="unknown"):
        store.set_alias("a", "无中生有", ["999"])
    with pytest.raises(ValueError, match="unknown"):
        store.set_alias("b", "小红", ["20"])


def test_observed_alias_collision_retains_both_members(store):
    store.ingest("a", [message(), message("2", "20", "阿明")])
    alias = store.overview("a")["aliases"][0]
    assert alias["alias"] == "阿明"
    assert alias["user_ids"] == ["10", "20"]
    assert alias["status"] == "ambiguous"
    store.save_analysis("a", analysis(alias="阿明"))
    assert store.overview("a")["aliases"][0] == alias


def test_model_alias_collision_is_ambiguous(store):
    store.ingest("a", [message(), message("2", "20", "小红")])
    store.save_analysis("a", analysis())
    store.save_analysis("a", analysis("20", "2"))
    alias = next(item for item in store.overview("a")["aliases"] if item["alias"] == "摸鱼王")
    assert alias["status"] == "ambiguous"
    assert alias["user_ids"] == ["10", "20"]


def test_ungrounded_and_other_group_analysis_cannot_create_profiles(store):
    store.ingest("a", [message()])
    store.ingest("b", [message("2", "20")])
    store.save_analysis("a", analysis(evidence="2"))
    store.save_analysis("a", analysis(user="20"))
    overview = store.overview("a")
    assert overview["stats"]["profiled_members"] == 0
    assert len(overview["members"]) == 1
    # Unknown ids may not become alias targets or meme participants.
    assert all("20" not in item["user_ids"] for item in overview["aliases"])
    assert all("20" not in item["user_ids"] for item in overview["memes"])


def test_merging_facts_deduplicates_and_attaches_evidence(store):
    store.ingest("a", [message(), message("2")])
    store.save_analysis("a", analysis())
    store.save_analysis("a", analysis(evidence="2"))
    overview = store.overview("a")
    facts = overview["members"][0]["profile"]["facts"]
    assert len(facts) == 1
    assert facts[0]["evidence_ids"] == ["1", "2"]
    assert {item["message_id"] for item in overview["evidence"]} == {"1", "2"}


def test_prompt_selects_alias_referents_and_stays_valid_and_bounded(store):
    store.ingest("a", [message(), message("2", "20", "小红")])
    store.save_analysis("a", analysis())
    store.save_analysis("a", analysis("20", "2", "潜水王"))
    context = json.loads(store.prompt_context("a", [], "摸鱼王快来"))
    assert [member["user_id"] for member in context["members"]] == ["10"]
    assert context["memes"][0]["name"] == "摸鱼"
    for limit in [2, 50, 150, 400, 800, 1500, 10000]:
        serialized = store.prompt_context("a", ["10"], "摸鱼王", max_chars=limit)
        assert len(serialized) <= limit
        assert isinstance(json.loads(serialized), dict)
    with pytest.raises(ValueError):
        store.prompt_context("a", [], "", max_chars=1)


def test_retention_keeps_newest_history_without_recounting_old_ids(store):
    store.ingest("a", [message(str(index)) for index in range(5002)])
    assert len(store.messages("a", limit=6000)) == 5000
    assert store.messages("a", limit=6000)[0]["message_id"] == "2"
    assert store.messages("a", limit=1)[0]["message_id"] == "5001"
    assert store.overview("a")["stats"]["messages"] == 5002
    assert store.ingest("a", [message("0")], source="history") == 0
    assert store.overview("a")["stats"]["messages"] == 5002
    # Importing an older message must not evict a newer message or restore an old nickname.
    store.ingest("a", [message("-1", nickname="过去昵称", time=-1)])
    assert len(store.messages("a", limit=6000)) == 5000
    assert store.overview("a")["members"][0]["display_name"] == "阿明"


def test_sql_values_are_bound_and_concurrent_ingest_deduplicates(store):
    group_id = "a'); DROP TABLE messages; --"
    with ThreadPoolExecutor(max_workers=6) as pool:
        inserted = list(pool.map(lambda _: store.ingest(group_id, [message()]), range(20)))
    assert sum(inserted) == 1
    assert store.messages(group_id)[0]["message_id"] == "1"
    assert store.overview(group_id)["stats"]["messages"] == 1


def test_bot_messages_remain_context_without_creating_human_profile(store):
    store.ingest("a", [message(), message("2", "100", "机器人", is_bot=True, emoji=["😂"])])
    store.save_analysis("a", analysis("100", "2"))
    overview = store.overview("a")
    assert overview["stats"]["messages"] == 2
    assert overview["stats"]["human_messages"] == 1
    assert [member["user_id"] for member in overview["members"]] == ["10"]
    assert len(store.messages("a")) == 2
    assert all("100" not in alias["user_ids"] for alias in overview["aliases"])


def test_large_profiles_share_budget_and_preserve_current_alias_and_meme(store):
    for index, user_id in enumerate(["10", "20", "30"], start=1):
        key = str(index)
        store.ingest("a", [message(key, user_id, f"成员{index}")])
        findings = analysis(user_id, key)
        findings["members"][0]["facts"] = [
            {
                "category": "topics",
                "text": "生产力是摸鱼的话术" + "具体习惯" * 400,
                "evidence_ids": [key],
                "confidence": "observed",
            },
            *[
                {
                    "category": "chat_traits",
                    "text": f"无关观察{number}" + "另外的习惯" * 300,
                    "evidence_ids": [key],
                    "confidence": "observed",
                }
                for number in range(59)
            ],
        ]
        findings["members"][0]["summary"] = "很长的历史总结" * 300
        store.save_analysis("a", findings)
    store.ingest("other", [message(nickname="别群不该出现")])
    store.save_analysis("other", analysis(alias="外群绰号"))
    for limit in [3000, 10000]:
        serialized = store.prompt_context(
            "a", ["10", "20", "30"], "摸鱼王这生产力如何", max_chars=limit
        )
        context = json.loads(serialized)
        assert len(serialized) <= limit
        assert [member["user_id"] for member in context["members"]] == ["30", "20", "10"]
        assert any(alias["alias"] == "摸鱼王" for alias in context["aliases"])
        assert any(meme["name"] == "摸鱼" for meme in context["memes"])
        for member in context["members"]:
            assert 1 <= len(member["profile"]["facts"]) <= 10
            assert member["profile"]["facts"][0]["text"].startswith("生产力是摸鱼的话术")
            assert len(member["names"]) <= 8
        assert "外群绰号" not in serialized


def test_prompt_alias_matching_is_case_insensitive_with_latin_boundaries(store):
    store.ingest("a", [message(nickname="CaptainCat"), message("2", "20", "路人")])
    store.set_alias("a", "K", ["10"])
    context = json.loads(store.prompt_context("a", ["20"], "CAPTAINCAT 来了吗"))
    assert [member["user_id"] for member in context["members"]] == ["20", "10"]
    assert any(alias["alias"] == "CaptainCat" for alias in context["aliases"])
    unrelated = json.loads(store.prompt_context("a", ["20"], "okay"))
    assert [member["user_id"] for member in unrelated["members"]] == ["20"]
    assert all(alias["alias"] != "K" for alias in unrelated["aliases"])
    direct = json.loads(store.prompt_context("a", ["20"], "k又来了"))
    assert any(alias["alias"] == "K" and alias["user_ids"] == ["10"] for alias in direct["aliases"])


def test_manual_identity_has_priority_over_old_aliases_and_memes_with_small_budget(store):
    store.ingest("a", [message()])
    for number in range(20):
        store.ingest_members("a", [{"user_id": "10", "nickname": f"aaa旧名字{number:02}"}])
        findings = analysis(alias=f"bbb模型绰号{number:02}")
        findings["memes"] = [
            {
                "name": f"旧梗{number}",
                "meaning": "旧的长解释" * 150,
                "user_ids": ["10"],
                "evidence_ids": ["1"],
            }
        ]
        store.save_analysis("a", findings)
    store.set_alias("a", "zzz已确认", ["10"], explanation="由用户补充的同音映射")
    store.ingest("other", [message(nickname="别群名字")])
    store.set_alias("other", "zzz别群手动", ["10"])
    serialized = store.prompt_context("a", ["10"], "aaa旧名字00 旧梗0", max_chars=1600)
    context = json.loads(serialized)
    assert len(serialized) <= 1600
    assert context["aliases"][0]["alias"] == "zzz已确认"
    assert context["aliases"][0]["source"] == "manual"
    assert context["aliases"][0]["explanation"] == "由用户补充的同音映射"
    assert "zzz别群手动" not in serialized


def test_explicit_manual_alias_precedes_other_manual_aliases_without_erasing_ambiguity(store):
    store.ingest("a", [message(), message("2", "20", "小红")])
    for number in range(10):
        store.set_alias("a", f"aaa已确认{number}", ["10"])
    store.set_alias("a", "ZETA", ["10"])
    context = json.loads(store.prompt_context("a", ["10"], "zeta呢", max_chars=1200))
    assert context["aliases"][0]["alias"] == "ZETA"
    store.ingest_members(
        "a", [{"user_id": "10", "nickname": "共同名字"}, {"user_id": "20", "nickname": "共同名字"}]
    )
    context = json.loads(store.prompt_context("a", [], "共同名字 是谁", max_chars=10000))
    alias = next(item for item in context["aliases"] if item["alias"] == "共同名字")
    assert alias["status"] == "ambiguous"
    assert alias["user_ids"] == ["10", "20"]


def test_memes_follow_current_topic_instead_of_speaker_identity(store):
    store.ingest("a", [message(nickname="阿明"), message("2", "20", "小红")])
    store.save_analysis(
        "a",
        {
            "memes": [
                {
                    "name": "CUDA家谱",
                    "meaning": "阿明用显卡族谱接技术话题的旧梗",
                    "user_ids": ["10"],
                    "evidence_ids": ["1"],
                },
                {
                    "name": "尊贵饿鬼",
                    "meaning": "拿点外卖也要自称皇帝这件事调侃",
                    "user_ids": ["20"],
                    "evidence_ids": ["2"],
                },
                {
                    "name": "虚空水冷",
                    "meaning": "用水冷散热比喻吵架",
                    "user_ids": [],
                    "evidence_ids": ["1"],
                },
            ]
        },
    )
    serialized = store.prompt_context("a", ["10"], "[阿明/10] 今天又点外卖了")
    context = json.loads(serialized)
    assert [meme["name"] for meme in context["memes"]] == ["尊贵饿鬼"]
    assert "CUDA家谱" not in serialized
    assert "虚空水冷" not in serialized
    addressed = json.loads(store.prompt_context("a", ["10"], "阿明 今天外卖吃啥"))
    assert [meme["name"] for meme in addressed["memes"]] == ["尊贵饿鬼"]
    unrelated = json.loads(store.prompt_context("a", ["10"], "准备出门跑步"))
    assert unrelated["memes"] == []
    assert len(store.overview("a")["memes"]) == 3
    explicitly_recalled = json.loads(store.prompt_context("a", ["10"], "cuda家谱那次呢"))
    assert explicitly_recalled["memes"][0]["name"] == "CUDA家谱"


def test_explicit_memes_precede_semantically_related_callbacks_with_limited_budget(store):
    store.ingest("a", [message()])
    store.save_analysis(
        "a",
        {
            "memes": [
                {
                    "name": "aaa旧饭梗",
                    "meaning": "拿外卖晚到调侃加班",
                    "user_ids": ["10"],
                    "evidence_ids": ["1"],
                },
                {
                    "name": "zzz饿鬼",
                    "meaning": "对点外卖也自称皇帝的调侃",
                    "user_ids": ["10"],
                    "evidence_ids": ["1"],
                },
            ]
        },
    )
    context = json.loads(store.prompt_context("a", ["10"], "zzz饿鬼你还在等外卖", max_chars=1500))
    assert context["memes"][0]["name"] == "zzz饿鬼"
    assert "aaa旧饭梗" in {meme["name"] for meme in context["memes"]}
