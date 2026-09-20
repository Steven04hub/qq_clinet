import json
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.services.memory_search import alias_in_text, excerpt, tokenize
from app.services.profile_store import SCHEMA_VERSION, GroupMemoryStore


def message(key, text, *, user="10", name="老张", is_bot=False):
    return {
        "message_id": str(key),
        "user_id": user,
        "nickname": name,
        "text": text,
        "time": int(key),
        "is_bot": is_bot,
    }


@pytest.fixture
def store(tmp_path):
    return GroupMemoryStore(tmp_path / "memory" / "groups.sqlite3")


def test_chinese_retrieval_ranks_topic_not_merely_recent_history(store):
    store.ingest(
        "a",
        [
            message(1, "上次火锅那个蘸料太咸了，老板还说秘制"),
            message(2, "今天在家打游戏"),
            message(3, "周末去吃火锅吗"),
            message(4, "刚下班到家"),
        ],
    )
    result = store.search("a", "火锅的蘸料是什么", limit=2)
    assert result[0]["message_id"] == "1"
    assert {item["message_id"] for item in result} == {"1", "3"}
    assert all(item["score"] > 0 for item in result)


def test_search_cannot_cross_groups_or_retrieve_bot_or_excluded_messages(store):
    store.ingest("a", [message(1, "火锅蘸料秘籍"), message(2, "火锅蘸料配方", is_bot=True)])
    store.ingest("b", [message(1, "另一群的火锅蘸料秘密")])
    assert [item["text"] for item in store.search("a", "火锅")] == ["火锅蘸料秘籍"]
    assert store.search("a", "火锅", exclude_message_ids=["1"]) == []
    assert store.search("unknown", "火锅") == []


def test_manual_alias_expands_people_and_their_other_names(store):
    # These members, nicknames and messages are entirely fictional test data.
    store.ingest(
        "a",
        [
            message(1, "周末想去爬山", user="10", name="samplepilot"),
            message(2, "SAMPLEPILOT 的路线记得发一下", user="20", name="示例成员乙"),
            message(3, "明早想吃面", user="20", name="示例成员乙"),
        ],
    )
    explanation = "虚构样例：纸飞机对应 samplepilot"
    store.set_alias("a", "纸飞机", ["10"], explanation=explanation)
    result = store.search("a", "纸飞机今天在干嘛")
    assert {item["message_id"] for item in result} == {"1", "2"}
    assert (
        next(a for a in store.overview("a")["aliases"] if a["alias"] == "纸飞机")["explanation"]
        == explanation
    )
    # Aliases from another group cannot add this group's authors to the query.
    store.ingest("b", [message(1, "只是另一个群的成员", user="10", name="示例成员甲")])
    assert store.search("b", "纸飞机今天在干嘛") == []


def test_ambiguous_alias_does_not_guess_a_person(store):
    store.ingest("a", [message(1, "今天吃饭了", user="10"), message(2, "准备睡觉", user="20")])
    store.set_alias("a", "共同昵称", ["10", "20"])
    assert store.search("a", "共同昵称") == []
    assert store.search("a", "未出现的话题", user_ids=["10"]) == []


def test_latin_alias_boundaries_and_emoji_queries(store):
    assert alias_in_text("R", "R今天来了吗")
    assert alias_in_text("samplepilot", "SAMPLEPILOT!")
    assert not alias_in_text("r", "orange")
    store.ingest("a", [message(1, "这个操作笑死😂"), message(2, "今天正常上班")])
    assert [row["message_id"] for row in store.search("a", "😂")] == ["1"]


def test_retrieval_deduplicates_repeats_and_bounds_long_evidence(store):
    store.ingest(
        "a",
        [
            message(1, "无关内容" * 400 + "火锅蘸料独家配方" + "还有其他内容" * 300),
            message(2, "火锅真好吃"),
            message(3, "火锅 真好吃"),
        ],
    )
    result = store.search("a", "火锅蘸料", limit=50)
    assert len(result) == 2
    assert all(len(row["text"]) <= 500 for row in result)
    assert any("独家配方" in row["text"] for row in result)
    assert store.search("a", "") == []
    assert store.search("a", "火锅", limit=0) == []


def test_punctuation_is_data_not_fts_query_syntax(store):
    store.ingest("a", [message(1, "火锅真好吃")])
    assert store.search("a", '"火锅" OR (NOT * :')
    assert tokenize("哈" * 100) == []


def test_old_database_is_indexed_automatically_and_survives_reopen(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE messages (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT NOT NULL, message_id TEXT NOT NULL,
            time INTEGER NOT NULL, payload TEXT NOT NULL,
            UNIQUE(group_id,message_id))""")
        payload = {
            **message(1, "上次奶茶喝得睡不着"),
            "emoji": [],
            "mentions": [],
            "reply_to": None,
        }
        db.execute(
            "INSERT INTO messages(group_id,message_id,time,payload) VALUES(?,?,?,?)",
            ("a", "1", 1, json.dumps(payload)),
        )
    store = GroupMemoryStore(path)
    assert store.schema_version == SCHEMA_VERSION
    assert store.search("a", "奶茶")[0]["message_id"] == "1"
    store.ingest("a", [message(2, "奶茶我不喝了")])
    reopened = GroupMemoryStore(path)
    assert len(reopened.search("a", "奶茶")) == 2
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert db.execute("SELECT COUNT(*) FROM memory_documents").fetchone()[0] == 2


def test_fallback_when_sqlite_has_no_fts5(tmp_path, monkeypatch):
    original_connect = sqlite3.connect

    class NoFTSConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if "CREATE VIRTUAL TABLE" in sql:
                raise sqlite3.OperationalError("no such module: fts5")
            return super().execute(sql, parameters)

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: original_connect(*args, **kwargs, factory=NoFTSConnection),
    )
    store = GroupMemoryStore(tmp_path / "memory.sqlite3")
    assert not store._fts_enabled
    store.ingest("a", [message(1, "上次火锅蘸料太咸"), message(2, "今天起床了")])
    store.ingest("b", [message(3, "火锅蘸料是秘密")])
    assert [row["message_id"] for row in store.search("a", "蘸料")] == ["1"]


def test_retention_prunes_search_indexes_together(store, monkeypatch):
    monkeypatch.setattr("app.services.profile_store.MAX_MESSAGES", 2)
    store.ingest(
        "a", [message(1, "独家消失的火锅"), message(2, "保留的奶茶"), message(3, "保留的面包")]
    )
    assert store.search("a", "独家消失") == []
    assert store.search("a", "奶茶")[0]["message_id"] == "2"
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM memory_documents").fetchone()[0] == 2
        if store._fts_enabled:
            assert db.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0] == 2


def test_backup_restores_profiles_aliases_and_search_with_private_permissions(store, tmp_path):
    store.ingest("a", [message(1, "火锅蘸料好咸")])
    store.set_alias("a", "火锅王", ["10"], explanation="用户明确确认")
    backup = tmp_path / "export" / "memory.sqlite3"
    assert store.backup_to(backup) == backup
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    restored = GroupMemoryStore(backup)
    assert restored.overview("a") == store.overview("a")
    assert restored.search("a", "火锅王")[0]["message_id"] == "1"
    store.ingest("a", [message(2, "之后才收到的消息")])
    assert len(restored.messages("a")) == 1
    with pytest.raises(FileExistsError):
        store.backup_to(backup)
    with pytest.raises(ValueError, match="differ"):
        store.backup_to(store.path)


def test_backup_is_consistent_during_concurrent_writes(store, tmp_path):
    store.ingest("a", [message(1, "奶茶初始记录")])
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(
            store.ingest, "a", [message(i, f"奶茶第{i}次") for i in range(2, 100)]
        )
        backup = store.backup_to(tmp_path / "snapshot.sqlite3")
        pending.result()
    restored = GroupMemoryStore(backup)
    with sqlite3.connect(backup) as db:
        count = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert db.execute("SELECT COUNT(*) FROM memory_documents").fetchone()[0] == count
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert restored.search("a", "奶茶")


def test_short_excerpt_handles_matches_at_end():
    text = "无" * 700 + "关键词在结尾"
    result = excerpt(text, ["关键词"])
    assert len(result) <= 500
    assert "关键词在结尾" in result
