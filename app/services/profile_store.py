"""Private, group-scoped memory of observed chat and evidence-backed profiles.

This store deliberately knows nothing about the model or OneBot. Callers pass only
messages from their selected group. Every database operation is additionally scoped
to a group, so changing the selected group never mixes members or chat history.
"""

import json
import os
import re
import sqlite3
import threading
from collections import Counter
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from app.services.memory_search import (
    SEARCH_VERSION,
    alias_in_text,
    bm25_fallback,
    excerpt,
    match_expression,
    query_terms,
    tokenize,
)

MAX_MESSAGES = 5000
SCHEMA_VERSION = 1


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _strings(value: Any, limit: int = 100, length: int = 200) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, (str, int)) and not isinstance(item, bool):
            item = str(item).strip()[:length]
            if item and item not in result:
                result.append(item)
        if len(result) >= limit:
            break
    return result


def _text(value: Any, limit: int = 2000) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


class GroupMemoryStore:
    """Synchronous API, safe to call from multiple ``asyncio.to_thread`` jobs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self._lock = threading.RLock()
        # Create with restrictive permissions before sqlite writes any chat data.
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        os.chmod(self.path, 0o600)
        self._fts_enabled = False
        with self._connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ValueError("memory database was created by a newer application version")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS members (
                    group_id TEXT NOT NULL, user_id TEXT NOT NULL,
                    display_name TEXT NOT NULL, names TEXT NOT NULL DEFAULT '[]',
                    message_count INTEGER NOT NULL DEFAULT 0,
                    emoji_habits TEXT NOT NULL DEFAULT '{}',
                    last_seen INTEGER NOT NULL DEFAULT 0, profile TEXT,
                    PRIMARY KEY (group_id, user_id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL, message_id TEXT NOT NULL,
                    time INTEGER NOT NULL, payload TEXT NOT NULL,
                    UNIQUE (group_id, message_id)
                );
                CREATE INDEX IF NOT EXISTS messages_group_time
                    ON messages(group_id, time DESC, sequence DESC);
                CREATE TABLE IF NOT EXISTS seen_messages (
                    group_id TEXT NOT NULL, message_id TEXT NOT NULL,
                    PRIMARY KEY (group_id, message_id)
                );
                CREATE TABLE IF NOT EXISTS aliases (
                    group_id TEXT NOT NULL, alias TEXT NOT NULL,
                    source TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (group_id, alias)
                );
                CREATE TABLE IF NOT EXISTS memes (
                    group_id TEXT NOT NULL, name TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (group_id, name)
                );
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_documents (
                    sequence INTEGER PRIMARY KEY, group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL, is_bot INTEGER NOT NULL, tokens TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS memory_documents_group
                    ON memory_documents(group_id,is_bot,user_id);
            """)
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._initialize_search(db)

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    def _initialize_search(self, db) -> None:
        version = db.execute(
            "SELECT value FROM memory_metadata WHERE key='search_version'"
        ).fetchone()
        rebuild = not version or version[0] != str(SEARCH_VERSION)
        if rebuild:
            db.execute("DELETE FROM memory_documents")
        db.execute(
            "DELETE FROM memory_documents WHERE sequence NOT IN (SELECT sequence FROM messages)"
        )
        rows = db.execute(
            """SELECT sequence,group_id,payload FROM messages
                WHERE sequence NOT IN (SELECT sequence FROM memory_documents)"""
        ).fetchall()
        for row in rows:
            self._index_document(db, row["sequence"], row["group_id"], json.loads(row["payload"]))
        try:
            db.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    tokens, group_id UNINDEXED, tokenize='unicode61'
                )"""
            )
            # Querying verifies that a migrated virtual table can actually be
            # opened by this machine's SQLite build, not only that it exists.
            db.execute("SELECT rowid FROM memory_fts LIMIT 1").fetchall()
        except sqlite3.OperationalError as error:
            if "fts5" not in str(error).lower() and "no such module" not in str(error).lower():
                raise
        else:
            self._fts_enabled = True
            if rebuild:
                db.execute("DELETE FROM memory_fts")
            db.execute(
                "DELETE FROM memory_fts WHERE rowid NOT IN (SELECT sequence FROM memory_documents)"
            )
            db.execute(
                """INSERT INTO memory_fts(rowid,tokens,group_id)
                    SELECT sequence,tokens,group_id FROM memory_documents
                    WHERE sequence NOT IN (SELECT rowid FROM memory_fts)"""
            )
        db.execute(
            "INSERT OR REPLACE INTO memory_metadata(key,value) VALUES('search_version',?)",
            (str(SEARCH_VERSION),),
        )

    def _index_document(self, db, sequence: int, group_id: str, message: dict) -> None:
        tokens = " ".join(tokenize(message.get("text", "")))
        db.execute(
            "INSERT OR REPLACE INTO memory_documents VALUES(?,?,?,?,?)",
            (sequence, group_id, message["user_id"], int(message.get("is_bot") is True), tokens),
        )
        if self._fts_enabled:
            db.execute(
                "INSERT INTO memory_fts(rowid,tokens,group_id) VALUES(?,?,?)",
                (sequence, tokens, group_id),
            )

    @contextmanager
    def _connection(self):
        with self._lock:
            db = sqlite3.connect(self.path, timeout=15)
            db.row_factory = sqlite3.Row
            try:
                with db:
                    yield db
            finally:
                db.close()

    @staticmethod
    def _observed_alias(db, group_id: str, alias: str, user_id: str) -> None:
        if not alias:
            return
        row = db.execute(
            "SELECT source, payload FROM aliases WHERE group_id=? AND alias=?",
            (group_id, alias),
        ).fetchone()
        if row and row["source"] == "manual":
            return
        previous = json.loads(row["payload"]) if row and row["source"] == "observed" else {}
        users = sorted(set(previous.get("user_ids", [])) | {user_id})
        payload = {
            "alias": alias,
            "user_ids": users,
            "status": "confirmed" if len(users) == 1 else "ambiguous",
            "source": "observed",
            "evidence_ids": [],
            "explanation": (
                "聊天发送者昵称或群成员名片" if len(users) == 1 else "多位成员使用过同一名称"
            ),
        }
        db.execute(
            "INSERT OR REPLACE INTO aliases(group_id,alias,source,payload) VALUES(?,?,?,?)",
            (group_id, alias, "observed", _json(payload)),
        )

    def _member(
        self, db, group_id: str, user_id: str, names: list[str], *, seen: int | None = None
    ) -> None:
        row = db.execute(
            "SELECT names,last_seen,display_name FROM members WHERE group_id=? AND user_id=?",
            (group_id, user_id),
        ).fetchone()
        known_names = json.loads(row["names"]) if row else []
        # Retain historical nicknames while keeping frequently renamed users bounded.
        known_names = list(dict.fromkeys([*known_names, *names]))[-100:]
        display = names[-1] if names else user_id
        if row and (not names or (seen is not None and seen < row["last_seen"])):
            display = row["display_name"]
        last_seen = max(seen or 0, row["last_seen"] if row else 0)
        db.execute(
            """INSERT INTO members(group_id,user_id,display_name,names,last_seen)
                VALUES(?,?,?,?,?) ON CONFLICT(group_id,user_id) DO UPDATE SET
                display_name=excluded.display_name,names=excluded.names,last_seen=excluded.last_seen
            """,
            (group_id, user_id, display, _json(known_names), last_seen),
        )
        for name in names:
            self._observed_alias(db, group_id, name, user_id)

    def ingest(self, group_id: str, messages: list[dict], *, source: str = "live") -> int:
        group_id = str(group_id)
        inserted = 0
        with self._connection() as db:
            for message in messages:
                if not isinstance(message, dict):
                    continue
                message_id = str(message.get("message_id", "")).strip()
                user_id = str(message.get("user_id", "")).strip()
                if not message_id or not user_id or message_id == "None" or user_id == "None":
                    continue
                if db.execute(
                    "SELECT 1 FROM seen_messages WHERE group_id=? AND message_id=?",
                    (group_id, message_id),
                ).fetchone():
                    continue
                timestamp = message.get("time", 0)
                timestamp = timestamp if isinstance(timestamp, int) else 0
                nickname = _text(message.get("nickname"), 200)
                # Emoji occurrences, unlike names/mentions, must retain repetitions.
                emoji = (
                    [
                        _text(item, 200)
                        for item in message.get("emoji", [])
                        if isinstance(item, str) and _text(item, 200)
                    ]
                    if isinstance(message.get("emoji"), list)
                    else []
                )
                normalized = {
                    "message_id": message_id,
                    "user_id": user_id,
                    "nickname": nickname,
                    "text": _text(message.get("text"), 20000),
                    "time": timestamp,
                    "mentions": _strings(message.get("mentions")),
                    "reply_to": str(message["reply_to"]) if message.get("reply_to") else None,
                    "emoji": emoji[:100],
                    "is_bot": message.get("is_bot") is True,
                    "source": _text(source, 40),
                }
                if not normalized["is_bot"]:
                    self._member(
                        db, group_id, user_id, [nickname] if nickname else [], seen=timestamp
                    )
                    habits = json.loads(
                        db.execute(
                            "SELECT emoji_habits FROM members WHERE group_id=? AND user_id=?",
                            (group_id, user_id),
                        ).fetchone()[0]
                    )
                    counts = Counter(habits)
                    counts.update(normalized["emoji"])
                    db.execute(
                        """UPDATE members SET message_count=message_count+1,emoji_habits=?
                            WHERE group_id=? AND user_id=?""",
                        (_json(dict(counts.most_common(200))), group_id, user_id),
                    )
                db.execute(
                    "INSERT INTO seen_messages(group_id,message_id) VALUES(?,?)",
                    (group_id, message_id),
                )
                cursor = db.execute(
                    "INSERT INTO messages(group_id,message_id,time,payload) VALUES(?,?,?,?)",
                    (group_id, message_id, timestamp, _json(normalized)),
                )
                self._index_document(db, cursor.lastrowid, group_id, normalized)
                inserted += 1
            db.execute(
                """DELETE FROM messages WHERE group_id=? AND sequence NOT IN (
                    SELECT sequence FROM messages WHERE group_id=?
                    ORDER BY time DESC,sequence DESC LIMIT ?
                )""",
                (group_id, group_id, MAX_MESSAGES),
            )
            if self._fts_enabled:
                db.execute(
                    """DELETE FROM memory_fts WHERE group_id=?
                        AND rowid NOT IN (SELECT sequence FROM messages WHERE group_id=?)""",
                    (group_id, group_id),
                )
            db.execute(
                """DELETE FROM memory_documents WHERE group_id=?
                    AND sequence NOT IN (SELECT sequence FROM messages WHERE group_id=?)""",
                (group_id, group_id),
            )
        return inserted

    def mark_message_source(self, group_id: str, message_id: str, source: str) -> None:
        """Mark a semantically recognized tool request without changing its evidence."""
        with self._connection() as db:
            row = db.execute(
                "SELECT payload FROM messages WHERE group_id=? AND message_id=?",
                (str(group_id), str(message_id)),
            ).fetchone()
            if row is None:
                return
            payload = json.loads(row["payload"])
            payload["source"] = _text(source, 40)
            db.execute(
                "UPDATE messages SET payload=? WHERE group_id=? AND message_id=?",
                (json.dumps(payload, ensure_ascii=False), str(group_id), str(message_id)),
            )

    def ingest_members(self, group_id: str, members: list[dict]) -> None:
        group_id = str(group_id)
        with self._connection() as db:
            for member in members:
                if not isinstance(member, dict):
                    continue
                user_id = str(member.get("user_id", "")).strip()
                if not user_id or user_id == "None":
                    continue
                names = _strings([member.get("nickname"), member.get("card")])
                self._member(db, group_id, user_id, names)

    def messages(self, group_id: str, limit: int = 500) -> list[dict]:
        limit = max(0, min(int(limit), MAX_MESSAGES))
        with self._connection() as db:
            rows = db.execute(
                """SELECT payload FROM messages WHERE group_id=?
                    ORDER BY time DESC,sequence DESC LIMIT ?""",
                (str(group_id), limit),
            ).fetchall()
        return [json.loads(row["payload"]) for row in reversed(rows)]

    def search(
        self,
        group_id: str,
        query: str,
        *,
        user_ids: list[str] | None = None,
        limit: int = 6,
        exclude_message_ids: list[str] | None = None,
    ) -> list[dict]:
        """Retrieve bounded, group-local human evidence using lexical relevance.

        ``user_ids`` boosts active speakers; it never makes unrelated messages
        match. Explicit, unambiguous aliases can also retrieve their author's
        recent messages when the nickname is absent from the message body.
        """
        group_id = str(group_id)
        query = _text(query, 2000)
        limit = max(0, min(int(limit), 12))
        if not query or not limit:
            return []
        excluded = set(_strings(exclude_message_ids or [], 1000))
        preferred = set(_strings(user_ids or [], 100))
        terms = query_terms(query)
        named_users = set()
        with self._connection() as db:
            aliases = [
                json.loads(row[0])
                for row in db.execute("SELECT payload FROM aliases WHERE group_id=?", (group_id,))
            ]
            confirmed = [
                alias
                for alias in aliases
                if alias.get("status") == "confirmed" and len(alias.get("user_ids", [])) == 1
            ]
            for alias in confirmed:
                if alias_in_text(alias["alias"], query):
                    named_users.update(alias["user_ids"])
            preferred.update(named_users)
            for alias in confirmed:
                if named_users.intersection(alias["user_ids"]):
                    terms.extend(query_terms(alias["alias"]))
            terms = list(dict.fromkeys(terms))[:128]
            scores = {}
            if terms and self._fts_enabled:
                rows = db.execute(
                    """SELECT memory_fts.rowid AS sequence,bm25(memory_fts) AS rank
                        FROM memory_fts JOIN memory_documents d ON d.sequence=memory_fts.rowid
                        WHERE memory_fts MATCH ? AND memory_fts.group_id=? AND d.is_bot=0
                        ORDER BY rank LIMIT ?""",
                    (match_expression(terms), group_id, limit * 20),
                ).fetchall()
                scores = {row["sequence"]: -row["rank"] for row in rows}
            elif terms:
                documents = [
                    (row[0], row[1])
                    for row in db.execute(
                        """SELECT sequence,tokens FROM memory_documents
                            WHERE group_id=? AND is_bot=0""",
                        (group_id,),
                    )
                ]
                scores = dict(bm25_fallback(documents, terms)[: limit * 20])
            if named_users:
                placeholders = ",".join("?" for _ in named_users)
                rows = db.execute(
                    f"""SELECT d.sequence FROM memory_documents d
                        JOIN messages m ON m.sequence=d.sequence
                        WHERE d.group_id=? AND d.is_bot=0 AND d.user_id IN ({placeholders})
                        ORDER BY m.time DESC,m.sequence DESC LIMIT ?""",
                    (group_id, *sorted(named_users), limit * 3),
                ).fetchall()
                for row in rows:
                    scores.setdefault(row[0], 0.0)
            if not scores:
                return []
            placeholders = ",".join("?" for _ in scores)
            rows = db.execute(
                f"""SELECT sequence,payload FROM messages
                    WHERE group_id=? AND sequence IN ({placeholders})""",
                (group_id, *scores),
            ).fetchall()
        candidates = []
        for row in rows:
            message = json.loads(row["payload"])
            if (
                message["message_id"] in excluded
                or message["is_bot"]
                or message.get("source") == "jev_test"
                or not message["text"]
            ):
                continue
            score = scores[row["sequence"]]
            if message["user_id"] in preferred:
                score *= 1.3
            candidates.append(
                {
                    "message_id": message["message_id"],
                    "user_id": message["user_id"],
                    "nickname": message["nickname"][:80],
                    "text": excerpt(message["text"], terms),
                    "time": message["time"],
                    "score": round(score, 10),
                }
            )
        candidates.sort(key=lambda item: (item["score"], item["time"]), reverse=True)
        result = []
        seen_text = set()
        for item in candidates:
            normalized = re.sub(r"\s+", "", item["text"]).casefold()
            if normalized in seen_text:
                continue
            seen_text.add(normalized)
            result.append(item)
            if len(result) >= limit:
                break
        return result

    def backup_to(self, path: Path) -> Path:
        """Create a consistent private snapshot, including the schema and indexes.

        Exclusive creation protects an existing backup from accidental replacement.
        SQLite's backup API remains valid while other connections receive messages.
        """
        path = Path(path)
        if path.resolve() == self.path.resolve():
            raise ValueError("backup destination must differ from the active database")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(descriptor)
        try:
            with self._connection() as source:
                with closing(sqlite3.connect(path)) as destination:
                    source.backup(destination)
                    if destination.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        raise ValueError("database backup failed its integrity check")
            os.chmod(path, 0o600)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path

    def overview(self, group_id: str) -> dict:
        group_id = str(group_id)
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM members WHERE group_id=? ORDER BY message_count DESC,user_id",
                (group_id,),
            ).fetchall()
            members = [
                {
                    "user_id": row["user_id"],
                    "display_name": row["display_name"],
                    "names": json.loads(row["names"]),
                    "message_count": row["message_count"],
                    "emoji_habits": [
                        {"emoji": emoji, "count": count}
                        for emoji, count in sorted(
                            json.loads(row["emoji_habits"]).items(),
                            key=lambda pair: (-pair[1], pair[0]),
                        )
                    ],
                    "profile": json.loads(row["profile"]) if row["profile"] else None,
                }
                for row in rows
            ]
            aliases = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT payload FROM aliases WHERE group_id=? ORDER BY alias", (group_id,)
                )
            ]
            memes = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT payload FROM memes WHERE group_id=? ORDER BY name", (group_id,)
                )
            ]
            evidence_ids = set()
            for member in members:
                for fact in (member["profile"] or {}).get("facts", []):
                    evidence_ids.update(fact.get("evidence_ids", []))
            for item in [*aliases, *memes]:
                evidence_ids.update(item.get("evidence_ids", []))
            recent = db.execute(
                "SELECT message_id,payload FROM messages WHERE group_id=? ORDER BY time DESC",
                (group_id,),
            )
            evidence = []
            for row in recent:
                if row["message_id"] in evidence_ids:
                    evidence.append(json.loads(row["payload"]))
                    if len(evidence) == 200:
                        break
            retained = db.execute(
                "SELECT COUNT(*) FROM messages WHERE group_id=?", (group_id,)
            ).fetchone()[0]
            total = db.execute(
                "SELECT COUNT(*) FROM seen_messages WHERE group_id=?", (group_id,)
            ).fetchone()[0]
        return {
            "group_id": group_id,
            "stats": {
                "messages": total,
                "human_messages": sum(member["message_count"] for member in members),
                "retained_messages": retained,
                "members": len(members),
                "profiled_members": sum(member["profile"] is not None for member in members),
            },
            "members": members,
            "aliases": aliases,
            "memes": memes,
            "evidence": evidence,
        }

    def save_analysis(self, group_id: str, analysis: dict) -> None:
        """Merge grounded findings; model output cannot create unknown members."""
        group_id = str(group_id)
        with self._connection() as db:
            known = {
                row[0]
                for row in db.execute("SELECT user_id FROM members WHERE group_id=?", (group_id,))
            }
            evidence = {
                row[0]
                for row in db.execute(
                    "SELECT message_id FROM messages WHERE group_id=?", (group_id,)
                )
            }

            def evidence_for(item):
                return [key for key in _strings(item.get("evidence_ids"), 30) if key in evidence]

            for member in analysis.get("members", []):
                if not isinstance(member, dict) or str(member.get("user_id")) not in known:
                    continue
                user_id = str(member["user_id"])
                facts = []
                member_facts = member.get("facts", [])
                for fact in member_facts if isinstance(member_facts, list) else []:
                    if not isinstance(fact, dict):
                        continue
                    ids = evidence_for(fact)
                    content = _text(fact.get("text"))
                    if ids and content:
                        facts.append(
                            {
                                "category": _text(fact.get("category"), 80),
                                "text": content,
                                "evidence_ids": ids,
                                "confidence": "observed"
                                if fact.get("confidence") == "observed"
                                else "likely",
                            }
                        )
                if not facts:
                    continue
                row = db.execute(
                    "SELECT profile FROM members WHERE group_id=? AND user_id=?",
                    (group_id, user_id),
                ).fetchone()
                previous = json.loads(row[0]) if row[0] else {}
                combined = {}
                for fact in [*previous.get("facts", []), *facts]:
                    key = (fact["category"], fact["text"])
                    old = combined.get(key, {})
                    fact["evidence_ids"] = _strings(
                        [*old.get("evidence_ids", []), *fact["evidence_ids"]], 30
                    )
                    combined[key] = fact
                profile = {
                    "user_id": user_id,
                    "summary": _text(member.get("summary")) or previous.get("summary", ""),
                    "facts": list(combined.values())[-60:],
                }
                db.execute(
                    "UPDATE members SET profile=? WHERE group_id=? AND user_id=?",
                    (_json(profile), group_id, user_id),
                )

            for item in analysis.get("aliases", []):
                if not isinstance(item, dict):
                    continue
                alias = _text(item.get("alias"), 200)
                users = [uid for uid in _strings(item.get("user_ids")) if uid in known]
                ids = evidence_for(item)
                if not alias or not users or not ids:
                    continue
                row = db.execute(
                    "SELECT source,payload FROM aliases WHERE group_id=? AND alias=?",
                    (group_id, alias),
                ).fetchone()
                if row and row["source"] in {"manual", "observed"}:
                    continue
                previous = json.loads(row["payload"]) if row else {}
                users = _strings([*previous.get("user_ids", []), *users])
                payload = {
                    "alias": alias,
                    "user_ids": users,
                    "status": "ambiguous"
                    if len(users) > 1 or item.get("status") == "ambiguous"
                    else "likely",
                    "source": "analysis",
                    "evidence_ids": _strings([*previous.get("evidence_ids", []), *ids], 30),
                    "explanation": _text(item.get("explanation")),
                }
                db.execute(
                    "INSERT OR REPLACE INTO aliases(group_id,alias,source,payload) VALUES(?,?,?,?)",
                    (group_id, alias, "analysis", _json(payload)),
                )

            for item in analysis.get("memes", []):
                if not isinstance(item, dict):
                    continue
                name = _text(item.get("name"), 200)
                meaning = _text(item.get("meaning"))
                ids = evidence_for(item)
                if not name or not meaning or not ids:
                    continue
                row = db.execute(
                    "SELECT payload FROM memes WHERE group_id=? AND name=?", (group_id, name)
                ).fetchone()
                previous = json.loads(row[0]) if row else {}
                users = [uid for uid in _strings(item.get("user_ids")) if uid in known]
                payload = {
                    "name": name,
                    "meaning": meaning,
                    "user_ids": _strings([*previous.get("user_ids", []), *users]),
                    "evidence_ids": _strings([*previous.get("evidence_ids", []), *ids], 30),
                }
                db.execute(
                    "INSERT OR REPLACE INTO memes(group_id,name,payload) VALUES(?,?,?)",
                    (group_id, name, _json(payload)),
                )

    def set_alias(
        self,
        group_id: str,
        alias: str,
        user_ids: list[str],
        *,
        explanation: str = "由控制台手动确认",
    ) -> None:
        group_id = str(group_id)
        alias = _text(alias, 200)
        users = _strings(user_ids)
        if not alias or not users:
            raise ValueError("alias and at least one member are required")
        with self._connection() as db:
            known = {
                row[0]
                for row in db.execute("SELECT user_id FROM members WHERE group_id=?", (group_id,))
            }
            if not set(users) <= known:
                raise ValueError("alias refers to an unknown group member")
            payload = {
                "alias": alias,
                "user_ids": users,
                "status": "confirmed",
                "source": "manual",
                "evidence_ids": [],
                "explanation": _text(explanation, 1000) or "由控制台手动确认",
            }
            db.execute(
                "INSERT OR REPLACE INTO aliases(group_id,alias,source,payload) VALUES(?,?,?,?)",
                (group_id, alias, "manual", _json(payload)),
            )

    def prompt_context(
        self, group_id: str, user_ids: list[str], text: str, max_chars: int = 10000
    ) -> str:
        """Return relevant memory as valid JSON, never slice a serialized document."""
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 2:
            raise ValueError("max_chars must be at least 2")
        overview = self.overview(str(group_id))
        # The controller supplies speakers in chronological order. Keep the latest
        # participants first, then add people explicitly referred to by a nickname.
        selected_ids = _strings(list(reversed(user_ids)))
        matching_aliases = [
            item for item in overview["aliases"] if alias_in_text(item["alias"], text)
        ]
        # Speaker labels identify people, not the current topic. A nickname in a
        # stored joke's explanation must not revive that joke on every message.
        topic_text = re.sub(r"(?m)^\[[^\n]*?/\d+\]\s*", "", text)
        matching_memes = [
            item for item in overview["memes"] if alias_in_text(item["name"], topic_text)
        ]
        for item in matching_aliases:
            selected_ids = _strings([*selected_ids, *item["user_ids"]])
        by_id = {member["user_id"]: member for member in overview["members"]}
        members = [by_id[uid] for uid in selected_ids if uid in by_id]
        if not selected_ids:
            members = overview["members"][:6]
        selected = {member["user_id"] for member in members}
        person_terms = {
            term
            for alias in matching_aliases
            if alias_in_text(alias["alias"], topic_text)
            for term in query_terms(alias["alias"])
        }
        topic_terms = (
            {term for term in query_terms(topic_text) if len(term) >= 2 and not term.isdigit()}
            - person_terms
            - {"今天", "昨天", "明天", "刚才", "大家", "群友", "聊天", "时候"}
        )

        def meme_relevance(meme: dict) -> int:
            memory_terms = set(query_terms(meme["name"] + " " + meme["meaning"]))
            return len(topic_terms & memory_terms)

        relevant_memes = sorted(
            (
                meme
                for meme in overview["memes"]
                if meme not in matching_memes and meme_relevance(meme) > 0
            ),
            key=lambda meme: (meme_relevance(meme), bool(set(meme["user_ids"]) & selected)),
            reverse=True,
        )
        result = {"group_id": str(group_id), "members": [], "aliases": [], "memes": []}
        if len(_json(result)) > max_chars:
            return "{}"

        def append_if_fits(field: str, item: dict) -> bool:
            result[field].append(item)
            if len(_json(result)) <= max_chars:
                return True
            result[field].pop()
            return False

        # Reserve most of the document for people. Human-confirmed identities go
        # ahead of old model guesses and observed nicknames, even when a detailed
        # old memory would otherwise consume the entire shared memory budget.
        memory_budget = int(max_chars * 0.4)
        memory_used = 0

        def append_memory(field: str, item: dict) -> None:
            nonlocal memory_used
            item = {
                key: value[:800] if isinstance(value, str) else value for key, value in item.items()
            }
            if field == "aliases":
                item["explanation"] = item.get("explanation", "")[:200]
            item["evidence_ids"] = item.get("evidence_ids", [])[:3]
            size = len(_json(item)) + 1
            if memory_used + size <= memory_budget and append_if_fits(field, item):
                memory_used += size

        manual_aliases = [
            alias
            for alias in overview["aliases"]
            if alias.get("source") == "manual"
            and alias.get("status") == "confirmed"
            and (alias in matching_aliases or set(alias["user_ids"]) & selected)
        ]
        manual_aliases.sort(key=lambda alias: alias not in matching_aliases)
        for alias in manual_aliases:
            append_memory("aliases", alias)
        for alias in matching_aliases:
            if alias not in manual_aliases:
                append_memory("aliases", alias)
        for meme in matching_memes:
            append_memory("memes", meme)

        included = []
        for member in members:
            item = {
                **member,
                "display_name": member["display_name"][:80],
                "names": [name[:64] for name in member["names"][-8:]],
                "emoji_habits": member["emoji_habits"][:5],
            }
            profile = member["profile"]
            if profile:
                item["profile"] = {**profile, "summary": profile["summary"][:180], "facts": []}
            if not append_if_fits("members", item):
                item["names"] = item["names"][-1:]
                item["emoji_habits"] = []
                if item["profile"]:
                    item["profile"]["summary"] = item["profile"]["summary"][:60]
                if not append_if_fits("members", item):
                    continue
            included.append((member, item))

        for alias in overview["aliases"]:
            if (
                alias not in matching_aliases
                and alias not in manual_aliases
                and set(alias["user_ids"]) & selected
            ):
                append_memory("aliases", alias)
        # Reuse a callback when its wording or meaning fits this conversation.
        # Being associated with the same person is only a relevance tie-breaker.
        for meme in relevant_memes:
            append_memory("memes", meme)

        # Chinese has no word separators. Short shared phrases make topic ranking
        # useful for both Chinese chats and literal English/game-name catchphrases.
        terms = set()
        for token in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9_]{2,}", text[-2000:].casefold()):
            if "\u4e00" <= token[0] <= "\u9fff":
                terms.update(token[index : index + 2] for index in range(len(token) - 1))
            else:
                terms.add(token)

        def relevance(fact: dict) -> int:
            content = fact["text"].casefold()
            return sum(term in content for term in terms)

        ranked = []
        for original, item in included:
            if not original["profile"]:
                continue
            recent_first = list(reversed(original["profile"]["facts"]))
            facts = sorted(recent_first, key=relevance, reverse=True)[:10]
            ranked.append((item["profile"], facts))
        remaining = max_chars - len(_json(result))
        fact_text_limit = min(800, max(80, remaining // max(1, len(ranked) * 3) - 150))
        # Round-robin allocation gives recent participants equal opportunities to
        # contribute relevant facts instead of the first profile consuming all space.
        for index in range(10):
            for profile, facts in ranked:
                if index >= len(facts):
                    continue
                fact = {**facts[index], "evidence_ids": facts[index]["evidence_ids"][:3]}
                content = fact["text"]
                fact["text"] = content[:fact_text_limit]
                if len(content) > fact_text_limit:
                    fact["text"] += "…"
                profile["facts"].append(fact)
                if len(_json(result)) > max_chars:
                    profile["facts"].pop()
        return _json(result)

    def close(self) -> None:
        """Connections are short lived; this method supports owner cleanup hooks."""
