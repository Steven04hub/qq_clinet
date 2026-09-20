import pytest

from app.services.dedup import MessageDeduplicator


def test_checks_and_remembers_message_atomically():
    dedup = MessageDeduplicator()
    assert not dedup.seen(123)
    assert dedup.seen(123)
    assert dedup.seen(123)
    assert not dedup.seen(456)


def test_group_message_pairs_keep_identical_ids_in_different_groups_separate():
    dedup = MessageDeduplicator()
    assert not dedup.seen((100, 10))
    assert not dedup.seen((200, 10))
    assert dedup.seen((100, 10))


def test_bounded_dedup_evicts_oldest_first_seen_id():
    dedup = MessageDeduplicator(max_size=2)
    assert not dedup.seen("first")
    assert not dedup.seen("second")
    assert dedup.seen("first")
    assert not dedup.seen("third")
    assert dedup.seen("second")
    assert not dedup.seen("first")


def test_single_entry_capacity():
    dedup = MessageDeduplicator(max_size=1)
    assert not dedup.seen(1)
    assert dedup.seen(1)
    assert not dedup.seen(2)
    assert not dedup.seen(1)


@pytest.mark.parametrize("max_size", [0, -1, 1.5, True])
def test_invalid_capacity_is_rejected(max_size):
    with pytest.raises(ValueError):
        MessageDeduplicator(max_size)
