"""Bounded, in-memory message deduplication for a single asyncio event loop."""

from collections import OrderedDict
from collections.abc import Hashable


class MessageDeduplicator:
    """Atomically check and remember identifiers, evicting oldest first-seen IDs."""

    def __init__(self, max_size: int = 1000) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
            raise ValueError("max_size must be a positive integer")
        self.max_size = max_size
        self._seen: OrderedDict[Hashable, None] = OrderedDict()

    def seen(self, message_id: Hashable) -> bool:
        # No await: checking and remembering are indivisible in the event loop.
        if message_id in self._seen:
            return True
        self._seen[message_id] = None
        if len(self._seen) > self.max_size:
            self._seen.popitem(last=False)
        return False
