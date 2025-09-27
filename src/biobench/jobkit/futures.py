import logging
import time
import typing

import beartype

T = typing.TypeVar("T")

logger = logging.getLogger(__name__)


@beartype.beartype
class FutureQueue(typing.Generic[T]):
    def __init__(self, max_size: int):
        """Create queue. max_size >= 0; 0 => always full."""
        self._max_size = max_size
        self._items = []  # FIFO queue of items

    def submit(self, item: T) -> None:
        """RuntimeError if full()."""
        if self.full():
            raise RuntimeError(f"Queue is full (max_size={self._max_size})")
        self._items.append(item)

    def pop(self) -> T:
        """Block until *some* contained Job is done, remove and return its payload."""
        if not self._items:
            return None

        # First, check if any job is already done (non-blocking)
        for i, item in enumerate(self._items):
            if self._is_done(item):
                return self._items.pop(i)

        # If no job is done, wait for the first one to complete
        while self._items:
            for i, item in enumerate(self._items):
                if self._is_done(item):
                    return self._items.pop(i)
            # No job is done yet, sleep briefly before checking again
            time.sleep(0.1)

        return None

    def _is_obj_done(self, obj) -> bool:
        # Direct check for objects with done() method
        if hasattr(obj, "done") and callable(obj.done):
            try:
                # done() may raise when the job failed. Swallow it.
                return obj.done()  # True => finished (success or failed)
            except Exception:
                # Treat "raises inside done()" as "finished but failed".
                logger.exception("obj.done() failed")
                return True

        return False

    def _is_done(self, obj) -> bool:
        """Check if an object or any of its nested items is done."""
        if self._is_obj_done(obj):
            return True

        # Check first level of nesting (tuples, lists, dicts)
        if isinstance(obj, (tuple, list)):
            for item in obj:
                if self._is_obj_done(item):
                    return True

                # Check second level of nesting
                if isinstance(item, (tuple, list)):
                    for subitem in item:
                        if self._is_obj_done(subitem):
                            return True
                elif isinstance(item, dict):
                    for subitem in item.values():
                        if self._is_obj_done(subitem):
                            return True
        elif isinstance(obj, dict):
            for item in obj.values():
                if self._is_obj_done(item):
                    return True

                # Check second level of nesting
                if isinstance(item, (tuple, list)):
                    for subitem in item:
                        if self._is_obj_done(subitem):
                            return True
                elif isinstance(item, dict):
                    for subitem in item.values():
                        if self._is_obj_done(subitem):
                            return True

        return False

    def full(self) -> bool:
        """Return True if the queue is at capacity."""
        if self._max_size == 0:
            return True
        return len(self._items) >= self._max_size

    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return len(self._items)

    def __bool__(self) -> bool:
        """Return True if the queue is not empty."""
        return bool(self._items)

    def __iter__(self):
        """Iterate over items in FIFO order."""
        return iter(self._items)
