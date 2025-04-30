import collections.abc
import threading
import typing

import beartype

HashableT = typing.TypeVar("HashableT", bound=collections.abc.Hashable)


@beartype.beartype
class ExitHook(typing.Generic[HashableT]):
    """
    Keep track of outstanding *claims* and make sure each one is released
    on SIGINT/SIGTERM or normal interpreter shutdown.

    Parameters
    ----------
    release_fn
        Callback that frees a single claim (e.g. lambda c: reporting.release_run(db, *c)).
    lock_factory
        Optional constructor—defaults to `threading.Lock` but can be swapped
        for a stub in tests.

    Typical usage
    -------------
    >>> reaper = ExitHook(release_fn).register()
    >>> if claim_row():              # user-defined “claim” operation
    ...     reaper.add(payload)
    ...     try:
    ...         run_job()
    ...     finally:
    ...         reaper.discard(payload)
    """

    def __init__(
        self,
        release_fn: collections.abc.Callable[[HashableT], None],
        *,
        lock_factory: collections.abc.Callable[[], threading.Lock] = threading.Lock,
    ): ...

    def register(self) -> "ExitHook[HashableT]":
        return self

    def add(self, claim: HashableT) -> None: ...
    def discard(self, claim: HashableT) -> None: ...
    def release_run(self, claim: HashableT) -> None: ...
