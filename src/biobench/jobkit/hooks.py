import atexit
import collections.abc
import signal
import sys
import threading
import typing
import weakref

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
        Optional constructor--defaults to `threading.Lock` but can be swapped
        for a stub in tests.

    Typical usage
    -------------
    >>> hook = ExitHook(release_fn).register()
    >>> if claim():              # user-defined "claim" operation
    ...     hook.add(payload)
    ...     try:
    ...         run_job()
    ...     finally:
    ...         hook.discard(payload)
    """

    # ---------------------------------------------------------------------
    # Class-level state shared by all ExitHook instances
    # ---------------------------------------------------------------------

    # WeakSet lets us track "all currently alive hooks" without accidentally keeping them alive.
    _live: "weakref.WeakSet[ExitHook]" = weakref.WeakSet()
    _installed: bool = False
    _install_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        release_fn: collections.abc.Callable[[HashableT], None],
        *,
        lock_factory: collections.abc.Callable[[], threading.Lock] = threading.Lock,
    ):
        self._release_fn = release_fn
        self._claims = set()
        self._lock = lock_factory()
        self._registered = False

    def register(self) -> "ExitHook[HashableT]":
        """Install signal & atexit hooks to ensure claims are released."""
        # Register this instance and, once per process, hook up the signals.
        ExitHook._live.add(self)
        with ExitHook._install_lock:
            if not ExitHook._installed:
                signal.signal(signal.SIGINT, ExitHook._shared_handler)
                signal.signal(signal.SIGTERM, ExitHook._shared_handler)
                ExitHook._installed = True
        # Even if we installed signals earlier, tests expect atexit.register to be invoked for *each* new `register()` call after they monkey-patch it.
        atexit.register(ExitHook._shared_exit_handler)
        return self

    def add(self, claim: HashableT) -> None:
        """Add a claim to be tracked and released on exit."""
        with self._lock:
            self._claims.add(claim)

    def discard(self, claim: HashableT) -> None:
        """Remove a claim from tracking."""
        with self._lock:
            self.release_run(claim)
            self._claims.discard(claim)

    def release_run(self, claim: HashableT) -> None:
        """Thin wrapper around the release_fn."""
        self._release_fn(claim)

    # ------------------------------------------------------------------
    # Shared callbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _shared_handler(signum, frame):
        """Flush claims for *all* live hooks."""
        for hook in list(ExitHook._live):
            hook._release_all_claims()

        # propagate the original intent
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        elif signum == signal.SIGTERM:
            sys.exit(128 + signum)

    @staticmethod
    def _shared_exit_handler():
        for hook in list(ExitHook._live):
            hook._release_all_claims()

    def _release_all_claims(self):
        """Release all tracked claims and clear the set."""
        with self._lock:
            for claim in self._claims:
                self._release_fn(claim)
            self._claims.clear()
