import atexit
import collections.abc
import signal
import sys
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
        Optional constructor--defaults to `threading.Lock` but can be swapped
        for a stub in tests.

    Typical usage
    -------------
    >>> reaper = ExitHook(release_fn).register()
    >>> if claim_row():              # user-defined "claim" operation
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
    ):
        self._release_fn = release_fn
        self._claims = set()
        self._lock = lock_factory()
        self._registered = False

    def register(self) -> "ExitHook[HashableT]":
        """Install signal & atexit hooks to ensure claims are released."""
        if not self._registered:
            # Register signal handlers
            self._old_sigint = signal.signal(signal.SIGINT, self._signal_handler)
            self._old_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Register atexit hook
            atexit.register(self._exit_handler)
            
            self._registered = True
        return self

    def add(self, claim: HashableT) -> None:
        """Add a claim to be tracked and released on exit."""
        with self._lock:
            self._claims.add(claim)

    def discard(self, claim: HashableT) -> None:
        """Remove a claim from tracking."""
        with self._lock:
            self._claims.discard(claim)

    def release_run(self, claim: HashableT) -> None:
        """Thin wrapper around the release_fn."""
        self._release_fn(claim)
        
    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM by releasing all claims and re-raising."""
        self._release_all_claims()
        
        # Re-raise the original signal
        if signum == signal.SIGINT:
            signal.signal(signal.SIGINT, self._old_sigint)
            raise KeyboardInterrupt
        elif signum == signal.SIGTERM:
            signal.signal(signal.SIGTERM, self._old_sigterm)
            sys.exit(128 + signum)

    def _exit_handler(self):
        """Handle normal interpreter shutdown."""
        self._release_all_claims()

    def _release_all_claims(self):
        """Release all tracked claims and clear the set."""
        with self._lock:
            for claim in self._claims:
                self._release_fn(claim)
            self._claims.clear()
