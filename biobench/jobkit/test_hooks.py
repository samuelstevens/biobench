import atexit
import signal
import threading
import time

import beartype
import pytest

from .hooks import ExitHook


def _invoke_handler(handler, signum):
    """
    Call the given *handler* as the signal machinery would:
    - pass (signum, current-frame)   (frame may be None for our purposes)
    - swallow KeyboardInterrupt / SystemExit so the suite keeps running.
    """
    try:
        handler(signum, None)
    except (KeyboardInterrupt, SystemExit):
        pass


def test_register_returns_self_and_is_idempotent():
    """
    *Spec:* `register()` **must** return the same object so callers can write
    `reaper = ClaimReaper(...).register()`.

    It must also be safe to call repeatedly (harmless no-ops) because some launchers may defensively register twice.
    """

    sink = []
    hook = ExitHook(lambda claim: sink.append(claim))

    assert hook.register() is hook  # first call
    assert hook.register() is hook  # second call - still fine


def test_add_then_discard_accepts_generic_payload():
    """
    With the new signature `add(claim)`, the class must accept *any* hashable
    payload the caller chooses.  The easiest non-trivial smoke-test is a tuple.
    """

    hook = ExitHook(release_fn=lambda c: None).register()

    claim = ("cfg-xyz", "task-abc")
    hook.add(claim)  # should not raise
    hook.discard(claim)  # should not raise


def test_multiple_outstanding_claims_do_not_interfere():
    """
    Adding several distinct claims before discarding them should not trigger
    internal errors such as "set changed size during iteration".
    """

    hook = ExitHook(lambda c: None).register()
    claims = [f"job-{i}" for i in range(5)]

    for c in claims:
        hook.add(c)

    for c in claims:
        hook.discard(c)


def test_release_run_calls_injected_release_fn():
    """
    `release_run()` is documented as a *thin wrapper* that forwards to the
    injected `release_fn`.  Verify that the exact same claim object reaches the
    callback once--and only once.
    """

    hits = []
    hook = ExitHook(lambda claim: hits.append(claim))

    claim = ("id", 7)
    hook.release_run(claim)

    assert hits == [claim], "release_fn should have been invoked exactly once"


def test_add_requires_hashable_claim():
    """
    Non-hashable objects cannot be stored in the internal set that tracks
    outstanding claims.  A correct implementation therefore raises TypeError.
    """

    hook = ExitHook(lambda c: None)

    # list is non-hashable -> should blow up
    with pytest.raises((TypeError, beartype.roar.BeartypeCallHintParamViolation)):
        hook.add(["unhashable"])


def test_release_run_invokes_callback_exactly_each_time():
    """
    *Behavioural guarantee:* every explicit release_run() call must translate
    into one--and only one--invocation of the injected release_fn, regardless of
    whether the claim was previously added/discarded.
    """

    hits = []
    hook = ExitHook(lambda c: hits.append(c)).register()

    claim = ("cfg-id-123", "task-foo")

    # Claim is active
    hook.add(claim)
    hook.release_run(claim)
    assert hits == [claim]  # called once

    # Claim is no longer tracked
    hook.discard(claim)
    hook.release_run(claim)
    assert hits == [claim, claim]  # called again, no extras


def test_sigint_releases_all_claims():
    hits = []
    claims = [f"claim-{i}" for i in range(3)]
    hook = ExitHook(lambda c: hits.append(c)).register()

    for c in claims:
        hook.add(c)

    old_handler = signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, old_handler)  # restore for other tests

    assert sorted(hits) == sorted(claims)
    assert len(hits) == len(claims)


def test_sigint_handler_releases_all_claims():
    """
    Core guarantee: when the SIGINT handler is invoked, every *current* claim
    is released exactly once.

    We call the handler directly rather than raising a real signal so the
    suite survives even if ClaimReaper forgot to install it.
    """

    hits, claims = [], [f"claim-{i}" for i in range(3)]
    hook = ExitHook(lambda c: hits.append(c)).register()
    for c in claims:
        hook.add(c)

    handler = signal.getsignal(signal.SIGINT)
    assert callable(handler) and handler not in (signal.SIG_DFL, signal.SIG_IGN), (
        "ClaimReaper failed to install a SIGINT handler"
    )

    _invoke_handler(handler, signal.SIGINT)

    assert sorted(hits) == sorted(claims)
    assert len(hits) == len(claims)


def test_duplicate_add_is_idempotent_under_sigint():
    """
    Adding the same claim twice must not produce duplicate releases when the
    SIGINT handler fires.
    """

    hits, claim = [], ("cfg-42", "task-alpha")
    hook = ExitHook(lambda c: hits.append(c)).register()
    hook.add(claim)
    hook.add(claim)  # duplicate

    handler = signal.getsignal(signal.SIGINT)
    assert callable(handler), "missing SIGINT handler"
    _invoke_handler(handler, signal.SIGINT)

    assert hits == [claim], "duplicate add produced duplicate release"


def test_sigterm_handler_releases_only_current_claims():
    """
    Discarded claims should not be released by the SIGTERM handler.
    """

    hits = []
    live1, live2, discarded = "stay-1", "stay-2", "gone-x"
    hook = ExitHook(lambda c: hits.append(c)).register()

    for c in (live1, live2, discarded):
        hook.add(c)
    hook.discard(discarded)  # no longer live

    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler), "missing SIGTERM handler"
    _invoke_handler(handler, signal.SIGTERM)

    assert sorted(hits) == sorted([live1, live2])


def test_discard_unknown_claim_is_noop():
    """
    Discarding a claim that was never added should *not* raise.  This keeps
    launcher code simple because it can unconditionally discard in finally-
    blocks without first checking membership.
    """

    hook = ExitHook(lambda c: None).register()

    # Should silently ignore
    hook.discard("non-existent")


def test_register_calls_atexit(monkeypatch):
    """
    *Contract*: `register()` must install an atexit hook so that claims are released on normal interpreter shutdown.

    We patch `atexit.register` to capture the callback and assert that:
    1) it was invoked exactly once; 2) the registered object is callable.
    """

    captured: list[callable] = []

    def _fake_register(fn, *args, **kwargs):
        captured.append(fn)

    monkeypatch.setattr(atexit, "register", _fake_register)

    # construction shouldn't trigger the hook -- only .register()
    hook = ExitHook(lambda _: None)
    assert not captured

    hook.register()
    assert len(captured) == 1
    assert callable(captured[0])


def test_atexit_handler_releases_all_live_claims(monkeypatch):
    """
    Verify that the function registered with `atexit` releases *every* claim
    that is still active when the interpreter would exit.

    Strategy
    --------
    * Capture the cleanup callback via a patched `atexit.register`.
    * Add several claims.
    * Manually invoke the captured callback.
    * Confirm `release_fn` was called once per tracked claim (order irrelevant).
    """

    cleanup_fns = []

    def _capture(fn, *a, **kw):
        cleanup_fns.append(fn)

    monkeypatch.setattr(atexit, "register", _capture)

    hits = []
    hook = ExitHook(lambda c: hits.append(c)).register()
    claims = [("cfg-0", "task-0"), ("cfg-1", "task-1"), ("cfg-2", "task-2")]
    for claim in claims:
        hook.add(claim)

    # Simulate interpreter shutdown
    assert cleanup_fns, "No atexit hook registered"
    cleanup_fns[0]()  # invoke captured function

    assert sorted(hits) == sorted(claims)
    assert len(hits) == len(claims)


def test_lock_prevents_set_mutation_during_massive_adds():
    """
    A classic failure mode is "set changed size during iteration" when the
    signal-handler walks `_claims` while another thread is adding claims.

    Strategy
    --------
    * Worker thread continuously adds new claims.
    * Main thread waits a short moment, then calls the SIGINT handler.
    * If locking is absent we'll almost certainly trigger the RuntimeError.
    * We also check that each claim is released **at most once**.
    """

    hits: list[str] = []
    hook = ExitHook(lambda c: hits.append(c)).register()

    stop = threading.Event()

    def _producer():
        i = 0
        while not stop.is_set():
            hook.add(f"claim-{i}")
            i += 1

    t = threading.Thread(target=_producer)
    t.start()

    # Give producer a head-start so the set is being modified.
    time.sleep(0.05)

    handler = signal.getsignal(signal.SIGINT)
    _invoke_handler(handler, signal.SIGINT)

    stop.set()
    t.join()

    # No duplicates => each claim released only once (lock prevented races)
    assert len(hits) == len(set(hits))


def test_lock_prevents_set_mutation_during_discards():
    """
    Another race: thread discarding while handler iterates.
    """

    hits: list[str] = []
    hook = ExitHook(lambda c: hits.append(c)).register()

    # Pre-populate many claims
    claims = [f"c{i}" for i in range(250)]
    for c in claims:
        hook.add(c)

    def _consumer():
        for c in claims:
            hook.discard(c)
            time.sleep(0.0005)  # keep the race window open

    t = threading.Thread(target=_consumer)
    t.start()

    # Let the consumer start discarding, then fire the handler
    time.sleep(0.02)
    handler = signal.getsignal(signal.SIGINT)
    _invoke_handler(handler, signal.SIGINT)

    t.join()

    # All *remaining* live claims were released once; any discarded before
    # the handler shouldn't re-appear, so no duplicates.
    assert len(hits) == len(set(hits))


def test_lock_serialises_multiple_concurrent_handlers():
    """
    If two threads invoke the handler almost simultaneously, the internal lock
    must guarantee:
      * no crashes,
      * each claim released <= 1 time,
      * after both finish `_claims` is empty (second call sees nothing).

    We mimic this by launching a second thread that calls the SIGTERM handler
    while the main thread does the same.
    """

    hits: list[tuple[str, str]] = []
    hook = ExitHook(lambda c: hits.append(c)).register()

    claims = [(f"cfg{i}", f"t{i}") for i in range(100)]
    for c in claims:
        hook.add(c)

    handler = signal.getsignal(signal.SIGTERM)

    barrier = threading.Barrier(2)

    def _invoke():
        barrier.wait()
        _invoke_handler(handler, signal.SIGTERM)

    t = threading.Thread(target=_invoke)
    t.start()

    barrier.wait()  # release both threads
    _invoke_handler(handler, signal.SIGTERM)
    t.join()

    # Every claim released exactly once
    assert sorted(hits) == sorted(claims)
    assert len(hits) == len(set(hits))


def test_handler_identity_shared_across_instances():
    """
    Registering two separate ClaimReaper objects must **not** replace the
    process-wide signal handler with two different callables.  Both `.register()`
    calls should leave exactly *one* shared handler installed.
    """

    old = signal.getsignal(signal.SIGINT)
    try:
        ExitHook(lambda _: None).register()
        handler1 = signal.getsignal(signal.SIGINT)

        ExitHook(lambda _: None).register()
        handler2 = signal.getsignal(signal.SIGINT)

        assert handler1 is handler2 is not signal.SIG_DFL
    finally:
        signal.signal(signal.SIGINT, old)


def test_multiple_hooks_each_release_their_own_claims():
    """
    When the shared handler runs, *every* live claim from *every* registered
    hook must be released exactly once.
    """

    hits1, hits2 = [], []
    r1 = ExitHook(lambda c: hits1.append(c)).register()
    r2 = ExitHook(lambda c: hits2.append(c)).register()

    claim1, claim2 = ("A", 1), ("B", 2)
    r1.add(claim1)
    r2.add(claim2)

    _invoke_handler(signal.getsignal(signal.SIGINT), signal.SIGINT)

    assert hits1 == [claim1]
    assert hits2 == [claim2]


def test_discarded_claims_on_one_hook_do_not_affect_others():
    """
    If hook-1 discards a claim before the signal arrives, only hook-2's
    live claim should be released.
    """

    h1, h2 = [], []
    r1 = ExitHook(lambda c: h1.append(c)).register()
    r2 = ExitHook(lambda c: h2.append(c)).register()

    gone = ("gone", 0)
    stay = ("stay", 1)

    r1.add(gone)
    r1.discard(gone)  # already finished
    r2.add(stay)

    _invoke_handler(signal.getsignal(signal.SIGINT), signal.SIGINT)

    assert h1 == []  # no spurious releases
    assert h2 == [stay]


def test_unregistered_hook_claims_are_not_released():
    """
    Claims tracked by a *non-registered* ClaimReaper instance must *not* be
    released when some other hook's handler fires.
    """

    ghost_hits, live_hits = [], []
    ghost = ExitHook(lambda c: ghost_hits.append(c))  # NOT registered
    live = ExitHook(lambda c: live_hits.append(c)).register()

    ghost.add(("ghost", 9))
    live.add(("live", 10))

    _invoke_handler(signal.getsignal(signal.SIGINT), signal.SIGINT)

    assert ghost_hits == []  # untouched
    assert live_hits == [("live", 10)]


def test_handler_is_reentrant_no_duplicate_releases():
    """
    After the first invocation empties the claim set, a *second* call should
    release nothing new.  This proves internal state was cleared.
    """

    hits = []
    r = ExitHook(lambda c: hits.append(c)).register()
    r.add(("cfg", "task"))

    handler = signal.getsignal(signal.SIGTERM)
    _invoke_handler(handler, signal.SIGTERM)  # first pass
    first_count = len(hits)

    _invoke_handler(handler, signal.SIGTERM)  # second pass
    assert len(hits) == first_count, "duplicate releases detected"


def test_can_add_new_claims_after_previous_release():
    """
    A hook should remain usable: after its claims are flushed by a signal,
    callers may add new claims and expect those to be released on the *next*
    signal.
    """

    hits = []
    r = ExitHook(lambda c: hits.append(c)).register()

    r.add("first")
    h = signal.getsignal(signal.SIGTERM)
    _invoke_handler(h, signal.SIGTERM)

    r.add("second")
    _invoke_handler(h, signal.SIGTERM)

    assert hits == ["first", "second"]


def test_second_signal_after_empty_claims_is_noop():
    """
    If no claims are outstanding, invoking the handler should simply return and
    **not** raise or append anything.
    """

    hits = []
    ExitHook(lambda c: hits.append(c)).register()

    handler = signal.getsignal(signal.SIGINT)
    _invoke_handler(handler, signal.SIGINT)  # nothing to release

    assert hits == []


def test_atexit_cleanup_is_idempotent():
    """
    The function registered with `atexit` must clear the claim set so that a second manual call is harmless (re-entrant) and releases nothing new.
    """

    captured = []

    def _capture(fn, *a, **kw):
        captured.append(fn)

    with pytest.MonkeyPatch().context() as m:
        m.setattr(atexit, "register", _capture)

        hits = []
        r = ExitHook(lambda c: hits.append(c)).register()
        r.add("x")

    # Simulate normal shutdown twice
    captured[0]()  # first call
    captured[0]()  # second call - should be a no-op

    assert hits == ["x"]
