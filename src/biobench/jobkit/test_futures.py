"""
Unit tests for the `JobQueue` helper described in Coding-spec: ClaimReaper & JobQueue.

We use a minimal `FakeJob` stub that mimics the `submitit.Job` API (`done()` -> bool) so the tests remain independent of SubmitIt. Each test includes a docstring explaining the contract being verified.
"""

import queue
import threading
import time

import pytest

from .futures import FutureQueue


class FakeJob:
    """Simple stub that reports completion via `done()`."""

    def __init__(self, done: bool = False):
        self._done = done

    def done(self) -> bool:
        return self._done

    def mark_done(self):
        self._done = True

    # Make debugging nicer.
    def __repr__(self) -> str:
        return f"<FakeJob done={self._done}>"


class DelayedJob(FakeJob):
    """`done()` flips to True after `delay` wall-seconds."""

    def __init__(self, delay: float):
        super().__init__(done=False)
        self._ready_at = time.time() + delay

    def done(self) -> bool:
        if not self._done and time.time() >= self._ready_at:
            self._done = True
        return self._done


def test_submit_and_len():
    """
    `submit()` appends items and `len(queue)` tracks the count.

    A queue with `max_size=3` should grow from 0 -> 1 -> 2 as items are added.
    """
    q = FutureQueue(max_size=3)
    assert len(q) == 0

    q.submit(FakeJob())
    assert len(q) == 1

    q.submit(FakeJob())
    assert len(q) == 2


def test_full_and_submit_raises():
    """
    `submit()` must raise `RuntimeError` when the queue is full.

    This covers both:
    * normal capacity exhaustion (`max_size=1`)
    * the "always full" sentinel (`max_size=0`)
    """
    single = FutureQueue(max_size=1)
    single.submit(FakeJob())  # fills the queue
    with pytest.raises(RuntimeError):
        single.submit(FakeJob())  # should fail

    always_full = FutureQueue(max_size=0)
    assert always_full.full() is True
    with pytest.raises(RuntimeError):
        always_full.submit(FakeJob())  # should fail immediately


def test_bool_dunder_and_full():
    """
    `__bool__` mirrors "non-empty" and `full()` reflects capacity status.

    * Empty queue -> `bool(q)` is `False`.
    * After one insert -> `bool(q)` is `True`.
    * When at capacity -> `full()` becomes `True`.
    """
    q = FutureQueue(max_size=2)
    assert not q and len(q) == 0

    q.submit(FakeJob())
    assert q and not q.full()

    q.submit(FakeJob())
    assert q.full()


@pytest.mark.timeout(5)
def test_pop_returns_first_finished_payload():
    """
    `pop()` must return the *earliest finished* payload, not simply FIFO order.

    We enqueue two jobs:
    * `slow` -> not yet done
    * `fast` -> already done

    Even though `slow` was submitted first, `pop()` should unblock on `fast`.
    """
    slow, fast = FakeJob(done=False), FakeJob(done=True)
    q = FutureQueue(max_size=3)
    q.submit(slow)  # inserted first
    q.submit(fast)  # inserted second

    popped = q.pop()
    assert popped is fast
    assert len(q) == 1 and q.full() is False


@pytest.mark.timeout(5)
def test_pop_recognises_nested_jobs():
    """
    The queue must scan *nested* payloads (<= 2 levels, depth-first).

    We wrap a finished `FakeJob` inside a tuple along with arbitrary metadata
    and verify that `pop()` still detects completion.
    """
    done_job = FakeJob(done=True)
    nested_payload = (done_job, {"cfg": "dummy"})
    q = FutureQueue(max_size=2)
    q.submit(nested_payload)

    popped = q.pop()
    assert popped is nested_payload


@pytest.mark.timeout(5)
def test_pop_frees_capacity_for_further_submissions():
    """
    After `pop()` removes a finished payload the queue should no longer be full.

    * Fill a queue of size 1 with a *finished* job.
    * `pop()` should return immediately and leave the queue empty.
    * A subsequent `submit()` must succeed without raising.
    """
    q = FutureQueue(max_size=1)
    q.submit(FakeJob(done=True))  # queue is full

    q.pop()  # frees the slot
    assert len(q) == 0 and not q.full()

    # Should accept a new item now.
    try:
        q.submit(FakeJob())
    except RuntimeError:
        pytest.fail("Queue did not free capacity after pop()")


@pytest.mark.timeout(5)
def test_depth_two_nesting_is_detected():
    """
    The spec promises a *depth-first scan <= 2* levels deep.

    We wrap a finished job two layers down:  [ ( job ) ].  `pop()` must still
    recognise completion and unblock.
    """
    deep = [[FakeJob(done=True)]]
    q = FutureQueue(max_size=2)
    q.submit(deep)

    popped = q.pop()
    assert popped is deep  # exact payload returned


@pytest.mark.timeout(5)
def test_payload_with_multiple_jobs_mixed_status():
    """
    If any *member* job is finished the *whole* payload counts as finished.

    We craft a container with (unfinished, finished).  Even though the first
    element is *not* done, the second is -- so `pop()` must still dequeue it.
    """
    first, second = FakeJob(done=False), FakeJob(done=True)
    mixed = (first, second)
    q = FutureQueue(max_size=1)
    q.submit(mixed)

    popped = q.pop()
    assert popped is mixed
    assert len(q) == 0  # queue now empty


@pytest.mark.timeout(5)
def test_non_job_payload_left_in_queue_if_no_done_job_inside():
    """
    Submitting a payload *without* any `done()` method must *not* starve later
    finished jobs.

    We enqueue a plain dict (no job) first, followed by a finished `FakeJob`.
    `pop()` should skip over the dict, return the finished job, and leave the
    dict in the queue.
    """
    sentinel = {"msg": "not a job"}
    finished = FakeJob(done=True)

    q = FutureQueue(max_size=3)
    q.submit(sentinel)
    q.submit(finished)

    assert q.pop() is finished
    assert list(q) == [sentinel]  # dict still waiting (never satisfies pop)


@pytest.mark.timeout(5)
def test_iter_preserves_fifo_after_interleaved_pops():
    """
    `__iter__` must always reflect current FIFO order of *remaining* items.

    Scenario:
    1. enqueue A(done=False), B(done=True), C(done=False)
    2. `pop()` removes B
    3. Order should now be [A, C] -- verify with `list(q)`.
    """
    a, b, c = FakeJob(), FakeJob(done=True), FakeJob()
    q = FutureQueue(max_size=5)
    q.submit(a)
    q.submit(b)
    q.submit(c)

    q.pop()  # removes B
    assert [a, c] == list(q)


def _call_pop(fq: FutureQueue, out: queue.Queue):
    """Run fq.pop() in a separate thread and capture result / exception."""
    try:
        out.put(fq.pop())
    except Exception as exc:  # pragma: no cover
        out.put(exc)


def test_pop_returns_first_runtime_finish_not_fifo():
    """
    The job that *completes first in wall time* must be returned, regardless
    of insertion order.

    Queue order:  slow(2 s)  ->  fast(0.1 s)

    Expected: `pop()` blocks ~0.1 s, returns *fast*.
    """
    slow = DelayedJob(2.0)
    fast = DelayedJob(0.1)

    fq = FutureQueue(max_size=2)
    fq.submit(slow)  # FIFO head
    fq.submit(fast)  # finishes first

    out = queue.Queue()
    t = threading.Thread(target=_call_pop, args=(fq, out), daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive(), "pop() blocked too long"
    popped = out.get_nowait()
    assert popped is fast
    assert list(fq) == [slow]  # slow still pending


def test_blocking_pop_unblocks_when_only_job_finishes_later():
    """
    When **all** payloads are unfinished, `pop()` must wait until one finishes.

    We enqueue a single `DelayedJob(0.2)`, start `pop()` in a thread, and
    verify it returns within the safety window.
    """
    delayed = DelayedJob(0.2)
    fq = FutureQueue(max_size=1)
    fq.submit(delayed)

    out = queue.Queue()
    t = threading.Thread(target=_call_pop, args=(fq, out), daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive(), "pop() blocked >5 s"
    assert out.get_nowait() is delayed
    assert len(fq) == 0


def test_pop_skips_non_job_then_unblocks_on_future_finish():
    """
    Interleaving:  [dict, DelayedJob].

    `pop()` must *skip* the non-job payload, block until the job finishes,
    then return **the job** while leaving the dict untouched.
    """
    sentinel = {"cfg": "noop"}
    job = DelayedJob(0.1)

    fq = FutureQueue(3)
    fq.submit(sentinel)
    fq.submit(job)

    out = queue.Queue()
    t = threading.Thread(target=_call_pop, args=(fq, out), daemon=True)
    t.start()
    t.join(timeout=5)

    assert out.get_nowait() is job
    assert list(fq) == [sentinel]


def test_multiple_done_jobs_in_one_payload_returns_first_depth_first():
    """
    If a single payload contains *several* finished jobs, the queue should treat the first depth-first hit as the trigger.

    We build:  ( doneA , [ doneB ] )  -- `pop()` may choose either, but spec allows returning the *payload* not the individual job; ensure dequeueing occurs immediately.
    """
    a, b = FakeJob(done=True), FakeJob(done=True)
    nested = (a, [b])

    fq = FutureQueue(2)
    fq.submit(nested)

    assert fq.pop() is nested
    assert len(fq) == 0
