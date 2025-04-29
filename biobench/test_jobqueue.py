import pytest

from . import jobqueue


def test_jobqueue_init():
    """Test that JobQueue initializes with the correct max_size."""
    queue = jobqueue.JobQueue(5)
    assert queue.max_size == 5
    assert len(queue) == 0


def test_jobqueue_submit():
    """Test that submit adds a job to the queue."""
    queue = jobqueue.JobQueue(5)
    queue.submit("job1")
    assert len(queue) == 1


def test_jobqueue_pop():
    """Test that pop removes and returns the oldest job."""
    queue = jobqueue.JobQueue(5)
    queue.submit("job1")
    queue.submit("job2")
    assert queue.pop() == "job1"
    assert len(queue) == 1


def test_jobqueue_fifo_order():
    """Test that jobs are returned in FIFO order."""
    queue = jobqueue.JobQueue(5)
    jobs = ["job1", "job2", "job3"]
    for job in jobs:
        queue.submit(job)

    for expected_job in jobs:
        assert queue.pop() == expected_job


def test_jobqueue_full():
    """Test that full() returns True when queue is at max capacity."""
    queue = jobqueue.JobQueue(2)
    assert not queue.full()
    queue.submit("job1")
    assert not queue.full()
    queue.submit("job2")
    assert queue.full()


def test_jobqueue_submit_when_full():
    """Test that submitting to a full queue raises ValueError."""
    queue = jobqueue.JobQueue(1)
    queue.submit("job1")
    with pytest.raises(ValueError, match="Queue is full"):
        queue.submit("job2")


def test_jobqueue_pop_when_empty():
    """Test that popping from an empty queue raises IndexError."""
    queue = jobqueue.JobQueue(5)
    with pytest.raises(IndexError, match="Queue is empty"):
        queue.pop()


def test_jobqueue_bool_conversion():
    """Test that bool(queue) returns True if queue has jobs, False otherwise."""
    queue = jobqueue.JobQueue(5)
    assert not bool(queue)
    queue.submit("job1")
    assert bool(queue)
    queue.pop()
    assert not bool(queue)


def test_jobqueue_with_complex_objects():
    """Test that JobQueue works with complex objects like tuples."""
    queue = jobqueue.JobQueue(5)
    job = (lambda: None, {"config": "value"}, "task_name")
    queue.submit(job)
    assert queue.pop() == job


def test_jobqueue_max_size_zero():
    """Test that a JobQueue with max_size=0 is always full."""
    queue = jobqueue.JobQueue(0)
    assert queue.full()
    with pytest.raises(ValueError, match="Queue is full"):
        queue.submit("job1")


def test_initial_state():
    # constructor sets capacity & starts empty
    q = jobqueue.JobQueue[int](max_size=3)
    assert len(q) == 0
    assert q.full() is False


def test_submit_increments_len():
    # submit increases length
    q = jobqueue.JobQueue(max_size=3)
    q.submit(10)
    assert len(q) == 1
    q.submit(20)
    assert len(q) == 2


def test_full_flag():
    # full() flips to True at capacity
    q = jobqueue.JobQueue(max_size=3)
    for i in range(3):
        q.submit(i)
    assert q.full() is True


def test_submit_when_full_raises():
    # submitting when full raises
    q = jobqueue.JobQueue(max_size=1)
    q.submit(1)
    with pytest.raises(RuntimeError):
        q.submit(2)


def test_fifo_pop():
    # FIFO semantics for pop()
    q = jobqueue.JobQueue(max_size=3)
    for i in range(3):
        q.submit(i)
    assert q.pop() == 0
    assert q.pop() == 1
    assert q.pop() == 2


def test_pop_updates_state():
    # pop decreases len and clears full flag
    q = jobqueue.JobQueue(max_size=3)
    for i in range(3):
        q.submit(i)
    _ = q.pop()
    assert len(q) == 2
    assert q.full() is False


def test_pop_empty_raises():
    # pop on empty raises
    q = jobqueue.JobQueue()
    with pytest.raises(IndexError):
        q.pop()


def test_len_never_exceeds_capacity():
    # length never exceeds capacity through random ops
    q = jobqueue.JobQueue(max_size=2)
    q.submit(1)
    q.submit(2)
    with pytest.raises(RuntimeError):
        q.submit(3)  # should raise because capacity==3 after pending pop
    _ = q.pop()
    q.submit(3)  # now fits
    assert len(q) <= 3
