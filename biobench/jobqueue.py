import collections

import beartype


@beartype.beartype
class JobQueue[T]:
    """
    A simple queue for managing jobs with a maximum size limit.

    This queue is used to throttle job submissions and track pending jobs.
    """

    def __init__(self, max_size: int):
        """
        Initialize a new JobQueue with a maximum size.

        Args:
            max_size: Maximum number of jobs that can be in the queue at once
        """
        self.max_size = max_size
        self.jobs = collections.deque()

    def submit(self, job: T) -> None:
        """
        Add a job to the queue.

        Args:
            job: The job to add to the queue

        Raises:
            ValueError: If the queue is already full
        """
        if self.full():
            raise ValueError("Queue is full")
        self.jobs.append(job)

    def pop(self) -> T:
        """
        Remove and return the oldest job from the queue.

        Returns:
            The oldest job in the queue

        Raises:
            IndexError: If the queue is empty
        """
        if not self.jobs:
            raise IndexError("Queue is empty")
        return self.jobs.popleft()

    def full(self) -> bool:
        """
        Check if the queue is at maximum capacity.

        Returns:
            True if the queue is full, False otherwise
        """
        return len(self.jobs) >= self.max_size

    def __len__(self) -> int:
        """
        Get the current number of jobs in the queue.

        Returns:
            The number of jobs currently in the queue
        """
        return len(self.jobs)

    def __bool__(self) -> bool:
        """
        Check if the queue has any jobs.

        Returns:
            True if the queue has at least one job, False otherwise
        """
        return bool(self.jobs)
