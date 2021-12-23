import multiprocessing as mp
from typing import Any
from typing import Generator
from typing import List
from typing import no_type_check

from NLEval.util import checkers

mp.set_start_method("fork")


class ParDat:
    """Run function over a list of args in parallel.

    This method is bulit upon ``multiprocessing`` using the parent-children
    communications. Specifically, it first spawns ``n_workers`` number of
    children (if this number is smaller than the total number of jobs), each
    remains a point communication with the parent process. The parent process
    then distribut jobs among the children processes. When a child process
    finshes a job, it puts the result in the queue, and when the parent process
    grab the result and either assign the child process with the next job if
    there's any, or kill the child process if there is no futher job.

    Examples:
        Let's say we have a function ``func`` and a list of things ``mylist``,
        and we want to apply our function to every element in the list, we
        can do the following list comprehension:

        >>> out_list = [func(i) for i in mylist]

        To run this in parallel using ``ParDat``, we need to decorate our
        function definition with ``ParDat``:

        >>> @ParDat(job_list=mylist, n_workers=4)
        >>> def func(main_arg, *args, **kwargs):
        >>>     result = ... # do something
        >>>     return result
        >>>
        >>> # Now can compute and show the results as follow
        >>> for result in func(*args, **kwargs):
        >>>     print(result)
        >>>
        >>> # Alternatively, wrap the function right before calling
        >>> for result in ParDat(job_list=mylist)(func)(*args, **kwargs):
        >>>     print(result)

        Notice that in this way, we don't have the guarantee that the result
        coming back have the same order.

    Todo:
        * Make a version of ParDat that return a list of results in the order
            of the input arguments.
        * Make a version of ParDat that run all the functions in parallel
            and do not gather reults. This is useful for cases where ``func``
            do not actually return anything, but just print some statements,
            for example.
    """

    def __init__(
        self,
        job_list: List[Any],
        n_workers: int = 5,
        verbose: bool = False,
        bar_length: int = 80,
        log_steps: int = 1,
    ) -> None:
        """Initialize ParDat.

        Args:
            job_list: List of main (or the first) argument for the function.
            n_workers: Number of parallel workers.
            verbose: Show progress bar if set to true.
            bar_length: Progress bar length.
            log_steps: Log interval.
        """
        self.job_list = job_list
        self.n_workers = n_workers
        self.verbose = verbose
        self.bar_length = bar_length
        self.log_steps = log_steps

        self._q: mp.Queue = mp.Queue()
        self._p: List = []
        self._parent_conn: List = []
        self._n_finished: int = 0

    @no_type_check
    def __call__(self, func):
        """Return the parallelized function over the input arguments."""

        def wrapper(*func_args, **func_kwargs):
            n_workers = self.n_workers

            if n_workers > 1:
                for job_id in range(self.n_jobs):
                    if len(self._p) < n_workers:
                        self.spawn(func, func_args, func_kwargs)
                    else:
                        yield self.get_result_and_assign_next(job_id)
                for result in self.terminate():
                    yield result
            else:
                for job in self.job_list:
                    self.log()
                    yield func(job, *func_args, **func_kwargs)

        return wrapper

    @no_type_check
    @staticmethod
    def worker(worker_id, conn, q, func, func_args, func_kwargs):
        """Worker instance.

        Args:
            worker_id: Index of the worker, used to identify child connection
                for communicaition from the parent process.
            conn: Connection with the parent process.
            q: Queue on which the results are placed.
            func: function to parallelize
            func_args: remaining positional arguments for the function
            func_kwargs: keyword arguments for the fucntion
        """
        main_arg = conn.recv()
        # TODO: create a end process indicator instead of using None.
        while main_arg is not None:
            result = func(main_arg, *func_args, **func_kwargs)
            q.put((worker_id, result))
            main_arg = conn.recv()
        conn.close()

    @property
    def n_workers(self) -> int:
        """Parallel workers number."""
        return self._n_workers

    @n_workers.setter
    def n_workers(self, n: int) -> None:
        """Setter for n_workers."""
        checkers.checkTypeErrNone("n_workers", int, n)
        if n == 0:
            n = mp.cpu_count()
        elif n < 0:
            raise ValueError("n_workers must be positive number")
        self._n_workers = n

    @property
    def n_jobs(self) -> int:
        """Total number of jobs."""
        return len(self.job_list)

    @property
    def job_list(self) -> List[Any]:
        """List of main arguments for the function."""
        return self._job_list

    @job_list.setter
    def job_list(self, obj: List[Any]) -> None:
        """Setter for job_list."""
        checkers.checkTypeErrNone("job_list", list, obj)
        self._job_list = obj

    @property
    def verbose(self) -> bool:
        """Show progress."""
        return self._verbose

    @verbose.setter
    def verbose(self, val: bool) -> None:
        """Setter for verbose."""
        checkers.checkTypeErrNone("verbose", bool, val)
        self._verbose = val

    @no_type_check
    def spawn(self, func, func_args, func_kwargs):
        """Spawn new child process.

        Set up communication with the child process and set up the result
        queue where the parent process can grab the results.
        """
        parent_conn, child_conn = mp.Pipe()
        worker_id = len(self._p)
        new_process = mp.Process(
            target=ParDat.worker,
            args=(
                worker_id,
                child_conn,
                self._q,
                func,
                func_args,
                func_kwargs,
            ),
        )
        new_process.daemon = True

        # launch process and send main arg
        new_process.start()
        parent_conn.send(self.job_list[worker_id])

        # put communication and process to master lists
        self._parent_conn.append(parent_conn)
        self._p.append(new_process)

    def get_result_and_assign_next(self, job_id: int) -> Any:
        """Retrieve result from queue and assign next job.

        Args:
            job_id: Index for next main argument to use.
        """
        worker_id, result = self._q.get()
        self._parent_conn[worker_id].send(self.job_list[job_id])
        self.log()
        return result

    def terminate(self) -> Generator[None, Any, None]:
        """Kill all children processes after the final round."""
        for _ in self._p:
            worker_id, result = self._q.get()
            self._parent_conn[worker_id].send(None)
            self.log()
            yield result

    def log(self) -> None:
        """Construct and show progress bar."""
        self._n_finished += 1
        # TODO: use tqdm instead
        if not self.verbose:
            return
        if (self._n_finished % self.log_steps == 0) | (
            self._n_finished == self.n_jobs
        ):
            filled_length = self._n_finished * self.bar_length // self.n_jobs
            empty_length = self.bar_length - filled_length
            bar_str = "|" + "#" * filled_length + " " * empty_length + "|"
            progress_str = (
                f"{bar_str} {self._n_finished} / {self.n_jobs} finished"
            )
            print(progress_str, end="\r", flush=True)
