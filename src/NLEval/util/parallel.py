import multiprocessing as mp

from tqdm import tqdm

from . import checkers
from ..typing import Any
from ..typing import Iterator
from ..typing import List
from ..typing import Tuple

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
        coming back have the same order. To enforce the order, use ParDatMap,
        which return the results in a list with the same order as the input
        main arguments.
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

    def __call__(self, func):
        """Return the parallelized function over the input arguments."""

        def wrapper(*func_args, **func_kwargs):
            n_workers = self.n_workers
            n_jobs = self.n_jobs
            disable = not self.verbose

            with tqdm(total=n_jobs, disable=disable) as pbar:
                if n_workers > 1:
                    for job_id in range(self.n_jobs):
                        if len(self._p) < n_workers:
                            self.spawn(func, func_args, func_kwargs)
                        else:
                            pbar.update(1)
                            yield self.get_result_and_assign_next(job_id)[1]
                    for result in self.terminate():
                        pbar.update(1)
                        yield result[1]
                else:
                    for job in self.job_list:
                        yield func(job, *func_args, **func_kwargs)
                        pbar.update(1)

        return wrapper

    @staticmethod
    def worker(worker_id, conn, job_list, q, func, func_args, func_kwargs):
        """Worker instance.

        Args:
            worker_id: Index of the worker, used to identify child connection
                for communicaition from the parent process.
            conn: Connection with the parent process.
            job_list: List of main arguments for the function.
            q: Queue on which the results are placed.
            func: Function to parallelize.
            func_args: Remaining positional arguments for the function.
            func_kwargs: Keyword arguments for the fucntion.
        """
        job_id = worker_id
        while job_id is not None:
            result = func(job_list[job_id], *func_args, **func_kwargs)
            q.put((worker_id, job_id, result))
            job_id = conn.recv()
        conn.close()

    @property
    def n_workers(self) -> int:
        """Parallel workers number."""
        return self._n_workers

    @n_workers.setter
    def n_workers(self, n: int) -> None:
        """Setter for n_workers."""
        checkers.checkType("n_workers", int, n)
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
        checkers.checkType("job_list", list, obj)
        self._job_list = obj

    @property
    def verbose(self) -> bool:
        """Show progress."""
        return self._verbose

    @verbose.setter
    def verbose(self, val: bool) -> None:
        """Setter for verbose."""
        checkers.checkType("verbose", bool, val)
        self._verbose = val

    def spawn(self, func, func_args, func_kwargs):
        """Spawn new child process.

        Set up communication with the child process and set up the result
        queue where the parent process can grab the results.
        """
        # Setup parent child connection and start a child process
        parent_conn, child_conn = mp.Pipe()
        worker_id = len(self._p)
        new_process = mp.Process(
            target=ParDat.worker,
            args=(
                worker_id,
                child_conn,
                self.job_list,
                self._q,
                func,
                func_args,
                func_kwargs,
            ),
        )
        new_process.daemon = True
        new_process.start()

        # Put communication and process to master lists
        self._parent_conn.append(parent_conn)
        self._p.append(new_process)

    def get_result_and_assign_next(self, job_id: int) -> Any:
        """Retrieve result from queue and assign next job.

        Args:
            job_id: Index for next main argument to use.
        """
        worker_id, prev_job_id, result = self._q.get()
        self._parent_conn[worker_id].send(job_id)
        return prev_job_id, result

    def terminate(self) -> Iterator[Tuple[int, Any]]:
        """Kill all children processes after the final round."""
        for _ in self._p:
            worker_id, prev_job_id, result = self._q.get()
            self._parent_conn[worker_id].send(None)
            yield prev_job_id, result


class ParDatMap(ParDat):
    """Run function over a list of args in parallel and return list of results.

    Examples:
        >>> @ParDatMap(job_list=mylist, n_workers=4)
        >>> def func(main_arg, *args, **kwargs):
        >>>     result = ... # do something
        >>>     return result
        >>>
        >>> # Now can compute and show the results as follow
        >>> results = func(*args, **kwargs):
        >>>
        >>> # Alternatively, wrap the function right before calling
        >>> results = ParDatMap(job_list=mylist)(func)(*args, **kwargs):
    """

    def __call__(self, func):
        """Return the parallelized function over the input arguments."""

        def wrapper(*func_args, **func_kwargs):
            n_workers = self.n_workers
            n_jobs = self.n_jobs
            disable = not self.verbose
            results = [None] * n_jobs

            with tqdm(total=n_jobs, disable=disable) as pbar:
                if n_workers > 1:
                    for job_id in range(self.n_jobs):
                        if len(self._p) < n_workers:
                            self.spawn(func, func_args, func_kwargs)
                        else:
                            pbar.update(1)
                            result = self.get_result_and_assign_next(job_id)
                            results[result[0]] = result[1]
                    for result in self.terminate():
                        pbar.update(1)
                        results[result[0]] = result[1]
                else:
                    for i, job in enumerate(self.job_list):
                        results[i] = func(job, *func_args, **func_kwargs)
                        pbar.update(1)

            return results

        return wrapper


class ParDatExe(ParDat):
    """Run function over a list of args in parallel and do not capture outputs.

    Examples:
        >>> @ParDatExe(job_list=mylist, n_workers=4)
        >>> def func(main_arg, *args, **kwargs):
        >>>     # do something, but do not output anything
        >>>
        >>> # Now can execute the functions in parallel over elements in mylist
        >>> func(*args, **kwargs):
        >>>
        >>> # Alternatively, wrap the function right before calling
        >>> ParDatExe(job_list=mylist)(func)(*args, **kwargs):
    """

    def __call__(self, func):
        """Return the parallelized function over the input arguments."""

        def wrapper(*func_args, **func_kwargs):
            n_workers = self.n_workers
            n_jobs = self.n_jobs
            disable = not self.verbose
            results = [None] * n_jobs

            with tqdm(total=n_jobs, disable=disable) as pbar:
                if n_workers > 1:
                    for job_id in range(self.n_jobs):
                        if len(self._p) < n_workers:
                            self.spawn(func, func_args, func_kwargs)
                        else:
                            pbar.update(1)
                            self.get_result_and_assign_next(job_id)
                    for _ in self.terminate():
                        pbar.update(1)
                else:
                    for job in self.job_list:
                        func(job, *func_args, **func_kwargs)
                        pbar.update(1)

            return results

        return wrapper
