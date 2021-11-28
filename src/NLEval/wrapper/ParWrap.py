import multiprocessing as mp

from NLEval.util import checkers

mp.set_start_method("fork")


class ParDat:
    # TODO: create doc string, with example(s)
    def __init__(
        self,
        job_list,
        n_workers=5,
        verbose=False,
        bar_length=80,
        log_steps=1,
    ):
        self.job_list = job_list
        self.n_workers = n_workers
        self.verbose = verbose
        self.bar_length = bar_length
        self.log_steps = log_steps

        self._q = mp.Queue()
        self._p = []
        self._parent_conn = []

    def __call__(self, func):
        def wrapper(*func_args, **func_kwargs):
            n_workers = self.n_workers
            n_jobs = self._n_jobs = len(self.job_list)
            self._n_finished = 0

            if n_workers > 1:
                for job_id in range(n_jobs):
                    if len(self._p) < n_workers:
                        self.spawn(func, func_args, func_kwargs)
                    else:
                        yield self.next_job(job_id)
                for result in self.terminate():
                    yield result
            else:
                for job in self.job_list:
                    self.log()
                    yield func(job, *func_args, **func_kwargs)

        return wrapper

    @staticmethod
    def worker(conn, q, job_list, func, func_args, func_kwargs):
        job_id = worker_id = conn.recv()
        while job_id is not None:
            result = func(job_list[job_id], *func_args, **func_kwargs)
            q.put((worker_id, result))
            job_id = conn.recv()
        conn.close()

    @property
    def n_workers(self):
        return self._n_workers

    @n_workers.setter
    def n_workers(self, n):
        checkers.checkTypeErrNone("n_workers", int, n)
        if n == 0:
            n = mp.cpu_count()
        elif n < 0:
            raise ValueError("n_workers must be positive number")
        self._n_workers = n

    @property
    def job_list(self):
        return self._job_list

    @job_list.setter
    def job_list(self, obj):
        checkers.checkTypeErrNone("job_list", list, obj)
        self._job_list = obj

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        checkers.checkTypeErrNone("verbose", bool, val)
        self._verbose = val

    def spawn(self, func, func_args, func_kwargs):
        # configure new child process and setup communication
        parent_conn, child_conn = mp.Pipe()
        new_process = mp.Process(
            target=ParDat.worker,
            args=(
                child_conn,
                self._q,
                self.job_list,
                func,
                func_args,
                func_kwargs,
            ),
        )
        new_process.daemon = True

        # launch process and send job id
        worker_id = len(self._p)
        new_process.start()
        parent_conn.send(worker_id)

        # put communication and process to master lists
        self._parent_conn.append(parent_conn)
        self._p.append(new_process)

    def next_job(self, job_id):
        worker_id, result = self._q.get()
        self._parent_conn[worker_id].send(job_id)
        self.log()
        return result

    def terminate(self):
        for _ in self._p:
            worker_id, result = self._q.get()
            self._parent_conn[worker_id].send(None)
            self.log()
            yield result

    def log(self):
        self._n_finished += 1
        if not self.verbose:
            return
        if (self._n_finished % self.log_steps == 0) | (
            self._n_finished == self._n_jobs
        ):
            filled_length = self._n_finished * self.bar_length // self._n_jobs
            empty_length = self.bar_length - filled_length
            bar_str = "|" + "#" * filled_length + " " * empty_length + "|"
            progress_str = (
                f"{bar_str} {self._n_finished} / {self._n_jobs} finished"
            )
            print(progress_str, end="\r", flush=True)
