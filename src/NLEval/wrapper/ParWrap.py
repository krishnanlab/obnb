import multiprocessing as mp
mp.set_start_method("fork")

from NLEval.util import checkers

class ParDat:
    # TODO: create doc string, with example(s)
    def __init__(self, job_list, n_workers=5, verbose=False, verb_kws={}):
        self.job_list = job_list
        self.n_workers = n_workers
        self.verbose = verbose
        self.verb_kws = verb_kws

        self._q = mp.Queue()
        self._p = []
        self._PrConn = []

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
                        yield self.next(job_id)
                for result in self.terminate():
                    yield result
            else:
                for job in self.job_list:
                    self.log(**self.verb_kws)
                    yield func(job, *func_args, **func_kwargs)

        return wrapper

    @staticmethod
    def worker(conn, q, job_list, func, func_args, func_kwargs):
        job_id = worker_id = conn.recv()
        while job_id != None:
            result = func(job_list[job_id], *func_args, **func_kwargs)
            q.put((worker_id, result))
            job_id = conn.recv()
        conn.close()

    @property
    def n_workers(self):
        return self._n_workers
    
    @n_workers.setter
    def n_workers(self, n):
        checkers.checkTypeErrNone('n_workers', int, n)
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
        checkers.checkTypeErrNone('verbose', bool, val)
        self._verbose = val

    def spawn(self, func, func_args, func_kwargs):
        # configure new child process and setup communication
        PrConn, ChConn = mp.Pipe()
        new_process = mp.Process(target=ParDat.worker, 
            args=(ChConn, self._q, self.job_list, func, func_args, func_kwargs))
        new_process.daemon = True

        # launch process and send job id
        worker_id = len(self._p)
        new_process.start()
        PrConn.send(worker_id)

        # put communication and process to master lists
        self._PrConn.append(PrConn)
        self._p.append(new_process)

    def next(self, job_id):
        worker_id, result = self._q.get()
        self._PrConn[worker_id].send(job_id)
        self.log(**self.verb_kws)
        return result

    def terminate(self):
        for _ in self._p:
            worker_id, result = self._q.get()
            self._PrConn[worker_id].send(None)
            self.log(**self.verb_kws)
            yield result

    def log(self, bar_length=80, log_steps=1):
        self._n_finished += 1
        if not self.verbose:
            return
        if (self._n_finished % log_steps == 0) | \
           (self._n_finished == self._n_jobs):
            filled_length = self._n_finished * bar_length // self._n_jobs
            empty_length = bar_length - filled_length
            bar_str = '|' + '#' * filled_length + ' ' * empty_length + '|'
            progress_str = f"{bar_str} {self._n_finished} / {self._n_jobs} finished"
            print(progress_str, end='\r', flush=True)
