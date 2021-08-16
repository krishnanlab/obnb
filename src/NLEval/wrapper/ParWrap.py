import multiprocessing as mp
mp.set_start_method("fork")

from NLEval.util import checkers

class ParDat:
	def __init__(self, job_list, n_workers=5, verbose=False):
		self.job_list = job_list
		self.n_workers = n_workers
		self.verbose = verbose

		self.q = mp.Queue()
		self.p = {}
		self.PrConn = {}

	def __call__(self, func):
		def wrapper(**kwargs):
			#n_workers = checkers.checkWorkers(self.n_workers, len(self.job_list))
			n_workers = self.n_workers
			n_jobs = len(self.job_list)

			if n_workers > 1:
				for job_id in range(n_jobs):
					if len(self.p) < n_workers:
						self.spawn(job_id, func, kwargs)
					else:
						yield self.next(job_id)
				for result in self.terminate():
					yield result
			else:
				for job in self.job_list:
					yield func(job, **kwargs)
		return wrapper

	@staticmethod
	def worker(conn, q, job_list, func, kwargs):
		worker_id = conn.recv()
		job_id = worker_id
		while job_id != None:
			result = func(job_list[job_id], **kwargs)
			q.put((worker_id, result))
			job_id = conn.recv()
		conn.close()

	@property
	def n_workers(self):
		return self._n_workers
	
	@n_workers.setter
	def n_workers(self, n):
		checkers.checkTypeErrNone('n_workers', int, n)
		if n < 1:
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

	def spawn(self, job_id, func, kwargs):
		self.PrConn[job_id], ChConn = mp.Pipe()
		self.p[job_id] = mp.Process(target=ParDat.worker, 
			args=(ChConn, self.q, self.job_list, func, kwargs))
		self.p[job_id].daemon = True
		self.p[job_id].start()
		self.PrConn[job_id].send(job_id)

	def next(self, job_id):
		worker_id, result = self.q.get()
		self.PrConn[worker_id].send(job_id)
		return result

	def terminate(self):
		for _ in self.p:
			worker_id, result = self.q.get()
			self.PrConn[worker_id].send(None)
			yield result
