import multiprocessing as mp
mp.set_start_method("fork")

from NLEval.util import checkers

class ParDat:
	def __init__(self, job_list, n_workers=5, verbose=False):
		self.job_list = job_list
		self.n_workers = n_workers
		self.verbose = verbose

	def __call__(self, func):
		def wrapper(**kwargs):
			#n_workers = checkers.checkWorkers(self.n_workers, len(self.job_list))
			n_workers = self.n_workers
			if n_workers > 1:
				q = mp.Queue()
				p = {}
				PrConn = {}

				for job_id in range(len(self.job_list)):
					if len(p) < n_workers:
						PrConn[job_id], ChConn = mp.Pipe()
						p[job_id] = mp.Process(target=ParDat.worker, 
							args=(ChConn, q, self.job_list, func, kwargs))
						p[job_id].daemon = True
						p[job_id].start()
						PrConn[job_id].send(job_id)
					else:
						worker_id, result = q.get()
						PrConn[worker_id].send(job_id)
						yield result
				for _ in p:
					worker_id, result = q.get()
					PrConn[worker_id].send(None)
					p[worker_id].join()
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



