import multiprocessing as mp
from NLEval.util import checkers

class ParDat:
	def __init__(self, iter_obj, n_jobs=5, verbose=False):
		self.iter_obj = iter_obj
		self.n_jobs = n_jobs
		self.verbose = verbose

	def __call__(self, fun):
		def wrapper(**kwargs):
			def worker(conn, q):
				worker_id = conn.recv()
				job_id = worker_id
				while job_id != None:
					result = fun(self.iter_obj[job_id], **kwargs)
					q.put((worker_id, result))
					job_id = conn.recv()
				conn.close()

			n_jobs = checkers.checkWorkers(self.n_jobs, len(self.iter_obj))
			q = mp.Queue()
			p = {}
			PrConn = {}

			for job_id in range(len(self.iter_obj)):
				if len(p) < n_jobs:
					PrConn[job_id], ChConn = mp.Pipe()
					p[job_id] = mp.Process(target=worker, args=(ChConn, q))
					p[job_id].daemon = True
					p[job_id].start()
					PrConn[job_id].send(job_id)
				else:
					worker_id, result = q.get()
					PrConn[worker_id].send(job_id)
					yield result
			for worker in p:
				worker_id, result = q.get()
				PrConn[worker_id].send(None)
				p[worker_id].join()
				yield result
		return wrapper

	@property
	def n_jobs(self):
		return self._n_jobs
	
	@n_jobs.setter
	def n_jobs(self, n):
		checkers.checkTypeErrNone('n_jobs', int, n)
		if n < 1:
			raise ValueError("n_jobs must be positive number")
		self._n_jobs = n

	@property
	def iter_obj(self):
		return self._iter_obj

	@iter_obj.setter
	def iter_obj(self, obj):
		iter(obj) # check if iterable
		self._iter_obj = obj

	@property
	def verbose(self):
		return self._verbose
	
	@verbose.setter
	def verbose(self, val):
		checkers.checkTypeErrNone('verbose', bool, val)
		self._verbose = val



