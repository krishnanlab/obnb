import multiprocessing as mp
mp.set_start_method("fork")

from NLEval.util import checkers

class ParDat:
	def __init__(self, job_list, n_workers=5, verbose=False):
		self.job_list = job_list
		self.n_workers = n_workers
		self.verbose = verbose

		self.q = mp.Queue()
		self.p = []
		self.PrConn = []

	def __call__(self, func):
		def wrapper(**kwargs):
			#n_workers = checkers.checkWorkers(self.n_workers, len(self.job_list))
			n_workers = self.n_workers
			n_jobs = len(self.job_list)

			if n_workers > 1:
				for job_id in range(n_jobs):
					if len(self.p) < n_workers:
						self.spawn(func, kwargs)
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
		job_id = worker_id = conn.recv()
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

	def spawn(self, func, kwargs):
		# configure new child process and setup communication
		PrConn, ChConn = mp.Pipe()
		new_process = mp.Process(target=ParDat.worker, 
			args=(ChConn, self.q, self.job_list, func, kwargs))
		new_process.daemon = True

		# spawn process and send job id
		worker_id = len(self.p)
		new_process.start()
		PrConn.send(worker_id)

		# put communication and process to master lists
		self.PrConn.append(PrConn)
		self.p.append(new_process)

	def next(self, job_id):
		worker_id, result = self.q.get()
		self.PrConn[worker_id].send(job_id)
		return result

	def terminate(self):
		for _ in self.p:
			worker_id, result = self.q.get()
			self.PrConn[worker_id].send(None)
			yield result
