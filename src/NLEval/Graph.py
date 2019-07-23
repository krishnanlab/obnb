from NLEval.util.IDmap import IDmap
import numpy as np

class AdjLst:
	def __init__(self, weighted=True, directed=False):
		self._edge_data = []
		self.IDmap = IDmap()
		self.weighted = weighted
		self.directed = directed

	@property
	def edge_data(self):
		return self._edge_data

	@property
	def weighted(self):
		return self._weighted
	
	@property
	def directed(self):
		return self._directed

	@weighted.setter
	def weighted(self, val):
		self.check_bool('weighted',val)
		self._weighted = val

	@directed.setter
	def directed(self, val):
		self.check_bool('directed',val)
		self._directed = val

	@staticmethod
	def check_bool(name, val):
		if not isinstance(val, bool):
			raise TypeError("Argument for '%s' must be bool type, not '%s'"%\
				(name, type(val)))

	def addID(self, ID):
		self.IDmap.addID(ID)
		self._edge_data.append({})

	def addEdge(self, ID1, ID2, weight):
		for ID in [ID1, ID2]:
			#check if ID exists, add new if not
			if ID not in self.IDmap:
				self.addID(ID)
		try:
			old_weight = self._edge_data[self.IDmap[ID1]][self.IDmap[ID2]]
			if old_weight != weight:
				#check if edge exists
				print("Warning: edge between '%s' and '%s' exists with weight \
					'%.2f', overwriting with '%.2f'"%\
					(self.IDmap[ID1], self.IDmap[ID2], old_weight, weight))
		except KeyError:
			self._edge_data[self.IDmap[ID1]][self.IDmap[ID2]] = weight
			if not self.directed:
				self._edge_data[self.IDmap[ID2]][self.IDmap[ID1]] = weight

	@staticmethod
	def edglst_reader(edg_file, weighted, directed, cut_threshold):
		'''
		Read line by line from a edge list file and yield ID1, ID2, weight
		'''
		with open(edg_file, 'r') as f:
			for line in f:
				try:
					ID1, ID2, weight = line.split('\t')
					weight = float(weight)
					if weight <= cut_threshold:
						continue
					if not weighted:
						weight = float(1)
				except ValueError:
					ID1, ID2 = line.split('\t')
					weight = float(1)
				ID1 = ID1.strip()
				ID2 = ID2.strip()
				yield ID1, ID2, weight

	@staticmethod
	def npy_reader(mat, weighted, directed, cut_threshold):
		'''
		Load an numpy matrix (either from file path or numpy matrix directly) 
		and yield ID1, ID2, weight
		Matrix should be in shape (N, N+1), where N is number of nodes
		First column of the matrix encodes IDs
		'''
		if isinstance(mat, str):
			#load numpy matrix from file if provided path instead of numpy matrix
			mat = np.load(mat)
		Nnodes = mat.shape[0]

		for i in range(Nnodes):
			ID1 = mat[i,0]

			for j in range(Nnodes):
				ID2 = mat[j,0]
				weight = mat[i,j+1]
				if weight > cut_threshold:
					try:
						yield str(int(ID1)), str(int(ID2)), weight
					except TypeError:
						yield str(ID1), str(ID2), weight

	def read(self, file, reader='edglst', cut_threshold=0):
		'''
		Construct sparse graph from edge list file
		Read line by line and implicitly discard zero weighted edges
		Input:
			- file:			(str) path to edge file
			- weighted:		(bool) if not weighted, all weights are set to 1
			- directed:		(bool) if not directed, automatically add 2 edges
			- reader:		generator function that yield edges from file, or 
							specify 'edglst' for default edge list reader or
							specify 'npy' for default numpy matrix reader
			- cut_threshold:(float) threshold beyound which edge are not consider as exist
		'''
		if reader is 'edglst':
			reader = AdjLst.edglst_reader
		elif reader == 'npy':
			reader = AdjLst.npy_reader

		for ID1, ID2, weight in reader(file, self.weighted, self.directed, cut_threshold):
			self.addEdge(ID1, ID2, weight)

	@staticmethod
	def edglst_writer(outpth, edge_gen, weighted, directed, cut_threshold):
		with open(outpth, 'w') as f:
			for srcID, dstID, weight in edge_gen():
				if weighted:
					if weight > cut_threshold:
						f.write('%s\t%s\t%.12f\n'%(srcID, dstID, weight))
				else:
					f.write('%s\t%s\n'%(srcID, dstID))

	@staticmethod
	def npy_writer():
		pass

	def edge_gen(self):
		edge_data_copy = self._edge_data[:]
		for src_idx in range(len(edge_data_copy)):
			src_nbrs = edge_data_copy[src_idx]
			srcID = self.IDmap.idx2ID(src_idx)
			for dst_idx in src_nbrs:
				dstID = self.IDmap.idx2ID(dst_idx)
				if not self.directed:
					edge_data_copy[dst_idx].pop(src_idx)
				weight = edge_data_copy[src_idx][dst_idx]
				yield srcID, dstID, weight

	def save(self, outpth, writer='edglst', cut_threshold=0):
		if writer == 'edglst':
			writer = self.edglst_writer
		elif writer == 'npy':
			writer = self.npy_writer
		writer(outpth, self.edge_gen, self.weighted, self.directed, cut_threshold)

	def to_adjmat(self):
		'''
		Construct adjacency matrix from edgelist data
		TODO: prompt for default value instead of implicitely set to 0
		'''
		Nnodes = self.IDmap.size
		mat = np.zeros((Nnodes, Nnodes))
		for src_node, src_nbrs in enumerate(self._edge_data):
			for dst_node in src_nbrs:
				mat[src_node, dst_node] = src_nbrs[dst_node]
		return mat

class BaseGraph:
	def __init__(self, IDmap, mat):
		self.IDmap = IDmap
		self._mat = mat

	@property
	def mat(self):
		return self._mat
	
	@classmethod
	def from_mat(cls, mat):
		idmap = IDmap()
		for ID in mat[:,0]:
			idmap.addID(ID)
		return cls(idmap, mat[:,1:].astype(float))

	@classmethod
	def from_npy(cls, path_to_npy, **kwargs):
		mat = np.load(path_to_npy, **kwargs)
		return BaseGraph.from_mat(mat)

	@classmethod
	def from_edglst(cls, path_to_edglst, weighted, directed):
		graph = AdjLst()
		graph.read_edglst(path_to_edglst, weighted, directed)
		return cls(graph.IDmap, graph.to_adjmat())

class FeatureVec(BaseGraph):
	'''
	Feature vectors with ID maps
	'''
	def __init__(self, IDmap, mat):
		super().__init__(IDmap, mat)

	def __getitem__(self, ID):
		return self.mat[ID2idx[ID]]

	def addVec(self, ID, vec):
		'''
		Add a new feature vector
		'''
		self.IDmap.addID(ID)
		if self.mat is not None:
			self.mat = np.append(self.mat, vec.copy(), axis=0)
		else:
			self.mat = vec.copy()
	
	@classmethod
	def from_npy(cls, path_to_npy, **kwargs):
		return super(BaseGraph, cls).from_npy(path_to_npy, **kwargs)






