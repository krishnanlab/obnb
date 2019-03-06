from util.IDmap import IDmap
import numpy as np

class SparseGraph:
	def __init__(self):
		self._edge_data = []
		self.IDmap = IDmap()

	@property
	def edge_data(self):
		return self._edge_data

	def addID(self, ID):
		self.IDmap.addID(ID)
		self._edge_data.append({})

	def addEdge(self, ID1, ID2, weight, directed):
		for ID in [ID1, ID2]:
			#check if ID exists, add new if not
			if ID not in self.IDmap:
				self.addID(ID)
		try:
			#check if edge exists
			print("Warning: edge between '%s' and '%s' exists with weight '%.2f', overwriting with '%.2f'"%\
				(self.IDmap[ID1], self.IDmap[ID2], self._edge_data[self.IDmap[ID1]][self.IDmap[ID2]], weight))
		except KeyError:
			self._edge_data[self.IDmap[ID1]][self.IDmap[ID2]] = weight
			if not directed:
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
		Load an numpy matrix and yield ID1, ID2, weight
		Matrix should be in shape (N, N+1), where N is number of nodes
		First column of the matrix encodes IDs
		'''
		if isinstance(mat, str):
			#load numpy matrix from file if provided path instead of numpy matrix
			mat = np.load(mat)
		Nnodes = mat.shape[0]

		for i in range(Nnodes):
			ID1 = mat[i,0]
			if directed:
				jstart = 0
			else:
				jstart = i

			for j in range(jstart,Nnodes):
				ID2 = mat[j,0]
				weight = mat[i,j]
				if weight > cut_threshold:
					yield ID1, ID2, weight

	def read(self, file, weighted, directed, reader='edglst', cut_threshold=0):
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
			reader = SparseGraph.edglst_reader
		elif reader == 'npy':
			reader = SparseGraph.npy_reader

		for ID1, ID2, weight in reader(file, weighted, directed, cut_threshold):
			self.addEdge(ID1, ID2, weight, directed)

	def save_edg(self, outpth, weighted, cut_threshold=-np.inf):
		with open( outpth, 'w' ) as f:
			for src_idx, src_nbrs in enumerate(self._edge_data):
				for dst_idx in src_nbrs:
					src = self.IDlst[ src_idx ]
					dst = self.IDlst[ dst_idx ]
					if weighted:
						weight = src_nbrs[dst_idx]
						if weight > cut_threshold:
							f.write('%d\t%d\t%.12f\n'%(src,dst,weight))
					else:
						f.write('%d\t%d\n'%(src,dst))

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
		graph = SparseGraph()
		graph.read_edglst(path_to_edglst, weighted, directed)
		return cls(graph.IDmap, graph.to_adjmat())

class FeatureVec(BaseGraph):
	'''
	Feature vectors with ID maps
	'''
	def __init__(self):
		self.mat = None
		self.IDmap = IDmap()

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






