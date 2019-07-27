from NLEval.graph.BaseGraph import BaseGraph
from NLEval.util import checkers
import numpy as np

class SparseGraph(BaseGraph):
	"""Sparse Graph object with data stored as adjacency list
	for efficient data retrieving and processing"""
	def __init__(self, weighted=True, directed=False):
		super().__init__()
		self._edge_data = []
		self.weighted = weighted
		self.directed = directed

	@property
	def edge_data(self):
		""":obj:`list` of :obj:`dict`: adjacency list data"""
		return self._edge_data

	@property
	def weighted(self):
		"""bool: Indicate whether weights (3rd column in edgelist) are available"""
		return self._weighted
	
	@property
	def directed(self):
		"""bool: Indicate whether edges are directed or not"""
		return self._directed

	@weighted.setter
	def weighted(self, val):
		checkers.checkType('weighted', bool, val)
		self._weighted = val

	@directed.setter
	def directed(self, val):
		checkers.checkType('directed', bool, val)
		self._directed = val

	def __getitem__(self, key):
		"""Return slices of constructed adjacency matrix

		Args:
			key(str): key of ID
			key(:obj:`list` of :obj:`str`): list of keys of IDs
		"""
		idx = self.IDmap[key]
		print(idx)
		if isinstance(idx, int):
			fvec = self.construct_adj_vec(idx)
		else:
			fvec_lst = []
			fvec_lst = [self.construct_adj_vec(int(i)) for i in idx]
			fvec = np.asarray(fvec_lst)
		return fvec

	def construct_adj_vec(self, src_idx):
		"""Construct and return a specific row vector of 
		adjacency matrix using correspondingadjacency list

		Args:
			src_idx(int): index of row
		"""
		checkers.checkType('src_idx', int, src_idx)
		fvec = np.zeros(self.size)
		for nbr_idx, weight in self.edge_data[src_idx].items():
			fvec[nbr_idx] = weight
		return fvec

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

	def get_edge(self, ID1, ID2):
		try:
			return self.edge_data[self.IDmap[ID1]][self.IDmap[ID2]]
		except KeyError:
				return 0

	@staticmethod
	def edglst_reader(edg_fp, weighted, directed, cut_threshold):
		"""Edge list file reader
		Read line by line from a edge list file and yield ID1, ID2, weight
		"""
		with open(edg_fp, 'r') as f:
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
		"""Numpy reader
		Load an numpy matrix (either from file path or numpy matrix directly) 
		and yield ID1, ID2, weight
		Matrix should be in shape (N, N+1), where N is number of nodes
		First column of the matrix encodes IDs
		"""
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
		"""Read data and construct sparse graph

		Args:
			file(str): path to input file
			weighted(bool): if not weighted, all weights are set to 1
			directed(bool): if not directed, automatically add 2 edges
			reader: generator function that yield edges from file
			cut_threshold(float): threshold below which edges are not considered
		"""
		for ID1, ID2, weight in reader(file, self.weighted, self.directed, cut_threshold):
			self.addEdge(ID1, ID2, weight)

	@classmethod
	def from_edglst(cls, path_to_edglst, weighted, directed, cut_threshold=0):
		graph = cls(weighted=weighted, directed=directed)
		reader = cls.edglst_reader
		graph.read(path_to_edglst, reader=reader, cut_threshold=cut_threshold)
		return graph

	@classmethod
	def from_npy(cls, npy, weighted, directed, cut_threshold=0):
		graph = cls(weighted=weighted, directed=directed)
		reader = cls.npy_reader
		graph.read(npy, reader=reader, cut_threshold=cut_threshold)
		return graph

	@staticmethod
	def edglst_writer(outpth, edge_gen, weighted, directed, cut_threshold):
		"""Edge list file writer
		Write line by line to edge list
		"""
		with open(outpth, 'w') as f:
			for srcID, dstID, weight in edge_gen():
				if weighted:
					if weight > cut_threshold:
						f.write('%s\t%s\t%.12f\n'%(srcID, dstID, weight))
				else:
					f.write('%s\t%s\n'%(srcID, dstID))

	@staticmethod
	def npy_writer():
		raise NotImplementedError

	def edge_gen(self):
		edge_data_copy = self._edge_data[:]
		for src_idx in range(len(edge_data_copy)):
			src_nbrs = edge_data_copy[src_idx]
			srcID = self.IDmap.getID(src_idx)
			for dst_idx in src_nbrs:
				dstID = self.IDmap.getID(dst_idx)
				if not self.directed:
					edge_data_copy[dst_idx].pop(src_idx)
				weight = edge_data_copy[src_idx][dst_idx]
				yield srcID, dstID, weight

	def save(self, outpth, writer='edglst', cut_threshold=0):
		"""Save graph to file

		Args:
			outpth(str): path to output file
			writer: writer function (or name of default writer) to generate file
						- 'edglst': edge list writer
						- 'npy': numpy writer
			cut_threshold(float): threshold below which edges are not considered
		"""
		if isinstance(writer, str):
			if writer == 'edglst':
				writer = self.edglst_writer
			elif writer == 'npy':
				writer = self.npy_writer
			else:
				raise ValueError('Unknown writer function name %s'%repr(writer))
		writer(outpth, self.edge_gen, self.weighted, self.directed, cut_threshold)

	def to_adjmat(self, default_val=0):
		"""Construct adjacency matrix from edgelist data

		Args:
			default_val(float): default value for missing edges
		"""
		Nnodes = self.IDmap.size
		mat = np.ones((Nnodes, Nnodes)) * default_val
		for src_node, src_nbrs in enumerate(self._edge_data):
			for dst_node in src_nbrs:
				mat[src_node, dst_node] = src_nbrs[dst_node]
		return mat