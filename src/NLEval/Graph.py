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
		try:
			#check if edge exists
			print("Warning: edge between '%s' and '%s' exists with weight '%.2f', overwriting with '%.2f'"%\
				(self.IDmap[ID1], self.IDmap[ID2], self._edge_data[self.IDmap[ID1]][self.IDmap[ID2]], weight))
		except KeyError:
			self._edge_data[self.IDmap[ID1]][self.IDmap[ID2]] = weight
			if not directed:
				self._edge_data[self.IDmap[ID2]][self.IDmap[ID1]] = weight

	def read_edglst(self, edg_file, weighted, directed):
		'''
		Construct sparse graph from edge list file
		Read line by line and implicitly discard zero weighted edges
		'''
		with open(edg_file, 'r') as f:

			"""print('Loading edgelist file {}...'.format(edg_file))
			tic = time.time()"""

			for line in f:
				try:
					ID1, ID2, weight = line.split('\t')
					weight = float(weight)
					if weight == 0:
						continue
					if not weighted:
						weight = float(1)
				except ValueError:
					ID1, ID2 = line.split('\t')
					weight = float(1)
				
				for ID in [ID1,ID2]:
					ID.strip() #clean up white spaces
					if ID not in self.IDmap:
						self.addID(ID)

				self.addEdge(ID1, ID2, weight, directed)

		"""toc = time.time() - tic
		print('Done, took %.2f seconds to load.' % toc)
		print('There are %d nodes in the graph.' % len(self.IDlst))"""

	def read_npy(self, mat, weighted, directed):
		if isinstance(mat, str):
			#load numpy matrix from file first
			mat = np.load(mat)
		Nnodes = mat.shape[0]

		for ID in mat[:,0]:
			self.addID(ID)

		for i in range(Nnodes):
			ID1 = mat[i,0]
			if directed:
				jstart = 0
			else:
				jstart = i

			for j in range(jstart,Nnodes):
				ID2 = mat[j,0]
				weight = mat[i,j]
				if weight > 0:
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
		self.mat = mat

	@classmethod
	def from_npy(cls, path_to_npy):
		mat = np.load(path_to_npy)
		idmap = IDmap()
		for ID in mat[:,0]:
			idmap.addID(ID)
		return cls(idmap, mat[:,1:])

	@classmethod
	def from_edglst(cls, path_to_edglst, weighted, directed):
		graph = SparseGraph()
		graph.read_edglst(path_to_edglst, weighted, directed)
		return cls(graph.IDmap, graph.to_adjmat())

class FeatureVec(BaseGraph):
	pass










