'''decides which node should come next in a walk.'''
# identity2vec.py is needed because it produces the “sentences” of nodes. Without it, train.py has nothing to train on.

import numpy as np 
import networkx as nx
from tqdm import tqdm
import time
from decimal import Decimal
from scipy.special import gammaln


class Graph():
    def __init__(self, nx_Graph, e):
        self.G = nx_Graph
        self.e = e
   
    # Returns each node’s degree
    def degree_node(self):
        G = self.G
        deg_dict = {}
        for node, deg in G.degree:
            deg_dict[node] = deg
        return deg_dict 
    
    # Returns each node's degree-distribution value Δ.
    # Δ_u = fraction of graph nodes that have the same degree as node u.
    def degree_distribution(self):
        deg_dict = self.degree_node()
        total_nodes = len(self.G.nodes)

        degree_counts = {}
        for deg in deg_dict.values():
            degree_counts[deg] = degree_counts.get(deg, 0) + 1

        dist_dict = {}
        for node, deg in deg_dict.items():
            dist_dict[node] = degree_counts[deg] / total_nodes

        return dist_dict
    
    # Computes eigenvector centrality for every node.
    # In our future version, this should be cached because it is recomputed many times although the graph does not change. The checklist also marks this as the first important fix.
    def eigenvector_centrality(self):
        G = self.G
        ev = nx.eigenvector_centrality(G, max_iter=1000)
        return ev
    
    # Creates a dictionary: each node → list of its neighbors. This is used again and again during walking.
    def node_neighbors(self):
        G = self.G
        node_neigh = {}
        for node in G.nodes:
            node_neigh[node] = list(G.neighbors(node))
        return node_neigh

    # For the first step of a walk, it randomly picks one neighbor of the starting node.
    def test_source(self, s):
        mnn = self.node_neighbors()[s] 
        arr_new_nn = np.array(mnn)
        np.random.shuffle(arr_new_nn)
        next_node = np.random.choice(arr_new_nn)
        return next_node
    
    # This function is used to skip the last visited node in the list of neighbors. It is called recursively until the last visited node is not in the list of neighbors.
    def skip_visited(self, snn, visited):
        if len(snn) != 1:
            if len(visited) > 1:
                last_visit = visited[-2]
                if last_visit in snn:
                    snn.remove(last_visit)
                    self.skip_visited(snn, visited)
        return snn
    
    # This function performs a random walk starting from a given node. It uses the Poisson distribution to decide which neighbor to visit next, while avoiding revisiting the last visited node. 
    # The walk continues until the specified walk length is reached or there are no more neighbors to visit.
    def identity_walker(self, node, walk_length):
        walk = [node]
        visited = [node]
        while len(walk) < walk_length:
            current_node = walk[-1]
            nn = self.node_neighbors()[current_node] 
            if len(nn) == 0:
                break 
            if visited[-1] == node:
                next_node = self.test_source(visited[-1])
                walk.append(next_node)
                visited.append(next_node)
            else:
                nn = self.skip_visited(nn, visited)
                bounded_curr = walk[-2]
            #     pdn = self.poisson_dist(nn, bounded_curr)
            #  #   next_node = max(pdn, key=pdn.get) 
            #     current_score = self.identity_score(current_node, bounded_curr)
            #     next_node = min(pdn, key=lambda x: abs(pdn[x] - current_score))
            
                pdn = self.poisson_dist(nn, bounded_curr)
                current_score = self.identity_score(current_node, bounded_curr)
                next_node = min(pdn, key=lambda x: abs(pdn[x] - current_score))
                walk.append(next_node)
                visited.append(next_node) 
                
        return walk
    
    # This function computes the shortest path and its length between two nodes in the graph. If there is no path, it returns an empty list and a length of zero. 
    # This is used as a distance penalty d in similarity score calculation.
    def s_path(self, source, destination): 
        G = self.G
        if nx.has_path(G, source, destination):
            path = nx.shortest_path(G, source, destination)
            path_length = nx.shortest_path_length(G, source, destination) 
        else:
            path = []
            path_length = len(path)
        return path, path_length
    
    # This function calculates the probability distributions p and q for a given node n and the current node curr.
    # p is based on the degree and eigenvector centrality of the neighbors of n, while q is based on the shortest path lengths from curr to each neighbor of n. 
    # Calculates p and q for KL-style comparison.
    # Fix 3:
    # p = Δ node degree-distribution
    # q = Ω eigenvector centrality × shortest-path distance
    def get_prob(self, n, curr):
        G = self.G
        neigh = list(G.neighbors(n))

        degree_dist = self.degree_distribution()
        ev = self.eigenvector_centrality()

        p = []
        q = []

        for node in neigh:
            _, path_length = self.s_path(curr, node)
            path_length += 0.01  # prevents zero distance

            p.append(degree_dist[node])
            q.append(ev[node] * path_length)

        return p, q
    
    # Computes one Poisson identity score Ψ for one node, using the current reference node.
    def identity_score(self, node, bounded_curr):
        e = self.e
        neigh = list(self.G.neighbors(node))

        if len(neigh) == 0:
            return 0.0
        
        rt = 0
        p, q = self.get_prob(node, bounded_curr)

        for i in range(len(p)):
            if p[i] > 0 and q[i] > 0:
                rt += p[i] * np.log(p[i] / q[i])

        k = len(neigh)

        # Fix 4A:
        # Normalize by the candidate node's own raw degree + eigenvector centrality.
        # Here, "node" is the candidate being scored.
        normalizer = self.degree_node()[node] + self.eigenvector_centrality()[node]
        drt = (1 / normalizer) * rt

        # Fix 8: compute Poisson score in log-space for numerical safety.
        # Original: (drt**k * e**(-drt)) / k!
        # Log version: k*log(drt) - drt - log(k!)
        # gammaln(k + 1) = log(k!)
        drt = max(drt, 1e-12)
        log_poiss = k * np.log(drt) - drt - gammaln(k + 1)

        return log_poiss
        
    # This is the mathematical heart of the code. It computes a KL-style divergence and then converts it into a Poisson-style score. The next node is selected using this score.

    # Computes Ψ scores for all candidate next nodes.
    def poisson_dist(self, mnn, bounded_curr):
        #k = Number of adjacent neighbors
        #λ = Divergence rate
        #pdn = (λ**k * e**-λ) / k! 
        #KLDivergence = #sum(p(x) * log(p(x)/q(x)))
        pdn = {}

        for node in mnn:
            pdn[node] = self.identity_score(node, bounded_curr)

        return pdn

    
    # This function performs multiple random walks starting from each node in the graph. It generates a corpus of walks, where each walk is a sequence of nodes visited during the random walk. 
    # The number of walks and the length of each walk can be specified as parameters.
    # This creates the full training corpus. It starts walks from every node, repeats this num_walk times, and returns all walks. This output is later given to Word2Vec/SkipGram
    def identity2vec_walk(self, num_walk, walk_length):
        
        G = self.G
        print("STARTING RANDOM WALK... ")
        print("Number of Nodes:", len(G.nodes))
        time.sleep(3)
        
        nodes = list(G.nodes)
        walk_corpus = []
        
        for cw in range(1, num_walk+1):
            print("\n")
            print("Current Walk: " + str(cw) + " of " + str(num_walk))
            for node in tqdm(nodes):
                node_walk = self.identity_walker(node, walk_length)
                walk_corpus.append(node_walk)            
               
        return walk_corpus
    




    
    