import torch_geometric
from torch_geometric.data import Data
import torch
from itertools import product, permutations
from math import factorial
import networkx as nx

# TODO: clean up
# TODO: Object oriented
# TODO: generalize to only unique classes on edge equivalences

def unique_edge(edge_index):
    # TODO: include this below instead of extra argument unique
    edge_index = edge_index.t().tolist()
    unique = []
    for e in edge_index:
        # print("hum, ", e)
        if (not (e[0], e[1]) in unique) and (not (e[1], e[0]) in unique):
            unique += [(e[0], e[1])]
    # print("unique edges: ",  unique)
    return unique


def reconstruct(base_data, sig, unique, k):
    '''base_data is from pytorch_geometric with edge_index
    k is the degree of the cover
    is the set of edges (i.e. with only one of (i, j) and (j, i) )
    sig is an element of the cartesian product over edges of base graph of permutations of k elements'''
    
    graph_list = []
    
    node_type = {}
    node_map = {} # this encodes the covering map, each node of the cover is a key, and the values are the respective images. TODO: replace this with th emod base_data.num_nodes
    
    for j in range(base_data.num_nodes):
        node_type[j] = [j + i*base_data.num_nodes for i in range(k)]
        for i in range(k):
            node_map[j + i*base_data.num_nodes] = j
            
    
    edge_index = []

    for i, e in enumerate(unique):
        # ith edge downstairs
        for j in range(k):
            # jth preimage of e (where it sends node number j of pre image of e[0])
            edge = (e[0] + j*base_data.num_nodes, e[1] + sig[i][j]*base_data.num_nodes)
            edge_index += [[edge[0], edge[1]],
                            [edge[1], edge[0]]]

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.zeros((base_data.num_nodes*k, 1), dtype=torch.float) # TODO: replace this with WL colouring
    cover = Data(x=x, edge_index = edge_index.t().contiguous())
    return cover, node_map, node_type

def reconstruct_list(data, L, unique, k):
    return [reconstruct(data, sig, unique, k) for sig in L]

def generate(data, k, stop=2):
    ''' data is a Data object from pytorch_geometric'''
    unique = unique_edge(data.edge_index)
    L = [] # list of covers
    print("total number of covers to check: ", factorial(k)**len(unique))
    for i, sig in enumerate(product(permutations(range(k)), repeat=len(unique))):
        # this loop enumerates all cover candidates
        # if i==9:
        #     return L[0], reconstruct(data, sig, unique, k)
        # print(i, sig)
        cover, node_map, node_type = reconstruct(data, sig, unique, k)
        # print("cover nb ", i, "edge_index: ", cover.edge_index.t().contiguous())
        # print("node map", node_map)
        # print("node_type", node_type)
        C = nx.Graph(cover.edge_index.t().tolist())
        if nx.is_connected(C):
            # print(cover)
            # L+=[(cover, edge_type, edge_map, node_type, node_map)] # missing all maps etc
            L = filter(cover, L, data, sig, node_map, node_type)
            #print("tried vs representatives ", i, len(L))
            if (not stop is None) and len(L)==stop:
                print("we found ", stop, "! Only in ", i, " tries!")
                break
    return L

def generate_with_cycle_data(data, k, cycle_edge, stop=2):
    '''like above, but cycle_edge is a list of edges given as indices (only one direction), and those are the only ones we want to sample a nontrivial permutation.'''
    unique = unique_edge(data.edge_index)
    cycle_edge_index = [i for i, e in enumerate(unique) if ([e[0], e[1]] in cycle_edge) or ([e[1], e[0]] in cycle_edge)]
    # print(cycle_edge_index)
    # print("cycle_edge: ", cycle_edge)
    L = []
    print("total number of covers to check: ", factorial(k)**len(cycle_edge))
    for i, mini_sig in enumerate(product(permutations(range(k)), repeat=len(cycle_edge))):
        # this loop enumerates all candidate covers
        sig = [tuple(range(k)) for i in range(len(unique))] # start with identity everywhere
        for j, x in enumerate(cycle_edge_index):
            sig[x] = mini_sig[j] # change the permutation of edges in distinguished elements
        sig=tuple(sig)

        cover, node_map, node_type = reconstruct(data, sig, unique, k)
        C = nx.Graph(cover.edge_index.t().tolist())
        if nx.is_connected(C):
            L = filter(cover, L, data, sig, node_map, node_type) # checks if this isomorphism type is already present 
            if (not stop is None) and len(L)==stop:
                print("we found ", stop, "! Only in ", i, " tries!")
                break
    return L


def filter(candidate, L, data, sig,
           cand_node_map, cand_node_type):
    for representative, rep_node_map, rep_node_type in L:
        if isom(candidate, representative,
                cand_node_map, cand_node_type,
               rep_node_map, rep_node_type):
            return L
    return L + [(candidate, cand_node_map, cand_node_type)]

def isom(candidate, representative,
         cand_node_map, cand_node_type,
         rep_node_map, rep_node_type):
    C = nx.Graph(candidate.edge_index.t().tolist())
    R = nx.Graph(representative.edge_index.t().tolist())
    node_mapping = []
    start = 0 # fix a node
    for v in rep_node_type[cand_node_map[0]]: # preimage of image for the two covers
        if extends(C, R, v,
                  cand_node_map, cand_node_type,
                   rep_node_map, rep_node_type):
            return True
    return False

def extends(C, R, v,
            cand_node_map, cand_node_type,
            rep_node_map, rep_node_type):
    '''  returns True is the partial map sending 0 to v extends to an isomorphism betwee the two covers C and R'''
    
    node_mapping = {0: v} # 0 is sent to v
    neighbor_edges = [(0, n) for n in C.neighbors(0)]
    checked_edges = set()
    
    while len(neighbor_edges)!=0:
        toadd = neighbor_edges.pop() # this is an edge, with initial already mapped, but terminal might not
        checked_edges.add(toadd)
        
        image = find_image(toadd, node_mapping[toadd[0]], R,
                           node_mapping,
                           cand_node_map, cand_node_type,
                           rep_node_map, rep_node_type)
        
        if not toadd[1] in node_mapping.keys(): # if map has not yet been defined on the terminal node
            # print("mapping: ", (toadd[1], image[1]))
            node_mapping[toadd[1]] = image[1]
        else:
            if node_mapping[toadd[1]] != image[1]:
                # print("no immersion possible!")
                return False

        # print("all good")
        neighbor_edges += [(toadd[1], n) for n in C.neighbors(toadd[1]) if not (toadd[1], n) in checked_edges]
    return True
            
        
def find_image(toadd, source, R,
               node_mapping, # redundant
               cand_node_map, cand_node_type,
               rep_node_map, rep_node_type):
    '''finds unique candidate edge
    source is in R, the target graph'''
    # print(toadd, source,
    #            node_mapping, # redundant
    #            cand_node_map, cand_node_type,
    #            rep_node_map, rep_node_type)
    rep_node_pre_im = rep_node_type[cand_node_map[toadd[1]]] # should be k of these, but only one being a neighbor of source, by assumption
    for temp in rep_node_pre_im:
        if rep_node_map[temp]!=cand_node_map[toadd[1]]:
            print("big problem!")
    # print(rep_node_pre_im)
    # print([x for x in R.neighbors(source)])
    source_neighb = [n for n in R.neighbors(source)]
    image = [(source, w) for w in rep_node_pre_im if w in source_neighb] # we want the w with same node type as toadd[1], which is  aneighbor of v
    if len(image) !=1:
        print("source: ", source)
        print("target type pre-im: ", rep_node_pre_im)
        print("toadd: ", toadd)
        print("node mapping", node_mapping)
        print("source neighbours: ", source_neighb)
        print(image)
        print("more, or less, than one candidate image!")
        print(R.edges)
    image = image[0]
    return image
        
    
def get_colour_signal(candidate, cand_node_map, cand_node_type):
    temp = torch.zeros((len(candidate.x), len(cand_node_type.keys())), dtype=torch.float)
    for i in range(len(candidate.x)):
        temp[i][cand_node_map[i]]=1.
    candidate.x = temp
    return candidate
    