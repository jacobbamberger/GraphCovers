from itertools import product, permutations
from math import factorial

import plotly.graph_objects as go
import numpy as np
import networkx as nx

# TODO: Generalize to disconnected covers.
#       Either by adapting isom, or by constructing from disjoint
#       unions of smaller covers...


class Cover():
    """
    Base class for covers of graphs.
    """

    def __init__(self, base_edge_index, degree, sigma):
        """
        Constructs all data necessary to define a cover of the base graph.
        The construction happens in the helper function _construct

        Parameters
        ----------
        base_edge_index : List of length 2 tuples
                          Represents edges present in the base graph. Each
                          element is a length tuple (i, j) indicating that
                          the ith node is connected to the jth node by an
                          edge. Since we are dealing with undirected graphs,
                          if (i, j) is in the list, so is (j, i)
        degree : Int, greater than 1.
                 Represents the degree (or number of sheets) of the cover to
                 generate.
        sigma : Tuple of length len(base_edge_index)/2.
                Represents the connection pattern of the construction, as
                described in Section 3 of the paper. Each element of `sigma`
                is another tuple, of length `degree`, which is defines the
                connection pattern of the pre-images of edge an edge e. The
                elements can be thought of as elements of the permutation group
                on `degree` elements, where the pth element is where p is sent
                to by the permutation. The connection pattern is defined as
                follows: we assume an order on the pre-images of each nodes.
                If nodes i and j are connected in the base graph through
                edge e, then we connect the pth pre-image of node i to the
                `sigma`[e](p)th pre-image of node j.
        """

        self.base_edge_index = base_edge_index
        self.degree = degree
        self.sigma = sigma

        self.unique = unique_edge(base_edge_index)

        edge_index, node_map, node_type = self._construct()

        self.edge_index = edge_index
        self.node_map = node_map
        self.node_type = node_type

        self.nxGraph = nx.Graph(edge_index)

    def _construct(self):
        """
        Constructs edge indices of the degree self.degree cover of the graph
        described by base_edge_index, where the cover construction is
        determined by self.sigma. This procedure is described at the end of
        Section 3 of the paper, or in the documentation of __init__ above.

        Returns
        ----------
        (edge_index, node_map, node_type) : Tuple, where each elements are
                                            described below.
        edge_index :  List of length 2 lists
                      Represents the connection patterns of the cover. If the
                      element [i, j] appears in it, then the ith node is
                      connected to the jth node through an edge. Morevover,
                      since the graph is undirected, if [i, j] appears, then
                      so does [j, i].
        node_map: Dictionary
                  Represents the map from nodes of covers, to the nodesd of
                  the base graph, by defining it on the indices of nodes. A
                  key is a node index (Int) of the cover, and a value is the
                  node index (Int) of the image in the base graph.
        node_type: Dictionary
                   Represents the 'inverse' map to `node_map`, which is a 1
                   to `degree` map. A key is a node index of the base graph,
                   and a value is the set of node indices in the cover which
                   map to the key (hence they are the fibers of `node_map`)
        """

        # Define the map from cover's node to base graph's nodes. This is a
        # dicitonary where keys are node indices of the cover, and values are
        # the node index of the image in the base graph.
        node_map = {}

        # Define the inverse map of node_map. A dictionary where keys are node
        # indices of base graph, and value is the set of node indices of the
        # cover correponsind to the pre-image of the base graph's node
        node_type = {}

        # number of nodes in the base graph
        base_num_nodes = len(set([e[0] for e in self.base_edge_index]))
        # Note that this assumes that every node has an edge coming out of it.

        for j in range(base_num_nodes):
            node_type[j] = [j + i*base_num_nodes for i in range(self.degree)]
            for i in range(self.degree):
                node_map[j + i*base_num_nodes] = j

        # edge_index of the cover
        edge_index = []
        for i, e in enumerate(self.unique):
            # ith edge in the base graph
            for j in range(self.degree):  # jth preimage of e
                # connect the jth pre-image of node e[0] to the
                # sigma[i][j]th preimage of node e[1]:
                edge = (e[0] + j*base_num_nodes,
                        e[1] + self.sigma[i][j]*base_num_nodes)
                edge_index += [[edge[0], edge[1]],
                               [edge[1], edge[0]]]
        return edge_index, node_map, node_type

    def isom(self, cover):
        """
        Determines if `self` is isomorphic as covers to `cover`.
        We do so in several steps. By Lemma 3.2 in the paper, it suffices to
        check if we can extend a the assignment 0 --> v to an isomorphsim of
        covers. We therefore loop through vertices of `cover`, and try to
        extend the assigment to an isomorphism by using the helper function
        _extends defined below.

        Parameters
        ----------
        cover : Cover object
                This is the cover we want to check isomorphism to.
                It is assumed that the base graphs are equal!
                TODO: Check that base graphs are equal first.

        Return
        ----------
        _ : Boolean
            If `self` and  `covers` are isomorphic this is True,
            otherwise it is False.
        """
        for v in cover.node_type[0]:
            # This loops over the candidates images of 0, since we need them
            # to be in the pre-image of 0 to make the diagram commute
            if self._extends(v, cover):
                return True
        return False

    def _extends(self, v, cover):
        """
        Determines if the assignment 0 to v extends to an isomorphism of covers

        Parameters
        ----------
        cover : Cover object

        Return
        ----------
        _ : Boolean
            True if the assignment extends, False otherwise
        """
        # initialize the map between nodes
        node_mapping = {0: v}  # 0 is sent to v
        # where to extend the map next:
        extension_edges = [(0, n) for n in self.nxGraph.neighbors(0)]
        checked_edges = set()  # track edges we have already extended on

        while len(extension_edges) != 0:
            # while there is somwhere to locally extend the map
            # toadd is an edge, source already mapped, but target might not:
            toadd = extension_edges.pop()
            # edges we have already mapped consistently:
            checked_edges.add(toadd)

            # Find where to send the target of toadd.
            # Out of the neighbours of the image of the source of toadd,
            # there is only one that covers the same base graph node as the
            # target of toadd
            source_im_neighb = [n for n in cover.nxGraph
                                                .neighbors(node_mapping[toadd[0]])]
            # these are the neighbours of the image of the source
            safety_count = 0
            for n in source_im_neighb:
                # TODO: extend on all neighbours at once, should be faster
                if cover.node_map[n] == self.node_map[toadd[1]]:
                    # by definitions of cover, this happens only once
                    image = n
                    safety_count += 1
            if safety_count != 1:
                print("Problem!")  # see previous comment

            # image is now the candidate image of toadd[1], the target of
            # the edge we want to extend along.
            # We need to check that extending it in this way does node
            # violate consistency of the map!
            if not (toadd[1] in node_mapping.keys()):
                # if map has not yet been defined on the terminal node
                # then we can define it as we want:
                node_mapping[toadd[1]] = image
            else:
                if node_mapping[toadd[1]] != image:
                    # if the new extension does not agree with an old one
                    return False  # then the map can't be extended consistently

            extension_edges += [(toadd[1], n)
                                for n in self.nxGraph.neighbors(toadd[1])
                                if not (toadd[1], n) in checked_edges]
        return True

    def get_colour_signal(self):
        """
        Returns the node_map as a list of one hot encodings, which can be taken
        as a signal of the node of our graph for downstream ML tasks. If the
        conditions of Theorem 3.1 on the base Graph are satisfied, this is
        equivalent to a one-hot-encoding WL colouring.
        Note: If you want to train a model with this node signal using
        pytorch and pytorch-geometric, use torch.from_numpy to turn it into
        a torch tensor.

        Returns :
        ----------

        signal : np.array of shape (degree*n, n), where n is number of nodes
                 in base graph, and degree is the degree of the cover
                 The ith element is a vector of dimension n, which is zero
                 everywhere, except at the jth dimension where j the the
                 base graph's node that i cover.
        """
        num_nodes = self.nxGraph.order()
        signal = np.zeros((num_nodes, num_nodes//self.degree), dtype=np.float)
        for i in range(num_nodes):
            signal[i][self.node_map[i]] = 1.
        return signal

    def plot(self, coord=None):
        """
        Plots the cover in 3 dimensions, by using the third dimension for
        fibers over a 2 dimenisonal embedding of the base cover

        Parameters
        ----------
        coord : Optional: Dictionary or None
                If available, the user can input a dictionary which defines
                a 2 dimensional embedding of the nodes of the base graph as
                follows: The keys are node indices of the base graph. The
                values are length 2 list, corresponding to the embedding of
                the key's node in the 2 dimensional plane.
                If None, then 2dimensional embeddings are obtained from a
                spring layout of the base graph.

        Returns
        ----------
        fig : a plotly graph object that one can vizualise with .show()
        """

        G = self.nxGraph
        base = nx.Graph(self.base_edge_index)
        if coord is None:
            # if no coordinates are available, compute some with spring layout
            coord = nx.spring_layout(base, seed=100)

        # fix the coordinates
        base_num_nodes = base.number_of_nodes()
        for i in range(base_num_nodes):
            base.nodes[i]['pos'] = (coord[i][0], coord[i][1], 0.)

        # set cover coordinates
        size = len(G.nodes())
        scale = 0.5
        for i in range(size):
            G.nodes[i]['pos'] = (base.nodes[i % base_num_nodes]['pos'][0],
                                 base.nodes[i % base_num_nodes]['pos'][1],
                                 base.nodes[i % base_num_nodes]['pos'][2]+scale*float(i//base_num_nodes))

        edge_x = []
        edge_y = []
        edge_z = []

        # Get coordinates of edges
        for edge in G.edges():
            x0, y0, z0 = G.nodes[edge[0]]['pos']
            x1, y1, z1 = G.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)
            edge_z.append(z1)
            edge_z.append(None)

        # plot all edges
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_z = []

        # get node coordinates
        for node in G.nodes():
            x, y, z = G.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

        # plot nodes
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                reversescale=True,
                color=np.arange(0, 1, 1/len(G.nodes())),
                size=10,
                line_width=2))

        # plot figure
        fig = go.Figure(data=[edge_trace, node_trace])

        # setting the right zoom and size
        fig.update_layout(width=1000,
                          height=700,
                          scene=dict(aspectratio=dict(x=1, y=0.2, z=0.7)),
                          showlegend=False
                          )

        # Removing background etc
        fig.update_layout(scene=dict(
                        xaxis=dict(
                            showbackground=False,
                            showgrid=False,
                            visible=False,
                        ),
                        yaxis=dict(
                            showbackground=False,
                            showgrid=False,
                            visible=False,
                        ),
                        zaxis=dict(
                            showbackground=False,
                            showgrid=False,
                            visible=False,
                        ),
                        camera=dict(
                                up=dict(
                                    x=0,
                                    y=0,
                                    z=1
                                ),
                                eye=dict(
                                    x=0.1,
                                    y=-1,
                                    z=0.5,
                                )
                            ),
                    ))
        return fig


def unique_edge(edge_index):
    """
    Computes unique direct representatives of each undirected edges
    in edge_index. Returns a list where for every (i, j) in edge_index,
    exactly one of (i, j) and (j, i) is in the list.

    Parameters
    ----------
    edge_index : List of edges describing a graph. Each element is
                 another list of length 2,and is equal to [i, j] if and
                 only if the ith node is connected to the jth node.

    Returns :
    ----------
    unique: List of tuples. Each element in the list is a tuple (i, j).
            For each node i connected to node j, edge_index contains
            both [i, j] and [j, i], but unique only keeps one.
    """
    unique = []
    for e in edge_index:
        if (not (e[0], e[1]) in unique) and (not (e[1], e[0]) in unique):
            unique += [(e[0], e[1])]
    return unique


def gen_graphCovers(base_edge_index, degree, nb_covers=2, cycle_edge=None,
                    use_nx=False):
    """
    Generates all (or nb_covers) isomorphism types of degree `degree`
    covers of the graph described byedge_index. If cycle_edge is chosen such
    that it consists of one edge per cycle for every cycle, such that
    no cycles have the same edge, then it covers all desired isomorphism
    classes. Having such a cycle_edge speeds up the algorithm considerably.
    Note that in order to ensure that the number of graphs generated is
    large, we need the base graph to satisfy the conditions of Theorem 3.1.
    TODO: Compute cycle_edge automatically.

    Parameters
    ----------
    base_edge_index : List of length 2 list/tuples
                      Represents edges present in the base graph. Each
                      element is a length tuple (i, j) indicating that the
                      ith node is connected to the jth node by an edge.
                      Since we are dealing with undirected graphs, if
                      (i, j) is in the list, so is (j, i).
    degree : Int, greater than 1.
             Represents the degree (or number of sheets) of the cover to
             generate.
    cycle_edge : Optional: None or List of length 2 lists/tuples
                 If None then all possible permutations are tested, which
                 is very slow. Otherwise, each element represents the edge
                 for which we want nontrivial perumatations in the cover.
                 If the there is one element for each cycle (for example
                 the complement of a spanning tree), then the isomorphism
                 classes will all be represented. When the base graph is
                 connected, the length of this list should be equal to the
                 1-chi(G), where chi is the Euler Characteristic.
                 This allows to speed up by a lot.
    nb_covers : Option Int or None
           If it is an Int, then it is the number of desired cover. Note
           that if it is too big, it won't be possible to generate nb_covers
           covers. If None, all isomorphism classes are generated
    use_nx : Boolean,
            We recommend setting this to False
            If True, we use networkx's function .is_isomorphic to check for
            isomorphism (as graphs) of two covers.
            Note that this is much much slower than our  implementation.
            It was mainly used for testing our implementation. Can also be
            used if the base graph does not satisfy conditions of Theorem 3.1.
            If False, our custom implementation is used.

    Returns
    ----------
    graphCovers : List of Covers object
                  The size of the list is either `nb_covers`, or the number
                  of pairwise non-simorphic degree `degree` covers. Each
                  element of the list is a Cover object, and they are
                  pairwise non-isomorphic.

    """
    unique = unique_edge(base_edge_index)
    if cycle_edge is None:
        cycle_edge_index = unique
    else:
        # make sure `cycle_edge_index` match with `unique` representatives
        cycle_edge_index = [i for i, e in enumerate(unique)
                            if ([e[0], e[1]] in cycle_edge)
                            or ([e[1], e[0]] in cycle_edge)]
    graphCovers = []
    print("Total number of covers to check: ",
          factorial(degree)**len(cycle_edge))

    for i, mini_sig in enumerate(product(permutations(range(degree)),
                                         repeat=len(cycle_edge))):
        # this loop enumerates all candidate covers

        # from minisig, we create the corresponding sigma
        # start with identity everywhere:
        sigma = [tuple(range(degree)) for i in range(len(unique))]
        for j, x in enumerate(cycle_edge_index):
            # change the permutation of edges in distinguished elements:
            sigma[x] = mini_sig[j]
        sigma = tuple(sigma)

        cover = Cover(base_edge_index, degree, sigma)
        if nx.is_connected(cover.nxGraph):
            # TODO: generalize to disconnected. To do this we can either
            #       use nx.is_isom, or extend our custom isom algorithm

            is_new = True  # checks if this is a new graph
            for old in graphCovers:
                if use_nx:
                    if nx.is_isomorphic(cover.nxGraph,
                                        old.nxGraph):
                        is_new = False
                        break
                else:
                    if cover.isom(old):
                        is_new = False
                        break
            if is_new:
                # if the isomorphism class is not in graphCovers,
                # add it to the list:
                graphCovers.append(cover)

        if (nb_covers is not None) and len(graphCovers) == nb_covers:
            print("We found ", nb_covers, " covers ! Only in ", i, " tries!")
            return graphCovers
    print("There are ", len(graphCovers), " degree ", degree, " covers!")
    return graphCovers
