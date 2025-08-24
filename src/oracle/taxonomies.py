import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import graphviz_layout

from oracle.constants import BTS_to_Astrophysical_mappings

# NOTE: I am not super worried about the performance of this code since it will not be used to compute the losses while training the model. That code exists in loss.py and is optimized for speed.

# Name of the root node for the self
root_label = "Alert"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# Generic class to represent a taxonomy. Specific taxonomy classes inherit from this class and add their own nodes and edges.
class Taxonomy(nx.DiGraph):
    """
    Class to represent a taxonomy as a directed graph. 
    """

    def __init__(self, directed=True, **attr):

        super().__init__(directed=directed, **attr)
        self.root_label = root_label

    def plot_taxonomy(self):
        """
        Plot the taxonomy using networkx and matplotlib.

        returns:
            None
        """

        # Plot the taxonomy.
        pos = graphviz_layout(self, prog='dot')
        nx.draw_networkx(self, with_labels=True, font_weight='bold', arrows=True, font_size=8, pos=pos, alpha=0.7)
        plt.show()

    def plot_colored_taxonomy(self, probabilities):
        """
        Plot the taxonomy, colored by values in 
        """

        level_order_traversal = self.get_level_order_traversal()

        color_map = {node: label for node, label in zip(level_order_traversal, probabilities.tolist())}
        node_colors = [color_map[node] for node in self.nodes()]

        pos = graphviz_layout(self, prog='dot')
        nx.draw_networkx(self, with_labels=True, font_weight='bold', arrows=True, node_color=node_colors, font_size = 8, pos=pos, cmap='Wistia')
        plt.tight_layout()

        plt.show()

    def get_level_order_traversal(self):
        """
        Get the level order traversal of the taxonomy.

        returns:
            level_order_nodes: list of nodes in the taxonomy ordered by level order traversal.
        """

        # Do a level order traversal of the tree to get an ordering of the nodes.
        level_order_nodes = nx.bfs_tree(self, source=self.root_label).nodes()
        return level_order_nodes
    
    def get_parent_nodes(self):
        """
        Get the parent nodes for each node in the taxonomy, ordered by level order traversal.

        returns:
            parents: list of parent nodes for each node in the taxonomy, ordered by level order traversal.
        """

        # Get the parent nodes for each node in the taxonomy, ordered by level order traversal.
        level_order_nodes = self.get_level_order_traversal()
        parents = [list(self.predecessors(node)) for node in level_order_nodes]

        # Make sure the root node has no parent
        for idx in range(len(parents)):

            # Make sure the graph is a tree.
            assert len(parents[idx]) == 0 or len(parents[idx]) == 1, 'Number of parents for each node should be 0 (for root) or 1.'
            
            if len(parents[idx]) == 0: # Make an exception for the root node since it should have no parent.
                parents[idx] = ''
            else:
                parents[idx] = parents[idx][0]

        return parents
    
    def get_sibling_masks(self):
        """
        Get the sibling masks for each node in the taxonomy, ordered by level order traversal.

        returns:
            masks: list of numpy arrays of sibling mask indices for the taxonomy.
        """

        # NOTE: Sibling nodes are nodes that share the same parent node.

        # Get a list of all parents for all nodes in the taxonomy, ordered by level order traversal.
        parent_nodes = self.get_parent_nodes()

        # Get the unique parents, sorted alphabetically
        unique_parent_nodes = list(set(parent_nodes))
        unique_parent_nodes.sort()

        # Create a mask for sibling nodes.
        masks = []
        for parent in unique_parent_nodes:
            masks.append(np.where(np.array(parent_nodes) == parent, 1, 0))

        return masks
    
    def get_leaf_nodes(self):
        """
        Get the leaf nodes in the taxonomy.

        returns:
            leaf_nodes: list of leaf nodes in the taxonomy.
        """

        # Get the leaf nodes in the taxonomy
        leaf_nodes = [x for x in self.nodes() if self.out_degree(x)==0 and self.in_degree(x)==1]
        return leaf_nodes
    
    def get_hierarchical_one_hot_encoding(self, labels):

        """
        Get the hierarchical one-hot encoding for the labels.
        
        args:
            labels: list of labels to encode (n_samples). 

        returns:
            all_encodings: 2D numpy array of hierarchical one-hot encodings (n_samples, n_nodes).
        """
        level_order_nodes = self.get_level_order_traversal()

        # Assert that all labels are one of the nodes in the taxonomy.
        for label in labels:
            assert label in level_order_nodes, f'Label {label} is not a node in the taxonomy.'

        # Array to store all the label encodings
        all_encodings = np.zeros([len(labels), len(level_order_nodes)])

        # Loop through all the labels 
        for i, label in enumerate(labels):
            
            # Get the path from the root to the label. Shortest path is guaranteed to be unique.
            path = nx.shortest_path(self, source=root_label, target=label)

            # Loop through all the nodes in the the path from the root to the leaf label.
            for node in path:
                
                # Get the index of the node in the level order traversal.
                idx = list(level_order_nodes).index(node)

                # Encoding of the labels where true nodes are 1 and everything else is 0. Ordered by level order traversal.
                all_encodings[i,idx] = 1

        return all_encodings
    
    def get_paths(self, labels):
        """
        Get the paths from the root to the labels.

        args:
            labels: list of labels to compute paths for (n_samples). Each label should be a leaf node in the taxonomy.

        returns:
            paths: list of paths from the root to the labels.
        """

        encodings = self.get_hierarchical_one_hot_encoding(labels)
        level_order_nodes = np.array(self.get_level_order_traversal())

        paths = []
        for i in range(encodings.shape[0]):
            idx = np.where(encodings[i,:]==1)[0]
            path = level_order_nodes[idx].tolist()
            paths.append(path)  

        return paths
    
    def get_depths(self):
        """
        Get the depths of the nodes in the taxonomy, ordered by level order traversal.

        returns:
            depths: list of depths of the nodes in the taxonomy, ordered by level order traversal.
        """
    
        path_lengths = []
        level_order_nodes = self.get_level_order_traversal()

        # Compute the shortest paths from the root node to each of the other nodes in the tree.
        for node in level_order_nodes:
            path_lengths.append(len(nx.shortest_path(self, root_label, node)) - 1)

        return np.array(path_lengths)
    
    def get_conditional_probabilities(self, logits, epsilon=1e-10):
        """
        Get the conditional probabilities from the logits, after applying the softmax function to sets of sibling nodes.
        
        args:
            logits: 2D torch tensor of logits (n_samples, n_nodes).
            epsilon: small value to avoid numerical problems with floating point numbers.
            
        returns:
            conditional_probabilities: 2D torch tensor of conditional probabilities (n_samples, n_nodes)."""

        masks = self.get_sibling_masks()

        for mask in masks:
            
            mask = torch.from_numpy(mask).to(device=device)

            # Get the e^logits
            exps = torch.exp(logits)

            # Multiply (dot product) the e^logits with the mask to maintain just the e^logits values that belong to this mask. All other values will be zeros.
            masked_exps = torch.multiply(exps, mask)

            # Find the sum of the e^logits values that belong to the mask. Do this for each element in the batch separately. Add a small value to avoid numerical problems with floating point numbers.
            masked_sums = torch.sum(masked_exps, dim=1, keepdim=True) + epsilon

            # Compute the softmax by dividing the e^logits with the sum (e^logits)
            softmax = masked_exps/masked_sums

            # (1 - mask) * y_pred gets the logits for all the values not in this mask and zeros out the values in the mask. Add those back so that we can repeat the process for other masks.
            logits = softmax + ((1 - mask) * logits)

        return logits
    
    def get_nodes_by_depth(self):
        """
        Get the nodes in the taxonomy grouped by their depth.

        returns:
            nodes_by_depth: dictionary where keys are depths and values are lists of nodes at that depth.
        """

        depths = self.get_depths()
        level_order_traversal = np.array(self.get_level_order_traversal())
        leaf_nodes = self.get_leaf_nodes()

        nodes_by_depth = {}

        # Loop through all depths
        for d in np.unique(depths):
            
            # Get the indices of the nodes at this depth in the level order traversal.
            idx = np.where(depths==d)[0]
            nodes_by_depth[int(d)] = level_order_traversal[idx]

        # Add the leaf nodes to the nodes_by_depth dictionary.
        nodes_by_depth[-1] = np.array(leaf_nodes)

        return nodes_by_depth
            
    def get_class_probabilities(self, conditional_probabilities):
        """
        Get the class probabilities from the conditional probabilities.

        args:
            conditional_probabilities: 2D numpy array of conditional probabilities (n_samples, n_nodes).

        returns:
            class_probabilities: 2D numpy array of class probabilities (n_samples, n_nodes).
        """

        assert conditional_probabilities.shape[1] == len(self.nodes()), 'Number of nodes in the taxonomy should match the number of columns in the conditional probabilities.'

        # TODO: This could probably be sped up. I am implementing the dumb version for now but we might want to optimize this, especially if we want to deploy since this will be used during inference.

        # Conditional probabilities are the output of the model. We use those to compute the class probabilities.
        class_probabilities = torch.ones(conditional_probabilities.shape)

        # Get the leaf nodes for the taxonomy.
        level_order_nodes = self.get_level_order_traversal()

        for i, node in enumerate(level_order_nodes):
            
            # Skip the root node since its class probability should be equal to its conditional probability and both should be 1.
            if i != 0: 
            
                # Get the path from the root to the another node in the taxonomy.
                path = nx.shortest_path(self, source=root_label, target=node)

                # Get the indices of the nodes in the path.
                indices = [list(self.get_level_order_traversal()).index(n) for n in path]

                # Multiply the conditional probabilities of the nodes in the path to get the class probability.
                class_probabilities[:,i] = torch.prod(conditional_probabilities[:,indices], dim=1)

        return class_probabilities

class BTS_Taxonomy(Taxonomy):

    def __init__(self, **attr):

        super().__init__(**attr)
        self.add_node(root_label)

        level_1_nodes = ['Persistent','Transient']
        self.add_nodes_from(level_1_nodes)
        self.add_edges_from([(root_label, node) for node in level_1_nodes])

        level_2a_nodes = ['AGN','CV']
        self.add_nodes_from(level_2a_nodes)
        self.add_edges_from([('Persistent', node) for node in level_2a_nodes])

        # Level 2b nodes for SN-like events
        level_2b_nodes = ['SN-Ia','SN-CC']
        self.add_nodes_from(level_2b_nodes)
        self.add_edges_from([('Transient', node) for node in level_2b_nodes])

        level_3a_nodes = ['SN-II','SN-Ib/c','SLSN']
        self.add_nodes_from(level_3a_nodes)
        self.add_edges_from([('SN-CC', node) for node in level_3a_nodes])


class ORACLE_Taxonomy(Taxonomy):

    def __init__(self, **attr):

        super().__init__(**attr)
        self.add_node(root_label)

        # Level 1
        level_1_nodes = ['Transient', 'Variable']
        self.add_nodes_from(level_1_nodes)
        self.add_edges_from([(root_label, level_1_node) for level_1_node in level_1_nodes])

        # Level 2a nodes for Transients
        level_2a_nodes = ['SN', 'Fast', 'Long']
        self.add_nodes_from(level_2a_nodes)
        self.add_edges_from([('Transient', level_2a_node) for level_2a_node in level_2a_nodes])

        # Level 2b nodes for Transients
        level_2b_nodes = ['Periodic', 'AGN']
        self.add_nodes_from(level_2b_nodes)
        self.add_edges_from([('Variable', level_2b_node) for level_2b_node in level_2b_nodes])

        # Level 3a nodes for SN Transients
        level_3a_nodes = ['SNIa', 'SNIb/c', 'SNIax', 'SNI91bg', 'SNII']
        self.add_nodes_from(level_3a_nodes)
        self.add_edges_from([('SN', level_3a_node) for level_3a_node in level_3a_nodes])

        # Level 3b nodes for Fast events Transients
        level_3b_nodes = ['KN', 'Dwarf Novae', 'uLens', 'M-dwarf Flare']
        self.add_nodes_from(level_3b_nodes)
        self.add_edges_from([('Fast', level_3b_node) for level_3b_node in level_3b_nodes])

        # Level 3c nodes for Long events Transients
        level_3c_nodes = ['SLSN', 'TDE', 'ILOT', 'CART', 'PISN']
        self.add_nodes_from(level_3c_nodes)
        self.add_edges_from([('Long', level_3c_node) for level_3c_node in level_3c_nodes])

        # Level 3d nodes for periodic stellar events
        level_3d_nodes = ['Cepheid', 'RR Lyrae', 'Delta Scuti', 'EB'] 
        self.add_nodes_from(level_3d_nodes)
        self.add_edges_from([('Periodic', level_3d_node) for level_3d_node in level_3d_nodes])

if __name__=='__main__':

    #<-- Example usage of the taxonomy class -->

    taxonomy = BTS_Taxonomy()
    taxonomy.plot_taxonomy()

    all_classes = list(BTS_to_Astrophysical_mappings.values())
    taxonomy.get_hierarchical_one_hot_encoding(all_classes)
    taxonomy.plot_colored_taxonomy(taxonomy.get_hierarchical_one_hot_encoding(['SN-II'])[0])

    print(taxonomy.get_nodes_by_depth())


    # Create a taxonomy object
    taxonomy = ORACLE_Taxonomy()

    # Print the nodes after doing a level order traversal
    print(taxonomy.get_level_order_traversal())
    print(taxonomy.get_parent_nodes())
    print(taxonomy.get_sibling_masks())
    print(taxonomy.get_leaf_nodes())
    print(taxonomy.get_hierarchical_one_hot_encoding(['SNIa', 'SNIb/c', 'SNIax', 'SNI91bg', 'SNII', 'SN']))
    print(taxonomy.get_paths(['SNIa', 'SNIb/c', 'SNIax', 'SNI91bg', 'SNII']))
    print(taxonomy.get_class_probabilities(np.random.rand(10, len(taxonomy.nodes()))))
    print(taxonomy.get_conditional_probabilities(torch.from_numpy(np.random.rand(10, len(taxonomy.nodes())))))

    taxonomy.plot_taxonomy()
