import networkx as nx
import random
import pickle
from collections import deque
import matplotlib.pyplot as plt
import os
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # Set NetworkX global random seed correctly
    nx.utils.random_sequence.MAPPING_SEED = seed  # For older versions of NetworkX
    try:
        nx.utils.random_sequence.seed(seed)
    except AttributeError:
        pass  # For some NetworkX versions this might not exist

def select_root_node(nodes, G, threshold):
    for node in nodes:
        if len(list(G.neighbors(node))) == threshold:
            return node
    raise ValueError("No suitable root node found with given threshold.")

def traverse(graph, start_node, size_limit, method):
    visited = set()

    if method == 'RANDOM_WALK':
        # Random walk implementation
        current = start_node
        visited.add(current)
    
        while len(visited) < size_limit:
            neighbors = list(set(graph.neighbors(current)) - visited)
            if not neighbors:
                # If stuck in a dead end, choose a random node from visited nodes
                # that still has unvisited neighbors
                escape_options = []
                for node in visited:
                    unvisited_neighbors = set(graph.neighbors(node)) - visited
                    if unvisited_neighbors:
                        escape_options.append(node)
                
                if not escape_options:
                    break  # No more nodes to explore
                
                current = random.choice(escape_options)
            else:
                current = random.choice(neighbors)
                visited.add(current)
    else:
        # Original BFS/DFS implementation
        stack_or_queue = deque([start_node])
        
        while stack_or_queue and len(visited) < size_limit:
            current = stack_or_queue.popleft() if method == 'BFS' else stack_or_queue.pop()
            if current not in visited:
                visited.add(current)
                # Sort neighbors to ensure deterministic ordering
                neighbors = sorted(list(set(graph.neighbors(current)) - visited))
                stack_or_queue.extend(neighbors)

    return visited

def partition_ppi_graph(G: nx.Graph, test_ratio=0.1, threshold=12, method='BFS'):
    assert method in ['BFS', 'DFS', 'RANDOM_WALK'], "Method must be one of 'BFS', 'DFS', or 'RANDOM_WALK'"

    # Check how many partitions are in the graph
    num_partitions = nx.number_connected_components(G)
    print("Number of partitions in the graph: ", num_partitions)

    # Find the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    print("Number of nodes in the largest connected component: ", len(largest_cc))
    print("Number of edges in the largest connected component: ", G.subgraph(largest_cc).number_of_edges())

    # set random seed
    # random.seed(1)

    nodes = list(G.nodes)
    print("Number of nodes in the graph: ", len(nodes))
    random.shuffle(nodes)

    total_nodes = len(nodes)
    test_size = int(total_nodes * test_ratio)

    # Test set
    test_root = select_root_node(nodes, G, threshold)
    test_nodes = traverse(G, test_root, test_size, method)

    # Validation set (avoid overlap)
    remaining_nodes = set(nodes) - test_nodes
    remaining_subgraph = G.subgraph(remaining_nodes)

    # Train set (remaining nodes)
    train_nodes = set(nodes) - test_nodes

    # calculate the statistics of each graph
    train_graph = G.subgraph(train_nodes)
    # val_graph = G.subgraph(val_nodes)
    test_graph = G.subgraph(test_nodes)
    # print the statistics: density, average degree, average clustering coefficient
    print("Train Graph: Density: {}, Average Degree: {}, Average Clustering Coefficient: {}".format(
        nx.density(train_graph), sum(dict(train_graph.degree()).values())/len(train_graph), nx.average_clustering(train_graph)))
    print("Train Graph: Nodes: {}, Edges: {}".format(len(train_nodes), train_graph.number_of_edges()))
    print("Test Graph: Density: {}, Average Degree: {}, Average Clustering Coefficient: {}".format(
        nx.density(test_graph), sum(dict(test_graph.degree()).values())/len(test_graph), nx.average_clustering(test_graph)))
    print("Test Graph: Nodes: {}, Edges: {}".format(len(test_nodes), test_graph.number_of_edges()))

    # check if there is self loop
    print("Train Graph: Self Loop: ", nx.number_of_selfloops(train_graph))
    # print("Validation Graph: Self Loop: ", nx.number_of_selfloops(val_graph))
    print("Test Graph: Self Loop: ", nx.number_of_selfloops(test_graph))

    return train_nodes, test_nodes


if __name__ == '__main__':
    # Set seed again before running main logic
    seed_everything(42)

    graph_path = '../species_processed_data/human/human_graph.pkl'
    human_graph = pickle.load(open(graph_path, 'rb'))

    # method: BFS, DFS, RANDOM_WALK
    method_list = ['BFS', 'DFS', 'RANDOM_WALK']
    for method in method_list:
        train_nodes, test_nodes = partition_ppi_graph(human_graph, test_ratio=0.2, threshold=5, method=method)
        split = {
            'train': train_nodes,
            'test': test_nodes
        }
        os.makedirs(f'../species_processed_data/human/{method}', exist_ok=True)
        with open(f'../species_processed_data/human/{method}/human_{method}_split.pkl', 'wb') as f:
            pickle.dump(split, f)
