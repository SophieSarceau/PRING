import networkx as nx
import random
import pickle
from collections import deque
import os
import numpy as np
from tqdm import tqdm


species_list = ['yeast', 'arath', 'ecoli', 'human']
method_list = ['BFS', 'DFS', 'RANDOM_WALK']
base_path = '../species_processed_data/'
sample_node_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
graph_num = 50

def sample_nodes(graph, num_nodes, method='BFS'):
    """
    Sample nodes from the graph using the specified method.

    Parameters:
    - graph: The input graph (networkx.Graph)
    - num_nodes: The number of nodes to sample
    - method: The sampling method ('BFS', 'DFS', 'RANDOM_WALK')

    Returns:
    - sampled_nodes: A set of sampled nodes
    """
    assert method in ['BFS', 'DFS', 'RANDOM_WALK'], "Method must be one of 'BFS', 'DFS', or 'RANDOM_WALK'"

    nodes = list(graph.nodes)
    start_node = random.choice(nodes)
    visited = set()

    if method == 'RANDOM_WALK':
        current = start_node
        visited.add(current)

        while len(visited) < num_nodes:
            neighbors = list(set(graph.neighbors(current)) - visited)
            if not neighbors:
                escape_options = [node for node in visited if set(graph.neighbors(node)) - visited]
                if not escape_options:
                    break
                current = random.choice(escape_options)
            else:
                current = random.choice(neighbors)
                visited.add(current)
    else:
        stack_or_queue = deque([start_node])

        while stack_or_queue and len(visited) < num_nodes:
            current = stack_or_queue.popleft() if method == 'BFS' else stack_or_queue.pop()
            if current not in visited:
                visited.add(current)
                neighbors = sorted(list(set(graph.neighbors(current)) - visited))
                stack_or_queue.extend(neighbors)

    return visited

if __name__ == "__main__":
    for species in species_list:
        if species == 'human':
            for method in method_list:
                print("Sampling nodes for", species, method)
                graph_path = os.path.join(base_path, species, method, 'human_test_graph.pkl')
                test_graph = pickle.load(open(graph_path, 'rb'))
                sampled_nodes_dict = {
                    num_nodes: [] for num_nodes in sample_node_list
                }
                for num_nodes in tqdm(sample_node_list):
                    for i in range(graph_num):
                        sampled_nodes = sample_nodes(test_graph, num_nodes, method)
                        sampled_nodes_dict[num_nodes].append(sampled_nodes)
                output_path = os.path.join(base_path, species, method, 'test_sampled_nodes.pkl')
                pickle.dump(sampled_nodes_dict, open(output_path, 'wb'))
        else:
            for method in method_list:
                print("Sampling nodes for", species, method)
                graph_path = os.path.join(base_path, species, species + '_test_graph.pkl')
                graph = pickle.load(open(graph_path, 'rb'))
                sampled_nodes_dict = {
                    num_nodes: [] for num_nodes in sample_node_list
                }
                for num_nodes in tqdm(sample_node_list):
                    for i in range(graph_num):
                        sampled_nodes = sample_nodes(graph, num_nodes, method)
                        sampled_nodes_dict[num_nodes].append(sampled_nodes)
                output_path = os.path.join(base_path, species, species + '_' + method + '_sampled_nodes.pkl')
                pickle.dump(sampled_nodes_dict, open(output_path, 'wb'))
