import os
import networkx as nx
import pickle
import random
from tqdm import tqdm
from collections import deque
import numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    # Compatible method for setting NetworkX randomness
    # This works regardless of NetworkX version
    random_state = random.Random(seed)
    
    # For modern NetworkX versions
    if hasattr(nx, "random"):
        nx.random.seed = seed
    
    # For older NetworkX versions
    try:
        nx.utils.random_sequence.seed(seed)
    except AttributeError:
        pass
        
    os.environ['PYTHONHASHSEED'] = str(seed)  # For hash randomization

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

def read_ppis(file_path):
    ppis = set()
    with open(file_path, "r") as f:
        for line in f:
            prot1, prot2 = line.strip().split()
            ppis.add((prot1, prot2))
    return ppis

def sample_negatives(block_pos, block_neg, all_ppis, factor=1):
    if block_neg is None:
        block_neg = set()
    size = len(block_neg)
    # should lead to power law distribution
    candidates = [ppi[0] for ppi in block_pos]
    candidates.extend([ppi[1] for ppi in block_pos])
    to_generate = (factor * len(block_pos)) - size
    while size < (factor * len(block_pos)):
        prot1 = random.choice(tuple(candidates))
        prot2 = random.choice(tuple(candidates))
        while prot1 == prot2 or (prot1, prot2) in all_ppis or (prot2, prot1) in all_ppis or (prot2, prot1) in block_neg:
            prot2 = random.choice(tuple(candidates))
        block_neg.add((prot1, prot2))
        if to_generate % 1000 == 0:
            print(f'Still {to_generate} proteins left to generate!')
        size = len(block_neg)
        to_generate = (factor * len(block_pos)) - size
    return list(block_neg)

# seed_everything(40)

species_list = ['ecoli', 'yeast', 'arath']
base_path = "../species_processed_data/"

for species in tqdm(species_list):
    total_graph_path = os.path.join(base_path, species, f'{species}_graph.pkl')
    pos_ppis = read_ppis(os.path.join(base_path, species, f'{species}_ppi.txt'))
    total_graph = pickle.load(open(total_graph_path, "rb"))

    print("The total number of ppi pairs: ", len(pos_ppis))
    print("The total number of edges in the graph: ", total_graph.number_of_edges())

    random.seed(40)
    prot_num = total_graph.number_of_nodes()
    if prot_num > 1000:
        # sample 1000 nodes using BFS method
        root_node = select_root_node(list(total_graph.nodes), total_graph, 10)
        sampled_nodes = traverse(total_graph, root_node, 1000, 'BFS')

    # extract the subgraph
    subgraph = total_graph.subgraph(sampled_nodes)
    print("The number of nodes in the subgraph: ", subgraph.number_of_nodes())
    print("The number of edges in the subgraph: ", subgraph.number_of_edges())

    test_graph = subgraph
    test_pos_ppis = list(set(test_graph.edges))

    # sample negative ppi pairs for each method
    test_neg_ppis = sample_negatives(test_pos_ppis, None, pos_ppis, factor=1)
    print("The number of negative ppi pairs: ", len(test_neg_ppis))

    # combine positive and negative ppi pairs
    all_ppis = [(prot1, prot2, 1) for prot1, prot2 in test_pos_ppis]
    all_ppis.extend([(prot1, prot2, 0) for prot1, prot2 in test_neg_ppis])
    print("The total number of ppi pairs: ", len(all_ppis))

    # save the combined ppi pairs to the txt file
    with open(os.path.join(base_path, species, f'{species}_test_ppi.txt'), "w") as f:
        for prot1, prot2, label in all_ppis:
            f.write(f'{prot1}\t{prot2}\t{label}\n')

    # save the test_graph to the pkl file
    pickle.dump(test_graph, open(os.path.join(base_path, species, f'{species}_test_graph.pkl'), "wb"))
    print(f"Finish processing {species} dataset!")
