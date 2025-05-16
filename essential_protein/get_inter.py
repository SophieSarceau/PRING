import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns

def read_uniprot_ids(txt_path):
    ids = []
    for line in open(txt_path):
        line = line.strip()
        ids.append(line)

    return ids

import networkx as nx

def calculate_LCCDC(G: nx.Graph) -> dict:
    LCCDC = {}
    for node in tqdm(G.nodes()):
        neighbors = list(G.neighbors(node))
        k = len(neighbors)  # deg(i)
        
        if k < 2:
            LCCDC[node] = 0
            continue

        # Step 1: Count actual links (APL) among neighbors
        subgraph = G.subgraph(neighbors)
        APL = subgraph.number_of_edges()

        # Step 2: Calculate maximum possible links (MPL)
        MPL = k * (k - 1) / 2

        # Step 3: Compute LCC
        LCC = APL / MPL

        # Step 4: Compute LCCDC
        LCCDC[node] = (1 - LCC) * k

    return LCCDC


def compute_betweenness_centrality(G):
    # normalized=True gives values between 0 and 1
    bc = nx.betweenness_centrality(G, normalized=True)
    return bc

def compute_network_centrality(G):
    nc_dict = {}

    for u in tqdm(G.nodes()):
        neighbors_u = set(G.neighbors(u))
        nc = 0

        for v in neighbors_u:
            neighbors_v = set(G.neighbors(v))
            common = neighbors_u & neighbors_v  # intersection

            denom = min(len(neighbors_u) - 1, len(neighbors_v) - 1)
            if denom > 0:
                ecc = len(common) / denom
                nc += ecc
        nc_dict[u] = nc

    return nc_dict


def compute_MNC(G):
    mnc_dict = {}

    for node in tqdm(G.nodes()):
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 0:
            mnc_dict[node] = 0
            continue

        # Induced subgraph from neighbors
        subgraph = G.subgraph(neighbors)
        # Get size of largest connected component
        components = nx.connected_components(subgraph)
        largest_cc_size = max((len(c) for c in components), default=0)

        mnc_dict[node] = largest_cc_size

    return mnc_dict

def compute_DMNC(G, epsilon=1.7):
    dmnc_dict = {}

    for node in tqdm(G.nodes()):
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            dmnc_dict[node] = 0
            continue

        # Induced subgraph of neighbors
        subgraph = G.subgraph(neighbors)
        # Find largest connected component
        components = list(nx.connected_components(subgraph))
        largest_cc = max(components, key=len)
        cc_subgraph = G.subgraph(largest_cc)

        num_edges = cc_subgraph.number_of_edges()
        num_nodes = cc_subgraph.number_of_nodes()

        if num_nodes > 0:
            dmnc = num_edges / (num_nodes ** epsilon)
        else:
            dmnc = 0
        dmnc_dict[node] = dmnc

    return dmnc_dict


def compute_subgraph_centrality(G):
    A = nx.to_numpy_array(G)
    eigvals, eigvecs = np.linalg.eigh(A)  # A is symmetric if G is undirected
    SC = {}

    for i in tqdm(range(len(G.nodes()))):
        sc_u = sum((eigvecs[i, j] ** 2) * np.exp(eigvals[j]) for j in range(len(eigvals)))
        node = list(G.nodes())[i]
        SC[node] = sc_u

    return SC

def compute_pagerank(G, damping=0.85):
    pr = nx.pagerank(G, alpha=damping)
    return pr

def compute_SLC(G):
    slc = {}

    for i in tqdm(G.nodes()):
        neighbors_i = G.neighbors(i)
        slc_i = 0
        for j in neighbors_i:
            neighbors_j = G.neighbors(j)
            q_j = sum(len(list(G.neighbors(w))) for w in neighbors_j)
            slc_i += q_j
        slc[i] = slc_i

    return slc

if __name__ == "__main__":
    ids = read_uniprot_ids("essential_proteins.txt")
    print("The number of essential proteins is: ", len(ids))
    non_ids = read_uniprot_ids("non_essential_proteins.txt")
    print("The number of non-essential proteins is: ", len(non_ids))

    human_ppi_graph = pickle.load(open("human_test_graph.pkl", "rb"))
    human_ppi_nodes = set(human_ppi_graph.nodes())
    print("The length of human ppi nodes is: ", len(human_ppi_nodes))

    essential_proteins = set(ids).intersection(human_ppi_nodes)
    print("The number of essential proteins in human ppi is: ", len(essential_proteins))

    non_essential_proteins = set(non_ids).intersection(human_ppi_nodes)
    print("The number of non-essential proteins in human ppi is: ", len(non_essential_proteins))

    # Calculate degree centrality
    id_degree = {}
    for node in human_ppi_graph.nodes():
        id_degree[node] = human_ppi_graph.degree(node)

    # Calculate network centrality
    network_centrality_dict = compute_network_centrality(human_ppi_graph)

    # Create ranking dictionaries for each metric
    network_centrality_ranking = {node: rank for rank, node in enumerate(sorted(human_ppi_nodes, 
                                                                   key=lambda x: network_centrality_dict[x], 
                                                                   reverse=True))}

    # For comparison - print individual results too
    network_top100 = set(sorted(human_ppi_nodes, key=lambda x: network_centrality_dict[x], reverse=True)[:100])
    # Last 100 nodes in the sorted list
    network_last100 = set(sorted(human_ppi_nodes, key=lambda x: network_centrality_dict[x], reverse=True)[-100:])

    # convert essential_proteins to a set for faster lookup
    essential_proteins = set(essential_proteins)
    print("The number of essential proteins in the top 100 (network centrality): ", 
          len(network_top100.intersection(essential_proteins)))

    non_essential_proteins = set(non_essential_proteins)
    print("The number of non-essential proteins in the top 100 (network centrality): ", 
          len(network_top100.intersection(non_essential_proteins)))
    print("The number of essential proteins in the last 100 (network centrality): ",
            len(network_last100.intersection(essential_proteins)))

    # Get the network centrality score for essential proteins and the non-essential proteins
    essential_prot_network_centrality = {k: network_centrality_dict[k] for k in essential_proteins}
    non_essential_prot_network_centrality = {k: network_centrality_dict[k] for k in non_essential_proteins}
    # plot the distribution of network centrality scores
    plt.figure(figsize=(6, 6))
    sns.kdeplot(list(essential_prot_network_centrality.values()), label='Essential Proteins', color='blue')
    sns.kdeplot(list(non_essential_prot_network_centrality.values()), label='Non-Essential Proteins', color='red')
    plt.title("Network Centrality Distribution")
    plt.xlabel("Network Centrality")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("network_centrality_distribution.png")
