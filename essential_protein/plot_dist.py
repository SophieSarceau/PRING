import pickle
import argparse
import networkx as nx
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_uniprot_ids(txt_path):
    ids = []
    for line in open(txt_path):
        line = line.strip()
        ids.append(line)

    return ids


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze protein network centrality.")
    parser.add_argument("--essential-proteins", type=str, default="essential_proteins.txt",
                        help="Path to the file with essential protein IDs.")
    parser.add_argument("--non-essential-proteins", type=str, default="non_essential_proteins.txt",
                        help="Path to the file with non-essential protein IDs.")
    parser.add_argument("--ppi-graph", type=str, 
                        help="Path to the human PPI graph pickle file.")
    parser.add_argument("--output-file", type=str, default="network_centrality_distribution.png",
                        help="Path to save the output plot.")
    
    args = parser.parse_args()

    ids = read_uniprot_ids(args.essential_proteins)
    print("The number of essential proteins is: ", len(ids))
    non_ids = read_uniprot_ids(args.non_essential_proteins)
    print("The number of non-essential proteins is: ", len(non_ids))

    human_ppi_graph = pickle.load(open(args.ppi_graph, "rb"))
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
    plt.savefig(args.output_file)
