import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns


def read_uniprot_ids(txt_path):
    ids = []
    for line in open(txt_path):
        line = line.strip()
        ids.append(line)

    return ids

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

    # Calculate network centrality
    network_centrality_dict = compute_network_centrality(human_ppi_graph)

    essential_dict = {prot: network_centrality_dict[prot] for prot in essential_proteins}
    non_essential_dict = {prot: network_centrality_dict[prot] for prot in non_essential_proteins}

    # Get the rank dict of essential and non-essential proteins
    essential_rank_dict = {k: v for k, v in sorted(essential_dict.items(), key=lambda item: item[1], reverse=True)}
    non_essential_rank_dict = {k: v for k, v in sorted(non_essential_dict.items(), key=lambda item: item[1], reverse=True)}

    # Filter essential proteins with score > 25 and select top 100
    essential_above_threshold = [(k, v) for k, v in essential_rank_dict.items() if v > 30]
    print(f"Number of essential proteins with score > 30: {len(essential_above_threshold)}")

    # Take top 100 essential proteins with score > 25 in descending order
    essential_selected = essential_above_threshold[-100:]
    if len(essential_selected) < 100:
        print(f"Warning: Only {len(essential_selected)} essential proteins with score > 25 available")

    # Filter non-essential proteins with score < 25 and select bottom 100
    non_essential_below_threshold = [(k, v) for k, v in non_essential_rank_dict.items() if v < 20]
    print(f"Number of non-essential proteins with score < 20: {len(non_essential_below_threshold)}")

    # Sort in ascending order and take the first 100
    non_essential_below_threshold.sort(key=lambda x: x[1])
    non_essential_selected = non_essential_below_threshold[-100:]
    if len(non_essential_selected) < 100:
        print(f"Warning: Only {len(non_essential_selected)} non-essential proteins with score < 25 available")

    # Extract protein IDs and scores
    essential_selected_ids = [item[0] for item in essential_selected]
    essential_selected_scores = [item[1] for item in essential_selected]
    non_essential_selected_ids = [item[0] for item in non_essential_selected]
    non_essential_selected_scores = [item[1] for item in non_essential_selected]

    # Save the selected proteins to files
    with open("selected_essential_proteins.txt", "w") as f:
        for prot_id in essential_selected_ids:
            f.write(f"{prot_id}\n")

    with open("selected_non_essential_proteins.txt", "w") as f:
        for prot_id in non_essential_selected_ids:
            f.write(f"{prot_id}\n")

    # Save the value of network centrality for selected proteins into a xlsx file
    # import pandas as pd
    # essential_df = pd.DataFrame({
    #     'Protein ID': essential_selected_ids,
    #     'Network Centrality Score': essential_selected_scores
    # })
    # essential_df.to_excel("selected_essential_proteins_scores.xlsx", index=False)
    # non_essential_df = pd.DataFrame({
    #     'Protein ID': non_essential_selected_ids,
    #     'Network Centrality Score': non_essential_selected_scores
    # })
    # non_essential_df.to_excel("selected_non_essential_proteins_scores.xlsx", index=False)

    # Plot the distribution of network centrality scores for selected proteins
    plt.figure(figsize=(6, 6))
    sns.histplot(essential_selected_scores, bins=20, alpha=0.5, color='blue',
                label='Essential Proteins', kde=True, stat='density')
    sns.histplot(non_essential_selected_scores, bins=20, alpha=0.5, color='red',
                label='Non-Essential Proteins', kde=True, stat='density')
    plt.xlabel('Network Centrality Score', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    # plt.axvline(x=25, color='r', linestyle='--', label='Threshold = 25')
    plt.legend(fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('selected_network_centrality_distribution.pdf', bbox_inches='tight')
    plt.show()

    print(f"Selected {len(essential_selected_ids)} essential proteins with scores > 25")
    print(f"Selected {len(non_essential_selected_ids)} non-essential proteins with scores < 25")
    print(f"Essential score range: {min(essential_selected_scores)} to {max(essential_selected_scores)}")
    print(f"Non-essential score range: {min(non_essential_selected_scores)} to {max(non_essential_selected_scores)}")
