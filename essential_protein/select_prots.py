import pickle
import argparse
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


def main(args):
    ids = read_uniprot_ids(args.essential_proteins_file)
    print("The number of essential proteins is: ", len(ids))
    non_ids = read_uniprot_ids(args.non_essential_proteins_file)
    print("The number of non-essential proteins is: ", len(non_ids))

    human_ppi_graph = pickle.load(open(args.graph_file, "rb"))
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

    # Filter essential proteins with score > threshold and select top N
    essential_above_threshold = [(k, v) for k, v in essential_rank_dict.items() if v > args.essential_threshold]
    print(f"Number of essential proteins with score > {args.essential_threshold}: {len(essential_above_threshold)}")

    # Take top N essential proteins with score > threshold in descending order
    essential_selected = essential_above_threshold[-args.top_n:]
    if len(essential_selected) < args.top_n:
        print(f"Warning: Only {len(essential_selected)} essential proteins with score > {args.essential_threshold} available")

    # Filter non-essential proteins with score < threshold and select bottom N
    non_essential_below_threshold = [(k, v) for k, v in non_essential_rank_dict.items() if v < args.non_essential_threshold]
    print(f"Number of non-essential proteins with score < {args.non_essential_threshold}: {len(non_essential_below_threshold)}")

    # Sort in ascending order and take the first N
    non_essential_below_threshold.sort(key=lambda x: x[1])
    non_essential_selected = non_essential_below_threshold[-args.top_n:]
    if len(non_essential_selected) < args.top_n:
        print(f"Warning: Only {len(non_essential_selected)} non-essential proteins with score < {args.non_essential_threshold} available")

    # Extract protein IDs and scores
    essential_selected_ids = [item[0] for item in essential_selected]
    essential_selected_scores = [item[1] for item in essential_selected]
    non_essential_selected_ids = [item[0] for item in non_essential_selected]
    non_essential_selected_scores = [item[1] for item in non_essential_selected]

    # Save the selected proteins to files
    with open(args.out_essential, "w") as f:
        for prot_id in essential_selected_ids:
            f.write(f"{prot_id}\n")

    with open(args.out_non_essential, "w") as f:
        for prot_id in non_essential_selected_ids:
            f.write(f"{prot_id}\n")

    # Plot the distribution of network centrality scores for selected proteins
    plt.figure(figsize=(6, 6))
    sns.histplot(essential_selected_scores, bins=20, alpha=0.5, color='blue',
                label='Essential Proteins', kde=True, stat='density')
    sns.histplot(non_essential_selected_scores, bins=20, alpha=0.5, color='red',
                label='Non-Essential Proteins', kde=True, stat='density')
    plt.xlabel('Network Centrality Score', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(args.out_plot, bbox_inches='tight')
    plt.show()

    print(f"Selected {len(essential_selected_ids)} essential proteins with scores > {args.essential_threshold}")
    print(f"Selected {len(non_essential_selected_ids)} non-essential proteins with scores < {args.non_essential_threshold}")
    if essential_selected_scores:
        print(f"Essential score range: {min(essential_selected_scores)} to {max(essential_selected_scores)}")
    if non_essential_selected_scores:
        print(f"Non-essential score range: {min(non_essential_selected_scores)} to {max(non_essential_selected_scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select essential and non-essential proteins based on network centrality.")
    parser.add_argument('--essential_proteins_file', type=str, default='essential_proteins.txt', help='Path to the file with essential protein IDs.')
    parser.add_argument('--non_essential_proteins_file', type=str, default='non_essential_proteins.txt', help='Path to the file with non-essential protein IDs.')
    parser.add_argument('--graph_file', type=str, help='Path to the human PPI graph file.')
    parser.add_argument('--out_essential', type=str, default='selected_essential_proteins2.txt', help='Output file for selected essential proteins.')
    parser.add_argument('--out_non_essential', type=str, default='selected_non_essential_proteins2.txt', help='Output file for selected non-essential proteins.')
    parser.add_argument('--out_plot', type=str, default='selected_network_centrality_distribution.png', help='Output file for the distribution plot.')
    parser.add_argument('--essential_threshold', type=float, default=30, help='Network centrality score threshold for essential proteins.')
    parser.add_argument('--non_essential_threshold', type=float, default=20, help='Network centrality score threshold for non-essential proteins.')
    parser.add_argument('--top_n', type=int, default=100, help='Number of proteins to select for each category.')

    args = parser.parse_args()
    main(args)
