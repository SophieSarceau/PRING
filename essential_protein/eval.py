import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def read_uniprot_ids(txt_path):
    ids = []
    for line in open(txt_path):
        line = line.strip()
        ids.append(line)

    return ids

def read_ppis(ppi_file: str) -> list:
    ppis = []
    with open(ppi_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # print(line)
                prot1, prot2, pred = line.split()
                if int(pred) == 1:
                    ppis.append((prot1, prot2))

    return ppis

def compute_network_centrality(G: nx.Graph, selected_prots: None) -> dict:
    # Pre-compute all neighbor sets
    if selected_prots is not None:
        neighbor_sets = {node: set(G.neighbors(node)) for node in G.nodes()}
        nc_dict = {}

        for u in tqdm(selected_prots):
            neighbors_u = neighbor_sets[u]
            nc = 0
            for v in neighbors_u:
                neighbors_v = neighbor_sets[v]
                common = len(neighbors_u & neighbors_v)
                denom = min(len(neighbors_u) - 1, len(neighbors_v) - 1)
                if denom > 0:
                    ecc = common / denom
                    nc += ecc
            nc_dict[u] = nc
    else:
        neighbor_sets = {node: set(G.neighbors(node)) for node in G.nodes()}
        nc_dict = {}

        for u in tqdm(G.nodes()):
            neighbors_u = neighbor_sets[u]
            nc = 0
            for v in neighbors_u:
                neighbors_v = neighbor_sets[v]
                common = len(neighbors_u & neighbors_v)  # Use the pre-computed sets
                denom = min(len(neighbors_u) - 1, len(neighbors_v) - 1)
                if denom > 0:
                    ecc = common / denom
                    nc += ecc
            nc_dict[u] = nc

    return nc_dict

def precision_at_k(gt_dict, pred_dict, K):
    """
    P@K = | top‐K(pred) ∩ top‐K(gt) | / K
    """
    # get the sets of top‐K keys
    gt_topk   = set(sorted(gt_dict,   key=gt_dict.get,   reverse=True)[:K])
    pred_topk = sorted(pred_dict, key=pred_dict.get, reverse=True)[:K]

    hits = sum(1 for p in pred_topk if p in gt_topk)
    return hits / K

def compute_distribution_overlap(essential_dict, nonessential_dict):
    """
    Computes the area of overlap between two centrality score distributions.

    Args:
        essential_dict (dict): {protein_id: centrality_score, ...} for essential proteins
        nonessential_dict (dict): {protein_id: centrality_score, ...} for non-essential proteins

    Returns:
        float: Overlap area between two distributions (0 to 1)
    """
    # scaler = MinMaxScaler()
    # ess_scores = scaler.fit_transform(np.array(list(essential_dict.values())).reshape(-1, 1)).flatten()
    # non_ess_scores = scaler.fit_transform(np.array(list(nonessential_dict.values())).reshape(-1, 1)).flatten()
    ess_scores = np.array(list(essential_dict.values()))
    non_ess_scores = np.array(list(nonessential_dict.values()))

    # Improved KDE with bandwidth tuning
    kde_ess = gaussian_kde(ess_scores)
    kde_non = gaussian_kde(non_ess_scores)

    # Define a range over which to evaluate the PDFs
    x = np.linspace(max(ess_scores.min(), non_ess_scores.min()), min(ess_scores.max(), non_ess_scores.max()), 10000)
    ess_pdf = kde_ess(x)
    non_ess_pdf = kde_non(x)
    min_pdf = np.minimum(ess_pdf, non_ess_pdf)
    overlap_area = np.trapz(min_pdf, x)
    overlap_self = np.trapz(ess_pdf, x)
    overlap_self2 = np.trapz(non_ess_pdf, x)

    return overlap_area, overlap_self, overlap_self2


if __name__ == "__main__":
    # Path to the essential and non-essential protein files
    gt_human_test_graph_path = './human_test_graph.pkl'
    pred_ppi_path = './human_all_test_ppi_pred.txt'
    #

    ids = read_uniprot_ids("selected_essential_proteins.txt")
    print("The number of essential proteins is: ", len(ids))
    non_ids = read_uniprot_ids("selected_non_essential_proteins.txt")
    print("The number of non-essential proteins is: ", len(non_ids))
    all_ids = ids + non_ids

    human_ppi_graph = pickle.load(open(gt_human_test_graph_path, "rb"))
    human_ppi_nodes = set(human_ppi_graph.nodes())
    print("The length of human ppi nodes is: ", len(human_ppi_nodes))

    pred_ppis = read_ppis(pred_ppi_path)
    pred_human_graph = nx.Graph()
    pred_human_graph.add_nodes_from(human_ppi_graph.nodes())
    pred_human_graph.add_edges_from(pred_ppis)

    # Calculate the network centrality for both gt graph and pred graph
    gt_human_test_graph_dict = compute_network_centrality(human_ppi_graph, selected_prots=all_ids)
    pred_human_test_graph_dict = compute_network_centrality(pred_human_graph, selected_prots=all_ids)

    p_100 = precision_at_k(gt_human_test_graph_dict, pred_human_test_graph_dict, K=100)
    pred_essential_dict = {k: pred_human_test_graph_dict[k] for k in ids}
    pred_non_essential_dict = {k: pred_human_test_graph_dict[k] for k in non_ids}
    overlap, overlap_self, overlap_self2 = compute_distribution_overlap(pred_essential_dict, pred_non_essential_dict)

    print("Precision: ", p_100)
    print("Overlap: ", overlap)

    # plot the distribution of network centrality scores for both essential and non_essential predictions
    plt.figure(figsize=(6, 6))
    pred_essential_proteins_dict = {k: pred_human_test_graph_dict[k] for k in ids}
    pred_non_essential_proteins_dict = {k: pred_human_test_graph_dict[k] for k in non_ids}
    sns.histplot(list(pred_essential_proteins_dict.values()), bins=20, alpha=0.5, color='blue',
                label='Essential Proteins', kde=True, stat='density')
    sns.histplot(list(pred_non_essential_proteins_dict.values()), bins=20, alpha=0.5, color='red',
                label='Non-Essential Proteins', kde=True, stat='density')
    plt.xlabel('Network Centrality Score', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig('network_centrality_distribution_pred_ppitrans.pdf', bbox_inches='tight')
