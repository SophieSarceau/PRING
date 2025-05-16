import pickle
import math
import networkx as nx
import numpy as np
from tqdm import tqdm
import community as community_louvain
from collections import defaultdict
from gprofiler import GProfiler
from collections import OrderedDict,deque,Counter
from scipy.stats import spearmanr
import random

def read_ppis(ppi_file):
    ppis = []
    with open(ppi_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # print(line)
                prot1, prot2, label, neg_prob, pos_prob = line.split()
                if pos_prob > neg_prob:
                    ppis.append((prot1, prot2))

    return ppis

def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

def clusters_to_protein_sets(cluster_dict):
    cluster_sets = defaultdict(list)
    for protein, cluster_id in cluster_dict.items():
        cluster_sets[cluster_id].append(protein)
    return list(cluster_sets.values())

def get_go_profile(cluster, sources=['GO:BP']):
    gp = GProfiler(return_dataframe=True)

    results = gp.profile(organism='hsapiens',  # human, adjust accordingly
                        query=cluster,
                        sources=sources)
    results['neg_log10_fdr'] = -np.log10(results['p_value'])
    results['f1'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'])

    return results

def match_go_terms(results, leaf_terms):
    """
    Params:
        results: DataFrame from gprofiler (source,native,name,p_value,significant,description,
                                           term_size,query_size,intersection_size,effective_domain_size,
                                           precision,recall,query,parents)
        leaf_terms: list of leaf terms
    """
    # keep the results that only the native in leaf_terms
    results = results[results['native'].isin(leaf_terms)]

    return results

def calculate_metrics(pred_cluster, true_cluster):
    pred_set = set(pred_cluster)
    true_set = set(true_cluster)

    intersection = pred_set & true_set
    intersection_size = len(intersection)

    precision = intersection_size / len(pred_set) if pred_set else 0
    recall = intersection_size / len(true_set) if true_set else 0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    jaccard = intersection_size / len(pred_set | true_set) if pred_set | true_set else 0

    return precision, recall, f1_score, jaccard

def jaccard_func(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0

def within_cluster_similarity(cluster, go_terms, source):
    # go_terms: dict protein_id → set of GO‑Slim terms
    if source == "GO:BP":
        source = 'go_p'
    elif source == "GO:MF":
        source = 'go_f'
    elif source == "GO:CC":
        source = 'go_c'
    # print(cluster)
    prots = cluster
    sims = []
    for i in tqdm(range(len(prots))):
        for j in range(i+1, len(prots)):
            go_terms_i = set(go_terms[prots[i]][source])
            go_terms_j = set(go_terms[prots[j]][source])
            if len(go_terms_i) == 0 or len(go_terms_j) == 0:
                continue
            similarity = jaccard_func(go_terms_i, go_terms_j)
            if similarity > 0:
                sims.append(similarity)
    similarity = sum(sims) / len(sims) if sims else 0.0

    return similarity, sims


if __name__ == "__main__":
    ### Your Path Here
    pred_file = "/home/bingxing2/ailab/group/ai4agr/zxz/PPI/benchmark/GenPPI-local/genppi_dataset/human/BFS_pred/human_all_test_ppi_pred.txt"
    gt_file = "/home/bingxing2/ailab/group/ai4agr/zxz/PPI/benchmark/GenPPI-local/genppi_dataset/human/BFS/human_test_graph.pkl"
    ### End of Your Path
    uniprot_to_goterms = "./test_go_terms.pkl"

    source = 'GO:BP' # GO:BP, GO:MF, GO:CC

    random.seed(42)
    np.random.seed(42)

    # Build network from edge list
    pred_edges = read_ppis(pred_file)
    G_pred = nx.Graph()
    G_true_ori = pickle.load(open(gt_file, 'rb'))
    G_nodes = list(G_true_ori.nodes())
    G_true_edges = list(G_true_ori.edges())
    G_true = nx.Graph()
    # Sort the nodes for G_nodes
    G_nodes = sorted(G_nodes)
    # Sort the edges by the first node then the second node for pred_edges
    pred_edges = sorted(pred_edges, key=lambda x: (x[0], x[1]))
    # Sort the edges by the first node for G_true_edges
    G_true_edges = sorted(G_true_edges, key=lambda x: (x[0], x[1]))
    # Add nodes to the graph
    G_pred.add_nodes_from(G_nodes)
    G_true.add_nodes_from(G_nodes)
    # Add edges to the graph
    G_pred.add_edges_from(pred_edges)
    G_true.add_edges_from(G_true_edges)
    uniprot2go_dict = pickle.load(open(uniprot_to_goterms, 'rb'))

    # Perform community detection
    clusters_pred = community_louvain.best_partition(G_pred, resolution=1.0, random_state=42)
    clusters_true = community_louvain.best_partition(G_true, resolution=1.0, random_state=42)

    pred_clusters = clusters_to_protein_sets(clusters_pred)
    true_clusters = clusters_to_protein_sets(clusters_true)

    print("The number of predicted clusters is: ", len(pred_clusters))
    print("The number of true clusters is: ", len(true_clusters))

    # pred_clusters and true_clusters are lists of sets of proteins
    matches = []
    for idx_pred, pred_cluster in tqdm(enumerate(pred_clusters)):
        best_match = None
        best_score = 0
        for idx_true, true_cluster in enumerate(true_clusters):
            score = jaccard_similarity(set(pred_cluster), set(true_cluster))
            if score > best_score:
                best_score = score
                best_match = idx_true
        if best_score >= 0.2:
            matches.append((idx_pred, best_match, best_score))

    jaccard_scores = []
    spearman_list = []
    pred_neg_log_10_fdr = []
    true_neg_log_10_fdr = []
    pred_within_cluster_sim = []
    true_within_cluster_sim = []
    pred_within_cluster_sim_list = []
    true_within_cluster_sim_list = []
    for match in tqdm(matches):
        pred_idx, true_idx, score = match
        pred_cluster = pred_clusters[pred_idx]
        true_cluster = true_clusters[true_idx]

        # get the GO profile for the predicted cluster
        pred_go = get_go_profile(pred_cluster, sources=[source])
        true_go = get_go_profile(true_cluster, sources=[source])

        pred_go_items = pred_go['native'].tolist()
        true_go_items = true_go['native'].tolist()
        # Get mutual GO terms
        mutual_go_terms = set(pred_go_items) & set(true_go_items)
        # print("The number of mutual GO terms is: ", len(mutual_go_terms))
        for go_term in mutual_go_terms:
            pred_row = pred_go[pred_go['native'] == go_term]
            true_row = true_go[true_go['native'] == go_term]
            if not np.isnan(pred_row['neg_log10_fdr'].values[0]) and pred_row['neg_log10_fdr'].values[0] != 0:
                if not np.isnan(true_row['neg_log10_fdr'].values[0]) and true_row['neg_log10_fdr'].values[0] != 0:
                    pred_neg_log_10_fdr.append(pred_row['neg_log10_fdr'].values[0])
                    true_neg_log_10_fdr.append(true_row['neg_log10_fdr'].values[0])
        # Calculate the precision, recall, f1 score, and Jaccard similarity between predicted and true GO terms
        precision, recall, f1_score, jaccard = calculate_metrics(pred_go_items, true_go_items)
        jaccard_scores.append(jaccard)

        # Calculate the within-cluster similarity for predicted and true clusters
        pred_within_sim, pred_within_sim_list = within_cluster_similarity(pred_cluster, uniprot2go_dict, source)
        true_within_sim, true_within_sim_list = within_cluster_similarity(true_cluster, uniprot2go_dict, source)
        pred_within_cluster_sim.append(pred_within_sim)
        true_within_cluster_sim.append(true_within_sim)
        pred_within_cluster_sim_list.append(pred_within_sim_list)
        true_within_cluster_sim_list.append(true_within_sim_list)

    spearman_list.append(spearmanr(pred_neg_log_10_fdr, true_neg_log_10_fdr)[0])

    # calculate the average Jaccard similarity
    avg_jaccard = np.mean(jaccard_scores)
    print("Average Jaccard similarity: ", avg_jaccard)
    # calculate the spearman correlation between the predicted and true GO terms
    print("Spearman correlation between predicted and true GO terms: ", np.mean(spearman_list))
    # calculate the average within-cluster similarity
    # remove 0 from pred_within_cluster_sim and true_within_cluster_sim
    pred_within_cluster_sim = [x for x in pred_within_cluster_sim if x != 0]
    true_within_cluster_sim = [x for x in true_within_cluster_sim if x != 0]
    avg_pred_within_sim = np.mean(pred_within_cluster_sim)
    avg_true_within_sim = np.mean(true_within_cluster_sim)
    print("Average within-cluster similarity for predicted clusters: ", avg_pred_within_sim)
    print("Average within-cluster similarity for true clusters: ", avg_true_within_sim)

    # save the within cluster similarity list
    pred_within_cluster_sim_list = np.array([item for sublist in pred_within_cluster_sim_list for item in sublist])
    true_within_cluster_sim_list = np.array([item for sublist in true_within_cluster_sim_list for item in sublist])
    np.save(f"{source}_ppitrans_pred_within_cluster_sim_list.npy", pred_within_cluster_sim_list)
    np.save(f"{source}_ppitrans_true_within_cluster_sim_list.npy", true_within_cluster_sim_list)
