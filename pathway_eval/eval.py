import pickle
import networkx as nx
import numpy as np
from tqdm import tqdm

def read_ppis(file_path):
    ppis = []
    for line in open(file_path):
        line = line.strip()
        if line:
            prot1, prot2, _, neg_prob, pos_prob = line.split()
            if pos_prob > neg_prob:
                ppis.append((prot1, prot2))

    return ppis

def pathway_recall_cal(pred_graph, gt_graph):
    pred_ppis = set(pred_graph.edges())
    gt_ppis = set(gt_graph.edges())
    recall_num = 0
    for ppi in pred_ppis:
        # check whether (A, B) or (B, A) in gt_ppis
        if ppi in gt_ppis or (ppi[1], ppi[0]) in gt_ppis:
            recall_num += 1

    return recall_num / len(gt_ppis) if len(gt_ppis) > 0 else 0

def pathway_prec_cal(pred_graph, gt_graph):
    pred_ppis = set(pred_graph.edges())
    gt_ppis = set(gt_graph.edges())
    prec_num = 0
    for ppi in pred_ppis:
        # check whether (A, B) or (B, A) in gt_ppis
        if ppi in gt_ppis or (ppi[1], ppi[0]) in gt_ppis:
            prec_num += 1

    return prec_num / len(pred_ppis) if len(pred_ppis) > 0 else 0

if __name__ == "__main__":
    # Read the PPI data
    pred_path = './complex_test_pred.txt'
    complex_graph_path = './complex_graphs.pkl'
    ppis = read_ppis(pred_path)
    # Load the complex graphs
    complex_graphs = pickle.load(open(complex_graph_path, 'rb'))

    # construct the pred whole graph
    pred_graph = nx.Graph()
    for prot1, prot2 in ppis:
        pred_graph.add_edge(prot1, prot2)

    unique_complex = []

    pathway_recall = []
    pathway_precision = []
    pathway_connectivity = []

    for graph in tqdm(complex_graphs):
        nodes = graph.nodes()
        if set(list(nodes)) in unique_complex:
            continue
        unique_complex.append(set(list(nodes)))
        pred_sub_graph = nx.Graph(pred_graph.subgraph(nodes))
        # add missing nodes if any
        for node in nodes:
            if node not in pred_sub_graph.nodes():
                pred_sub_graph.add_node(node)

        num_components = nx.number_connected_components(pred_sub_graph)
        if num_components == 1:
            pathway_connectivity.append(1)
        else:
            pathway_connectivity.append(0)
        # calculate the recall and precision
        recall = pathway_recall_cal(pred_sub_graph, graph)
        precision = pathway_prec_cal(pred_sub_graph, graph)
        pathway_recall.append(recall)
        pathway_precision.append(precision)

    # calculate the average recall and precision
    avg_recall = np.mean(pathway_recall)
    std_recall = np.std(pathway_recall)
    avg_precision = np.mean(pathway_precision)
    std_precision = np.std(pathway_precision)
    avg_connectivity = np.mean(pathway_connectivity)
    std_connectivity = np.std(pathway_connectivity)
    print(f"Average Recall: {avg_recall}, Std Recall: {std_recall}")
    print(f"Average Precision: {avg_precision}, Std Precision: {std_precision}")
    print(f"Average Connectivity: {avg_connectivity}, Std Connectivity: {std_connectivity}")
