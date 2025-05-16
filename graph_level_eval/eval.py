import os
import networkx as nx
import pickle
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
from multiprocessing import Pool
from scipy.linalg import eigh
from scipy.linalg import toeplitz
from sklearn import metrics
import concurrent.futures
from functools import partial
from datetime import datetime
from scipy.linalg import eigvalsh

def read_ppis(file_path):
    ppis = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            protein1 = line.split()[0]
            protein2 = line.split()[1]
            pred_neg = float(line.split()[3])
            pred_pos = float(line.split()[4])
            if pred_pos > pred_neg:
                ppis.append((protein1, protein2))

    return ppis

def reconstruct_graph(ppis):
    G = nx.Graph()
    G.add_edges_from(ppis)

    return G

def compute_graph_similarity(pred_graph: nx.Graph, gt_graph: nx.Graph) -> float:
    # Ensure both graphs have the same nodes
    if set(pred_graph.nodes) != set(gt_graph.nodes):
        raise ValueError("Graphs must have the same set of nodes")

    # Sort nodes to ensure consistent adjacency matrix ordering
    sorted_nodes = sorted(gt_graph.nodes)

    # Get adjacency matrices
    A_hat = nx.to_numpy_array(pred_graph, nodelist=sorted_nodes)
    A = nx.to_numpy_array(gt_graph, nodelist=sorted_nodes)

    diff = np.abs(A_hat - A).sum()
    graph_sim = 1 - diff / (np.sum(A) + np.sum(A_hat))

    return graph_sim

def compute_relative_density(pred_graph: nx.Graph, gt_graph: nx.Graph) -> float:
    d_hat = nx.density(pred_graph)
    d = nx.density(gt_graph)

    if d == 0:
        return float('inf') if d_hat != 0 else 1.0  # Avoid division by zero

    return d_hat / d

def gaussian_tv(x, y, sigma=1.0):  
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.abs(x - y).sum() / 2.0

    return np.exp(-dist * dist / (2 * sigma * sigma))

def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d

def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)

def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    ''' Discrepancy between 2 samples '''
    d = 0

    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        # Create the task list first
        tasks = [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1]

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            # Wrap executor.map with tqdm for progress bar
            for dist in tqdm(
                executor.map(kernel_parallel_worker, tasks),
                total=len(tasks),
                desc="Computing discrepancy"
            ):
                d += dist
    if len(samples1) * len(samples2) > 0:
        d /= len(samples1) * len(samples2)
    else:
        d = 1e+6
    return d

def compute_mmd(samples1, samples2, kernel=gaussian_tv, is_hist=True, *args, **kwargs):
    ''' MMD between two samples '''
    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / (np.sum(s1) + 1e-6) for s1 in samples1]
        samples2 = [s2 / (np.sum(s2) + 1e-6) for s2 in samples2]
    return disc(samples1, samples1, kernel, *args, **kwargs) + \
                    disc(samples2, samples2, kernel, *args, **kwargs) - \
                    2 * disc(samples1, samples2, kernel, *args, **kwargs)

def degree_distribution(pred_graph: nx.Graph, gt_graph: nx.Graph, sigma=1.0) -> float:
    deg_pred = np.array(nx.degree_histogram(pred_graph))
    deg_gt = np.array(nx.degree_histogram(gt_graph))
    # Pad 0 to make the two arrays have the same length
    max_len = max(len(deg_pred), len(deg_gt))
    deg_pred = np.pad(deg_pred, (0, max_len - len(deg_pred)))
    deg_gt = np.pad(deg_gt, (0, max_len - len(deg_gt)))

    return deg_pred, deg_gt

def clustering_worker(graph: nx.Graph):
    G = graph
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
            clustering_coeffs_list, bins=100, range=(0.0, 1.0), density=False)

    return hist

def clustering_stats(graph_ref_list, graph_pred_list,
                     bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
            G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, [G for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, [G for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10, distance_scaling=bins)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    # print('Time computing clustering mmd: ', elapsed)

    return mmd_dist

def spectral_worker(G, n_eigvals=-1):
    # eigs = nx.laplacian_spectrum(G)
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())  
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1:n_eigvals+1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf

def spectral_stats(graph_ref_list, graph_pred_list,
                   is_parallel=True, n_eigvals=-1):
    sample_ref = []
    sample_pred = []

    prev = datetime.now()
    if is_parallel:
        # Use ProcessPoolExecutor instead of ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(spectral_worker, G, n_eigvals): i 
                for i, G in enumerate(graph_ref_list)
            }

            # Add progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(graph_ref_list), 
                              desc="Processing reference graphs"):
                sample_ref.append(future.result())

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(spectral_worker, G, n_eigvals): i 
                for i, G in enumerate(graph_pred_list)
            }

            for future in tqdm(concurrent.futures.as_completed(futures), 
                              total=len(graph_pred_list),
                              desc="Processing prediction graphs"):
                sample_pred.append(future.result())

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    # print('Time computing spectral mmd: ', elapsed)

    return mmd_dist


if __name__ == "__main__":
    ### Path you need to change
    ppi_path = '../genppi_dataset/human/human_all_test_ppi_pred.txt' # path to the predicted PPIs (Take BFS as an example)
    out_path = '../genppi_dataset/human/BFS/' # path to save the evaluation results
    gt_graph_path = '../genppi_dataset/human/human_test_graph.pkl' # path to the ground truth graph
    test_graph_node_path = '../genppi_dataset/human/human_BFS_sampled_nodes.pkl' # path to the test graph nodes
    ###
    gt_graph = pickle.load(open(gt_graph_path, 'rb'))
    test_graph_nodes = pickle.load(open(test_graph_node_path, 'rb'))
    node_size_list = test_graph_nodes.keys()

    # # load ppis
    ppis = read_ppis(ppi_path)
    pred_graph = reconstruct_graph(ppis)

    # make sure the pred_graph has the same nodes as gt_graph
    if len(gt_graph.nodes()) > len(pred_graph.nodes()):
        # add missing nodes to the pred graph
        missing_nodes = set(gt_graph.nodes()) - set(pred_graph.nodes())
        pred_graph.add_nodes_from(missing_nodes)

    print("GT graph nodes: ", len(gt_graph.nodes()))
    print("GT graph edges: ", len(gt_graph.edges()))
    print("Pred graph nodes: ", len(pred_graph.nodes()))
    print("Pred graph edges: ", len(pred_graph.edges()))

    graph_level_results = {
        'graph_sim': {},
        'relative_density': {},
        'deg_dist_mmd': {},
        'cc_mmd': {},
        'laplacian_eigen_mmd': {}
    }

    # loop node size
    for node_size in tqdm(node_size_list):
        node_list = test_graph_nodes[node_size]
        gt_deg_dist = []
        pred_deg_dist = []
        gt_cc = []
        pred_cc = []
        gt_lap_eigen = []
        pred_lap_eigen = []
        gt_graphs = []
        pred_graphs = []
        for nodes in tqdm(node_list):
            # extract subgraph for both gt and pred
            gt_subgraph = gt_graph.subgraph(nodes)
            pred_subgraph = pred_graph.subgraph(nodes)

            # calculate the graph similarity
            graph_sim = compute_graph_similarity(pred_subgraph, gt_subgraph)
            if node_size not in graph_level_results['graph_sim']:
                graph_level_results['graph_sim'][node_size] = []
            graph_level_results['graph_sim'][node_size].append(graph_sim)

            # calculate the relative density
            relative_density = compute_relative_density(pred_subgraph, gt_subgraph)
            if node_size not in graph_level_results['relative_density']:
                graph_level_results['relative_density'][node_size] = []
            graph_level_results['relative_density'][node_size].append(relative_density)

            # calculate the degree distribution
            deg_pred, deg_gt = degree_distribution(pred_subgraph, gt_subgraph)
            pred_deg_dist.append(deg_pred)
            gt_deg_dist.append(deg_gt)

            # add graph to the list
            gt_graphs.append(gt_subgraph)
            pred_graphs.append(pred_subgraph)

        # calculate the degree distribution MMD
        deg_dist_mmd = compute_mmd(pred_deg_dist, gt_deg_dist)
        graph_level_results['deg_dist_mmd'][node_size] = deg_dist_mmd

        # calculate the clustering coefficient MMD
        cc_mmd = clustering_stats(gt_graphs, pred_graphs)
        graph_level_results['cc_mmd'][node_size] = cc_mmd

        # calculate the Laplacian eigenvalues MMD
        laplacian_eigen_mmd = spectral_stats(gt_graphs, pred_graphs)
        graph_level_results['laplacian_eigen_mmd'][node_size] = laplacian_eigen_mmd

    # print the results
    for node_size in node_size_list:
        print(f"Node size: {node_size}")
        print(f"Graph similarity: {np.mean(graph_level_results['graph_sim'][node_size])}")
        print(f"Relative density: {np.mean(graph_level_results['relative_density'][node_size])}")
        print(f"Degree distribution MMD: {np.mean(graph_level_results['deg_dist_mmd'][node_size])}")
        print(f"Clustering coefficient MMD: {np.mean(graph_level_results['cc_mmd'][node_size])}")
        print(f"Laplacian eigenvalues MMD: {np.mean(graph_level_results['laplacian_eigen_mmd'][node_size])}")

    # save the results
    with open(os.path.join(out_path, 'graph_eval_results.pkl'), 'wb') as f:
        pickle.dump(graph_level_results, f)

    print("The average results of all node sizes are:\n")
    results = pickle.load(open('graph_eval_results.pkl', 'rb'))

    keys = ['graph_sim', 'relative_density', 'deg_dist_mmd', 'cc_mmd', 'laplacian_eigen_mmd']

    results_list = {k: [] for k in keys}

    for k in keys:
        for sub_key in results[k]:
            results_list[k].append(results[k][sub_key])

    for k in keys:
        print(f'{k}: {np.mean(results_list[k])}')
