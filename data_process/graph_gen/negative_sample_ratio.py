import os
import networkx as nx
import pickle
import random
from tqdm import tqdm

random.seed(0)

neg_ratio = 10

method_list = ["BFS", "DFS", "RANDOM_WALK"]
base_path = "../string_ppi_network/human/"

graph_path = "../string_ppi_network/human/human_graph.pkl"
ppi_path = "../string_ppi_network/human/human_ppi.txt"

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

if __name__ == '__main__':
    total_ppi_graph = pickle.load(open(graph_path, "rb"))
    total_ppis = read_ppis(ppi_path)

    # sample train and test graphs for each method
    for method in tqdm(method_list):
        split_path = os.path.join(base_path, method, 'human_'+method+'_split.pkl')
        split_file = pickle.load(open(split_path, "rb"))
        train_split = split_file['train']
        test_split = split_file['test']
        train_graph = total_ppi_graph.subgraph(train_split)
        test_graph = total_ppi_graph.subgraph(test_split)

        # train ppi pairs & test ppi pairs
        train_ppi_pos_pairs = list(train_graph.edges())
        test_ppi_pos_pairs = list(test_graph.edges())
        print("The number of train ppi pairs: ", len(train_ppi_pos_pairs))
        print("The number of test ppi pairs: ", len(test_ppi_pos_pairs))

        # generate negative samples
        train_ppi_neg_pairs = sample_negatives(train_ppi_pos_pairs, None, total_ppis, factor=neg_ratio)
        test_ppi_neg_pairs = sample_negatives(test_ppi_pos_pairs, None, total_ppis, factor=neg_ratio)
        print("The number of train negative ppi pairs: ", len(train_ppi_neg_pairs))
        print("The number of test negative ppi pairs: ", len(test_ppi_neg_pairs))

        # save the train_graph and test_graph
        # train_graph_path = os.path.join(base_path, method, 'human_train_graph.pkl')
        # test_graph_path = os.path.join(base_path, method, 'human_test_graph.pkl')
        # pickle.dump(train_graph, open(train_graph_path, "wb"))
        # pickle.dump(test_graph, open(test_graph_path, "wb"))

        # save the train and test ppi pairs, combine the pos and neg pairs
        # with 1 indicating positive and 0 indicating negative
        # write to txt file
        train_ppi_pairs = [(ppi, 1) for ppi in train_ppi_pos_pairs]
        train_ppi_pairs.extend([(ppi, 0) for ppi in train_ppi_neg_pairs])
        test_ppi_pairs = [(ppi, 1) for ppi in test_ppi_pos_pairs]
        test_ppi_pairs.extend([(ppi, 0) for ppi in test_ppi_neg_pairs])
        print("The number of train ppi pairs: ", len(train_ppi_pairs))
        print("The number of test ppi pairs: ", len(test_ppi_pairs))

        # random split the train_ppi_pairs into train and validation: 8: 2
        random.shuffle(train_ppi_pairs)
        train_size = int(0.8 * len(train_ppi_pairs))
        train_ppi_pairs, val_ppi_pairs = train_ppi_pairs[:train_size], train_ppi_pairs[train_size:]

        # write to txt file
        train_path = os.path.join(base_path, method, 'human_train_ppi_'+str(neg_ratio)+'.txt')
        val_path = os.path.join(base_path, method, 'human_val_ppi_'+str(neg_ratio)+'.txt')
        test_path = os.path.join(base_path, method, 'human_test_ppi_'+str(neg_ratio)+'.txt')

        with open(train_path, "w") as f:
            for ppi, label in train_ppi_pairs:
                f.write(ppi[0] + '\t' + ppi[1] + '\t' + str(label) + '\n')
        with open(val_path, "w") as f:
            for ppi, label in val_ppi_pairs:
                f.write(ppi[0] + '\t' + ppi[1] + '\t' + str(label) + '\n')
        with open(test_path, "w") as f:
            for ppi, label in test_ppi_pairs:
                f.write(ppi[0] + '\t' + ppi[1] + '\t' + str(label) + '\n')

# cp -f GenPPI-local/string_ppi_network/human/BFS/human_train_ppi_5.txt ./GenPPI-local/genppi_dataset/human/BFS/
# cp -f GenPPI-local/string_ppi_network/human/BFS/human_val_ppi_5.txt ./GenPPI-local/genppi_dataset/human/BFS/
# cp -f GenPPI-local/string_ppi_network/human/BFS/human_test_ppi_5.txt ./GenPPI-local/genppi_dataset/human/BFS/

# cp -f GenPPI-local/string_ppi_network/human/DFS/human_train_ppi_5.txt ./GenPPI-local/genppi_dataset/human/DFS/
# cp -f GenPPI-local/string_ppi_network/human/DFS/human_val_ppi_5.txt ./GenPPI-local/genppi_dataset/human/DFS/
# cp -f GenPPI-local/string_ppi_network/human/DFS/human_test_ppi_5.txt ./GenPPI-local/genppi_dataset/human/DFS/

# cp -f GenPPI-local/string_ppi_network/human/RANDOM_WALK/human_train_ppi_5.txt ./GenPPI-local/genppi_dataset/human/RANDOM_WALK/
# cp -f GenPPI-local/string_ppi_network/human/RANDOM_WALK/human_val_ppi_5.txt ./GenPPI-local/genppi_dataset/human/RANDOM_WALK/
# cp -f GenPPI-local/string_ppi_network/human/RANDOM_WALK/human_test_ppi_5.txt ./GenPPI-local/genppi_dataset/human/RANDOM_WALK/


# cp -f GenPPI-local/string_ppi_network/human/BFS/human_train_ppi_10.txt ./GenPPI-local/genppi_dataset/human/BFS/
# cp -f GenPPI-local/string_ppi_network/human/BFS/human_val_ppi_10.txt ./GenPPI-local/genppi_dataset/human/BFS/
# cp -f GenPPI-local/string_ppi_network/human/BFS/human_test_ppi_10.txt ./GenPPI-local/genppi_dataset/human/BFS/

# cp -f GenPPI-local/string_ppi_network/human/DFS/human_train_ppi_10.txt ./GenPPI-local/genppi_dataset/human/DFS/
# cp -f GenPPI-local/string_ppi_network/human/DFS/human_val_ppi_10.txt ./GenPPI-local/genppi_dataset/human/DFS/
# cp -f GenPPI-local/string_ppi_network/human/DFS/human_test_ppi_10.txt ./GenPPI-local/genppi_dataset/human/DFS/

# cp -f GenPPI-local/string_ppi_network/human/RANDOM_WALK/human_train_ppi_10.txt ./GenPPI-local/genppi_dataset/human/RANDOM_WALK/
# cp -f GenPPI-local/string_ppi_network/human/RANDOM_WALK/human_val_ppi_10.txt ./GenPPI-local/genppi_dataset/human/RANDOM_WALK/
# cp -f GenPPI-local/string_ppi_network/human/RANDOM_WALK/human_test_ppi_10.txt ./GenPPI-local/genppi_dataset/human/RANDOM_WALK/
