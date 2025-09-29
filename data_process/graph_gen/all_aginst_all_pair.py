import os
import pickle
import networkx as nx
from tqdm import tqdm

species_list = ['yeast', 'arath', 'ecoli', 'human']
base_path = '../species_processed_data/'

def read_ppis(file_path):
    ppis = set()
    graph = pickle.load(open(file_path, 'rb'))
    for edge in graph.edges():
        ppis.add((edge[0], edge[1]))

    return ppis

if __name__ == "__main__":
    for species in tqdm(species_list):
        if species == 'human':
            method_list = ['BFS', 'DFS', 'RANDOM_WALK']
            for method in method_list:
                file_path = os.path.join(base_path, species, method, 'human_test_graph.pkl')
                ppis = read_ppis(file_path)
                unique_proteins = set()
                for ppi in ppis:
                    unique_proteins.add(ppi[0])
                    unique_proteins.add(ppi[1])
                print(f'{method} {species} unique proteins: {len(unique_proteins)}')
                unique_proteins = list(unique_proteins)
                # generate all against all pairs, (a, b) and (b, a) are the same, so only record one
                all_against_all_pairs = []
                for i in range(len(unique_proteins)):
                    for j in range(i, len(unique_proteins)):
                        all_against_all_pairs.append((unique_proteins[i], unique_proteins[j]))
                all_against_all_pairs = set(all_against_all_pairs)
                print(f'{method} {species} all against all pairs: {len(all_against_all_pairs)}')
                positive_ppis = ppis
                positive_ppis_reverse = set([(ppi[1], ppi[0]) for ppi in positive_ppis])
                negative_ppis = set(all_against_all_pairs) - positive_ppis - positive_ppis_reverse
                print("positive ppis: ", len(positive_ppis))
                print("negative ppis: ", len(negative_ppis))

                # save all against all pairs to txt file
                output_file = os.path.join(base_path, species, method, 'all_test_ppi.txt')
                with open(output_file, "w") as f:
                    for pair in positive_ppis:
                        f.write(f'{pair[0]}\t{pair[1]}\t1\n')
                    for pair in negative_ppis:
                        f.write(f'{pair[0]}\t{pair[1]}\t0\n')
        else:
            file_path = os.path.join(base_path, species, species + '_test_graph.pkl')
            ppis = read_ppis(file_path)
            unique_proteins = set()
            for ppi in ppis:
                unique_proteins.add(ppi[0])
                unique_proteins.add(ppi[1])
            print(f'{species} unique proteins: {len(unique_proteins)}')
            unique_proteins = list(unique_proteins)
            # generate all against all pairs
            all_against_all_pairs = []
            for i in range(len(unique_proteins)):
                for j in range(i, len(unique_proteins)):
                    all_against_all_pairs.append((unique_proteins[i], unique_proteins[j]))
            print(f'{species} all against all pairs: {len(all_against_all_pairs)}')
            positive_ppis = ppis
            positive_ppis_reverse = set([(ppi[1], ppi[0]) for ppi in positive_ppis])
            negative_ppis = set(all_against_all_pairs) - positive_ppis - positive_ppis_reverse
            print("positive ppis: ", len(positive_ppis))
            print("negative ppis: ", len(negative_ppis))

            # save all against all pairs to txt file
            output_file = os.path.join(base_path, species, species+'_all_test_ppi.txt')
            with open(output_file, "w") as f:
                for pair in positive_ppis:
                    f.write(f'{pair[0]}\t{pair[1]}\t1\n')
                for pair in negative_ppis:
                    f.write(f'{pair[0]}\t{pair[1]}\t0\n')
