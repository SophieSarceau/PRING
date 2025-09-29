import networkx as nx
import os
import pickle

if __name__ == '__main__':
    species_folder = '../species_processed_data'
    # read folders in the directory
    folders = os.listdir(species_folder)

    for folder in folders:
        # read the ppi pairs
        species = folder
        ppi_file = os.path.join(species_folder, folder, species+'_ppi.txt')
        ppi_pairs = []
        proteins = []
        for line in open(ppi_file):
            line = line.strip()
            protein1, protein2 = line.split('\t')
            ppi_pairs.append((protein1, protein2))
            proteins.append(protein1)
            proteins.append(protein2)
        print("The number of PPI pairs in", species, "is", len(ppi_pairs))
        print("The number of unique PPI pairs in", species, "is", len(set(ppi_pairs)))
        print("The number of proteins in", species, "is", len(set(proteins)))
        # write the unique ppi pairs to the original file
        ppi_file = os.path.join(species_folder, folder, species+'_ppi_unique.txt')
        with open(ppi_file, 'w') as f:
            for pair in set(ppi_pairs):
                f.write(pair[0]+'\t'+pair[1]+ '\n')
        print('Unique PPI pairs saved in', ppi_file)

        # construct the graph
        G = nx.Graph()
        G.add_edges_from(ppi_pairs)
        print('Number of nodes:', G.number_of_nodes())
        print('Number of edges:', G.number_of_edges())

        # save the graph in a pkl
        graph_file = os.path.join(species_folder, folder, species+'_graph.pkl')
        with open(graph_file, 'wb') as f:
            pickle.dump(G, f)
        print('Graph saved in', graph_file)
