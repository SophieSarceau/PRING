import pandas as pd
import os
from tqdm import tqdm
import pandas as pd

def read_ppi(file_path):
    ppi = []
    with open(file_path, 'r') as f:
        for line in f:
            # O00206	PPI	P08571 string
            protein1, protein2 = line.split('\t')[0], line.split('\t')[2]
            ppi.append((protein1, protein2))

    return ppi

def read_uniprot_id(file_path):
    # read the csv file
    # uniprot_id, organism_id, sequence, sequence_length
    df = pd.read_csv(file_path)

    return df

if __name__ == '__main__':
    ppi_file_path = '../raw_data/ppi.txt'
    uniprot_id_folder = '../func_removed_seq'
    output_folder = '../species_ppis'

    os.makedirs(output_folder, exist_ok=True)

    ppis = read_ppi(ppi_file_path)
    # read the files end with csv in the folder
    uniprot_id_files = [file for file in os.listdir(uniprot_id_folder) if file.endswith('.csv')]

    for file in uniprot_id_files:
        species = file.split('_')[0]
        # read the csv file
        uniprot_id_df = read_uniprot_id(os.path.join(uniprot_id_folder, file))
        uniprot_ids = set(uniprot_id_df['uniprot_id'].tolist())
        print("The number of uniprot ids in ", species, " is: ", len(uniprot_ids))
        # filter the ppis
        species_ppis = [ppi for ppi in ppis if ppi[0] in uniprot_ids and ppi[1] in uniprot_ids]
        species_ppis = list(set(species_ppis))
        print("species: ", species, "ppis: ", len(species_ppis))
        unique_proteins = set()
        for ppi in species_ppis:
            unique_proteins.add(ppi[0])
            unique_proteins.add(ppi[1])
        print("unique proteins: ", len(unique_proteins))
        # write the ppi to the file
        with open(os.path.join(output_folder, species + '_ppi.txt'), 'w') as f:
            for ppi in species_ppis:
                f.write(ppi[0] + '\t' + ppi[1] + '\n')
