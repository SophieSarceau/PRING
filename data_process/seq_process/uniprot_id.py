import os
from tqdm import tqdm

def read_ppi(file_path):
    ppi = []
    with open(file_path, 'r') as f:
        for line in f:
            # O00206	PPI	P08571
            ppi.append(line.strip().split('\t'))

    return ppi


if __name__ == '__main__':
    file_path = '../raw_data/ppi.txt'
    ppi = read_ppi(file_path)
    print("The length of PPI is: ", len(ppi))

    # Get all uniprot ids
    uniprot_ids = []
    for i in tqdm(range(len(ppi))):
        uniprot_ids.append(ppi[i][0])
        uniprot_ids.append(ppi[i][2])

    # Get the set of uniprot ids and save to a txt file
    uniprot_ids = list(set(uniprot_ids))
    print("The length of uniprot ids is: ", len(uniprot_ids))
    with open('../raw_data/uniprot_ids.txt', 'w') as f:
        for uniprot_id in uniprot_ids:
            f.write(uniprot_id + '\n')
