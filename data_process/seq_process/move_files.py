import os

if __name__ == '__main__':
    species_list = ['yeast', 'arath', 'ecoli', 'human']
    output_folder = '../species_processed_data'

    for species in species_list:
        # create the folder for the species
        species_folder = os.path.join(output_folder, species)
        os.makedirs(species_folder, exist_ok=True)

        # copy clustered fasta and csv file to the species folder
        os.system(f'cp ../func_removed_seq/{species}.fasta {species_folder}')
        os.system(f'cp ../func_removed_seq/{species}_protein_id.csv {species_folder}')
        # copy ppi file to the species folder
        os.system(f'cp ../species_ppis/{species}_ppi.txt {species_folder}')
