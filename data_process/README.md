# Data Processing Pipeline

Please first download the raw data from [PRING - Hugging Face](https://huggingface.co/datasets/piaolaidangqu/PRING) and place them in the `./data_process/raw_data` folder.

The data processing pipeline consists of two parts: (1) Protein sequence processing and (2) Graph generation.
Please follow the instructions below to preprocess the data.

## 1 Protein Sequence Processing
First go to the `./data_process/seq_process` folder.
### 1.1 Get the UniProt ID
Use `python uniprot_id.py`. This operation will extract the uniprot id from the `ppi.txt` file, which contains all the PPI pairs from four data sources: 
You will need those ids to download the protein sequences from the UniProt database.

### 1.2 Download Protein Sequences
Use `python download_fasta.py`. This operation will download the protein sequences from the UniProt database using the uniprot ids obtained in Step 1.

### 1.3 Length Filtering
Use `python seq_len_filter.py`. The threshold is between 50 and 1000.

### 1.4 Separate Species
Use `python seperate_species.py`.
This operation will seperate the protein sequences in the total fasta file into different species. The species information is extracted from the sequence header.

### 1.5 Sequence-similarity Filtering
Before running this script, please install MMseqs2 first.
Use `python seq_sim.py`. The threshold is 0.4.

### 1.6 Similar Function Protein Filtering
Use `python similar_function_remove.py`. This operation will filter out the proteins in other species that have similar functions to the proteins in human.

### 1.7 Organism Mapping
Use `python organism_mapping.py`. This process constructs the csv file to store the uniprot id, organism code, sequence, and sequence length.

### 1.8 Separate PPIs
Use `python seperate_ppis.py`.

### 1.9 Move Files
Use `python move_files.py`. This operation will move the files to the corresponding directories.

## 2 Graph Generation
Go to the `./data_process/graph_gen` folder.
### 2.1 Graph Construction
Use `python graph_cons.py` to construct the ppi network for each species.

### 2.2 Graph Split for HUMAN
Use `python graph_split.py` to split the ppi network for HUMAN.

### 2.3 Sample negative PPI samples for model training (HUMAN)
Use `python negative_sample.py` to sample negative PPI samples for model training.

### 2.4 Sample negative PPI samples for other species
Use `python otherspecies_negative_sample.py` to sample negative PPI samples for other species.

### 2.5 Simplify fasta file
Use `python fasta_simplify.py` to simplify the fasta file.

### 2.6 Generate all against all pairs for further testing
Use `python all_against_all.py` to generate all against all pairs for further testing.

### 2.7 Sample subgraphs for graph-level testing
Use `python graph_sample.py` to sample subgraphs for graph-level testing.

### 2.8 Rename the final dataset folder
```bash
mv ../species_processed_data ../pring_dataset
```
