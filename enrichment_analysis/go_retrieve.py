import pickle
import re
from io import StringIO
import concurrent.futures
import time
from functools import partial

import networkx as nx
import pandas as pd
from bioservices import UniProt
from tqdm import tqdm

def retrieve_go_terms(uniprot_ids):
    """
    Retrieve GO terms for a list of UniProt IDs.

    Args:
        uniprot_ids (str or list): A single UniProt ID or a list of UniProt IDs

    Returns:
        dict: Dictionary with UniProt entry names as keys, and values are dictionaries 
              containing GO terms categorized by biological process (go_p), 
              molecular function (go_f), and cellular component (go_c)
    """
    # Initialize the UniProt service
    u = UniProt()

    # Convert single ID to list if needed
    if isinstance(uniprot_ids, str):
        uniprot_ids = [uniprot_ids]

    # Join the UniProt IDs for the query
    query = ' '.join(uniprot_ids)

    # Define the columns to retrieve: GO ID, GO terms for BP, MF, CC
    columns = "id,go_p,go_f,go_c"

    # Perform the search (format now must be 'tsv')
    result = u.search(query, columns=columns, frmt="tsv")

    # Convert the result to a pandas DataFrame
    df = pd.read_csv(StringIO(result), sep='\t')

    # Extract GO term IDs using regex
    go_dict = {}

    for i in range(len(df)):
        entry = uniprot_ids[i]
        go_p = re.findall(r'\[GO:\d+\]', str(df.iloc[i]['Gene Ontology (biological process)']))
        go_f = re.findall(r'\[GO:\d+\]', str(df.iloc[i]['Gene Ontology (molecular function)']))
        go_c = re.findall(r'\[GO:\d+\]', str(df.iloc[i]['Gene Ontology (cellular component)']))

        # Remove the brackets from GO IDs
        go_dict[entry] = {
            'go_p': [x[1:-1] for x in go_p],
            'go_f': [x[1:-1] for x in go_f],
            'go_c': [x[1:-1] for x in go_c]
        }

    return go_dict

def process_entry(entry):
    entry_dict = retrieve_go_terms(entry)
    return entry, entry_dict[entry]


if __name__ == "__main__":
    test_graph_path = "/home/bingxing2/ailab/group/ai4agr/zxz/PPI/benchmark/GenPPI-local/genppi_dataset/human/BFS/human_test_graph.pkl"
    test_graph = pickle.load(open(test_graph_path, "rb"))

    node_list = list(test_graph.nodes())
    go_dict = {}

    # Use ThreadPoolExecutor for concurrent API calls
    # Adjust max_workers as needed based on your system capabilities
    max_workers = 20  # You can adjust this value

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_entry = {executor.submit(process_entry, entry): entry for entry in node_list}

        # Process results as they complete using tqdm to track progress
        for future in tqdm(concurrent.futures.as_completed(future_to_entry), total=len(node_list), desc="Retrieving GO terms"):
            try:
                entry, go_terms = future.result()
                go_dict[entry] = go_terms
            except Exception as exc:
                print(f"Entry {future_to_entry[future]} generated an exception: {exc}")

    # Save the GO terms to a pickle file
    output_path = "./test_go_terms.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(go_dict, f)
