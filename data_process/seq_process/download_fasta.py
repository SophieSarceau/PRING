import requests
import re
import time
from tqdm import tqdm
import os

# URL for UniProt batch download
UNIPROT_BATCH_URL = "https://rest.uniprot.org/uniprotkb/accessions"

def get_uniprot_ids(file_path):
    """
    Extract UniProt IDs from a file.
    Supports two formats:
    1. One ID per line.
    2. FASTA header format, e.g., >sp|P50399|GDIB_RAT...
    """
    ids = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Check if it is in FASTA header format
                if line.startswith('>'):
                    match = re.search(r'\|(.*?)\|', line)
                    if match:
                        ids.add(match.group(1))
                # Otherwise, assume it's a plain ID
                else:
                    # Simple ID format validation (6-10 alphanumeric characters)
                    if re.match(r'^[A-Z0-9]{6,10}$', line, re.IGNORECASE):
                        ids.add(line.upper())
    except FileNotFoundError:
        print(f"Error: Input file not found -> {file_path}")
        return None
    return list(ids)

def download_sequences_in_batches(ids, batch_size=1000):
    """
    Download FASTA sequences from UniProt in batches.
    """
    all_sequences = ""
    for i in tqdm(range(0, len(ids), batch_size), desc="Downloading sequences"):
        batch_ids = ids[i:i+batch_size]

        params = {
            "accessions": ",".join(batch_ids),
            "format": "fasta"
        }

        try:
            response = requests.get(UNIPROT_BATCH_URL, params=params, stream=True)
            response.raise_for_status()  # Raise an exception if the request fails
            
            # Stream processing of response content
            for chunk in response.iter_content(chunk_size=8192):
                all_sequences += chunk.decode('utf-8')

        except requests.exceptions.RequestException as e:
            print(f"\nError downloading batch {i//batch_size + 1}: {e}")
            print(f"Failed IDs: {','.join(batch_ids)}")
        
        # Respect API usage rules, wait a bit between requests
        time.sleep(1)
        
    return all_sequences

def main():
    """
    Main function
    """
    base_dir = "../"
    input_file = os.path.join(base_dir, 'raw_data', 'uniprot_ids.txt')
    output_file = os.path.join(base_dir, 'raw_data', 'idmapping.fasta')

    print(f"Reading UniProt IDs from {input_file}...")
    uniprot_ids = get_uniprot_ids(input_file)

    if not uniprot_ids:
        print("Could not find any valid UniProt IDs.")
        return

    print(f"Found {len(uniprot_ids)} unique UniProt IDs.")
    print("Starting to download sequences...")

    sequences = download_sequences_in_batches(uniprot_ids)

    if sequences:
        with open(output_file, 'w') as f:
            f.write(sequences)
        print(f"\nAll sequences have been successfully downloaded and saved to {output_file}")
    else:
        print("\nFailed to download any sequences.")

if __name__ == "__main__":
    # Before running, please make sure you have installed the requests and tqdm libraries:
    # pip install requests tqdm
    #
    # Also, make sure the '../raw_data/ppi.txt' file exists and contains UniProt IDs.
    main()
