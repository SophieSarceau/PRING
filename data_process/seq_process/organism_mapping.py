import os
import pandas as pd
from Bio import SeqIO
import re
from tqdm import tqdm

def parse_fasta_to_dataframe(fasta_file):
    """
    Parse a FASTA file to extract UniProt ID, organism ID (OX), and sequence.

    Args:
        fasta_file: Path to the FASTA file

    Returns:
        pandas DataFrame with columns: uniprot_id, organism_id, sequence, sequence length
    """
    records = []

    # First, count the total number of records for the progress bar
    total_records = sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))

    # Now process each record with a progress bar
    with tqdm(total=total_records, desc=f"Parsing {os.path.basename(fasta_file)}") as pbar:
        for record in SeqIO.parse(fasta_file, "fasta"):
            # Extract UniProt ID from the record ID (format: sp|P50399|GDIB_RAT)
            uniprot_id = record.id.split('|')[1] if '|' in record.id else record.id

            # Extract OX from the description using regex
            ox_match = re.search(r'OX=(\d+)', record.description)
            organism_id = ox_match.group(1) if ox_match else None

            # Get the sequence as a string
            sequence = str(record.seq)

            records.append({
                'uniprot_id': uniprot_id,
                'organism_id': organism_id,
                'sequence': sequence,
                'sequence_length': len(sequence)
            })

            # Update the progress bar
            pbar.update(1)

    return pd.DataFrame(records)

def process_clustered_fasta_folder(folder_path, output_folder):
    """
    Process all FASTA files in a clustered_fasta folder and create a CSV file for each species.

    Args:
        folder_path: Path to the folder containing FASTA files for different species
        output_folder: Path to save the CSV files
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all FASTA files in the folder
    fasta_files = [f for f in os.listdir(folder_path) if f.endswith(('.fasta', '.fa'))]

    print(f"Found {len(fasta_files)} FASTA files in {folder_path}")

    # Process each FASTA file
    for fasta_file in tqdm(fasta_files):
        file_path = os.path.join(folder_path, fasta_file)

        # Get species name from filename (assuming format like "species_name.fasta")
        species_name = os.path.splitext(fasta_file)[0]

        print(f"Processing {species_name}...")

        # Parse FASTA file to DataFrame
        df = parse_fasta_to_dataframe(file_path)

        # Create output CSV filename
        output_file = os.path.join(output_folder, f"{species_name}_protein_id.csv")

        # Save DataFrame to CSV
        df.to_csv(output_file, index=False)
        print(f"Data for {species_name} saved to {output_file}")

        # Print summary
        print(f"  - Total proteins: {len(df)}")
        print(f"  - Unique organism IDs: {df['organism_id'].nunique()}")

def main():
    # Path to the folder containing clustered FASTA files
    clustered_fasta_folder = "../func_removed_seq"  # Adjust the path as needed

    # Path to save the output CSV files
    output_folder = "../func_removed_seq"  # Adjust the path as needed

    # Check if the folder exists
    if not os.path.exists(clustered_fasta_folder):
        print(f"Error: Folder {clustered_fasta_folder} not found.")
        return

    # Process all FASTA files in the folder
    process_clustered_fasta_folder(clustered_fasta_folder, output_folder)

if __name__ == "__main__":
    main()
