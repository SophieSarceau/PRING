import os
import subprocess
import tempfile
import shutil
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def filter_by_length(input_fasta, output_fasta, min_length=50, max_length=1000):
    """Filter sequences by length and write to a new FASTA file."""
    filtered_records = []

    print("The total number of protein sequences in the fasta file is: ", sum(1 for _ in SeqIO.parse(input_fasta, "fasta")))
    # Read the input FASTA file
    for record in tqdm(SeqIO.parse(input_fasta, "fasta"), desc="Filtering sequences by length"):
        seq_length = len(record.seq)
        if min_length <= seq_length <= max_length:
            filtered_records.append(record)

    # Write filtered sequences to output file
    SeqIO.write(filtered_records, output_fasta, "fasta")

    print(f"Filtered by length: {len(filtered_records)} sequences (from {min_length} to {max_length} amino acids)")

    return output_fasta

def main():
    # Input and output file paths
    input_fasta = "../raw_data/idmapping.fasta"
    length_filtered_fasta = "../raw_data/idmapping_len50-1k.fasta"

    # Create a temporary directory
    tmp_dir = tempfile.mkdtemp()

    try:
        # Step 1: Filter by sequence length
        length_filtered = filter_by_length(input_fasta, length_filtered_fasta, 
                                          min_length=50, max_length=1000)

    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()
