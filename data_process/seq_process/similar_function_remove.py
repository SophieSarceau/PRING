import os
import subprocess
import tempfile
import shutil
from tqdm import tqdm
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

species_list = ['ecoli', 'yeast', 'arath']

def extract_function_notation(header):
    """Extract function notation (like TCP4) from FASTA header."""
    try:
        # Format: >sp|Q5R6D0|TCP4_PONAB
        parts = header.split('|')
        if len(parts) >= 3:
            func_species = parts[2].split('_')
            if len(func_species) >= 1:
                return func_species[0]
    except Exception:
        pass
    return None

def calculate_sequence_similarity(seq1, seq2):
    """
    Calculate sequence similarity between two protein sequences using global alignment.
    Returns percentage similarity (0-100).
    """
    # Use global alignment with simple scoring (match=1, mismatch=0)
    alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)

    if not alignments:
        return 0

    alignment = alignments[0]
    identical = 0
    total_positions = 0
    
    for i, j in zip(alignment[0], alignment[1]):
        if i != '-' or j != '-':  # Count any position where at least one sequence has a residue
            total_positions += 1
            if i == j and i != '-' and j != '-':  # Both positions have the same residue
                identical += 1
    
    # Avoid division by zero
    if total_positions == 0:
        return 0

    similarity = (identical / total_positions) * 100
    return similarity


def cluster_by_similarity(input_fasta, output_fasta, tmp_dir, identity_threshold=0.4):
    """Cluster sequences by similarity using MMseqs2."""
    # Create temporary directories
    os.makedirs(tmp_dir, exist_ok=True)

    # Base names for MMseqs2 files
    db_name = os.path.join(tmp_dir, "seqdb")
    cluster_name = os.path.join(tmp_dir, "clusterdb")
    rep_name = os.path.join(tmp_dir, "rep_seq")

    # Create database
    subprocess.run(["mmseqs", "createdb", input_fasta, db_name], check=True)

    # Cluster sequences
    subprocess.run([
        "mmseqs", "cluster", db_name, cluster_name, tmp_dir,
        "--min-seq-id", str(identity_threshold),
        "-c", "0.8",  # coverage threshold
        "--cov-mode", "1"  # coverage mode (1: bi-directional)
    ], check=True)

    # Extract representative sequences
    subprocess.run(["mmseqs", "createsubdb", cluster_name, db_name, rep_name], check=True)

    # Convert to FASTA
    subprocess.run(["mmseqs", "convert2fasta", rep_name, output_fasta], check=True)

    # Count sequences in the output file
    count = sum(1 for _ in SeqIO.parse(output_fasta, "fasta"))
    print(f"Clustered at {identity_threshold*100}% identity: {count} representative sequences")

    return output_fasta

def filter_by_global_similarity(human_fasta, species_fasta, output_fasta, identity_threshold=0.4):
    """
    Filter sequences by comparing species proteins against all human proteins.
    Removes any species protein that has >40% similarity to any human protein.
    """
    print(f"Starting global similarity filtering for {os.path.basename(species_fasta)}")
    
    # Create temporary directory
    tmp_dir = tempfile.mkdtemp()
    try:
        # Create a combined FASTA file with both human and species proteins
        combined_fasta = os.path.join(tmp_dir, "combined.fasta")
        with open(combined_fasta, 'w') as outfile:
            # Add human sequences with a special prefix to identify them later
            for record in SeqIO.parse(human_fasta, "fasta"):
                modified_record = SeqRecord(
                    seq=record.seq,
                    id="HUMAN_" + record.id,
                    description=record.description
                )
                SeqIO.write(modified_record, outfile, "fasta")

            # Add species sequences
            for record in SeqIO.parse(species_fasta, "fasta"):
                modified_record = SeqRecord(
                    seq=record.seq,
                    id="SPECIES_" + record.id,
                    description=record.description
                )
                SeqIO.write(modified_record, outfile, "fasta")
        
        # Run MMseqs2 clustering
        clustered_fasta = os.path.join(tmp_dir, "clustered.fasta")
        cluster_by_similarity(combined_fasta, clustered_fasta, tmp_dir, identity_threshold)

        # Parse the clusters to identify species proteins that cluster with human proteins
        species_to_remove = set()
        cluster_mapping = {}

        # Create a reverse mapping of sequences to their clusters
        # For each cluster representative, find all clustered sequences
        tsv_file = os.path.join(tmp_dir, "clusterdb_cluster.tsv")

        # Extract cluster relationships
        subprocess.run([
            "mmseqs", "createtsv", 
            os.path.join(tmp_dir, "seqdb"), 
            os.path.join(tmp_dir, "seqdb"), 
            os.path.join(tmp_dir, "clusterdb"),
            tsv_file
        ], check=True)

        # Read the TSV file to identify clusters
        with open(tsv_file, 'r') as f:
            for line in f:
                rep_seq, member_seq = line.strip().split('\t')

                if rep_seq not in cluster_mapping:
                    cluster_mapping[rep_seq] = []
                cluster_mapping[rep_seq].append(member_seq)

                # If representative is human and member is species, mark for removal
                if rep_seq.startswith("HUMAN_") and member_seq.startswith("SPECIES_"):
                    species_to_remove.add(member_seq.replace("SPECIES_", ""))
                # If representative is species and any member is human, mark representative for removal
                elif rep_seq.startswith("SPECIES_"):
                    for member in cluster_mapping[rep_seq]:
                        if member.startswith("HUMAN_"):
                            species_to_remove.add(rep_seq.replace("SPECIES_", ""))
                            break

        # Write filtered species sequences to output file
        kept_records = []
        total_count = 0
        removed_count = 0

        for record in SeqIO.parse(species_fasta, "fasta"):
            total_count += 1
            if record.id not in species_to_remove:
                kept_records.append(record)
            else:
                removed_count += 1

        SeqIO.write(kept_records, output_fasta, "fasta")
        print(f"Global similarity filtering: Removed {removed_count} out of {total_count} sequences ({removed_count/max(total_count, 1)*100:.2f}%)")

    finally:
        # Clean up temporary files
        shutil.rmtree(tmp_dir)

    return output_fasta

def main():
    # Path setup
    species_fasta_dir = "../clustered_seq"
    human_fasta_path = "../clustered_seq/human_clustered.fasta"
    output_dir = "../func_removed_seq"
    final_output_dir = "../func_removed_seq"

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)

    # copy human.fasta to output_dir
    os.system(f"cp {human_fasta_path} {output_dir+'/human.fasta'}")

    # Read human function notations
    human_functions = {}
    try:
        for record in SeqIO.parse(human_fasta_path, "fasta"):
            func_notation = extract_function_notation(record.id)
            if func_notation:
                # add the protein sequence to the dictionary
                human_functions[func_notation] = record.seq
        print(f"Loaded {len(human_functions)} unique function notations from human.fasta")
    except FileNotFoundError:
        print(f"Error: Could not find human.fasta at {human_fasta_path}")
        return

    # Process each species fasta file
    for species in species_list:
        species_file = os.path.join(species_fasta_dir, f"{species}_clustered.fasta")
        intermediate_output_file = os.path.join(output_dir, f"{species}.fasta")
        final_output_file = os.path.join(final_output_dir, f"{species}.fasta")

        if not os.path.exists(species_file):
            print(f"Warning: {species_file} not found, skipping...")
            continue

        # Filter sequences by function notation
        kept_records = []
        skipped_count = 0
        total_count = 0

        # Get total number of sequences first (for accurate progress tracking)
        total_sequences = sum(1 for _ in SeqIO.parse(species_file, "fasta"))

        for record in tqdm(SeqIO.parse(species_file, "fasta"), 
                        total=total_sequences, 
                        desc=f"Processing {species}", 
                        unit="seq", 
                        ncols=80, 
                        colour="green"):
            total_count += 1
            func_notation = extract_function_notation(record.id)

            if func_notation and func_notation in human_functions:
                # Calculate sequence similarity
                human_seq = human_functions[func_notation]
                current_seq = record.seq
                similarity = calculate_sequence_similarity(human_seq, current_seq)

                # Keep sequence if similarity < 40%
                if similarity < 10:
                    kept_records.append(record)
                    continue

                skipped_count += 1
                continue

            kept_records.append(record)

        # Write filtered sequences
        SeqIO.write(kept_records, intermediate_output_file, "fasta")

        print(f"{species}: Removed {skipped_count} out of {total_count} sequences ({skipped_count/total_count*100:.2f}%)")

        # Further filter using global sequence similarity
        print(f"Starting global similarity filtering for {species}...")
        filter_by_global_similarity(
            human_fasta=os.path.join(output_dir, "human.fasta"), 
            species_fasta=intermediate_output_file,
            output_fasta=final_output_file,
            identity_threshold=0.4
        )

        # Count sequences in the final output
        final_count = sum(1 for _ in SeqIO.parse(final_output_file, "fasta"))
        print(f"Final {species} dataset: {final_count} sequences after all filtering steps")

if __name__ == "__main__":
    main()
