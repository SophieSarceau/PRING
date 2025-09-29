import os
import subprocess
import tempfile
import shutil
from tqdm import tqdm
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


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

def calculate_sequence_statistics(original_fastas, clustered_seqs):
    """
    Calculate and display statistics about original and clustered sequence files.

    Args:
        original_fastas (list): List of paths to original FASTA files
        clustered_seqs (list): List of paths to clustered FASTA files

    Returns:
        dict: Dictionary containing statistics
    """
    stats = []
    total_original = 0
    total_clustered = 0

    print("\n=== Sequence Clustering Statistics ===\n")
    print(f"{'Species':<10} {'Original':<10} {'Clustered':<10} {'Reduction %':<12}")
    print("-" * 45)

    # Process each pair of files
    for orig_path, clustered_path in zip(original_fastas, clustered_seqs):
        species = os.path.basename(orig_path).split('.')[0]

        # Count sequences
        try:
            original_count = sum(1 for _ in SeqIO.parse(orig_path, "fasta"))
            clustered_count = sum(1 for _ in SeqIO.parse(clustered_path, "fasta"))

            # Calculate reduction percentage
            reduction = ((original_count - clustered_count) / original_count * 100) if original_count > 0 else 0

            # Add to totals
            total_original += original_count
            total_clustered += clustered_count

            # Store and display statistics
            stats.append({
                'species': species,
                'original': original_count,
                'clustered': clustered_count,
                'reduction': reduction
            })

            print(f"{species:<10} {original_count:<10} {clustered_count:<10} {reduction:.2f}%")

        except Exception as e:
            print(f"{species:<10} Error: {str(e)}")

    # Calculate and display totals
    total_reduction = ((total_original - total_clustered) / total_original * 100) if total_original > 0 else 0
    print("-" * 45)
    print(f"{'TOTAL':<10} {total_original:<10} {total_clustered:<10} {total_reduction:.2f}%")

    # Save statistics to CSV
    stats_df = pd.DataFrame(stats)
    stats_file = "../clustered_seq/clustering_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"\nStatistics saved to {stats_file}")

    return {
        'statistics': stats,
        'total_original': total_original,
        'total_clustered': total_clustered,
        'total_reduction': total_reduction
    }

def main():
    # Input and output file paths
    input_fastas = [
        '../species_seq/human.fasta',
        '../species_seq/arath.fasta',
        '../species_seq/ecoli.fasta',
        '../species_seq/yeast.fasta'
    ]

    # Create output directory for clustered sequences
    output_dir = '../clustered_seq'
    os.makedirs(output_dir, exist_ok=True)

    # Create temporary directory for MMseqs2 files
    tmp_dir = '../tmp_mmseqs2'
    os.makedirs(tmp_dir, exist_ok=True)

    # Keep track of processed output files
    output_fastas = []

    # Process each species file
    for input_fasta in input_fastas:
        # Get species name from filename
        species_name = os.path.basename(input_fasta).split('.')[0]
        print(f"\nProcessing {species_name}...")

        # Set output path
        output_fasta = os.path.join(output_dir, f"{species_name}_clustered.fasta")
        output_fastas.append(output_fasta)

        # Create species-specific temp directory
        species_tmp_dir = os.path.join(tmp_dir, species_name)

        # Cluster sequences
        try:
            clustered_seq = cluster_by_similarity(
                input_fasta=input_fasta,
                output_fasta=output_fasta,
                tmp_dir=species_tmp_dir,
                identity_threshold=0.4  # 40% sequence identity threshold
            )
            print(f"Successfully clustered {species_name} sequences: {clustered_seq}")
        except Exception as e:
            print(f"Error processing {species_name}: {str(e)}")

    print("\nClustering complete. Clustered sequences are in:", output_dir)

    # Calculate and display statistics
    try:
        import pandas as pd  # Make sure pandas is imported
        stats = calculate_sequence_statistics(input_fastas, output_fastas)
    except Exception as e:
        print(f"Could not generate statistics: {str(e)}")

    # Optionally clean up temporary files
    try:
        shutil.rmtree(tmp_dir)
        print(f"Removed temporary directory: {tmp_dir}")
    except Exception as e:
        print(f"Note: Could not remove temporary directory {tmp_dir}: {str(e)}")

if __name__ == "__main__":
    main()
