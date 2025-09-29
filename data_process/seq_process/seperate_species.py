import os
from tqdm import tqdm
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import re

SPECIES_LIST = {
    '9606': 'HUMAN',
    '3702': 'ARATH',
    '83333': 'ECOLI',
    '559292': 'YEAST'
}

def extract_taxonomy_id(header):
    """Extract taxonomy ID (OX) from FASTA header"""
    match = re.search(r'OX=(\d+)', header)
    if match:
        return match.group(1)
    return None

def separate_species(input_fasta, output_dir):
    """
    Separate sequences in a FASTA file according to different species using OX codes.

    Args:
        input_fasta (str): Path to the input FASTA file
        output_dir (str): Directory to save species-specific FASTA files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize containers for sequences by species
    species_records = defaultdict(list)
    other_species = []

    # Count total records for progress bar
    total_records = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))
    print(f"Processing {total_records} sequences...")

    # Process each sequence in the FASTA file
    for record in tqdm(SeqIO.parse(input_fasta, "fasta"), total=total_records, desc="Separating species"):
        # Extract taxonomy ID from the record description
        taxon_id = extract_taxonomy_id(record.description)

        if taxon_id and taxon_id in SPECIES_LIST:
            species_records[taxon_id].append(record)
        else:
            other_species.append(record)

    # Write species-specific FASTA files
    for taxon_id, records in species_records.items():
        species_name = SPECIES_LIST[taxon_id]
        output_file = os.path.join(output_dir, f"{species_name.lower()}.fasta")

        SeqIO.write(records, output_file, "fasta")
        print(f"Wrote {len(records)} sequences for {species_name} (Taxon ID: {taxon_id}) to {output_file}")

    # Write sequences from other species
    if other_species:
        other_output = os.path.join(output_dir, "other_species.fasta")
        SeqIO.write(other_species, other_output, "fasta")
        print(f"Wrote {len(other_species)} sequences from other species to {other_output}")

    # Create detailed stats about the species distribution
    all_species = defaultdict(int)
    for record in SeqIO.parse(input_fasta, "fasta"):
        taxon_id = extract_taxonomy_id(record.description)
        if taxon_id:
            all_species[taxon_id] += 1

    # Create summary stats for tracked species
    summary = []
    for taxon_id, count in all_species.items():
        species_name = SPECIES_LIST.get(taxon_id, "Unknown")
        is_tracked = taxon_id in SPECIES_LIST
        summary.append({
            'Taxon ID': taxon_id,
            'Species': species_name,
            'Sequence Count': count,
            'Is Tracked': is_tracked
        })

    # Sort by sequence count descending
    summary = sorted(summary, key=lambda x: x['Sequence Count'], reverse=True)

    # Save statistics
    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(output_dir, "species_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to {summary_file}")

if __name__ == "__main__":
    # Input file path - change this to your input FASTA file
    input_fasta = "../raw_data/idmapping_len50-1k.fasta"

    # Output directory for species-specific files
    output_dir = "../species_seq"

    separate_species(input_fasta, output_dir)
