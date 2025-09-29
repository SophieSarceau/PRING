import os
import glob
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


def simplify_fasta(input_file, output_file):
    """
    Simplify FASTA headers to only contain UniProt IDs using BioPython.
    Also ensures sequences are written as a single line.
    Example: >sp|P25786|PSA1_HUMAN Proteasome ... -> >P25786
    """
    simplified_records = []

    for record in SeqIO.parse(input_file, "fasta"):
        # Get header and extract UniProt ID
        header = record.id
        parts = header.split('|')

        if len(parts) >= 3 and parts[0] == 'sp':
            # Standard UniProt format
            new_id = parts[1]
        else:
            # Keep original ID without description
            new_id = header.split()[0]

        # Create new record with simplified ID and original sequence
        new_record = SeqRecord(record.seq, id=new_id, description="")
        simplified_records.append(new_record)

    # Write all records to output file with sequences on one line
    with open(output_file, 'w') as f_out:
        for record in simplified_records:
            f_out.write(f">{record.id}\n")
            f_out.write(f"{str(record.seq)}\n")


def process_all_species():
    """
    Find all species_clustered.fasta files in species folders and simplify them
    """
    base_dir = "../species_processed_data"
    species_dirs = glob.glob(os.path.join(base_dir, "*/"))

    for species_dir in species_dirs:
        species_name = os.path.basename(os.path.normpath(species_dir))
        fasta_file = os.path.join(species_dir, f"{species_name}.fasta")
        if os.path.isfile(fasta_file):
            output_file = os.path.join(species_dir, f"{species_name}_simple.fasta")
            print(f"Processing {fasta_file}")
            simplify_fasta(fasta_file, output_file)
            print(f"Simplified file saved to {output_file}")

if __name__ == "__main__":
    process_all_species()
