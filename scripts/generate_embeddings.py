"""
Script to generate embeddings of protein sequence from just sequence data
"""

from bio_embeddings.embed import ProtTransBertBFDEmbedder
from Bio import SeqIO
import pickle as pkl
from joblib import Parallel, delayed
import argparse


def get_embedding(sequence):
    """
    Function that calculates embedding for the protein sequence and stores it in the list variable
    :param
    sequence: Protein sequence object from biopython
    embedder: bio_embedding.embed object
    :return:
    None
    """
    per_residue_embedding = embedder.embed(str(sequence.seq))
    per_protein_embedding = embedder.reduce_per_protein(per_residue_embedding)
    dict_of_embeddings[sequence.id] = per_protein_embedding
    # print(sequence.id, per_protein_embedding)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', help='Input file containing protein sequences (.fasta) format', required=True)
    parser.add_argument('--o', help='Output file to store the sequence IDs and embeddings', required=True)
    args = parser.parse_args()

    embedder = ProtTransBertBFDEmbedder()
    dict_of_embeddings = {}

    input_file = args.i
    Parallel(n_jobs=8, require='sharedmem')(
        delayed(get_embedding)(sequence) for sequence in SeqIO.parse(input_file, 'fasta'))

    output_file = args.o
    with open(output_file, 'wb') as f:
        pkl.dump(dict_of_embeddings, f)
