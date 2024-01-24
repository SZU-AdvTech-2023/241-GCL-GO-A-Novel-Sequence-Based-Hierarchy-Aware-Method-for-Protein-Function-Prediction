import numpy as np
from config import get_config

# Configuration and amino acid mappings
args = get_config()
amino_acids = 'ARNDCEQGHILKMFPSTWYVBZOUX'
AA_index = {aa: idx for idx, aa in enumerate(['pad'] + list(amino_acids), start=0)}

def convert_to_onehot(sequence, start_index=0):
    # Create a zero matrix for one-hot encoding
    encoded_matrix = np.zeros((args.maxlen, len(AA_index)), dtype=np.int32)
    
    # Determine the length of the sequence to process
    seq_length = min(args.maxlen, len(sequence))

    # Encode the sequence into the one-hot matrix
    for i in range(start_index, start_index + seq_length):
        amino_acid = sequence[i - start_index]
        encoded_matrix[i, AA_index.get(amino_acid, 0)] = 1

    # Padding the start and end of the sequence if necessary
    encoded_matrix[:start_index, 0] = 1
    encoded_matrix[start_index + seq_length:, 0] = 1

    return encoded_matrix
