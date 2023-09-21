import biovec  # Import the biovec library for working with protein sequence embeddings
import numpy as np  # Import the numpy library for numerical computations
import pandas as pd  # Import the pandas library for data manipulation
import torch  # Import the PyTorch library for deep learning operations
from torch import tensor  # Import the tensor class from PyTorch for tensor operations


def split_string(input_string, chunk_size):
    """
    Split an input string into chunks of a given size.

    Parameters:
        input_string (str): The input string to be split.
        chunk_size (int): The size of each chunk.

    Returns:
        List[str]: A list of chunks obtained from the input string.
    """
    return [input_string[i:i + chunk_size] for i in range(0, len(input_string) - chunk_size, chunk_size)]


def seq_to_3gram(sequence):
    """
    Convert a protein sequence into a list of 3-grams.

    Parameters:
        sequence (str): The protein sequence to be converted.

    Returns:
        List[str]: A list of 3-grams obtained from the protein sequence.
    """
    n_grams = split_string(sequence, 3)
    # in the case where the input's length is not divisible by 3 there is an overlapping sequence at the end of the returned list
    n_grams.append(sequence[-3:])
    return n_grams


def embed(tokens):
    """
    Embed a list of protein sequence tokens using pretrained ProtVec embeddings.

    Parameters:
        tokens (List[str]): The list of protein sequence tokens to be embedded.

    Returns:
        torch.Tensor: A tensor containing the embeddings of the input tokens.
    """
    global prot_vec  # Reference to the ProtVec model
    embeddings = []
    for token in tokens:
        embeddings.append(prot_vec.wv[token])  # Obtain embeddings using ProtVec model
    return tensor(np.vstack(embeddings))  # Convert embeddings to a PyTorch tensor


# Load the pretrained ProtVec model
prot_vec = biovec.models.load_protvec('trained_models/swissprot-reviewed-protvec.model')

if __name__ == '__main__':
    # Read the binding affinity dataset
    dataset = pd.read_csv('data/binding_affinity_data.csv', index_col=0)
    print('binding_affinity_data is imported!')

    # Load the MHC embeddings
    MHC = dataset['MHC_sequence'].unique().tolist()
    print('MHC_embeddings are imported!')
    # Create 3-gram embeddings for MHC sequences and save them
    torch.save(pd.Series(dataset['MHC_sequence'].unique()).apply(seq_to_3gram).apply(embed).to_dict(),
               'trained_models/mhc_embedding_dict')
    print('mhc_3gram are saved in trained_models/mhc_embedding_dict')

    # Create the mapping from MHC sequences to integer indices using a dictionary
    mhc_to_index = {sequence: index for index, sequence in enumerate(MHC)}

    mhc_sequences = list(dataset['MHC_sequence'])
    # Convert MHC sequences to their corresponding integer indices
    mhc_indices = [mhc_to_index.get(sequence, -1) for sequence in mhc_sequences]

    # Add the mapping to dataset
    dataset['MHC_sequence_index'] = pd.Series(mhc_indices)
    print('MHC_embeddings mappings are added to the dataset!')

    # Load the peptide embeddings
    peptide = dataset['peptide_sequence'].unique().tolist()
    print('peptide_embeddings are imported!')
    # Create 3-gram embeddings for peptide sequences and save them
    torch.save(pd.Series(dataset['peptide_sequence'].unique()).apply(seq_to_3gram).apply(embed).to_dict(),
               'trained_models/peptide_embedding_dict')
    print('peptides_3gram are saved in trained_models/peptide_embedding_dict')

    # Create the mapping from peptide sequences to integer indices using a dictionary
    peptides_to_index = {sequence: index for index, sequence in enumerate(peptide)}

    peptide_sequences = list(dataset['peptide_sequence'])
    # Convert peptide sequences to their corresponding integer indices
    peptides_indices = [peptides_to_index.get(sequence, -1) for sequence in peptide_sequences]

    # Add the mapping to dataset
    dataset['peptide_sequence_index'] = pd.Series(peptides_indices)
    print('peptide_embeddings mappings are added to the dataset!')

    # Save the result
    dataset.loc[:, ['label', 'MHC_sequence_index', 'peptide_sequence_index']].to_csv('data/final_dataset.csv')
    print('The result is saved in data/final_dataset.csv.')
