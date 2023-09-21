import torch
import torch.nn as nn

from device import DEVICE  # Import the device configuration


class CustomEmbedding(nn.Module):
    """
    A custom embedding layer that maps input tokens to their corresponding embeddings.

    Parameters:
        embedding_dict (dict): A dictionary containing token_index-to-embedding mappings.

    Returns:
        padded_embeddings (torch.Tensor): Padded sequence of embeddings for input token_indexes.
    """

    def __init__(self, embedding_dict):
        super(CustomEmbedding, self).__init__()
        self.embedding_dict = embedding_dict

    def forward(self, input_tokens):
        embeddings = [self.embedding_dict[token.item()] for token in input_tokens]
        padded_embeddings = nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        return padded_embeddings


class MyModel(nn.Module):
    """
    A custom neural network model for predicting binding affinities between peptides and MHC alleles.

    Parameters:
        mhc_embedder (nn.Module): Embedding layer for MHC sequences.
        peptide_embedder (nn.Module): Embedding layer for peptide sequences.
        embed_dim (int): Dimension of embeddings.
        peptide_lstm_hidden_size (int): Hidden size of LSTM for peptides.
        mhc_lstm_hidden_size (int): Hidden size of LSTM for MHC alleles.
        lstm_peptide_num_layers (int): Number of layers in the peptide LSTM.
        lstm_mhc_num_layers (int): Number of layers in the MHC LSTM.
        linear1_out (int): Output dimension of the first linear layer.
        linear2_out (int): Output dimension of the final linear layer.

    Returns:
        out (torch.Tensor): Predicted binding affinities.
    """

    def __init__(
            self,
            mhc_embedder,
            peptide_embedder,
            embed_dim,
            peptide_lstm_hidden_size,
            mhc_lstm_hidden_size,
            lstm_peptide_num_layers,
            lstm_mhc_num_layers,
            linear1_out,
            linear2_out,
    ):
        super(MyModel, self).__init__()
        self.peptide_LSTM_hidden_size = peptide_lstm_hidden_size
        self.mhc_LSTM_hidden_size = mhc_lstm_hidden_size

        self.peptide_num_layers = lstm_peptide_num_layers
        self.mhc_num_layers = lstm_mhc_num_layers

        self.mhc_embedder = mhc_embedder
        self.peptide_embedder = peptide_embedder
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

        self.peptide_lstm = nn.LSTM(embed_dim, self.peptide_LSTM_hidden_size, num_layers=self.peptide_num_layers,
                                    batch_first=True)
        self.mhc_lstm = nn.LSTM(embed_dim, self.mhc_LSTM_hidden_size, num_layers=self.mhc_num_layers, batch_first=True)

        self.peptide_linear = nn.Linear(self.peptide_LSTM_hidden_size * 2, linear1_out)
        self.mhc_linear = nn.Linear(self.mhc_LSTM_hidden_size * 2, linear1_out)
        self.hidden_linear = nn.Linear(2 * linear1_out, linear1_out)
        self.out_linear = nn.Linear(linear1_out, linear2_out)

    def forward(self, sequences_indexes):
        mhc = sequences_indexes[:, 0]
        peptide = sequences_indexes[:, 1]
        pep_emb = self.peptide_embedder(peptide).to(DEVICE)
        mhc_emb = self.mhc_embedder(mhc).to(DEVICE)

        pep_lstm_output, (pep_last_hidden_state, pep_last_cell_state) = self.peptide_lstm(pep_emb)
        mhc_lstm_output, (mhc_last_hidden_state, mhc_last_cell_state) = self.mhc_lstm(mhc_emb)

        pep_last_hidden_layer = pep_last_hidden_state[-1]
        mhc_last_hidden_layer = mhc_last_hidden_state[-1]
        pep_last_cell_layer = pep_last_cell_state[-1]
        mhc_last_cell_layer = mhc_last_cell_state[-1]

        pep_linear_out = self.relu(self.peptide_linear(torch.cat((pep_last_hidden_layer, pep_last_cell_layer), dim=1)))
        mhc_linear_out = self.relu(self.mhc_linear(torch.cat((mhc_last_hidden_layer, mhc_last_cell_layer), dim=1)))

        conc = torch.cat((pep_linear_out, mhc_linear_out), dim=1)

        out = self.sigmoid(self.out_linear(self.hidden_linear(conc)))

        return out
