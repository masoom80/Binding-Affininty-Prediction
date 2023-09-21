from torch import nn, optim

import config
from train_utils import train, evaluate

from data_utils import load_dataset
from networks import MyModel, CustomEmbedding
from device import DEVICE
import torch.optim.lr_scheduler as lr_scheduler
import torch
from tqdm import tqdm

data_path = 'data/final_dataset.csv'
train_loader, valid_loader, test_loader = load_dataset(data_path)
torch.save(test_loader, 'data/test_dataloader')

# Load embeddings
mhc_embeddings = torch.load('trained_models/mhc_embedding_dict')
peptide_embeddings = torch.load('trained_models/peptide_embedding_dict')
mhc_embedding_layer = CustomEmbedding(mhc_embeddings)
peptide_embedding_layer = CustomEmbedding(peptide_embeddings)
print('Adding the embeddings layer to the model.')

# Define the constants
EMBED_DIM = config.EMBED_DIM
PEPTIDE_LSTM_HIDDEN_SIZE = config.PEPTIDE_LSTM_HIDDEN_SIZE
MHC_LSTM_HIDDEN_SIZE = config.MHC_LSTM_HIDDEN_SIZE
LSTM_PEPTIDE_NUM_LAYERS = config.LSTM_PEPTIDE_NUM_LAYERS
LSTM_MHC_NUM_LAYERS = config.LSTM_MHC_NUM_LAYERS
LINEAR1_OUT = config.LINEAR1_OUT
LINEAR2_OUT = config.LINEAR2_OUT

# Instantiate the model
classifier = MyModel(
    mhc_embedder=mhc_embedding_layer,
    peptide_embedder=peptide_embedding_layer,
    embed_dim=EMBED_DIM,
    peptide_lstm_hidden_size=PEPTIDE_LSTM_HIDDEN_SIZE,
    mhc_lstm_hidden_size=MHC_LSTM_HIDDEN_SIZE,
    lstm_peptide_num_layers=LSTM_PEPTIDE_NUM_LAYERS,
    lstm_mhc_num_layers=LSTM_MHC_NUM_LAYERS,
    linear1_out=LINEAR1_OUT,
    linear2_out=LINEAR2_OUT
).to(DEVICE)
# Define loss function and optimizer
BCE_criterion = nn.BCELoss().to(DEVICE)
adam_optimizer = optim.Adam(classifier.parameters(), lr=0.01)
# Define a learning rate scheduler
scheduler = lr_scheduler.StepLR(adam_optimizer, step_size=1, gamma=0.1)

# Training loop
EPOCHS = 5
check_validation_loss = True
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    train_loss = train(classifier, train_loader, BCE_criterion, adam_optimizer, epoch, EPOCHS,
                       convergence_test_period=500)
    torch.save(classifier.state_dict(), f'trained_models/model_epoch{epoch}')
    if check_validation_loss:
        validation_loss = evaluate(classifier, valid_loader, BCE_criterion, epoch, EPOCHS)
        print(f'[{epoch + 1}] Training Loss: {train_loss:.5f} Validation Loss: {validation_loss:.5f}')
    if train_loss == -1:
        break
    scheduler.step()

# Save test dataset and model
print('Finished Training')
