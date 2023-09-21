import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from data_utils import load_dataset
from device import DEVICE
from networks import MyModel, CustomEmbedding  # Import your custom model module
from train_utils import train, evaluate


def choose_random(param_dist):
    random_params = {}
    for key, value_list in param_dist.items():
        random_params[key] = np.random.choice(value_list)
    return random_params


# Define your hyperparameter search space
param_dist = {
    'embed_dim': [100],
    'peptide_lstm_hidden_size': [32, 64, 128],
    'mhc_lstm_hidden_size': [64, 128, 256],
    'lstm_peptide_num_layers': [2, 3, 4],
    'lstm_mhc_num_layers': [3, 4, 5],
    'linear1_out': [32, 64, 128],
    'linear2_out': [1]
}

# Load the dataset using the load_dataset function
data_path = 'data/final_dataset.csv'
train_loader, valid_loader, test_loader = load_dataset(data_path)  # Replace data_path with the actual path
mhc_embeddings = torch.load('trained_models/mhc_embedding_dict')
peptide_embeddings = torch.load('trained_models/peptide_embedding_dict')
mhc_embedding_layer = CustomEmbedding(mhc_embeddings)
peptide_embedding_layer = CustomEmbedding(peptide_embeddings)
BCE_criterion = nn.BCELoss().to(DEVICE)
EPOCHS = 3


# Define a function to create and evaluate the model
def create_and_evaluate_model(**params):
    model = MyModel(
        mhc_embedder=mhc_embedding_layer,
        peptide_embedder=peptide_embedding_layer,
        **params
    ).to(DEVICE)
    adam_optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Define your training loop and evaluation logic here
    loss = 0
    epoch = 0
    for _ in tqdm(range(EPOCHS), desc="Epochs"):
        test = train(model, train_loader, BCE_criterion, adam_optimizer, epoch, EPOCHS)
        loss += evaluate(model, valid_loader, BCE_criterion, epoch, EPOCHS)
        epoch += 1
        # convergence test
        if test == -1:
            break
    # Use validation data for hyperparameter tuning
    return loss / epoch, model.state_dict()  # Return the evaluation metric (e.g., loss) to minimize


min_loss = float('inf')
best_model = dict()
best_param = dict()
iterations = 10
for iteration in range(iterations):
    print(f'iteration {iteration}:')
    parameters = choose_random(param_dist=param_dist)
    loss, model_state_dict = create_and_evaluate_model(**parameters)
    if loss < min_loss:
        min_loss = loss
        best_model = model_state_dict
        best_param = parameters

print('best hyperparameters found:')
print(best_param)
