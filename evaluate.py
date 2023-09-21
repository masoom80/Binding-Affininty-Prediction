import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, \
    precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

import config
from device import DEVICE
from networks import MyModel, CustomEmbedding

# Load the saved dataset and model

dataloader = torch.load('data/test_dataset')

mhc_embeddings = torch.load('trained_models/mhc_embedding_dict')
peptide_embeddings = torch.load('trained_models/peptide_embedding_dict')
mhc_embedding_layer = CustomEmbedding(mhc_embeddings)
peptide_embedding_layer = CustomEmbedding(peptide_embeddings)
# Define the constants
EMBED_DIM = config.EMBED_DIM
PEPTIDE_LSTM_HIDDEN_SIZE = config.PEPTIDE_LSTM_HIDDEN_SIZE
MHC_LSTM_HIDDEN_SIZE = config.MHC_LSTM_HIDDEN_SIZE
LSTM_PEPTIDE_NUM_LAYERS = config.LSTM_PEPTIDE_NUM_LAYERS
LSTM_MHC_NUM_LAYERS = config.LSTM_MHC_NUM_LAYERS
LINEAR1_OUT = config.LINEAR1_OUT
LINEAR2_OUT = config.LINEAR2_OUT

# Instantiate the model
model = MyModel(
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
model.load_state_dict(torch.load('trained_models/model_epoch4'))
# Set the model to evaluation mode
model.eval()
print('the model is ready for evaluation.')

# Create a data loader for evaluation
true_labels = []
probs = []
predicted_labels = []
with torch.no_grad():
    for data in dataloader:
        inputs, labels = data
        true_labels.extend(labels.cpu().numpy())
        outputs = model(inputs)
        predicted = (outputs >= 0.5).float()  # Convert probabilities to binary predictions (0 or 1)
        probs.extend(outputs.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
true_labels = np.vstack(true_labels)
probs = np.vstack(probs)
print('predicted_labels are ready!')
# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

# Calculate and display confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(true_labels, probs[:, 0])
roc_auc = auc(fpr, tpr)

# Calculate and plot precision-recall curve
precision_curve, recall_curve, _ = precision_recall_curve(true_labels, probs[:, 0])
average_precision = average_precision_score(true_labels, probs[:, 0])

# Display ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Display precision-recall curve
plt.figure()
plt.plot(recall_curve, precision_curve, color='darkorange', lw=2,
         label='Precision-Recall curve (AP = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")

plt.show()

# Display evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")
