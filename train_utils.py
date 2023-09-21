import torch
from tqdm import tqdm

from device import DEVICE


def train(model, data_loader, criterion, optimizer, epoch, EPOCHS, convergence_test_period=100,
          convergence_test_threshold=1e-6):
    """
    Train the given model using the provided data_loader for a single epoch.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        data_loader (torch.utils.data.DataLoader): DataLoader providing training data.
        criterion: The loss function to measure the training loss.
        optimizer: The optimization algorithm for updating model parameters.
        epoch (int): The current epoch number.
        EPOCHS (int): The max number of epochs.
        convergence_test_period (int): The period length for testing convergence.
        convergence_test_threshold (int): Threshold for convergence test.
    Returns:
        float: Average training loss for the epoch.
    """

    model.train()
    training_loss = 0.0
    with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False) as pbar:
        last_training_loss = float('inf')
        for i, data in enumerate(pbar):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            if i % 10 == 0:
                pbar.set_postfix({"Training Loss per batch": f"{training_loss / (i + 1):.5f}"})
            if i % convergence_test_period == 0:
                if last_training_loss - training_loss / (i + 1) < convergence_test_threshold:
                    print(f"converged at {i + 1} minibatch")
                    return -1
                last_training_loss = training_loss / (i + 1)

    return training_loss / len(data_loader)


def evaluate(model, data_loader, criterion, epoch, EPOCHS):
    """
    Evaluate the given model's performance on the provided data_loader.

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader providing evaluation data.
        criterion: The loss function to measure the evaluation loss.
        epoch (int): The current epoch number.
        EPOCHS (int): The max number of epochs
    Returns:
        float: Average evaluation loss.

    """
    model.eval()
    loss = 0.0
    with torch.no_grad():
        with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False) as pbar:
            for i, data in enumerate(data_loader):
                inputs, labels = data
                outputs = model(inputs)
                loss += criterion(outputs.squeeze(), labels.to(DEVICE)).item()
                if i % 10 == 0:
                    pbar.set_postfix({"validation Loss per batch": f"{loss / (i + 1):.5f}"})
    return loss / len(data_loader)
