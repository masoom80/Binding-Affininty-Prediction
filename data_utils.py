from Dataset import BindingAffinityDataset  # Import your custom dataset module
from torch.utils.data import DataLoader, random_split


def load_dataset(data_path, batch_size=256):
    """
    Load the dataset, split it into training, validation, and test sets, and create data loaders.

    Args:
        data_path (str): Path to the dataset file.
        batch_size (int, optional): Batch size for data loaders. Defaults to 256.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        valid_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    print("Loading dataset.")

    # Load the custom dataset using BindingAffinityDataset class
    dataset = BindingAffinityDataset(data_path)

    # Split dataset into training, validation, and test sets
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, rest_of_dataset = random_split(dataset, [train_size, valid_size])
    validation_dataset, test_dataset = random_split(rest_of_dataset, [int(0.5 * len(rest_of_dataset)),
                                                                      len(rest_of_dataset) - int(
                                                                          0.5 * len(rest_of_dataset))])

    # Create data loaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader
