import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

def load_cifar10_data(val_ratio=0.1, seed=42):
    # Create a generator and set the random seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Define transformations for the training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load the CIFAR-10 training and test datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    # Split the training set into training and validation sets
    train_size = int((1 - val_ratio) * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size], generator=generator)
    
    # Define the classes in the CIFAR-10 dataset
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, valset, testset, classes

def split_retain_forget(trainset, forget_size: int|float = 100, seed=42):
    # Create a generator and set the random seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    
    # Split the training set into a retain set and a forget set
    if forget_size < 1:
        forget_size = int(forget_size * len(trainset))
    retain_size = len(trainset) - forget_size
    
    retainset, forgetset = random_split(trainset, [retain_size, forget_size], generator=generator)

    return retainset, forgetset

def to_dataloader(dataset, batch_size=64, shuffle=True, num_workers=2):
    # Create a DataLoader for the given dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader



# Example usage: iterate through the training data
if __name__ == "__main__":
    trainloader, valloader, testloader, forgetloader, classes = load_cifar10_data(forget_split=0.8)
    for images, labels in trainloader:
        print(f'Batch of images has shape: {images.shape}')
        print(f'Batch of labels has shape: {labels.shape}')
        break  # Remove this break to iterate through the entire dataset