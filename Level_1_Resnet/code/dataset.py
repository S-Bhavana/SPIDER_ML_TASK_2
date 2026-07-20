from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define image transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Download CIFAR-10 training dataset
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

# Download CIFAR-10 testing dataset
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# Check dataset
if __name__ == "__main__":
    print("Training Images :", len(train_dataset))
    print("Testing Images :", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("Image Batch Shape :", images.shape)
    print("Label Batch Shape :", labels.shape)