import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

def load_data(data_dir):
    """Loads and preprocesses the data."""
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # Define transformations
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Define dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

    # Return class_to_idx for use in the model
    return trainloader, validloader, train_dataset.class_to_idx

def process_image(image_path):
    """Preprocesses an image for prediction."""
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img = img.crop((16, 16, 240, 240))
    np_img = np.array(img) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) / std
    np_img = np_img.transpose((2, 0, 1))
    return torch.from_numpy(np_img).type(torch.FloatTensor)
