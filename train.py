
import torch
from torchvision import models
from collections import OrderedDict
import os  # For directory creation

def build_model(arch='resnet34', hidden_layer_1_units=512, hidden_layer_2_units=256, output_size=102):
    """Builds a pre-trained model with a custom classifier."""
    if arch == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        input_size = model.fc.in_features
    elif arch == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        input_size = model.classifier[1].in_features
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define custom classifier
    classifier = torch.nn.Sequential(OrderedDict([
        ('fc1', torch.nn.Linear(input_size, hidden_layer_1_units)),
        ('relu1', torch.nn.ReLU()),
        ('fc2', torch.nn.Linear(hidden_layer_1_units, hidden_layer_2_units)),
        ('relu2', torch.nn.ReLU()),
        ('output', torch.nn.Linear(hidden_layer_2_units, output_size)),
        ('logsoftmax', torch.nn.LogSoftmax(dim=1))
    ]))

    # Replace the classifier
    if arch == 'resnet34':
        model.fc = classifier
    elif arch == 'efficientnet_v2_s':
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            classifier
        )

    return model

def save_checkpoint(model, filepath, arch, hidden_layer_1_units, hidden_layer_2_units, output_size):
    """Saves the model checkpoint."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'arch': arch,
        'hidden_layer_1_units': hidden_layer_1_units,
        'hidden_layer_2_units': hidden_layer_2_units,
        'output_size': output_size,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, filepath, weights_only=True)  # Add weights_only=True

def load_checkpoint(filepath):
    """Loads a model checkpoint."""
    checkpoint = torch.load(filepath, weights_only=True)  # Add weights_only=True
    model = build_model(
        checkpoint['arch'],
        checkpoint['hidden_layer_1_units'],
        checkpoint['hidden_layer_2_units'],
        checkpoint['output_size']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
