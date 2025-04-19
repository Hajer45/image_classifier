
import argparse
import torch
from model import load_checkpoint
from utils import process_image
import json

def predict(image_path, model, topk=1, gpu=False):
    """Predicts the class probabilities for an input image."""
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    img = process_image(image_path)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        ps = torch.exp(outputs)
        top_ps, top_classes = ps.topk(topk, dim=1)

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[c.item()] for c in top_classes[0]]
    return top_ps[0].tolist(), top_classes

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained network.")
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    args = parser.parse_args()

    # Load model
    model = load_checkpoint(args.checkpoint)

    # Predict
    probs, classes = predict(args.input, model, args.top_k, args.gpu)

    # Map categories to real names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name.get(cls, cls) for cls in classes]

    # Print results
    print("Predicted Classes:", classes)
    print("Probabilities:", probs)

if __name__ == '__main__':
    main()
