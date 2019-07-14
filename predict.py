import argparse
import json
import torch
from utils import process_image
from model import predict

# Get the command line inputs
parser = argparse.ArgumentParser()
parser.add_argument('image_path', action='store',
                    default = 'flowers/test/1/image_06743.jpg',
                    help='Path to image, e.g., "flowers/test/1/image_06743.jpg"')
parser.add_argument('checkpoint', action='store',
                    default = '.',
                    help='Directory of saved checkpoints, e.g., "assets"')
parser.add_argument('--top_k', action='store',
                    default = 3,
                    dest='top_k',
                    help='Return top KK most likely classes; default = 3')
parser.add_argument('--category_names', action='store',
                    default = 'cat_to_name.json',
                    dest='category_names',
                    help='File name of the mapping of flower categories to real names; default = "cat_to_name.json"')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for inference, set a switch to true')
parse_results = parser.parse_args()
image_path = parse_results.image_path
checkpoint = parse_results.checkpoint
top_k = int(parse_results.top_k)
category_names = parse_results.category_names
gpu = parse_results.gpu

# Label mapping
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the checkpoint
filepath = checkpoint + '/checkpoint.pth'
checkpoint = torch.load(filepath, map_location='cpu')
model = checkpoint["model"]
model.load_state_dict(checkpoint['model_state_dict'])

# Image preprocessing
np_image = process_image(image_path)

# Predict class and probabilities
print(f"Predicting top {top_k} most likely flower names from image {image_path}.")
probs, classes = predict(np_image, model, top_k, gpu)
classes_name = [cat_to_name[class_i] for class_i in classes]
print("\nFlower name (probability): ")
print("---")
for i in range(len(probs)):
    print(f"{classes_name[i]} ({round(probs[i], 3)})")
print("")