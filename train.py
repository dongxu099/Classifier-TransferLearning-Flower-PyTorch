'''
Basic usage: python train.py data_directory
'''
import argparse
import torch
from utils import preprocess
from model import build_model, train_model

# Get command line input
parser = argparse.ArgumentParser()
parser.add_argument('data_directory', action='store',
                    default = 'flowers',
                    help='Set directory to load training and validating data; default = "flowers"')
parser.add_argument('--save_dir', action='store',
                    default = '.',
                    dest='save_dir',
                    help='Set directory to save checkpoints; default = "assets"')
parser.add_argument('--arch', action='store',
                    default = 'densenet121',
                    dest='arch',
                    help='Choose architecture; default = "vgg13"')
parser.add_argument('--learning_rate', action='store',
                    default = 0.0001,
                    dest='learning_rate',
                    help='Choose architecture learning rate; default =  0.0001')
parser.add_argument('--hidden_units', action='store',
                    default = 512,
                    dest='hidden_units',
                    help='Choose architecture hidden units; default = 512')
parser.add_argument('--epochs', action='store',
                    default = 1,
                    dest='epochs',
                    help='Choose architecture number of epochs; default = 1')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for training, set a switch to true; default = False')
parse_results = parser.parse_args()
# print('data_directory     = {!r}'.format(parse_results.data_directory))
data_dir = parse_results.data_directory
save_dir = parse_results.save_dir
arch = parse_results.arch
learning_rate = float(parse_results.learning_rate)
hidden_units = int(parse_results.hidden_units)
epochs = int(parse_results.epochs)
gpu = parse_results.gpu

# Load and preprocess data
image_datasets, train_loader, valid_loader, test_loader = preprocess(data_dir)

# Building and training the classifier
model_init = build_model(arch, hidden_units)
model, optimizer, criterion = train_model(model_init, train_loader, valid_loader, learning_rate, epochs, gpu)

# Save the checkpoint 
model.to('cpu')
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'model': model,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'criterion': criterion,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx}
torch.save(checkpoint, save_dir + '/checkpoint.pth')
if save_dir == ".":
    save_dir_name = "current folder"
else:
    save_dir_name = save_dir + " folder"
print(f'Checkpoint saved to {save_dir_name}.')