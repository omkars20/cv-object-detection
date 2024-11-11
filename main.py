# main.py
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T

from train import get_model, train_model

def get_coco_data_loader(data_dir, annotations_file, batch_size=4, shuffle=True):
    transform = T.Compose([T.ToTensor()])
    dataset = CocoDetection(root=data_dir, annFile=annotations_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    return data_loader

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to data
    train_data_dir = 'anottated-data/train'
    valid_data_dir = 'anottated-data/valid'
    annotations_file_train = 'anottated-data/train/_annotations.coco.json'
    annotations_file_valid = 'anottated-data/valid/_annotations.coco.json'

    # Load data
    train_loader = get_coco_data_loader(train_data_dir, annotations_file_train, batch_size=4, shuffle=True)
    valid_loader = get_coco_data_loader(valid_data_dir, annotations_file_valid, batch_size=4, shuffle=False)

    # Initialize model
    model = get_model()
    model.to(device)

    # Train the model
    train_model(model, train_loader, valid_loader, device, num_epochs=5)

if __name__ == "__main__":
    main()


