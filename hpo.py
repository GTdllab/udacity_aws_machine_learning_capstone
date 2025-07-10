# hpo.py
import argparse
import os
import logging
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def net(dropout_rate=0.3):
    num_classes = 5
    model = models.resnet50(pretrained=True)

    # Freeze layers
    for name, param in model.named_parameters():
        if 'layer1' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Modified classifier with tunable dropout
    num_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_inputs, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(256, num_classes)
    )
    return model

def create_data_loaders(data, batch_size):
    # Same as in train.py
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, validation_loader

def train(model, train_loader, validation_loader, epochs, patience, criterion, optimizer, device, scheduler):
    best_loss = 1e6
    
    loss_counter = 0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()
        
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0
            
            for inputs, labels in (train_loader if phase == 'train' else validation_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples += len(inputs)
            
            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    loss_counter = 0
                else:
                    loss_counter += 1
            else:
                if scheduler:
                    scheduler.step()
            
            logger.info(f"{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}, best loss: {best_loss:.4f}")
        
        if loss_counter == patience:
            logger.info("Early stopping triggered")
            break
    
    return model

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += float(loss.item() * inputs.size(0))
            running_corrects += float(torch.sum(preds == labels.data))

    total_loss = float(running_loss) / float(len(test_loader.dataset))
    total_acc = float(running_corrects) / float(len(test_loader.dataset))

    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    
    return total_loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device: {device}')
    
    # Initialize model with hyperparameters
    model = net(dropout_rate=args.dropout_rate).to(device)
    
    # Create data loaders
    train_loader, test_loader, validation_loader = create_data_loaders(args.data, args.batch_size)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5) if args.use_scheduler else None
    
    
    # Train the model
    model = train(
        model, train_loader, validation_loader, args.epochs, args.early_stop_patience,
        criterion, optimizer, device, scheduler
    )
    
    # Test and return the test loss (objective metric)
    test_loss = test(model, test_loader, criterion, device)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    
    return test_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--dropout-rate', type=float, default=0.3)
    parser.add_argument('--use-scheduler', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--early-stop-patience', type=int, default=3)
    
    # SageMaker specific arguments
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args = parser.parse_args()
    
    main(args)