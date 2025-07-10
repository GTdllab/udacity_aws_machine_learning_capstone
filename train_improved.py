
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import logging
import os
import time
import sys
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from smdebug import modes
from smdebug.pytorch import get_hook
import smdebug.pytorch as smd

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



def test(model, test_loader, criterion, hook, device):
    model.eval()
    hook.set_mode(modes.EVAL)
    running_loss=0.0
    running_corrects=0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            running_loss += float(loss.item() * inputs.size(0))
            running_corrects += float(torch.sum(preds == labels.data))

    total_loss = float(running_loss) / float(len(test_loader.dataset))
    total_acc = float(running_corrects) / float(len(test_loader.dataset))

    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    
    return all_preds, all_labels

def print_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    '''
    Print and optionally save confusion matrix with detailed metrics
    '''
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print confusion matrix as text
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")
    
    # Print classification report
    logger.info("Classification Report:")
    if class_names:
        report = classification_report(y_true, y_pred, target_names=class_names)
    else:
        report = classification_report(y_true, y_pred)
    logger.info(f"\n{report}")
    
    # Create and save visual confusion matrix if matplotlib is available
    try:
        plt.figure(figsize=(10, 8))
        
        if class_names:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}/confusion_matrix.png")
        
        plt.close()  # Close to free memory
        
    except Exception as e:
        logger.warning(f"Could not create visual confusion matrix: {e}")
    
    return cm

    
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

    return train_loader, test_loader, validation_loader, train_data



def train(model, train_loader, validation_loader, epochs, criterion, optimizer, hook, device, scheduler):
    best_loss = 1e6
    patience = 3
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            model.train() if phase == 'train' else model.eval()
            hook.set_mode(modes.TRAIN if phase == 'train' else modes.EVAL)

            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
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

                if running_samples % 2000 == 0:
                    acc = running_corrects / running_samples
                    print(f"Images [{running_samples}/{len(image_dataset[phase].dataset)}] Loss: {loss.item():.2f} Accuracy: {acc*100:.2f}%")

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1
            else:
                scheduler.step()

            logger.info(f"{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}, best loss: {best_loss:.4f}")

        if loss_counter == patience:
            print("Early stopping due to no improvement in validation loss.")
            break

    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device: {device}')

    model = net().to(device)

    train_loader, test_loader, validation_loader, train_dataset = create_data_loaders(args.data, args.batch_size)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)

    model = train(model, train_loader, validation_loader, args.epochs, loss_criterion, optimizer, hook, device, scheduler)

    test_preds, test_labels = test(model, test_loader, loss_criterion, hook, device)

    class_names = ["1", "2", "3", "4", "5"]
    print_confusion_matrix(test_labels, test_preds, class_names, save_path=args.output_dir)

    model.cpu()
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    
# Updated argument defaults
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=20) # increase epochs
    parser.add_argument('--lr', type=float, default=0.0001)  
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dropout-rate', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--data', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir',
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()
    print(args)

    main(args)