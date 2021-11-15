import os
import time
import copy
import argparse
import numpy as np 
from collections import defaultdict
from PIL import Image as image
from tqdm import tqdm
from datetime import datetime
import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
from torchvision.models import vgg16
import matplotlib.pyplot as plt 


# def train_model(model, criterion, optimizer, scheduler, num_epochs:int):
#     """train the model

#     Parameters
#     ----------
#     model : torchvision.models
#         set target model
#     criterion : torch.nn.modules.loss
#         set target loss
#     optimizer : torch.optim
#         set target optimizer
#     scheduler : torch.optim.lr_scheduler
#         set scheduler
#     num_epochs : int
#         set epochs

#     Returns
#     -------
#     torchvision.models
#         trained model
#     """
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#             if phase == 'train':
#                 scheduler.step()

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))

#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         print()

#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))

#     # load best model weights
#     model.load_state_dict(best_model_wts)
    
#     return model

def VGG11():
    unfreeze_layer = ["features", 
                            "classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias",
                            "classifier.6.weight", "classifier.6.bias"
    ]
    model = vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=9)

    for name, param in model.named_parameters():
        if name in unfreeze_layer:
            param.requires_grad = True
        else:
            param.requires_grad = False

    
    return model


def train_model(model, dataloader_dict, criterion, optimizer, device):
    model = model.train()
    
    epoch_loss = 0.0
    correct_prediction = 0
    global epoch_accuracy
    for images, labels in tqdm(dataloader_dict):
        
        images = images.to(device)
        labels = labels.to(device)
        
             
        # Forward
        outputs = model(images)
        # Calculate loss
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        epoch_loss += loss.item()*images.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        correct_prediction += torch.sum(preds == labels.data)
        
        epoch_loss = epoch_loss / len(dataloader_dict.dataset)
        epoch_accuracy = correct_prediction.double() / len(dataloader_dict.dataset)
        
    return epoch_accuracy, epoch_loss

def evaluate_epoch(model, dataloader_dict, criterion, optimizer, device):
    model.eval()
    
    epoch_loss = 0.0
    correct_prediction = 0
    global epoch_accuracy
    with torch.no_grad():
        for images, labels in tqdm(dataloader_dict):
            # Load images, labels to device
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            
            epoch_loss += loss.item()*images.size(0)
            correct_prediction += torch.sum(preds == labels.data)
            
            epoch_loss = epoch_loss / len(dataloader_dict.dataset)
            epoch_accuracy = correct_prediction.double() / len(dataloader_dict.dataset)
            
        return epoch_accuracy, epoch_loss


def Plot_Loss_Save(history,save_path, today):
    # Plot results
    plt.title('Loss')
    plt.plot(history['train_loss'], color='blue', label='Train loss')
    plt.plot(history['val_loss'], color='red', label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'Loss_{today}.jpg'))
    plt.show()

def Plot_ACC_Save(history,save_path, today):
    plt.title('Accuracy')
    plt.plot(history['train_acc'], 'bo--', label='Train Acc')
    plt.plot(history['val_acc'], color='red', label='Test Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'Acc_{today}.jpg'))
    plt.show()

def ResNet18():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)
    return model 

if __name__ == "__main__":
    parser = argparse.ArgumentParser('train', add_help=False)
    parser.add_argument('--data_dir', default='.', type=str)
    parser.add_argument('--save_dir', default='checkpoint', type=str)
    parser.add_argument('--save_result', default='result', type=str)
    parser.add_argument('--save_name', default='base', type=str)
    parser.add_argument('--epochs', default=100_000_000_000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=1e-7, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--step_size', default=7, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    args = parser.parse_args()
    
    data_dir = args.data_dir

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_result, exist_ok=True)

    data_transforms = {'train': transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ]),
                    'val': transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ]),
                    }

    image_datasets = datasets.ImageFolder(data_dir)
    dataset_sizes = len(image_datasets)
    train_datasets = torch.utils.data.Subset(image_datasets, 
                                            range(dataset_sizes // 10 * 9)
                                            )
    val_datasets = torch.utils.data.Subset(image_datasets, 
                                        range(dataset_sizes // 10 * 9, dataset_sizes)
                                        )
    train_datasets.dataset.transform = data_transforms['train']
    val_datasets.dataset.transform = data_transforms['val']

    dataloaders = {'train': torch.utils.data.DataLoader(train_datasets, 
                                                        batch_size=args.batch_size,
                                                        shuffle=True, 
                                                        num_workers=args.num_workers
                                                        ),
                'val': torch.utils.data.DataLoader(val_datasets, 
                                                batch_size=args.batch_size,
                                                shuffle=True, 
                                                num_workers=args.num_workers
                                                )
                }

    dataset_sizes = {'train': len(train_datasets),
                    'val': len(val_datasets)
                    }
    class_names = image_datasets.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set model
    model = ResNet18()
    # model = VGG11()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer_ft = optim.SGD(model.parameters(), 
                            # lr=args.learning_rate, 
                            # momentum=args.momentum
                            # )
    optimizer_ft = optim.Adam(model.parameters(), 
                            lr=args.learning_rate, 
                            # momentum=args.momentum
                            )
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, 
                                                step_size=args.step_size, 
                                                gamma=args.gamma
                                                )
    # model_ft = train_model(model, 
    #                     criterion, 
    #                     optimizer_ft, 
    #                     exp_lr_scheduler,
    #                     num_epochs=args.epochs
    #               )
    history = defaultdict(list)
    best_val_acc = 0.0
    EPOCHS = args.epochs
    for epoch in range(EPOCHS):
        print(f'\nEpoch: [{epoch+1}/{EPOCHS}]')
        print('-'*30)
        
        train_acc, train_loss = train_model(model, dataloaders['train'], criterion, optimizer_ft,device)
        val_acc, val_loss = evaluate_epoch(model, dataloaders['val'], criterion, optimizer_ft, device)
        
        print('Train Loss: {:.4f}\t Train Acc: {:.4f}'.format(train_loss, train_acc))
        print('Val Loss: {:.4f}\t Val Acc: {:.4f}'.format(val_loss, val_acc))
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        # torch.save(model.state_dict(), f'{args.save_name}_{epoch}.ckpt')
    
    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    # PlotSave(history, args.save_result, today)
    # Plot_Loss_Save(history,args.save_result, today)
    # Plot_ACC_Save(history,args.save_result, today)
    print('[INFO] end...')
    # file_name = f'{args.save_name}_{args.epochs}.ckpt'
    # torch.save(model_ft.state_dict(), os.path.join(args.save_dir,file_name))
