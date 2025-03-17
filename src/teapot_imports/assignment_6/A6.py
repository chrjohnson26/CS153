import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import matplotlib.pyplot as plt

#******Helper functions********
# These are all as seen in class (with slight modifications to adapt them)
# and do not need to be modified by you.

def tensor_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def visualize_model(model, dataloaders, class_names, device, num_images=6):
    """Visualize model predictions."""
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                print(j)
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                tensor_show(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

#******Dataloader functions********
# You are provided with a dataloader that takes in a version argument.
# You will be asked questions about the affect of the version argument,
# but don't need to modify this code.

def build_dataloader(data_dir, version = 0):
    """Construct a dataloader.
    Input:
    - data_dir: The root folder of the dataset. Data is assumed to be organized
      as shown in the transfer learning exercise in class.
    - version: A version flag (value 0 or 1) that swaps between two different
      transform specifications.
    Returns:
    - dataloaders: A torch dataloader object that provides data to the training
      procedure.
    - datadict: A formatted dictionary with useful information about the data.
    """
    if version == 0:
        data_transforms = {
            'train': transforms.Compose([
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
    elif version == 1:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
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
    else:
        error("Not a recognized dataloader version flag.")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                          data_transforms[x])
                     for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
                     for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    datadict = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "dataset_sizes": dataset_sizes,
    }

    return dataloaders, datadict

#******Model Handling********
# Here you are provided with part of a model setup function, and you will need
# to provide the rest of the implementation.

#TODO: Finish this function by adding lines at the TODO marked below.
def setup_model(datadict, device):
    """Set up a model for transfer learning according to the description
    given in the assignment notebook.
    Input:
    - datadict: A formatted dictionary with useful information about the
      data. The fields are defined in build_dataloader() above.
    - device: The torch device where we will be performing computations.
    Returns:
    - model_dict: A formatted dictionary of model information consisting of:
        --model: A network with layers set up ready to be trained for
          recognition on our custom dataset.
        --criterion: The loss function.
        --optimizer: The optimizer.
        --scheduler: The learning rate scheduler.
    """

    # we are going to be using VGG16 for our transfer learning.
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
#     print(model)

    for param in model.parameters():
        param.requires_grad = False

    # we have preset a block of the fully-connected layers to be pass-through
    # instead.
    for l in [3,4,5]:
        model.classifier[l] = nn.Identity()

    # TODO: Set the final fully connected layer to be a new fully connected
    # layer (nn.Linear in PyTorch) set up to learn our data.
    num_ftrs = model.classifier[6].in_features
    print(num_ftrs)
    class_names = datadict['class_names']
    model.classifier[6] = nn.Linear(num_ftrs, len(class_names))

    # this command sends the model to our device
    model = model.to(device)

    # here we set what type of loss we plan to use. Since this is a recognition task,
    # cross entropy is a good loss function.
    criterion = nn.CrossEntropyLoss()

    # we also need to set up an optimizer. Our standard SGD works fine.
    optimizer = optim.SGD(model.classifier[3:].parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model_dict = {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler
    }

    return model_dict

#******Training Setup********
# Here you are provided with part of training procedure, and you will need to
# complete the code.

#TODO: Modify this to enable logging of validation accuracy on a per-class basis
def train_model(model_dict, dataloaders, datadict, device, num_epochs=15):
    """
    Defines a training procedure for our transfer learning problem.
    Inputs:
    - model_dict: Our model to be trained and its accompanying information.
    - dataloaders: Our training and validation dataloader.
    - datadict: A data dictionary with information about the data being
      processed; this will be useful for data logging.
    - device: The torch device performing the evaluations.
    - num_epochs: The number of loops through our training data.
    Outputs:
    - model: The version of our trained model with the best validation score.
    - logs: A dictionary of performance logs.
    """

    since = time.time()

    # We are going to run for a set number of epochs, but that doesn't mean our final epoch is our best.
    # Keep track of which version of the model worked the best.
    best_model_wts = copy.deepcopy((model_dict["model"]).state_dict())
    best_acc = 0.0

    numval = datadict["dataset_sizes"]["val"]/datadict["num_classes"]

    # data logging
    losslog = [[],[]]
    acclog = [[],[]]
    # classlog, numpy array, contains the class specific validation accuracy for each epoch
    # length => num_epochs
    # classlog[i] => array of length num_classes (length 8)
    # classlog[i] => ratio of true positive predictions of each class
    classlog = [] # this is the list you will use to append class accuracy

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                (model_dict["model"]).train()  # Set model to training mode
            else:
                (model_dict["model"]).eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # class_true_positives: numpy array of length (num_classes) variable keeps track of number of times each class has a true positive (model predicts that class)
            # print(datadict["num_classes"])
            class_true_positives = np.zeros((datadict["num_classes"]), dtype=int)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # print(inputs, labels)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                model_dict["optimizer"].zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_dict["model"](inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = model_dict["criterion"](outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        (model_dict["optimizer"]).step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)                
                # add stat for running class corrects
                for label in range(datadict["num_classes"]):
                    class_true_positives[label] += torch.sum((preds == label) & (labels.data == label))

            if phase == 'train':
                model_dict["scheduler"].step()

            epoch_loss = running_loss / datadict["dataset_sizes"][phase]
            epoch_acc = running_corrects.double() / datadict["dataset_sizes"][phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # data logging
            if phase == 'train':
                losslog[0].append(epoch_loss)
                acclog[0].append(epoch_acc.to('cpu'))
            else:
                losslog[1].append(epoch_loss)
                acclog[1].append(epoch_acc.to('cpu'))
                # updating the class epoch accuracy in the evaluation
                # Calculating the class-specific accuracy and append to class log
                class_accuracy = class_true_positives / numval
                classlog.append(class_accuracy)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_dict["model"].state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    (model_dict["model"]).load_state_dict(best_model_wts)

    logs = {
        "train_loss": losslog[0],
        "val_loss": losslog[1],
        "train_acc": acclog[0],
        "val_acc": acclog[1],
        "class_acc": classlog
    }

    return model_dict, logs
