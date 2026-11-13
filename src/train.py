import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from models import (
    CNN_Mode_Proposed,
    CNN_Model_Test2,
    CNN_Model_Test3
)


# Training configuration
batch_size_model1 = 128
batch_size_model2 = 128
batch_size_model3 = 128

epochs = 30
learning_rate = 0.0001


# Accuracy function
def accuracy_fn(pred, true):
    pred = torch.argmax(pred, dim=1)
    acc = (true == pred).float().mean().item()
    return acc


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_Mode_Proposed().to(device)

# loss function
loss_fn = nn.CrossEntropyLoss()
# optimzier
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

top1_acc_train = []
top1_acc_val = []
loss_avg_train = []
loss_avg_val = []

for epoch in range(1, epochs + 1):
    sum_train_acc_epoch = 0
    sum_val_acc_epoch = 0
    sum_train_loss_epoch = 0
    sum_val_loss_epoch = 0

    model.train()
    mode = "train"

    for batch_idx, (images, labels) in enumerate(train_loader,1):
        images = images.to(device)

        unique_classes = list(set(labels))
      
        labels = labels.to(device)
        optimizer.zero_grad()    # prevent accumulate the gradients

        labels_pred = model(images.float())
        loss = loss_fn(labels_pred, labels)

        loss.backward()
        optimizer.step()   # update the weights

        acc1 = accuracy_fn(labels_pred, labels)
        sum_train_acc_epoch += acc1
        sum_train_loss_epoch += loss.detach().item()    # make independent from graph
        if batch_idx % max(1, int(round(len(train_loader)/6))) == 0:   # set how many times show the acc in each epoch
            print(f"At epoch {epoch}, till batch_index {batch_idx}: average accuracy for train ------>  {(sum_train_acc_epoch/batch_idx)*100}")
            print(f"At epoch {epoch}, till batch_index {batch_idx}: average loss for train ---------->  {sum_train_loss_epoch/batch_idx}")
    top1_acc_train.append(sum_train_acc_epoch / batch_idx)
    loss_avg_train.append(sum_train_loss_epoch / batch_idx)


    model.eval()
    mode = "val"

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader,1):
            images = images.to(device)

            unique_classes = list(set(labels))

            labels = labels.to(device)
            labels_pred = model(images.float())
            loss = loss_fn(labels_pred, labels)
            acc1 = accuracy_fn(labels_pred, labels)
            sum_val_acc_epoch += acc1
            sum_val_loss_epoch += loss.detach().item()
            if batch_idx % max(1, int(round(len(val_loader)/6))) == 0:
                print(f"At epoch {epoch}, till batch_index {batch_idx}: average accuracy for valid ------>  {(sum_val_acc_epoch/batch_idx)*100}")
                print(f"At epoch {epoch}, till batch_index {batch_idx}: average loss for valid ---------->  {sum_val_loss_epoch/batch_idx}")

        top1_acc_val.append(sum_val_acc_epoch / batch_idx)
        loss_avg_val.append(sum_val_loss_epoch / batch_idx)


def accloss_plots(train_loss1, test_loss1, train_acc1, test_acc1):
    num_epochs = len(train_loss1)
    plt.figure(figsize=(13, 10))

    # Plot training loss for all sets
    plt.subplot(121)
    plt.plot(train_loss1, '-.', label='Train Set CNN')
    plt.xlabel('Epochs', fontsize=10, labelpad=8)
    plt.title('Loss', fontsize=25, pad=15)
    plt.grid()
    plt.legend()
    sns.despine()

    # Plot testing loss for all sets
    #plt.subplot(223)
    plt.plot(test_loss1, '-.', label='Validation Set CNN')
    plt.xlabel('Epochs', fontsize=10, labelpad=8)
    plt.title('Loss', fontsize=25, pad=15)
    plt.grid()
    plt.legend()
    sns.despine()

    # Plot training accuracy for all sets
    plt.subplot(122)
    plt.plot(train_acc1, '-.', label='Train Set CNN')
    plt.xlabel('Epochs', fontsize=10, labelpad=8)
    plt.title('Accuracy', fontsize=25, pad=15)
    plt.tight_layout(pad=3)
    plt.grid()
    plt.legend()
    sns.despine()

    # Plot testing accuracy for all sets
    #plt.subplot(224)
    plt.plot(test_acc1, '-.', label='Validation Set CNN')
    plt.xlabel('Epochs', fontsize=10, labelpad=8)
    plt.title('Accuracy', fontsize=25, pad=15)
    plt.tight_layout(pad=3)
    plt.grid()
    plt.legend()
    sns.despine()

    plt.show()

accloss_plots(loss_avg_train, loss_avg_val, top1_acc_train, top1_acc_val)




###############################################################################
# train the model test2
#batch_size_model2 = 128
epochs_model2 = 30
learning_rate_model2 = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_test2 = CNN_Model_Test2().to(device)

# loss function
loss_fn = nn.CrossEntropyLoss()
# optimzier
optimizer_model_test2 = optim.Adam(model_test2.parameters(), lr=learning_rate_model2)


acc_train_model_test2 = []
acc_val_model_test2 = []
loss_avg_train_model_test2 = []
loss_avg_val_model_test2 = []


for epoch in range(1, epochs_model2 + 1):
    sum_train_acc_epoch = 0
    sum_val_acc_epoch = 0
    sum_train_loss_epoch = 0
    sum_val_loss_epoch = 0

    model_test2.train()
    mode = "train"

    for batch_idx, (images, labels) in enumerate(train_loader,1):
        images = images.to(device)

        unique_classes = list(set(labels))
        optimizer_model_test2.zero_grad()    # prevent accumulate the gradients

        labels = labels.to(device)
        labels_pred = model_test2(images.float())
        loss = loss_fn(labels_pred, labels.squeeze())

        loss.backward()
        optimizer_model_test2.step()   # update the weights

        acc1 = accuracy_fn(labels_pred, labels.squeeze())
        sum_train_acc_epoch += acc1
        sum_train_loss_epoch += loss.detach().item()    # make independent from graph
        if batch_idx % max(1, int(round(len(train_loader)/6))) == 0:   # set how many times show the acc in each epoch
            print(f"At epoch {epoch}, till batch_index {batch_idx}: average accuracy for train ------>  {(sum_train_acc_epoch/batch_idx)*100}")
            print(f"At epoch {epoch}, till batch_index {batch_idx}: average loss for train ---------->  {sum_train_loss_epoch/batch_idx}")
    acc_train_model_test2.append(sum_train_acc_epoch / batch_idx)
    loss_avg_train_model_test2.append(sum_train_loss_epoch / batch_idx)

    model_test2.eval()
    mode = "val"

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader,1):
            images = images.to(device)

            unique_classes = list(set(labels))
            labels = labels.to(device)
            labels_pred = model_test2(images.float())
            loss = loss_fn(labels_pred, labels.squeeze())
            acc1 = accuracy_fn(labels_pred, labels.squeeze())
            sum_val_acc_epoch += acc1
            sum_val_loss_epoch += loss.detach().item()
            if batch_idx % max(1, int(round(len(val_loader)/6))) == 0:
                print(f"At epoch {epoch}, till batch_index {batch_idx}: average accuracy for valid ------>  {(sum_val_acc_epoch/batch_idx)*100}")
                print(f"At epoch {epoch}, till batch_index {batch_idx}: average loss for valid ---------->  {sum_val_loss_epoch/batch_idx}")

        acc_val_model_test2.append(sum_val_acc_epoch / batch_idx)
        loss_avg_val_model_test2.append(sum_val_loss_epoch / len(val_loader))

accloss_plots(loss_avg_train_model_test2, loss_avg_val_model_test2, acc_train_model_test2, acc_val_model_test2)



########################################################################################
#train model test3
#batch_size_model3 = 128
epochs_model3 = 30
learning_rate_model3 = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_test3 = CNN_Model_Test3().to(device)

# loss function
loss_fn = nn.CrossEntropyLoss()
# optimzier
optimizer_model_test3 = optim.Adam(model_test3.parameters(), lr=learning_rate_model3)


acc_train_model_test3 = []
acc_val_model_test3 = []
loss_avg_train_model_test3 = []
loss_avg_val_model_test3 = []


for epoch in range(1, epochs_model3 + 1):
    sum_train_acc_epoch = 0
    sum_val_acc_epoch = 0
    sum_train_loss_epoch = 0
    sum_val_loss_epoch = 0

    model_test3.train()
    mode = "train"

    for batch_idx, (images, labels) in enumerate(train_loader,1):
        images = images.to(device)

        unique_classes = list(set(labels))
        optimizer_model_test3.zero_grad()    # prevent accumulate the gradients

        labels = labels.to(device)
        labels_pred = model_test3(images.float())
        loss = loss_fn(labels_pred, labels.squeeze())

        loss.backward()
        optimizer_model_test3.step()   # update the weights

        acc1 = accuracy_fn(labels_pred, labels.squeeze())
        sum_train_acc_epoch += acc1

        sum_train_loss_epoch += loss.detach().item()    # make independent from graph
        if batch_idx % max(1, int(round(len(train_loader)/6))) == 0:   # set how many times show the acc in each epoch
            print(f"At epoch {epoch}, till batch_index {batch_idx}: average accuracy for train ------>  {(sum_train_acc_epoch/batch_idx)*100}")
            print(f"At epoch {epoch}, till batch_index {batch_idx}: average loss for train ---------->  {sum_train_loss_epoch/batch_idx}")
    acc_train_model_test3.append(sum_train_acc_epoch / batch_idx)
    loss_avg_train_model_test3.append(sum_train_loss_epoch / batch_idx)

    model_test3.eval()
    mode = "val"

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader,1):
            images = images.to(device)

            unique_classes = list(set(labels))
          
            labels = labels.to(device)
            labels_pred = model_test3(images.float())
            loss = loss_fn(labels_pred, labels.squeeze())
            acc1 = accuracy_fn(labels_pred, labels.squeeze())
            sum_val_acc_epoch += acc1
            sum_val_loss_epoch += loss.detach().item()
            if batch_idx % max(1, int(round(len(val_loader)/6))) == 0:
                print(f"At epoch {epoch}, till batch_index {batch_idx}: average accuracy for valid ------>  {(sum_val_acc_epoch/batch_idx)*100}")
                print(f"At epoch {epoch}, till batch_index {batch_idx}: average loss for valid ---------->  {sum_val_loss_epoch/batch_idx}")

        acc_val_model_test3.append(sum_val_acc_epoch / batch_idx)
        loss_avg_val_model_test3.append(sum_val_loss_epoch / len(val_loader))

#plot the accuracy and loss curve
accloss_plots(loss_avg_train_model_test3, loss_avg_val_model_test3, acc_train_model_test3, acc_val_model_test3)