from torch.utils import data
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn
from torch import optim
from time import time
import os
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd
import shutil
# import torch.nn.functional as F

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0")

img_train = np.load('/home/hwi3319/COMP_SCI_496_SML/train/train_img.npy')
lb_train = np.load('/home/hwi3319/COMP_SCI_496_SML/train/train_lb.npy')

img_test = np.load('/home/hwi3319/COMP_SCI_496_SML/test/test_img.npy')
lb_test = np.load('/home/hwi3319/COMP_SCI_496_SML/test/test_lb.npy')

img_val = np.load('/home/hwi3319/COMP_SCI_496_SML/validate/val_img.npy')
lb_val = np.load('/home/hwi3319/COMP_SCI_496_SML/validate/val_lb.npy')

transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class Dataset(data.Dataset):
    def __init__(self, img, lb):
        self.labels = lb
        self.images = img


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        scan = self.images[index]
        scan = np.rollaxis(scan,2,0).reshape(28,3,224,224)
        scan = torch.from_numpy(scan).float().to(device)
        # for i in range(28):
        #     scan[i, :, :, :] = transform(scan[i, :, :, :])
        X = scan
        y = self.labels[index][0]

        return X, y


params = {'batch_size': 8,
          'shuffle': True}

# use VGG16 pre-trained
training_set = Dataset(img_train, lb_train)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(img_val, lb_val)
validation_generator = data.DataLoader(validation_set, **params)

test_set = Dataset(img_test, lb_test)
test_generator = data.DataLoader(test_set, **params)

model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 14)

model = model.to(device)
model = nn.DataParallel(model)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()


time0 = time()
epochs = 60
loss_valid_bingo = 0
loss_valid_list = []
loss_train_list = []
valid_acc_list = []
train_acc_list = []

for e in range(epochs):
    running_loss_train = 0
    running_loss_valid = 0
    pred_eval_train = np.zeros(14)
    print("Epoch {} , Training Starts".format(e))
    for images, labels in training_generator:
        images = images.to(device)
        labels = labels.to(device)
        # Training pass
        optimizer.zero_grad()

        output_metric = torch.empty([images.shape[0],28,14])
        for i in range(28):
            output_metric[:,i,:] = model(images[:,i,:,:,:])

        output = torch.empty([images.shape[0],14])
        for i in range(images.shape[0]):
            output[i,:] = torch.mean(output_metric[i,:,:], dim=0)

        output = output.to(device)
        loss_train = criterion(output, labels.float())

        # This is where the model learns by backpropagating
        loss_train.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss_train += loss_train.item()

        pred_label = torch.sigmoid(output).cpu().data.numpy() > 0.5
        pred_eval_train += np.sum((pred_label == labels.cpu().numpy()), axis = 0)

    print("\nTraining Accuracy =", pred_eval_train/len(img_train))

    print("Epoch {} , Validation Starts".format(e))
    pred_eval_val = np.zeros(14)
    for images, labels in validation_generator:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output_metric = torch.empty([images.shape[0], 28, 14])
            for i in range(28):
                output_metric[:, i, :] = model(images[:, i, :, :, :])

            output = torch.empty([images.shape[0], 14])
            for i in range(images.shape[0]):
                output[i, :] = torch.mean(output_metric[i, :, :], dim=0)
            output = output.to(device)
        loss_valid = criterion(output, labels.float())
        running_loss_valid += loss_valid.item()

        pred_label = torch.sigmoid(output).cpu().data.numpy() > 0.5
        pred_eval_val += np.sum((pred_label == labels.cpu().numpy()), axis = 0)
    print("\nValidation Accuracy =", pred_eval_val / len(img_val))

    print("Epoch {} - Training loss: {}".format(e, running_loss_train/len(training_generator)), "Epoch {} - Validating loss: {}".format(e, running_loss_valid/len(validation_generator)))
print("\nTraining Time (in minutes) =", (time() - time0) / 60)

#test

outputs = np.array([]).reshape(0,14)
true_labels = np.array([]).reshape(0,14)
pred_labels = np.array([]).reshape(0,14)
for images, labels in test_generator:
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        output_metric = torch.empty([images.shape[0], 28, 14])
        for i in range(28):
            output_metric[:, i, :] = model(images[:, i, :, :, :])

        output = torch.empty([images.shape[0], 14])
        for i in range(images.shape[0]):
            output[i, :] = torch.mean(output_metric[i, :, :], dim=0)
        output = output.to(device)
    proba = torch.sigmoid(output).cpu().data.numpy()
    pred_label = torch.sigmoid(output).cpu().data.numpy() > 0.5

    outputs = np.concatenate([outputs, proba])
    pred_labels = np.concatenate([pred_labels, pred_label])
    true_labels = np.concatenate([true_labels, labels.cpu().numpy()])


y_true = true_labels
y_pred = pred_labels
y_score = outputs

roc = []
sensitivity = []
specificity = []
accuracy = []
for i in range(14):
    try:
        roc.append(roc_auc_score(y_true[:,i], y_score[:,i]))
    except:
        roc.append(0.0)

    if all(y_true[:,i] == y_pred[:,i]):
        sensitivity.append(1)
        specificity.append(1)
    else:
        tn, fp, fn, tp=confusion_matrix(y_true[:,i], y_pred[:,i]).ravel()
        sensitivity.append(tp/(tp+fn))
        specificity.append(tn/(tn+fp))
    accuracy.append((np.sum((y_pred[:,i] == y_true[:,i]), axis=0))/len(img_test))

print(roc, sensitivity, specificity, accuracy)
os.chdir('/home/hwi3319/COMP_SCI_496_SML')
pd.DataFrame(roc).to_csv('roc_resnet152_fine_tune.csv')
pd.DataFrame(sensitivity).to_csv('sensitivity_resnet152_fine_tune.csv')
pd.DataFrame(specificity).to_csv('specificity_resnet152_fine_tune.csv')
pd.DataFrame(accuracy).to_csv('accuracy_resnet152_fine_tune.csv')