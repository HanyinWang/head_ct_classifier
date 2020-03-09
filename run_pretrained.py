from torch.utils import data
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn
from torch import optim
from time import time
import torch.nn.functional as F
import os

cuda_device = 0

img_train = np.load('/home/hwi3319/COMP_SCI_496_SML/train/train_img.npy')
lb_train = np.load('/home/hwi3319/COMP_SCI_496_SML/train/train_lb.npy')

img_test = np.load('/home/hwi3319/COMP_SCI_496_SML/test/test_img.npy')
lb_test = np.load('/home/hwi3319/COMP_SCI_496_SML/test/test_lb.npy')

img_val = np.load('/home/hwi3319/COMP_SCI_496_SML/validate/val_img.npy')
lb_val = np.load('/home/hwi3319/COMP_SCI_496_SML/validate/val_lb.npy')

transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class Dataset(data.Dataset):
    def __init__(self, img, lb, pretrained_models):
        self.labels = lb
        self.images = img
        self.pretrained_models = pretrained_models.float().cuda(cuda_device)
        for param in self.pretrained_models.parameters():
            param.requires_grad = False

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        scan = self.images[index]
        scan = np.rollaxis(scan,2,0).reshape(28,3,224,224)
        scan = torch.from_numpy(scan).float().cuda(cuda_device)
        # for i in range(28):
        #     scan[i, :, :, :] = transform(scan[i, :, :, :])
        X = self.pretrained_models(scan).flatten()
        y = self.labels[index]

        return X, y

params = {'batch_size': 16,
          'shuffle': True}

# use VGG16 pre-trained
training_set = Dataset(img_train, lb_train, models.vgg16(pretrained=True))
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(img_val, lb_val, models.vgg16(pretrained=True))
validation_generator = data.DataLoader(validation_set, **params)

test_set = Dataset(img_test, lb_test, models.vgg16(pretrained=True))
test_generator = data.DataLoader(test_set, **params)

# fully connected layer for classification
input_size = 28000
hidden_sizes = [128, 64]
output_size = 2

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size))
                      # nn.Softmax(dim=1))

model = model.cuda(cuda_device)
model = nn.DataParallel(model)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

time0 = time()
epochs = 20
loss_valid_bingo = 0
loss_valid_list = []
loss_train_list = []
valid_acc_list = []
train_acc_list = []

for e in range(epochs):
    # switch to train mode
    model.train()
    running_loss_train = 0
    running_loss_valid = 0
    print("Epoch {} , Training Starts".format(e))
    running_loss = 0
    for images, labels in training_generator:
        images = images.cuda(cuda_device)
        labels = labels.cuda(cuda_device)
        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss_train = criterion(output, labels.long())

        # This is where the model learns by backpropagating
        loss_train.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss_train += loss_train.item()

    correct_count, all_count = 0, 0
    for images, labels in training_generator:
        images = images.cuda(cuda_device)
        labels = labels.cuda(cuda_device)
        for i in range(len(labels)):
            img = images[i].view(1, 28000)
            img = img.float()
            with torch.no_grad():
                ps = model(img)
            probab = list(ps.cpu().numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.cpu().numpy()[i]
            if (true_label == pred_label):
                correct_count += 1
            all_count += 1
    valid_acc_list.append(correct_count / all_count)
    print("\nTraining Accuracy =", (correct_count / all_count))

    print("Epoch {} , Validation Starts".format(e))

    for images, labels in validation_generator:
        images = images.cuda(cuda_device)
        labels = labels.cuda(cuda_device)
        with torch.no_grad():
            ps = model(images)
        loss_valid = criterion(ps, labels.long())
        running_loss_valid += loss_valid.item()

    correct_count, all_count = 0, 0
    for images, labels in validation_generator:
        images = images.cuda(cuda_device)
        labels = labels.cuda(cuda_device)
        for i in range(len(labels)):
            img = images[i].view(1, 28000)
            img = img.float()
            with torch.no_grad():
                ps = model(img)
            probab = list(ps.cpu().numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.cpu().numpy()[i]
            if (true_label == pred_label):
                correct_count += 1
            all_count += 1
    valid_acc_list.append(correct_count / all_count)
    print("\nValidating Accuracy =", (correct_count / all_count))

    print("Epoch {} - Training loss: {}".format(e, running_loss_train / len(training_generator)),
          "Epoch {} - Validating loss: {}".format(e, running_loss_valid / len(validation_generator)))

    loss_train_list.append(running_loss_train / len(training_generator))
    loss_valid_list.append(running_loss_valid / len(validation_generator))

    if e >= 1:
        loss_valid_chg = loss_valid_list[e - 1] - loss_valid_list[e]
        if loss_valid_chg < 1e-4:
            loss_valid_bingo += 1
        else:
            loss_valid_bingo = 0
    if loss_valid_bingo >= 3:
        break

print("\nTraining Time (in minutes) =", (time() - time0) / 60)

#test
model.eval()

outputs = np.array([]).reshape(0, 2)
labels = np.array([]).reshape(0, 2)
end = time.time()
for i, (img, label) in enumerate(test_generator):
    inputs, targets = img, label
    output = F.sigmoid(model(inputs))
    outputs = np.concatenate([outputs, output.cpu().data.numpy()])
    labels = np.concatenate([labels, targets.numpy()])

    # measure elapsed time
    batch_time = time.time() - end
    end = time.time()
    print("batch: [{}/{}], \t time:{}".format(i, len(test_generator), batch_time))
    i += 1

y_true = labels
y_score = outputs
roc=[]
for i in range(8):
    roc.append(roc_auc_score(y_true[:,i], y_score[:,i]))
print(roc)