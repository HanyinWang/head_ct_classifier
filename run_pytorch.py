from torch.utils import data
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
from torch import optim
from dcnn import train2, validate, test, save_checkpoint, AverageMeter
from sklearn.metrics import roc_auc_score


img_train = np.load('/home/hwi3319/COMP_SCI_496_SML/train/train_img.npy')
lb_train = np.load('/home/hwi3319/COMP_SCI_496_SML/train/train_lb.npy')

img_test = np.load('/home/hwi3319/COMP_SCI_496_SML/test/test_img.npy')
lb_test = np.load('/home/hwi3319/COMP_SCI_496_SML/test/test_lb.npy')

img_val = np.load('/home/hwi3319/COMP_SCI_496_SML/validate/val_img.npy')
lb_val = np.load('/home/hwi3319/COMP_SCI_496_SML/validate/val_lb.npy')

class Dataset(data.Dataset):
    def __init__(self, img, lb):
        self.labels = lb
        self.images = img

    def __len__(self):
        'Denotes the total number of samples'
        return len(list(self.labels.tolist().values()))

    def __getitem__(self, index):
        'Generates one sample of data'
        X = np.rollaxis(list(self.images.tolist().values())[index],2,0)
        y = list(self.labels.tolist().values())[index]

        return X, y


params = {'batch_size': 16, 'shuffle': False}

training_set = Dataset(img_train, lb_train)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(img_val, lb_val)
validation_generator = data.DataLoader(validation_set, **params)

test_set = Dataset(img_test, lb_test)
test_generator = data.DataLoader(test_set, **params)

for tepimg, teplb in training_generator:
    print(tepimg.shape, teplb.shape)
    break

# load models
model = models.resnet18(pretrained=True)
# Freeze model weights
for param in model.parameters():
    param.requires_grad = False


# model.conv1 = torch.nn.Conv2d(84, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.fc = torch.nn.Linear(512, 2)
# x = torch.randn(16, 84, 224, 224)
# output = model(x)

# check weight freezed or not
for name, param in model.named_parameters():
    print(name, param.requires_grad)

# model = model.cuda()
# model = nn.DataParallel(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

bestloss = np.inf
for epoch in range(8):
    print('start training epoch: ' + str(epoch))
    train2(training_generator, model, criterion, optimizer, epoch, 20)
    loss_val = validate(validation_generator, model, criterion)
    isbest = loss_val < bestloss
    bestloss = min(bestloss, loss_val)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'loss_val': loss_val,
        'best_loss': bestloss,
        'optimizer': optimizer.state_dict(),
    }, isbest, filename='checkpoint_res50.pth.tar')

y_true, y_score = test(test_generator, model)
roc=[]
for i in range(8):
    roc.append(roc_auc_score(y_true[:,i], y_score[:,i]))


