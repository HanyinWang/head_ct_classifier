from __future__ import print_function, division
from os.path import join
import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from PIL import Image
import time
import shutil

path = '/home/hddraid/shared_data/chest_xray8/'


class ChestXray_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_labelfile=join(path, 'Data_Entry_2017.csv'), \
                 csv_bboxfile=join(path, 'BBox_list_2017.csv'), \
                 root_dir=join(path, 'images/images'), \
                 use='train', transform=None):
        """
        Args:
            csv_labelfile (string): Path to the csv file with labels.
            csv_bboxfile (string): Path to the csv file with bbox.
            root_dir (string): Directory with all the images.
            use (string): 'train' or 'validation' or 'test'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        label_df = pd.read_csv(csv_labelfile)
        n = len(label_df)
        np.random.seed(0)
        split = np.random.permutation(range(n))
        if use == 'train':
            self.label_df = label_df.iloc[split[:int(n * 0.7)], :]
        elif use == 'validation':
            self.label_df = label_df.iloc[split[int(n * 0.7):int(n * 0.8)], :]
        elif use == 'test':
            self.label_df = label_df.iloc[split[int(n * 0.8):], :]
        elif use == "bboxtest":
            self.bbox = pd.read_csv(csv_bboxfile)
            # self.bbox['bbox']=self.bbox.iloc[:,[2,3,4,5]].apply(lambda x: tuple(x),axis=1)
            self.label_df = label_df.loc[label_df['Image Index'].isin(self.bbox['Image Index']), :]
        else:
            raise Error('use must be "train" or "validation" or "test" or "bboxtest"')

        self.root_dir = root_dir
        self.classes = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltration': 3, \
                        'Mass': 4, 'Nodule': 5, 'Pneumonia': 6, 'Pneumothorax': 7, \
                        'Consolidation': 8, 'Edema': 9, 'Emphysema': 10, 'Fibrosis': 11, \
                        'Pleural_Thickening': 12, 'Hernia': 13}
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        img_name = self.label_df.iloc[idx, 0]
        image = Image.open(join(self.root_dir, img_name)).convert('RGB')
        labels = np.zeros(len(self.classes), dtype=np.float32)
        labels[
            [self.classes[x.strip()] for x in self.label_df.iloc[idx, 1].split('|') if x.strip() in self.classes]] = 1
        # bbox = self.box_loc.loc[self.box_loc['Image Index']==img_name,['Finding Label','bbox']] \
        #        .set_index('Finding Label').to_dict()['bbox']

        sample = {'image': image, 'label': labels, 'pid': self.label_df.iloc[idx, 3], \
                  'age': self.label_df.iloc[idx, 4], 'gender': self.label_df.iloc[idx, 5], \
                  'position': self.label_df.iloc[idx, 6], 'name': img_name}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        original_model = models.alexnet(pretrained=True)
        self.features = original_model.features
        self.features.add_module('transit', nn.Sequential(nn.Conv2d(256, 1024, 3, padding=1),
                                                          nn.ReLU(inplace=True), nn.MaxPool2d(2, padding=1)))
        self.features.add_module('gpool', nn.MaxPool2d(16))
        self.classifier = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x


class MyResNet50(nn.Module):
    def __init__(self):
        super(MyResNet50, self).__init__()
        original_model = models.resnet50(pretrained=True)

        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.features.add_module('transit', nn.Sequential(nn.Conv2d(2048, 1024, 3, padding=1),
                                                          nn.ReLU(inplace=True), nn.MaxPool2d(2, padding=1)))
        self.features.add_module('gpool', nn.MaxPool2d(16))
        self.classifier = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x


class FineTuneModel(nn.Module):
    def __init__(self, net, num_classes):
        super(FineTuneModel, self).__init__()
        original_model = net(pretrained=True)
        arch = net.__name__
        if arch.startswith('alexnet'):
            self.features = original_model.features
            self.features.add_module('transit', nn.Sequential(nn.Conv2d(256, 1024, 3, padding=1),
                                                              nn.ReLU(inplace=True), nn.MaxPool2d(2, padding=1)))
            self.features.add_module('gpool', nn.MaxPool2d(16))
            self.classifier = nn.Linear(1024, num_classes)
            self.modelName = 'alexnet'
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        else:
            raise ("Finetuning not supported on this architecture yet")

    def forward(self, x):
        f = self.features(x)
        f = f.view(-1, 1024)
        y = self.classifier(f)
        return y


class W_BCEWithLogitsLoss(nn.Module):
    def __init__(self, size_average=True):
        super(W_BCEWithLogitsLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        p = int(target.sum().cpu().data.numpy())
        s = int(np.prod(target.size()))
        weight = target * (s / p - s / (s - p)) + s / (s - p) if p != 0 else target + 1
        return F.binary_cross_entropy_with_logits(input, target, weight, self.size_average)


## train the CNN
def train(train_loader, model, criterion, optimizer, epoch, iter_size=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    loss = 0.0
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = Variable(data['image'].cuda()), Variable(data['label'][:, :8].cuda())
        output = model(inputs)
        loss += criterion(output, targets)

        # compute gradient and do SGD step
        if (i + 1) % iter_size == 0:
            optimizer.zero_grad()
            loss /= iter_size
            loss.backward()
            optimizer.step()
            losses.update(loss.data[0])
            loss = 0.0
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


def train2(train_loader, model, criterion, optimizer, epoch, iter_size=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (img, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = img, label
        inputs = inputs.float()
        targets = targets.long()
        # inputs = inputs.cuda()
        # targets = targets.cuda()
        output = model(inputs)

        loss = criterion(output, targets)
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % iter_size == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i,(img, label) in enumerate(val_loader):
        inputs, targets = img, label
        inputs = inputs.float()
        targets = targets.long()
        # inputs = inputs.cuda()
        # targets = targets.cuda()
        output = model(inputs)
        loss = criterion(output, targets)
        losses.update(loss.item, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % 10 == 0:
            print('Validation: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))
    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('checkpoint', 'best'))


def test(test_loader, model):
    # switch to evaluate mode
    model.eval()

    outputs = np.array([]).reshape(0, 8)
    labels = np.array([]).reshape(0, 8)
    end = time.time()
    for i, (img, label) in enumerate(test_loader):
        inputs, targets = img, label
        output = F.sigmoid(model(inputs))
        outputs = np.concatenate([outputs, output.cpu().data.numpy()])
        labels = np.concatenate([labels, targets.numpy()])

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()
        print("batch: [{}/{}], \t time:{}".format(i, len(test_loader), batch_time))
        i += 1
    return (labels, outputs)

def single_docvec(self, x):
#         x is a list of word ids.
    size = len(x)
    x = torch.LongTensor([x]).cuda()
    x = self.features(self.embedding(x).transpose(1,2))
    x = F.max_pool1d(x, size)
    x = x.squeeze()
    return x


def forward(self, x, bowvec=None):
    # x is a batch of lists of lists with word ids.
    n_docs = [len(d) for d in x]
    rep = []
    for bat in x:
        for doc in bat:
            rep.append(self.single_docvec(doc))
    rep = torch.stack(rep)
    rep = self.dens(rep)
    rep = torch.stack([t.max(0)[0] for t in rep.split(n_docs)])
    if bowvec is not None:
        rep = torch.cat([rep, self.bowdens(bowvec)], dim=1)
    rep = self.final(rep)
    return rep

transform = transforms.Compose([transforms.ToTensor(), \
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

