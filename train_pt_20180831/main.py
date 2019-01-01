import random
import sys
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import PIL

import pandas as pd
import numpy as np

import transforms_extra
#import transforms_extended as transforms
import torchvision.transforms as transforms
import models

import sklearn.metrics as sklm

parser = argparse.ArgumentParser(description='PyTorch Pneumonia Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', default='model_best.pth.tar', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 15)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--mu', '--momentum', default=0.9, type=float,
                    metavar='MU', help='momentum (when used by optimizer, otherwise ignored)')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--print-freq', '-p', default=1600, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-c', '--classes', default=3, type=int,
                    metavar='N', help='number of classes (default: 4)')
parser.add_argument('--opt-method', default='Adam', type=str,
                    help='Optimizer method: SGD, Adam, YF')                    
parser.add_argument('--nesterov', default='False', type=bool,
                    help='Nesterov acceleration when available')      

train_epoch_size = 1600 #25684/16              

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        checkpoint = torch.load(args.arch)
        model = models.ret1_bn()
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> creating model:")
        model = models.ret1_bn()
        print(model)

    model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    mean = [124.92 / 255.]
    std = [63.27 / 255.]
    
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=mean, std=std)

    dataset1 = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.01, contrast=0.01),
            transforms.Resize(640),
            transforms.ToTensor(),            
            normalize,
    ]))
    
    dataset2 = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms_extra.RandomVerticalFlip(),
            transforms_extra.RandomRotation(),
            transforms.ColorJitter(brightness=0.025, contrast=0.025),
            transforms.RandomAffine(degrees=5, translate=(0.01, 0.01), shear=5, resample=PIL.Image.BICUBIC, fillcolor=0),
            transforms.RandomCrop(1024*0.9),
            transforms.Resize(640),
            transforms.ToTensor(),            
            normalize,
    ]))


    dataset_vals = [dataset1, dataset2]
    dataset_train = torch.utils.data.ConcatDataset(dataset_vals)

    # For unbalanced dataset we create a weighted sampler
    weights = make_weights_for_balanced_classes(dataset1.imgs + dataset2.imgs, len(dataset1.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle = False,
        sampler = sampler,
        num_workers=args.workers,
        pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(640),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    if args.opt_method == "SGD":
        print("using SGD")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mu, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.opt_method == "Adam":
        print("using Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt_method == "YF":
        print("using YF")
        from yellowfin import YFOptimizer
        optimizer = YFOptimizer(model.parameters(), lr=args.lr, mu=args.mu, weight_decay=args.weight_decay)
    else:
        raise Exception("Optimizer not supported")


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def saveImageFromTensor(inputBatchTensor):
    trans = transforms.ToPILImage()
    for i in range(inputBatchTensor.size(0)):
        tensor = inputBatchTensor[i]
        img = trans(tensor)
        img.save("batch-" + str(i) + ".jpg")
        
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    y = np.ndarray((0), dtype='int64')
    t = np.ndarray((0), dtype='int64')
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #transforms.functional.to_pil_image(input[0]).save(str(int(random.uniform(0,1000000))) + ".png")
        
        #target = target.cuda(async=True)

        # image has only one channel (BW)
        input = input[:,0,:,:].unsqueeze(1)

        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # measure qwk and record loss
        losses.update(loss.data, input.size(0))
        top1.update(loss.data, input.size(0))
        y = np.append(y, np.argmax(output.data.cpu().numpy(), axis=1))
        t = np.append(t, target.cpu().numpy())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'''Epoch: [{epoch}][{i}/{len(train_loader)}]\t \
                  Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t \
                  Data {data_time.val:.3f} ({data_time.avg:.3f})\t \
                  Loss {losses.val:.4f} ({losses.avg:.4f})\t \
                  QWK {top1.val:.3f} ({top1.avg:.3f})'''
            print("TRAIN Classification report:\n{}".format(sklm.classification_report(t,y)))

        if i > train_epoch_size:
            break
    #saveImageFromTensor(input)
    #qwk_train = qwk.quadratic_weighted_kappa(y, t, 0, 4)
    #print('Epoch: [{}]  * Train QWK {}'.format(epoch, qwk_train))


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    y = np.ndarray((0), dtype='int64')
    t = np.ndarray((0), dtype='int64')
    for i, (input, target) in enumerate(val_loader):
        # only one channel BW
        input = input[:,0,:,:].unsqueeze(1)
        #target = target.cuda(async=True)
        with torch.no_grad(): # new volatile way
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        y = np.append(y, np.argmax(output.data.cpu().numpy(), axis=1))
        t = np.append(t, target.cpu().numpy())
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))

    #qwk_val = qwk.quadratic_weighted_kappa(y, t, 0, 4)
    cnf_matrix = sklm.confusion_matrix(y, t)
    np.set_printoptions(precision=2)

    print("Confusion matrix VALIDATION:")
    print_cm(cnf_matrix, labels=['0','1','2'])
    positive_class = 2 # check that melanoma is class 0
    sensitivity, specificity = calculate_sensitivity_specificity(cnf_matrix, positive_class)
    #precision, recall, f1, support = sklm.precision_recall_fscore_support(t, y, average='weighted')
    #print("VAL EP:{} i:{} PRE:{:0.3f} REC:{:0.3f} F1:{:0.3f} SUP:{:0.3f}".format(epoch, i, precision, recall, f1, support))
    #print("VAL EP:{} i:{} PRE:{} REC:{} F1:{} SUP:{}".format(epoch, i, precision, recall, f1, support))
    #print("VALIDATION Classification report:\n{}".format(sklm.classification_report(t,y)))
    print('Epoch: [{}]  * Val loss {}'.format(epoch, loss))
    print("Positive class: class {} Sensitivity={:0.3f} Specificity={:0.3f} multiply={:0.3f}".format(positive_class, sensitivity, specificity, sensitivity*specificity))
    return sensitivity*specificity


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("SAVING BEST")
        shutil.copyfile(filename, 'model_best.pth.tar')


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
        
# function for balancing the sampling of the classes
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def calculate_sensitivity_specificity(confusion, positive_class):
    pc = positive_class
    TP = confusion[pc][pc]
    FN = np.sum(confusion[:, pc]) - TP
    FP = np.sum(confusion[pc, :]) - TP
    TN = np.sum(confusion) - (TP + FN + FP)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    return sensitivity, specificity

# Pretty print of confusion matrix
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, file=sys.stdout):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ", file=file)
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ", file=file)
    print(file=file)
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ", file=file)
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ", file=file)
        print(file=file)


if __name__ == '__main__':
    main()

