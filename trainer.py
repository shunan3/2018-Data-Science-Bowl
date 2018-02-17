import argparse
import os
import shutil
import time


from pathlib import Path
from PIL import Image
from skimage import io
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import models
from sklearn.model_selection import train_test_split

model_names = sorted(name for name in models.__dict__
    if  not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='2018 Data Science Bowl Competition')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vanilla',
                    choices=model_names)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--resume', default=False, type=bool, metavar='N',
                    help='resume from checkpoint (default: False)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec = 0

def main():
    global args, best_prec
    args = parser.parse_args()

    # create model
    model = models.__dict__[args.arch]()
    model.cuda() 

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume == True:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
    	start_epoch = 0    	

    cudnn.benchmark = True

    # Data loading code
    train_data = load_data('./data/stage1_train/', has_mask=True)
    valid_data = load_data('./data/stage1_valid/', has_mask=True)
    test_data = load_data('./data/stage1_test/', has_mask=False)
    
    
    # Data processing    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    s_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    normalize,
    ])
    
    t_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    ])
    
    train_dataset = image_processing(train_data,s_trans,t_trans)
    val_dataset = image_processing(valid_data,s_trans,t_trans)
	
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
	val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(start_epoch, args.epochs):        
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec = validate(val_loader, model, criterion)

        # remember best prec and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))

        top.update(prec, input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top.val:.3f} ({top.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top=top))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top.update(prec, input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top.val:.3f} ({top.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top=top))

    print(' * Prec {top.avg:.3f} '.format(top=top))

    return top.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def computeIoU(y_pred_batch, y_true_batch, threshold):
    return np.mean(np.asarray([IoU(y_pred_batch[i], y_true_batch[i], threshold) for i in range(len(y_true_batch))])) 

def IoU(y_pred, y_true, threshold):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    y_pred_tran = y_pred * (y_pred > threshold)
    TP = np.sum((y_pred_tran==y_true)*(y_true>0))
    FNFP = np.sum(y_pred_tran!=y_true)
    return 1.0*TP/(TP+FNFP)

def accuracy(y_pred_batch, y_true_batch):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    total = 0.0
    for threshold in thresholds:
        total += computeIoU(y_pred_batch, y_true_batch, threshold)
    return total*1.0 *10


def load_data(file_path, has_mask=True):
    file_path = Path(file_path)
    files = sorted(list(Path(file_path).iterdir()))
    datas = []

    for file in files:
        item = {}
        imgs = []
        for image in (file/'images').iterdir():
            img = io.imread(image)
            imgs.append(img)
        assert len(imgs)==1
        if img.shape[2]>3:
            assert(img[:,:,3]!=255).sum()==0
        img = img[:,:,:3]

        if has_mask:
            mask_files = list((file/'masks').iterdir())
            masks = None
            for ii,mask in enumerate(mask_files):
                mask = io.imread(mask)
                assert (mask[(mask!=0)]==255).all()
                if masks is None:
                    H,W = mask.shape
                    masks = np.zeros((len(mask_files),H,W))
                masks[ii] = mask
            tmp_mask = masks.sum(0)
            assert (tmp_mask[tmp_mask!=0] == 255).all()
            for ii,mask in enumerate(masks):
                masks[ii] = mask/255 * (ii+1)
            mask = masks.sum(0)
            item['mask'] = torch.from_numpy(mask)
        item['name'] = str(file).split('/')[-1]
        item['img'] = torch.from_numpy(img)
        datas.append(item)
    return datas

class image_processing():
    def __init__(self,data,source_transform,target_transform):
        self.datas = data
        self.s_transform = source_transform
        self.t_transform = target_transform
    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        mask = data['mask'][:,:,None].byte().numpy()
        img = self.s_transform(img)
        mask = self.t_transform(mask)
        mask = mask > 0
        return img, mask
    def __len__(self):
        return len(self.datas)
       
if __name__ == '__main__':
    main()
