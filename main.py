import argparse
import os
import time
import datetime
import shutil
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
from collections import OrderedDict
from utils.video_funcs import smooth_crossentropy, LabelSmoothingCrossEntropy

from dataset import TrearDataSet
from transforms import *
from opts import parser

from trear import Trear

import math
import warnings
warnings.filterwarnings("ignore")

best_prec1 = 0
flag = False
loss_check = 100

def main():
    global args, best_prec1, loss_check
    args = parser.parse_args()


    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'ntu60':
        num_class = 60
    elif args.dataset == 'ntu120':
        num_class = 120
    elif args.dataset == 'sysu':
        num_class = 12
    elif args.dataset == 'pku':
        num_class = 51
    elif args.dataset == 'thu':
        num_class = 40
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'FPHA':
        num_class = 45
    else:
        raise ValueError('Unknown dataset '+args.dataset)
    

    model = Trear(num_class, args.num_segments, base_model=args.arch, dropout=args.dropout)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    train_augmentation = model.get_augmentation()
    
    print('Current GPUs id:', args.gpus)
    torch.cuda.set_device('cuda:{}'.format(args.gpus[0]))
    model = torch.nn.DataParallel(model.cuda(), device_ids=args.gpus)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True  

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)
    data_length = 1

    train_loader = torch.utils.data.DataLoader(
        TrearDataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   image_tmpl=args.flow_prefix+"img_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=1)
    val_loader = torch.utils.data.DataLoader(
        TrearDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   image_tmpl=args.flow_prefix+"img_{:05d}.jpg",
                   random_shift=False,      # in 'test' or 'val' mode must be False
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=1, shuffle=False, drop_last=True,
        num_workers=len(args.gpus) * 2 if len(args.gpus) >= 2 else 4, pin_memory=True)


    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        epoch_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("**" * 20)
        print('epoch_start_time: ' + epoch_start_time)
        
        prec1 = train(train_loader, model, criterion, optimizer, epoch)
        epoch_end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("**" * 20)
        print('epoch_end_time: ' + epoch_end_time)
        print("**" * 20)
        
        if epoch < args.epochs - 1:
            prec1 = validate(val_loader, model, criterion)  # use val set
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
            
        save_name = './train_result/' + '_'.join(('trear', str(epoch), 'model.pth.tar'))
        
        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': prec1,
        }, save_name)
            
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()
    epoch_accuracy=0
    data_size=0

    # switch to train mode

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        output = model(input_var)     
        loss = criterion(output, target_var)       
        
        # measure accuracy and record loss
        prec = accuracy(input, output.data, target)
        losses.update(loss.item(), input.size(0))      
        top.update(prec, input.size(0))
        
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure batch accuracy
        _,predicted=torch.max(output, dim=1)
        acc=(predicted==target).sum().item()
        epoch_accuracy+=acc
        data_size+=len(input)


        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.avg:.3f}\t'   
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'              
                   'Fusion_Prec {top5.avg:.3f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=losses, top5=top, lr=optimizer.param_groups[-1]['lr'])))
    
    epoch_accuracy/=data_size
    print("Avg Train Acc:",round(epoch_accuracy, 4))
    return top.avg  

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    top_fusion = AverageMeter()
    loss_fusion = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)

            # measure accuracy and record loss
            loss = criterion(output, target_var)
            loss_fusion.update(loss, input.size(0))
            prec = accuracy(input, output.data, target)
            top_fusion.update(prec, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        print('===========', 'Val Results', '===========')
        print(('Accuracy {top1_fusion.avg:.3f} Loss {loss_fusion.avg:.5f}'.format(top1_fusion=top_fusion, loss_fusion=loss_fusion)),)

        return top_fusion.avg

def save_checkpoint(state, is_best):
    if is_best:
        best_name = './train_result/' + '_'.join(('trear', 'best.pth.tar'))
        torch.save(state,  best_name)

class AverageMeter(object):
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

def adjust_learning_rate(optimizer, epoch, lr_steps):
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    optimizer.param_groups[-1]['lr'] = args.lr * decay

def accuracy(input, output, target):
    _,predicted=torch.max(output, dim=1)
    acc=(predicted==target).sum().item()
    acc/=len(input)

    return acc

if __name__ == '__main__':
    main()