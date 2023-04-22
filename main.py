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

from dataset import TSNDataSet
from transforms import *
from opts import parser

from trear import TSN

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
    

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
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
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality in ['Appearance', 'Motion']:
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=args.flow_prefix+"{}_{:05d}.jpg" if args.modality == 'Flow' else "img_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        #num_workers=len(args.gpus) * 2 if len(args.gpus) >= 2 else 4, pin_memory=True)
        num_workers=1)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=args.flow_prefix+"{}_{:05d}.jpg" if args.modality == 'Flow' else "img_{:05d}.jpg",
                   random_shift=False,      # in 'test' or 'val' mode must be False
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=len(args.gpus) * 2 if len(args.gpus) >= 2 else 4, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
        # criterion = LabelSmoothingCrossEntropy().cuda()   # github
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    #optimizer = torch.optim.SGD(policies,
    #                            args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)

    # if args.evaluate:
    #     validate(val_loader, model, criterion, 0)
    #     return

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

        #if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
        if epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))  # use val set
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

        save_name = './train_result/' + '_'.join((args.modality.lower(), str(epoch), 'model.pth.tar'))
        
        #if epoch % 10 == 0:
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
    top1_1 = AverageMeter()
    top5_1 = AverageMeter()
    top1_2 = AverageMeter()
    top5_2 = AverageMeter()
    top1_3 = AverageMeter()
    top5_3 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)
    # switch to train mode

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
       
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output_1, output_2, output_3 = model(input_var)     # shape：batch-size * class
        loss_1 = criterion(output_1, target_var)
        loss_2 = criterion(output_2, target_var)
        loss_3 = criterion(output_3, target_var)        # fusion loss

        # measure accuracy and record loss
        prec5_1 = accuracy(output_1.data, target, topk=(0, 44))    
        prec5_2 = accuracy(output_2.data, target, topk=(0, 44))
        prec5_3 = accuracy(output_3.data, target, topk=(0, 44))

        losses.update(loss_3.item(), input.size(0))       # just for recording fusion loss

        top5_1.update(prec5_1.item(), input.size(0))
        top5_2.update(prec5_2.item(), input.size(0))
        top5_3.update(prec5_3.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # loss_1.backward()
        loss_3.backward()
        '''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            #     print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
        '''
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.avg:.3f}\t'   # batch training time
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'               # batch loss (average loss until this batch)
                   'Fusion_Prec {top5.avg:.3f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    loss=losses, top5=top5_3, lr=optimizer.param_groups[-1]['lr'])))
    return top5_3.avg   # (top1_1.avg + top1_2.avg + top1_3.avg) / 2

def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    losses_2 = AverageMeter()
    losses_3 = AverageMeter()
    top1_2 = AverageMeter()
    top5_2 = AverageMeter()

    top1_3 = AverageMeter()
    top5_3 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output_1, output_2, output_3 = model(input_var)

            # measure accuracy and record loss
            loss = criterion(output_1, target_var)
            prec = accuracy(output_1.data, target, topk=(0, 44))
            top5.update(prec.item(), input.size(0))

            loss2 = criterion(output_2, target_var)
            prec_2 = accuracy(output_2.data, target, topk=(0, 44))
            top5_2.update(prec_2.item(), input.size(0))

            loss3 = criterion(output_3, target_var)
            prec_3 = accuracy(output_3.data, target, topk=(0, 44))
            top5_3.update(prec_3.item(), input.size(0))

            losses.update(loss.item(), input.size(0))
            losses_2.update(loss2.item(), input.size(0))
            losses_3.update(loss3.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('===========', ('Test Results: rgb Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}\t'
                              'depth Prec@5 {top5_2.avg:.3f} Loss {loss2.avg:.5f}\t'
                              'fus Prec@5 {top5_3.avg:.3f} Loss {loss3.avg:.5f}'
                              .format(top5=top5, loss=losses, top5_2=top5_2, loss2=losses_2,
                                      top5_3=top5_3, loss3=losses_3)),
              '============')

        return top5_3.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    # torch.save(state, filename)
    if is_best:
        #best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'best.pth.tar'))
        best_name = './train_result/' + '_'.join((args.modality.lower(), 'best.pth.tar'))
        # shutil.copyfile(filename, best_name)
        torch.save(state,  best_name)


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


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
'''
    for param_group in optimizer.param_groups:
        if param_group['name'] not in ['fusion_weight', 'fusion_bais']:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = decay * param_group['decay_mult']
        else:
            d = 0.1 ** (sum(epoch >= np.array([30, 50])))
            param_group['lr'] = d * args.lr * param_group['lr_mult']
            param_group['weight_decay'] = decay * param_group['decay_mult']
'''
            # ======= warm up =============
            # if epoch < 8:
            #     param_group['lr'] = (args.lr * math.exp(epoch / 8 - 1)) * param_group['lr_mult']
            # else:
            #     param_group['lr'] = d * args.lr * param_group['lr_mult']
            # param_group['weight_decay'] = 1e-4 * param_group['decay_mult']
            # ============================

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True) # (topk, 정렬 dimension, largest, sorted) -> index, value 반환
    pred = pred.t() # transpose
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # 행렬에서 같으면 true, 다르면 false
    res=[]
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum()        
        res.append(correct_k.mul_(100.0 / batch_size))
    return max(res)

if __name__ == '__main__':
    main()