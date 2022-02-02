import argparse
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from lenet_mix_ab_all_smo_full_new_2 import LeNet, LabelSmoothLoss
import numpy
from metann import Learner
import scipy
from scipy import io
import pickle
import imageio

torch.manual_seed(0)
numpy.random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch Codes')
parser.add_argument('--data_dir', default='data', type=str,
                    help='dataset dir')
parser.add_argument('--dataset', default='mnist', type=str,
                    help='dataset mnist or cifar10')
parser.add_argument('--num_iters', default=10001, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_iters', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--min-learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_max', '--adv-learning-rate', default=1, type=float,
                    help='adversarial learning rate')

parser.add_argument('--gamma', default=1, type=float,
                    help='to make them closer in latent space, less gamma, larger distance 1')
parser.add_argument('--T_adv', default=30, type=int,
                    help='max iterations: 30')
parser.add_argument('--T_sample', default=10, type=int,
                    help='MC sample times')
parser.add_argument('--advstart_iter', default=100, type=int,
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--K', default=20, type=int,
                    help='num of augmented test domains')
parser.add_argument('--T_min', default=1, type=int,
                    help='min iterations')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    help='print frequency (default: 10)')

parser.add_argument('--resume', default=None, type=str,
                    help='p ath to latest checkpoint (default: none)')

parser.add_argument('--name', default='exp', type=str,
                    help='name of experiment: ')

parser.add_argument('--mode',  default='train', type=str,
                    help='train or test')
parser.add_argument('--tensorboard',  default=True, type=bool,
                    help='Log progress to TensorBoard')

parser.add_argument('--GPU_ID', default=1, type=int,
                    help='GPU_id')
parser.add_argument('--num_updates', default=1, type=int,
                    help='number of meta-train')

best_prec1 = 0

def main():

    global args, best_prec1
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 on stackoverflow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_ID)

    exp_name = args.name

    exp_dir = os.path.join('Digits', exp_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    logfile = open(os.path.join(exp_dir, '.log'), 'w')

    # Data loading code
    kwargs = {'num_workers': 4}

    # construct train and val dataloader
    train_loader, val_loader = construct_datasets(args.data_dir, args.batch_size, kwargs)

    model = Learner(LeNet())

    parameters_theta = []
    parameters_phi = []

    for name, param in model.named_parameters():
        if "_" in name:
            parameters_phi.append(param)
        else:
            parameters_theta.append(param)

    model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['iter']
            prec = checkpoint['prec']
            print(prec)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (iter {})"
                  .format(args.resume, checkpoint['iter']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_smooth = LabelSmoothLoss().cuda()

    mse_loss = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer_theta = torch.optim.Adam(parameters_theta, args.lr)
    aug_optimizer = torch.optim.SGD(parameters_phi, args.lr_max)

    if args.mode == 'train':
        print('Training')

        train_loader_iter = iter(train_loader)

        counter_k = 0

        for t in range(args.start_iters, args.num_iters):

            losses = AverageMeter()
            top1 = AverageMeter()
            model.train()

            try:
                input, target = next(train_loader_iter)
            except:
                train_loader_iter = iter(train_loader)
                input, target = next(train_loader_iter)

            input, target = input.cuda(non_blocking=True).float(), target.cuda(non_blocking=True).long()

            if t > args.advstart_iter and (t + 1) % args.T_min == 0 and counter_k < args.K:

                for n in range(args.T_adv):
                    params = list(model.parameters())
                    output_a_1, output_a_2, output_a_4, output_a_o = model.functional(params, True, input, return_feat=True, noise_layer=False)
                    mask, output_b_1, output_b_2, output_b_4, output_b_o = model.functional(params, True, input, mix=False, return_feat=True, noise_layer=True)

                    ce_loss = criterion(output_b_o, target)

                    constraint_loss = mse_loss(output_a_4, output_b_4)

                    aug_loss = -ce_loss + args.gamma*constraint_loss

                    aug_optimizer.zero_grad()
                    aug_loss.backward()
                    aug_optimizer.step()

                counter_k += 1

            params = list(model.parameters())
            output_a_1, output_a_2, output_a_4, output_a_o = model.functional(params, True, input, return_feat=True, noise_layer=False)
            ce_loss = criterion(output_a_o, target)

            if counter_k == 0:
                optimizer_theta.zero_grad()
                ce_loss.backward()
                optimizer_theta.step()
            else:
                grads = torch.autograd.grad(ce_loss, params, create_graph=True, allow_unused=True)
                params_new = []
                for param, grad in zip(params, grads):
                    if grad is None:
                        params_new.append((param).requires_grad_())
                    else:
                        params_new.append((param - args.lr * grad).requires_grad_())

                loss_b_mc = 0
                for _ in range(args.T_sample):
                    lam, mask, output_b_n = model.functional(params_new, True, input,  mix=False, noise_layer=True)
                    lam2, mask2, output_b_n_mix = model.functional(params_new, True, input, noise_layer=True)
                    # loss_b_mc += criterion(output_b_n, target) + args.alpha*mixup_criterion(criterion, output_b_n_mix, y_a, y_b, lam)
                    # loss_b_mc += criterion(output_b_n, target) + 0.1*criterion(output_b_n_mix, target)
                    loss_b_mc += criterion(output_b_n, target) + 0.1*criterion_smooth(output_b_n_mix, target, mask2, lam2) # soft label
                    # loss_b_mc += 0*criterion(output_b_n, target) + criterion(output_b_n_mix, target)
                    # loss_b_mc += criterion_smooth(output_b_n, target, mask) + args.alpha*criterion(output_b_n_mix, target)

                loss_b = loss_b_mc/args.T_sample + ce_loss
                # loss_b = loss_b_mc/args.T_sample

                optimizer.zero_grad()
                loss_b.backward(create_graph=True)
                optimizer.step()

            # scheduler.step()
            # measure accuracy and record loss
            prec1 = accuracy(output_a_o, target, topk=(1,))[0]
            losses.update(ce_loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            if t % args.print_freq == 0:
                acc_info = 'Iter: [{0}][{1}/{2}]\t' \
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    t, t, args.num_iters,  loss=losses, top1=top1)

                print(acc_info)

                logfile.write(acc_info + '\n')

                prec1 = validate(val_loader, model)
                print("validation set acc", prec1)
                logfile.write('validation set acc: ' + str(prec1) + '\n')

                if t >= 1000:
                    validate_all(model, args.data_dir, exp_name, args.batch_size, logfile, t, kwargs)

        validate_all(model, args.data_dir, exp_name, args.batch_size, logfile, t, kwargs)
        logfile.close()

def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def construct_datasets(data_dir, batch_size, kwargs):

    def data2loader(imgs, labels):
        assert len(imgs) == len(labels)
        y = torch.stack([torch.from_numpy(np.array(i)) for i in labels])
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # pytorch CHW, tf HWC
        X = torch.stack([torch.from_numpy(imgs[i]) for i in range(len(labels))])
        X_dataset = torch.utils.data.TensorDataset(X, y)
        X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return  X_loader

    train_imgs, train_labels = load_mnist(data_dir, 'train')
    val_imgs, val_labels = load_mnist(data_dir, 'test')

    return data2loader(train_imgs, train_labels), data2loader(val_imgs, val_labels)

def validate(val_loader, model):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    params = list(model.parameters())
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True).long()
        input = input.cuda(non_blocking=True).float()
        with torch.no_grad():
            output = model.functional(params, False, input)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg

def validate_all(model, data_dir, exp_name, batch_size, logfile, t, kwargs):

    def data2loader(imgs, labels):
        assert len(imgs) == len(labels)
        y = torch.stack([torch.from_numpy(np.array(i)) for i in labels])
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # pytorch CHW, tf HWC
        X = torch.stack([torch.from_numpy(imgs[i]) for i in range(len(labels))])
        X_dataset = torch.utils.data.TensorDataset(X, y)
        X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return X_loader

    model.eval()
    params = list(model.parameters())
    accs = []
    target_domains = ['mnist', 'svhn', 'mnist_m', 'syn', 'usps']
    for td in target_domains:
        print(td)
        logfile.write(td + '\n')
        target_test_images, target_test_labels = load_test_data(data_dir, td)
        test_loader = data2loader(target_test_images, target_test_labels)

        top1 = AverageMeter()

        for i, (input, target) in enumerate(test_loader):
            target = target.cuda(non_blocking=True).long()
            input = input.cuda(non_blocking=True).float()
            with torch.no_grad():
                output = model.functional(params, False, input)

            prec1 = accuracy(output.data, target, topk=(1,))[0]
            top1.update(prec1.item(), input.size(0))

        accs.append(top1.avg)
        acc_info = ' * Prec@1 {top1.avg:.3f}'.format(top1=top1)
        print(acc_info)
        logfile.write(acc_info + '\n')

    avg_acc = np.mean(accs[1:])
    accs.append(avg_acc)
    print('avg acc', avg_acc)
    logfile.write(str(t) +' avg acc: ' + str(avg_acc) + '\n')

def save_checkpoint(state, dataset, exp_name, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/%s/"%(dataset, exp_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def asarray_and_reshape(imgs, labels):
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    imgs = np.reshape(imgs, (-1, 3, 32, 32)) # pytorch CHW
    labels = np.reshape(labels, (-1,))
    return imgs, labels

def load_svhn(data_dir, split='train'):
    print('Loading SVHN dataset.')
    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'
    image_dir = os.path.join(data_dir, 'svhn', image_file)
    svhn = io.loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 255.
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    return images, labels

def load_mnist(data_dir, split='train'):

    print('Loading MNIST dataset.')
    image_file = 'train.pkl' if split == 'train' else 'test.pkl'
    image_dir = os.path.join(data_dir, 'mnist', image_file)
    with open(image_dir, 'rb') as f:
        mnist = pickle.load(f, encoding="bytes")
    images = mnist['X']
    labels = mnist['y']
    images = images / 255.
    images = np.stack((images, images, images), axis=3)  # grayscale to rgb
    return np.squeeze(images[:10000]), labels[:10000]

def load_mnist_m(data_dir, split='train'):
    print('Loading MNIST_M dataset.')

    image_dir = os.path.join(data_dir, 'mnist_m')

    if split == 'train':
        data_dir = os.path.join(image_dir, 'mnist_m_train')
        with open(os.path.join(image_dir, 'mnist_m_train_labels.txt')) as f:
            content = f.readlines()

    elif split == 'test':
        data_dir = os.path.join(image_dir, 'mnist_m_test')
        with open(os.path.join(image_dir, 'mnist_m_test_labels.txt')) as f:
            content = f.readlines()

    content = [c.split('\n')[0] for c in content]
    images_files = [c.split(' ')[0] for c in content]
    labels = np.array([int(c.split(' ')[1]) for c in content]).reshape(-1)
    images = np.zeros((len(labels), 32, 32, 3))
    for no_img, img in enumerate(images_files):
        img_dir = os.path.join(data_dir, img)
        im = imageio.imread(img_dir)
        im = np.expand_dims(im, axis=0)
        images[no_img] = im
    images = images
    images = images / 255.

    return images, labels

def load_syn(data_dir, split='train'):
    print('Loading SYN dataset.')
    image_file = 'synth_train_32x32.mat' if split == 'train' else 'synth_test_32x32.mat'
    image_dir = os.path.join(data_dir, 'syn', image_file)
    syn = scipy.io.loadmat(image_dir)
    images = np.transpose(syn['X'], [3, 0, 1, 2])
    labels = syn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    images = images / 255.
    return images, labels

def load_usps(data_dir, split='train'):
    print('Loading USPS dataset.')
    image_file = 'usps_train_32x32.pkl' if split == 'train' else 'usps_test_32x32.pkl'
    # image_file = 'usps_32x32.pkl'
    image_dir = os.path.join(data_dir, 'usps', image_file)
    with open(image_dir, 'rb') as f:
        usps = pickle.load(f, encoding="bytes")
    images = usps['X']
    labels = usps['y']
    print('label range [{0}-{1}]'.format(np.min(labels), np.max(labels)))
    # labels -= 1
    # labels[labels == 255] = 9
    if np.max(images) == 255:
        images = images / 255.
    assert np.max(images) == 1
    images = np.squeeze(images)
    images = np.stack((images, images, images), axis=3)  # grayscale to rgb
    return images, labels

def load_test_data(data_dir, target):

    if target == 'svhn':
        target_test_images, target_test_labels = load_svhn(data_dir, split='test')
    elif target == 'mnist':
        target_test_images, target_test_labels = load_mnist(data_dir, split='test')
    elif target == 'syn':
        target_test_images, target_test_labels = load_syn(data_dir, split='test')
    elif target == 'usps':
        target_test_images, target_test_labels = load_usps(data_dir, split='test')
    elif target == 'mnist_m':
        target_test_images, target_test_labels = load_mnist_m(data_dir, split='test')
    return target_test_images, target_test_labels

if __name__ == '__main__':
    main()
