import argparse
import torch
from parallel_nets.ResNet_for_Gesture import *
import torch.optim as optim
import os
import torch.nn.functional as F
from torchvision import transforms
import time
from tensorboardX import SummaryWriter
from DVS_dataload.DVS_Gesture_dataset import *
import copy


def data_model_load(args, model, kwargs):
    path = os.path.join(os.path.join(os.getcwd(), 'data'), 'DVS_Gesture')
    train_dataset = DVSGestureDataset(path, train=True, transform=Compose([Normalize_ToTensor()]))
    test_dataset = DVSGestureDataset(path, train=False, transform=Compose([Normalize_ToTensor()]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    if args.pretrained:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('Pretrained model loaded.')
    else:
        start_epoch = 0
        print('Model loaded.')
    return train_loader, test_loader, start_epoch


def train(args, model, train_loader, optimizer, device, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data_temp, target = data.to(device), target.to(device)
        bs = data_temp.shape[0]
        data = torch.zeros((TimeStep * bs,) + data_temp.shape[2:], device=data_temp.device)
        for t in range(TimeStep):
            data[t*bs:(t+1)*bs, ...] = data_temp[:, t, :, :, :]

        output = model(data)
        target = target.long().to(target.device)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.tensorboard:
                writer.add_scalar('Train Loss / batch_idx', loss.item(), batch_idx + len(train_loader) * epoch)


def test(args, model, test_loader, device, writer):
    model.eval()
    total_loss = 0.
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data_temp, target = data.to(device), target.to(device)
            bs = data_temp.shape[0]
            data = torch.zeros((TimeStep * bs,) + data_temp.shape[2:], device=data_temp.device)
            for t in range(TimeStep):
                data[t * bs:(t + 1) * bs, ...] = data_temp[:, t, :, :, :]

            output = model(data)
            target = target.long().to(target.device)
            total_loss += F.cross_entropy(output, target, reduction='sum').item()
            pre_result = output.argmax(dim=1, keepdim=True)
            correct += pre_result.eq(target.view_as(pre_result)).sum().item()

    total_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        total_loss, correct, len(test_loader.dataset),
        accuracy))

    if args.tensorboard:
        writer.add_scalar('Test Loss / epoch', total_loss, epoch)
        writer.add_scalar('Test Accuracy / epoch', accuracy, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trian')
    parser.add_argument('--batch-size', type=int, default=28, help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=28, help='input batch size for testing')
    parser.add_argument('--total-epochs', type=int, default=120, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--use-cuda', action='store_true', default=True, help='use CUDA training')
    parser.add_argument('--save', action='store_true', default=True, help='save model')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='write tensorboard')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pre-trained model')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model-interval', type=int, default=5,
                        help='save model every save_model_interval')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoint/dvs_gesture/result_gesture.pth',
                        help='use CUDA training')
    args = parser.parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.manual_seed(2)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    writer = None
    writer_path = './summaries/dvs_gesture/result_gesture' + '_' + str(len(os.listdir('./summaries/dvs_gesture')))
    if args.tensorboard:
        writer = SummaryWriter(writer_path)
        
    model = resnet20().to(device)

    train_loader, test_loader, start_epoch = data_model_load(args, model, kwargs)
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=momentum_SGD, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, 0.1)
    for _ in range(start_epoch):
        scheduler.step()

    for epoch in range(start_epoch + 1, args.total_epochs + 1):
        start_time = time.time()
        train(args, model, train_loader, optimizer, device, epoch, writer)
        test(args, model, test_loader, device, writer)
        waste_time = time.time() - start_time
        print('One epoch wasting time:{:.0f}s, learning rate:{:.8f}\n'.format(
            waste_time, optimizer.state_dict()['param_groups'][0]['lr']))
        if epoch % args.save_model_interval == 0:
            if args.save:
                state = {'model': model.state_dict(), 'epoch': epoch}
                torch.save(state, args.checkpoint_path)
        scheduler.step()

    if args.tensorboard:
        writer.close()
