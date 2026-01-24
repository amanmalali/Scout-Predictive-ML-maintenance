'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

# import torchvision
# import torchvision.transforms as transforms

import os

from .models.resnet import ResNet18
# from .utils import progress_bar
import torchvision.transforms as transforms
import torch
import numpy as np
from utils import run_inference_dataset,image_dataset
from sensitivity.gen_train_data import generate_image_embeddings_v2,build_sensitivity_training_set_v2,build_sensitivity_training_ddla




device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ",device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# Model

criterion = nn.CrossEntropyLoss()



# Training
def train(net,trainloader,optimizer,epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(net,testloader,epoch,if_train=True,retrain=False,retrain_path=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    print('acc :',100.*correct/total)
    if if_train:
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..',acc)
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if retrain:
                if retrain_path is None:
                    torch.save(state, './cifar10/checkpoint/retrain_ckpt.pth')
                else:
                    torch.save(state, retrain_path)
            else:
                if not os.path.isdir('./cifar10/checkpoint'):
                    os.mkdir('./cifar10/checkpoint')
                torch.save(state, './cifar10/checkpoint/cifar_image_classifier.pth')
            best_acc = acc

def train_img_model(train_dataset,test_dataset=None,eval_dataset=None,epochs=100,lr=0.1,retrain=False,retrain_path=None):

    trainloader=torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
    testloader=torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=1)
    evalloader=torch.utils.data.DataLoader(eval_dataset, batch_size=100, shuffle=False, num_workers=1)

    net = ResNet18()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    for epoch in range(start_epoch, start_epoch+epochs):
        train(net,trainloader,optimizer,epoch)
        test(net,testloader,epoch,retrain=retrain,retrain_path=retrain_path)
        scheduler.step()
    
    if retrain:
        if retrain_path is None:
            state=torch.load("./cifar10/checkpoint/retrain_ckpt.pth")
        else:
            state=torch.load(retrain_path)
        net.load_state_dict(state['net'])
    else:
        state=torch.load("./cifar10/checkpoint/cifar_image_classifier.pth")
        net.load_state_dict(state['net'])

    
    if not retrain:
        print("PERFORMANCE ON DRIFTED DATA")
        test(net,evalloader,0,if_train=False)

    return net

def model_inf(model,X):
    # x = transforms.ToPILImage()(X)
    model.eval()
    X.to(device)
    pred=model(X.unsqueeze(0))
    return pred

def calc_loss(y_pred,y):
    y=torch.tensor([y],dtype=torch.long)
    y=y.to(device)
    loss=criterion(y_pred,y)
    return loss
