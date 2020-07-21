% matplotlib inline
import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms  # data augmentation and normalization
import torch.optim as optim

import torch.nn as nn

import matplotlib.pyplot as plt

import numpy as np


# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_kernel(model):
    model_weights = model.state_dict()
    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    for idx, filt in enumerate(model_weights['conv1.weight']):
        # print(filt[0, :, :])
        if idx >= 32: continue
        plt.subplot(4, 8, idx + 1)
        plt.imshow(filt[0, :, :], cmap="gray")
        plt.axis('off')

    plt.show()


def plot_kernel_output(model, images):
    fig1 = plt.figure()
    plt.figure(figsize=(1, 1))

    img_normalized = (images[0] - images[0].min()) / (images[0].max() - images[0].min())
    plt.imshow(img_normalized.numpy().transpose(1, 2, 0))
    plt.show()
    output = model.conv1(images)
    layer_1 = output[0, :, :, :]
    layer_1 = layer_1.data

    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    for idx, filt in enumerate(layer_1):
        if idx >= 32: continue
        plt.subplot(4, 8, idx + 1)
        plt.imshow(filt, cmap="gray")
        plt.axis('off')
    plt.show()


def test_accuracy(net, dataloader):  # net=model, dataloader=iterable obj
    ########TESTING PHASE###########

    # check accuracy on whole test set
    correct = 0
    total = 0
    net.eval()  # important for deactivating dropout and correctly use batchnorm accumulated statistics
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)  # predictions
            _, predicted = torch.max(outputs.data, 1)  # predicted labels
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # compare with ground truth
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: %d %%' % (
        accuracy))
    return accuracy


n_classes = 100

        ####RUNNING CODE FROM HERE:


# transform are heavily used to do simple and complex transformation and data augmentation
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),  # resolution
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # data normalized between -1 and 1
    ])

transform_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4, drop_last=True)
# trainloader: to be given to the function
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=4, drop_last=True)

dataiter = iter(trainloader)

# show images just to understand what is inside the dataset ;)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))


# Residual Network:
net = models.resnet18(pretrained=True)
net.fc = nn.Linear(512, n_classes)  # changing the fully connected layer of the already allocated network

print("####plotting kernels of conv1 layer:####")
plot_kernel(net)

net = net.cuda()  # cast of the network for GPU computation

criterion = nn.CrossEntropyLoss().cuda()  # it already does softmax computation for use!
optimizer = optim.Adam(net.parameters(), lr=0.0001)  # better convergency w.r.t simple SGD :)


print("####plotting output of conv1 layer:#####")
plot_kernel_output(net,images)

########TRAINING PHASE###########
n_loss_print = len(trainloader)

n_epochs = 10
losses = np.empty(n_epochs)
accuracies = np.empty(n_epochs)
j = 0
for epoch in range(n_epochs):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % n_loss_print == (n_loss_print - 1):
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / n_loss_print))
            losses[j] = running_loss / n_loss_print
            running_loss = 0.0
    accuracies[j] = test_accuracy(net, testloader)  # at each epoch
    j += 1
print('Finished Training')

plt.plot(accuracies, 'r', label='Accuracy')
plt.plot(losses, 'b', label='Training Loss')
plt.legend(loc='best')
plt.show()