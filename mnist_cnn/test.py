import numpy as npimport torchimport torch.optim as optimimport torchvision.utilsimport torchvision.transforms as transformsfrom torch.utils.data import DataLoaderfrom torchvision.datasets import MNISTfrom model import CNN_mnist# prepare train data and test datatest_data = MNIST('~/Project/deeplearning/mnist_data',        train=False, download=True,        transform=transforms.ToTensor())test_loader = DataLoader(test_data, batch_size=4,        shuffle=False)classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')param = torch.load('mnist_model.pth')model = CNN_mnist()model.load_state_dict(param)data_iter = iter(test_loader)images, labels = data_iter.next()print(images.shape)print(labels.shape)correct = 0total = 0class_correct = list(0. for i in range(10))class_total = list(0. for i in range(10))with torch.no_grad():    for data in test_loader:        images, labels = data        outputs = model(images)        # outputsはdata(4つ分)のdataのsoftmaxの前の結果？        _, predicted = torch.max(outputs.data, 1)        # _はそれぞれのdataのoutputのなかでもっとも値の高い要素        # predictedは_のもっとも高い場所(つまりoutputの予測)        total += labels.size(0)        correct += (predicted == labels).sum().item()        c = (predicted == labels).squeeze()        for i in range(4):            label = labels[i]            class_correct[label] += c[i].item()            class_total[label] += 1print('Accuracy of the network on the 10000 test images: %d %%' %        (100 * correct / total))for i in range(10):    print('Accuracy of %5s : %0.6f %%' %            (classes[i], 100 * class_correct[i] / class_total[i]))