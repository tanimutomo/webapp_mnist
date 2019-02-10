import argparseimport numpy as npimport torchimport torch.optim as optimimport torchvision.utilsimport torchvision.transforms as transformsfrom torch.utils.data import DataLoaderfrom torchvision.datasets import MNISTfrom linear_model import Linear_mnistfrom model import CNN_mnistfrom model_tmp import CNN# argsparser = argparse.ArgumentParser(description='which model you use')parser.add_argument('--model', type=str, default='cnn', help='specify the model you use')args = parser.parse_args()# prepare train data and test data# test_data = MNIST('~/Project/deeplearning/mnist_data',transform = transforms.Compose(    [transforms.ToTensor(),     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])test_data = MNIST('~/Project/kronos/cnn/data/mnist_data/',        train=False, download=True,        transform=transform)test_loader = DataLoader(test_data, batch_size=4,        shuffle=False)classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')if args.model == 'cnn':    param = torch.load('./old_model_new_dataset.pth', map_location='cpu')    model = CNN_mnist()    # model = CNN()elif args.model == 'linear':    param = torch.load('mnist_model_linear.pth')    model = Linear_mnist()# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')model.load_state_dict(param)data_iter = iter(test_loader)images, labels = data_iter.next()correct = 0total = 0class_correct = list(0. for i in range(10))class_total = list(0. for i in range(10))with torch.no_grad():    for data in test_loader:        images, labels = data        outputs = model(images)        # outputsはdata(4つ分)のdataのsoftmaxの前の結果？        _, predicted = torch.max(outputs.data, 1)        # _はそれぞれのdataのoutputのなかでもっとも値の高い要素        # predictedは_のもっとも高い場所(つまりoutputの予測)        total += labels.size(0)        correct += (predicted == labels).sum().item()        c = (predicted == labels).squeeze()        for i in range(4):            label = labels[i]            class_correct[label] += c[i].item()            class_total[label] += 1print('Accuracy of the network on the 10000 test images: %d %%' %        (100 * correct / total))for i in range(10):    print('Accuracy of %5s : %0.6f %%' %            (classes[i], 100 * class_correct[i] / class_total[i]))