import argparseimport torchfrom mnist_cnn.model import CNN_mnistfrom PIL import Image, ImageOpsimport torchvision.transforms as transforms# from pillow import convert_imgdef load_img(input_img):    pil_img = Image.open(input_img)    flg = False    if pil_img.width != pil_img.height:        flg = True    pil_img = ImageOps.invert(pil_img)    pil_img = pil_img.resize((28, 28)).convert("L")    if flg:        pil_img = pil_img.rotate(270)    input = transforms.ToTensor()(            pil_img            ).unsqueeze(0)    return inputdef predict(input_img):    param = torch.load('mnist_cnn/mnist_model.pth')    model = CNN_mnist()    model.load_state_dict(param)    input = load_img(input_img)    # input = convert_img(input_img)    outputs = model(input)    _, predicted = torch.max(outputs, 1)    # print("prediction:", predicted[0].item())    return predicted[0].item()if __name__ == '__main__':    parser = argparse.ArgumentParser(description='predict number of input image')    parser.add_argument('--img_path', type=str, default='./templates/images/noimage.png',                        help='input image path')    args = parser.parse_args()    input_img = args.img_path    predict(input_img)