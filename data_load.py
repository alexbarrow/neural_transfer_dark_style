import torch

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_loader(image_name, img_height, img_width):
    loader = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor()])

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)

    return image.to(device, torch.float)


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)

    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show(block=False)
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img_path = 'data\input_img.jpg'
    style_img_path = 'data\style_img.jpg'

    height = 520 if torch.cuda.is_available() else 128
    width = 740 if torch.cuda.is_available() else 128

    style_img = image_loader(style_img_path, height, width)
    content_img = image_loader(content_img_path, height, width)

    print(style_img.size())
    print(content_img.size())
    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')