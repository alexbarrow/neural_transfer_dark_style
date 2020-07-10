import torch
import pickle
import matplotlib.pyplot as plt


from model import check_cuda,  run_style_transfer
from data_load import image_loader, imshow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_img_path = 'data/input_img.jpg'
style_img_path = 'data/style_img.jpg'

height = 550 if torch.cuda.is_available() else 128
width = 650 if torch.cuda.is_available() else 128

# Load images
style_img = image_loader(style_img_path, height, width)
content_img = image_loader(content_img_path, height, width)
input_img = content_img.clone()

print('Style Image size: {}, Content Image size {}'.format(style_img.size(), content_img.size()))

assert style_img.size() == content_img.size(), \
    "We need to import style and content images of the same size"

# plt.figure()
# imshow(style_img, title='Style Image')
# plt.figure()
# imshow(content_img, title='Content Image')

try:
    with open("models/vgg19_pretrained.pickle", "rb") as pickle_in:
        cnn = pickle.load(pickle_in)
except FileNotFoundError:
    print('Cannot find the model. Please download vgg19 pretrained model.')

cnn = cnn.features.to(device).eval()

# mean and std for pretrained model
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

check_cuda()

content_layers_default = ['conv_2', 'conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6']

output_img = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img,
                                content_layers=content_layers_default, style_layers=style_layers_default,
                                show_img=False, num_steps=300, style_weight=200000, content_weight=20, tv_reg=0.01)

check_cuda()

plt.figure()
imshow(output_img, title='Output Image', save=False, name='best_res')

