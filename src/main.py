
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
import torchvision.models as models
import argparse
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
# print(imsize)
# imsize = 512

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    print("original image", image.size)
    # fake batch dimension required to fit network's input dimensions
    # crop image
    # image = fn.center_crop(image, (imsize, imsize))
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


## SETUP USING MAKE
parser = argparse.ArgumentParser(
    description="Command Line Arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
# parser.add_argument('--style_img', type=str, default='../data/picasso.jpg', help='path to style image')
# parser.add_argument('--content_img', type=str, default='../data/dancing.jpg', help='path to content image')
parser.add_argument('--folder', type=str, default='1', help='path to content, style image folders')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs train')
parser.add_argument('--model',type=str,default='vgg', help='pretrained model to use')
parser.add_argument('--content_layers',type=str,nargs='+',default='conv_4', help='content layers to use')
parser.add_argument('--style_layers',type=str,default='conv_1 conv_2 conv_3 conv_4 conv_5', help='style layers to use')
parser.add_argument('--show_img',type=str,default='false', help='see images')
parser.add_argument('--style_weight',type=int,default=1000000, help='setup weights')
# parser.add_argument('--trained_on',type=str,default='gap', help='Use whether gcnconv or gatconv')
# parser.add_argument('--need_training',type=str,default='false', help='training or testing')

args = parser.parse_args()

style_img_path = '../data/'+args.folder+'/style.jpg'
content_img_path = '../data/'+args.folder+'/content.jpg'
result_img_path = '../diff_models/'+args.folder+'/'       # TODO change this path acc to params
print(result_img_path, style_img_path, content_img_path)
os.makedirs('../diff_models/'+args.folder, exist_ok=True)


show_img = True if args.show_img.lower() == 'true' else False

style_img = image_loader(style_img_path)
content_img = image_loader(content_img_path)

print(style_img.size())
print(content_img.size())

# CROP the images(according to imsize)
style_img = fn.center_crop(style_img, (imsize, imsize))
content_img = fn.center_crop(content_img, (imsize, imsize))
# print(style_img.size())
# print(content_img.size())
# exit(0)
assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"



unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


# save an image
def imsave(tensor,path, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    if title is not None:
        plt.title(title)
    image.save(path)


######################################################################
# Loss Functions
# --------------
# Content Loss

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input



######################################################################
# Style Loss

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


if args.model=='vgg':
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
elif args.model=='resnet':
    # cnn = models.resnet50(pretrained=True).features.to(device).eval()
    cnn = models.resnet50(pretrained=True).to(device).eval()
elif args.model=='inception':
    cnn = models.inception_v3(pretrained=True).to(device).eval()
elif args.model=='googlenet':
    cnn = models.googlenet(pretrained=True).to(device).eval()


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
content_layers_default = args.content_layers
style_layers_default = args.style_layers

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    print(cnn)
    # exit(0)
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            # i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            i += 1
            name='other_{}'.format(i)
        # elif isinstance(layer, nn.Sequential):
        #     i += 1
        #     name = 'sequential_{}'.format(i)
        # elif isinstance(layer, nn.AdaptiveAvgPool2d):
        #     name = 'adaptiveavgpool_{}'.format(i)
        # elif isinstance(layer, nn.Linear):
        #     name = 'linear_{}'.format(i)
        # else:
        #     raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        print("name: ", name, "layer: ", layer)
        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses



# input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
input_img = torch.randn(content_img.data.size(), device=device)


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            if run[0] % 10 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                # print("Saving Output Image")

                # plt.figure()
                # toSavePath = result_img_path+'_'+str(run[0])+'_result.jpg'
                # print(toSavePath)
                # with torch.no_grad():
                #     output = input_img.clone().clamp_(0,1)
                #     imsave(output,path=toSavePath , title='Output Image')
                #     plt.close()
                # exit(0)

            run[0] += 1
            


            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


######################################################################
# Finally, we can run the algorithm.
# 

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img, num_steps = args.epochs, style_weight=args.style_weight)

if show_img:
    plt.figure()
    imshow(output, title='Output Image')

    plt.figure()
    imshow(style_img, title='Style Image')

    plt.figure()
    imshow(content_img, title='Content Image')

    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()

plt.figure()
print(result_img_path)
print(content_layers_default)
result_img_path = result_img_path+str(args.model)+'_result.jpg'
imsave(output,path=result_img_path , title='Output Image')