import os
import argparse
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import pretrainedmodels
from pytorchcv.model_provider import get_model as ptcv_get_model
from tqdm import tqdm

from utils import AverageMeter
from models.attack_generator import AttackGenerator, weights_init_normal
import gradcam


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


parser = argparse.ArgumentParser(description='Training for Attentive Diversity Attack')
parser.add_argument('--surrogate', type=str, help='target model')
parser.add_argument('--target_layer', type=str,
                    help='target layer : '
                         'ex) Mixed_7c for inc-v3,'
                         '    features.21 for inc-v4,'
                         '    conv2d_7b for incres-v2,'
                         '    features.stage4.unit3 for res-v2')

parser.add_argument('--epoch', default=100, type=int, help='total number of epochs')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--lam_div', default=1000., type=float, help='weight for div_loss')
parser.add_argument('--lam_attn', default=10., type=float, help='weight for attn_loss')
parser.add_argument('--z_dim', default=16, type=int, help='dimension of the latent vector')
parser.add_argument('--epsilon', default=16., type=float, help='perturbation constraint')

parser.add_argument('--save_dir', default='./weights', type=str, help='directory for saving model weights')
parser.add_argument('--save_name', default='default', type=str, help='name of the saved model weights')
parser.add_argument('--device', default=0, type=int, help='GPU device id')
args = parser.parse_args()


device = torch.device("cuda:{}".format(args.device) if (torch.cuda.is_available()) else "cpu")

if 'inception' in args.surrogate or 'resnet_v2' in args.surrogate:
    mean, stddev = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
else:
    mean, stddev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stddev)
])
dataset = torchvision.datasets.ImageFolder('./data/imagenet_subset/train', transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True)
testset = torchvision.datasets.folder.ImageFolder('./data/imagenet_subset/val', transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False)

target_models = {
    "resnet_v2": ptcv_get_model("preresnet152", pretrained=True),
    "inception_v3": pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet'),
    "inception_v4": pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet'),
    "inception_resnet_v2": pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet'),
    "vgg_16": torchvision.models.vgg16(pretrained=True)
}

class_model = target_models[args.surrogate]
if args.surrogate == 'inception_v3':
    class_model.aux_logits = False
if args.surrogate == 'resnet_v2':
    class_model.features.final_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
class_model = class_model.to(device)
class_model.eval()
target_layer = [args.target_layer]

G = AttackGenerator(base_channel_dim=64, input_img_channel=3, z_channel=args.z_dim, deeper_layer=False,
                    num_class=1000, last_dim=3)
G = G.to(device)
G.apply(weights_init_normal)

criterion_gcam = nn.MSELoss(reduction='mean')
criterion_ce = nn.CrossEntropyLoss()
G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=0.00001)


def normalize_and_scale_imagenet(delta_im, epsilon):
    for ci in range(3):
        mag_in_scaled = epsilon / stddev[ci]
        delta_im[:, ci] = delta_im[:, ci].clone().clamp(-mag_in_scaled, mag_in_scaled)

    return delta_im


def renormalization(X, X_pert, epsilon):
    eps_added = normalize_and_scale_imagenet(X_pert - X.clone(), epsilon) + X.clone()
    for i in range(3):
        min_clamp = (0 - mean[i]) / stddev[i]
        max_clamp = (1 - mean[i]) / stddev[i]
        eps_added[:, i] = eps_added[:, i].clone().clamp(min_clamp, max_clamp)
    return eps_added


def test():
    G.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            image, label = data
            image, label = image.to(device), label.to(device)
            double_image = torch.cat((image, image), dim=0)
            double_label = torch.cat((label, label), dim=0)
            z = torch.FloatTensor(image.shape[0] * 2, args.z_dim).normal_().to(device)
            adv_noise = G(double_image, z)

            double_adv_image = double_image + adv_noise
            double_adv_image = renormalization(double_image, double_adv_image, args.epsilon / 255)

            output = class_model(double_adv_image)
            _, predicted = torch.max(output.data, 1)
            total += double_label.size(0)
            correct += (predicted == double_label).sum().item()

    avg_acc = 100 * float(correct) / total
    return avg_acc


def train(epoch):
    G.train()

    #############################################################################################################
    # Training
    #############################################################################################################
    attn_losses = AverageMeter()
    div_losses = AverageMeter()
    cls_losses = AverageMeter()

    with tqdm(total=(len(dataset) - len(dataset) % args.batch_size)) as _tqdm:
        _tqdm.set_description('Epoch: {}/{}'.format(epoch, args.epoch))
        for data in dataloader:
            image, label = data
            image, label = image.to(device), label.to(device)
            double_image = torch.cat((image, image), dim=0)
            double_label = torch.cat((label, label), dim=0)
            z = torch.FloatTensor(image.shape[0] * 2, args.z_dim).normal_().to(device)
            adv_noise = G(double_image, z)

            double_adv_image = double_image + adv_noise
            double_adv_image = renormalization(double_image, double_adv_image, args.epsilon / 255)

            adv_output = class_model(double_adv_image)

            loss = 0.
            cls_loss = criterion_ce(adv_output, double_label)
            loss += -1 * cls_loss

            ori_gcam = gradcam.GradCAM(model=class_model, candidate_layers=target_layer)
            index, output = ori_gcam.forward(double_image)
            ori_gcam.backward(ids=index, output=output)
            ori_attn = ori_gcam.generate(target_layer=args.target_layer)

            adv_gcam = gradcam.GradCAM(model=class_model, candidate_layers=target_layer)
            _, adv_output = adv_gcam.forward(double_adv_image)
            adv_gcam.backward(ids=index, output=adv_output)
            adv_attn = adv_gcam.generate(target_layer=args.target_layer)

            attn_loss = 0.
            for a_attn, o_attn in zip(adv_attn, ori_attn):
                attn_loss += criterion_gcam(a_attn, o_attn)
            attn_loss /= len(adv_attn)
            loss += -1 * args.lam_attn * attn_loss

            div_loss = 0.
            for a_attn in adv_attn:
                numerator = torch.mean(torch.abs(a_attn[:image.shape[0]] - a_attn[image.shape[0]:]),
                                       dim=[_ for _ in range(1, len(a_attn.shape))])
                denominator = torch.mean(torch.abs(z[:image.shape[0]] - z[image.shape[0]:]),
                                         dim=[_ for _ in range(1, len(z.shape))])
                div_loss += torch.mean(numerator / (denominator + 1e-8))
            div_loss /= len(adv_attn)

            loss += -1 * args.lam_div * div_loss

            loss.backward()
            G_optimizer.step()

            ori_gcam.remove_hook()
            ori_gcam.clear()
            adv_gcam.remove_hook()
            adv_gcam.clear()

            cls_losses.update(cls_loss.data, len(image) * 2)
            attn_losses.update(attn_loss.data, len(image))
            div_losses.update(div_loss.data, len(image))

            _tqdm.set_postfix(
                attn_loss='{:.4f}'.format(attn_losses.avg),
                cls_loss='{:.4f}'.format(cls_losses.avg),
                div_loss='{:.4f}'.format(div_losses.avg),
            )
            _tqdm.update(args.batch_size)


if __name__ == '__main__':
    for epoch in range(args.epoch):
        train(epoch=epoch)
        avg_acc = test()
        print("AVG ACC:{}".format(avg_acc))

    torch.save(G.state_dict(), os.path.join(args.save_dir, args.save_name + '.pth'))
