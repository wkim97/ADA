import os
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import pretrainedmodels
from pytorchcv.model_provider import get_model as ptcv_get_model
from models.attack_generator import AttackGenerator, weights_init_normal
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Evaluation for Attentive Diversity Attack')
parser.add_argument('--surrogate', type=str, help='target model')
parser.add_argument('--target_layer', type=str,
                    help='target layer : '
                         'ex) Mixed_7c for inc-v3,'
                         '    features.21 for inc-v4,'
                         '    conv2d_7b for incres-v2,'
                         '    features.stage4.unit3 for res-v2')

parser.add_argument('--load_dir', default='./weights', type=str, help='directory for loading model weights')
parser.add_argument('--load_name', default='default', type=str, help='name of the loaded model weights')
parser.add_argument('--device', default=0, type=int, help='GPU device id')
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.device) if (torch.cuda.is_available()) else "cpu")

if 'inception' in args.surrogate or 'resnet_v2' in args.surrogate:
    mean_surrogate = [0.5, 0.5, 0.5]
    stddev_surrogate = [0.5, 0.5, 0.5]
else:
    mean_surrogate = [0.485, 0.456, 0.406]
    stddev_surrogate = [0.229, 0.224, 0.225]

target_models = {
    "inception_v3": pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet'),
    "inception_v4": pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet'),
    "inception_resnet_v2": pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet'),
    "resnet_v2": ptcv_get_model("preresnet152", pretrained=True),
    "vgg16": torchvision.models.vgg16(pretrained=True)
}


class NormalizeInverse(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean).to(device)
        std = torch.as_tensor(std).to(device)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


invTrans = NormalizeInverse(mean_surrogate, stddev_surrogate)


def normalize_and_scale_imagenet(delta_im, epsilon):
    for ci in range(3):
        mag_in_scaled = epsilon / stddev_surrogate[ci]
        delta_im[:, ci] = delta_im[:, ci].clone().clamp(-mag_in_scaled, mag_in_scaled)

    return delta_im


def renormalization(X, X_pert, epsilon):
    eps_added = normalize_and_scale_imagenet(X_pert - X.clone(), epsilon) + X.clone()
    for i in range(3):
        min_clamp = (0 - mean_surrogate[i]) / stddev_surrogate[i]
        max_clamp = (1 - mean_surrogate[i]) / stddev_surrogate[i]
        eps_added[:, i] = eps_added[:, i].clone().clamp(min_clamp, max_clamp)
    return eps_added


t_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_surrogate, std=stddev_surrogate),
])
testset = torchvision.datasets.ImageFolder('./data/imagenet_subset/val', transform=t_transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=8, shuffle=False)


def test():
    success = [0] * (len(target_models) + 1)
    total = 0
    for i, (image, label) in tqdm(enumerate(testloader)):
        ensemble_logit = torch.zeros(image.size(0), 1000).to(device)
        image, label = image.to(device), label.to(device)
        attack_model = AttackGenerator(base_channel_dim=64, input_img_channel=3, z_channel=16,
                                                 deeper_layer=False, num_class=1000, last_dim=3)
        attack_model = attack_model.to(device)
        attack_model.load_state_dict(torch.load(os.path.join(args.load_dir, args.load_name + '.pth'),
                                                map_location=device))
        attack_model.eval()

        with torch.no_grad():
            double_image = torch.cat((image, image), dim=0)
            z = torch.FloatTensor(image.shape[0] * 2, 16).normal_().to(device)
            adv_noise = attack_model(double_image, z)
            double_adv_image = double_image + adv_noise
            double_adv_image = renormalization(double_image, double_adv_image, 16.0 / 255.0)
            test_image = double_adv_image[:image.shape[0]]

        for i, model_name in enumerate(target_models.keys()):
            target_model = target_models[model_name].to(device)
            target_model.eval()
            if model_name == 'inception_v3':
                target_model.aux_logits = False
            if model_name == 'resnet_v2':
                target_model.features.final_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
            if 'vgg' in model_name:
                target_size = 224
                mean_target = [0.485, 0.456, 0.406]
                stddev_target = [0.229, 0.224, 0.225]
            else:
                target_size = 299
                mean_target = [0.5, 0.5, 0.5]
                stddev_target = [0.5, 0.5, 0.5]
            normalize = torchvision.transforms.Normalize(mean_target, stddev_target)
            resize = torchvision.transforms.Resize(target_size)

            with torch.no_grad():
                adv_image = resize(normalize(invTrans(test_image)))
                adv_output = target_model(adv_image)
                _, adv_predicted = torch.max(adv_output.data, 1)
                success[i] += (adv_predicted != label).sum().item()
                if model_name != args.surrogate:
                    ensemble_logit += adv_output.detach().clone()

        _, ensemble_predicted = torch.max(ensemble_logit.data, 1)
        success[-1] += (ensemble_predicted != label).sum().item()
        total += image.size(0)

    for elem, model_name in zip(success, target_models):
        asr = 100 * float(elem) / total
        print("{} : ASR {:.2f}%".format(model_name, asr))
    ensemble_asr = 100 * float(success[-1]) / total
    print("Ensemble : ASR {:.2f}%".format(ensemble_asr))
    print('----------------------------------------')


if __name__ == '__main__':
    test()
