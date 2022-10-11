import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_image(image, path, normalize=True):
    fig = plt.figure()
    sample = np.transpose(vutils.make_grid(image, normalize=normalize, nrow=1, padding=0).cpu().detach().numpy(),
                          (1, 2, 0))
    plt.imsave(path, sample)
    plt.close(fig)

def show_image(image):
    fig = plt.figure()
    sample = np.transpose(vutils.make_grid(image, normalize=True).cpu().detach().numpy(), (1, 2, 0))
    plt.imshow(sample)
    plt.show()
