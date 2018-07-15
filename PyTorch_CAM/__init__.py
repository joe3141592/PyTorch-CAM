import torch
import numpy as np
from PIL import Image
from matplotlib import cm


def torch_normalize(x):
    xmin = torch.min(x)
    xmax = torch.max(x)
    z = (x - xmin) / (xmax - xmin)
    return z


def np_normalize(x):
    xmin = np.min(x)
    xmax = np.max(x)
    z = (x - xmin) / (xmax - xmin)
    return z * 255.0


class CAM(object):

    def __init__(self, model):
        model_name = model.__class__.__name__
        print("hooking {}".format(model_name))
        assert model_name == "ResNet", "Model architecture not supported"

        model.layer4.register_forward_hook(self.__hook)
        self.conv_features = None
        self.model = model
        params = list(model.parameters())
        self.weights = params[-2]

    def __hook(self, _, inp, out):
        self.conv_features = out

    def generateCAM(self, idx, numpy=False):
        bz, n_kernels, h, w = self.conv_features.size()
        flat_features = self.conv_features.view(n_kernels, h * w)
        maps = self.weights.mm(flat_features)
        maps = maps.view(self.weights.size()[0], h, w)
        assert idx < maps.size()[0], "idx: [{}] but ony {} classes in output".format(idx, maps.size()[0])
        cam = (torch_normalize(maps[idx]) * 255).int()
        if numpy:
            return cam.cpu().detach().numpy()
        return cam

    def visualize(self, idx, img=None, f=50,alpha=0.9):
        cam = self.generateCAM(idx, numpy=True)
        cam[cam < f] = 0
        f = np.zeros((cam.shape[0], cam.shape[1], 3))
        f[:, :, 2] = alpha * cam
        cam = np.concatenate((np.expand_dims(cam, 2), f), 2).astype("uint8")
        for i in range(cam.shape[0]):
            for j in range(cam.shape[1]):
                cam[i, j, 0:3] = np.array(cm.jet(cam[i, j, 0]))[0:3] * 255
        if img:
            assert type(img) == Image.Image, "Please pass a PIL Image"
            assert img.size[0]==img.size[1], "x and y dimension must match"
            img = img.copy()
            FILTER = Image.BILINEAR
            overlay = Image.fromarray(cam).resize((int(img.size[0]), int(img.size[1])), FILTER)
            img.paste(overlay, (0, 0), overlay)
            return img
        else:
            overlay = Image.fromarray(cam).resize((100, 100))
            return overlay
