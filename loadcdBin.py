import torch
import torch.nn as nn
from Utils import L2Norm

import argparse
import cv2 as cv
import numpy as np


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant_(m.bias.data, 0.01)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant_(m.bias.data, 0.01)
        except:
            pass
    return

def input_norm(x):
    # print(x.shape)
    flat = x.view(x.size(0), -1)
    # print((flat.shape))
    mp = torch.mean(flat, dim=1)
    sp = torch.std(flat, dim=1) + 1e-7
    return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x))/sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    # x = x.detach().unsqueeze(-1).unsqueeze(-1)
    # return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x))/sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

def L2Norm(x):
    # def __init__(self):
    #     super(L2Norm,self).__init__()
    #     self.eps = 1e-10
    # def forward(self, x):
    #     norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
    #     x= x / norm.unsqueeze(-1).expand_as(x)
    #     return x
    eps = 1e-10
    norm = torch.sqrt(torch.sum(x * x, dim = 1) + eps)
    x= x / norm.unsqueeze(-1).expand_as(x)
    return x


class CDbin_NET_deep4_1(nn.Module):
    """CDbin_NET_deep4_1 model definition
    """
    def __init__(self,fcnum):
        super(CDbin_NET_deep4_1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
        )
        self.features1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, fcnum, kernel_size=8, bias=True),
            nn.BatchNorm2d(fcnum, affine=False),
        )
        self.features.apply(weights_init)
        self.features1.apply(weights_init)
        return


    # insert channel-wise pooling
    def forward(self, input):
        # print(input.size())
        x_features = self.features(input_norm(input))
        y = self.features1(x_features)
        x = y.view(y.size(0), -1)
        print("x= ")
        print(x)
        return L2Norm(x)



if __name__ == '__main__':
    model = CDbin_NET_deep4_1(256)
    # print(torch.load('CDbin_5_binary_256.pth',  map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load('CDbin_4_binary_256.pth', map_location=torch.device('cpu')), strict=False) # Load weights from .pth file
    model.load_state_dict(torch.load('CDbin_5_binary_256.pth', map_location=torch.device('cpu'))) # Load weights from .pth file
    model.eval()

    # Converting to TorchScript
    sm = torch.jit.script(model)
    print(sm)

    sm.save('sm.pt') #Serializing the Script Module to a File

    loadedSM = torch.jit.load('sm.pt') #Loading the Script Module from the File


    parser = argparse.ArgumentParser(description='Code for Feature Detection tutorial.')
    parser.add_argument('--input', help='Path to input image.', default='data/images/istockphoto-1129813604-612x612.jpg')
    args = parser.parse_args()

    src = cv.imread(cv.samples.findFile(args.input), cv.IMREAD_GRAYSCALE)
    if src is None:
        print('Could not open or find the image:', args.input)
        exit(0)

    # #-- Step 1: Detect the keypoints using SURF Detector
    # minHessian = 400
    # detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    # keypoints = detector.detect(src)
    # print(keypoints)

    ## MSER detector
    detector = cv.MSER_create()
    keypoints = detector.detect(src)
    # print((keypoints[51].pt))
    # print((keypoints[15].angle))
    # print((keypoints[154].class_id))
    # print((keypoints[56].octave))
    # print((keypoints[12].response))
    # print((keypoints[152].size))
    # print(len(keypoints))


    #-- Draw keypoints
    img_keypoints = np.empty((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    cv.drawKeypoints(src, keypoints, img_keypoints)

    #-- Show detected (drawn) keypoints
    cv.imshow('Keypoints', img_keypoints)
    cv.waitKey()

    # Loop through the keypoints

    i = 0
    for kp in keypoints:
        (x_x, y_y) = kp.pt 
        x_x = int(x_x)
        y_y = int(y_y)
        (srcX, srcY) = src.shape
        if x_x>32 and y_y>32 and x_x<(srcX-32) and y_y<(srcY-32):
            patch = src[x_x-32:x_x+32, y_y-32:y_y+32]
            i = i+ 1
            # cv.imshow('Patch', patch)
            # cv.waitKey()
            patch = torch.from_numpy(patch).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            # print(patch.shape) ## torch.Size([1, 1, 64, 64])
            desc = model(patch)
            descSM = sm(patch)
            # print(torch.equal(desc,descSM)) #https://pytorch.org/docs/stable/generated/torch.equal.html
            # print(desc)
            # print(desc[0][0].type()) ## [1 256] Float Tensor.
    # print(i)

