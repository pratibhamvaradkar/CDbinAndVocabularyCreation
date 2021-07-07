import torch
import torch.nn as nn
from Utils import L2Norm

import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math 


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

def getDesc(img, kpts):
    i = 0
    model = CDbin_NET_deep4_1(256)
    model.load_state_dict(torch.load('CDbin_5_binary_256.pth', map_location=torch.device('cpu'))) # Load weights from .pth file
    model.eval()

    des = []
    for kp in kpts:
        (x_x, y_y) = kp.pt 
        x_x = int(x_x)
        y_y = int(y_y)
        (srcX, srcY) = img.shape
        if x_x>32 and y_y>32 and x_x<(srcX-32) and y_y<(srcY-32):
            patch = img[x_x-32:x_x+32, y_y-32:y_y+32]
            i = i + 1
            # cv.imshow('Patch', patch)
            # cv.waitKey()
            patch = torch.from_numpy(patch).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            # print(patch.shape) ## torch.Size([1, 1, 64, 64])
            desc = model(patch)
            desc = desc.squeeze(0)
            desc = desc.detach().numpy()
            desc = np.sign(desc)
            desc = (desc > 0).astype(int)
            desc = np.packbits(desc)
            # print("desc: " + str(desc))

            des.append(desc)
    return des


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
        # print("x= ")
        # print(x)
        return L2Norm(x)



if __name__ == '__main__':
    # model = CDbin_NET_deep4_1(256)
    # # print(torch.load('CDbin_5_binary_256.pth',  map_location=torch.device('cpu')))
    # # model.load_state_dict(torch.load('CDbin_4_binary_256.pth', map_location=torch.device('cpu')), strict=False) # Load weights from .pth file
    # model.load_state_dict(torch.load('CDbin_5_binary_256.pth', map_location=torch.device('cpu'))) # Load weights from .pth file
    # model.eval()

    # # Converting to TorchScript
    # sm = torch.jit.script(model)
    # print(sm)

    # sm.save('sm.pt') #Serializing the Script Module to a File

    # loadedSM = torch.jit.load('sm.pt') #Loading the Script Module from the File


    # parser = argparse.ArgumentParser(description='Code for Feature Detection tutorial.')
    # parser.add_argument('--input', help='Path to input image.', default='data/images/istockphoto-1129813604-612x612.jpg')
    # args = parser.parse_args()

    # src = cv.imread(cv.samples.findFile(args.input), cv.IMREAD_GRAYSCALE)
    # if src is None:
    #     print('Could not open or find the image:', args.input)
    #     exit(0)

    # # #-- Step 1: Detect the keypoints using SURF Detector
    # # minHessian = 400
    # # detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    # # keypoints = detector.detect(src)
    # # print(keypoints)

    # ## MSER detector
    # detector = cv.MSER_create()
    # keypoints = detector.detect(src)
    # # print((keypoints[51].pt))
    # # print((keypoints[15].angle))
    # # print((keypoints[154].class_id))
    # # print((keypoints[56].octave))
    # # print((keypoints[12].response))
    # # print((keypoints[152].size))
    # # print(len(keypoints))


    # #-- Draw keypoints
    # img_keypoints = np.empty((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    # cv.drawKeypoints(src, keypoints, img_keypoints)

    # #-- Show detected (drawn) keypoints
    # cv.imshow('Keypoints', img_keypoints)
    # cv.waitKey()

    # # Loop through the keypoints

    # i = 0
    # for kp in keypoints:
    #     (x_x, y_y) = kp.pt 
    #     x_x = int(x_x)
    #     y_y = int(y_y)
    #     (srcX, srcY) = src.shape
    #     if x_x>32 and y_y>32 and x_x<(srcX-32) and y_y<(srcY-32):
    #         patch = src[x_x-32:x_x+32, y_y-32:y_y+32]
    #         i = i+ 1
    #         # cv.imshow('Patch', patch)
    #         # cv.waitKey()
    #         patch = torch.from_numpy(patch).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    #         # print(patch.shape) ## torch.Size([1, 1, 64, 64])
    #         desc = model(patch)
    #         descSM = sm(patch)
    #         # print(torch.equal(desc,descSM)) #https://pytorch.org/docs/stable/generated/torch.equal.html
    #         # print(desc)
    #         # print(desc[0][0].type()) ## [1 256] Float Tensor.
    # # print(i)

################################################################################################################
    # img1 = cv.imread('1305031102.175304.png',cv.IMREAD_GRAYSCALE)          # queryImage
    # img2 = cv.imread('1305031102.211214.png',cv.IMREAD_GRAYSCALE) # trainImage
    # # Initiate ORB detector
    # orb = cv.ORB_create()
    # # find the keypoints and descriptors with ORB
    # kp1 = orb.detect(img1,None)
    # kp2 = orb.detect(img2,None)
    # # kp1, des1 = orb.detectAndCompute(img1,None)
    # # kp2, des2 = orb.detectAndCompute(img2,None)
    # kp1d, des1_orb = orb.compute(img1,kp1)
    # kp2d, des2_orb = orb.compute(img2,kp2)

    # # print("des11 type: "+ str(np.shape(des11)))
    # # print(len(kp1))
    # des1_cdbin = np.array(getDesc(img1, kp1))
    # des2_cdbin = np.array(getDesc(img2, kp2))
    # # print("des1 type: "+ str(np.shape(des1)))

    # # create BFMatcher object
    # bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=False)
    # # Match descriptors.
    # matches_brief = bf.match(des1_orb,des2_orb)
    # matches_cdbin = bf.match(des1_cdbin,des2_cdbin)
    # print("matches: "+ str(len(matches_brief)))
    # print("matches: "+ str(len(matches_cdbin)))
    # # Sort them in the order of their distance.
    # matches_brief = sorted(matches_brief, key = lambda x:x.distance)
    # matches_cdbin = sorted(matches_cdbin, key = lambda x:x.distance)
    # # Draw first 10 matches.
    # img3 = cv.drawMatches(img1,kp1,img2,kp2,matches_cdbin[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()

##################################################################################################################
    img1 = cv.imread('/home/pratibha/Downloads/hpatches-release/i_autannes/e1.png',cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread('/home/pratibha/Downloads/hpatches-release/i_autannes/e2.png',cv.IMREAD_GRAYSCALE) # trainImage
    print(type(img1))
    
    height = img1.shape[0]
    model = CDbin_NET_deep4_1(256)
    model.load_state_dict(torch.load('CDbin_5_binary_256.pth', map_location=torch.device('cpu'))) # Load weights from .pth file
    model.eval()

    des1 = []
    des2 = []
    print(height)
    print(math.floor(height/64))
    for i in range(math.floor(height/64)):
        patch1 = img1[(64*i):(64*(i+1)-1), 0:63]
        patch2 = img2[(64*i):(64*(i+1)-1), 0:63]
        patch1 = torch.from_numpy(patch1).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
        desc1 = model(patch1)
        desc1 = desc1.squeeze(0)
        desc1 = desc1.detach().numpy()
        desc1 = np.sign(desc1)
        desc1 = (desc1 > 0).astype(int)
        desc1 = np.packbits(desc1)
        des1.append(desc1)

        patch2 = torch.from_numpy(patch2).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
        desc2 = model(patch2)
        desc2 = desc2.squeeze(0)
        desc2 = desc2.detach().numpy()
        desc2 = np.sign(desc2)
        desc2 = (desc2 > 0).astype(int)
        desc2 = np.packbits(desc2)
        des2.append(desc2)
        # print(i)

    des1 = np.array(des1)
    des2 = np.array(des2)
    print(len(des1))
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches_brief = bf.match(des1,des2)
    matches_brief = sorted(matches_brief, key = lambda x:x.distance)

    print("matches: "+ str(len(matches_brief)))


