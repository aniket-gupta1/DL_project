import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



class output2env():
    def __init__(self, SGNum, envWidth = 16, envHeight = 8, isCuda = True ):
        self.envWidth = envWidth
        self.envHeight = envHeight

        Az = ( (np.arange(envWidth) + 0.5) / envWidth - 0.5 )* 2 * np.pi
        El = ( (np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0
        Az, El = np.meshgrid(Az, El)
        Az = Az[np.newaxis, :, :]
        El = El[np.newaxis, :, :]
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis = 0)
        ls = ls[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :, :]
        self.ls = Variable(torch.from_numpy(ls.astype(np.float32 ) ) )

        self.SGNum = SGNum
        if isCuda:
            self.ls = self.ls.cuda()

        self.ls.requires_grad = False

    def fromSGtoIm(self, axis, lamb, weight ):
        bn = axis.size(0)
        envRow, envCol = weight.size(2), weight.size(3)

        # Turn SG parameters to environmental maps
        axis = axis.unsqueeze(-1).unsqueeze(-1)

        weight = weight.view(bn, self.SGNum, 3, envRow, envCol, 1, 1)
        lamb = lamb.view(bn, self.SGNum, 1, envRow, envCol, 1, 1)

        mi = lamb.expand([bn, self.SGNum, 1, envRow, envCol, self.envHeight, self.envWidth] )* \
                (torch.sum(axis.expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth]) * \
                self.ls.expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth] ), dim = 2).unsqueeze(2) - 1)
        envmaps = weight.expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth] ) * \
            torch.exp(mi).expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth] )

        envmaps = torch.sum(envmaps, dim=1)

        return envmaps

    def output2env(self, axisOrig, lambOrig, weightOrig ):
        bn, _, envRow, envCol = weightOrig.size()

        axis = axisOrig

        weight = 0.999 * weightOrig
        weight = torch.tan(np.pi / 2 * weight )

        lambOrig = 0.999 * lambOrig
        lamb = torch.tan(np.pi / 2 * lambOrig )

        envmaps = self.fromSGtoIm(axis, lamb, weight )

        return envmaps, axis, lamb, weight