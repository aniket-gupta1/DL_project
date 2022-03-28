import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
import os
import backbone_changed
import light_enc_dec
import torchvision.utils as vutils
import utils
import dataLoader
from outenv import output2env
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default=None, help='path to input images')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
parser.add_argument('--nepoch0', type=int, default=14, help='the number of epochs for training')
parser.add_argument('--batchSize0', type=int, default=16, help='input batch size')
opt = parser.parse_args()

opt.nepoch = opt.nepoch0
opt.batchSize = opt.batchSize0


lr_scale = 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



brdfDataset = dataLoader.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth, imHeight = opt.imHeight,
        cascadeLevel = opt.cascadeLevel )
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize,
        num_workers = 8, shuffle = True )

lightEncoder = light_enc_dec.encoderLight(cascadeLevel = 0, SGNum = 12 )
axisDecoder = light_enc_dec.decoderLight(mode=0, SGNum = 12 )
lambDecoder = light_enc_dec.decoderLight(mode = 1, SGNum = 12 )
weightDecoder = light_enc_dec.decoderLight(mode = 2, SGNum = 12 )

j = 0
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
depthErrsNpList = np.ones( [1, 1], dtype = np.float32 )

for epoch in list(range(opt.epochIdFineTune+1, opt.nepoch) ):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
    for i, dataBatch in enumerate(brdfLoader):
        j += 1
        # Load data from cpu to gpu
        albedo_cpu = dataBatch['albedo']
        albedoBatch = Variable(albedo_cpu ).cuda()

        normal_cpu = dataBatch['normal']
        normalBatch = Variable(normal_cpu ).cuda()

        rough_cpu = dataBatch['rough']
        roughBatch = Variable(rough_cpu ).cuda()

        depth_cpu = dataBatch['depth']
        depthBatch = Variable(depth_cpu ).cuda()

        segArea_cpu = dataBatch['segArea']
        segEnv_cpu = dataBatch['segEnv']
        segObj_cpu = dataBatch['segObj']

        seg_cpu = torch.cat([segArea_cpu, segEnv_cpu, segObj_cpu], dim=1 )
        segBatch = Variable(seg_cpu ).cuda()

        segBRDFBatch = segBatch[:, 2:3, :, :]
        segAllBatch = segBatch[:, 0:1, :, :]  + segBatch[:, 2:3, :, :]

        # Load the image from cpu to gpu
        im_cpu = (dataBatch['im'] )
        imBatch = Variable(im_cpu ).cuda()
        inputBatch = imBatch

        envmaps_cpu = dataBatch['envmaps']
        envmapsBatch = Variable(envmaps_cpu ).cuda()

        envmapsInd_cpu = dataBatch['envmapsInd']
        envmapsIndBatch = Variable(envmapsInd_cpu ).cuda()

        albedo,normal,rough,depth = backbone_changed.MnasMulti(inputBatch)
        
        albedoPred = 0.5 * albedo + 1
        normalPred = normal
        roughPred = rough
        depthPred = 0.5 * depth + 1

        albedoBatch = segBRDFBatch * albedoBatch
        # change this
        albedoPred1 = models.LSregress(albedoPred.detach() * segBRDFBatch.expand_as(albedoPred),
                albedoBatch * segBRDFBatch.expand_as(albedoBatch), albedoPred ) 
        albedoPred1 = torch.clamp(albedoPred1, 0, 1)

        depthPred1 = models.LSregress(depthPred.detach() *  segAllBatch.expand_as(depthPred),
                depthBatch * segAllBatch.expand_as(depthBatch), depthPred)
        #################3

        ## Compute Errors
        pixelObjNum = (torch.sum(segBRDFBatch ).cpu().data).item()
        pixelAllNum = (torch.sum(segAllBatch ).cpu().data).item()

        albedoErr = torch.sum( (albedoPred1 - albedoBatch )
                * (albedoPred1 - albedoBatch) * segBRDFBatch.expand_as(albedoBatch) / pixelObjNum / 3.0)
        normalErr = torch.sum( (normalPred - normalBatch)
                * (normalPred - normalBatch) * segAllBatch.expand_as(normalBatch) ) / pixelAllNum / 3.0
        roughErr = torch.sum( (roughPred - roughBatch)
                * (roughPred - roughBatch) * segBRDFBatch ) / pixelObjNum
        depthErr = torch.sum( (torch.log(depthPred1 + 1) - torch.log(depthBatch + 1) )
                * ( torch.log(depthPred1 + 1) - torch.log(depthBatch + 1) ) * segAllBatch.expand_as(depthBatch ) ) / pixelAllNum

        # Normalize Albedo and depth
        bn, ch, nrow, ncol = albedoPred.size()
        albedoPred = albedoPred.view(bn, -1)
        albedoPred = albedoPred / torch.clamp(torch.mean(albedoPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        albedoPred = albedoPred.view(bn, ch, nrow, ncol)

        bn, ch, nrow, ncol = depthPred.size()
        depthPred = depthPred.view(bn, -1)
        depthPred = depthPred / torch.clamp(torch.mean(depthPred, dim=1), min=1e-10).unsqueeze(1) / 3.0
        depthPred = depthPred.view(bn, ch, nrow, ncol)

        imBatchLarge = F.interpolate(imBatch, [480, 640], mode='bilinear')
        albedoPredLarge = F.interpolate(albedoPred, [480, 640], mode='bilinear')
        normalPredLarge = F.interpolate(normalPred, [480, 640], mode='bilinear')
        roughPredLarge = F.interpolate(roughPred, [480,640], mode='bilinear')
        depthPredLarge = F.interpolate(depthPred, [480, 640], mode='bilinear')

        x1, x2, x3, x4, x5, x6 = lightEncoder(inputBatch.detach() )
        axisPred = axisDecoder(x1, x2, x3, x4, x5, x6, envmapsBatch )
        lambPred = lambDecoder(x1, x2, x3, x4, x5, x6, envmapsBatch )
        weightPred = weightDecoder(x1, x2, x3, x4, x5, x6, envmapsBatch )

        bn, SGNum, _, envRow, envCol = axisPred.size()
        envmapsPred = torch.cat([axisPred.view(bn, SGNum * 3, envRow, envCol ), lambPred, weightPred], dim=1)

        imBatchSmall = F.adaptive_avg_pool2d(imBatch, (opt.envRow, opt.envCol) )
        segBatchSmall = F.adaptive_avg_pool2d(segBRDFBatch, (opt.envRow, opt.envCol) )
        notDarkEnv = (torch.mean(torch.mean(torch.mean(envmapsBatch, 4), 4), 1, True ) > 0.001 ).float()
        segEnvBatch = (segBatchSmall * envmapsIndBatch.expand_as(segBatchSmall) ).unsqueeze(-1).unsqueeze(-1)
        segEnvBatch = segEnvBatch * notDarkEnv.unsqueeze(-1).unsqueeze(-1)
        
        # Compute the recontructed error
        envmapsPredImage, axisPred, lambPred, weightPred = output2env.output2env(axisPred, lambPred, weightPred )

        pixelNum = max( (torch.sum(segEnvBatch ).cpu().data).item(), 1e-5)

        # figure out LSGress
        envmapsPredScaledImage = models.LSregress(envmapsPredImage.detach() * segEnvBatch.expand_as(envmapsBatch ),
                envmapsBatch * segEnvBatch.expand_as(envmapsBatch), envmapsPredImage )