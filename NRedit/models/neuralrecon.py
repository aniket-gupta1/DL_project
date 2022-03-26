import torch
import torch.nn as nn

from .backbone import MnasMulti
from .neucon_network import NeuConNet
from .gru_fusion import GRUFusion
from .lighting_models import Decoder, Encoder_Light, Decoder_Light
from utils import tocuda, output2img, LSregression


class NeuralRecon(nn.Module):
    '''
    NeuralRecon main class.
    '''

    def __init__(self, cfg):
        super(NeuralRecon, self).__init__()
        self.cfg = cfg.MODEL
        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.n_scales = len(self.cfg.THRESHOLDS) - 1

        # backbone encoder network
        self.backbone2d = MnasMulti(alpha)

        # Lighting estimation branch - Data decoders
        self.albedo_decoder = Decoder(mode=0)
        self.normal_decoder = Decoder(mode=1)
        self.depth_decoder = Decoder(mode=2)
        self.rough_decoder = Decoder(mode=3)

        # Lighting estimation branch - Lighting estimator
        self.Light_encoder = Encoder_Light(12)
        self.Axis_Decoder = Decoder_Light(12, mode=0)
        self.Lambda_Decoder = Decoder_Light(12, mode=1)
        self.Weight_Decoder = Decoder_Light(12, mode=2)

        # Mesh creation branch
        self.neucon_net = NeuConNet(cfg.MODEL)

        # for fusing to global volume
        self.fuse_to_global = GRUFusion(cfg.MODEL, direct_substitute=True)

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def forward(self, inputs, save_mesh=False):
        '''

        :param inputs: dict: {
            'imgs':                    (Tensor), images,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            when we have ground truth:
            'tsdf_list':               (List), tsdf ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            'occ_list':                (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            others: unused in network
        }
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results:
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
        }
                 loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        '''
        inputs = tocuda(inputs)
        outputs = {}
        imgs = torch.unbind(inputs['imgs'], 1)

        # image feature extraction
        # in: images; out: feature maps
        features = [self.backbone2d(self.normalizer(img)) for img in imgs]

        # send these features to 4 decoders for depth, albedo, normal and lighting
        albedo = self.albedo_decoder(features)
        normal = self.normal_decoder(features)
        rough = self.rough_decoder(features)
        depth = self.depth_decoder(features)

        # Compute the losses
        # albedo_loss =
        # normal_loss =
        # rough_loss =
        # depth_loss =

        for n in range(0, len(albedo)):
            albedo_loss = torch.sum((albedo[n] - albedoBatch) * (albedo[n] - albedoBatch))

        for n in range(0, len(normal)):
            normal_loss = torch.sum((normal[n] - normalBatch) * (normal[n] - normalBatch))

        for n in range(0, len(rough)):
            rough_loss = torch.sum((rough[n] - roughBatch) * (rough[n] - roughBatch))

        for n in range(0, len(depth)):
            depth_loss = torch.sum((torch.log(depth[n] + 1) - torch.log(depthBatch + 1))
                    * (torch.log(depth[n] + 1) - torch.log(depthBatch + 1)))

        # # Back propagate the gradients
        # totalErr = 6 * albedo_loss[-1] + normal_loss[-1] + 0.5 * rough_loss[-1] + 0.5 * depth_loss[-1]
        # totalErr.backward()

        # send the output of the 4 decoders to the lighting estimator encoder-decoder network
        # Concatenate the outputs of the 4 decoders
        feature_decoder_output = torch.cat([inputs, albedo, normal, rough, depth], dim=1)
        light_features = self.Light_encoder(feature_decoder_output)

        axis = self.Axis_Decoder(light_features)
        lamb = self.Lambda_Decoder(light_features)
        weight = self.Weight_Decoder(light_features)

        light_image_pred = output2img.cvt2img(axis, lamb, weight)
        light_error = LSregression(light_image_pred, light_image_gt)

        # Gather lighting branch loss
        print_lighting_branch_loss = f'Lighting_loss: Albedo: {albedo_loss} | Normal: {normal_loss} | Roughness: {rough_loss} | Depth: {depth_loss}'


        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, loss_dict = self.neucon_net(features, inputs, outputs)

        # fuse to global volume.
        if not self.training and 'coords' in outputs.keys():
            outputs = self.fuse_to_global(outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh)

        # gather loss.
        print_loss = 'Loss: '
        for k, v in loss_dict.items():
            print_loss += f'{k}: {v} '

        weighted_loss = 0

        for i, (k, v) in enumerate(loss_dict.items()):
            weighted_loss += v * self.cfg.LW[i]

        loss_dict.update({'total_loss': weighted_loss})
        loss_dict.update({'lighting_loss': light_error})
        loss_dict.update({'albedo_loss': albedo_loss})
        loss_dict.update({'normal_loss': normal_loss})
        loss_dict.update({'rough_loss': rough_loss})
        loss_dict.update({'depth_loss': depth_loss})
        return outputs, loss_dict
