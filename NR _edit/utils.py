import os
import torch
import trimesh
import numpy as np
import torchvision.utils as vutils
from skimage import measure
from loguru import logger
from tools.render import Visualizer
from torch.autograd import Variable
import cv2

class output2img():
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

    def cvt2img(self, axisOrig, lambOrig, weightOrig ):
        bn, _, envRow, envCol = weightOrig.size()

        axis = axisOrig

        weight = 0.999 * weightOrig
        weight = torch.tan(np.pi / 2 * weight )

        lambOrig = 0.999 * lambOrig
        lamb = torch.tan(np.pi / 2 * lambOrig )

        envmaps = self.fromSGtoIm(axis, lamb, weight )

        return envmaps

def LSregression(pred, gt):
    nb = pred.size(0)
    origSize = pred.size()
    pred = pred.reshape(nb, -1)
    gt = gt.reshape(nb, -1)

    coef = (torch.sum(pred * gt, dim=1) / torch.clamp(torch.sum(pred * pred, dim=1), min=1e-5)).detach()
    coef = torch.clamp(coef, 0.001, 1000)
    for n in range(0, len(origSize) - 1):
        coef = coef.unsqueeze(-1)

    coef = coef.expand(origSize)

    return coef


# print arguments
def print_args(args):
    logger.info("################################  args  ################################")
    for k, v in args.__dict__.items():
        logger.info("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    logger.info("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        if len(vars.shape) == 0:
            return vars.data.item()
        else:
            return [v.data.item() for v in vars]
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


def coordinates(voxel_dim, device=torch.device('cuda')):
    """ 3d meshgrid of given size.

    Args:
        voxel_dim: tuple of 3 ints (nx,ny,nz) specifying the size of the volume

    Returns:
        torch long tensor of size (3,nx*ny*nz)
    """

    nx, ny, nz = voxel_dim
    x = torch.arange(nx, dtype=torch.long, device=device)
    y = torch.arange(ny, dtype=torch.long, device=device)
    z = torch.arange(nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


def apply_log_transform(tsdf):
    sgn = torch.sign(tsdf)
    out = torch.log(torch.abs(tsdf) + 1)
    out = sgn * out
    return out


def sparse_to_dense_torch_batch(locs, values, dim, default_val):
    dense = torch.full([dim[0], dim[1], dim[2], dim[3]], float(default_val), device=locs.device)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], locs[:, 3]] = values
    return dense


def sparse_to_dense_torch(locs, values, dim, default_val, device):
    dense = torch.full([dim[0], dim[1], dim[2]], float(default_val), device=device)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_channel(locs, values, dim, c, default_val, device):
    dense = torch.full([dim[0], dim[1], dim[2], c], float(default_val), device=device)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


def sparse_to_dense_np(locs, values, dim, default_val):
    dense = np.zeros([dim[0], dim[1], dim[2]], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense


class SaveScene(object):
    def __init__(self, cfg):
        self.cfg = cfg
        log_dir = cfg.LOGDIR.split('/')[-1]
        self.log_dir = os.path.join('results', 'scene_' + cfg.DATASET + '_' + log_dir)
        self.scene_name = None
        self.global_origin = None
        self.tsdf_volume = []  # not used during inference.
        self.weight_volume = []

        self.coords = None

        self.keyframe_id = None

        if cfg.VIS_INCREMENTAL:
            self.vis = Visualizer()

    def close(self):
        self.vis.close()
        cv2.destroyAllWindows()

    def reset(self):
        self.keyframe_id = 0
        self.tsdf_volume = []
        self.weight_volume = []

        # self.coords = coordinates(np.array([416, 416, 128])).float()

        # for scale in range(self.cfg.MODEL.N_LAYER):
        #     s = 2 ** (self.cfg.MODEL.N_LAYER - scale - 1)
        #     dim = tuple(np.array([416, 416, 128]) // s)
        #     self.tsdf_volume.append(torch.ones(dim).cuda())
        #     self.weight_volume.append(torch.zeros(dim).cuda())

    @staticmethod
    def tsdf2mesh(voxel_size, origin, tsdf_vol):
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)
        verts = verts * voxel_size + origin  # voxel grid coordinates to world coordinates
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
        return mesh

    def vis_incremental(self, epoch_idx, batch_idx, imgs, outputs):
        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()
        if self.cfg.DATASET == 'demo':
            origin[2] -= 1.5

        if (tsdf_volume == 1).all():
            logger.warning('No valid partial data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # vis
            key_frames = []
            for img in imgs[::3]:
                img = img.permute(1, 2, 0)
                img = img[:, :, [2, 1, 0]]
                img = img.data.cpu().numpy()
                img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                key_frames.append(img)
            key_frames = np.concatenate(key_frames, axis=0)
            cv2.imshow('Selected Keyframes', key_frames / 255)
            cv2.waitKey(1)
            # vis mesh
            self.vis.vis_mesh(mesh)

    def save_incremental(self, epoch_idx, batch_idx, imgs, outputs):
        save_path = os.path.join('incremental_' + self.log_dir + '_' + str(epoch_idx), self.scene_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()
        if self.cfg.DATASET == 'demo':
            origin[2] -= 1.5

        if (tsdf_volume == 1).all():
            logger.warning('No valid partial data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # save
            mesh.export(os.path.join(save_path, 'mesh_{}.ply'.format(self.keyframe_id)))

    def save_scene_eval(self, epoch, outputs, batch_idx=0):
        tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
        origin = outputs['origin'][batch_idx].data.cpu().numpy()

        if (tsdf_volume == 1).all():
            logger.warning('No valid data for scene {}'.format(self.scene_name))
        else:
            # Marching cubes
            mesh = self.tsdf2mesh(self.cfg.MODEL.VOXEL_SIZE, origin, tsdf_volume)
            # save tsdf volume for atlas evaluation
            data = {'origin': origin,
                    'voxel_size': self.cfg.MODEL.VOXEL_SIZE,
                    'tsdf': tsdf_volume}
            save_path = '{}_fusion_eval_{}'.format(self.log_dir, epoch)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savez_compressed(
                os.path.join(save_path, '{}.npz'.format(self.scene_name)),
                **data)
            mesh.export(os.path.join(save_path, '{}.ply'.format(self.scene_name)))

    def __call__(self, outputs, inputs, epoch_idx):
        # no scene saved, skip
        if "scene_name" not in outputs.keys():
            return

        batch_size = len(outputs['scene_name'])
        for i in range(batch_size):
            scene = outputs['scene_name'][i]
            self.scene_name = scene.replace('/', '-')

            if self.cfg.SAVE_SCENE_MESH:
                self.save_scene_eval(epoch_idx, outputs, i)
