import numpy as np
import math
import marching_cubes as mcubes
import torch
import torchvision

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import pytorch3d.structures
import pytorch3d.renderer
import pytorch3d.structures
from pytorch3d.utils import ico_sphere

import matplotlib
import pdb


def extract_meshes(sdf):
    '''
        sdf: B X 1 X nC X nC X nC
    '''
    n_cells = sdf.shape[-1]
    verts = []
    faces = []
    verts_rgb = []
    for b in range(sdf.shape[0]):
        verts_b, faces_b = mcubes.marching_cubes(sdf[b,0].detach().cpu().numpy(), 0.02)
        verts_b = verts_b/n_cells - 0.5
        verts_b = torch.Tensor(verts_b).to(sdf.device)
        faces_b = torch.Tensor(faces_b.astype(np.int64)).to(sdf.device)
        verts.append(verts_b)
        faces.append(faces_b)
        verts_rgb.append(torch.ones_like(verts_b))

    return pytorch3d.structures.Meshes(verts, faces, textures=pytorch3d.renderer.Textures(verts_rgb=verts_rgb))



def sdf_samples_mesh(sample_pos, sample_val):
    '''
    Args:
        sample_pos: nS X B X 3
        sample_val: nS X B X 1
    '''
    nS = sample_val.shape[0]
    bs = sample_val.shape[1]
    sphere_mesh_init = ico_sphere(level=2, device=sample_pos.device)
    nV = sphere_mesh_init.num_verts_per_mesh().item()
    sphere_mesh_init = sphere_mesh_init.extend(nS)

    verts = []
    faces = []
    verts_rgb = []
    cmap = matplotlib.cm.get_cmap(name='bwr')

    for b in range(bs):
        trans_b = sample_pos[:,[b],:].repeat(1,nV,1)
        
        colors = torch.zeros_like(trans_b)
        # colors[:,:,0] += torch.sign(sample_val[:,[b],0]-0.1)
        # colors[:,:,2] += -1*torch.sign(sample_val[:,[b],0]-0.1)
        # colors = torch.clamp(colors,0,1).view(-1,3)

        for sx in range(nS):
            c_s = cmap(sample_val[sx,b,0].item()*0.5 + 0.4)
            colors[sx] += torch.Tensor(c_s[0:3]).view(1,3).to(colors.device)
        verts_rgb.append(colors.view(-1,3))

        # mesh_b = sphere_mesh_init.scale_verts(torch.abs(sample_val[:,b,0])*0.1)
        # sample_val_vis = torch.clamp(sample_val, -0.1, 0.1)
        # mesh_b = sphere_mesh_init.scale_verts(torch.abs(sample_val_vis[:,b,0]))

        mesh_b = sphere_mesh_init.scale_verts(0.05)
        mesh_b.offset_verts_(trans_b.view(-1,3))
        mesh_b = pytorch3d.structures.join_meshes_as_scene(mesh_b)
        verts.append(mesh_b.verts_list()[0])
        faces.append(mesh_b.faces_list()[0])

    return pytorch3d.structures.Meshes(verts, faces, textures=pytorch3d.renderer.Textures(verts_rgb=verts_rgb))
        


def visualize(positions, val_gt, val_pred=None, nz_slices=4, **kwargs):
    '''
    Args:
        positions: nS X B X 3 torch tensors
        val_{}: nS X B X nc values
    Returns:
        image array
    '''
    ns = val_gt.shape[0]
    bs = val_gt.shape[1]
    nC = val_gt.shape[2]
    H = int(math.sqrt(ns//nz_slices))*nz_slices
    W = ns // H
    # val_gt = torch.clamp(val_gt, -0.5, 1)
    val_gt = val_gt.view(H,W,bs,nC).permute(2,0,1,3).contiguous()
    val_gt = val_gt.view(-1,W,nC)
    vis_img = val_gt.detach().cpu().numpy()

    if val_pred is not None:
        val_pred = torch.clamp(val_pred, -0.5, 1)
        val_pred = val_pred.view(H,W,bs,nC).permute(2,0,1,3).contiguous()
        val_pred = val_pred.view(-1,W,nC)
        vis_img = np.concatenate((vis_img, val_pred.detach().cpu().numpy()), axis=1)

    vis_img = (np.clip(vis_img,-0.1,0.5) + 0.1)/0.6
    return vis_img
    

def coord_grid(D, H, W):
    '''
    grid:
        D X H X W X 3 (x,y,z) coord
    '''
    delta_z = 1/D
    delta_y = 1/H
    delta_x = 1/W
    xs = torch.linspace(-1 + delta_x, 1 - delta_x, W)
    ys = torch.linspace(-1 + delta_y, 1 - delta_y, H)
    zs = torch.linspace(-1 + delta_z, 1 - delta_z, D)
    Zs, Ys, Xs = torch.meshgrid([zs, ys, xs])

    grid = torch.stack([Xs, Ys, Zs], dim=-1)
    return grid


# -------------- Dataset ------------- #
# ------------------------------------ #
class SDFDataset(Dataset):
    def __init__(self, num_samples=10, num_queries=10, unif_query_sampling=False, jitter=False, nz_slices=4):
        ## child class should instantiate self.dset
        self.num_samples = num_samples
        self.num_queries = num_queries
        self.unif_query_sampling = unif_query_sampling
        self._jitter = jitter
        self.nz_slices = nz_slices


    def _sample_sdf(self, sdf, positions):
        '''
        Sample given image at specified positions
        Args:
            sdf: nC X H X W X D
            positions: S X 3 samples
        Returns:
            values: S X nC
        '''
        nC = sdf.shape[0]
        sdf = sdf[None,:,:,:,:]
        positions = positions.view(1,1,1,-1,3)
        values = torch.nn.functional.grid_sample(sdf, positions, align_corners=False)
        values = values.view(nC,-1).permute(1,0).contiguous()
        return values
    
    def __len__(self):
        return len(self.dset)

    def _scale(self, val, pos, scale=1.):
        return val*scale, pos*scale
    
    def _translate(self, pos, trans):
        return pos+trans
        
    def __getitem__(self, index):
        sdf = self.dset[index]

        sample_positions = (torch.rand(self.num_samples, 3)*2 - 1)
        if self.unif_query_sampling:
            nq = int(math.sqrt(self.num_queries//self.nz_slices))
            query_positions = coord_grid(self.nz_slices, nq, nq).view(-1,3)
        else:
            query_positions = (torch.rand(self.num_queries, 3)*2 - 1)

        sample_values = self._sample_sdf(sdf, sample_positions)
        query_values = self._sample_sdf(sdf, query_positions)

        if self._jitter:
            scale = torch.rand(1)*0.2 + 0.9
            trans = torch.rand(1,3)*0.2 - 0.1
            sdf = sdf*scale
            sample_values, sample_positions = self._scale(sample_values, sample_positions, scale)
            query_values, query_positions = self._scale(query_values, query_positions, scale)
            sample_positions = self._translate(sample_positions, trans)
            query_positions = self._translate(query_positions, trans)

        return {
            'sdf': sdf,
            'sample_positions': sample_positions,
            'sample_values': sample_values,
            'query_positions': query_positions,
            'query_values': query_values
        }
