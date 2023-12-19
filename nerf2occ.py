import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import trimesh

class NeRF2Occ:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.set_network()
        self.load_model()
    
    def set_network(self):
        if self.opt.ff:
            self.opt.fp16 = True
            assert self.opt.bg_radius <= 0, "background model is not implemented for --ff"
            from nerf.network_ff import NeRFNetwork
        elif self.opt.tcnn:
            self.opt.fp16 = True
            assert self.opt.bg_radius <= 0, "background model is not implemented for --tcnn"
            from nerf.network_tcnn import NeRFNetwork
        else:
            from nerf.network import NeRFNetwork
            
        self.model = NeRFNetwork( # TODO: consider replement this network
            encoding="hashgrid",
            bound=self.opt.bound,
            cuda_ray=self.opt.cuda_ray,
            density_scale=1,
            min_near=self.opt.min_near,
            density_thresh=self.opt.density_thresh,
            bg_radius=self.opt.bg_radius,
        )
        
    def load_model(self):
        if self.model and os.path.isfile(self.opt.ckpt):
            state_dict = torch.load(self.opt.ckpt, map_location="cpu")["model"]
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
        else:
            print(f"[Warning] Model can not be loaded from '{self.opt.ckpt}'!")
    
    @torch.no_grad      
    def run(
        self,
        grid_size = 64,
        sample_interval = 8 # if grid_size set to 64, max value of sample_interval should be 16
    ):
        bound = 1.0
        # sample points
        gsize = grid_size*sample_interval
        Y = (torch.arange(gsize, dtype=torch.float32, device=self.device)/(gsize) - 0.5) * (2 * bound)
        Z = (torch.arange(gsize, dtype=torch.float32, device=self.device)/(gsize) - 0.5) * (2 * bound)
        X = (torch.arange(gsize, dtype=torch.float32, device=self.device)/(gsize) - 0.5) * (2 * bound)
        points = torch.concat([t.unsqueeze(dim=3) for t in torch.meshgrid(X, Y, Z, indexing='ij')], dim=3)
        # get density (sigma and alpha)
        points = points.view(-1, 3)
        sigmas = self.model.density(points)['sigma']
        alphas = 1.0 - torch.exp(sigmas * (-1.0/gsize)) # TODO: check how to set d
        # max pooling
        alphas = alphas.view(1, gsize, gsize, gsize, 1)
        alphas = torch.permute(alphas, (0, 4, 1, 2, 3)) # [batch, channel, h, w, l]
        alphas = F.max_pool3d(alphas, kernel_size=sample_interval, stride=sample_interval)
        alphas = alphas.view(grid_size, grid_size, grid_size)
        # occupancy thresholding
        thres = 0.3
        if thres < 1:
            alphas[alphas > thres] = 1
            alphas[alphas <= thres] = 0
        else:
            alphas[alphas < thres] = 0
            alphas[alphas >= thres] = 1
        mesh = voxel2mesh(alphas.cpu().numpy())
        mesh.export("test.obj")
        
        
def voxel2mesh(voxel, threshold=0.4, use_vertex_normal: bool = False):
    verts, faces, vertex_normals = _voxel2mesh(voxel, threshold)
    if use_vertex_normal:
        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=vertex_normals)
    else:
        return trimesh.Trimesh(vertices=verts, faces=faces)


def _voxel2mesh(voxels, threshold=0.5):

    top_verts = [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    top_faces = [[0, 1, 3], [1, 2, 3]]
    top_normals = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]

    bottom_verts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    bottom_faces = [[1, 0, 3], [2, 1, 3]]
    bottom_normals = [[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]]

    left_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    left_faces = [[0, 1, 3], [2, 0, 3]]
    left_normals = [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0]]

    right_verts = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    right_faces = [[1, 0, 3], [0, 2, 3]]
    right_normals = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

    front_verts = [[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]
    front_faces = [[1, 0, 3], [0, 2, 3]]
    front_normals = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]

    back_verts = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]]
    back_faces = [[0, 1, 3], [2, 0, 3]]
    back_normals = [[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]]

    top_verts = np.array(top_verts)
    top_faces = np.array(top_faces)
    bottom_verts = np.array(bottom_verts)
    bottom_faces = np.array(bottom_faces)
    left_verts = np.array(left_verts)
    left_faces = np.array(left_faces)
    right_verts = np.array(right_verts)
    right_faces = np.array(right_faces)
    front_verts = np.array(front_verts)
    front_faces = np.array(front_faces)
    back_verts = np.array(back_verts)
    back_faces = np.array(back_faces)

    dim = voxels.shape[0]
    new_voxels = np.zeros((dim+2, dim+2, dim+2))
    new_voxels[1:dim+1, 1:dim+1, 1:dim+1] = voxels
    voxels = new_voxels

    scale = 2/dim
    verts = []
    faces = []
    vertex_normals = []
    curr_vert = 0
    a, b, c = np.where(voxels > threshold)

    for i, j, k in zip(a, b, c):
        if voxels[i, j, k+1] < threshold:
            verts.extend(scale * (top_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(top_faces + curr_vert)
            vertex_normals.extend(top_normals)
            curr_vert += len(top_verts)

        if voxels[i, j, k-1] < threshold:
            verts.extend(
                scale * (bottom_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(bottom_faces + curr_vert)
            vertex_normals.extend(bottom_normals)
            curr_vert += len(bottom_verts)

        if voxels[i-1, j, k] < threshold:
            verts.extend(scale * (left_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(left_faces + curr_vert)
            vertex_normals.extend(left_normals)
            curr_vert += len(left_verts)

        if voxels[i+1, j, k] < threshold:
            verts.extend(scale * (right_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(right_faces + curr_vert)
            vertex_normals.extend(right_normals)
            curr_vert += len(right_verts)

        if voxels[i, j+1, k] < threshold:
            verts.extend(scale * (front_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(front_faces + curr_vert)
            vertex_normals.extend(front_normals)
            curr_vert += len(front_verts)

        if voxels[i, j-1, k] < threshold:
            verts.extend(scale * (back_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(back_faces + curr_vert)
            vertex_normals.extend(back_normals)
            curr_vert += len(back_verts)

    return np.array(verts) - 1, np.array(faces), np.array(vertex_normals)     
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='scratch')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")
    parser.add_argument('--sparsity_loss_weight', type=float, default=0, help="set it > 0 to enable sparsity loss")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=1.0, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()
    return opt
    
if __name__ == "__main__":
    opt = parse_args()
    nerf2occ = NeRF2Occ(opt=opt, device=torch.device("cuda"))
    nerf2occ.run()