from run_dnerf import config_parser, create_nerf
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from load_blender import pose_spherical
from run_dnerf import render_path
from run_dnerf import *
from run_dnerf_helpers import to8b
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

to_tensor = lambda x: x.to(device) if isinstance(
    x, torch.Tensor) else torch.Tensor(x).to(device)

to_array = lambda x: x if isinstance(x, np.ndarray) else x.data.cpu().numpy()
def get_rand_pose():
    '''Random sampling. Random origins and directions.
    '''
    theta1 = -180
    theta2 = 180
    phi1 = -90
    phi2 = 0
    theta = theta1 + np.random.rand() * (theta2 - theta1)
    phi = phi1 + np.random.rand() * (phi2 - phi1)
    return to_tensor(pose_spherical(theta, phi, 4))

# set cuda
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_name = 'jumpingjacks' #standup, jumpingjacks
# get config file
config_file = f'configs/{set_name}.txt'
parser = config_parser()
args = parser.parse_args(f'--config {config_file}')

# set render params
hwf = [400, 400, 555.5555155968841]
# hwf = [800, 800, 1111.1110311937682]

H, W, focal = hwf
_, render_kwargs_test, _, _, _ = create_nerf(args)
render_kwargs_test.update({'near' : 2., 'far' : 6.})

datadir_kd_new = f'./data/{set_name}_pseudo_images10k_dnerf/'
os.makedirs(datadir_kd_new, exist_ok=True)
n_pose_kd = 10000
split = 0
i_save, split_size = 100, 4096  # every 4096 rays will make up a .npy file
data = []

# time_rand = np.zeros((1,1)) #+ 0.2 #np.random.rand(1,1)
# render_time = torch.Tensor([time_rand]).to(device)
# render_time0 = render_time.repeat([H, W, 3])

for i in range(1, n_pose_kd + 1):
    time_rand = np.random.rand(1,1)
    render_time = torch.Tensor([time_rand]).to(device)
    
    render_pose = torch.unsqueeze(get_rand_pose(), 0).to(device)

    focal_ = focal * (np.random.rand() + 1) 
    hwf = [H, W, focal_]

    with torch.no_grad():
            rgb, _, rays_o, rays_d, x = render_path(render_pose, render_time, hwf, args.chunk, render_kwargs_test, render_factor=args.render_factor)

    render_time0 = render_time.repeat([H, W, 3])

    rays_o = rays_o.reshape([H, W, 3])
    rays_d = rays_d.reshape([H, W, 3])
    rgb = torch.squeeze(torch.Tensor(rgb), 0)
    data_ = torch.cat([torch.tensor(rays_o), torch.tensor(rays_d), render_time0, torch.tensor(rgb)], dim=-1)  # [H, W, 9]

    data += [data_.view(rays_o.shape[0] * rays_o.shape[1], -1)]
    if i <= 20:
        filename = f'{datadir_kd_new}/pseudo_sample_{i}.png'
        rgb = rgb.cpu().numpy()
        imageio.imwrite(filename, to8b(rgb))

    if i % i_save == 0:
        print('iter: ' + str(i))
        data = torch.cat(data, dim=0)
        # shuffle rays
        rand_ix1 = np.random.permutation(data.shape[0])
        rand_ix2 = np.random.permutation(data.shape[0])
        data = data[rand_ix1][rand_ix2]
        data = to_array(data)

        # save
        num = data.shape[0] // split_size * split_size
        for ix in range(0, num, split_size):
                split += 1
                save_path = f'{datadir_kd_new}/data_{split}.npy'
                d = data[ix:ix + split_size]
                np.save(save_path, d)
        print(
        f'[{i}/{n_pose_kd}] Saved data at "{datadir_kd_new}"')
        data = []  # reset