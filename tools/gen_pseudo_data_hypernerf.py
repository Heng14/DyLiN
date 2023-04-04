import sys
sys.path.append("..")
import numpy as np
import math
import imageio
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import jax
print('Detected Devices:', jax.devices())

# import pycolmap
# from third_party.pycolmap.pycolmap import Quaternion
import plotly.graph_objs as go
from pathlib import Path
from jax import numpy as jnp
from jax import random
from utils import *

from flax.training import checkpoints
from flax import jax_utils
from flax import optim
import functools
import gin

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from hypernerf import evaluation
from hypernerf import datasets
from hypernerf import configs
from hypernerf import models
from hypernerf import model_utils
from hypernerf import schedules
from hypernerf import training
from hypernerf import visualization as viz

from absl import logging
def myprint(msg, *args, **kwargs):
 print(msg % args)

logging.info = myprint 



colmap_image_scale = 4 #4 #4  # @param {type: 'number'}

save_dir = '/home/hengyu/projects/hypernerf/dataset'  # @param {type: 'string'} 
# save_dir = '/home/hengyu/projects/hypernerf/dataset_custom/dataset_output'
# @markdown The name of this capture. The working directory will be `$save_dir/$capture_name`. **Make sure you change this** when processing a new video.
# capture_name = 'vrig-chicken_0203'  # @param {type: 'string'}
# capture_name = 'americano_0130'  # @param {type: 'string'}
# capture_name = 'captures_7361'  # @param {type: 'string'}
# capture_name = '3dprinter++'  # @param {type: 'string'}
# capture_name = 'banana++'  # @param {type: 'string'}
capture_name = 'broom++'  # @param {type: 'string'}
# The root directory for this capture.
root_dir = Path(save_dir, capture_name)
# root_dir = Path(save_dir, capture_name, 'capture1')
# Where to save RGB images.
rgb_dir = root_dir / 'rgb'
rgb_raw_dir = root_dir / 'rgb-raw'
# Where to save the COLMAP outputs.
colmap_dir = root_dir / 'colmap'

# @markdown The working directory where the trained model is.
# train_dir = '/home/hengyu/projects/hypernerf/experiments/vrig-chicken_0203_test'  # @param {type: "string"}
# train_dir = '/home/hengyu/projects/hypernerf/experiments/americano_0130_test'  # @param {type: "string"}
# train_dir = '/home/hengyu/projects/hypernerf/experiments/captures_7361_test'  # @param {type: "string"}
# train_dir = '/home/hengyu/projects/hypernerf/experiments/3dprinter++'  # @param {type: "string"}
# train_dir = '/home/hengyu/projects/hypernerf/experiments_joel/banana++'
train_dir = '/home/hengyu/projects/hypernerf/experiments_joel/broom++'
# @markdown The directory to the dataset capture.
# data_dir = '/home/hengyu/projects/hypernerf/dataset_custom/dataset_output/captures_7361/capture1'  # @param {type: "string"}
data_dir = root_dir


scene_manager = SceneManager.from_pycolmap(
    colmap_dir / 'sparse/0', 
    rgb_dir / f'4x', 
    min_track_length=5)

if colmap_image_scale > 1:
  print(f'Scaling COLMAP cameras back to 1x from {colmap_image_scale}x.')
  for item_id in scene_manager.image_ids:
    camera = scene_manager.camera_dict[item_id]
    scene_manager.camera_dict[item_id] = camera.scale(colmap_image_scale)

new_scene_manager = scene_manager

near_far = estimate_near_far(new_scene_manager)
print('Statistics for near/far computation:')
print(near_far.describe())
print()

near = near_far['near'].quantile(0.001) / 0.8
far = near_far['far'].quantile(0.999) * 1.2
print('Selected near/far values:')
print(f'Near = {near:.04f}')
print(f'Far = {far:.04f}')

points = filter_outlier_points(new_scene_manager.points, 0.95)
bbox_corners = get_bbox_corners(
    np.concatenate([points, new_scene_manager.camera_positions], axis=0))

scene_center = np.mean(bbox_corners, axis=0)
scene_scale = 1.0 / np.sqrt(np.sum((bbox_corners[1] - bbox_corners[0]) ** 2))

checkpoint_dir = Path(train_dir, 'checkpoints')
checkpoint_dir.mkdir(exist_ok=True, parents=True)

config_path = Path(train_dir, 'config.gin')
with open(config_path, 'r') as f:
  logging.info('Loading config from %s', config_path)
  config_str = f.read()
gin.parse_config(config_str)

exp_config = configs.ExperimentConfig()
train_config = configs.TrainConfig()
eval_config = configs.EvalConfig()


dummy_model = models.NerfModel({}, 0, 0)
datasource = exp_config.datasource_cls(
    image_scale=exp_config.image_scale,
    random_seed=exp_config.random_seed,
    # Enable metadata based on model needs.
    use_warp_id=dummy_model.use_warp,
    use_appearance_id=(
        dummy_model.nerf_embed_key == 'appearance'
        or dummy_model.hyper_embed_key == 'appearance'),
    use_camera_id=dummy_model.nerf_embed_key == 'camera',
    use_time=dummy_model.warp_embed_key == 'time')



rng = random.PRNGKey(exp_config.random_seed)
np.random.seed(exp_config.random_seed + jax.process_index())
devices_to_use = jax.devices()

learning_rate_sched = schedules.from_config(train_config.lr_schedule)
nerf_alpha_sched = schedules.from_config(train_config.nerf_alpha_schedule)
warp_alpha_sched = schedules.from_config(train_config.warp_alpha_schedule)
elastic_loss_weight_sched = schedules.from_config(
train_config.elastic_loss_weight_schedule)
hyper_alpha_sched = schedules.from_config(train_config.hyper_alpha_schedule)
hyper_sheet_alpha_sched = schedules.from_config(
    train_config.hyper_sheet_alpha_schedule)

rng, key = random.split(rng)
params = {}
model, params['model'] = models.construct_nerf(
      key,
      batch_size=train_config.batch_size,
      embeddings_dict=datasource.embeddings_dict,
      near=datasource.near,
      far=datasource.far)

optimizer_def = optim.Adam(learning_rate_sched(0))
optimizer = optimizer_def.create(params)

state = model_utils.TrainState(
    optimizer=optimizer,
    nerf_alpha=nerf_alpha_sched(0),
    warp_alpha=warp_alpha_sched(0),
    hyper_alpha=hyper_alpha_sched(0),
    hyper_sheet_alpha=hyper_sheet_alpha_sched(0))
scalar_params = training.ScalarParams(
    learning_rate=learning_rate_sched(0),
    elastic_loss_weight=elastic_loss_weight_sched(0),
    warp_reg_loss_weight=train_config.warp_reg_loss_weight,
    warp_reg_loss_alpha=train_config.warp_reg_loss_alpha,
    warp_reg_loss_scale=train_config.warp_reg_loss_scale,
    background_loss_weight=train_config.background_loss_weight,
    hyper_reg_loss_weight=train_config.hyper_reg_loss_weight)

logging.info('Restoring checkpoint from %s', checkpoint_dir)
state = checkpoints.restore_checkpoint(checkpoint_dir, state)
step = state.optimizer.state.step + 1
state = jax_utils.replicate(state, devices=devices_to_use)
del params


devices = jax.devices()


def _model_fn(key_0, key_1, params, rays_dict, extra_params):
  out = model.apply({'params': params},
                    rays_dict,
                    extra_params=extra_params,
                    rngs={
                        'coarse': key_0,
                        'fine': key_1
                    },
                    mutable=False)
  return jax.lax.all_gather(out, axis_name='batch')

pmodel_fn = jax.pmap(
    # Note rng_keys are useless in eval mode since there's no randomness.
    _model_fn,
    in_axes=(0, 0, 0, 0, 0),  # Only distribute the data input.
    devices=devices_to_use,
    axis_name='batch',
)

render_fn = functools.partial(evaluation.render_image,
                              model_fn=pmodel_fn,
                              device_count=len(devices),
                              chunk=eval_config.chunk)



ref_cameras = [c for c in new_scene_manager.camera_list]
print ('ref_cameras len: ', len(ref_cameras))
# raise
origins = np.array([c.position for c in ref_cameras])
directions = np.array([c.optical_axis for c in ref_cameras])
look_at = triangulate_rays(origins, directions)
print('look_at', look_at)

avg_position = np.mean(origins, axis=0)
print('avg_position', avg_position)

# print (origins.shape)
# print (ref_cameras[1].orientation.shape)
# raise

# up = -np.mean([c.orientation[..., 1] for c in ref_cameras], axis=0)
# print('up', up)

# bounding_size = points_bounding_size(origins) / 2
# x_scale =   0.75# @param {type: 'number'}
# y_scale = 0.75  # @param {type: 'number'}
# xs = x_scale * bounding_size
# ys = y_scale * bounding_size
# radius = 0.75  # @param {type: 'number'}
# # num_frames = 100  # @param {type: 'number'}

# origin = np.zeros(3)

# print (tmp.position, tmp.orientation)
# look_at_tmp = triangulate_rays(origins, directions)
# tmp1 = tmp.look_at(origins[1], look_at_tmp, tmp.orientation[..., 1])
# print (tmp1.position, tmp1.orientation)
# raise

datadir_kd_new = f'./data/{capture_name}_pseudo_images10k_hypernerf/'
os.makedirs(datadir_kd_new, exist_ok=True)
n_pose_kd = 10000 #10000
split = 0
i_save, split_size = 100, 4096  # every 4096 rays will make up a .npy file
data = []

for i in range(1, n_pose_kd + 1):
    print (f'processing {i} ...')

    # idx1 = np.random.randint(len(ref_cameras))
    # idx2 = np.random.randint(len(ref_cameras))
    idx = np.random.choice(range(len(ref_cameras)), 2, replace=False)

    tmp1 = ref_cameras[idx[0]]
    tmp2 = ref_cameras[idx[1]]
    tmp = np.stack([tmp1.orientation, tmp2.orientation], axis=0)
    r = R.from_matrix(tmp)
    key_times = [0, 1]
    slerp = Slerp(key_times, r)
    interp_p = np.random.rand()
    interp_rots = slerp(interp_p)
    orientation_tmp = interp_rots.as_matrix()
    position_tmp = tmp1.position * (1-interp_p) + tmp2.position * interp_p

    camera = tmp1.copy()
    camera.position = position_tmp
    camera.orientation = orientation_tmp

    # ref_camera = ref_cameras[idx]
    # z_offset = -0.1

    # # angles = np.linspace(0, 2*math.pi, num=num_frames)
    # angle = np.random.rand() * 2*math.pi

    # x = np.cos(angle) * radius * xs
    # y = np.sin(angle) * radius * ys
    # # x = xs * radius * np.cos(angle) / (1 + np.sin(angle) ** 2)
    # # y = ys * radius * np.sin(angle) * np.cos(angle) / (1 + np.sin(angle) ** 2)

    # position = np.array([x, y, z_offset])
    # # Make distance to reference point constant.
    # position = avg_position + position
  
    # camera = ref_camera.look_at(position, look_at, up)

    if colmap_image_scale != 1.0:
        camera = camera.scale(1 / colmap_image_scale)

    if scene_center is not None:
        camera.position = camera.position - scene_center
    if scene_scale is not None:
        camera.position = camera.position * scene_scale

    batch = datasets.camera_to_rays(camera)

    warp_id = np.random.randint(len(ref_cameras))

    batch['metadata'] = {
        'appearance': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) + warp_id,
        'warp': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) + warp_id,
    }

    render = render_fn(state, batch, rng=rng)
    rgb = np.array(render['rgb'])
    # depth_med = np.array(render['med_depth'])
    # depth_viz = viz.colorize(depth_med.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)  
    # print (rgb.shape, depth_med.shape, depth_viz.shape)
    # imageio.imsave(f'test{i}.png', rgb)
    # raise

    rays_o = batch['origins']
    rays_d = batch['directions']
    render_time = np.zeros_like(rgb) + warp_id/(len(ref_cameras)-1)
    data_ = np.concatenate([rays_o, rays_d, render_time, rgb], axis=-1)  # [H, W, 12]

    data += [data_.reshape(rays_o.shape[0] * rays_o.shape[1], -1)]
    if i <= 20:
        filename = f'{datadir_kd_new}/pseudo_sample_{i}.png'
        imageio.imsave(filename, rgb)
    if i % i_save == 0:
        print('iter: ' + str(i))
        data = np.concatenate(data, axis=0)
        # shuffle rays
        rand_ix1 = np.random.permutation(data.shape[0])
        rand_ix2 = np.random.permutation(data.shape[0])
        data = data[rand_ix1][rand_ix2]
        
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



