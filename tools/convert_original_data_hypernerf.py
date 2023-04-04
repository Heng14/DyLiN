import sys
sys.path.append("..")
import numpy as np
import math
import imageio
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import jax
print('Detected Devices:', jax.devices())

# import pycolmap
from third_party.pycolmap.pycolmap import Quaternion
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
import json

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

np.random.seed(2)

colmap_image_scale = 4 #4  # @param {type: 'number'}

# save_dir = '/home/hengyu/projects/hypernerf/dataset'  # @param {type: 'string'} 
save_dir = '/home/hengyu/projects/hypernerf/dataset_custom/dataset_output'
# @markdown The name of this capture. The working directory will be `$save_dir/$capture_name`. **Make sure you change this** when processing a new video.
# capture_name = 'vrig-chicken_0203'  # @param {type: 'string'}
# capture_name = 'americano_0130'  # @param {type: 'string'}
capture_name = 'captures_7361'  # @param {type: 'string'}
# The root directory for this capture.
# root_dir = Path(save_dir, capture_name)
root_dir = Path(save_dir, capture_name, 'capture1')
# Where to save RGB images.
rgb_dir = root_dir / 'rgb'
rgb_raw_dir = root_dir / 'rgb-raw'
# Where to save the COLMAP outputs.
colmap_dir = root_dir / 'colmap'

# @markdown The working directory where the trained model is.
# train_dir = '/home/hengyu/projects/hypernerf/experiments/vrig-chicken_0203_test'  # @param {type: "string"}
# train_dir = '/home/hengyu/projects/hypernerf/experiments/americano_0130_test'  # @param {type: "string"}
train_dir = '/home/hengyu/projects/hypernerf/experiments/captures_7361_test'  # @param {type: "string"}
# @markdown The directory to the dataset capture.
# data_dir = '/home/hengyu/projects/hypernerf/dataset_custom/dataset_output/captures_7361/capture1'  # @param {type: "string"}
data_dir = root_dir


scene_manager = SceneManager.from_pycolmap(
    colmap_dir / 'sparse/0', 
    rgb_dir / f'{colmap_image_scale}x', 
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
origins = np.array([c.position for c in ref_cameras])
directions = np.array([c.optical_axis for c in ref_cameras])
look_at = triangulate_rays(origins, directions)
print('look_at', look_at)

avg_position = np.mean(origins, axis=0)
print('avg_position', avg_position)

up = -np.mean([c.orientation[..., 1] for c in ref_cameras], axis=0)
print('up', up)

bounding_size = points_bounding_size(origins) / 2
x_scale =   0.75# @param {type: 'number'}
y_scale = 0.75  # @param {type: 'number'}
xs = x_scale * bounding_size
ys = y_scale * bounding_size
radius = 0.75  # @param {type: 'number'}
num_frames = 100  # @param {type: 'number'}
origin = np.zeros(3)
# z_offset = -0.1

datadir_kd_new = f'./data/{capture_name}_hypernerf_real_train/'
os.makedirs(datadir_kd_new, exist_ok=True)

split_size = 4096  # every 4096 rays will make up a .npy file
all_data = []

# origin = origins[i]

# angles = np.linspace(0, 2*math.pi, num=num_frames)
# positions = []

img_dir = rgb_dir / f'{colmap_image_scale}x'
meta_dir = root_dir / f'metadata.json'
dataset_dir = root_dir / f'dataset.json'

with open(meta_dir) as f:
    meta_data = json.load(f)

with open(dataset_dir) as f:
    dataset_data = json.load(f)

for i, camera in enumerate(ref_cameras):
# for i, angle in enumerate(angles):

    img_id = scene_manager.image_ids[i]
    img_path = f'{img_dir}/{img_id}.png'
    print (f'processing {img_path} ...')
    if not os.path.exists(img_path):
        print ('no such file: ', img_path)
        continue 

    if img_id not in meta_data:
        print ('no such meta: ', img_id)
        continue         

    if img_id not in dataset_data['train_ids']:
        print ('no train: ', img_id)
        continue    

    rgb_gt = imageio.imread(img_path)
    meta_i = meta_data[img_id]

#     print (f'processing {i} ...')

#     x = np.cos(angle) * radius * xs
#     y = np.sin(angle) * radius * ys
#     # x = xs * radius * np.cos(angle) / (1 + np.sin(angle) ** 2)
#     # y = ys * radius * np.sin(angle) * np.cos(angle) / (1 + np.sin(angle) ** 2)

#     position = np.array([x, y, z_offset])
#     # Make distance to reference point constant.
#     position = avg_position + position
  
    # camera = ref_camera.look_at(position, look_at, up)

    if colmap_image_scale != 1.0:
        camera = camera.scale(1 / colmap_image_scale)

    if scene_center is not None:
        camera.position = camera.position - scene_center
    if scene_scale is not None:
        camera.position = camera.position * scene_scale

    batch = datasets.camera_to_rays(camera)

    # warp_id = i

    batch['metadata'] = {
        'appearance': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) + meta_i['appearance_id'],
        'warp': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32) + meta_i['warp_id'],
    }

    # render = render_fn(state, batch, rng=rng)
    # rgb = np.array(render['rgb'])

    rays_o = batch['origins']
    rays_d = batch['directions']
    render_time = np.zeros_like(rgb_gt) + meta_i['warp_id']/(len(ref_cameras)-1)

    rgb_gt = rgb_gt / 255
    data = np.concatenate([rays_o, rays_d, render_time, rgb_gt], axis=-1)  # [H, W, 12]

    all_data += [data.reshape(rays_o.shape[0] * rays_o.shape[1], -1)]

    if i <= 20:
        # filename = f'{datadir_kd_new}/test_sample_{i}.png'
        # imageio.imsave(filename, rgb)

        filename_gt = f'{datadir_kd_new}/test_sample_{i}_gt.png'
        imageio.imsave(filename_gt, rgb_gt)

all_data = np.concatenate(all_data, axis=0)
# shuffle rays
rand_ix1 = np.random.permutation(all_data.shape[0])
rand_ix2 = np.random.permutation(all_data.shape[0])
all_data = all_data[rand_ix1][rand_ix2]

# save
split = 0
num = all_data.shape[0] // split_size * split_size
for ix in range(0, num, split_size):
    split += 1
    save_path = f'{datadir_kd_new}/train_{split}.npy'
    d = all_data[ix:ix + split_size]
    np.save(save_path, d)


