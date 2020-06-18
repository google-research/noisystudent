# coding=utf-8
# Copyright 2019 The Google NoisyStudent Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''Model utilities.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json
import re
from absl import flags
from PIL import Image
import collections
import os
import functools
import numpy as np
import tensorflow as tf
import tensorflow.compat.v2 as tf2

from tensorflow.python.tpu import tpu_function


FLAGS = flags.FLAGS


def build_learning_rate(
    initial_lr,
    global_step,
    steps_per_epoch=None,
    lr_decay_type='exponential',
    decay_factor=0.97,
    decay_epochs=2.4,
    total_steps=None,
    warmup_epochs=5,
    start_from_step=0,
):
  '''Build learning rate.'''
  lr_step = global_step + start_from_step
  if lr_decay_type == 'exponential':
    assert steps_per_epoch is not None
    decay_steps = steps_per_epoch * decay_epochs
    lr = tf.train.exponential_decay(
        initial_lr, lr_step, decay_steps, decay_factor, staircase=True)
  elif lr_decay_type == 'cosine':
    assert total_steps is not None
    lr = 0.5 * initial_lr * (
        1 + tf.cos(np.pi * tf.cast(lr_step, tf.float32) / total_steps))
  elif lr_decay_type == 'constant':
    lr = initial_lr
  else:
    assert False, 'Unknown lr_decay_type : %s' % lr_decay_type

  if warmup_epochs:
    tf.logging.info('Learning rate warmup_epochs: %d' % warmup_epochs)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    warmup_lr = (
        initial_lr * tf.cast(lr_step, tf.float32) / tf.cast(
            warmup_steps, tf.float32))
    lr = tf.cond(lr_step < warmup_steps, lambda: warmup_lr, lambda: lr)

  return lr


def build_optimizer(learning_rate,
                    optimizer_name='rmsprop',
                    decay=0.9,
                    epsilon=0.001,
                    momentum=0.9):
  '''Build optimizer.'''
  if optimizer_name == 'sgd':
    tf.logging.info('Using SGD optimizer')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  elif optimizer_name == 'momentum':
    tf.logging.info('Using Momentum optimizer')
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=momentum)
  elif optimizer_name == 'rmsprop':
    tf.logging.info('Using RMSProp optimizer')
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay, momentum,
                                          epsilon)
  else:
    tf.logging.fatal('Unknown optimizer:', optimizer_name)

  return optimizer


class TpuBatchNormalization(tf.layers.BatchNormalization):
  # class TpuBatchNormalization(tf.layers.BatchNormalization):
  '''Cross replica batch normalization.'''

  def __init__(self, fused=False, **kwargs):
    if fused in (True, None):
      raise ValueError('TpuBatchNormalization does not support fused=True.')
    super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

  def _cross_replica_average(self, t, num_shards_per_group):
    '''Calculates the average value of input tensor across TPU replicas.'''
    num_shards = tpu_function.get_tpu_context().number_of_shards
    group_assignment = None
    if num_shards_per_group > 1:
      if num_shards % num_shards_per_group != 0:
        raise ValueError('num_shards: %d mod shards_per_group: %d, should be 0'
                         % (num_shards, num_shards_per_group))
      num_groups = num_shards // num_shards_per_group
      group_assignment = [[
          x for x in range(num_shards) if x // num_shards_per_group == y
      ] for y in range(num_groups)]
      return tf.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
          num_shards_per_group, t.dtype)
    else:
      tf.logging.info('TpuBatchNormalization None')
      return tf.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
          num_shards, t.dtype)

  def _moments(self, inputs, reduction_axes, keep_dims):
    '''Compute the mean and variance: it overrides the original _moments.'''
    shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = tpu_function.get_tpu_context().number_of_shards or 1
    if FLAGS.num_shards_per_group != -1:
      num_shards_per_group = FLAGS.num_shards_per_group
    else:
      if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
        num_shards_per_group = 1
      else:
        num_shards_per_group = max(8, num_shards // 8)
    tf.logging.info('TpuBatchNormalization with num_shards_per_group %s',
                    num_shards_per_group)
    if num_shards_per_group > 1 or num_shards_per_group == -2:
      # Compute variance using: Var[X]= E[X^2] - E[X]^2.
      shard_square_of_mean = tf.math.square(shard_mean)
      shard_mean_of_square = shard_variance + shard_square_of_mean
      group_mean = self._cross_replica_average(
          shard_mean, num_shards_per_group)
      group_mean_of_square = self._cross_replica_average(
          shard_mean_of_square, num_shards_per_group)
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)


def stochastic_depth(inputs, is_training, stochastic_depth_rate):
  '''Apply stochastic depth.'''
  if not is_training:
    return inputs

  # Compute keep_prob
  # TODO(tanmingxing): add support for training progress.
  keep_prob = 1.0 - stochastic_depth_rate

  # Compute stochastic_depth tensor
  batch_size = tf.shape(inputs)[0]
  random_tensor = keep_prob
  random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  output = tf.div(inputs, keep_prob) * binary_tensor
  return output


def archive_ckpt(ckpt_eval, ckpt_objective, ckpt_path):
  '''Archive a checkpoint if the metric is better.'''
  ckpt_dir, ckpt_name = os.path.split(ckpt_path)

  saved_objective_path = os.path.join(ckpt_dir, 'best_objective.txt')
  saved_objective = float('-inf')
  if tf.gfile.Exists(saved_objective_path):
    with tf.gfile.GFile(saved_objective_path, 'r') as f:
      saved_objective = float(f.read())
  if saved_objective > ckpt_objective:
    tf.logging.info('Ckpt %s is worse than %s', ckpt_objective, saved_objective)
    return False

  filenames = tf.gfile.Glob(ckpt_path + '.*')
  if filenames is None:
    tf.logging.info('No files to copy for checkpoint %s', ckpt_path)
    return False

  # Clear the old folder.
  dst_dir = os.path.join(ckpt_dir, 'archive')
  if tf.gfile.Exists(dst_dir):
    tf.gfile.DeleteRecursively(dst_dir)
  tf.gfile.MakeDirs(dst_dir)

  # Write checkpoints.
  for f in filenames:
    dest = os.path.join(dst_dir, os.path.basename(f))
    tf.gfile.Copy(f, dest, overwrite=True)
  ckpt_state = tf.train.generate_checkpoint_state_proto(
      dst_dir,
      model_checkpoint_path=ckpt_name,
      all_model_checkpoint_paths=[ckpt_name])
  with tf.gfile.GFile(os.path.join(dst_dir, 'checkpoint'), 'w') as f:
    f.write(str(ckpt_state))
  with tf.gfile.GFile(os.path.join(dst_dir, 'best_eval.txt'), 'w') as f:
    f.write('%s' % ckpt_eval)

  # Update the best objective.
  with tf.gfile.GFile(saved_objective_path, 'w') as f:
    f.write('%f' % ckpt_objective)

  tf.logging.info('Copying checkpoint %s to %s', ckpt_path, dst_dir)
  return True


# TODO(hongkuny): Consolidate this as a common library cross models.
class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
  '''Wrap keras DepthwiseConv2D to tf.layers.'''

  pass


def save_pic(uint8_arr, filename, log=True):
  if log:
    tf.logging.info('saving {}'.format(filename))
  img = Image.fromarray(uint8_arr)
  with tf.gfile.Open(filename, 'wb') as ouf:
    img.save(ouf, subsampling=0, quality=100)


def int64_feature(value):
  '''Wrapper for inserting int64 features into Example proto.'''
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
  '''Wrapper for inserting float features into Example proto.'''
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
  '''Wrapper for inserting bytes features into Example proto.'''
  if six.PY3 and isinstance(value, six.text_type):
    value = six.binary_type(value, encoding='utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class ImageCoder(object):
  '''Helper class that provides TensorFlow image coding utilities.'''

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
    self._encode_jpeg = tf.image.encode_jpeg(self._encode_jpeg_data,
                                             format='rgb', quality=100)

  def encode_jpeg(self, image):
    image_data = self._sess.run(self._encode_jpeg,
                                feed_dict={self._encode_jpeg_data: image})
    return image_data



def iterate_through_dataset(dst):
  iter = dst.make_initializable_iterator()
  elem = iter.get_next()
  cnt = 0
  with tf.Session() as sess:
    sess.run(iter.initializer)
    try:
      while True:
        features = sess.run(elem)
        yield features
    except tf.errors.OutOfRangeError:
      pass


def get_assignment_map_from_checkpoint(vars_list, init_checkpoint, only_teacher_model=False):
  graph_to_ckpt_map = {}
  assignment_map = {}
  for var in vars_list:
    ori_name = var.name
    ckpt_name = ori_name[:-len(':0')]
    if 'global_step' in ori_name:
      continue
    if only_teacher_model:
      # only initialize the teacher model
      if 'teacher_model' not in ori_name:
        continue
      ckpt_name = ckpt_name[len('teacher_model/'):]
    if 'RMSProp' not in ckpt_name and 'ExponentialMovingAverage' not in ckpt_name:
      ckpt_name += '/ExponentialMovingAverage'
    graph_to_ckpt_map[ori_name] = ckpt_name
    assignment_map[ckpt_name] = var

  init_vars = tf.train.list_variables(init_checkpoint)
  initialized_variable = {}
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in assignment_map:
      continue
    initialized_variable[name] = True
  new_assignment_map = {}
  for ckpt_name in assignment_map:
    if ckpt_name not in initialized_variable:
      block_name = ckpt_name.split('/')[1]
      assert False, ckpt_name + ' not found'
    else:
      new_assignment_map[ckpt_name] = assignment_map[ckpt_name]
  return new_assignment_map, graph_to_ckpt_map


def construct_scalar_host_call(
    metric_dict,
):
  metric_names = list(metric_dict.keys())

  def host_call_fn(gs, *args):
    gs = gs[0]
    # Host call fns are executed FLAGS.iterations_per_loop times after one
    # TPU loop is finished, setting max_queue value to the same as number of
    # iterations will make the summary writer only flush the data to storage
    # once per loop.
    with tf2.summary.create_file_writer(
        FLAGS.model_dir, max_queue=FLAGS.iterations_per_loop).as_default():
      with tf2.summary.record_if(tf.math.equal(tf.math.floormod(gs, FLAGS.iterations_per_loop), 0)):
        for i, name in enumerate(metric_names):
          scalar = args[i][0]
          # with tf.contrib.summary.record_summaries_every_n_global_steps(100, gs):
          tf2.summary.scalar(name, scalar, step=gs)
        return tf.summary.all_v2_summary_ops()
  global_step_tensor = tf.reshape(tf.train.get_or_create_global_step(), [1])
  other_tensors = [tf.reshape(metric_dict[key], [1]) for key in metric_names]
  host_call = (host_call_fn, [global_step_tensor] + other_tensors)
  return host_call


def get_all_variable():
  var_list = tf.trainable_variables() + tf.get_collection('moving_vars')
  for v in tf.global_variables():
    # We maintain mva for batch norm moving mean and variance as well.
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      var_list.append(v)
  var_list = list(set(var_list))
  var_list = sorted(var_list, key=lambda var: var.name)
  return var_list


def init_from_ckpt(scaffold_fn):
  all_var_list = get_all_variable()
  all_var_list = sorted(all_var_list, key=lambda var: var.name)

  if FLAGS.teacher_model_name:
    init_ckpt = FLAGS.teacher_model_path
  else:
    init_ckpt = FLAGS.init_model_path
  assignment_map, graph_to_ckpt_map = get_assignment_map_from_checkpoint(
      all_var_list, init_ckpt, FLAGS.teacher_model_name is not None)
  if FLAGS.use_tpu:
    def tpu_scaffold():
      tf.logging.info('initializing from {}'.format(init_ckpt))
      tf.train.init_from_checkpoint(init_ckpt, assignment_map)
      return tf.train.Scaffold()
    scaffold_fn = tpu_scaffold
  else:
    tf.train.init_from_checkpoint(init_ckpt, assignment_map)
  tf.logging.info('**** Variables ****')
  for var in all_var_list:
    init_string = ''
    if var.name in graph_to_ckpt_map:
      init_string = ', *INIT_FROM_CKPT* <== {}'.format(
          graph_to_ckpt_map[var.name])
    tf.logging.info('  name = %s, shape = %s%s', var.name, var.shape,
                    init_string)
  return scaffold_fn


def get_filename(data_dir, file_prefix, shard_id, num_shards):
  filename = os.path.join(
      data_dir,
      '%s-%05d-of-%05d' % (file_prefix, shard_id, num_shards))
  tf.logging.info('processing %s', filename)
  return filename


def get_dst_from_filename(filename, data_type, total_replicas=1, worker_id=0, get_label=False):
  input_files = [filename]
  if FLAGS.data_type == 'tfrecord':
    buffer_size = 8 * 1024 * 1024
    dst = tf.data.TFRecordDataset(input_files, buffer_size=buffer_size)
    dst = dst.shard(total_replicas, worker_id)
    dst = dst.map(parse_tfrecord, num_parallel_calls=16)
  else:
    assert False

  return dst


def parse_tfrecord(encoded_example):
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string),
  }
  parsed = tf.parse_single_example(encoded_example, keys_to_features)
  return parsed


def decode_raw_image(contents, channels=0):
  '''Decodes an image, ensuring that the result is height x width x channels.'''
  image = tf.image.decode_image(contents, channels)

  # Note: GIFs are decoded with 4 dimensions [num_frames, height, width, 3]
  image = tf.cond(
      tf.equal(tf.rank(image), 4),
      lambda: image[0, :],  # Extract first frame
      lambda: image)
  image_channel_shape = tf.shape(image)[2]
  image = tf.cond(
      tf.equal(image_channel_shape, 1),
      lambda: tf.image.grayscale_to_rgb(image), lambda: image)
  image.set_shape([None, None, 3])

  return image


def get_reassign_filename(data_dir, file_prefix, shard_id, num_shards, worker_id):
  filename = os.path.join(
      data_dir,
      '%s-%d-%05d-of-%05d' % (file_prefix, worker_id, shard_id, num_shards))
  tf.logging.info('writing to %s', filename)
  return filename

def get_uid_list():
    # get the mapping from class index to class name
    return [str(i) for i in range(FLAGS.num_label_classes)]


def label_dataset(worker_id, prediction_dir, shard_id, num_shards):
  def label_dst_parser(value):
    keys_to_features = {
        'probabilities': tf.FixedLenFeature([FLAGS.num_label_classes], tf.float32),
        'classes': tf.FixedLenFeature([], tf.int64),
    }
    parsed = tf.parse_single_example(value, keys_to_features)
    features = {}
    features['probabilities'] = tf.cast(
        tf.reshape(parsed['probabilities'], shape=[FLAGS.num_label_classes]), dtype=tf.float32)
    features['classes'] = tf.cast(
        tf.reshape(parsed['classes'], shape=[]), dtype=tf.int32)
    return features
  input_file = os.path.join(
      prediction_dir,
      'train-info-%.5d-of-%.5d-%.5d' % (shard_id, num_shards, worker_id))
  dst = tf.data.Dataset.list_files(input_file)
  def fetch_dataset(filename):
    buffer_size = 8 * 1024 * 1024
    dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
    return dataset
  dst = dst.apply(
      tf.data.experimental.parallel_interleave(
          fetch_dataset, cycle_length=1))
  dst = dst.apply(
      tf.data.experimental.map_and_batch(
          label_dst_parser, batch_size=1,
          num_parallel_batches=16))
  dst = dst.prefetch(tf.data.experimental.AUTOTUNE)
  return dst
