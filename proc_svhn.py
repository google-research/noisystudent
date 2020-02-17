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
'''Process SVHN.'''
from absl import app
from absl import flags
import tensorflow as tf
import os
import numpy as np
import tarfile
import scipy.io
import sys
from six.moves import cPickle as pickle
try:
  from urllib import urlretrieve
except Exception:
  from urllib.request import urlretrieve
import utils

SVHN_DOWNLOAD_URL = 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat'
DOWNLOAD_DATA_FOLDER = 'downloaded_data'
MERGE_DATA_FOLDER = 'merged_raw_data'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'task_name', 'svhn',
    help='Task to use.')
flags.DEFINE_string(
    'raw_data_dir', None, 'Path of the raw data.')
flags.DEFINE_string('output_dir', '', 'Path of the tfrecord')
flags.DEFINE_bool(
    'full_train_data', False, 'Whether to use the full training data')


def save_merged_data(images, labels, split, merge_folder):
  with tf.gfile.Open(
      os.path.join(merge_folder, '{}_images.npy'.format(split)), 'wb') as ouf:
    np.save(ouf, images)
  with tf.gfile.Open(
      os.path.join(merge_folder, '{}_labels.npy'.format(split)), 'wb') as ouf:
    np.save(ouf, labels)


def download_and_extract():
  all_exist = True
  download_folder = os.path.join(FLAGS.raw_data_dir, DOWNLOAD_DATA_FOLDER)
  merge_folder = os.path.join(FLAGS.raw_data_dir, MERGE_DATA_FOLDER)
  splits = ['train', 'test']
  if FLAGS.task_name == 'svhn':
    splits += ['extra']
  for split in ['train', 'test']:
    for field in splits:
      if not tf.gfile.Exists(os.path.join(merge_folder, '{}_{}.npy'.format(
          split, field))):
        all_exist = False
  if all_exist:
    tf.logging.info('found all merged files')
    return
  tf.logging.info('downloading dataset')
  tf.gfile.MakeDirs(download_folder)
  tf.gfile.MakeDirs(merge_folder)
  if FLAGS.task_name == 'svhn':
    for split in splits:
      filename = os.path.join(download_folder, '{}_32x32.mat'.format(split))
      urlretrieve(SVHN_DOWNLOAD_URL.format(split), filename)
      tf.logging.info('downloaded {}'.format(filename))
      filename = os.path.join(download_folder, '{}_32x32.mat'.format(split))
      data_dict = scipy.io.loadmat(tf.gfile.Open(filename, "rb"))
      images = np.transpose(data_dict['X'], [3, 0, 1, 2])
      labels = data_dict['y'].reshape(-1)
      labels[labels == 10] = 0
      save_merged_data(images, labels, split, merge_folder)


def load_dataset():
  data = {}
  download_and_extract()
  merge_folder = os.path.join(FLAGS.raw_data_dir, MERGE_DATA_FOLDER)
  splits = ['train', 'test']
  if FLAGS.task_name == 'svhn':
    splits += ['extra']
  for split in splits:
    with tf.gfile.Open(
        os.path.join(merge_folder, '{}_images.npy'.format(split)), 'rb') as inf:
      images = np.load(inf)
    with tf.gfile.Open(
        os.path.join(merge_folder, '{}_labels.npy'.format(split)), 'rb') as inf:
      labels = np.load(inf)
    data[split] = {'images': images, 'labels': labels}
  return data


def save_tfrecord(data):
  coder = utils.ImageCoder()
  tf.gfile.MakeDirs(FLAGS.output_dir)
  for subset in data:
    if subset == 'train' or subset == 'extra':
      num_shards = 128
    else:
      num_shards = 64
    shard_spacing = np.linspace(
        0, data[subset]['images'].shape[0], num_shards + 1).astype(np.int)
    cnt = 0
    if subset == 'extra':
      output_dir = os.path.join(FLAGS.output_dir, 'unlabeled')
      tf.gfile.MakeDirs(output_dir)
    else:
      output_dir = FLAGS.output_dir
    for i in range(len(shard_spacing) - 1):
      if FLAGS.full_train_data:
        output_filename = '%s_full-%.5d-of-%.5d' % (subset, i, num_shards)
      else:
        output_filename = '%s-%.5d-of-%.5d' % (subset, i, num_shards)
      if i % 10 == 0:
        tf.logging.info('saving {}'.format(output_filename))
      writer = tf.python_io.TFRecordWriter(
          os.path.join(output_dir, output_filename))
      for j in range(shard_spacing[i], shard_spacing[i + 1]):
        cnt += 1
        image = data[subset]['images'][j]
        label = data[subset]['labels'][j] + 1
        image_data = coder.encode_jpeg(image)
        features = {
            'image/encoded': utils.bytes_feature(image_data),
            'image/class/label': utils.int64_feature(label),
        }
        if subset == 'extra':
          features['probabilities'] = utils.float_feature([0] * 10)
          features['label'] = utils.int64_feature(0)
          features['prob'] = utils.float_feature(0)
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
      writer.close()
    assert cnt == data[subset]['images'].shape[0]


def shuffle_split(data, split):
  perm_idx = np.arange(data[split]['images'].shape[0])
  np.random.shuffle(perm_idx)
  data[split]['images'] = data[split]['images'][perm_idx]
  data[split]['labels'] = data[split]['labels'][perm_idx]
  return data


def main(argv):
  data = load_dataset()
  np.random.seed(0)
  data = shuffle_split(data, 'train')
  if 'extra' in data:
    data = shuffle_split(data, 'extra')
  if FLAGS.task_name == 'svhn':
    dev_size = 4000 # 7300
  else:
    assert False
  data['dev'] = {}
  if not FLAGS.full_train_data:
    data['dev']['images'] = data['train']['images'][:dev_size]
    data['dev']['labels'] = data['train']['labels'][:dev_size]
    data['train']['images'] = data['train']['images'][dev_size:]
    data['train']['labels'] = data['train']['labels'][dev_size:]
  else:
    data['dev'] = data['test']
  tf.logging.info("dev labels" + str(data['dev']['labels'][:1000]) + '\n' * 5)
  if not FLAGS.full_train_data:
    if FLAGS.task_name == 'svhn':
      assert data['train']['images'].shape[0] == 73257 - 4000

  save_tfrecord(data)

if __name__ == '__main__':
  app.run(main)

