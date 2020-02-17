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
'''Efficient input pipeline using tf.data.Dataset.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import abc
import collections
import functools
import os
import tensorflow as tf

import preprocessing
import efficientnet_builder

FLAGS = tf.app.flags.FLAGS


class TFExampleInput(object):
  '''Base class for input_fn generator.

  Args:
    is_training: `bool` for whether the input is for training
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    num_cores: `int` for the number of TPU cores
    image_size: `int` for image size (both width and height).
    transpose_input: 'bool' for whether to use the double transpose trick
  '''
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               is_training,
               use_bfloat16,
               num_cores=8,
               image_size=224,
               transpose_input=False,
               label_minus_one=True):
    self.image_preprocessing_fn = preprocessing.preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.num_cores = num_cores
    self.transpose_input = transpose_input
    self.image_size = image_size
    self.label_minus_one = label_minus_one

  def set_shapes(self, batch_size, features):
    '''Statically set the batch_size dimension.'''
    if self.is_training and self.transpose_input:
      features['image'].set_shape(features['image'].get_shape().merge_with(
          tf.TensorShape([None, None, None, batch_size])))
      features['label'].set_shape(features['label'].get_shape().merge_with(
          tf.TensorShape([batch_size])))
    else:
      features['image'].set_shape(features['image'].get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))
      features['label'].set_shape(features['label'].get_shape().merge_with(
          tf.TensorShape([batch_size])))

    return features

  def dataset_parser(self, value):
    '''Parses an image and its label from a serialized ResNet-50 TFExample.

    Args:
      value: serialized string containing an ImageNet TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    '''
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

    image = self.image_preprocessing_fn(
        input_tensor=image_bytes,
        is_training=self.is_training and not FLAGS.remove_aug,
        image_size=self.image_size,
        use_bfloat16=self.use_bfloat16,
        augment_name=FLAGS.augment_name,
        randaug_mag=FLAGS.randaug_mag,
    )
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32)
    # Subtract one so that labels are in [0, 1000).
    if self.label_minus_one:
      label = label - 1
    parsed_results = {'image': image, 'label': label}
    if FLAGS.teacher_model_name:
      teacher_image_size = efficientnet_builder.efficientnet_params(
          FLAGS.teacher_model_name)[2]
      if FLAGS.small_image_model:
        teacher_image_size = FLAGS.input_image_size
      teacher_image = self.image_preprocessing_fn(
          input_tensor=image_bytes,
          is_training=False,
          image_size=teacher_image_size,
          use_bfloat16=self.use_bfloat16)
      parsed_results['teacher_image'] = teacher_image
    return parsed_results

  @abc.abstractmethod
  def make_source_dataset(self,
                          index,
                          num_hosts,
                          all_data_dir=None,
                          cache=None,
                          unl=False,
                          num_train_shards=None):
    return


  def unl_dst_parser(self, value):
    keys_to_features = {
        'probabilities':
            tf.FixedLenFeature([FLAGS.num_label_classes], tf.float32),
        'label':
            tf.FixedLenFeature([], tf.int64, -1),
        'prob':
            tf.FixedLenFeature([], tf.float32),
        'image/encoded':
            tf.FixedLenFeature((), tf.string, ''),
    }
    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    ori_image = tf.image.decode_jpeg(image_bytes, channels=3)

    if FLAGS.unl_aug == 'default':
      augment_name = FLAGS.augment_name
    else:
      augment_name = FLAGS.unl_aug

    image = self.image_preprocessing_fn(
        input_tensor=ori_image,
        is_training=self.is_training and not FLAGS.remove_aug,
        image_size=self.image_size,
        use_bfloat16=self.use_bfloat16,
        augment_name=augment_name,
        randaug_mag=FLAGS.randaug_mag,
        is_image_bytes=False,
    )

    label = tf.cast(tf.reshape(parsed['label'], shape=[]), dtype=tf.int32)
    probabilities = tf.cast(
        tf.reshape(parsed['probabilities'], shape=[FLAGS.num_label_classes]),
        dtype=tf.float32)
    top_1_prob = tf.cast(tf.reshape(parsed['prob'], shape=[]), dtype=tf.float32)
    parsed_results = {
        'unl_image': image,
        'unl_label': label,
        'unl_probs': probabilities,
        'top_1_prob': top_1_prob,
    }
    if FLAGS.teacher_model_name:
      teacher_image_size = efficientnet_builder.efficientnet_params(
          FLAGS.teacher_model_name)[2]
      if FLAGS.small_image_model:
        teacher_image_size = FLAGS.input_image_size
      teacher_image = self.image_preprocessing_fn(
          input_tensor=image_bytes,
          is_training=False,
          image_size=teacher_image_size,
          use_bfloat16=self.use_bfloat16,
          augment_name=augment_name,
          randaug_mag=FLAGS.randaug_mag)
      parsed_results['unl_teacher_image'] = teacher_image
    return parsed_results

  def flatten_input(self, *features_list):
    flatten_result = {}
    for features in features_list:
      for key in features:
        assert key not in flatten_result
        flatten_result[key] = features[key]
    new_result = {}
    image_fields = ['image', 'unl_image']
    label_fields = ['label', 'unl_label']
    new_result['image'] = tf.concat(
        [flatten_result[key] for key in image_fields], 0)

    new_result['label'] = tf.concat(
        [flatten_result[key] for key in label_fields], 0)
    new_result['unl_probs'] = flatten_result['unl_probs']
    new_result['top_1_prob'] = flatten_result['top_1_prob']
    if FLAGS.teacher_model_name:
      new_result['teacher_image'] = tf.concat(
          [flatten_result['teacher_image'], flatten_result['unl_teacher_image']],
          0)
    return new_result

  def input_fn(self, params):
    '''input function which provides a single batch for train or eval.

    args:
      params: `dict` of parameters passed from the `tpuestimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    returns:
      a `tf.data.dataset` object.
    '''
    # retrieves the batch size for the current shard. the # of shards is
    # computed according to the input pipeline deployment. see
    # tf.contrib.tpu.runconfig for details.
    batch_size = params['batch_size']

    if self.is_training and 'context' in params:
      current_host = params['context'].current_input_fn_deployment()[1]
      num_hosts = params['context'].num_hosts
    else:
      # when evaluation, always let the first host read all data
      current_host = 0
      num_hosts = 1

    # use the fused map-and-batch operation.
    #
    # for xla, we must used fixed shapes. because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=true` to get fixed-size
    # batches without dropping any training examples.
    #
    # when evaluating, `drop_remainder=true` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. as long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    dataset = self.make_source_dataset(
        current_host, num_hosts, cache=self.cache,
        num_train_shards=FLAGS.num_train_shards)  # Thang
    dataset_parser = self.dataset_parser
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            dataset_parser,
            batch_size=batch_size,
            num_parallel_batches=self.num_cores,
            drop_remainder=True))

    if FLAGS.unlabel_ratio != 0 and self.is_training:
      real_unl_bsz = batch_size * FLAGS.label_data_sample_prob * FLAGS.unlabel_ratio
      unl_bsz = int(math.ceil(real_unl_bsz))
      unl_dst = self.make_source_dataset(
          current_host,
          num_hosts,
          all_data_dir=FLAGS.unlabel_data_dir,
          cache=self.cache,
          unl=True)
      unl_dst = unl_dst.map(
          self.unl_dst_parser, num_parallel_calls=self.num_cores * unl_bsz)

      unl_dst = unl_dst.batch(unl_bsz, drop_remainder=True)
      dataset = tf.data.Dataset.zip((dataset, unl_dst))
      dataset = dataset.map(
          self.flatten_input, num_parallel_calls=self.num_cores)
    else:
      unl_bsz = 0

    # transpose for performance on tpu
    if self.transpose_input and self.is_training:

      def transpose_fn(features):
        for key in features:
          if 'image' in key:
            # image and teacher_image
            features[key] = tf.transpose(features[key], [1, 2, 3, 0])
        return features

      dataset = dataset.map(transpose_fn, num_parallel_calls=self.num_cores)

    # assign static batch size dimension
    total_batch_size = batch_size + unl_bsz
    dataset = dataset.map(functools.partial(self.set_shapes, total_batch_size))

    # prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class DataInput(TFExampleInput):
  '''generates imagenet input_fn from a series of tfrecord files.

  the training data is assumed to be in tfrecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:

      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  the validation data is in the same format but sharded in 128 files.

  the format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  '''

  def __init__(self,
               is_training,
               use_bfloat16,
               transpose_input,
               data_dir,
               image_size=224,
               num_parallel_calls=64,
               cache=False,
               label_minus_one=True,
               subset=None
               ):
    '''create an input from tfrecord files.

    args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: if True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      data_dir: `str` for the directory of the training and validation data;
          if 'null' (the literal string 'null') or implicitly false
          then construct a null pipeline, consisting of empty images
          and blank labels.
      image_size: `int` for image size (both width and height).
      num_parallel_calls: concurrency level to use when reading data from disk.
      cache: if True, fill the dataset by repeating from its cache
    '''
    super(DataInput, self).__init__(
        is_training=is_training,
        image_size=image_size,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input,
        label_minus_one=label_minus_one)
    self.data_dir = data_dir
    if self.data_dir == 'null' or not self.data_dir:
      self.data_dir = None
    self.num_parallel_calls = num_parallel_calls
    self.cache = cache
    self.subset = subset

  def _get_null_input(self, data):
    '''returns a null image (all black pixels).

    args:
      data: element of a dataset, ignored in this method, since it produces
          the same null image regardless of the element.

    returns:
      a tensor representing a null image.
    '''
    del data  # unused since output is constant regardless of input
    return tf.zeros([self.image_size, self.image_size, 3],
                    tf.bfloat16 if self.use_bfloat16 else tf.float32)

  def dataset_parser(self, value):
    '''see base class.'''
    if not self.data_dir:
      return value, tf.constant(0, tf.int32)
    return super(DataInput, self).dataset_parser(value)

  def make_source_dataset(self,
                          index,
                          num_hosts,
                          all_data_dir=None,
                          cache=None,
                          unl=False,
                          num_train_shards=None):
    '''see base class.'''
    if not self.data_dir:
      tf.logging.info('undefined data_dir implies null input')
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)
    if cache is None:
      cache = self.cache

    # shuffle the filenames to ensure better randomization.
    if all_data_dir is None:
      all_data_dir = self.data_dir
    file_list = []
    for data_dir in all_data_dir.split(';'):
      if self.subset:
        subset = self.subset
      else:
        if unl:
          subset = 'train'
        else:
          subset = 'train' if self.is_training else 'validation'
      file_pattern = os.path.join(data_dir, '{}*'.format(subset))
      new_files = tf.gfile.Glob(file_pattern)
      if subset == 'train' and unl:
        file_pattern = os.path.join(data_dir, 'extra*')
        new_files += tf.gfile.Glob(file_pattern)
      tf.logging.info('# files={} for file_pattern: {}'.format(
          len(new_files), file_pattern))
      file_list += new_files

    file_list = sorted(file_list)

    # Thang: limit num_train_shards
    if self.is_training and num_train_shards:
      tf.logging.info('Thang: use %d out of %d shards' % (
          num_train_shards, len(file_list)))
      file_list = file_list[:num_train_shards]

    dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(file_list, dtype=tf.string))

    tf.logging.info('file stats for {}, num: {}, all: {}'.format(
        'unl' if unl else 'in', len(file_list), str(file_list[:10])))
    assert len(file_list) >= num_hosts, 'File list len %d vs num_hosts %d' % (
        len(file_list), num_hosts)
    dataset = dataset.shard(num_hosts, index)
    # this should be greater than number of files
    # shuffle should be able to prevent reading the same files after preemption
    if self.is_training:
      dataset = dataset.shuffle(len(file_list))

    if self.is_training and not cache:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 mib per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # TODO(qizhex) if the model is preempted, will this read the same file?
    # read the data from disk in parallel
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            fetch_dataset, cycle_length=self.num_parallel_calls, sloppy=True))
    if self.is_training:
      if cache:
        dataset = dataset.cache().apply(
            tf.data.experimental.shuffle_and_repeat(1024 * 16))
      else:
        dataset = dataset.shuffle(1024)
    return dataset
