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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import time
import functools
import json
import tensorflow as tf
import numpy as np
import utils
import preprocessing
import data_input


FLAGS = flags.FLAGS

flags.DEFINE_string('predict_ckpt_path', '', 'The path to the checkpoint for prediction.')

flags.DEFINE_string('output_dir', '', 'Directory to store prediction results.')

flags.DEFINE_string('info_dir', '', 'Directory to store information for each shard.')

flags.DEFINE_integer(
    'num_shards', default=128, help='Total number of shards for the dataset.')

flags.DEFINE_string(
    'shard_id', default='0', help='Between 0 and num_shards - 1. The shard number to run prediction on.')

flags.DEFINE_integer(
    'total_replicas', default=1, help='Divide a shard into total_replicas copies of data.')

flags.DEFINE_integer(
    'worker_id', default=0, help='Between 0 and total_replicas - 1.')

flags.DEFINE_bool(
    'reassign_label', default=False, help='')

flags.DEFINE_string(
    'data_type', default='tfrecord', help='')

flags.DEFINE_string('file_prefix', 'train', '')

shard_id = 0


def set_shapes(batch_size, features):
  """Statically set the batch_size dimension."""
  for key in features:
    if 'image' in key:
      images = features[key]
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))
  if 'label' in features:
    features['label'].set_shape(features['label'].get_shape().merge_with(
        tf.TensorShape([batch_size])))
  return features


def preprocess(parsed):
  """Preprocess image for inference."""
  features = {}
  if FLAGS.data_type == 'tfrecord':
    image = tf.image.decode_jpeg(parsed['image/encoded'], channels=3)

  features['image'] = preprocessing.preprocess_image(
      image,
      is_training=False,
      image_size=FLAGS.input_image_size,
      use_bfloat16=FLAGS.use_bfloat16,
      is_image_bytes=False,
  )
  return features


def get_input_fn(params, raw_data=False):
  batch_size = params['batch_size']
  global shard_id
  if FLAGS.reassign_label:
    assert FLAGS.data_type == 'tfrecord'
    filename = os.path.join(
        FLAGS.label_data_dir,
        '%s-%d-%05d-of-%05d' % (
            FLAGS.file_prefix, FLAGS.worker_id,
            shard_id, FLAGS.num_shards))
    tf.logging.info('processing {}'.format(filename))
    # do not use replica here
    dst = utils.get_dst_from_filename(filename, FLAGS.data_type)
  else:
    filename = utils.get_filename(FLAGS.label_data_dir, FLAGS.file_prefix,
                                  shard_id, FLAGS.num_shards)
    tf.logging.info('processing files: {}'.format(str(filename)))
    dst = utils.get_dst_from_filename(filename, FLAGS.data_type,
                                      FLAGS.total_replicas, FLAGS.worker_id)

  if raw_data:
    return dst
  dst = dst.apply(
      tf.data.experimental.map_and_batch(
          functools.partial(preprocess), batch_size=batch_size,
          num_parallel_batches=16, drop_remainder=False))
  dst = dst.map(functools.partial(set_shapes, batch_size))
  dst = dst.prefetch(tf.data.experimental.AUTOTUNE)
  return dst


def predict_on_dataset(estimator, worker_image_num):
  if not worker_image_num:
    return 0, []
  global shard_id
  start_time = time.time()
  cnt = 0
  predict_result_list = []
  for i, result in enumerate(estimator.predict(
      get_input_fn,
      yield_single_examples=True,
      checkpoint_path=FLAGS.predict_ckpt_path)):

    classes = result['probabilities'].argmax()
    top_1_prob = result['probabilities'].max()
    new_result = {
        'probabilities': result['probabilities'].tolist(),
        'label': classes,
        'prob': top_1_prob,
    }
    predict_result_list += [
        new_result
    ]
    if i % 100 == 0:
      elp_time = (time.time() - start_time) / 3600
      tf.logging.info(
          'prediction finished sample {:d}, expected sample number {:d}'.format(
              i, worker_image_num))
      tf.logging.info(
          'elpased time: {:.2f} h, remaining time: {:.2f} h'.format(
              elp_time, elp_time / (i + 1) * (worker_image_num - i - 1)))
    cnt += 1
    if cnt >= worker_image_num:
      break
  print(cnt, worker_image_num, len(predict_result_list), '\n' * 5)
  return cnt, predict_result_list

def get_num_image():
  info_file = os.path.join(
      FLAGS.info_dir,
      'info-%.5d-of-%.5d-%.5d.txt' % (
          shard_id, FLAGS.num_shards, FLAGS.worker_id))
  print(info_file + '\n' * 10)
  if not FLAGS.reassign_label and tf.gfile.Exists(info_file):
    with tf.gfile.Open(info_file) as inf:
      info = json.load(inf)
    worker_image_num = info['image_num']
    tf.logging.info('\n\n\nloaded worker image num')
  else:
    tf.logging.info(
        '\n\n\ngetting worker image num since %s does not exist', info_file)
    dst = get_input_fn({'batch_size': 1}, raw_data=True)
    worker_image_num = 0
    tf.gfile.MakeDirs(FLAGS.info_dir)
    for _ in utils.iterate_through_dataset(dst):
      worker_image_num += 1
      if worker_image_num % 100 == 0:
        tf.logging.info('image num %d', worker_image_num)
    if not FLAGS.reassign_label:
      with tf.gfile.Open(info_file, 'w') as ouf:
        info = {
            'image_num': worker_image_num,
        }
        json.dump(info, ouf)
  tf.logging.info('worker image num: %d', worker_image_num)
  return worker_image_num


def run_prediction(estimator):
  global shard_id
  shard_id_list = FLAGS.shard_id.split(',')
  for cur_shard_id in shard_id_list:
    shard_id = int(cur_shard_id)

    worker_image_num = get_num_image()
    cnt, predict_result_list = predict_on_dataset(
        estimator, worker_image_num)
    tf.logging.info('predicted on %d images', cnt)
    assert cnt == worker_image_num, (cnt, worker_image_num)

    tf.gfile.MakeDirs(FLAGS.output_dir)
    if FLAGS.reassign_label:
      sample_dir = os.path.join(FLAGS.output_dir, 'samples')
      uid_list = utils.get_uid_list()
      for uid in uid_list:
        tf.gfile.MakeDirs(os.path.join(sample_dir, uid))

      image_bytes_placeholder = tf.placeholder(dtype=tf.string)
      decoded_image = utils.decode_raw_image(image_bytes_placeholder)

      raw_dst = get_input_fn({'batch_size': 1}, raw_data=True)
      raw_iter = raw_dst.make_initializable_iterator()
      raw_elem = raw_iter.get_next()

      filename = utils.get_reassign_filename(
          FLAGS.label_data_dir, FLAGS.file_prefix,
          shard_id, FLAGS.num_shards, FLAGS.worker_id)
      record_writer = tf.python_io.TFRecordWriter(os.path.join(
          FLAGS.output_dir, os.path.basename(filename)))
      sample_prob = 30000. / (worker_image_num * FLAGS.num_shards)
      with tf.Session() as sess:
        sess.run(raw_iter.initializer)
        for i in range(worker_image_num):
          features = sess.run(raw_elem)
          encoded_image = features['image/encoded']
          features = {}
          label = predict_result_list[i]['label']
          prob = predict_result_list[i]['prob']
          features['image/encoded'] = utils.bytes_feature(encoded_image)
          features['prob'] = utils.float_feature(prob)
          features['label'] = utils.int64_feature(label)
          features['probabilities'] = utils.float_feature(predict_result_list[i]['probabilities'])
          example = tf.train.Example(features=tf.train.Features(feature=features))
          record_writer.write(example.SerializeToString())
          if np.random.random() < sample_prob:
            uid = uid_list[label]
            filename = os.path.join(
                sample_dir, uid, 'image_{:d}_{:d}_{:.2f}.jpeg'.format(
                    shard_id, i, prob))
            tf.logging.info('saving {:s}'.format(filename))
            image = sess.run(
                decoded_image,
                feed_dict={image_bytes_placeholder: encoded_image}
            )
            utils.save_pic(image, filename)

      record_writer.close()
    else:
      filename = 'train-info-%.5d-of-%.5d-%.5d' % (
          shard_id, FLAGS.num_shards, FLAGS.worker_id)
      writer = tf.python_io.TFRecordWriter(
          os.path.join(FLAGS.output_dir, filename))
      for result in predict_result_list:
        features = {}
        features['probabilities'] = utils.float_feature(result['probabilities'])
        features['classes'] = utils.int64_feature(result['label'])

        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
      writer.close()
