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
import collections
import json
import copy
import os
import time
import numpy as np
import tensorflow as tf
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', '', '')

flags.DEFINE_string('prediction_dir', '', '')

flags.DEFINE_string('info_dir', '', '')

flags.DEFINE_string('prelim_stats_dir', '', '')

flags.DEFINE_string('output_dir', '', '')

flags.DEFINE_integer(
    'num_shards', default=128, help='')

flags.DEFINE_integer(
    'only_use_num_shards', default=-1, help='')

flags.DEFINE_integer(
    'shard_id', default=0, help='')

flags.DEFINE_integer(
    'num_image', default=1300, help='')

flags.DEFINE_integer(
    'total_replicas', default=1, help='')

flags.DEFINE_integer(
    'total_label_replicas', default=-1, help='')

flags.DEFINE_integer(
    'task', default=-1, help='')

flags.DEFINE_integer(
    'debug', default=0, help='')

flags.DEFINE_float(
    'min_threshold', default=0.0, help='')

flags.DEFINE_float(
    'max_prob', default=2, help='sometimes the probability can be greater than 1 due to floating point.')

flags.DEFINE_integer(
    'num_label_classes', default=1000, help='')

flags.DEFINE_integer(
    'upsample', default=1, help='')

flags.DEFINE_integer(
    'only_get_stats', default=0, help='')

flags.DEFINE_string('file_prefix', 'train', '')

flags.DEFINE_string(
    'data_type', default='tfrecord', help='')

flags.DEFINE_integer(
    'use_top', default=1, help='')

flags.DEFINE_bool(
    'eval_imagenet_p', default=False, help='')

flags.DEFINE_bool(
    'use_all', default=False, help='')


def preprocess_jft(features):
  encoded_image = features['image/encoded']
  image = utils.decode_raw_image(encoded_image)
  encoded_image = tf.image.encode_jpeg(
      image,
      format='rgb', quality=100)
  features['image/encoded'] = encoded_image
  return features


def input_dataset(worker_id):
  filename = utils.get_filename(FLAGS.input_dir, FLAGS.file_prefix,
                                FLAGS.shard_id, FLAGS.num_shards)
  dst = utils.get_dst_from_filename(filename, FLAGS.data_type,
                                    FLAGS.total_label_replicas, worker_id)

  dst = dst.apply(
      tf.data.experimental.map_and_batch(
          preprocess_jft if FLAGS.data_type == 'sstable' else lambda x: x, batch_size=1,
          num_parallel_batches=16, drop_remainder=False))
  dst = dst.prefetch(tf.data.experimental.AUTOTUNE)
  return dst


def get_worker_id_list():
  if FLAGS.debug == 1:
    worker_id_list = [0]
  else:
    if FLAGS.task != -1:
      num_label_replica_per_worker = FLAGS.total_label_replicas // FLAGS.total_replicas
      worker_id_list = list(range(
          FLAGS.task * num_label_replica_per_worker,
          (FLAGS.task + 1) * num_label_replica_per_worker))
      tf.logging.info('worker_id_list {:s}'.format(str(worker_id_list)))
    else:
      worker_id_list = list(range(FLAGS.total_label_replicas))
  return worker_id_list


def get_label_to_image_idx():
  tf.logging.info('\n\ngetting label to image idx')
  label_to_image_idx = {}
  num_image_for_worker = {}
  for worker_id in get_worker_id_list():
    with tf.gfile.Open(
        os.path.join(
            FLAGS.info_dir,
            'info-%.5d-of-%.5d-%.5d.txt' % (
                FLAGS.shard_id, FLAGS.num_shards, worker_id
            ))) as inf:
      info = json.load(inf)
      image_num = info['image_num']
    num_image_for_worker[worker_id] = image_num
    if image_num == 0:
      continue
    label_dst = utils.label_dataset(
        worker_id,
        FLAGS.prediction_dir, FLAGS.shard_id, FLAGS.num_shards)
    iter = label_dst.make_initializable_iterator()
    elem = iter.get_next()
    cnt = 0

    with tf.Session() as sess:
      sess.run(iter.initializer)
      for j in range(image_num):
        features = sess.run(elem)
        label_arr = features['classes']
        prob_arr = features['probabilities']
        for i in range(label_arr.shape[0]):
          label = label_arr[i]
          prob = prob_arr[i][label]
          if label not in label_to_image_idx:
            label_to_image_idx[label] = []
          label_to_image_idx[label] += [{
              'worker_id': worker_id,
              'idx': cnt,
              'prob': prob,
              'probabilities': prob_arr[i].tolist(),
          }]
          cnt += 1
    assert cnt == image_num
  return label_to_image_idx, num_image_for_worker


def get_keep_image_idx(label_to_image_idx, selected_threshold, uid_list):
  tf.logging.info('\n\ngetting keep image idx')
  stats_dir = os.path.join(
      FLAGS.output_dir,
      'stats')
  tf.gfile.MakeDirs(stats_dir)

  keep_idx = {}
  for i in label_to_image_idx:
    label_to_image_idx[i] = sorted(label_to_image_idx[i],
                                   key=lambda x: -x['prob'])
    k = 0
    uid = uid_list[i]
    while k < len(label_to_image_idx[i]):
      if (label_to_image_idx[i][k]['prob'] >= selected_threshold[uid][0]
          and label_to_image_idx[i][k]['prob'] <= FLAGS.max_prob):
        if FLAGS.use_all:
          include_copy = 1
        else:
          include_copy = FLAGS.num_image / selected_threshold[uid][1]
          if not FLAGS.upsample:
            include_copy = min(include_copy, 1)
        prob = include_copy - int(include_copy)
        include_copy = int(include_copy) + int(np.random.random() < prob)
        if include_copy:
          info = label_to_image_idx[i][k]
          worker_id = info['worker_id']
          print('include_copy', include_copy, FLAGS.num_image, selected_threshold[uid][1], '\n\n\n')
          if worker_id not in keep_idx:
            keep_idx[worker_id] = {}
          keep_idx[worker_id][info['idx']] = [i, info['prob'], include_copy, info['probabilities']]
      k += 1
  counts = collections.defaultdict(int)
  total_keep_example = 0
  for worker_id in keep_idx:
    for label, prob, include_copy, _ in keep_idx[worker_id].values():
      counts[uid_list[label]] += include_copy
      total_keep_example += 1

  tf.logging.info('counts: {:s}'.format(json.dumps(counts, indent=4)))
  return keep_idx, total_keep_example, counts


def filter_image_by_idx(
    keep_idx,
    uid_list,
    total_keep_example,
    num_image_for_worker):
  sample_prob = 30000. / (FLAGS.num_image * 1000)
  image_list = []
  np.random.seed(12345)

  def get_image_list(features):
    dump_features = {}
    prob = keep_idx[worker_id][cnt][1]
    label = keep_idx[worker_id][cnt][0]
    include_copy = keep_idx[worker_id][cnt][2]
    image_bytes = features['image/encoded'][0]
    dump_features['image/encoded'] = utils.bytes_feature(image_bytes)
    dump_features['prob'] = utils.float_feature(prob)
    dump_features['probabilities'] = utils.float_feature(keep_idx[worker_id][cnt][3])
    dump_features['label'] = utils.int64_feature(label)
    example = tf.train.Example(features=tf.train.Features(feature=dump_features))
    cur_image_list = []
    for j in range(include_copy):
      image_info = {
          'example': example,
          'label': label,
          'prob': prob,
          'image_bytes': image_bytes,
          'cnt': cnt,
      }
      cur_image_list += [image_info]
    return cur_image_list

  def flush(sess):
    tf.logging.info('saving images')
    np.random.shuffle(image_list)
    for image_info in image_list:
      image_bytes = image_info['image_bytes']
      prob = image_info['prob']
      label = image_info['label']
      example = image_info['example']
      cnt = image_info['cnt']
      record_writer.write(example.SerializeToString())
      if np.random.random() < sample_prob:
        uid = uid_list[label]
        filename = os.path.join(
            sample_dir, uid, 'image_{:d}_{:d}_{:d}_{:.2f}.jpeg'.format(
                FLAGS.shard_id, FLAGS.task, cnt, prob))
        tf.logging.info('saving {:s}'.format(filename))
        image = sess.run(decoded_image,
                        feed_dict={image_bytes_placeholder: image_bytes}
                        )
        utils.save_pic(image, filename)

        tf.logging.info(
            '{:d}/{:d} images saved, elapsed time: {:.2f} h'.format(
                num_picked_images, total_keep_example,
                (time.time() - start_time) / 3600))

  tf.logging.info('\n\nfilter image by index')
  num_picked_images = 0
  sample_dir = os.path.join(FLAGS.output_dir, 'samples')
  data_dir = os.path.join(FLAGS.output_dir, 'data')
  for uid in uid_list:
    tf.gfile.MakeDirs(os.path.join(sample_dir, uid))
  tf.gfile.MakeDirs(data_dir)

  image_bytes_placeholder = tf.placeholder(dtype=tf.string)
  decoded_image = utils.decode_raw_image(image_bytes_placeholder)

  total_cnt = 0
  start_time = time.time()
  image_list = []
  if len(keep_idx) == 0:
    return
  record_writer = tf.python_io.TFRecordWriter(
      os.path.join(data_dir, 'train-%d-%.5d-of-%.5d' % (
          FLAGS.task, FLAGS.shard_id, FLAGS.num_shards)))
  for worker_id in get_worker_id_list():
    tf.logging.info('worker_id: {:d}, elapsed time: {:.2f} h'.format(
        worker_id, (time.time() - start_time) / 3600.))
    dst = input_dataset(worker_id)
    iter = dst.make_initializable_iterator()
    elem = iter.get_next()
    cnt = 0
    hit_samples = {}
    with tf.Session() as sess:
      sess.run(iter.initializer)
      for i in range(num_image_for_worker[worker_id]):
        features = sess.run(elem)
        key = 'image/encoded'
        # encoded_image_arr = features['image/encoded']
        # assert encoded_image_arr.shape[0] == 1
        # for j in range(encoded_image_arr.shape[0]):
        for j in range(features[key].shape[0]):
          if worker_id in keep_idx and cnt in keep_idx[worker_id]:
            num_picked_images += 1
            # image_list += get_image_list(encoded_image_arr[j])
            image_list += get_image_list(features)
            hit_samples[cnt] = 1
          if total_cnt % 1000 == 0:
            elapsed_time = (time.time() - start_time) / 3600
            total_image = num_image_for_worker[worker_id]
            tf.logging.info(
                'scanning idx {:d} of {:d} images, {:d}/{:d} images saved, elapsed time: {:.2f} h, remaining time {:.2f} h'.format(
                    total_cnt, total_image,
                    num_picked_images, total_keep_example,
                    elapsed_time,
                    elapsed_time / (total_cnt + 1) * (total_image - total_cnt)
                )
            )
          cnt += 1
          total_cnt += 1
        if len(image_list) >= 10000:
          flush(sess)
          image_list = []
      try:
        sess.run(elem)
        assert False, "count isn't right"
      except tf.errors.OutOfRangeError:
        tf.logging.info('count is right')
    assert cnt == num_image_for_worker[worker_id], (cnt, num_image_for_worker[worker_id])
    for idx in keep_idx[worker_id]:
      if idx not in hit_samples:
        tf.logging.info('\n\nnot hit, %d %d', worker_id, idx)

  assert num_picked_images == total_keep_example
  if len(image_list):
    with tf.Session() as sess:
      flush(sess)
    image_list = []
  record_writer.close()


def is_master_job():
  return FLAGS.shard_id == 0 and (FLAGS.task == -1 or FLAGS.task == 0)


def get_total_counts(uid_list, prelim_stats_dir, prob_threshold):
  if FLAGS.only_use_num_shards != -1:
    num_shards = FLAGS.only_use_num_shards
  else:
    num_shards = FLAGS.num_shards
  to_read_filenames = []
  for i in range(num_shards):
    for j in range(FLAGS.total_replicas):
      if FLAGS.debug == 1 and (i != FLAGS.shard_id or j != FLAGS.task):
        continue
      prelim_stats_filename = os.path.join(
          prelim_stats_dir,
          'prelim_stats_%.5d_%d.json' % (i, j))
      to_read_filenames += [prelim_stats_filename]
  total_counts = {}
  total_counts_sum = {}
  for uid in uid_list:
    total_counts[uid] = []
    total_counts_sum[uid] = []
    for threshold in prob_threshold:
      total_counts[uid] += [[threshold, 0]]
      total_counts_sum[uid] += [[threshold, 0]]
  tf.logging.info('reading prelim stats')

  while len(to_read_filenames):
    new_to_read_filenames = []
    for filename in to_read_filenames:
      completed, counts = load_json(filename)
      if completed:
        for uid in counts:
          for k in range(len(prob_threshold)):
            total_counts[uid][k][1] += counts[uid][k][1]
        tf.logging.info('finished reading prelim stats for {:s}'.format(filename))
      else:
        new_to_read_filenames += [filename]
        tf.logging.info('not ready: {:s}'.format(filename))
    to_read_filenames = new_to_read_filenames
  return total_counts, total_counts_sum


def get_threshold(label_to_image_idx, uid_list, prob_threshold):
  tf.logging.info('\n\ngetting threshold')
  threshold_stats = {}
  prelim_stats_dir = FLAGS.prelim_stats_dir
  prelim_stats_filename = os.path.join(prelim_stats_dir, 'prelim_stats_%.5d_%d.json' % (FLAGS.shard_id, FLAGS.task))
  if not load_json(prelim_stats_filename)[0]:
    tf.gfile.MakeDirs(prelim_stats_dir)
    for i in label_to_image_idx:
      label_to_image_idx[i] = sorted(label_to_image_idx[i],
                                     key=lambda x: -x['prob'])
      num_samples = []
      n = len(label_to_image_idx[i])
      start_idx = 0
      cur_sample_idx = 0
      for j in reversed(range(len(prob_threshold))):
        while cur_sample_idx < n and label_to_image_idx[i][cur_sample_idx]['prob'] >= prob_threshold[j]:
          cur_sample_idx += 1
        num_samples += [(prob_threshold[j], cur_sample_idx - start_idx)]
        start_idx = cur_sample_idx
      threshold_stats[uid_list[i]] = copy.deepcopy(list(reversed(num_samples)))

    with tf.gfile.Open(
        prelim_stats_filename, 'w') as ouf:
      json.dump(threshold_stats, ouf)
      tf.logging.info('threshold_stats: {:s}'.format(json.dumps(threshold_stats, indent=4)))
  if is_master_job():
    total_counts_file = os.path.join(prelim_stats_dir, 'total_counts.json')
    if not tf.gfile.Exists(total_counts_file):
      total_counts, total_counts_sum = get_total_counts(
          uid_list, prelim_stats_dir, prob_threshold)
      for uid in uid_list:
        for i in range(len(prob_threshold) - 1, -1, -1):
          if i < len(prob_threshold) - 1:
            total_counts_sum[uid][i][1] = total_counts_sum[uid][i + 1][1] + total_counts[uid][i][1]
          else:
            total_counts_sum[uid][i][1] = total_counts[uid][i][1]

      total_counts_sum_file = os.path.join(prelim_stats_dir, 'total_counts_sum.json')
      with tf.gfile.Open(total_counts_sum_file, 'w') as ouf:
        json.dump(total_counts_sum, ouf)
      with tf.gfile.Open(total_counts_file, 'w') as ouf:
        json.dump(total_counts, ouf)
    else:
      with tf.gfile.Open(total_counts_file) as inf:
        total_counts = json.load(inf)

    tf.gfile.MakeDirs(FLAGS.output_dir)
    threshold_file = os.path.join(FLAGS.output_dir, 'threshold.json')
    if not tf.gfile.Exists(threshold_file):
      selected_threshold = {}
      num_image_across_cat = 0
      for uid in uid_list:
        threshold_idx = -1
        total_image = 0
        for i in range(len(prob_threshold) - 1, -1, -1):
          if prob_threshold[i] < FLAGS.max_prob and prob_threshold[i] >= FLAGS.min_threshold:
            total_image += total_counts[uid][i][1]
            if not FLAGS.use_all:
              if FLAGS.use_top and total_image >= FLAGS.num_image:
                threshold_idx = i
                break
            if prob_threshold[i] == FLAGS.min_threshold:
              threshold_idx = i
              break
        assert threshold_idx != -1
        if not FLAGS.use_all:
          if total_image < FLAGS.num_image:
            assert prob_threshold[threshold_idx] == FLAGS.min_threshold
            tf.logging.info(
                'warning: too few images, {:s} only has {:d} images while we expect {:d} images, upsampling, threshold {:.3f}'.format(
                    uid, total_image, FLAGS.num_image, prob_threshold[threshold_idx]))
          else:
            tf.logging.info('warning: too many images, {:s} has {:d} images while we expect {:d} images, down sampling, threshold {:.3f}'.format(
                uid, total_image, FLAGS.num_image, prob_threshold[threshold_idx]))
        selected_threshold[uid] = (
            prob_threshold[threshold_idx],
            total_image)
        num_image_across_cat += min(total_image, FLAGS.num_image)
      with tf.gfile.Open(threshold_file, 'w') as ouf:
        json.dump(selected_threshold, ouf)

      image_across_cat_filename = os.path.join(FLAGS.output_dir, 'num_image_across_cat.json')
      with tf.gfile.Open(image_across_cat_filename, 'w') as ouf:
        json.dump({'num_image_acorss_cat': num_image_across_cat}, ouf)
    else:
      with tf.gfile.Open(threshold_file) as inf:
        selected_threshold = json.load(inf)
  else:
    if FLAGS.only_get_stats:
      return None
    threshold_file = os.path.join(FLAGS.output_dir, 'threshold.json')
    while not tf.gfile.Exists(threshold_file):
      tf.logging.info('waiting for the threshold file')
      time.sleep(300) # sleep 5 min
    selected_threshold = None
    while True:
      try:
        with tf.gfile.Open(threshold_file) as inf:
          selected_threshold = json.load(inf)
        break
      except:
        pass
  return selected_threshold


def load_json(filename):
  if tf.gfile.Exists(filename):
    counts = None
    try:
      with tf.gfile.Open(filename) as inf:
        counts = json.load(inf)
      return (True, counts)
    except:
      tf.logging.info('having error loading {:s}, not exist'.format(
          filename))
  return (False, None)


def read_stats():
  total_counts = collections.defaultdict(int)
  filename_list = []
  stats_dir = os.path.join(FLAGS.output_dir, 'stats')

  if FLAGS.only_use_num_shards != -1:
    num_shards = FLAGS.only_use_num_shards
  else:
    num_shards = FLAGS.num_shards
  for i in range(num_shards):
    for j in range(FLAGS.total_replicas):
      filename = os.path.join(stats_dir, 'stats_%.5d_%d.json' % (i, j))
      filename_list += [filename]
  if FLAGS.debug == 1:
    filename_list = [os.path.join(
        stats_dir, 'stats_%.5d_%d.json' % (FLAGS.shard_id, FLAGS.task))]
  while len(filename_list):
    new_filename_list = []
    for filename in filename_list:
      if load_json(filename)[0]:
        counts = None
        while True:
          try:
            with tf.gfile.Open(filename) as inf:
              counts = json.load(inf)
            break
          except:
            tf.logging.info('having error loading {:s}, retrying'.format(
                filename))
            pass
        for uid in counts:
          total_counts[uid] += counts[uid]
      else:
        new_filename_list += [filename]
    filename_list = new_filename_list
    tf.logging.info('waiting for: {:s}'.format(' '.join(filename_list)))
  count_pairs = total_counts.items()
  count_pairs = sorted(count_pairs, key=lambda x: -x[1])
  num_images_all_label = 0
  for key, value in count_pairs:
    num_images_all_label += value
  final_stats = {
        'cat_count': total_counts,
        'cat_sorted_pairs': count_pairs,
        'total_cnt': num_images_all_label
    }
  with tf.gfile.Open(
      os.path.join(FLAGS.output_dir, 'stats', 'final_stats.json'), 'w') as ouf:
    json.dump(final_stats, ouf)
  tf.logging.info(json.dumps(final_stats, indent=4))


def get_label_replicas():
  # infer number of replicas from data
  FLAGS.total_label_replicas = 1
  while True:
    filename = os.path.join(
        FLAGS.prediction_dir,
        'train-info-%.5d-of-%.5d-%.5d' % (
            0, FLAGS.num_shards, FLAGS.total_label_replicas - 1))
    if tf.gfile.Exists(filename):
      FLAGS.total_label_replicas *= 2
    else:
      break
  FLAGS.total_label_replicas = FLAGS.total_label_replicas // 2
  tf.logging.info('total_label_replicas {:d}'.format(FLAGS.total_label_replicas))
  assert FLAGS.total_label_replicas > 0


def main(argv):
  stats_dir = os.path.join(FLAGS.output_dir, 'stats')
  stats_filename = os.path.join(stats_dir, 'stats_%.5d_%d.json' % (FLAGS.shard_id, FLAGS.task))
  if load_json(stats_filename)[0]:
    if is_master_job():
      read_stats()
    tf.logging.info('stats already finished, returning')
    return
  prelim_stats_filename = os.path.join(
      FLAGS.prelim_stats_dir,
      'prelim_stats_%.5d_%d.json' % (FLAGS.shard_id, FLAGS.task))
  completed, _ = load_json(prelim_stats_filename)
  if FLAGS.only_get_stats and completed and not is_master_job():
    return
  get_label_replicas()
  assert FLAGS.total_label_replicas == FLAGS.total_replicas
  # must be sorted
  prob_threshold = []
  # 0 to 0.99
  for i in range(0, 101):
    prob = i / 100.
    prob_threshold += [prob]

  uid_list = utils.get_uid_list()
  print(len(uid_list))
  print("\n" * 10)
  label_to_image_idx, num_image_for_worker = get_label_to_image_idx()
  selected_threshold = get_threshold(
      label_to_image_idx, uid_list, prob_threshold)
  if FLAGS.only_get_stats:
    return

  keep_idx, total_keep_example, counts = get_keep_image_idx(
      label_to_image_idx, selected_threshold, uid_list)
  filter_image_by_idx(keep_idx, uid_list, total_keep_example, num_image_for_worker)

  with tf.gfile.Open(
      os.path.join(stats_dir, 'stats_%.5d_%d.json' % (FLAGS.shard_id, FLAGS.task)),
      'w') as ouf:
    json.dump(counts, ouf)
  if is_master_job():
    read_stats()


if __name__ == '__main__':
  app.run(main)

