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
'''Train Noisy Student.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time
import math
import functools
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

import efficientnet_builder
import data_input
import utils
import task_info
import predict_label
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator


FLAGS = flags.FLAGS

FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'

# Experiment configs
flags.DEFINE_string(
    'task_name', default='imagenet', help='imagenet or svhn')

flags.DEFINE_string(
    'label_data_dir',
    default=None,
    help=('The directory where the labeled data is stored.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_bool(
    'init_model', default=False,
    help='whether to initialize the student')

flags.DEFINE_string(
    'init_model_path', default=None,
    help='initialize the student from checkpoint')

flags.DEFINE_string(
    'model_name',
    default='efficientnet-b0',
    help='The model name among existing configurations.')

flags.DEFINE_string(
    'mode', default='train_and_eval',
    help='One of {train_and_eval, train, eval}.')

flags.DEFINE_integer(
    'train_steps', default=109474,
    help='The number of steps to use for training. 350 epochs on ImageNet.')

flags.DEFINE_integer(
    'input_image_size', default=None,
    help='Input image size: it depends on specific model name.')

flags.DEFINE_float(
    'train_ratio', default=1.0,
    help=('The train_steps and decay steps are multiplied by train_ratio.'
          'When train_ratio > 1, training is going to take longer.'))

flags.DEFINE_integer(
    'train_batch_size', default=4096, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=8, help='Batch size for evaluation.')


# Cloud TPU Cluster Resolvers
flags.DEFINE_bool(
    'use_tpu', default=True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

flags.DEFINE_string(
    'master', default=None,
    help='not used')

flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Model specific flags
flags.DEFINE_integer(
    'num_train_images', default=None, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=None, help='Size of validation data set.')

flags.DEFINE_integer(
    'num_test_images', default=None, help='Size of test data set.')

flags.DEFINE_integer(
    'steps_per_eval', default=3000,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_bool(
    'skip_host_call', default=False,
    help=('Skip the host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))

flags.DEFINE_integer(
    'iterations_per_loop', default=1000,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_string(
    'data_format', default='channels_last',
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))

flags.DEFINE_integer(
    'num_label_classes', default=1000, help='Number of classes, at least 2')

flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Use TPU double transpose optimization')

flags.DEFINE_bool(
    'use_bfloat16',
    default=True,
    help=('Whether to use bfloat16 as activation for training.'))

flags.DEFINE_float(
    'base_learning_rate',
    default=0.016,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'moving_average_decay', default=0.9999,
    help=('Moving average decay rate.'))

flags.DEFINE_float(
    'weight_decay', default=1e-5,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_float(
    'dropout_rate', default=None,
    help=('Dropout rate for the final output layer.'))

flags.DEFINE_float(
    'stochastic_depth_rate', default=None,
    help=('Drop connect rate for the network.'))

flags.DEFINE_integer('log_step_count_steps', 64, 'The number of steps at '
                     'which the global step information is logged.')

flags.DEFINE_bool(
    'use_cache', default=True, help=('Enable cache for training input.'))

flags.DEFINE_float(
    'depth_coefficient', default=None,
    help=('Depth coefficient for scaling number of layers.'))

flags.DEFINE_float(
    'width_coefficient', default=None,
    help=('Width coefficient for scaling channel size.'))

flags.DEFINE_integer('debug', 0, '')

flags.DEFINE_string(
    'unlabel_data_dir', default='', help='unlabeled data dir')

flags.DEFINE_float(
    'unlabel_ratio', default=0,
    help='batch size of unlabeled data: unlabel_ratio * train_batch_size')

flags.DEFINE_float(
    'teacher_softmax_temp', default=-1,
    help=('The softmax temperature when teacher computes the predicted distribution.'
          '-1 means to use an one-hot distribution'))

flags.DEFINE_integer(
    'train_last_step_num', -1,
    ('Used for finetuning. Only train for train_last_step_num out of the '
     'total train_steps'))

flags.DEFINE_string(
    'teacher_model_name', default=None,
    help='the model_name of the teacher model')

flags.DEFINE_string(
    'teacher_model_path', default=None,
    help='teacher model checkpoint path')

flags.DEFINE_string(
    'augment_name', default=None,
    help='None: normal cropping and flipping. v1: RandAugment')

flags.DEFINE_bool(
    'remove_aug', False,
    help='Whether to use center crop for augmentation')

flags.DEFINE_integer(
    'save_checkpoints_steps', default=1000,
    help='Batch size for training.')

flags.DEFINE_integer(
    'fix_layer_num', default=-1,
    help='Fix the first fix_layer_num layers when fintuning')

flags.DEFINE_integer(
    'randaug_mag', default=27, help='randaugment magnitude')

flags.DEFINE_integer(
    'randaug_layer', default=2, help='number of ops in randaugment')

flags.DEFINE_integer(
    'num_shards_per_group', default=-1,
    help='Tpu specific batch norm hyperparameters')

flags.DEFINE_float(
    'label_data_sample_prob', default=1,
    help=('Tpu specific hyperparameter. On Tpu, there should be at least one '
          'labeled image on each core. When we want to use a train_batch_size '
          'smaller than the num_tpu_cores, we set this hyperparameter to mask '
          'out some labeled images in the loss function. '))

flags.DEFINE_integer(
    'num_tpu_cores', default=None, help='not used')

flags.DEFINE_string(
    'unl_aug', 'default', 'augmentation for unlabeled data.')

flags.DEFINE_bool(
    'cutout_op', default=True, help='use cutout in RandAugment')

flags.DEFINE_bool(
    'small_image_model', default=False, help='whether the image size is 32x32')

flags.DEFINE_float(
    'final_base_lr', default=None, help='final learning rate.')

flags.DEFINE_integer(
    'num_train_shards', default=None, help='Number of training shards to use.')

flags.DEFINE_integer(
    'keep_checkpoint_max', default=5, help='Number of checkpoints to keep.')


def _scaffold_fn(restore_vars_dict):
  max_to_keep = FLAGS.keep_checkpoint_max
  saver = tf.train.Saver(restore_vars_dict, max_to_keep=max_to_keep)
  return tf.train.Scaffold(saver=saver)


def cross_entropy(target_prob, logits, return_mean=False):
  ce_loss = tf.reduce_sum(target_prob * (-tf.nn.log_softmax(logits)), -1)
  if return_mean:
    ce_loss = tf.reduce_mean(ce_loss, 0)
  return ce_loss


def model_fn(features, mode, params):
  '''The model_fn to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images.
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  '''
  def preprocess_image(image):
    # In most cases, the default data format NCHW instead of NHWC should be
    # used for a significant performance boost on GPU. NHWC should be used
    # only if the network needs to be run on CPU since the pooling operations
    # are only supported on NHWC. TPU uses XLA compiler to figure out best layout.
    if FLAGS.data_format == 'channels_first':
      assert not FLAGS.transpose_input    # channels_first only for GPU
      image = tf.transpose(image, [0, 3, 1, 2])

    if FLAGS.transpose_input and mode == tf.estimator.ModeKeys.TRAIN:
      image = tf.transpose(image, [3, 0, 1, 2])  # HWCN to NHWC
    return image

  def normalize_image(image):
    # Normalize the image to zero mean and unit variance.
    if FLAGS.data_format == 'channels_first':
      stats_shape = [3, 1, 1]
    else:
      stats_shape = [1, 1, 3]
    mean, std = task_info.get_mean_std(FLAGS.task_name)
    image -= tf.constant(mean, shape=stats_shape, dtype=image.dtype)
    image /= tf.constant(std, shape=stats_shape, dtype=image.dtype)
    return image

  image = features['image']
  image = preprocess_image(image)

  image_shape = image.get_shape().as_list()
  tf.logging.info('image shape: {}'.format(image_shape))
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  if mode != tf.estimator.ModeKeys.PREDICT:
    labels = features['label']
  else:
    labels = None

  # If necessary, in the model_fn, use params['batch_size'] instead the batch
  # size flags (--train_batch_size or --eval_batch_size).
  batch_size = params['batch_size']   # pylint: disable=unused-variable

  if FLAGS.unlabel_ratio and is_training:
    unl_bsz = features['unl_probs'].shape[0]
  else:
    unl_bsz = 0

  lab_bsz = image.shape[0] - unl_bsz
  assert lab_bsz == batch_size

  metric_dict = {}
  global_step = tf.train.get_global_step()

  has_moving_average_decay = (FLAGS.moving_average_decay > 0)
  # This is essential, if using a keras-derived model.
  tf.keras.backend.set_learning_phase(is_training)
  tf.logging.info('Using open-source implementation.')
  override_params = {}
  if FLAGS.dropout_rate is not None:
    override_params['dropout_rate'] = FLAGS.dropout_rate
  if FLAGS.stochastic_depth_rate is not None:
    override_params['stochastic_depth_rate'] = FLAGS.stochastic_depth_rate
  if FLAGS.data_format:
    override_params['data_format'] = FLAGS.data_format
  if FLAGS.num_label_classes:
    override_params['num_classes'] = FLAGS.num_label_classes
  if FLAGS.depth_coefficient:
    override_params['depth_coefficient'] = FLAGS.depth_coefficient
  if FLAGS.width_coefficient:
    override_params['width_coefficient'] = FLAGS.width_coefficient

  def build_model(scope=None, reuse=tf.AUTO_REUSE, model_name=None,
                  model_is_training=None, input_image=None, use_adv_bn=False, is_teacher=False):
    model_name = model_name or FLAGS.model_name
    if model_is_training is None:
      model_is_training = is_training
    if input_image is None:
      input_image = image
    input_image = normalize_image(input_image)

    scope_model_name = model_name

    if scope:
      scope = scope + '/'
    else:
      scope = ''
    with tf.variable_scope(scope + scope_model_name, reuse=reuse):
      if model_name.startswith('efficientnet'):
        logits, _ = efficientnet_builder.build_model(
            input_image,
            model_name=model_name,
            training=model_is_training,
            override_params=override_params,
            model_dir=FLAGS.model_dir,
            use_adv_bn=use_adv_bn,
            is_teacher=is_teacher)
      else:
        assert False, 'model {} not implemented'.format(model_name)
    return logits

  if params['use_bfloat16']:
    with tf.tpu.bfloat16_scope():
      logits = tf.cast(build_model(), tf.float32)
  else:
    logits = build_model()


  if FLAGS.teacher_model_name:
    teacher_image = preprocess_image(features['teacher_image'])
    if params['use_bfloat16']:
      with tf.tpu.bfloat16_scope():
        teacher_logits = tf.cast(build_model(
            scope='teacher_model',
            model_name=FLAGS.teacher_model_name,
            model_is_training=False,
            input_image=teacher_image,
            is_teacher=True), tf.float32)
    else:
      teacher_logits = build_model(
          scope='teacher_model',
          model_name=FLAGS.teacher_model_name,
          model_is_training=False,
          input_image=teacher_image,
          is_teacher=True)
    teacher_logits = tf.stop_gradient(teacher_logits)
    if FLAGS.teacher_softmax_temp != -1:
      teacher_prob = tf.nn.softmax(teacher_logits / FLAGS.teacher_softmax_temp)
    else:
      teacher_prob = None
      teacher_one_hot_pred = tf.argmax(
          teacher_logits, axis=1, output_type=labels.dtype)

  if mode == tf.estimator.ModeKeys.PREDICT:
    if has_moving_average_decay:
      ema = tf.train.ExponentialMovingAverage(
          decay=FLAGS.moving_average_decay)
      ema_vars = utils.get_all_variable()
      restore_vars_dict = ema.variables_to_restore(ema_vars)
      tf.logging.info('restored variables:\n%s',
                      json.dumps(sorted(restore_vars_dict.keys()), indent=4))

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    return tf.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions=predictions,
        scaffold_fn=functools.partial(
            _scaffold_fn,
            restore_vars_dict=restore_vars_dict) if has_moving_average_decay else None
    )

  if has_moving_average_decay:
    ema_step = global_step
    ema = tf.train.ExponentialMovingAverage(
        decay=FLAGS.moving_average_decay, num_updates=ema_step)
    ema_vars = utils.get_all_variable()

  lab_labels = labels[:lab_bsz]
  lab_logits = logits[:lab_bsz]
  lab_pred = tf.argmax(lab_logits, axis=-1, output_type=labels.dtype)
  lab_prob = tf.nn.softmax(lab_logits)
  lab_acc = tf.to_float(tf.equal(lab_pred, lab_labels))
  metric_dict['lab/acc'] = tf.reduce_mean(lab_acc)
  metric_dict['lab/pred_prob'] = tf.reduce_mean(
      tf.reduce_max(lab_prob, axis=-1)
  )
  one_hot_labels = tf.one_hot(lab_labels, FLAGS.num_label_classes)

  if FLAGS.unlabel_ratio:
    unl_labels = labels[lab_bsz:]
    unl_logits = logits[lab_bsz:]
    unl_pred = tf.argmax(unl_logits, axis=-1, output_type=labels.dtype)
    unl_prob = tf.nn.softmax(unl_logits)
    unl_acc = tf.to_float(tf.equal(unl_pred, unl_labels))
    metric_dict['unl/acc_to_dump'] = tf.reduce_mean(unl_acc)
    metric_dict['unl/pred_prob'] = tf.reduce_mean(
        tf.reduce_max(unl_prob, axis=-1)
    )

  # compute lab_loss
  one_hot_labels = tf.one_hot(lab_labels, FLAGS.num_label_classes)
  lab_loss = tf.losses.softmax_cross_entropy(
      logits=lab_logits,
      onehot_labels=one_hot_labels,
      label_smoothing=FLAGS.label_smoothing,
      reduction=tf.losses.Reduction.NONE)
  if FLAGS.label_data_sample_prob != 1:
    # mask out part of the labeled data
    random_mask = tf.floor(
        FLAGS.label_data_sample_prob + tf.random_uniform(
            tf.shape(lab_loss), dtype=lab_loss.dtype))
    lab_loss = tf.reduce_mean(lab_loss * random_mask)
  else:
    lab_loss = tf.reduce_mean(lab_loss)
  metric_dict['lab/loss'] = lab_loss

  if FLAGS.unlabel_ratio:
    if FLAGS.teacher_softmax_temp == -1:  # Hard labels
      # Get one-hot labels
      if FLAGS.teacher_model_name:
        ext_teacher_pred = teacher_one_hot_pred[lab_bsz:]
        one_hot_labels = tf.one_hot(ext_teacher_pred, FLAGS.num_label_classes)
      else:
        one_hot_labels = tf.one_hot(unl_labels, FLAGS.num_label_classes)
      # Compute cross entropy
      unl_loss = tf.losses.softmax_cross_entropy(
          logits=unl_logits,
          onehot_labels=one_hot_labels,
          label_smoothing=FLAGS.label_smoothing)
    else:  # Soft labels
      # Get teacher prob
      if FLAGS.teacher_model_name:
        unl_teacher_prob = teacher_prob[lab_bsz:]
      else:
        scaled_prob = tf.pow(
            features['unl_probs'], 1 / FLAGS.teacher_softmax_temp)
        unl_teacher_prob = scaled_prob / tf.reduce_sum(scaled_prob, axis=-1,
                                                       keepdims=True)
      metric_dict['unl/target_prob'] = tf.reduce_mean(
          tf.reduce_max(unl_teacher_prob, axis=-1))
      unl_loss = cross_entropy(unl_teacher_prob, unl_logits, return_mean=True)

    metric_dict['ext/loss'] = unl_loss
  else:
    unl_loss = 0

  real_lab_bsz = tf.to_float(lab_bsz) * FLAGS.label_data_sample_prob
  real_unl_bsz = batch_size * FLAGS.label_data_sample_prob * FLAGS.unlabel_ratio
  data_loss = lab_loss * real_lab_bsz + unl_loss * real_unl_bsz
  data_loss = data_loss / real_lab_bsz

  # Add weight decay to the loss for non-batch-normalization variables.
  loss = data_loss + FLAGS.weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])
  metric_dict['train/data_loss'] = data_loss
  metric_dict['train/loss'] = loss

  host_call = None
  restore_vars_dict = None

  if is_training:
    # Compute the current epoch and associated learning rate from global_step.
    current_epoch = (
        tf.cast(global_step, tf.float32) / params['steps_per_epoch'])
    real_train_batch_size = FLAGS.train_batch_size
    real_train_batch_size *= FLAGS.label_data_sample_prob
    scaled_lr = FLAGS.base_learning_rate * (real_train_batch_size / 256.0)
    if FLAGS.final_base_lr:
      # total number of training epochs
      total_epochs = FLAGS.train_steps * FLAGS.train_batch_size * 1. / FLAGS.num_train_images - 5
      decay_times = math.log(FLAGS.final_base_lr / FLAGS.base_learning_rate) / math.log(0.97)
      decay_epochs = total_epochs / decay_times
      tf.logging.info(
          'setting decay_epochs to {:.2f}'.format(decay_epochs) + '\n' * 3)
    else:
      decay_epochs = 2.4 * FLAGS.train_ratio
    learning_rate = utils.build_learning_rate(
        scaled_lr, global_step,
        params['steps_per_epoch'],
        decay_epochs=decay_epochs,
        start_from_step=FLAGS.train_steps - FLAGS.train_last_step_num,
        warmup_epochs=5,
    )
    metric_dict['train/lr'] = learning_rate
    metric_dict['train/epoch'] = current_epoch
    optimizer = utils.build_optimizer(learning_rate)
    if FLAGS.use_tpu:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    tvars = tf.trainable_variables()
    g_vars = []
    tvars = sorted(tvars, key=lambda var: var.name)
    for var in tvars:
      if 'teacher_model' not in var.name:
        g_vars += [var]
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step, var_list=g_vars)

    if has_moving_average_decay:
      with tf.control_dependencies([train_op]):
        train_op = ema.apply(ema_vars)

    if not FLAGS.skip_host_call:
      host_call = utils.construct_scalar_host_call(metric_dict)
    scaffold_fn = None
    if FLAGS.teacher_model_name or FLAGS.init_model:
      scaffold_fn = utils.init_from_ckpt(scaffold_fn)
  else:
    train_op = None
    if has_moving_average_decay:
      # Load moving average variables for eval.
      restore_vars_dict = ema.variables_to_restore(ema_vars)

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    scaffold_fn = functools.partial(
        _scaffold_fn,
        restore_vars_dict=restore_vars_dict) if has_moving_average_decay else None
    def metric_fn(labels, logits):
      '''Evaluation metric function. Evaluates accuracy.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      '''

      predictions = tf.argmax(logits, axis=1)
      top_1_accuracy = tf.metrics.accuracy(labels, predictions)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      top_5_accuracy = tf.metrics.mean(in_top_5)

      result_dict = {
          'top_1_accuracy': top_1_accuracy,
          'top_5_accuracy': top_5_accuracy,
      }

      return result_dict

    eval_metrics = (metric_fn, [labels, logits])

  num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
  tf.logging.info('number of trainable parameters: {}'.format(num_params))

  return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics,
      scaffold_fn=scaffold_fn)


def get_ent(logits):
  log_prob = tf.nn.log_softmax(logits, axis=-1)
  prob = tf.exp(log_prob)
  ent = tf.reduce_sum(-prob * log_prob, axis=-1)
  return ent


def main(unused_argv):
  if FLAGS.task_name == 'svhn':
    FLAGS.input_image_size = 32
    FLAGS.small_image_model = True
    FLAGS.num_label_classes = 10
  if FLAGS.num_train_images is None:
    FLAGS.num_train_images = task_info.get_num_train_images(FLAGS.task_name)
  if FLAGS.num_eval_images is None:
    FLAGS.num_eval_images = task_info.get_num_eval_images(FLAGS.task_name)
  if FLAGS.num_test_images is None and FLAGS.task_name != 'imagenet':
    FLAGS.num_test_images = task_info.get_num_test_images(FLAGS.task_name)

  steps_per_epoch = (FLAGS.num_train_images /
                     (FLAGS.train_batch_size * FLAGS.label_data_sample_prob))
  if FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval':
    tf.gfile.MakeDirs(FLAGS.model_dir)
    flags_dict = tf.app.flags.FLAGS.flag_values_dict()
    with tf.gfile.Open(os.path.join(FLAGS.model_dir, 'FLAGS.json'), 'w') as ouf:
      json.dump(flags_dict, ouf)
  input_image_size = FLAGS.input_image_size
  if not input_image_size:
    _, _, input_image_size, _ = efficientnet_builder.efficientnet_params(
        FLAGS.model_name)
    FLAGS.input_image_size = input_image_size
  if FLAGS.train_last_step_num == -1:
    FLAGS.train_last_step_num = FLAGS.train_steps
  if FLAGS.train_ratio != 1:
    FLAGS.train_last_step_num *= FLAGS.train_ratio
    FLAGS.train_steps *= FLAGS.train_ratio
    FLAGS.train_last_step_num = int(FLAGS.train_last_step_num)
    FLAGS.train_steps = int(FLAGS.train_steps)

  if (FLAGS.tpu or FLAGS.use_tpu) and not FLAGS.master:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project)
  else:
    tpu_cluster_resolver = None

  if FLAGS.use_tpu:
    tpu_config = tf.estimator.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
        .PER_HOST_V2)
  else:
    tpu_config = tf.estimator.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
        .PER_HOST_V2)
  config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=max(FLAGS.save_checkpoints_steps, FLAGS.iterations_per_loop),
      log_step_count_steps=FLAGS.log_step_count_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      session_config=tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True))),
      tpu_config=tpu_config)  # pylint: disable=line-too-long
  # Initializes model parameters.
  params = dict(
      steps_per_epoch=steps_per_epoch,
      use_bfloat16=FLAGS.use_bfloat16)
  est = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=8,
      params=params)

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  if FLAGS.label_data_dir == FAKE_DATA_DIR:
    tf.logging.info('Using fake dataset.')
  else:
    tf.logging.info('Using dataset: %s', FLAGS.label_data_dir)

  train_data = data_input.DataInput(
      is_training=True,
      data_dir=FLAGS.label_data_dir,
      transpose_input=FLAGS.transpose_input,
      cache=FLAGS.use_cache,
      image_size=input_image_size,
      use_bfloat16=FLAGS.use_bfloat16)
  if FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval':
    current_step = estimator._load_global_step_from_checkpoint_dir(
        FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long

    tf.logging.info(
        'Training for %d steps (%.2f epochs in total). Current'
        ' step %d.', FLAGS.train_last_step_num,
        FLAGS.train_last_step_num / params['steps_per_epoch'],
        current_step)

    start_timestamp = time.time()  # This time will include compilation time

    if FLAGS.mode == 'train':
      est.train(
          input_fn=train_data.input_fn,
          max_steps=FLAGS.train_last_step_num,
          hooks=[])
  elif FLAGS.mode == 'eval':
    input_fn_mapping = {}
    for subset in ['dev', 'test']:
      input_fn_mapping[subset] = data_input.DataInput(
          is_training=False,
          data_dir=FLAGS.label_data_dir,
          transpose_input=FLAGS.transpose_input,
          cache=False,
          image_size=input_image_size,
          use_bfloat16=FLAGS.use_bfloat16,
          subset=subset).input_fn
      if subset == 'dev':
        num_images = FLAGS.num_eval_images
      else:
        num_images = FLAGS.num_test_images
      eval_results = est.evaluate(
          input_fn=input_fn_mapping[subset],
          steps=num_images // FLAGS.eval_batch_size)
      tf.logging.info('%s, results: %s', subset, eval_results)
  elif FLAGS.mode == 'predict':
      predict_label.run_prediction(est)
  else:
      assert False


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
