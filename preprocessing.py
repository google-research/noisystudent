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
'''ImageNet preprocessing.'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import randaugment

IMAGE_SIZE = 224
CROP_PADDING = 32

FLAGS = tf.app.flags.FLAGS

def _distorted_bounding_box_crop(image,
                                 bbox,
                                 min_object_covered=0.1,
                                 aspect_ratio_range=(0.75, 1.33),
                                 area_range=(0.05, 1.0),
                                 max_attempts=100,
                                 scope=None):
  '''Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    cropped image `Tensor`
  '''
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    shape = tf.shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    image = tf.slice(image, bbox_begin, bbox_size)
    return image


def _at_least_x_are_equal(a, b, x):
  '''At least `x` of `a` and `b` `Tensors` are equal.'''
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _random_crop(ori_image, image_size):
  '''Make a random crop of image_size.'''
  original_shape = tf.shape(ori_image)
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = _distorted_bounding_box_crop(
      ori_image,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      scope=None)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: center_crop(ori_image, image_size),
      lambda: tf.image.resize_bicubic([image],  # pylint: disable=g-long-lambda
                                      [image_size, image_size])[0])

  return image


def center_crop(image, image_size):
  '''Crops to center of image with padding then scales image_size.'''
  if FLAGS.small_image_model:
    return image
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2

  bbox_begin = tf.stack([
      offset_height, offset_width,
      tf.constant(0, dtype=tf.int32)
  ])
  bbox_size = tf.stack([
      padded_center_crop_size, padded_center_crop_size,
      tf.constant(-1, dtype=tf.int32)
  ])
  bbox_begin = tf.Print(bbox_begin, [bbox_begin], message='bbox_begin \n\n\n', summarize=1000)
  bbox_size = tf.Print(bbox_size, [bbox_size], message='bbox_size \n\n\n', summarize=1000)
  image = tf.slice(image, bbox_begin, bbox_size)

  image = tf.image.resize_bicubic([image], [image_size, image_size])[0]

  return image


def _flip(image):
  '''Random horizontal image flip.'''
  image = tf.image.random_flip_left_right(image)
  return image


def small_image_crop(image):
  amount = 4
  pad_inp = tf.pad(image,
                   tf.constant([[amount, amount],
                                [amount, amount],
                                [0, 0]]),
                   'REFLECT')
  cropped_data = tf.random_crop(pad_inp, tf.shape(image))
  return cropped_data


def _preprocess_for_train(image, use_bfloat16, image_size=IMAGE_SIZE,
                          augment_name=None, randaug_mag=None,
                          randaug_layer=None):
  '''Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  '''
  image = _random_crop(image, image_size)
  image = _flip(image)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.cast(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)

  if augment_name:
    input_image_type = image.dtype
    tf.logging.info('Apply augment {}'.format(augment_name))
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, dtype=tf.uint8)
    if augment_name == 'v1':
      image = randaugment.distort_image_with_randaugment(
          image, randaug_layer or FLAGS.randaug_layer,
          randaug_mag or FLAGS.randaug_mag)
    else:
      assert False
    image = tf.cast(image, dtype=input_image_type)

  return image


def _cifar10_preprocess_for_train(image, use_bfloat16, image_size=IMAGE_SIZE,
                                  augment_name=None, randaug_mag=None,
                                  randaug_layer=None):
  '''Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  '''
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.cast(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)

  if augment_name:
    input_image_type = image.dtype
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, dtype=tf.uint8)
    if augment_name == 'v1':
      image = randaugment.distort_image_with_randaugment(
          image, randaug_layer or FLAGS.randaug_layer,
          randaug_mag or FLAGS.randaug_mag,
          cutout_const=image_size // 8, translate_const=image_size // 8,
          )
      image = randaugment.cutout(image, pad_size=image_size // 4, replace=128)
    image = tf.cast(image, dtype=input_image_type)

  assert image_size == 32
  image = small_image_crop(image)
  if FLAGS.task_name != 'svhn':
    image = _flip(image)

  return image


def _preprocess_for_eval(input_tensor, use_bfloat16, image_size=IMAGE_SIZE):
  '''Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  '''
  image = center_crop(input_tensor, image_size)


  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.cast(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
  return image


def preprocess_image(input_tensor,
                     is_training=False,
                     use_bfloat16=False,
                     image_size=IMAGE_SIZE,
                     is_image_bytes=True,
                     augment_name=None,
                     randaug_mag=None,
                     randaug_layer=None):
  '''Preprocesses the given image.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor` with value range of [0, 255].
  '''
  if is_image_bytes:
    image = tf.image.decode_jpeg(input_tensor, channels=3)
  else:
    image = input_tensor
  if is_training:
    if FLAGS.small_image_model:
      return _cifar10_preprocess_for_train(
          image, use_bfloat16, image_size,
          augment_name, randaug_mag, randaug_layer)
    else:
      return _preprocess_for_train(
          image, use_bfloat16, image_size,
          augment_name, randaug_mag, randaug_layer)
  else:
    return _preprocess_for_eval(image, use_bfloat16, image_size)
