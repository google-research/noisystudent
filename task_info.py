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
'''Task-specific info.'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def get_mean_std(task_name):
  '''Get mean and std.'''
  if task_name == 'imagenet':
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    stddev = [0.229 * 255, 0.224 * 255, 0.225 * 255]
  elif task_name == 'svhn':
    mean = [0.4376821 * 255, 0.4437697 * 255, 0.47280442 * 255]
    stddev = [0.19803012 * 255, 0.20101562 * 255, 0.19703614 * 255]
  else:
    assert False
  return mean, stddev


def get_num_train_images(task_name):
  if task_name == 'imagenet':
    return 1281167
  elif task_name == 'svhn':
    return 73257 - 4000
  else:
    tf.logging.info(task_name)
    assert False, task_name


def get_num_eval_images(task_name):
  if task_name == 'imagenet':
    return 50000
  elif task_name == 'svhn':
    return 4000
  else:
    tf.logging.info(task_name)
    assert False, task_name


def get_num_test_images(task_name):
  if task_name == 'svhn':
    return 26032
  else:
    tf.logging.info(task_name)
    assert False, task_name
