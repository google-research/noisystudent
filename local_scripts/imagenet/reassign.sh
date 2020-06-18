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
#!/bin/bash


data_root=./data/imagenet/predict/
info_dir=${data_root}/info
data_dir=${data_root}/filter_data/data
reassign_data_dir=${data_root}/reassign
teacher_ckpt=

# Prediction jobs for different shards can run in parallel if you have multiple GPUs/TPUs
for shard_id in {0..127}
do
    python main.py \
        --model_name=efficientnet-b0 \
        --use_tpu=False \
        --use_bfloat16=False \
        --task_name=svhn \
        --mode=predict \
        --predict_ckpt_path=${teacher_ckpt} \
        --worker_id=0 \
        --num_shards=128 \
        --shard_id=${shard_id} \
        --file_prefix=train \
        --label_data_dir=${data_dir} \
        --data_type=tfrecord \
        --info_dir=${info_dir} \
        --output_dir=${reassign_data_dir} \
        --reassign_label=True
done
