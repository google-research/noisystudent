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

data_root=./data/svhn/predict/
info_dir=${data_root}/info
prelim_stats_dir=${data_root}/prelim_stats
output_dir=${data_root}/filter_data
prediction_dir=${data_root}/predict_label
only_get_stats=$1

# Predict
for ((shard_id=127;shard_id>=0;shard_id--));
do
    python filter_unlabel.py \
        --input_dir=./data/svhn/proc/unlabeled \
        --file_prefix=extra \
        --info_dir=${info_dir} \
        --prediction_dir=${prediction_dir} \
        --prelim_stats_dir=${prelim_stats_dir} \
        --output_dir=${output_dir} \
        --num_shards=128 \
        --shard_id=${shard_id} \
        --min_threshold=0.3 \
        --num_label_classes=10 \
        --task=0 \
        --num_image=53113 \
        --only_get_stats=${only_get_stats}
done
