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
data_dir=data

python proc_svhn.py \
  --task_name=svhn \
  --raw_data_dir=data/svhn/raw \
  --output_dir=data/svhn/proc

mkdir ckpt
wget https://storage.googleapis.com/noisystudent/ckpts/svhn/teacher_ckpt.tar.gz -O ckpt/teacher_ckpt.tar.gz
cd ckpt && tar xzvf teacher_ckpt.tar.gz
