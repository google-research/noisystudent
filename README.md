# Noisy Student Training

## Overview

[Noisy Student Training](https://arxiv.org/abs/1911.04252) is a semi-supervised learning method which achieves 88.4% top-1 accuracy on ImageNet (SOTA) and surprising gains on robustness and adversarial benchmarks.
Noisy Student Training is based on the self-training framework and trained with 4 simple steps:
1. Train a classifier on labeled data (teacher).
2. Infer labels on a much larger unlabeled dataset.
3. Train a larger classifier on the combined set, adding noise (noisy student).
4. Go to step 2, with student as teacher

For ImageNet checkpoints trained by Noisy Student Training, please refer to the [EfficientNet github](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet#2-using-pretrained-efficientnet-checkpoints).

## SVHN Experiments
Here we show an implementation of Noisy Student Training on SVHN, which boosts the performance of a
supervised model from 97.9% accuracy to 98.6% accuracy.

```shell
# Download and preprocess SVHN. Download the teacher model trained on labeled data with accuracy 97.9.
bash local_scripts/svhn/prepro.sh

# Train & Eval (expected accuracy: 98.6 +- 0.1) 
# The teacher model generates predictions on the fly in this script. To store the teacher model's prediction to save training time, see the following instructions.
bash local_scripts/svhn/run.sh
```

Instructions on running prediction on unlabeled data, filtering and balancing data and training using the stored predictions.

```shell
# Run prediction on multiple shards.
# Run predictions in parallel if you have multiple GPUs/TPUs
bash local_scripts/svhn/predict.sh

# Get statistics of different shards (parallelizable).
bash local_scripts/svhn/filter_unlabel.sh 1

# Output the filtered and balanced data (parallelizable).
bash local_scripts/svhn/filter_unlabel.sh 0

# Training & Eval the stored predictions.
bash local_scripts/svhn/run_offline.sh
```

If you get a better model, you can use the model to predict pseudo-labels on the filtered data.

```shell
# Reassign pseudo-labels.
# Run predictions in parallel if you have multiple GPUs/TPUs
bash local_scripts/svhn/reassign.sh
```

You can also use the colab script [noisystudent_svhn.ipynb](https://github.com/google-research/noisystudent/blob/master/noisystudent_svhn.ipynb) to try the method on free Colab GPUs. 

## ImageNet Experiments
Scripts used for our ImageNet experiments: 
```shell
# Train:
# See the scripts for hyperparameters for EfficientNet-B0 to B7.
# You need to fill in the label_data_dir, unlabel_data_dir, model_name, teacher_model_path in the script.
bash local_scripts/imagenet/train.sh

# Eval
bash local_scripts/imagenet/eval.sh
```

Similar scripts to run predictions on unlabeled data, filter and balance data and train using the filtered data.

```shell
# Run prediction on multiple shards.
bash local_scripts/imagenet/predict.sh

# Get statistics of different shards (parallelizable).
bash local_scripts/imagenet/filter_unlabel.sh 1

# Output the filtered and balanced data (parallelizable).
bash local_scripts/imagenet/filter_unlabel.sh 0

# Training & Eval using the filtered data.
bash local_scripts/imagenet/run_offline.sh
bash local_scripts/imagenet/eval.sh
```

Use a model to predict pseudo-labels on the filtered data:


```shell
# Reassign pseudo-labels.
# Run predictions in parallel if you have multiple GPUs/TPUs
bash local_scripts/imagenet/reassign.sh
```

## Bibtex 

```
@article{xie2019self,
  title={Self-training with Noisy Student improves ImageNet classification},
  author={Xie, Qizhe and Luong, Minh-Thang and Hovy, Eduard and Le, Quoc V},
  journal={arXiv preprint arXiv:1911.04252},
  year={2019}
}
```

This is not an officially supported Google product.
