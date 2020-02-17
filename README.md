# NoisyStudent

## Overview

NoisyStudent is a semi-supervised learning method which achieves 88.4% top-1 accuracy on ImageNet (SOTA) and surprising gains on robustness and adversarial benchmarks.
NoisyStudent is based on the self-training framework and trained with 4 simple steps:
1. Train a classifier on labeled data (teacher).
2. Infer labels on a much larger unlabeled dataset.
3. Train a larger classifier on the combined set, adding noise (noisy student).
4. Go to step 2, with student as teacher

For ImageNet checkpoints trained by NoisyStudent, please refer to the [EfficientNet github](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet#2-using-pretrained-efficientnet-checkpoints).

## SVHN Experiments
Our ImageNet experiments requires using JFT-300M which is not publicly
available. We will release the full code for ImageNet trained on a public
dataset as unlabeled data in a few weeks.

Here we show an implementation of NoisyStudent on SVHN, which boosts the performance of a
supervised model from 97.9% accuracy to 98.6% accuracy.

```shell
# Download and preprocess SVHN. Download the teacher model trained on labeled data with accuracy 97.9.
bash local_scripts/prepro.sh

# Training & Eval (expected accuracy: 98.6 +- 0.1)
bash local_scripts/run_svhn.sh
```

You can also use the colab script [noisystudent_svhn.ipynb](https://github.com/google-research/noisystudent/blob/master/noisystudent_svhn.ipynb) to try the method on free Colab GPUs. 

## Relevant Papers 

NoisyStudent
```
@article{xie2019self,
  title={Self-training with Noisy Student improves ImageNet classification},
  author={Xie, Qizhe and Hovy, Eduard and Luong, Minh-Thang and Le, Quoc V},
  journal={arXiv preprint arXiv:1911.04252},
  year={2019}
}
```

EfficientNet: Our backbone model 
```
@article{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc V},
  journal={arXiv preprint arXiv:1905.11946},
  year={2019}
}
```

RandAugment: An effective data augmentation noise 
```
@article{cubuk2019randaugment,
  title={RandAugment: Practical data augmentation with no separate search},
  author={Cubuk, Ekin D and Zoph, Barret and Shlens, Jonathon and Le, Quoc V},
  journal={arXiv preprint arXiv:1909.13719},
  year={2019}
}
```

This is not an officially supported Google product.
