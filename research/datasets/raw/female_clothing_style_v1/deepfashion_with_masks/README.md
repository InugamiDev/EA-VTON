---
license: apache-2.0
tags:
- code
pretty_name: fashion clothes segmentation
dataset_info:
  features:
  - name: images
    dtype: image
  - name: gender
    dtype: string
  - name: pose
    dtype: string
  - name: cloth_type
    dtype: string
  - name: pid
    dtype: string
  - name: caption
    dtype: string
  - name: mask
    dtype: image
  - name: mask_overlay
    dtype: image
  splits:
  - name: train
    num_bytes: 1821511821.448
    num_examples: 40658
  download_size: 1449380618
  dataset_size: 1821511821.448
---


# Dataset 

Dataset name is deepfashion2 datasest, the dataset is in raw form with annotations, for original dataset repo. see `https://github.com/switchablenorms/DeepFashion2` 

This dataset is just the extracted version of original deepfashion2 dataset and can be used for training **Controlnet Model**.