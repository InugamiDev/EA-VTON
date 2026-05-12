---
license: apache-2.0
task_categories:
- text-to-image
- image-to-image
---
# DeepFashion MultiModal Parts2Whole

<!-- Provide a quick summary of the dataset. -->

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6375d136dee28348a9c63cbf/BvkNNQxx_DfgZFG9XsDRG.png)

## Dataset Details

### Dataset Description

This human image dataset comprising about 41,500 reference-target pairs. Each pair in this dataset includes multiple reference images, which encompass human pose images (e.g., OpenPose, Human Parsing, DensePose), various aspects of human appearance (e.g., hair, face, clothes, shoes) with their short textual labels, and a target image featuring the same individual (ID) in the same outfit but in a different pose, along with textual captions.

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://github.com/huanngzh/Parts2Whole
- **Paper:** https://arxiv.org/pdf/2404.15267

## Uses

<!-- This section describes suitable use cases for the dataset. -->
Please refer to our dataset file: https://github.com/huanngzh/Parts2Whole/blob/main/parts2whole/data/ref_trg.py.

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

We provide train and test jsonl file for indexing reference and target images. Each sample in the jsonl file contains:
```json
{
  "target_id": "target person id in the original DeepFashion-MultiModal dataset",
  "reference_id": "reference person id in the original DeepFashion-MultiModal dataset",
  "target": "the relative path of target human image",
  "caption": "text descriptions for the target human image",
  "appearance": {},
  "structure": {},
  "mask": {}
}
```

Example:
```json
{
  "target_id": "MEN-Denim-id_00000265-01_1_front",
  "reference_id": "MEN-Denim-id_00000265-01_2_side",
  "target": "images/MEN-Denim-id_00000265-01_1_front.jpg",
  "caption": "This person is wearing a short-sleeve shirt with solid color patterns. The shirt is with cotton fabric. It has a crew neckline. The pants this person wears is of short length. The pants are with cotton fabric and pure color patterns. There is a hat in his head.",
  "appearance": {
    "upper body clothes": "upper_body_clothes/MEN-Denim-id_00000265-01_2_side_rgb.jpg",
    "lower body clothes": "lower_body_clothes/MEN-Denim-id_00000265-01_2_side_rgb.jpg",
    "whole body clothes": "whole_body_clothes/MEN-Denim-id_00000265-01_2_side_rgb.jpg",
    "hair or headwear": "hair_headwear/MEN-Denim-id_00000265-01_2_side_rgb.jpg",
    "face": "face/MEN-Denim-id_00000265-01_2_side_rgb.jpg",
    "shoes": "shoes/MEN-Denim-id_00000265-01_2_side_rgb.jpg"
  },
  "mask": {
    "upper body clothes": "upper_body_clothes/MEN-Denim-id_00000265-01_2_side_mask.jpg",
    "lower body clothes": "lower_body_clothes/MEN-Denim-id_00000265-01_2_side_mask.jpg",
    "whole body clothes": "whole_body_clothes/MEN-Denim-id_00000265-01_2_side_mask.jpg",
    "hair or headwear": "hair_headwear/MEN-Denim-id_00000265-01_2_side_mask.jpg",
    "face": "face/MEN-Denim-id_00000265-01_2_side_mask.jpg",
    "shoes": "shoes/MEN-Denim-id_00000265-01_2_side_mask.jpg"
  },
  "structure": {
    "densepose": "densepose/MEN-Denim-id_00000265-01_1_front_densepose.png",
    "openpose": "openpose/MEN-Denim-id_00000265-01_1_front.png"
  }
}
```

## Dataset Creation

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

DeepFashion MultiModal dataset (a large-scale high-quality human dataset with rich multi-modal annotations):
https://github.com/yumingj/DeepFashion-MultiModal

#### Data Collection and Processing

We build a multi-modal dataset comprising about 41,500 reference-target pairs from the open-source DeepFashion-MultiModal dataset. Each pair in this newly constructed dataset includes multiple reference images, which encompass hu- man pose images (e.g., OpenPose, Human Parsing, DensePose), various aspects of human appearance (e.g., hair, face, clothes, shoes) with their short textual labels, and a target image featuring the same individual (ID) in the same outfit but in a different pose, along with textual captions.

The DeepFashion-MultiModal dataset exhibits noise in its ID data. For example, different images are tagged with the same ID but depict different individuals. To address this issue, we first cleanse the IDs by extracting facial ID features from images tagged with the same ID using InsightFace[5, 6]. Cosine similarity is then used to evaluate the similarity between image ID feature pairs to distinguish between different ID images within the same ID group. Subsequently, we utilize DWPose to generate pose images corresponding to each image. Guided by human parsing files, we crop human images into various parts. Due to the low resolution of the cropped parts, we apply Real-ESRGAN[46] to enhance the image resolution, thus obtaining clearer reference images. Textual descriptions of the original dataset are used as captions. For constructing pairs, we select images with cleaned IDs that feature the same clothes and individual but in different poses. Specifically, a pair contains multiple parts from one human image as reference images, and an image of the person in another pose as the target. Finally, we build a total of about 41,500 pairs, of which the training set is about 40,000 and the test set is about 1,500 pairs.

## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

```
@article{huang2024parts2whole,
  title={From Parts to Whole: A Unified Reference Framework for Controllable Human Image Generation},
  author={Huang, Zehuan and Fan, Hongxing and Wang, Lipeng and Sheng, Lu},
  journal={arXiv preprint arXiv:2404.15267},
  year={2024}
}
```

If you find the original dataset helps, please consider also citing:
```
@article{jiang2022text2human,
  title={Text2Human: Text-Driven Controllable Human Image Generation},
  author={Jiang, Yuming and Yang, Shuai and Qiu, Haonan and Wu, Wayne and Loy, Chen Change and Liu, Ziwei},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={4},
  articleno={162},
  pages={1--11},
  year={2022},
  publisher={ACM New York, NY, USA},
  doi={10.1145/3528223.3530104},
}
```
