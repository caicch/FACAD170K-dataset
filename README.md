# ACC
Towards Attribute-Controlled Fashion Image Captioning [paper](https://dl.acm.org/doi/abs/10.1145/3671000).
[paper](https://ieeexplore.ieee.org/document/9897417)

# Training
Run python train_ACC.py for fashion image only captioning

Run python train_ACC_adaptive.py for attribute controlled captioning

## Dataset:
The FACAD170K [dataset](https://drive.google.com/file/d/1JyNN3eNyDuvyAcsxvTsh-CIBy3OkMTHr/view?usp=share_link) .json file can be downloaded from this link. The fashion images can be downloaded using the URL link in the JSON file.

Run python create_input_files17k.py to create .hdf5 files

## Acknowledgments:
The code of this paper is implemented based on [zyj0021200/simpleImageCaptionZoo](https://github.com/zyj0021200/simpleImageCaptionZoo), [xuewyang/Fashion_Captioning](https://github.com/xuewyang/Fashion_Captioning) and [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).
