# SAM-DA: UAV Tracks Anything at Night with SAM-Powered Domain Adaptation
This is the official code for the paper "SAM-DA: UAV Tracks Anything at Night with SAM-Powered Domain Adaptation".

**[Vision4robotics](https://vision4robotics.github.io/)**

Liangliang Yao†, Haobo Zuo†, Guangze Zheng†, Changhong Fu*, Jia Pan
† Equal contribution. * Corresponding author.

## 🏗️ Framework
![Framework](https://github.com/vision4robotics/SAM-DA/blob/main/assets/framework.png)
## 👀 Visualization of SAM-DA
![One-to-many generation](https://github.com/vision4robotics/SAM-DA/blob/main/assets/one-to-many_generation.png)

## 📅 Todo
* Video demos for more night scenes with SAM-DA.
* Test with your own videos.
* Interactive demo on your video with your instruction.
## 🛠️ Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. 

Install Segment Anything:

```
bash install.sh
```
## 😀 Getting started
### Test SAM-DA
* Download a **model checkpoint** below and put it in `./tracker/BAN/snapshot`.

  | Training data | Model | Source 1 | Source 2 | Source 3 |
  |  ----  | ----  |  ----  | ----  | ----  |
  |  SAM-NAT-B (base, default) | `sam-da-track-b` |  [Baidu](https://pan.baidu.com/s/1c_hlOxnyv-4bGyHzymlpRA?pwd=6prk)  | [Google](https://drive.google.com/file/d/1yiUTYQty52cAacmtGuqdgb53CnIe2l1W/view?usp=sharing)  | [Hugging face](https://huggingface.co/George-Zhuang/SAM-DA/resolve/main/sam-da-track-b.pth)  |
  |  SAM-NAT-S (small) | `sam-da-track-s` |  [Baidu](https://pan.baidu.com/s/1kUCZMXgRZs1HgD6gtx9hrQ?pwd=a48s)  | [Google](https://drive.google.com/file/d/1fxShaJ67XB1nMnE9ioQg7_LXYQBd6snI/view?usp=sharing)  | [Hugging face](https://huggingface.co/George-Zhuang/SAM-DA/resolve/main/sam-da-track-s.pth)   |
  |  SAM-NAT-T (tiny) | `sam-da-track-t` |  [Baidu](https://pan.baidu.com/s/11LrJwoz--AO3UzXavwa_GA?pwd=5qkj)  | [Google](https://drive.google.com/file/d/10Y9td4CJt4DqbcvCCLVUkCEx67MyilYC/view?usp=sharing)  | [Hugging face](https://huggingface.co/George-Zhuang/SAM-DA/resolve/main/sam-da-track-t.pth)   | 
  |  SAM-NAT-N (nano) | `sam-da-track-n` |  [Baidu](https://pan.baidu.com/s/1h1OROv17qINJmGU7zR4LTA?pwd=ujag)  | [Google](https://drive.google.com/file/d/1xR5i2XqHoDRoBEXH7O4ko5JZok0EPHTF/view?usp=sharing)  | [Hugging face](https://huggingface.co/George-Zhuang/SAM-DA/resolve/main/sam-da-track-n.pth)  | 

* Download **[NUT-L]()** dataset and put it in `./tracker/BAN/test_dataset`.
* Test and evalute on NUT-L with `default` settings. 

```bash
cd tracker/BAN
python tools/test.py 
python tools/eval.py
```

* (optional) Test with other checkpoints (e.g., `sam-da-track-s`):
```bash
cd tracker/BAN
python tools/test.py --snapshot sam-da-track-s
python tools/eval.py
```

### Train SAM-DA
* SAM-powered target domain training sample swelling on NAT2021-*train*.

  1. Download original nighttime dataset [NAT2021-*train*](https://vision4robotics.github.io/NAT2021/) and put it in `./tracker/BAN/train_dataset/sam_nat`. 
  2. Sam-powered target domain training sample swelling!
   ```
   bash swell.sh
   ```
    > ⚠️ warning: A huge passport is necessary for about ~G data.
* Prepare daytime dataset [VID] and [GOT-10K].
  1. Download [VID](https://image-net.org/challenges/LSVRC/2017/) and [GOT-10K](http://got-10k.aitestunion.com/downloads) and put them in `./tracker/BAN/train_dataset/vid` and `./tracker/BAN/train_dataset/got10k`, respectively.
  2. Crop data following the instruction for [VID](./tracker/BAN/train_dataset/vid/readme.md) and [GOT-10k](./tracker/BAN/train_dataset/got10k/readme.md).

* Train `sam-da-track-b` (default) and other models. 
  ```bash
  cd tracker/BAN
  python tools/train.py --model sam-da-track-b
  ```




## <a name="Performance"></a> 🌈 Fewer data, better performance
**SAM-DA** aims to reach the few-better training for quick deployment of night-time tracking methods for UAVs.
* **SAM-DA** enriches the training samples and attributes (ambient intensity) of target domain.
<img src="/assets/ai_dist.png" width = "600"  />

* **SAM-DA** can achieve better performance on fewer raw images with quicker training.
    
  | Method | Training data | Images | Propotion | Training | AUC (NUT-L) |
  |  ----  | ----  |  :----:  | :----:  | :----:  |  :----:  | 
  |  Baseline  | NAT2021-*train*  | 276k | 100%  | 12h  | 0.377  | 
  |  **SAM-DA**  | SAM-NAT-N | 28k | 10%  | **2.4h**  | 0.411  | 
  |  **SAM-DA**  | SAM-NAT-T | 92k | 33%  | 4h  | 0.414  | 
  |  **SAM-DA**  | SAM-NAT-S | 138k | 50%  | 6h  | 0.419  | 
  |  **SAM-DA**  | SAM-NAT-B | 276k | 100%  | 12h  | **0.430**  | 

  For more details, please refer to the [paper](https://arxiv.org/abs/2307.01024).

  <img src="/assets/suc_data.png" width = "600"  />

> Training duration on a single A100 GPU.



# License
The model is licensed under the Apache License 2.0 license.

# Citations
Please consider citing the related paper(s) in your publications if it helps your research.
```
@article{Yao2023SAMDA,
  title={{SAM-DA: UAV Tracks Anything at Night with SAM-Powered Domain Adaptation}},
  author={Yao, Liangliang and Zuo, Haobo and Zheng, Guangze and Fu, Changhong and Pan, Jia},
  journal={arXiv preprint arXiv:2307.01024},
  year={2023}
  pages={1-12}
}
@article{kirillov2023segment,
  title={{Segment Anything}},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
  pages={1-30}
}
@Inproceedings{Ye2022CVPR,
title={{Unsupervised Domain Adaptation for Nighttime Aerial Tracking}},
author={Ye, Junjie and Fu, Changhong and Zheng, Guangze and Paudel, Danda Pani and Chen, Guang},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2022},
pages={1-10}
}
```
# Acknowledgments
We sincerely thank the contribution of following repos: [SAM](https://github.com/facebookresearch/segment-anything), [SiamBAN](https://github.com/hqucv/siamban), and [UDAT](https://github.com/vision4robotics/UDAT).

# Contact
If you have any questions, please contact Liangliang Yao at [1951018@tongji.edu.cn](mailto:1951018@tongji.edu.cn) or Changhong Fu at [changhongfu@tongji.edu.cn](mailto:changhongfu@tongji.edu.cn).
