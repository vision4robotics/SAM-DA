# SAM-DA: UAV Tracks Anything at Night with SAM-Powered Domain Adaptation

**[Vision4robotics](https://vision4robotics.github.io/)**

Liangliang Yao†, Haobo Zuo†, Guangze Zheng†, Changhong Fu*, Jia Pan
† Equal contribution. * Corresponding author.

## Abstract
Domain adaptation (DA) has demonstrated significant promise for real-time nighttime unmanned aerial vehicle (UAV)
tracking. However, the state-of-the-art (SOTA) DA still lacks the potential object with accurate pixel-level location and boundary to
generate the high-quality target domain training sample. This key issue constrains the transfer learning of the real-time daytime SOTA
trackers for challenging nighttime UAV tracking. Recently, the notable Segment Anything Model (SAM) has achieved remarkable
zero-shot generalization ability to discover abundant potential objects due to its huge data-driven training approach. To solve the
aforementioned issue, this work proposes a novel SAM-powered DA framework for real-time nighttime UAV tracking, i.e., SAM-DA.
Specifically, an innovative SAM-powered target domain training sample swelling is designed to determine enormous high-quality target
domain training samples from every single raw nighttime image. This novel one-to-many method significantly expands the high-quality
target domain training sample for DA. Comprehensive experiments on extensive nighttime UAV videos prove the robustness and
domain adaptability of SAM-DA for nighttime UAV tracking. Especially, compared to the SOTA DA, SAM-DA can achieve better
performance with fewer raw nighttime images, i.e., the fewer-better training. This economized training approach facilitates the quick
validation and deployment of algorithms for UAVs.
## Framework
![Framework](https://github.com/vision4robotics/SAM-DA/blob/main/assets/framework.png)
## Visualization of one-to-many generation
![One-to-many generation](https://github.com/vision4robotics/SAM-DA/blob/main/assets/one-to-many_generation.png)
## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```
bash install.sh
```
## Getting started
### Data prepration

SAM-powered target domain training sample swelling on NAT2021-*train*.

1. Download the proposed [NAT2021-*train*](https://vision4robotics.github.io/NAT2021/) and put it in "./". 
2. Download a [model checkpoint](#model-checkpoints) and put it in "./sam/snapshot/". 
3. run the script
```
bash swell.sh
```
> ⚠️ warning: A huge passport is necessary.

### <a name=""></a> Model checkpoints

| Training data | Model | Source 1 | Source 2 | Source 3 |
|  ----  | ----  |  ----  | ----  | ----  |
|  SAM-NAT-B (base) | `sam-da-track-b` |  [Baidu](https://pan.baidu.com/s/1c_hlOxnyv-4bGyHzymlpRA?pwd=6prk)  | [Google](https://drive.google.com/file/d/1yiUTYQty52cAacmtGuqdgb53CnIe2l1W/view?usp=sharing)  | [Hugging face](https://huggingface.co/George-Zhuang/SAM-DA/resolve/main/sam-da-track-b.pth)  |
|  SAM-NAT-S (small) | `sam-da-track-s` |  [Baidu](https://pan.baidu.com/s/1kUCZMXgRZs1HgD6gtx9hrQ?pwd=a48s)  | [Google](https://drive.google.com/file/d/1fxShaJ67XB1nMnE9ioQg7_LXYQBd6snI/view?usp=sharing)  | [Hugging face](https://huggingface.co/George-Zhuang/SAM-DA/resolve/main/sam-da-track-s.pth)   |
|  SAM-NAT-T (tiny) | `sam-da-track-t` |  [Baidu](https://pan.baidu.com/s/11LrJwoz--AO3UzXavwa_GA?pwd=5qkj)  | [Google](https://drive.google.com/file/d/10Y9td4CJt4DqbcvCCLVUkCEx67MyilYC/view?usp=sharing)  | [Hugging face](https://huggingface.co/George-Zhuang/SAM-DA/resolve/main/sam-da-track-t.pth)   | 
|  SAM-NAT-N (nano) | `sam-da-track-n` |  [Baidu](https://pan.baidu.com/s/1h1OROv17qINJmGU7zR4LTA?pwd=ujag)  | [Google](https://drive.google.com/file/d/1xR5i2XqHoDRoBEXH7O4ko5JZok0EPHTF/view?usp=sharing)  | [Hugging face](https://huggingface.co/George-Zhuang/SAM-DA/resolve/main/sam-da-track-n.pth)  | 

## <a name="Performance"></a> Fewer data, better performance
**SAM-DA** aims to reach the few-better training for quick deployment of night-time tracking methods for UAVs
* **SAM-DA** enriches the training samples and attributes (ambient intensity) of target domain.
<img src="/assets/ai_dist.png" width = "600"  />

* **SAM-DA** can achieve better performance on fewer raw images with quicker training.
  
| Method | Training data | Images | Propotion | Training$^1$ | AUC (NUT-L) |
|  ----  | ----  |  :----:  | :----:  | :----:  |  :----:  | ----  |  ----  | ----  | ----  |
|  Baseline  | NAT2021-*train*  | 276k | 100%  | 12h  | 0.377  | 
|  **SAM-DA**  | SAM-NAT-N | 28k | 10%  | **2.4h**  | 0.411  | 
|  **SAM-DA**  | SAM-NAT-T | 92k | 33%  | 4h  | 0.414  | 
|  **SAM-DA**  | SAM-NAT-S | 138k | 50%  | 6h  | 0.419  | 
|  **SAM-DA**  | SAM-NAT-B | 276k | 100%  | 12h  | **0.430**  | 

<img src="/assets/suc_data.png" width = "600"  />

> 1. Training duration on a single A100 GPU.



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
We sincerely thank the contribution of following repos: [SiamBAN](https://github.com/hqucv/siamban), and [UDAT](https://github.com/vision4robotics/UDAT).

# Contact
If you have any questions, please contact Liangliang Yao at [1951018@tongji.edu.cn](mailto:1951018@tongji.edu.cn) or Changhong Fu at [changhongfu@tongji.edu.cn](mailto:changhongfu@tongji.edu.cn).
