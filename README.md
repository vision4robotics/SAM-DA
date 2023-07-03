# SAM-DA: UAV Tracks Anything at Night with SAM-Powered Domain Adaptation

**[Vision4robotics](https://vision4robotics.github.io/)**

Liangliang Yao†, Haobo Zuo†, Guangze Zheng†, Changhong Fu*, Jia Pan

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
validation and deployment of algorithms for UAV.
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
# SAM-powered target domain training sample swelling

1. Download the proposed [NAT2021-train set](https://vision4robotics.github.io/NAT2021/) and put it in "./". 
2. Download a [model checkpoint](#model-checkpoints) and put it in "./sam/snapshot/". 
3. run the script
```
bash swell.sh
```

Then 
## <a name="Models"></a>Model Checkpoints

Three model versions of the model are available with different backbone sizes. 
Click the links below to download the checkpoint for the corresponding model name. The default model in bold can also be instantiated with `build_sam`, as in the examples in [Getting Started](#getting-started).

* **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
* `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
* `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

