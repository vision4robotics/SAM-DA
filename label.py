# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import os
# import sys
# sys.path.append("./segment-anything/")
from sam.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import torch.multiprocessing as mp
import argparse
import json

parser = argparse.ArgumentParser(description='SAM segmentation')
parser.add_argument('--dataset', type=str, default='NAT2021', 
        help='datasets')

parser.add_argument(
    "--model-type",
    type=str,
    default="default",
    help="The type of model to load, in ['default', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default='./sam/snapshot/sam_vit_h_4b8939.pth', # pth path 
    # required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--output",
    type=str,
    default='./',
    # required=True,
    help="The output path.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
parser.add_argument("--dataset_root", type=str, default="./NAT2021-train/train_clip", help="The dataset to process.")

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)

def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

def main(args: argparse.Namespace) -> None:

    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    
    os.makedirs(os.path.join(args.output,"result"), exist_ok=True)

    # multi-process
    mp.set_start_method('spawn', force=True)
    
    threads = 4
    # Create json file
    for i in range(threads):
        save_base = os.path.join(args.output,"result", str(i)) 
        save_file = save_base + ".json"
        with open(save_file, 'w') as file:
            pass 
    if os.path.exists("list.txt"):
        file = open("list.txt", "r") # the sequences that has been run
        content = file.read()
        file.close()
        exist_list = content.split("\n")
    else:
        exist_list = []

    target_list = os.listdir(args.dataset_root)
    result_list = [x for x in target_list if x not in exist_list]
    print(len(result_list))

    param_list = [(sub_dir, sam, args) for sub_dir in result_list]
    with mp.Pool(processes=threads) as pool:
        pool.starmap(run_sequence, param_list)
    print('Done')
    
def run_sequence(sub_dir, sam, args, num_gpu=4):
    output_mode = "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    
    try:
        worker_name = mp.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda:"+str(gpu_id))
        _ = sam.to(device=device)
    except:
        pass
    model = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    
    file_name = os.path.basename(sub_dir).split('.')[0]
    sub_dir_path = os.path.join(args.dataset_root,sub_dir)
    img_list = sorted(os.listdir(sub_dir_path))
    bbox_list = []
    for img_file in img_list:
        img_file_path = os.path.join(sub_dir_path,img_file)
        print(f"Processing '{img_file_path}'...")
        # start_time = time.time()
        image = cv2.imread(img_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = model.generate_bbox(image)
        # end_time = time.time()
        # run_time = end_time - start_time
        # print("程序运行时间：", run_time, "秒")
        # print(bbox)
        base = os.path.basename(img_file)
        base = os.path.splitext(base)[0]
        name_box = {
            base:bbox,
        }
        bbox_list.append(name_box)

    bbox_print = {
        file_name:bbox_list
    }
    save_base = os.path.join(args.output,"result", str(worker_id)) 
    save_file = save_base + ".json"
    with open(save_file, 'a') as file:
        json.dump(bbox_print, file, indent=4, sort_keys=True)
        file.write('\n')
   
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



