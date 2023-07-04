# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
from genericpath import isdir, isfile
from siamban.core.config import cfg
# from siamban.models.model_builder import ModelBuilder
# from siamban.tracker.tracker_builder import build_tracker
# visualization
from siamban.models.model_builder_v import ModelBuilder
from siamban.tracker.tracker_builder import build_tracker
from siamban.utils.bbox import get_axis_aligned_bbox
from siamban.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory

import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--dataset', type=str,default='NUT_L',
        help='datasets')
parser.add_argument('--datasest_root', default='path/to/your/dataset', type=str,      
        help='dataset root path')
parser.add_argument('--config', default=os.getcwd() + '/experiments/udatban_r50_l234/config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='sam-da-track-b', type=str,        
        help='snapshot of models to eval')
parser.add_argument('--snapshot_path', default='./tracker/BAN/snapshot/', type=str,        
        help='path to snapshot')
parser.add_argument('--threads', default=12, type=int,        
        help='number of threads')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
parser.add_argument('--gpu_id', default='not_set', type=str, 
        help="gpu id")

args = parser.parse_args()

def main():
    # load config
    cfg.merge_from_file(args.config)
    print(f'Now testing {args.snapshot}')
    params = getattr(cfg.HP_SEARCH, args.dataset)
    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}
    snapshot_path = os.path.join(args.snapshot_path, args.snapshot + '.pth')

    # create model
    model = ModelBuilder()
    # load model
    model = load_pretrain(model, snapshot_path).cuda().eval()
    # build tracker
    tracker = build_tracker(model)
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=args.dataset_root,
                                            load_img=False)

    model_name = f'{args.snapshot}' + str(args.snapshot.split('/')[-1][:-4]) + '_'+ str(hp['lr']) + '_' + str(hp['penalty_k']) + '_' + str(hp['window_lr'])
    model_path = os.path.join('results', args.dataset, model_name)
    # multi-process
    mp.set_start_method('spawn', force=True)
    threads = args.threads
    param_list = [(seq, tracker, args, hp, model_name, model_path) for seq in dataset]
    with mp.Pool(processes=threads) as pool:
        pool.starmap(run_sequence, param_list)
    print('Done')

def run_sequence(video, tracker, args, hp, model_name, model_path, num_gpu=1):
    try:
        worker_name = mp.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass
    
    toc = 0
    pred_bboxes = []
    scores = []
    track_times = []
    for idx, (img, gt_bbox) in enumerate(video):
        tic = cv2.getTickCount()
        if idx == 0:
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
            gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
            tracker.init(img, gt_bbox_)
            pred_bbox = gt_bbox_
            scores.append(None)
            if 'VOT2018-LT' == args.dataset:
                pred_bboxes.append([1])
            else:
                pred_bboxes.append(pred_bbox)
        else:
            outputs = tracker.track(img, hp)
            # visualization
            # outputs = tracker.track(img, idx, hp, video.name, model_abbr)
            pred_bbox = outputs['bbox']
            pred_bboxes.append(pred_bbox)
            scores.append(outputs['best_score'])
        toc += cv2.getTickCount() - tic
        track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
        if idx == 0:
            cv2.destroyAllWindows()
        if args.vis and idx > 0:
            gt_bbox = list(map(int, gt_bbox))
            pred_bbox = list(map(int, pred_bbox))
            cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                            (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
            cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                            (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
            cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(video.name, img)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()
    # save results
    model_path = os.path.join('results', args.dataset, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    result_path = os.path.join(model_path, '{}.txt'.format(video.name))
    with open(result_path, 'w') as f:
        for x in pred_bboxes:
            f.write(','.join([str(i) for i in x])+'\n')
    print('Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
         video.name, toc, idx / toc))

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    import sys
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()

if __name__ == '__main__':
    main()
