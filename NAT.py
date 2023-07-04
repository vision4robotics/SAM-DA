import json
import time
from collections import defaultdict
import itertools
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class NAT2021:
    def __init__(self, annotation_file=None):
        """
        Thanks coco
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.imgs = dict(),dict(),dict()
        # self.imgToAnns = defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
            
    def createIndex(self):
        # create index
        print('creating index...')
        anns, imgs = {}, {}
        imgToAnns = defaultdict(list)
        

        dataType = list(self.dataset.keys())[0]
        json_dict = self.dataset[dataType]
        for ann in json_dict:
            id = list(ann.keys())[0]
            masks = ann[id]
            imgToAnns[id] = masks  

        print('index created!')
        self.imgToAnns = imgToAnns


if __name__ == '__main__':
    jsonFile = './seg_result/annotations/0421truck2_10.json'
    coco = NAT2021(jsonFile)
    print(len(list(coco.imgToAnns.keys())))
    for id in list(coco.imgToAnns.keys()):
        print(coco.loadImgs(id)[0])