from pathlib import Path
from typing import List

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import json


def read_img(path):
    img = np.array(Image.open(path).convert('RGB'))

    return torch.tensor(img.transpose(2, 0, 1).astype(np.float32))


class SPairDataset(Dataset):
    def __init__(self, pair_ann_path: Path, layout_path: Path, image_path: Path, dataset_size, pck_alpha: List[float], datatype, preprocess = None):

        self.datatype = datatype
        self.pck_alpha = pck_alpha
        # build the layout file path and read non-empty lines
        ann_file_path = Path(layout_path) / dataset_size / f"{datatype}.txt"
        with open(ann_file_path, 'r') as f:
            self.ann_files = [line.strip() for line in f.readlines() if line.strip()]

        self.pair_ann_path = Path(pair_ann_path)
        self.image_path = Path(image_path)
        # use pathlib to list category directories under image_path
        self.categories = [p.name for p in self.image_path.iterdir() if p.is_dir()]
        self.categories.sort()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.ann_files)

    def __getitem__(self, idx):
        # get pre-processed images
        ann_file = self.ann_files[idx].replace(':', '_') + '.json'
        ann_path = Path(self.pair_ann_path) / self.datatype / ann_file
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        category = annotation['category']
        src_img = read_img(Path(self.image_path) / category / annotation['src_imname'])
        trg_img = read_img(Path(self.image_path) / category / annotation['trg_imname'])

        trg_bbox = annotation['trg_bndbox']


        sample = {'pair_id': annotation['pair_id'],
                  'filename': annotation['filename'],
                  'src_imname': annotation['src_imname'],
                  'trg_imname': annotation['trg_imname'],
                  'og_src_imsize': src_img.size(),
                  'og_trg_imsize': trg_img.size(),

                  'src_bbox': annotation['src_bndbox'],
                  'trg_bbox': annotation['trg_bndbox'],
                  'category': annotation['category'],

                  'src_pose': annotation['src_pose'],
                  'trg_pose': annotation['trg_pose'],

                  'src_img': src_img,
                  'trg_img': trg_img,
                  'src_kps': torch.tensor(annotation['src_kps']).float(),
                  'trg_kps': torch.tensor(annotation['trg_kps']).float(),

                  'mirror': annotation['mirror'],
                  'vp_var': annotation['viewpoint_variation'],
                  'sc_var': annotation['scale_variation'],
                  'truncn': annotation['truncation'],
                  'occlsn': annotation['occlusion'],

                  }

        if self.preprocess:
            sample = self.preprocess(sample)

        pck_threshold = []
        for alpha in self.pck_alpha:
            pck_threshold.append(max(trg_bbox[2] - trg_bbox[0],  trg_bbox[3] - trg_bbox[1]) * alpha)

        sample['pck_threshold'] = pck_threshold

        return sample
