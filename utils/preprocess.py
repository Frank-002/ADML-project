from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F
from torchvision import transforms


class PreProcess(object):
    def __init__(self, *, long_side_length: int, apply_norm:bool = False):
        self.long_side_length = long_side_length
        self.apply_norm = apply_norm
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _get_preprocess_shape(self, oldh: int, oldw: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = self.long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return newh, neww

    @staticmethod
    def _resize_points(pts, sy, sx):
        # pts: (..., 2), coordinates are (x, y)
        pts[..., 0] = pts[..., 0] * sx
        pts[..., 1] = pts[..., 1] * sy
        return pts

    @staticmethod
    def _resize_bbox_xyxy(bb, sy, sx):
        # bb: (x1, y1, x2, y2)
        bb[0] *= sx
        bb[2] *= sx
        bb[1] *= sy
        bb[3] *= sy
        return bb

    def __call__(self, sample: Dict[str, Any]):
        for key in ["src", "trg"]:
            img = sample[f"{key}_img"]

            # original size coherent with kps/bbox before resize
            oldh, oldw = int(img.shape[-2]), int(img.shape[-1])

            # resize the image to the longest size while keeping the aspect ratio
            newh, neww = self._get_preprocess_shape(oldh, oldw)
            img_resized = F.interpolate(img.unsqueeze(0), (newh, neww), mode="bilinear", align_corners=False, antialias=True)

            if self.apply_norm:
                img_resized /= 255.0
                img_resized = self.normalize(img_resized)

            padh = self.long_side_length - newh
            padw = self.long_side_length - neww
            img_padded = F.pad(img_resized, (0, padw, 0, padh))

            # Compute scaling factors
            sy = newh / oldh
            sx = neww / oldw

            sample[f"{key}_img"] = img_padded.squeeze(0)
            sample[f"{key}_kps"] = self._resize_points(sample[f"{key}_kps"], sy, sx)
            sample[f"{key}_bbox"] = self._resize_bbox_xyxy(sample[f"{key}_bbox"], sy, sx)
            sample[f"{key}_nopad_size"] = torch.tensor([newh, neww])


        return sample