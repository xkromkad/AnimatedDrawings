"""Drawing -> character annotations (texture, mask, rig), with the models in-process.

Adapted from examples/image_to_annotations.py: the same cropping, segmentation
and skeleton, but the detector and pose estimator are called directly instead
of through TorchServe (archived upstream in 2025), and a drawing without a
usable figure raises NoFigure rather than asserting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage
from skimage import measure

from models import Models

log = logging.getLogger(__name__)

# The detector's own threshold in the TorchServe handler.
DETECTION_THRESHOLD = 0.5
# Below this mean keypoint confidence the rig is guesswork and the dance looks
# broken; telling the child to draw the whole figure is the better answer.
MIN_POSE_SCORE = 0.2


class NoFigure(Exception):
    """No drawn person (or no usable skeleton) in the drawing."""


def image_to_annotations(img: np.ndarray, out_dir: Path, models: Models) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if img is None or img.ndim != 3:
        raise NoFigure('not an RGB image')

    if np.max(img.shape) > 1000:
        scale = 1000 / np.max(img.shape)
        img = cv2.resize(img, (round(scale * img.shape[1]), round(scale * img.shape[0])))
    cv2.imwrite(str(out_dir / 'image.png'), img)

    detections = [d for d in models.detect(img) if d['score'] >= DETECTION_THRESHOLD]
    if not detections:
        raise NoFigure('no humanoid detected')
    detections.sort(key=lambda d: d['score'], reverse=True)

    h, w = img.shape[:2]
    l, t, r, b = [round(x) for x in detections[0]['bbox']]
    l, t, r, b = max(l, 0), max(t, 0), min(r, w), min(b, h)
    if r - l < 16 or b - t < 16:
        raise NoFigure('bounding box too small')

    with open(out_dir / 'bounding_box.yaml', 'w') as f:
        yaml.dump({'left': l, 'top': t, 'right': r, 'bottom': b}, f)

    cropped = img[t:b, l:r]
    mask = segment(cropped)

    keypoints = models.pose(cropped)
    if keypoints is None:
        raise NoFigure('no skeleton')
    if float(np.mean(keypoints[:, 2])) < MIN_POSE_SCORE:
        raise NoFigure(f'skeleton too uncertain ({np.mean(keypoints[:, 2]):.2f})')
    kpts = keypoints[:, :2]

    def loc(p):
        return [round(x) for x in p]

    skeleton = [
        {'loc': loc((kpts[11] + kpts[12]) / 2), 'name': 'root', 'parent': None},
        {'loc': loc((kpts[11] + kpts[12]) / 2), 'name': 'hip', 'parent': 'root'},
        {'loc': loc((kpts[5] + kpts[6]) / 2), 'name': 'torso', 'parent': 'hip'},
        {'loc': loc(kpts[0]), 'name': 'neck', 'parent': 'torso'},
        {'loc': loc(kpts[6]), 'name': 'right_shoulder', 'parent': 'torso'},
        {'loc': loc(kpts[8]), 'name': 'right_elbow', 'parent': 'right_shoulder'},
        {'loc': loc(kpts[10]), 'name': 'right_hand', 'parent': 'right_elbow'},
        {'loc': loc(kpts[5]), 'name': 'left_shoulder', 'parent': 'torso'},
        {'loc': loc(kpts[7]), 'name': 'left_elbow', 'parent': 'left_shoulder'},
        {'loc': loc(kpts[9]), 'name': 'left_hand', 'parent': 'left_elbow'},
        {'loc': loc(kpts[12]), 'name': 'right_hip', 'parent': 'root'},
        {'loc': loc(kpts[14]), 'name': 'right_knee', 'parent': 'right_hip'},
        {'loc': loc(kpts[16]), 'name': 'right_foot', 'parent': 'right_knee'},
        {'loc': loc(kpts[11]), 'name': 'left_hip', 'parent': 'root'},
        {'loc': loc(kpts[13]), 'name': 'left_knee', 'parent': 'left_hip'},
        {'loc': loc(kpts[15]), 'name': 'left_foot', 'parent': 'left_knee'},
    ]

    char_cfg = {'skeleton': skeleton, 'height': cropped.shape[0], 'width': cropped.shape[1]}
    cv2.imwrite(str(out_dir / 'texture.png'), cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA))
    cv2.imwrite(str(out_dir / 'mask.png'), mask)
    with open(out_dir / 'char_cfg.yaml', 'w') as f:
        yaml.dump(char_cfg, f)


def segment(img: np.ndarray) -> np.ndarray:
    """Unchanged from examples/image_to_annotations.py, except NoFigure."""
    img = np.min(img, axis=2)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 8)
    img = cv2.bitwise_not(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=2)

    mask = np.zeros([img.shape[0] + 2, img.shape[1] + 2], np.uint8)
    mask[1:-1, 1:-1] = img.copy()
    im_floodfill = np.full(img.shape, 255, np.uint8)
    h, w = img.shape[:2]
    for x in range(0, w - 1, 10):
        cv2.floodFill(im_floodfill, mask, (x, 0), 0)
        cv2.floodFill(im_floodfill, mask, (x, h - 1), 0)
    for y in range(0, h - 1, 10):
        cv2.floodFill(im_floodfill, mask, (0, y), 0)
        cv2.floodFill(im_floodfill, mask, (w - 1, y), 0)
    im_floodfill[0, :] = 0
    im_floodfill[-1, :] = 0
    im_floodfill[:, 0] = 0
    im_floodfill[:, -1] = 0

    mask2 = cv2.bitwise_not(im_floodfill)
    best = None
    biggest = 0
    for c in measure.find_contours(mask2, 0.0):
        x = np.zeros(mask2.T.shape, np.uint8)
        cv2.fillPoly(x, [np.int32(c)], 1)
        size = len(np.where(x == 1)[0])
        if size > biggest:
            best = x
            biggest = size
    if best is None:
        raise NoFigure('no contours')

    best = ndimage.binary_fill_holes(best).astype(int)
    return (255 * best.astype(np.uint8)).T
