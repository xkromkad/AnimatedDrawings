"""The two AnimatedDrawings models, loaded in-process.

Both come from the v0.0.1 release .mar files, unpacked by the Dockerfile into
$ANIMATOR_MODELS/{detector,pose} (config.py + the checkpoint named in
MAR-INF/MANIFEST.json). The calls mirror the TorchServe handlers shipped inside
those archives (mmdet_handler.py, mmpose_handler.py), so results match what
TorchServe returned.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _model_files(model_dir: Path) -> tuple[str, str]:
    manifest = json.loads((model_dir / 'MAR-INF' / 'MANIFEST.json').read_text())
    return (
        str(model_dir / manifest['model']['modelFile']),
        str(model_dir / manifest['model']['serializedFile']),
    )


@dataclass
class Models:
    detector: Any
    pose_model: Any

    @classmethod
    def load(cls) -> 'Models':
        from mmdet.apis import init_detector
        from mmpose.apis import init_pose_model

        threads = int(os.environ.get('OMP_NUM_THREADS', '0') or 0)
        if threads > 0:
            torch.set_num_threads(threads)

        root = Path(os.environ.get('ANIMATOR_MODELS', '/models'))
        det_cfg, det_ckpt = _model_files(root / 'detector')
        pose_cfg, pose_ckpt = _model_files(root / 'pose')
        detector = init_detector(det_cfg, det_ckpt, device='cpu')
        # The detector was trained at 1333x800. A drawing is one big figure
        # on white, and at 800x480 the scores and boxes match within a few
        # pixels (annotate.py pads the box) while detection takes 7 s
        # instead of 18 s on the VM's core.
        w, h = (int(x) for x in os.environ.get('ANIMATOR_DET_SCALE', '800x480').split('x'))
        for step in detector.cfg.data.test.pipeline:
            if 'img_scale' in step:
                step['img_scale'] = (w, h)
        return cls(
            detector=detector,
            pose_model=init_pose_model(pose_cfg, pose_ckpt, device='cpu'),
        )

    @torch.inference_mode()
    def detect(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Drawn humanoids as [{'bbox': [l, t, r, b], 'score': s}] (mmdet_handler.postprocess)."""
        from mmdet.apis import inference_detector

        result = inference_detector(self.detector, img_bgr)
        bbox_result = result[0] if isinstance(result, tuple) else result
        out = []
        for class_result in bbox_result:
            for bbox in class_result:
                out.append({'bbox': bbox[:4].tolist(), 'score': float(bbox[4])})
        return out

    @torch.inference_mode()
    def pose(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """17 COCO keypoints (x, y, score) of the figure filling the crop, or None.

        As mmpose_handler: top-down model with the whole crop as the person box.
        """
        from mmpose.apis import inference_top_down_pose_model

        preds, _ = inference_top_down_pose_model(self.pose_model, crop_bgr, person_results=None)
        if len(preds) != 1:
            return None
        return np.asarray(preds[0]['keypoints'], dtype=np.float32)
