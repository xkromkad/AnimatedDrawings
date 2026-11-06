"""Render one animation: python render_job.py <annotation_dir> <motion> <retarget> <out.gif|out.webp>

Runs in its own process for every request. The renderer keeps global OpenGL
(OSMesa) state, so a fresh process per render is what keeps one child's
dance from leaking into the next, and a renderer crash from taking down the
server.

The renderer itself only writes .gif or .mp4. For .webp the GIF writer's
final step is swapped for an animated WebP: real alpha (no jagged GIF edges
around the figure) at a fraction of the size.
"""

import sys
from pathlib import Path

import yaml

from animated_drawings import render
from animated_drawings.controller import video_render_controller

REPO = Path(__file__).resolve().parent.parent


FPS = 15
MAX_SECONDS = 10.0


def sample_frames() -> None:
    """Animate at ~FPS for at most MAX_SECONDS (the result loops).

    The BVH clips run at 30 or 100 fps for up to 28 s (up to 1007 frames).
    Keeping every n-th frame of the clip as it is loaded (and scaling its
    frame time by n) means the retargeter, which precomputes every frame of
    the clip, and the renderer both only touch frames that are shown; each
    kept pose is exact. The controller then renders the clip as given.
    """
    from animated_drawings.model.bvh import BVH

    original = BVH.from_file.__func__

    def from_file(cls, bvh_fn, start_frame_idx=0, end_frame_idx=None):
        bvh = original(cls, bvh_fn, start_frame_idx, end_frame_idx)
        step = max(1, round(1.0 / FPS / bvh.frame_time))
        keep = max(1, min(-(-bvh.frame_max_num // step), int(MAX_SECONDS / (bvh.frame_time * step))))
        bvh.pos_data = bvh.pos_data[::step][:keep]
        bvh.rot_data = bvh.rot_data[::step][:keep]
        bvh.frame_max_num = keep
        bvh.frame_time *= step
        bvh.apply_frame(0)
        return bvh

    BVH.from_file = classmethod(from_file)


def cache_joint_map(anno: Path) -> None:
    """Reuse the joint -> triangles map across the dances of one drawing.

    It is a pure-Python BFS over every mask pixel (~4 s on the VM) and
    depends only on the drawing and its skeleton, never on the motion. The
    original also flips each skeleton joint's y in place (later steps read
    it), so a cache hit repeats exactly that.
    """
    import os
    import pickle
    from animated_drawings.model.animated_drawing import AnimatedDrawing

    cache = anno / 'joint_map.pkl'
    original = AnimatedDrawing._initialize_joint_to_triangles_dict

    def cached(self) -> None:
        if cache.exists():
            for joint in self.char_cfg.skeleton:
                joint['loc'][1] = 1 - joint['loc'][1]
            with open(cache, 'rb') as f:
                self.joint_to_tri_v_idx = pickle.load(f)
            return
        original(self)
        tmp = cache.with_suffix(f'.{os.getpid()}.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(self.joint_to_tri_v_idx, f)
        os.replace(tmp, cache)

    AnimatedDrawing._initialize_joint_to_triangles_dict = cached


def write_webp_instead(webp_path: Path) -> None:
    def cleanup(self) -> None:
        from PIL import Image
        # 256 colours (with alpha) then lossless: a crayon drawing loses
        # nothing visible, and the file is ~40 % smaller than lossy q82.
        frames = [
            Image.fromarray(f).quantize(256, method=Image.Quantize.FASTOCTREE).convert('RGBA')
            for f in self.frames
        ]
        frames[0].save(
            webp_path, format='WEBP', save_all=True, append_images=frames[1:],
            duration=self.duration, loop=0, lossless=True, method=4,
        )
    video_render_controller.GIFWriter.cleanup = cleanup


def main(anno_dir: str, motion: str, retarget: str, out_path: str) -> None:
    anno = Path(anno_dir)
    out = Path(out_path)
    sample_frames()
    cache_joint_map(anno)
    if out.suffix == '.webp':
        write_webp_instead(out)
        out_path = str(out.with_suffix('.gif'))
    mvc_cfg = {
        'scene': {
            'ANIMATED_CHARACTERS': [{
                'character_cfg': str((anno / 'char_cfg.yaml').resolve()),
                'motion_cfg': str(REPO / 'examples/config/motion' / f'{motion}.yaml'),
                'retarget_cfg': str(REPO / 'examples/config/retarget' / f'{retarget}.yaml'),
            }],
        },
        'view': {'USE_MESA': True},
        'controller': {
            'MODE': 'video_render',
            'OUTPUT_VIDEO_PATH': str(Path(out_path).resolve()),
        },
    }
    cfg_fn = Path(out_path).with_suffix('.mvc.yaml')
    with open(cfg_fn, 'w') as f:
        yaml.dump(mvc_cfg, f)
    render.start(str(cfg_fn))


if __name__ == '__main__':
    main(*sys.argv[1:5])
