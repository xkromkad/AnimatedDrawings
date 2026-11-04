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
    """Render at ~FPS and at most MAX_SECONDS (the result loops).

    The BVH clips run at 30 or 100 fps for up to 28 s (up to 1007 frames),
    which on the VM's one core is minutes. Stepping the scene clock by whole
    BVH frames keeps every pose exact; only in-between poses are skipped.
    """
    cls = video_render_controller.VideoRenderController
    original = cls._set_frames_left_to_render_and_delta_t

    def patched(self) -> None:
        original(self)
        frame_time = self.delta_t
        step = max(1, round(1.0 / FPS / frame_time))
        self.delta_t = frame_time * step
        frames = -(-self.frames_left_to_render // step)
        self.frames_left_to_render = min(frames, max(1, int(MAX_SECONDS / self.delta_t)))

    cls._set_frames_left_to_render_and_delta_t = patched


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
