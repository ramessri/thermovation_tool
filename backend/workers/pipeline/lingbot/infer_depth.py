#!/usr/bin/env python3
"""LingBot-Map depth inference -> per-frame npz.

Runs INSIDE the isolated lingbot venv (torch 2.8 / cu128) and is invoked as a
subprocess by backend/workers/pipeline/lingbot_fusion.py. It must NOT be
imported by the main worker process (torch 2.4 there) — file-path execution only.

Reads a folder of frame images, runs feed-forward depth inference, and writes a
compressed npz with per-frame {depth, intrinsic, conf, rgb, names}. The fusion
step (affine calibration + TSDF) happens in the main worker, not here.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

# LingBot-Map repo (cloned in the Docker image). Importable in this venv only.
REPO = os.environ.get("LINGBOT_REPO", "/opt/lingbot-map")
sys.path.insert(0, REPO)
import demo  # noqa: E402  (load_images, load_model, postprocess, prepare_for_visualization)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--num_scale_frames", type=int, default=2)
    ap.add_argument("--camera_num_iterations", type=int, default=1)
    ap.add_argument("--mode", default="auto", choices=["auto", "streaming", "windowed"])
    ap.add_argument("--windowed_threshold", type=int, default=120,
                    help="frame count above which windowed mode is used in --mode auto")
    ap.add_argument("--window_size", type=int, default=24)
    ap.add_argument("--overlap_size", type=int, default=6)
    ap.add_argument("--mask_sky", action="store_true",
                    help="zero out sky pixels' depth via skyseg (outdoor scenes)")
    ap.add_argument("--skyseg_onnx", default="/app/models/lingbot/skyseg.onnx")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available in lingbot venv", file=sys.stderr)
        sys.exit(2)
    device = torch.device("cuda")

    t0 = time.time()
    images, paths, _ = demo.load_images(
        image_folder=args.image_folder, video_path=None, fps=10,
        first_k=None, stride=1, image_size=args.image_size,
        patch_size=14, rotate_clockwise_90=False,
    )
    n_frames = images.shape[0]
    mode = args.mode
    if mode == "auto":
        mode = "windowed" if n_frames > args.windowed_threshold else "streaming"
    print(f"Loaded {n_frames} images {tuple(images.shape)} in {time.time()-t0:.1f}s | mode={mode}",
          flush=True)

    ns = argparse.Namespace(
        mode=mode, image_size=args.image_size, patch_size=14,
        enable_3d_rope=True, max_frame_num=1024, kv_cache_sliding_window=64,
        num_scale_frames=args.num_scale_frames, use_sdpa=True,
        camera_num_iterations=args.camera_num_iterations, model_path=args.model_path,
    )
    model = demo.load_model(ns, device)
    # Cast DINOv2 trunk to bf16 (heads stay fp32, upcast internally). ~2-3 GB saved.
    if getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=torch.bfloat16)
    images = images.to(device)
    keyframe_interval = (n_frames + 319) // 320

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if mode == "windowed":
            predictions = model.inference_windowed(
                images, window_size=args.window_size, overlap_size=args.overlap_size,
                overlap_keyframes=None, num_scale_frames=args.num_scale_frames,
                keyframe_interval=keyframe_interval, output_device=torch.device("cpu"),
            )
        else:
            predictions = model.inference_streaming(
                images, num_scale_frames=args.num_scale_frames,
                keyframe_interval=keyframe_interval, output_device=torch.device("cpu"),
            )
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Inference {time.time()-t1:.1f}s | GPU peak {peak:.2f} GB", flush=True)

    del images
    torch.cuda.empty_cache()
    images_cpu = predictions["images"]  # offloaded to CPU
    predictions, images_cpu = demo.postprocess(predictions, images_cpu)
    vis = demo.prepare_for_visualization(predictions, images_cpu)

    depth = np.asarray(vis["depth"])
    if depth.ndim == 4:
        depth = depth[..., 0]                       # (S,H,W)
    intr = np.asarray(vis["intrinsic"])             # (S,3,3)
    dconf = np.asarray(vis["depth_conf"])           # (S,H,W)
    imgs = np.asarray(vis["images"])                # (S,3,H,W) or (S,H,W,3)
    if imgs.ndim == 4 and imgs.shape[1] == 3:
        imgs = np.transpose(imgs, (0, 2, 3, 1))     # -> (S,H,W,3)
    names = [os.path.basename(p) for p in paths]

    # Sky masking (outdoor): zero depth on sky pixels so they don't fuse into
    # floating garbage. LingBot predicts unreliable/far depth for sky.
    if args.mask_sky:
        try:
            import cv2, urllib.request, onnxruntime
            # Inlined from lingbot_map.vis.sky_segmentation (avoids that package's
            # matplotlib import). Higher score = sky. ImageNet-normed 320x320 input.
            SZ = (320, 320)
            def _skyseg(sess, bgr):
                r = cv2.resize(bgr, SZ)
                x = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32)
                mean = np.array([0.485, 0.456, 0.406], np.float32)
                std = np.array([0.229, 0.224, 0.225], np.float32)
                x = ((x / 255.0 - mean) / std).transpose(2, 0, 1).reshape(-1, 3, SZ[1], SZ[0]).astype("float32")
                inn = sess.get_inputs()[0].name; outn = sess.get_outputs()[0].name
                res = np.array(sess.run([outn], {inn: x})).squeeze()
                res = (res - res.min()) / max(res.max() - res.min(), 1e-8) * 255.0
                return res.astype(np.uint8)
            skyp = args.skyseg_onnx
            if not os.path.exists(skyp):
                os.makedirs(os.path.dirname(skyp), exist_ok=True)
                urllib.request.urlretrieve(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", skyp)
            sess = onnxruntime.InferenceSession(skyp, providers=["CPUExecutionProvider"])
            H, W = depth.shape[1:]
            zeroed = 0
            for i in range(depth.shape[0]):
                bgr = cv2.cvtColor((imgs[i] * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                sky = cv2.resize(_skyseg(sess, bgr), (W, H)) > 128
                depth[i][sky] = 0.0
                zeroed += int(sky.sum())
            print(f"Sky mask: zeroed {zeroed} px across {depth.shape[0]} frames", flush=True)
        except Exception as e:
            print(f"Sky mask skipped ({e})", file=sys.stderr, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        depth=depth.astype(np.float32),
        intrinsic=intr.astype(np.float32),
        conf=dconf.astype(np.float32),
        rgb=(imgs * 255).clip(0, 255).astype(np.uint8),
        names=np.array(names),
    )
    print(f"Wrote {args.out_npz}  (depth {depth.shape}, {len(names)} frames, "
          f"peak {peak:.2f} GB, total {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
