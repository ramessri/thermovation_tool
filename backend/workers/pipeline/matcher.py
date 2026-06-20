"""
Stage 2: Feature Matching with DISK + LightGlue on GPU.

For each frame we run DISK to extract keypoints + descriptors, then run
LightGlue pairwise on a sliding window (each frame matched to the next
`window` frames) — the sequential-video assumption.

Output: a single matches.h5 file uploaded to storage that contains
- /frames/<i>/keypoints  (N_i, 2)  float32, pixel coords
- /frames/<i>/scores     (N_i,)    float32
- /frames/<i>/image_size (2,)      int32  (W, H)
- /pairs/<i>_<j>/matches (M, 2)    int32  (idx_i, idx_j)
- /pairs/<i>_<j>/scores  (M,)      float32
- attrs: frame_keys (list[str]), window (int), feature ("disk")

The pipeline contract is preserved:
- async function
- uses get_storage()
- progress_cb(float, str)
- returns a manifest dict
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from backend.core.storage import get_storage

logger = logging.getLogger(__name__)

# Tunables ---------------------------------------------------------------------

DEFAULT_WINDOW = 15               # match each frame to next 15 frames
MAX_KEYPOINTS = 4096              # DISK keypoints per frame
RESIZE_LONG_SIDE = None           # None = native resolution (4K: ~3–5GB VRAM peak, safe for 12GB)
                                  # Set to e.g. 4096 if shooting 8K and hitting OOM.
MIN_MATCH_THRESHOLD = 0           # keep all matches; SfM does its own filtering
LONG_RANGE_PAIRS = 20             # extra pairs sampled uniformly for loop closure
CROSS_SOURCE_MAX_SAMPLES = 15     # max frames sampled per source for cross-source bridging

# Stills at native resolution — same policy as video frames.
# DISK descriptors are offloaded to CPU immediately after extraction so VRAM
# pressure is bounded to one frame at a time regardless of dataset size.
STILL_MAX_KEYPOINTS    = 4096
STILL_RESIZE_LONG_SIDE = None     # None = native resolution


def _load_image(
    path: Path, resize_long: int | None
) -> tuple["np.ndarray", tuple[int, int], tuple[int, int]]:
    """Read an image as uint8 RGB, optionally resizing so long-side==resize_long.
    Pass resize_long=None to use the image at its native resolution.
    Returns (rgb_array, (W_orig, H_orig), (W_new, H_new))."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2 failed to read {path}")
    h0, w0 = bgr.shape[:2]
    if resize_long is not None:
        long_side = max(h0, w0)
        scale = resize_long / long_side if long_side > resize_long else 1.0
        if scale != 1.0:
            new_w = int(round(w0 * scale))
            new_h = int(round(h0 * scale))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, (w0, h0), (rgb.shape[1], rgb.shape[0])


async def run_feature_matching(
    project_id: str,
    frame_keys: list[str],
    tmp: Path,
    progress_cb: Callable[[float, str], None],
    window: int = DEFAULT_WINDOW,
    max_keypoints: int = MAX_KEYPOINTS,
    frame_source_boundaries: list[dict] | None = None,
) -> dict:
    if not frame_keys:
        raise RuntimeError("feature_matching: no frame_keys received from extract_frames.")

    # Heavy imports kept lazy so module import stays cheap.
    import torch
    from lightglue import DISK, LightGlue
    from lightglue.utils import numpy_image_to_torch

    storage = get_storage()
    progress_cb(0.0, "Starting feature matching (DISK + LightGlue)…")

    # ── 1. Download frames ────────────────────────────────────────────────────
    frames_dir = tmp / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    local_paths: list[Path] = []
    n = len(frame_keys)
    for i, key in enumerate(frame_keys):
        local = frames_dir / Path(key).name
        await storage.download(key, local)
        local_paths.append(local)
        if i % 10 == 0:
            progress_cb(0.05 * (i + 1) / n, f"Downloaded {i+1}/{n} frames")
    progress_cb(0.05, f"Downloaded {n} frames")

    # ── 2. Init models ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Identify which frame indices are stills — they get a larger keypoint budget
    # and higher-resolution feature extraction to exploit their native quality.
    still_indices: set[int] = set()
    if frame_source_boundaries:
        for b in frame_source_boundaries:
            if b["source"] == "stills":
                still_indices.update(range(b["start"], b["end"]))

    n_stills = len(still_indices)
    if n_stills:
        progress_cb(0.06, (
            f"Loading DISK+LightGlue on {device} — native resolution, "
            f"{n_stills} stills @ {STILL_MAX_KEYPOINTS} kpts, "
            f"{n - n_stills} video frames @ {max_keypoints} kpts…"
        ))
        extractor_still = DISK(max_num_keypoints=STILL_MAX_KEYPOINTS).eval().to(device)
    else:
        progress_cb(0.06, f"Loading DISK+LightGlue on {device} — native resolution, {max_keypoints} kpts…")
        extractor_still = None

    extractor_video = DISK(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="disk").eval().to(device)

    # ── 3. Extract features per frame ─────────────────────────────────────────
    feats: list[dict] = []  # one dict per frame (already on CPU after extraction)
    orig_sizes: list[tuple[int, int]] = []  # (W, H) original (before resize)

    for i, p in enumerate(local_paths):
        is_still  = i in still_indices
        resize    = STILL_RESIZE_LONG_SIDE if is_still else RESIZE_LONG_SIDE
        extractor = extractor_still if is_still else extractor_video

        rgb, (w0, h0), (wn, hn) = _load_image(p, resize)
        orig_sizes.append((w0, h0))
        img_t = numpy_image_to_torch(rgb).to(device)
        with torch.no_grad():
            f = extractor.extract(img_t[None])
        # Keypoints are in resized-image pixel coords. Rescale to original
        # so SfM can use them against the full-resolution images.
        sx, sy = w0 / wn, h0 / hn
        kpts = f["keypoints"][0].cpu().numpy().astype(np.float32)
        kpts[:, 0] *= sx
        kpts[:, 1] *= sy
        # Store on CPU — only move to GPU per-pair during matching.
        # PCIe transfer of 2 frames (~4MB) costs ~0.2ms vs ~50ms LightGlue forward.
        feats.append({
            "keypoints":   f["keypoints"].cpu(),    # (1, K, 2), resized coords
            "descriptors": f["descriptors"].cpu(),  # (1, K, D)
            "image_size":  f["image_size"].cpu(),   # (1, 2)
            "kpts_orig":   kpts,                    # (K, 2) numpy, ORIGINAL coords
            "scores": (f.get("keypoint_scores", f.get("scores", torch.zeros(1, kpts.shape[0])))[0]
                       .cpu().numpy().astype(np.float32)),
        })
        if (i + 1) % 10 == 0 or i + 1 == n:
            tag = "still" if is_still else "frame"
            progress_cb(0.05 + 0.35 * (i + 1) / n,
                        f"DISK features: {i+1}/{n} ({tag}, K={kpts.shape[0]})")

    # Free DISK extractor(s) — not needed during matching, reclaim VRAM.
    del extractor_video
    if extractor_still is not None:
        del extractor_still
    torch.cuda.empty_cache()

    # ── 4. Pairwise matching on sliding window + long-range pairs ─────────────
    # Build the full set of pairs: sliding window + uniform long-range samples.
    pairs_to_match: set[tuple[int, int]] = set()
    for i in range(n - 1):
        for j in range(i + 1, min(i + 1 + window, n)):
            pairs_to_match.add((i, j))
    # Long-range: evenly-spaced anchors matched to each other — provides loop
    # closure links so COLMAP doesn't produce fragmented sub-reconstructions.
    if n > window * 2:
        step = max(1, n // (LONG_RANGE_PAIRS + 1))
        anchors = list(range(0, n, step))
        for a in anchors:
            for b in anchors:
                if a < b:
                    pairs_to_match.add((a, b))

    # ── Cross-source bridging pairs ───────────────────────────────────────────
    # For multi-source projects (nook clips, supplemental stills, etc.), frames
    # from different sources have no temporal proximity, so the sliding window
    # alone cannot connect them. Sample up to CROSS_SOURCE_MAX_SAMPLES evenly-
    # spaced frames from each source and cross-match all combinations. This is
    # capped per-source so pair count stays O(n_sources² × MAX_SAMPLES²) and
    # doesn't explode with long main videos.
    #
    # Example: 3 general + 4 nook sources (7 total), 15 samples each →
    #   C(7,2)=21 source pairs × 15×15=225 pairs = ~4,700 extra pairs.
    if frame_source_boundaries and len(frame_source_boundaries) > 1:
        n_before = len(pairs_to_match)
        for si in range(len(frame_source_boundaries)):
            for sj in range(si + 1, len(frame_source_boundaries)):
                src_i = frame_source_boundaries[si]
                src_j = frame_source_boundaries[sj]
                n_i   = src_i["end"] - src_i["start"]
                n_j   = src_j["end"] - src_j["start"]
                step_i = max(1, n_i // CROSS_SOURCE_MAX_SAMPLES)
                step_j = max(1, n_j // CROSS_SOURCE_MAX_SAMPLES)
                samp_i = list(range(src_i["start"], src_i["end"], step_i))[:CROSS_SOURCE_MAX_SAMPLES]
                samp_j = list(range(src_j["start"], src_j["end"], step_j))[:CROSS_SOURCE_MAX_SAMPLES]
                for a in samp_i:
                    for b in samp_j:
                        pairs_to_match.add((min(a, b), max(a, b)))
        n_cross = len(pairs_to_match) - n_before
        logger.info(
            "[%s] cross-source bridging: %d extra pairs across %d sources (≤%d samples/source)",
            project_id, n_cross, len(frame_source_boundaries), CROSS_SOURCE_MAX_SAMPLES,
        )

    all_pairs = sorted(pairs_to_match)

    pair_count = 0
    total_matches = 0
    first_pair_count = None
    matches_to_write: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    total_pairs_estimate = len(all_pairs)
    done = 0

    _GPU_KEYS = ("keypoints", "descriptors", "image_size")
    for i, j in all_pairs:
        feats_i = {k: feats[i][k].to(device) for k in _GPU_KEYS}
        feats_j = {k: feats[j][k].to(device) for k in _GPU_KEYS}
        with torch.no_grad():
            out = matcher({"image0": feats_i, "image1": feats_j})
        del feats_i, feats_j  # free GPU tensors immediately
        matches = out["matches"][0].cpu().numpy().astype(np.int32)   # (M, 2)
        scores = out["scores"][0].cpu().numpy().astype(np.float32)   # (M,)
        if MIN_MATCH_THRESHOLD > 0:
            keep = scores >= MIN_MATCH_THRESHOLD
            matches = matches[keep]
            scores = scores[keep]
        matches_to_write.append((i, j, matches, scores))
        pair_count += 1
        total_matches += int(matches.shape[0])
        if first_pair_count is None:
            first_pair_count = int(matches.shape[0])
        done += 1
        if done % 20 == 0:
            progress_cb(
                0.40 + 0.50 * done / max(total_pairs_estimate, 1),
                f"Matched pair ({i},{j}): {matches.shape[0]} matches  "
                f"({done}/{total_pairs_estimate})",
            )

    progress_cb(0.92, f"All {pair_count} pairs matched ({total_matches} matches total)")

    # ── 5. Write HDF5 + upload ────────────────────────────────────────────────
    matches_path = tmp / "matches.h5"
    with h5py.File(matches_path, "w") as h5:
        h5.attrs["window"] = window
        h5.attrs["feature"] = "disk"
        h5.attrs["frame_keys"] = np.array(frame_keys, dtype=h5py.string_dtype())
        h5.attrs["n_frames"] = n

        g_frames = h5.create_group("frames")
        for i, f in enumerate(feats):
            grp = g_frames.create_group(str(i))
            grp.create_dataset("keypoints", data=f["kpts_orig"], compression="gzip")
            grp.create_dataset("scores", data=f["scores"], compression="gzip")
            grp.create_dataset(
                "image_size",
                data=np.array(orig_sizes[i], dtype=np.int32),  # (W, H)
            )

        g_pairs = h5.create_group("pairs")
        for (i, j, m, s) in matches_to_write:
            grp = g_pairs.create_group(f"{i}_{j}")
            grp.create_dataset("matches", data=m, compression="gzip")
            grp.create_dataset("scores", data=s, compression="gzip")

    match_data_key = f"{project_id}/matching/matches.h5"
    await storage.upload(matches_path, match_data_key)
    progress_cb(1.0, f"Uploaded matches.h5 ({matches_path.stat().st_size/1e6:.1f} MB)")

    return {
        "match_data_key": match_data_key,
        "kpts_data_key": match_data_key,  # same file
        "frame_keys": frame_keys,
        "pair_count": pair_count,
        "total_matches": total_matches,
        "first_pair_match_count": first_pair_count,
        "window": window,
        "n_frames": n,
    }
