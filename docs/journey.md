# Research Journey

The driving question behind this project: **where can a vision-language model add meaningful value inside a classical photogrammetry pipeline?**

Photogrammetry was chosen as the domain because it is well-understood and well-tooled. Feature matching, structure-from-motion, and multi-view stereo are solved problems with reliable, battle-tested implementations. That makes it a good research surface — the baseline is strong enough that any VLM contribution can be evaluated against a known-good result, and failures are informative rather than confounded by a weak foundation.

### The original target

Classical photogrammetry has two well-known failure modes that no amount of better footage fully solves:

- **Featureless surfaces** — plain white walls, ceilings, uniform floors. MVS has nothing to match across views, so these areas either reconstruct poorly or produce phantom geometry.
- **Thin structures** — cables, chair legs, plant stems, narrow edges. The depth estimates at boundaries are unreliable, and these objects tend to either disappear or bloat into thick blobs in the dense cloud.

The hypothesis was that a VLM, given a frame and the partial reconstruction at that point, could identify these problem areas and guide either re-sampling (capture more views of this region) or post-processing (apply a different reconstruction strategy here). A model that can say "this wall has no texture" or "there is a thin vertical structure at this location" could in principle close the gap that geometry alone cannot.

**This objective was never properly tested.** To reach it, the pipeline first needed reliable metric scale — without it, any VLM-guided geometry correction has no coordinate system to work in. The work on VLM-based scale detection (Phases 1–4 below) was meant to be a prerequisite step. Those experiments ran into fundamental reliability problems that consumed the available research time, and the original featureless-surface / thin-structure question remains open.

This document is a complete record of every approach tried, what failed, what was learned, and what that led to. It is more useful than the current code for understanding the design space.

### The headline finding

**VLM output is too inconsistent to trust in an automated pipeline without a human in the loop.** Every VLM integration attempted here eventually hit the same wall: the model would produce correct, useful output most of the time, and confidently wrong output the rest of the time — with no reliable signal to distinguish the two. In a fully automated pipeline, that means either accepting silent failures or adding a human confirmation gate after every VLM step. The confirmation gate kills the automation; the silent failures degrade the output in ways that are hard to debug.

The deeper lesson is about problem framing. VLMs were applied here to compensate for missing information — a length reference that wasn't in the footage. That is fundamentally an *information* problem, not a *perception* problem. The right fix was to put the reference in the scene before filming (ArUco markers), not to infer it afterward. A consistent physical process — print markers, place them, shoot — turned out to be less frustrating, faster, and far more accurate than any AI-based approach.

Where VLMs likely *do* add value in this domain is in tasks where approximate or subjective output is acceptable: describing what a reconstruction shows, suggesting where coverage is thin in natural language, helping a non-technical user understand what went wrong. Tasks where correctness is binary and the cost of a wrong answer is high (metric scale, camera pose, reconstruction quality gates) are poor fits.

---

## Where we started

The initial commit was a complete photogrammetry platform scaffolding — FastAPI backend, Celery workers, Next.js frontend, COLMAP integration, and a 7-stage pipeline. The core geometry pipeline (LightGlue → SfM → MVS → coverage → export) was solid from the beginning and remains largely unchanged today.

The contested part was always: **how do you get metric scale?**

---

## Phase 0 — Foundation (commit `efe8f74`)

The base pipeline: extract frames → LightGlue feature matching → pycolmap SfM → COLMAP MVS → coverage → export. This produced good dense clouds but in arbitrary units. You could not measure anything.

---

## Phase 1–3 — VLM-anchored scale detection

### The idea

Use AI to recognise known objects in the scene and derive metric scale from their known dimensions. A keyboard is 440 mm wide. A credit card is 85.6 mm. If we can detect one in the cloud, we have scale.

### Implementation

**Phase 1 (commit `36986e2`)** — Grounding DINO (GDINO) object detection. Scanned sampled frames for 18 known objects: credit cards, keyboards, CD cases, RPi boards, floppy disks, etc. Estimated scale factor per detection by projecting the 3D depth cloud into the frame and comparing pixel span to known physical dimensions.

**Phase 2a/2b (commits `127abc3`, `ecf36d5`)** — Added LLaVA 1.5 7B (4-bit quantised) to:
1. Classify the scene (indoor_room / outdoor / object / etc.)
2. Detect thin structures and featureless surfaces
3. Validate GDINO plane fits ("is this actually a floor?") to reject hallucinated detections

**Phase 3 (commit `c3f9741`)** — Depth Anything v2 Large for depth fusion:
1. DA2 predicts relative depth per pixel
2. SfM sparse points project into the frame as metric anchors for scale alignment
3. Aligned depth maps back-projected to world space and merged with MVS cloud

This was intended to fill in geometry that MVS couldn't recover: plain white walls, ceilings, featureless tabletops.

**Phase 4 (commit `048e653`)** — Render-and-compare hallucination flagger: rendered the 3D cloud from the camera viewpoint and compared against the original frame to catch cases where depth fusion added geometry that didn't match the real appearance.

### What we learned

**GDINO detections were unreliable.** The model confidently detected "keyboard" on decorative patterns, "credit card" on any rectangular object, and "A4 paper" on walls. After VLM validation, the detection rate dropped to near zero for scenes that didn't have textbook-perfect examples of known objects.

The approach required objects that were:
- Flat (so depth projection worked)
- Fully visible (no occlusion)
- Not too small in the frame (< 50px width → calibration was noisy)
- In a texture-compatible region of the MVS cloud

In practice, these conditions were rarely met simultaneously. Most successful detections came from carefully staged test scenes.

**LLaVA was slow and difficult to prompt reliably.** The 4-bit model took ~1 min per inference pass. More critically, the structured JSON output was inconsistent — the model would hallucinate fields, use different key names, or fail to produce valid JSON entirely. We added 4 fallback parsing passes, which helped but didn't solve the root problem.

The VLM's most useful contribution was scene type classification (indoor vs. outdoor vs. object) which drove adaptive MVS parameters. But that could be done with a simpler heuristic.

**Three failure modes that no prompt engineering could fix — observed consistently across both local (Ollama) and external (Gemini) models:**

*Over-specification.* When asked to identify an object for scale reference, the model would go too deep. A ZX Spectrum 48K on a desk would come back as "the rubber key variant" or a specific revision — detail that is irrelevant for scale derivation and introduces a new failure surface (the sub-model identification could itself be wrong). Every prompt variation tried — asking for the category only, constraining output to a fixed vocabulary, instructing it to stop at the first level of identification — eventually broke down. The model's training pushes it toward specificity, and that instinct cannot be reliably suppressed for objects it recognises well.

*Context hallucination.* The model would use scene context to infer objects that weren't visible. A frame showing a vintage computer, a disk drive, and a controller would produce a detection for a monitor — because a monitor *belongs* in that scene. The model was completing a plausible scene description rather than reporting what was in the frame. This was particularly damaging for scale derivation: a hallucinated object has no physical presence in the 3D cloud, so any scale factor derived from it is pure noise. The problem was not detectable from the model's output — confidence scores were high on hallucinated detections.

*Bounding box inconsistency.* Even when the object was correctly identified, the model could not reliably draw a tight, consistent bounding box around it. Boxes would drift between runs on the same frame, clip the object's edges, or expand to include nearby objects. Since the scale derivation depended on pixel span mapped to known physical dimensions, a box that was 20% too wide produced a scale factor that was 20% off. The error was silent — the pipeline had no way to know the box was wrong.

**Depth Anything v2 scale alignment was fragile.** The relative-to-metric alignment (`metric_depth = scale × da2_depth + shift`, fitted on SfM sparse points) was sensitive to the number and distribution of anchor points. With fewer than ~15 anchors (common in texture-poor scenes), the alignment was unstable. DA2 also uses a disparity convention (larger value = closer) which we got backwards in the initial implementation — the `scale` term came out negative, and an overly strict positivity guard (`if scale <= 0: return None`) silently dropped all depth fusion output for these scenes.

The OOM issues were severe. DA2 ViT-L requires ~3 GB VRAM. When running alongside COLMAP patch_match_stereo (8–10 GB), there wasn't enough headroom. We eventually serialised the stages (empty_cache between them), but the back-projection at 8K resolution (7680×4320) created 26M points per frame, crashing on 12 GB cards. We added a `_MAX_BACKPROJECT_PX = 1920` cap which brought it under control.

> **Note — revisiting this.** Of all six approaches, this is the one that fought hardest without being fundamentally wrong. Every failure mode listed above is an implementation problem, not a conceptual one: the scale alignment issue disappears now that ArUco provides reliable metric scale and well-distributed anchor points; the disparity convention bug is a one-line fix; the VRAM crash goes away when you feed it the pipeline's already-resized 2K frames instead of the raw 8K source. The original research target — featureless surfaces and thin structures — was never actually tested because the prerequisite (reliable scale) wasn't in place. It is now. The decision at the time was to stop fighting it and perfect the platform instead, which was the right call. But this is the most promising thing left to try.

**3DGS detour (commits `32729f6` – `5b11c7d`)** — Gaussian Splatting was integrated (first nerfstudio splatfacto, then graphdeco-inria's reference implementation). The training ran but the output rendering quality was inconsistent, and the pipeline to extract a usable point cloud from a trained Gaussian scene for coverage analysis was complex. The feature remains available (as `gaussian_splatting_task`) but is not in the standard chain.

### Local vs. paid VLM — a cost/consistency experiment

One hypothesis worth testing: maybe the inconsistency was partly a *cost* problem. Paid API calls are expensive enough that you use them sparingly and batch them, which limits how much you can iterate on prompts and retry on bad outputs. A local model running on-device has near-zero marginal cost per call — you can retry freely, run it on every frame, and experiment without watching a billing meter.

To test this, the pipeline was structured with two tiers:

- **Local (Ollama)** — `qwen2.5:7b` and `llama3.1:8b` running on the host, accessed via `host.docker.internal`. Used for high-frequency tasks: per-frame scene classification, dimension extraction, structured JSON parsing of detection results. Near-zero cost, low latency, unlimited retries.
- **External (Gemini)** — used selectively for tasks requiring stronger reasoning or vision capability: complex scene understanding, validating ambiguous detections, cases where the local model produced low-confidence output.

The result confirmed the hypothesis was wrong. **The bottleneck was not cost — it was consistency.** The local model produced wrong structured output just as often as the hosted one, just faster and cheaper. A confidently wrong JSON field from a free local inference is still a wrong field. Removing the cost constraint freed up experimentation bandwidth but did not change the fundamental failure mode.

The two-tier architecture does remain useful conceptually for tasks where *approximate* output is acceptable — the local model handles volume, the external API handles the cases that matter. But for pipeline stages where correctness is binary, neither tier is reliable enough without a human review step.

### Why we abandoned VLM-anchored scale

1. Fragile — worked in staged demos, failed in real scans
2. Slow — added 5–10 min to every pipeline run, even with local inference
3. VRAM pressure — LLaVA + DA2 + COLMAP = OOM risk on 12 GB cards
4. User friction — still required the user to confirm the detected scale factor
5. Most importantly: **a simpler, more reliable mechanism existed**

---

## The overhaul — ArUco markers (commit `2037804`)

### The insight

The scale problem is fundamentally an information problem: the video contains no absolute length reference. Rather than trying to extract one post-hoc with AI (unreliable), just **put a physical reference in the scene** before filming.

ArUco markers are small printed squares that:
- Are designed to be detectable by OpenCV in any lighting condition
- Come in standardised dictionaries (4×4 to 7×7 bits, up to 1000 unique IDs)
- Have a physical size you control (we default to 15 cm side)
- Provide not just detection but full pose estimation (solvePnP: rotation + translation)

The scale derivation becomes a clean geometric problem: triangulate each marker's 3D position from the SfM reconstruction, measure the inter-marker distance in SfM units, compare to the known physical distance, take the ratio.

### What changed

**Removed:**
- LLaVA (scene understanding)
- Grounding DINO (anchor detection)
- Depth Anything v2 (depth fusion)
- The `awaiting_scale_confirmation` status and user confirmation gate
- Anchor wizard UI
- `auto_detect_anchors_task`
- `depth_fusion_task`
- `scene_understanding_task`
- `apply_scale` (Phase 2 of old pipeline) → now `apply_known_scale_task` runs automatically

**Added:**
- `detect_aruco_task` (pre-SfM, FAIL FAST)
- `detect_aruco_sfm_task` (post-SfM, refined intrinsics)
- `scale_from_aruco_task` (triangulation + scale derivation)
- `fill_planes_task` (room layout, plane projection)
- `refine_cloud_task` (SOR + voxel downsample)

**Result:** The pipeline became fully automatic. Upload video or photos, get a metric-scaled point cloud, mesh, and Gaussian splat. No user intervention. The first real end-to-end run with a properly placed marker sheet yielded scale accurate to ~2%.

### What we kept

The geometry pipeline (LightGlue → SfM → MVS → coverage → export) was largely untouched. The smart frame selection and calibration photo support were added. The coverage scoring and re-shoot suggestions remained.

---

## Scout + Full mode

### The problem it solves

A full pipeline run takes 2–3 hours. If the footage quality is poor (low registration rate, high reprojection error), the full run wastes 2 hours to produce a poor reconstruction. You need early feedback.

### The design

A **scout run** takes 40 frames, uses a narrow matching window (5), skips MVS, and completes in ~15 minutes. It goes through the full geometry pipeline up to SfM, derives ArUco scale if possible, and then runs `scout_calibrate` to analyse quality metrics and tune the full run:

**Metrics measured:**
- Registration rate (% of 40 frames registered by SfM) — primary connectivity signal
- Mean reprojection error (pixels) — geometric accuracy of the sparse reconstruction
- Sparse point count — scene density proxy

**Parameters tuned (ranges):**
- `target_frames`: 120–400 (default 200) — scales inversely with registration rate; a scene that registered poorly at 40 frames needs more frames for loop closure
- `match_window`: 10–25 — wider window improves loop closure at the cost of runtime; widened when registration rate is low
- `mvs_min_consistent`: 2–4 — COLMAP's patch consistency threshold; tighter when geometry is clean, looser when coverage is sparse

After calibration, the full run is automatically dispatched with the calibrated parameters. The user sees the scout phase complete, the ScoutResultCard shows the calibration summary, and then the full pipeline progress begins.

### Implementation challenges

**Frontend state management** was the main difficulty. Two pipeline phases share the same stage IDs (`extract_metadata`, `sfm`, etc.), running as separate Celery chains. This caused:

1. `pipelineStatus` being set to `complete` after `scout_calibrate` emitted `progress: 1.0` (fixed: removed per-stage completion detection; polling drives final status)
2. Stage log overwriting itself when the full pipeline reused stage IDs (fixed: mode transition detection in polling resets the stage log)
3. `ProjectResponse` API missing `pipeline_mode` and `scout_calibration` fields (fixed: added both)
4. `export_outputs` using the unrefined layout cloud because `refine_cloud` only updated `dense_cloud_key` but not `scaled_cloud_key` (fixed: both keys updated)
5. Room layout RANSAC fallback running without metric scale, adding 150k synthetic floor fill points to an incorrectly-determined plane (fixed: RANSAC fallback requires `fitted_planes` to be non-empty)

---

## Trajectory correction

### The problem

COLMAP SfM can produce *teleport* artefacts in scenes with featureless spaces — a long corridor, a closet, a stairwell where the camera briefly loses tracking and gets re-registered far from where it should be. The result is a camera pose that jumps several metres in a single step, with the subsequent frames anchored to the wrong location. The dense cloud then splits into two overlapping copies of the same geometry at different positions: the classic "ghost wall" failure.

### The approach

After MVS, `trajectory_correction.py` scans the registered camera path for jump discontinuities — steps where the inter-frame distance is more than 10× the median step. For each detected jump, it:

1. Classifies which side is the "correct" anchor (using surrounding pose continuity)
2. Estimates an ICP correction transform between the dense point block after the jump and the block before it
3. Applies the rigid transform to the misregistered block if ICP fitness exceeds a threshold (0.5)

This corrects the cloud without touching the registered camera poses — the geometry moves, not the cameras.

### What we learned from test8

In a native-resolution run, a 2.531m jump was detected at frame 1016/1017 (25× the median step of 0.102m). The correction was applied to the last 205 frames (~1M points). ICP fitness was 0.696 but RMSE was 0.246m — too high for a room-scale scene. The ghost walls the correction was trying to fix were replaced by subtly misaligned walls: the block moved to roughly the right place, but not accurately enough.

The root cause was that the jump itself originated from SfM misregistration due to repetitive staircase geometry at native resolution — not a genuine physical teleport. At 2K resize (test7), the same footage produced a clean trajectory with no detected jumps. This suggests the trajectory correction is most useful for genuine featureless-space teleports, not for recovering from SfM initialisation failures caused by repetitive texture.

---

## What didn't make the final cut (still in codebase)

**MASt3R** — the model is installed at `/opt/mast3r` but never called. We investigated using it as an alternative to LightGlue + COLMAP for joint feature extraction and matching, which would have eliminated the separate SfM step by doing pose estimation and feature matching jointly. The adaptation required for our multi-frame batch workflow was significant, and the runtime on a single GPU was worse than the LightGlue + pycolmap combination. Left in place — the integration path is clear if someone wants to revisit it.

**3D Gaussian Splatting** — `gaussian_splatting_task` exists in `tasks.py` and the full training pipeline works (nerfstudio splatfacto, 15K iterations, ~15 min; outputs `.splat`, `.obj`, `.ply`). Notably, 3DGS handles featureless surfaces better than MVS — plain walls and ceilings that COLMAP leaves empty often reconstruct with reasonable density in a Gaussian scene. This makes it relevant to the original research target (featureless surfaces and thin structures). Removed from the standard chain because (a) extracting a metric-scaled, uniformly-sampled point cloud from a trained Gaussian scene for coverage analysis is a non-trivial pipeline, (b) VRAM during training conflicts badly with MVS if both run in the same session, (c) coverage quality for the survey use case (complete, uniform density) is still better from COLMAP MVS. The Gaussian path is a natural direction for further experimentation.

**Depth Anything v2** — removed but could be re-integrated. The core issue was reliability of the scale alignment on texture-poor scenes (<15 anchor points). With ArUco providing metric scale and post-COLMAP clouds being dense in textured regions, the marginal gain from DA2 in featureless areas doesn't justify its fragility for most scans — but it remains the most direct route toward filling the original research target.

---

## Current status and open questions

**Scale still fails when markers aren't co-visible.** If the scene only allows each marker to be filmed alone (no frame captures 2+ markers simultaneously), triangulation gives positions but no baselines. The single-marker solvePnP estimate (which would give scale from one marker's known size and estimated distance) is not currently used as a fallback in the post-SfM path. This is a known limitation; a fallback using the pre-SfM solvePnP estimates is on the roadmap.

**Room layout only runs with marker-derived planes.** Without a reliable floor plane from markers, the fill algorithm adds synthetic geometry in the wrong place. The fallback was disabled. If you want fill on a scan without coplanar markers, the room layout stage is effectively skipped.

**Coverage at < 50% → `needs_more`.** The threshold is hard-coded. For large outdoor scenes, 50% may be unreachable; for small object scans, 90%+ should be expected. An adaptive threshold based on scene type is desirable.

---

## Frame extraction evolution

**The original approach** used `stable_frame_timestamps` from `extract_metadata`: an optical flow scan of the video selected the least-motion frames, which were then used by `extract_frames` (Strategy A). The idea was to give COLMAP only sharp, stable frames.

**What went wrong:** the optical flow scan applied a global quality threshold that silently dropped entire video sections (e.g., when the camera was panning down to film the floor). SfM got good frames but they only covered part of the room.

Multiple iterations:
1. Global threshold → dropped critical sections
2. Windowed threshold (12s windows, then 4s windows) → better coverage but still dropped the minimum frames from high-motion sections via a secondary Laplacian filter
3. Final design: **always step-based, no secondary filter.** Take frames every N source frames, pick the sharpest of each burst of 5 (already what the original step-based fallback did). Flag bursts whose best frame is below a quality threshold in `blurry_sections` for user feedback. No section is ever skipped. The `stable_frame_timestamps` metadata is still computed and stored (for potential future guided-capture use) but is not used for extraction.

The optical flow scan adds ~6 minutes to `extract_metadata` on 8K footage. This compute runs regardless of whether the timestamps are used.

---

## /reprocess API endpoint

After enough debugging sessions of manually reconstructing `prev_result` from Redis task IDs and dispatching partial chains from inside the worker container, a proper `/reprocess` endpoint was added.

`POST /api/projects/{id}/reprocess?from_stage=scale_from_aruco` reads the checkpoint saved by `detect_aruco_sfm_task` (at `project:{id}:reprocess_checkpoint` in Redis) and dispatches the downstream chain from any stage. This makes it trivial to re-run geometry processing after a code fix without wasting 45 minutes on MVS again.

---

## SfM gap detection on camera path

The Scene Overview viewer already showed the camera trajectory. We added structured gap data to `cameras.json` (`gaps` field) so the viewer can draw a red segment between the last registered camera before a gap and the first after it. The red marker sits in 3D space exactly where the camera jumped without coverage, giving the user an unambiguous physical target for the re-shoot.

---

## TripoSR: single-image 3D reconstruction (June 2026)

After several iterations on the mesh pipeline (switching from Poisson to Ball Pivoting, adding connected-component filtering, hole filling, and Taubin smoothing), the mesh quality for the indoor lawn mower scan remained limited by the input data — dark environment, reflective black surfaces, incomplete orbital coverage. The question became: could a VLM-based generative 3D model fill in the missing geometry?

**The approach (TripoSR):** take the best-lit photo from the scan, run background removal (rembg), and pass it to [TripoSR](https://github.com/VAST-AI-Research/TripoSR) (Stability AI, single-image to 3D). The intent was to use the photogrammetry result for accurate scale and position, and the generative mesh for complete surface geometry.

**What happened:** the background removal worked well — clean isolation of the mower on white background. TripoSR inference ran in ~0.5s on an RTX 4070 Ti. The output was 50K vertices, 100K faces. The mesh was not recognisable as a lawn mower.

The failure modes were predictable in hindsight:
- TripoSR is trained on centred, front-facing product photos of symmetric objects (chairs, shoes, vases). A mower photographed diagonally from below with the handle extending off-frame is outside its training distribution.
- Complex geometry — four wheels, an irregular body, a long handle at an oblique angle — is outside what the model handles well.
- Models like TripoSR and InstantMesh produce plausible completions for canonical-pose product imagery; they do not generalise to arbitrary real-world equipment in cluttered environments.

**The honest conclusion:** approach 2 (generative 3D) joins approach 1 (monocular depth estimation, tried earlier) as a VLM technique that doesn't survive contact with real scan conditions. The information problem here is the same as always — dark surfaces, incomplete coverage — and the right solution remains capturing better source data, not inferring missing geometry after the fact.

The more promising direction, not yet implemented: **Depth Anything V2 fused with the MVS cloud**, using the known camera poses to back-project dense depth maps from each photo into 3D space. This adds geometry where MVS failed (dark areas, featureless surfaces) using learned depth priors without hallucinating shape. The key difference from approach 1 (where this was tried and deemed "not useful"): using per-frame poses for back-projection rather than a single-image standalone depth map. Whether this actually closes the gap for dark indoor scans is still an open question.
