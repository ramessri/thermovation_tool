'use client';

/**
 * CameraWalkthroughViewer
 *
 * Stand at a recorded SfM camera and look at the reconstruction the way the
 * operator saw it while filming. Pick an evenly-spaced subset of the registered
 * camera poses; step through them (Prev/Next); look around in place (rotate
 * only — the camera position never moves); toggle the background geometry
 * between point cloud and mesh; and overlay the actual extracted video frame
 * (opacity-blended) to ground-truth the reconstruction against reality.
 *
 * Alignment: the geometry exports are metric-scaled but raw SfM poses are not,
 * so camera centres are multiplied by `scaleFactor` and then pushed through the
 * geometry's own world matrix (centre + COLMAP→Three.js X-flip), guaranteeing
 * the camera lands in exactly the same scene space as the geometry. FOV is set
 * from the camera's real intrinsics so the render registers with the photo.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { PLYLoader, OBJLoader } from 'three-stdlib';
import { Maximize, Minimize } from 'lucide-react';
import { worldPosition, worldBasis } from './lib/poseMath';

type Mode = 'cloud' | 'mesh';

interface CameraIntrinsics {
  model: string;
  width: number;
  height: number;
  params: number[];
}

interface Pose {
  name: string;            // frame filename, e.g. frame_000123.jpg
  matrix3x4: number[][];   // cam_from_world
  intr: CameraIntrinsics | null;
}

interface Props {
  cloudUrl: string | null;
  meshUrl: string | null;
  framesBaseUrl: string;   // `${apiBase}/files/${projectId}/frames`
  camerasJsonUrl: string | null;
  scaleFactor?: number;
}

const TARGET_POSES = 9;
const LOOK_SENS = 0.005;            // rad per pixel
const PITCH_LIMIT = THREE.MathUtils.degToRad(89);

/** Vertical FOV (degrees) from intrinsics; falls back to 60° if unavailable. */
function fovYFromIntrinsics(intr: CameraIntrinsics | null): number {
  if (!intr || !intr.params?.length || !intr.height) return 60;
  const m = intr.model || '';
  // PINHOLE/OPENCV-family store [fx, fy, ...]; simple models store [f, ...].
  const fy = (m === 'PINHOLE' || m.startsWith('OPENCV') || m.startsWith('FULL_OPENCV'))
    ? intr.params[1]
    : intr.params[0];
  if (!fy || fy <= 0) return 60;
  return THREE.MathUtils.radToDeg(2 * Math.atan(intr.height / (2 * fy)));
}

function intrinsicAspect(intr: CameraIntrinsics | null): number {
  if (intr && intr.width && intr.height) return intr.width / intr.height;
  return 16 / 9;
}

/** Fit a rect of the given aspect inside (w, h), centred (letterbox). */
function fitRect(w: number, h: number, aspect: number) {
  let rw = w, rh = w / aspect;
  if (rh > h) { rh = h; rw = h * aspect; }
  return { width: Math.round(rw), height: Math.round(rh) };
}

export function CameraWalkthroughViewer({
  cloudUrl, meshUrl, framesBaseUrl, camerasJsonUrl, scaleFactor = 1,
}: Props) {
  const rootRef      = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef     = useRef<HTMLDivElement>(null);

  const [ready, setReady]       = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [status, setStatus]     = useState('Loading…');
  const [mode, setMode]         = useState<Mode>('cloud');
  const [compare, setCompare]   = useState(false);
  const [opacity, setOpacity]   = useState(0.5);
  const [index, setIndex]       = useState(0);
  const [count, setCount]       = useState(0);
  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  const [imgAspect, setImgAspect] = useState(16 / 9);
  const [fullscreen, setFullscreen] = useState(false);

  // three.js objects (imperative — kept in refs across renders)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef    = useRef<THREE.Scene | null>(null);
  const cameraRef   = useRef<THREE.PerspectiveCamera | null>(null);
  const geomCache   = useRef<Partial<Record<Mode, THREE.Object3D>>>({});
  const currentGeom = useRef<THREE.Object3D | null>(null);
  const sceneMatrix = useRef<THREE.Matrix4>(new THREE.Matrix4());

  const posesRef = useRef<Pose[]>([]);
  const indexRef = useRef(0);
  const q0Ref    = useRef<THREE.Quaternion>(new THREE.Quaternion());
  const yawRef   = useRef(0);
  const pitchRef = useRef(0);

  // ── Place the camera at the active pose using the current geometry matrix ──
  const placeActive = useCallback(() => {
    const cam = cameraRef.current;
    const pose = posesRef.current[indexRef.current];
    if (!cam || !pose) return;
    const S = sceneMatrix.current;

    // Position: metric-scaled centre → geometry world space.
    const cMetric = worldPosition(pose.matrix3x4).multiplyScalar(scaleFactor);
    cam.position.copy(cMetric.applyMatrix4(S));

    // Orientation from the SfM basis, rotated into geometry world space.
    const Rs = new THREE.Matrix4().extractRotation(S);
    const { right, up, forward } = worldBasis(pose.matrix3x4);
    right.transformDirection(Rs);
    up.transformDirection(Rs);
    forward.transformDirection(Rs);
    // Three.js camera looks down -Z; local +Z maps to -forward.
    const basis = new THREE.Matrix4().makeBasis(right, up, forward.clone().negate());
    q0Ref.current.setFromRotationMatrix(basis);

    yawRef.current = 0;
    pitchRef.current = 0;
    cam.fov = fovYFromIntrinsics(pose.intr);
    cam.updateProjectionMatrix();
  }, [scaleFactor]);

  // ── Resize renderer/stage to a letterboxed rect matching the photo aspect ──
  const resize = useCallback(() => {
    const container = containerRef.current;
    const stage = stageRef.current;
    const renderer = rendererRef.current;
    const cam = cameraRef.current;
    if (!container || !stage || !renderer || !cam) return;
    const { width, height } = fitRect(container.clientWidth, container.clientHeight, imgAspect);
    stage.style.width = `${width}px`;
    stage.style.height = `${height}px`;
    renderer.setSize(width, height);
    cam.aspect = imgAspect;
    cam.updateProjectionMatrix();
  }, [imgAspect]);

  // ── Fullscreen toggle ──────────────────────────────────────────────────────
  const toggleFullscreen = useCallback(() => {
    const root = rootRef.current;
    if (!root) return;
    if (!document.fullscreenElement) {
      root.requestFullscreen?.().catch(() => {});
    } else {
      document.exitFullscreen?.().catch(() => {});
    }
  }, []);

  useEffect(() => {
    const onChange = () => {
      setFullscreen(!!document.fullscreenElement);
      // Container size changes on enter/exit — re-letterbox once layout settles.
      requestAnimationFrame(resize);
    };
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, [resize]);

  // ── One-time scene setup + cameras.json load ──────────────────────────────
  useEffect(() => {
    if (!camerasJsonUrl) { setError('No camera data available'); return; }
    const container = containerRef.current;
    const stage = stageRef.current;
    if (!container || !stage) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x0f172a);
    stage.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dl = new THREE.DirectionalLight(0xffffff, 0.7);
    dl.position.set(1, 2, 3);
    scene.add(dl);
    sceneRef.current = scene;

    const cam = new THREE.PerspectiveCamera(60, 1, 0.001, 100000);
    cameraRef.current = cam;

    // Free-look: drag rotates in place; position never changes.
    let dragging = false, lastX = 0, lastY = 0;
    const el = renderer.domElement;
    const onDown = (e: PointerEvent) => { dragging = true; lastX = e.clientX; lastY = e.clientY; el.setPointerCapture(e.pointerId); };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      yawRef.current -= dx * LOOK_SENS;
      pitchRef.current = THREE.MathUtils.clamp(pitchRef.current - dy * LOOK_SENS, -PITCH_LIMIT, PITCH_LIMIT);
    };
    const onUp = (e: PointerEvent) => { dragging = false; try { el.releasePointerCapture(e.pointerId); } catch {} };
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerup', onUp);
    el.addEventListener('pointerleave', onUp);

    const tmpEuler = new THREE.Euler(0, 0, 0, 'YXZ');
    const tmpQ = new THREE.Quaternion();
    let animId = 0;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      tmpEuler.set(pitchRef.current, yawRef.current, 0, 'YXZ');
      tmpQ.setFromEuler(tmpEuler);
      cam.quaternion.copy(q0Ref.current).multiply(tmpQ);
      renderer.render(scene, cam);
    };
    animate();

    const ro = new ResizeObserver(() => resize());
    ro.observe(container);

    (async () => {
      try {
        const res = await fetch(camerasJsonUrl);
        if (!res.ok) throw new Error(`cameras.json fetch failed: ${res.status}`);
        const data = await res.json();
        const images: any[] = data.images ?? [];
        if (images.length === 0) throw new Error('No registered cameras found');
        images.sort((a, b) => String(a.name).localeCompare(String(b.name)));

        const intrById = new Map<number, CameraIntrinsics>();
        for (const c of (data.cameras ?? [])) {
          intrById.set(c.camera_id, { model: c.model, width: c.width, height: c.height, params: c.params });
        }

        const step = Math.max(1, Math.floor(images.length / TARGET_POSES));
        const poses: Pose[] = [];
        for (let i = 0; i < images.length; i += step) {
          const img = images[i];
          const m = img.cam_from_world?.matrix_3x4;
          if (!m) continue;
          poses.push({ name: img.name, matrix3x4: m, intr: intrById.get(img.camera_id) ?? null });
        }
        if (poses.length === 0) throw new Error('Could not parse camera poses');

        posesRef.current = poses;
        setCount(poses.length);
        setReady(true);
      } catch (e: any) {
        setError(e?.message ?? 'Failed to load camera data');
      }
    })();

    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerup', onUp);
      el.removeEventListener('pointerleave', onUp);
      Object.values(geomCache.current).forEach(o => {
        o?.traverse((c: any) => { c.geometry?.dispose?.(); c.material?.dispose?.(); });
      });
      geomCache.current = {};
      renderer.dispose();
      try { stage.removeChild(renderer.domElement); } catch {}
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camerasJsonUrl]);

  // ── Load (or swap to) the geometry for the current mode ───────────────────
  useEffect(() => {
    if (!ready) return;
    const scene = sceneRef.current;
    if (!scene) return;
    const url = mode === 'cloud' ? cloudUrl : meshUrl;
    if (!url) return;

    let cancelled = false;

    const orient = (obj: THREE.Object3D) => {
      const box = new THREE.Box3().setFromObject(obj);
      const centre = box.getCenter(new THREE.Vector3());
      obj.position.sub(centre);
      // COLMAP Y-down/Z-forward → Three.js Y-up: rotate 180° about X.
      obj.rotation.x = Math.PI;
      obj.updateMatrixWorld(true);
    };

    const install = (obj: THREE.Object3D) => {
      if (cancelled) return;
      if (currentGeom.current && currentGeom.current !== obj) scene.remove(currentGeom.current);
      scene.add(obj);
      currentGeom.current = obj;
      sceneMatrix.current.copy(obj.matrixWorld);
      placeActive();
      setStatus(mode === 'cloud' ? 'Point cloud' : 'Mesh');
    };

    const cached = geomCache.current[mode];
    if (cached) { install(cached); return () => { cancelled = true; }; }

    setStatus('Loading geometry…');
    if (mode === 'cloud') {
      new PLYLoader().load(url, (geo) => {
        if (cancelled) return;
        geo.computeVertexNormals();
        const hasColor = !!geo.attributes.color;
        const obj = new THREE.Points(geo, new THREE.PointsMaterial({
          size: 0.004, vertexColors: hasColor,
          color: hasColor ? undefined : 0x64748b, sizeAttenuation: true,
        }));
        orient(obj);
        geomCache.current.cloud = obj;
        install(obj);
      }, undefined, () => !cancelled && setError('Could not load point cloud'));
    } else {
      new OBJLoader().load(url, (obj) => {
        if (cancelled) return;
        obj.traverse((c) => {
          if (c instanceof THREE.Mesh) {
            c.material = new THREE.MeshStandardMaterial({
              color: 0x94a3b8, roughness: 0.7, metalness: 0.1, side: THREE.DoubleSide,
            });
          }
        });
        orient(obj);
        geomCache.current.mesh = obj;
        install(obj);
      }, undefined, () => !cancelled && setError('Could not load mesh'));
    }
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, mode, cloudUrl, meshUrl]);

  // ── React to pose changes: reposition camera, update photo + stage aspect ─
  useEffect(() => {
    if (!ready) return;
    indexRef.current = index;
    const pose = posesRef.current[index];
    if (!pose) return;
    placeActive();
    setPhotoUrl(`${framesBaseUrl}/${pose.name}`);
    setImgAspect(intrinsicAspect(pose.intr));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, ready]);

  // Re-letterbox whenever the target aspect changes.
  useEffect(() => { resize(); }, [imgAspect, resize]);

  const step = (delta: number) => {
    setIndex(i => Math.min(Math.max(i + delta, 0), Math.max(count - 1, 0)));
  };

  const btn = 'px-2 py-1 rounded text-xs border border-slate-600 text-slate-300 hover:text-white hover:bg-slate-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed';

  return (
    <div
      ref={rootRef}
      style={{
        width: '100%',
        height: fullscreen ? '100vh' : '100%',
        position: 'relative',
        background: '#0f172a',
      }}
    >
      {/* Letterboxed stage centred in the container; renderer canvas + photo overlay live here */}
      <div ref={containerRef} style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div ref={stageRef} style={{ position: 'relative' }}>
          {compare && photoUrl && (
            <img
              src={photoUrl}
              alt="recorded frame"
              style={{
                position: 'absolute', inset: 0, width: '100%', height: '100%',
                objectFit: 'contain', opacity, pointerEvents: 'none',
              }}
            />
          )}
        </div>
      </div>

      {/* Top controls */}
      {!error && (
        <div className="absolute top-3 left-3 flex items-center gap-2">
          <button className={btn} onClick={() => step(-1)} disabled={index <= 0}>‹ Prev</button>
          <span className="text-xs text-white/80 bg-black/50 px-2 py-1 rounded">Camera {count ? index + 1 : 0} / {count}</span>
          <button className={btn} onClick={() => step(1)} disabled={index >= count - 1}>Next ›</button>
        </div>
      )}

      {/* Right controls */}
      {!error && (
        <div className="absolute top-3 right-3 flex flex-col items-end gap-2">
          <div className="flex gap-1">
            <button className={`${btn} ${mode === 'cloud' ? 'bg-slate-700 text-white' : ''}`} onClick={() => setMode('cloud')}>Point Cloud</button>
            <button className={`${btn} ${mode === 'mesh' ? 'bg-slate-700 text-white' : ''}`} onClick={() => setMode('mesh')} disabled={!meshUrl}>Mesh</button>
          </div>
          <button className={`${btn} ${compare ? 'bg-slate-700 text-white' : ''}`} onClick={() => setCompare(c => !c)}>
            {compare ? 'Hide photo' : 'Compare to photo'}
          </button>
          {compare && (
            <div className="flex items-center gap-1 bg-black/50 px-2 py-1 rounded">
              <span className="text-[10px] text-slate-400">Photo</span>
              <input type="range" min={0} max={1} step={0.01} value={opacity}
                     onChange={e => setOpacity(parseFloat(e.target.value))} className="w-24" />
            </div>
          )}
          <button className={btn} onClick={toggleFullscreen} title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}>
            {fullscreen ? <Minimize className="h-3.5 w-3.5" /> : <Maximize className="h-3.5 w-3.5" />}
          </button>
        </div>
      )}

      {/* Status / error */}
      <div className="absolute bottom-3 left-3 pointer-events-none">
        {error
          ? <span className="text-xs text-red-400 bg-black/50 px-2 py-1 rounded">{error}</span>
          : <span className="text-xs text-slate-400 bg-black/50 px-2 py-1 rounded">{status}</span>}
      </div>
    </div>
  );
}
