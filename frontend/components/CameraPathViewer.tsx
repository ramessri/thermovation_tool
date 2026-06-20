'use client';

/**
 * CameraPathViewer
 *
 * Renders the photographer's trajectory through the scene using data from
 * cameras.json (SfM output). Shows:
 *   - Sparse point cloud as small grey dots
 *   - Camera positions as coloured spheres (green → red along the path)
 *   - A line connecting cameras in recording order
 *   - Camera frustums (small pyramids) showing where each shot was aimed
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { PLYLoader } from 'three-stdlib';
import { OrbitControls } from 'three-stdlib';
import { worldPosition, worldDirection } from './lib/poseMath';

interface Suggestion {
  position: number[];   // [x, y, z] in scaled (mm) units
  direction: number[];  // [dx, dy, dz]
  message: string;
}

interface Props {
  sparseCloudUrl: string | null;       // URL to cloud (coverage or sparse)
  fallbackSparseUrl?: string | null;   // fallback if coverage cloud not yet available
  camerasJsonUrl: string | null;       // URL to cameras.json
  suggestions?: Suggestion[];          // re-shoot suggestions to visualise
  scaleFactor?: number;                // to convert suggestion coords → SfM units
}

function pathColor(t: number): THREE.Color {
  // Green (start) → yellow (mid) → red (end)
  const c = new THREE.Color();
  if (t < 0.5) c.setRGB(t * 2, 1, 0);
  else         c.setRGB(1, 1 - (t - 0.5) * 2, 0);
  return c;
}

export function CameraPathViewer({ sparseCloudUrl, fallbackSparseUrl, camerasJsonUrl, suggestions = [], scaleFactor = 1 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus]   = useState<string>('Loading…');
  const [error, setError]     = useState<string | null>(null);
  const [gapCount, setGapCount] = useState(0);

  useEffect(() => {
    if (!camerasJsonUrl) { setError('No camera data available'); return; }
    const container = containerRef.current;
    if (!container) return;

    let animId = 0;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x0f172a);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(1, 2, 3);
    scene.add(dirLight);

    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.001, 100000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    (async () => {
      try {
        // 1. Load cameras.json
        setStatus('Loading camera data…');
        const res = await fetch(camerasJsonUrl);
        if (!res.ok) throw new Error(`cameras.json fetch failed: ${res.status}`);
        const camsData = await res.json();
        const images: any[] = camsData.images ?? [];

        if (images.length === 0) throw new Error('No registered cameras found');

        // Sort by image name (frame order)
        images.sort((a, b) => a.name.localeCompare(b.name));

        // 2. Load coverage cloud (preferred) or sparse cloud as fallback
        const cloudUrls = [sparseCloudUrl, fallbackSparseUrl].filter(Boolean) as string[];
        for (const cloudUrl of cloudUrls) {
          setStatus('Loading scene cloud…');
          let loaded = false;
          await new Promise<void>((resolve) => {
            const loader = new PLYLoader();
            loader.load(cloudUrl, (geo) => {
              geo.computeVertexNormals();
              const hasColor = geo.attributes.color != null;
              const mat = new THREE.PointsMaterial({
                size: hasColor ? 0.008 : 0.005,
                vertexColors: hasColor,
                color: hasColor ? undefined : 0x6b7280,
                sizeAttenuation: true,
              });
              scene.add(new THREE.Points(geo, mat));
              loaded = true;
              resolve();
            }, undefined, () => resolve());
          });
          if (loaded) break; // stop at first successful load
        }

        setStatus('Building camera path…');

        // 3. Compute camera world positions
        const positions: THREE.Vector3[] = [];
        const directions: THREE.Vector3[] = [];
        for (const img of images) {
          const m = img.cam_from_world?.matrix_3x4;
          if (!m) continue;
          positions.push(worldPosition(m));
          directions.push(worldDirection(m));
        }

        if (positions.length === 0) throw new Error('Could not parse camera poses');

        // Estimate scene scale for sizing markers
        const box = new THREE.Box3();
        positions.forEach(p => box.expandByPoint(p));
        const diagonal = box.getSize(new THREE.Vector3()).length();
        const sphereR = diagonal * 0.005;
        const frustumSize = diagonal * 0.02;

        // 4. Path line
        const lineGeo = new THREE.BufferGeometry().setFromPoints(positions);
        const lineColors: number[] = [];
        positions.forEach((_, i) => {
          const c = pathColor(i / Math.max(positions.length - 1, 1));
          lineColors.push(c.r, c.g, c.b);
        });
        lineGeo.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));
        scene.add(new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 })));

        // 5. Camera spheres + frustum cones
        const sphereGeo = new THREE.SphereGeometry(sphereR, 8, 8);
        const coneGeo = new THREE.ConeGeometry(frustumSize * 0.4, frustumSize, 4);

        positions.forEach((pos, i) => {
          const t = i / Math.max(positions.length - 1, 1);
          const col = pathColor(t);
          const mat = new THREE.MeshStandardMaterial({ color: col });

          // Sphere at camera centre
          const sphere = new THREE.Mesh(sphereGeo, mat);
          sphere.position.copy(pos);
          scene.add(sphere);

          // Small cone showing view direction
          const cone = new THREE.Mesh(coneGeo, new THREE.MeshStandardMaterial({ color: col, opacity: 0.6, transparent: true }));
          cone.position.copy(pos);
          // Align cone tip with view direction
          const dir = directions[i];
          const axis = new THREE.Vector3(0, 1, 0);
          cone.quaternion.setFromUnitVectors(axis, dir);
          cone.translateOnAxis(dir, frustumSize * 0.5);
          scene.add(cone);
        });

        // 6. Start/end markers
        const startMat = new THREE.MeshStandardMaterial({ color: 0x22c55e, emissive: 0x22c55e, emissiveIntensity: 0.4 });
        const endMat   = new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xef4444, emissiveIntensity: 0.4 });
        const markerGeo = new THREE.SphereGeometry(sphereR * 2.5, 12, 12);
        const startMarker = new THREE.Mesh(markerGeo, startMat);
        startMarker.position.copy(positions[0]);
        scene.add(startMarker);
        const endMarker = new THREE.Mesh(markerGeo, endMat);
        endMarker.position.copy(positions[positions.length - 1]);
        scene.add(endMarker);

        // 7. Re-shoot suggestion arrows (orange cones + pulsing sphere)
        if (suggestions.length > 0) {
          const sf = scaleFactor || 1;
          const sugGeo = new THREE.SphereGeometry(sphereR * 3, 12, 12);
          const sugConGeo = new THREE.ConeGeometry(frustumSize * 0.6, frustumSize * 1.5, 6);
          suggestions.forEach((sug) => {
            if (!sug.position || sug.position.length < 3) return;
            // Convert from scaled (mm) to SfM units
            const pos = new THREE.Vector3(
              sug.position[0] / sf,
              sug.position[1] / sf,
              sug.position[2] / sf,
            );
            // Pulsing orange sphere
            const sugMat = new THREE.MeshStandardMaterial({
              color: 0xf97316, emissive: 0xf97316, emissiveIntensity: 0.5,
            });
            const sphere = new THREE.Mesh(sugGeo, sugMat);
            sphere.position.copy(pos);
            scene.add(sphere);
            // Arrow cone pointing toward the under-covered area
            if (sug.direction && sug.direction.length >= 3) {
              const dir = new THREE.Vector3(...(sug.direction as [number,number,number])).normalize();
              const cone = new THREE.Mesh(sugConGeo,
                new THREE.MeshStandardMaterial({ color: 0xf97316, opacity: 0.8, transparent: true }));
              cone.position.copy(pos);
              const axis = new THREE.Vector3(0, 1, 0);
              cone.quaternion.setFromUnitVectors(axis, dir);
              cone.translateOnAxis(dir, frustumSize * 0.8);
              scene.add(cone);
            }
            // Index label as tiny sphere
            const labelGeo = new THREE.SphereGeometry(sphereR * 1.5, 6, 6);
            const labelMat = new THREE.MeshStandardMaterial({ color: 0xfbbf24 });
            const label = new THREE.Mesh(labelGeo, labelMat);
            label.position.copy(pos.clone().addScaledVector(new THREE.Vector3(0, 1, 0), sphereR * 5));
            scene.add(label);
          });
        }

        // 8. SfM gap segments — red dashed line + warning marker between bracketing cameras
        const gaps: {before_frame?: string; after_frame?: string; n_missing: number; pct_start: number; pct_end: number}[] =
          camsData.gaps ?? [];
        const nameToPos = new Map<string, THREE.Vector3>(
          images
            .map((img, i) => [img.name as string, positions[i]] as [string, THREE.Vector3])
            .filter(([, p]) => p != null)
        );

        setGapCount(gaps.length);
        gaps.forEach((gap) => {
          const before = gap.before_frame ? nameToPos.get(gap.before_frame) : null;
          const after  = gap.after_frame  ? nameToPos.get(gap.after_frame)  : null;
          if (!before || !after) return;

          // Red dashed segment between the two bracketing cameras
          const gapGeo = new THREE.BufferGeometry().setFromPoints([before, after]);
          scene.add(new THREE.Line(gapGeo,
            new THREE.LineBasicMaterial({ color: 0xef4444, linewidth: 3 })));

          // Warning sphere at the midpoint
          const mid = before.clone().lerp(after, 0.5);
          const warnGeo  = new THREE.SphereGeometry(sphereR * 3.5, 12, 12);
          const warnMat  = new THREE.MeshStandardMaterial({
            color: 0xef4444, emissive: 0xef4444, emissiveIntensity: 0.6,
          });
          const warnMesh = new THREE.Mesh(warnGeo, warnMat);
          warnMesh.position.copy(mid);
          scene.add(warnMesh);
        });

        // 9. Frame camera on scene
        const centre = box.getCenter(new THREE.Vector3());
        camera.position.set(centre.x, centre.y + diagonal * 0.5, centre.z + diagonal * 1.2);
        camera.lookAt(centre);
        controls.target.copy(centre);
        controls.update();

        const gapCount = gaps.length;
        setStatus(
          `${positions.length} cameras · ${images.length} frames` +
          (suggestions.length ? ` · ${suggestions.length} re-shoot suggestion${suggestions.length > 1 ? 's' : ''}` : '') +
          (gapCount ? ` · ⚠ ${gapCount} coverage gap${gapCount > 1 ? 's' : ''}` : '')
        );
      } catch (e: any) {
        setError(e.message ?? 'Failed to load camera path');
        setStatus('');
      }
    })();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [sparseCloudUrl, camerasJsonUrl]);

  return (
    <div style={{ width: '100%', height: '24rem', background: '#0f172a', position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      {/* Status overlay */}
      <div className="absolute bottom-3 left-3 pointer-events-none">
        {error ? (
          <span className="text-xs text-red-400 bg-black/50 px-2 py-1 rounded">{error}</span>
        ) : (
          <span className="text-xs text-slate-400 bg-black/50 px-2 py-1 rounded">{status}</span>
        )}
      </div>
      {/* Legend */}
      {!error && (
        <div className="absolute top-3 right-3 flex flex-col gap-1 pointer-events-none">
          <div className="flex items-center gap-1.5 text-xs text-white/80 bg-black/50 px-2 py-1 rounded">
            <span className="h-2 w-2 rounded-full bg-green-400 inline-block" /> Start
          </div>
          <div className="flex items-center gap-1.5 text-xs text-white/80 bg-black/50 px-2 py-1 rounded">
            <span className="h-2 w-2 rounded-full bg-red-400 inline-block" /> End
          </div>
          {suggestions.length > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-white/80 bg-black/50 px-2 py-1 rounded">
              <span className="h-2 w-2 rounded-full bg-orange-400 inline-block" /> Re-shoot
            </div>
          )}
          {gapCount > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-white/80 bg-black/50 px-2 py-1 rounded">
              <span className="h-2 w-2 rounded-full bg-red-500 inline-block" /> Coverage gap
            </div>
          )}
        </div>
      )}
    </div>
  );
}
