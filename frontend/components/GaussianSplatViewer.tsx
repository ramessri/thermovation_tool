'use client';

/**
 * GaussianSplatViewer
 *
 * Renders a .splat file in real-time using the @mkkellogg/gaussian-splats-3d
 * library. Optionally overlays camera trajectory + re-shoot arrows on top.
 *
 * The library owns the renderer/scene/camera but lets us add overlays via
 * threeScene + onUpdate hooks.
 */

import { useEffect, useRef, useState } from 'react';

interface Suggestion {
  position: number[];
  direction: number[];
  message: string;
}

interface Props {
  splatUrl: string;
  camerasJsonUrl?: string | null;
  suggestions?: Suggestion[];
  scaleFactor?: number;
}

export function GaussianSplatViewer({ splatUrl, camerasJsonUrl, suggestions = [], scaleFactor = 1 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef    = useRef<any>(null);
  const [status, setStatus] = useState<string>('Loading 3DGS viewer…');
  const [error, setError]   = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current || !splatUrl) return;
    let cancelled = false;

    (async () => {
      try {
        // Dynamic import — gaussian-splats-3d depends on browser globals
        const GaussianSplats3D = await import('@mkkellogg/gaussian-splats-3d');
        const THREE            = await import('three');

        const viewer = new GaussianSplats3D.Viewer({
          rootElement: containerRef.current!,
          cameraUp:     [0, 1, 0],
          initialCameraPosition:  [0, 0, 3],
          initialCameraLookAt:    [0, 0, 0],
          sharedMemoryForWorkers: false,
        });
        viewerRef.current = viewer;

        setStatus('Loading splat scene…');
        await viewer.addSplatScene(splatUrl, {
          showLoadingUI: false,
          progressiveLoad: true,
        });

        if (cancelled) return;

        // Add camera path + suggestions as overlay meshes in the viewer's scene
        const threeScene = viewer.threeScene;
        if (camerasJsonUrl) {
          try {
            const res = await fetch(camerasJsonUrl);
            const cams = await res.json();
            const positions: any[] = [];
            for (const img of cams.images || []) {
              const m = img?.cam_from_world?.matrix_3x4;
              if (!m) continue;
              const R = [[m[0][0], m[0][1], m[0][2]],
                         [m[1][0], m[1][1], m[1][2]],
                         [m[2][0], m[2][1], m[2][2]]];
              const t = [m[0][3], m[1][3], m[2][3]];
              // C = -R^T t
              const pos = [
                -(R[0][0]*t[0] + R[1][0]*t[1] + R[2][0]*t[2]),
                -(R[0][1]*t[0] + R[1][1]*t[1] + R[2][1]*t[2]),
                -(R[0][2]*t[0] + R[1][2]*t[1] + R[2][2]*t[2]),
              ];
              positions.push(new (THREE as any).Vector3(...pos));
            }
            if (positions.length > 1) {
              const lineGeo = new (THREE as any).BufferGeometry().setFromPoints(positions);
              const colors = new Float32Array(positions.length * 3);
              for (let i = 0; i < positions.length; i++) {
                const f = i / (positions.length - 1);
                colors[i*3]   = f;       // R goes 0→1
                colors[i*3+1] = 1 - f;   // G goes 1→0
                colors[i*3+2] = 0;
              }
              lineGeo.setAttribute('color', new (THREE as any).BufferAttribute(colors, 3));
              const lineMat = new (THREE as any).LineBasicMaterial({ vertexColors: true, linewidth: 2 });
              const line = new (THREE as any).Line(lineGeo, lineMat);
              threeScene.add(line);
            }
          } catch (e) {
            console.warn('Camera path overlay failed:', e);
          }
        }

        // Re-shoot suggestion arrows
        if (suggestions.length > 0 && scaleFactor) {
          for (const sug of suggestions) {
            if (!sug.position || sug.position.length < 3) continue;
            const pos = new (THREE as any).Vector3(
              sug.position[0] / scaleFactor,
              sug.position[1] / scaleFactor,
              sug.position[2] / scaleFactor,
            );
            const geo = new (THREE as any).SphereGeometry(0.05, 12, 12);
            const mat = new (THREE as any).MeshStandardMaterial({
              color: 0xf97316, emissive: 0xf97316, emissiveIntensity: 0.6,
            });
            const sphere = new (THREE as any).Mesh(geo, mat);
            sphere.position.copy(pos);
            threeScene.add(sphere);
          }
        }

        viewer.start();
        setStatus(`Splat loaded${suggestions.length ? ` · ${suggestions.length} re-shoot suggestion${suggestions.length > 1 ? 's' : ''}` : ''}`);
      } catch (e: any) {
        console.error('GaussianSplatViewer error:', e);
        setError(`Failed to load splat: ${e?.message ?? String(e)}`);
      }
    })();

    return () => {
      cancelled = true;
      const viewer = viewerRef.current;
      if (viewer) {
        try {
          viewer.stop();
          viewer.dispose?.();
        } catch (e) {
          console.warn('Viewer dispose failed:', e);
        }
        viewerRef.current = null;
      }
      // The library appends its canvas to rootElement — clear it
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
    };
  }, [splatUrl, camerasJsonUrl, JSON.stringify(suggestions), scaleFactor]);

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="absolute inset-0" />
      {error ? (
        <div className="absolute top-2 left-2 text-xs text-rose-300 bg-rose-950/80 px-2 py-1 rounded">
          {error}
        </div>
      ) : (
        <div className="absolute top-2 left-2 text-xs text-white/80 bg-black/60 px-2 py-1 rounded pointer-events-none">
          {status}
        </div>
      )}
    </div>
  );
}
