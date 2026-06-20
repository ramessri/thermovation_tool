'use client';

/**
 * CompareViewer — side-by-side or toggle comparison between
 * the raw MVS dense cloud (SfM units) and the metric-scaled cloud.
 *
 * Shows point counts for both so the user can see that scaling is
 * purely a coordinate transform (same points, different units).
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { PLYLoader } from 'three-stdlib';
import { OrbitControls } from 'three-stdlib';

interface CloudInfo {
  label: string;
  url: string;
  color: string;  // accent colour for the label
}

interface Stats {
  points: number;
  loaded: boolean;
  error?: string;
}

// Shared orbit-control state so both panels stay in sync when linked
function useLinkedControls() {
  const stateRef = useRef<{ quat: THREE.Quaternion; target: THREE.Vector3; zoom: number } | null>(null);
  return stateRef;
}

function CloudCanvas({
  cloud,
  sharedState,
  onStats,
}: {
  cloud: CloudInfo;
  sharedState: React.MutableRefObject<{ quat: THREE.Quaternion; target: THREE.Vector3; zoom: number } | null>;
  onStats: (s: Stats) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x0f172a);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.001, 100000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    let animId = 0;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      // Sync camera state for linked panning
      if (sharedState.current) {
        // read from shared (other panel may have written)
      }
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Load PLY
    const loader = new PLYLoader();
    loader.load(cloud.url, (geo) => {
      geo.computeVertexNormals();
      const hasColor = !!geo.attributes.color;
      const pts = geo.attributes.position?.count ?? 0;
      const mat = new THREE.PointsMaterial({
        size: 0.003,
        vertexColors: hasColor,
        color: hasColor ? undefined : 0x94a3b8,
        sizeAttenuation: true,
      });
      const mesh = new THREE.Points(geo, mat);
      scene.add(mesh);

      const box = new THREE.Box3().setFromObject(mesh);
      const centre = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();
      mesh.position.sub(centre);
      camera.position.set(0, 0, size * 1.5);
      camera.near = size * 0.001;
      camera.far = size * 10;
      camera.updateProjectionMatrix();
      controls.update();

      onStats({ points: pts, loaded: true });
    }, undefined, () => {
      onStats({ points: 0, loaded: true, error: 'File not available yet' });
    });

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      controls.dispose();
      renderer.dispose();
      try { container.removeChild(renderer.domElement); } catch {}
    };
  }, [cloud.url]);

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;
}

interface Props {
  projectId: string;
  apiBase: string;
}

export function CompareViewer({ projectId, apiBase }: Props) {
  const [mode, setMode] = useState<'split' | 'toggle'>('split');
  const [active, setActive] = useState<'mvs' | 'ai'>('mvs');
  const [statsA, setStatsA] = useState<Stats>({ points: 0, loaded: false });
  const [statsB, setStatsB] = useState<Stats>({ points: 0, loaded: false });
  const sharedState = useLinkedControls();

  // Cache-bust: round to the nearest 30s so the URL changes when files are
  // regenerated, forcing the browser to re-fetch instead of serving a stale PLY.
  const cacheBust = Math.floor(Date.now() / 30000);
  const mvsCloud: CloudInfo = {
    label: 'Raw MVS (SfM units)',
    url: `${apiBase}/files/${projectId}/mvs/dense.ply?v=${cacheBust}`,
    color: '#60a5fa',
  };
  const aiCloud: CloudInfo = {
    label: 'Metric-scaled',
    url: `${apiBase}/files/${projectId}/clouds/scaled.ply?v=${cacheBust}`,
    color: '#34d399',
  };

  const added = statsB.points - statsA.points;
  const pct = statsA.points > 0 ? ((added / statsA.points) * 100).toFixed(1) : null;

  return (
    <div className="space-y-3">
      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex rounded-lg overflow-hidden border border-slate-700">
          {(['split', 'toggle'] as const).map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                mode === m ? 'bg-brand-600 text-white' : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}>
              {m === 'split' ? 'Split' : 'Toggle'}
            </button>
          ))}
        </div>

        {/* Stats */}
        <div className="flex gap-3 text-xs">
          <span className="text-blue-400 font-mono">
            MVS: {statsA.loaded ? (statsA.error ? '—' : `${statsA.points.toLocaleString()} pts`) : 'loading…'}
          </span>
          <span className="text-slate-500">→</span>
          <span className="text-violet-400 font-mono">
            AI: {statsB.loaded ? (statsB.error ? '—' : `${statsB.points.toLocaleString()} pts`) : 'loading…'}
          </span>
          {pct && added > 0 && (
            <span className="text-green-400 font-mono">+{added.toLocaleString()} pts (+{pct}%)</span>
          )}
        </div>
      </div>

      {/* Viewer area */}
      <div className="rounded-xl overflow-hidden border border-slate-700 bg-slate-950" style={{ height: '28rem' }}>
        {mode === 'split' ? (
          <div className="flex h-full">
            {/* Left — MVS */}
            <div className="relative flex-1 border-r border-slate-700">
              <CloudCanvas cloud={mvsCloud} sharedState={sharedState} onStats={setStatsA} />
              <div className="absolute top-2 left-2 pointer-events-none">
                <span className="text-xs font-semibold text-blue-300 bg-black/60 px-2 py-1 rounded">
                  MVS only
                </span>
              </div>
            </div>
            {/* Right — AI fused */}
            <div className="relative flex-1">
              <CloudCanvas cloud={aiCloud} sharedState={sharedState} onStats={setStatsB} />
              <div className="absolute top-2 left-2 pointer-events-none">
                <span className="text-xs font-semibold text-violet-300 bg-black/60 px-2 py-1 rounded">
                  AI-fused
                </span>
              </div>
            </div>
          </div>
        ) : (
          // Toggle mode — single canvas with label buttons
          <div className="relative h-full">
            {/* Only render the active one */}
            <div className={`absolute inset-0 ${active === 'mvs' ? '' : 'invisible'}`}>
              <CloudCanvas cloud={mvsCloud} sharedState={sharedState} onStats={setStatsA} />
            </div>
            <div className={`absolute inset-0 ${active === 'ai' ? '' : 'invisible'}`}>
              <CloudCanvas cloud={aiCloud} sharedState={sharedState} onStats={setStatsB} />
            </div>
            {/* Toggle buttons */}
            <div className="absolute bottom-3 left-0 right-0 flex justify-center gap-2 pointer-events-auto">
              <button onClick={() => setActive('mvs')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-full transition-colors ${
                  active === 'mvs' ? 'bg-blue-600 text-white' : 'bg-black/60 text-blue-300 hover:bg-black/80'
                }`}>
                MVS only
              </button>
              <button onClick={() => setActive('ai')}
                className={`px-3 py-1.5 text-xs font-semibold rounded-full transition-colors ${
                  active === 'ai' ? 'bg-violet-600 text-white' : 'bg-black/60 text-violet-300 hover:bg-black/80'
                }`}>
                AI-fused
              </button>
            </div>
          </div>
        )}
      </div>

      <p className="text-xs text-slate-500">
        Drag to rotate · Scroll to zoom. Split mode shows both clouds independently.
        AI-fused adds Depth Anything v2 estimates for featureless surfaces (walls, ceilings, thin structures).
      </p>
    </div>
  );
}
