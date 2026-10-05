'use client';

/**
 * ResultsViewer — tabbed 3D results panel for completed projects.
 *
 * Tabs:
 *   Point Cloud  — coverage heatmap PLY (vertex-coloured), with colour-mode toggle
 *   Mesh         — Poisson-reconstructed OBJ
 *   Camera Path  — sparse cloud + photographer trajectory
 *   Objects      — detected object inventory (when object_labels present)
 *   Segmentation — HVAC detections drawn on real photos (when HVAC ran);
 *                  the placement is also drawn in 3D on Point Cloud / Mesh
 */

import { useState, useEffect, useRef, type ReactNode } from 'react';
import * as THREE from 'three';
import { PLYLoader, OBJLoader } from 'three-stdlib';
import { OrbitControls } from 'three-stdlib';
import { CameraPathViewer } from './CameraPathViewer';
import { CameraWalkthroughViewer } from './CameraWalkthroughViewer';
import { GaussianSplatViewer } from './GaussianSplatViewer';
import { SuggestionsPanel } from './SuggestionsPanel';
import { HvacSegmentationViewer, type HvacSegmentation, type HvacPlacementResult } from './HvacSegmentationViewer';
import { buildHvacOverlay, hasPlacementGeometry } from './lib/hvacOverlay';
import { Download } from 'lucide-react';

type Tab = 'cloud' | 'mesh' | 'path' | 'walkthrough' | 'rgb' | 'suggestions' | 'objects' | 'hvac';
type ColorMode = 'coverage' | 'confidence' | 'semantic';

interface Export {
  label: string;
  key: string;
  mime_type: string;
}

interface ObjectLabel {
  object_name: string;
  detection_level: number;
  center_3d: number[];
  bbox_3d_min: number[];
  bbox_3d_max: number[];
  point_count: number;
  catalog_source: string;
  catalog_confidence: number;
  semantic_confidence: number;
  geometry_correction_mode: number;
  scale_factor: number;
  width_mm: number;
  height_mm: number;
  depth_mm: number;
}

interface Props {
  projectId: string;
  exports?: Export[];
  suggestions?: any[];
  scaleFactor?: number;
  splatKey?: string | null;
  gsMeshKey?: string | null;
  lingbotCloudKey?: string | null;
  lingbotMeshKey?: string | null;
  metricanythingCloudKey?: string | null;
  metricanythingMeshKey?: string | null;
  hvacSegmentation?: HvacSegmentation | null;
  hvacPlacement?: HvacPlacementResult | null;
  objectLabels?: ObjectLabel[];
  apiBase: string;
}

function storageUrl(apiBase: string, key: string) {
  return `${apiBase}/files/${key}`;
}

// ── Generic Three.js canvas viewer ───────────────────────────────────────────

function ThreeCanvas({ loader, overlay }: {
  loader: () => Promise<THREE.Object3D | null>;
  // Extra geometry in the loaded file's own coordinate frame (e.g. the HVAC
  // placement). Added as a child so it shares the re-centring and flip below.
  overlay?: (() => THREE.Object3D | null) | null;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState('Loading…');
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x0f172a);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dl = new THREE.DirectionalLight(0xffffff, 0.8);
    dl.position.set(1, 2, 3);
    scene.add(dl);

    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.001, 100000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    let animId = 0;
    const animate = () => { animId = requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); };
    animate();

    const handleResize = () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    loader().then(obj => {
      if (!obj) { setErr('Could not load file'); return; }
      scene.add(obj);

      const box = new THREE.Box3().setFromObject(obj);
      const centre = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();

      obj.position.sub(centre);
      const extra = overlay?.();
      if (extra) obj.add(extra);

      // COLMAP is Y-down / Z-forward; Three.js is Y-up / Z-toward-viewer.
      // Rotating 180° around X maps COLMAP → Three.js so the scene is right-side up.
      obj.rotation.x = Math.PI;

      // Start camera slightly above and in front for a natural room overview
      camera.position.set(0, size * 0.4, size * 1.4);
      camera.near = size * 0.001;
      camera.far  = size * 10;
      camera.updateProjectionMatrix();
      controls.target.set(0, 0, 0);
      controls.update();

      const pts = obj instanceof THREE.Points ? obj.geometry.attributes.position?.count : null;
      setStatus(pts ? `${pts.toLocaleString()} points` : 'Loaded');
    }).catch(e => setErr(e.message ?? 'Load error'));

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      controls.dispose();
      renderer.dispose();
      try { container.removeChild(renderer.domElement); } catch {}
    };
  }, []);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      <div className="absolute bottom-3 left-3 pointer-events-none">
        {err
          ? <span className="text-xs text-red-400 bg-black/50 px-2 py-1 rounded">{err}</span>
          : <span className="text-xs text-slate-400 bg-black/50 px-2 py-1 rounded">{status}</span>}
      </div>
    </div>
  );
}

function plyLoader(url: string) {
  return () => new Promise<THREE.Object3D | null>((resolve) => {
    new PLYLoader().load(url, (geo) => {
      geo.computeVertexNormals();
      const hasColor = !!geo.attributes.color;
      const mat = new THREE.PointsMaterial({
        size: 0.004,
        vertexColors: hasColor,
        color: hasColor ? undefined : 0x64748b,
        sizeAttenuation: true,
      });
      resolve(new THREE.Points(geo, mat));
    }, undefined, () => resolve(null));
  });
}

function objLoader(url: string, meshSide: THREE.Side) {
  return () => new Promise<THREE.Object3D | null>((resolve) => {
    new OBJLoader().load(url, (obj) => {
      obj.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.material = new THREE.MeshStandardMaterial({
            color: 0x94a3b8,
            roughness: 0.7,
            metalness: 0.1,
            side: meshSide,
          });
        }
      });
      resolve(obj);
    }, undefined, () => resolve(null));
  });
}

function PlacementToggle({ shown, onToggle }: { shown: boolean; onToggle: () => void }) {
  return (
    <div className="absolute top-2 right-2 z-10 flex flex-col items-end gap-1">
      <button
        onClick={onToggle}
        className={`px-3 py-1.5 rounded border border-slate-600 text-xs shadow-lg transition-colors ${shown ? 'bg-brand-600 text-white' : 'bg-slate-900/90 text-slate-300 hover:text-white hover:bg-slate-700'}`}
      >
        {shown ? 'Hide placement' : 'Show placement'}
      </button>
      {shown && <PlacementLegend />}
    </div>
  );
}

function PlacementLegend() {
  const item = (swatch: ReactNode, label: string) => (
    <span className="flex items-center gap-1.5">{swatch}{label}</span>
  );
  const dot = (color: string) => <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: color }} />;
  return (
    <div className="rounded border border-slate-700 bg-slate-900/90 px-2 py-1.5 text-[11px] text-slate-300 space-y-0.5 shadow-lg">
      {item(<span className="inline-block w-3 h-2 bg-green-500/70 border border-green-500" />, 'Best placement')}
      {item(<span className="inline-block w-3 h-2 border border-amber-500" />, 'Alternatives')}
      {item(dot('#2563eb'), 'Rücklauf')}
      {item(dot('#dc2626'), 'Vorlauf')}
      {item(dot('#f59e0b'), 'Fixtures')}
    </div>
  );
}

// ── Object inventory helpers ──────────────────────────────────────────────────

// ── Main exported component ───────────────────────────────────────────────────

export function ResultsViewer({ projectId, exports = [], suggestions = [], scaleFactor = 1, splatKey, gsMeshKey, lingbotCloudKey, lingbotMeshKey, metricanythingCloudKey, metricanythingMeshKey, hvacSegmentation, hvacPlacement, apiBase }: Props) {
  const [tab, setTab] = useState<Tab>('cloud');
  const [colorMode, setColorMode] = useState<ColorMode>('coverage');
  const [meshSide, setMeshSide] = useState<THREE.Side>(THREE.BackSide);  // default inside for rooms
  const [rgbSource, setRgbSource] = useState<'dense' | 'export'>('export');
  const [meshSource, setMeshSource] = useState<'poisson' | 'gs' | 'lingbot' | 'metricanything'>('poisson');
  const [cloudSource, setCloudSource] = useState<'colmap' | 'lingbot' | 'metricanything'>('colmap');
  const placementAvailable = hasPlacementGeometry(hvacPlacement);
  const [showPlacement, setShowPlacement] = useState(true);
  const placementOverlay = placementAvailable && showPlacement
    ? () => buildHvacOverlay(hvacPlacement, hvacSegmentation)
    : null;

  const plyExport        = exports.find(e => e.key === `${projectId}/exports/output.ply`);
  const objExport        = exports.find(e => e.key.endsWith('.obj'));
  const lasExport        = exports.find(e => e.key.endsWith('.las'));
  const confidenceExport = exports.find(e => e.key.endsWith('confidence_cloud.ply'));
  const semanticExport   = exports.find(e => e.key.endsWith('semantic_cloud.ply'));

  const cloudUrlByMode: Record<ColorMode, string | null> = {
    coverage:   plyExport        ? storageUrl(apiBase, plyExport.key)        : null,
    confidence: confidenceExport ? storageUrl(apiBase, confidenceExport.key) : null,
    semantic:   semanticExport   ? storageUrl(apiBase, semanticExport.key)   : null,
  };
  const lingbotCloudUrl = lingbotCloudKey ? storageUrl(apiBase, lingbotCloudKey) : null;
  const lingbotMeshUrl  = lingbotMeshKey  ? storageUrl(apiBase, lingbotMeshKey)  : null;
  const maCloudUrl = metricanythingCloudKey ? storageUrl(apiBase, metricanythingCloudKey) : null;
  const maMeshUrl  = metricanythingMeshKey  ? storageUrl(apiBase, metricanythingMeshKey)  : null;
  const activeCloudUrl = (cloudSource === 'lingbot' && lingbotCloudUrl) ? lingbotCloudUrl
    : (cloudSource === 'metricanything' && maCloudUrl) ? maCloudUrl
    : (cloudUrlByMode[colorMode] ?? cloudUrlByMode['coverage']);
  const isFusedMesh = meshSource === 'lingbot' || meshSource === 'metricanything';

  const poissonUrl   = objExport  ? storageUrl(apiBase, objExport.key)  : null;
  const gsMeshUrl    = gsMeshKey  ? storageUrl(apiBase, gsMeshKey)       : null;
  const hasBothMeshes = !!(poissonUrl && gsMeshUrl);
  const objUrl       = meshSource === 'lingbot' && lingbotMeshUrl ? lingbotMeshUrl
                     : meshSource === 'metricanything' && maMeshUrl ? maMeshUrl
                     : meshSource === 'gs' && gsMeshUrl ? gsMeshUrl
                     : (poissonUrl ?? gsMeshUrl ?? lingbotMeshUrl ?? maMeshUrl);
  const sparseUrl    = storageUrl(apiBase, `${projectId}/sfm/sparse.ply`);
  const coverageUrl  = storageUrl(apiBase, `${projectId}/coverage/cloud_colored.ply`);
  const camerasUrl   = storageUrl(apiBase, `${projectId}/sfm/cameras.json`);

  // RGB Cloud: export PLY (processed/filled) by default; raw dense MVS as alternative
  const exportUrl = plyExport ? storageUrl(apiBase, plyExport.key) : storageUrl(apiBase, `${projectId}/mvs/dense.ply`);
  const rawUrl    = storageUrl(apiBase, `${projectId}/mvs/dense.ply`);
  const rgbUrl    = rgbSource === 'export' ? exportUrl : rawUrl;

  const tabs: { id: Tab; label: string; available: boolean }[] = [
    { id: 'cloud',       label: 'Point Cloud',    available: !!activeCloudUrl },
    { id: 'mesh',        label: 'Mesh',           available: !!objUrl },
    { id: 'path',        label: splatKey ? '✨ 3DGS Scene' : 'Scene Overview', available: true },
    { id: 'walkthrough', label: 'Walkthrough',    available: !!activeCloudUrl },
    { id: 'rgb',         label: 'RGB Cloud',      available: true },
    { id: 'suggestions', label: suggestions.length > 0 ? `Re-shoot (${suggestions.length})` : 'Re-shoot', available: true },
    ...(hvacSegmentation || hvacPlacement
      ? [{ id: 'hvac' as Tab, label: 'Segmentation', available: true }]
      : []),
  ];

  const hasAlternateColors = !!(confidenceExport || semanticExport);

  return (
    <div className="rounded-xl border border-slate-200 overflow-hidden bg-slate-950">
      {/* Tab bar */}
      <div className="flex items-center gap-0 border-b border-slate-800 bg-slate-900 px-1">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => t.available && setTab(t.id)}
            disabled={!t.available}
            className={`px-4 py-2.5 text-sm font-medium transition-colors ${
              tab === t.id
                ? 'text-white border-b-2 border-brand-400 -mb-px'
                : t.available
                  ? 'text-slate-400 hover:text-slate-200'
                  : 'text-slate-600 cursor-not-allowed'
            }`}
          >
            {t.label}
          </button>
        ))}

        {/* Mesh controls */}
        {tab === 'mesh' && (
          <div className="ml-2 flex items-center gap-1.5">
            {(hasBothMeshes || lingbotMeshUrl || maMeshUrl) && (
              <div className="flex rounded border border-slate-600 overflow-hidden text-xs">
                {poissonUrl && (
                  <button
                    onClick={() => setMeshSource('poisson')}
                    className={`px-2 py-1 transition-colors ${meshSource === 'poisson' ? 'bg-brand-600 text-white' : 'text-slate-400 hover:text-white hover:bg-slate-700'}`}
                  >
                    Dense Cloud
                  </button>
                )}
                {gsMeshUrl && (
                  <button
                    onClick={() => setMeshSource('gs')}
                    className={`px-2 py-1 transition-colors ${meshSource === 'gs' ? 'bg-brand-600 text-white' : 'text-slate-400 hover:text-white hover:bg-slate-700'}`}
                  >
                    Gaussian Splat
                  </button>
                )}
                {lingbotMeshUrl && (
                  <button
                    onClick={() => setMeshSource('lingbot')}
                    className={`px-2 py-1 transition-colors ${meshSource === 'lingbot' ? 'bg-brand-600 text-white' : 'text-slate-400 hover:text-white hover:bg-slate-700'}`}
                  >
                    Densified (LingBot)
                  </button>
                )}
                {maMeshUrl && (
                  <button
                    onClick={() => setMeshSource('metricanything')}
                    className={`px-2 py-1 transition-colors ${meshSource === 'metricanything' ? 'bg-brand-600 text-white' : 'text-slate-400 hover:text-white hover:bg-slate-700'}`}
                  >
                    Densified (MetricAnything)
                  </button>
                )}
              </div>
            )}
            <button
              onClick={() => setMeshSide(s => s === THREE.BackSide ? THREE.FrontSide : THREE.BackSide)}
              className="px-2 py-1 rounded text-xs text-slate-400 hover:text-white hover:bg-slate-700 transition-colors border border-slate-600"
            >
              View: {meshSide === THREE.BackSide ? 'Inside' : 'Outside'}
            </button>
          </div>
        )}

        {/* RGB source toggle */}
        {tab === 'rgb' && (
          <div className="ml-2 flex items-center gap-1">
            <button
              onClick={() => setRgbSource(s => s === 'dense' ? 'export' : 'dense')}
              className="px-2 py-1 rounded text-xs text-slate-400 hover:text-white hover:bg-slate-700 transition-colors border border-slate-600"
            >
              {rgbSource === 'export' ? 'Post-processed' : 'Raw MVS'}
            </button>
          </div>
        )}

        {/* Cloud source toggle is rendered as an overlay on the canvas (below),
            not here — the tab bar is too crowded and clipped it. */}

        {/* Colour-mode dropdown (Point Cloud tab only, when alternates exist) */}
        {tab === 'cloud' && cloudSource === 'colmap' && hasAlternateColors && (
          <div className="ml-2 flex items-center gap-1">
            <span className="text-xs text-slate-500">Color:</span>
            <select
              value={colorMode}
              onChange={e => setColorMode(e.target.value as ColorMode)}
              className="text-xs bg-slate-800 border border-slate-600 text-slate-200 rounded px-2 py-1 focus:outline-none"
            >
              <option value="coverage">Coverage</option>
              {confidenceExport && <option value="confidence">Confidence</option>}
              {semanticExport   && <option value="semantic">Semantic</option>}
            </select>
          </div>
        )}

        {/* Download buttons pushed to the right */}
        <div className="ml-auto flex items-center gap-1 pr-2">
          {[plyExport, objExport, lasExport,
            gsMeshKey ? { key: gsMeshKey, label: 'GS Mesh (.obj)', mime_type: 'model/obj' } : null,
            lingbotCloudKey ? { key: lingbotCloudKey, label: 'Densified Cloud (.ply)', mime_type: 'model/ply' } : null,
            lingbotMeshKey ? { key: lingbotMeshKey, label: 'Densified Mesh (.obj)', mime_type: 'model/obj' } : null,
            metricanythingCloudKey ? { key: metricanythingCloudKey, label: 'MetricAnything Cloud (.ply)', mime_type: 'model/ply' } : null,
            metricanythingMeshKey ? { key: metricanythingMeshKey, label: 'MetricAnything Mesh (.obj)', mime_type: 'model/obj' } : null,
          ].filter(Boolean).map(ex => ex && (
            <a
              key={ex.key}
              href={storageUrl(apiBase, ex.key)}
              download
              className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
              title={`Download ${ex.label}`}
            >
              <Download className="h-3 w-3" />
              {ex.key.split('.').pop()?.toUpperCase()}
            </a>
          ))}
        </div>
      </div>

      {/* Viewer area */}
      <div style={{ height: tab === 'objects' ? 'auto' : '28rem' }}>
        {tab === 'cloud' && activeCloudUrl && (
          <div className="relative h-full">
            {(lingbotCloudUrl || maCloudUrl) && (
              <div className="absolute top-2 left-2 z-10 flex rounded border border-slate-600 overflow-hidden text-xs shadow-lg">
                <button
                  onClick={() => setCloudSource('colmap')}
                  className={`px-3 py-1.5 transition-colors ${cloudSource === 'colmap' ? 'bg-brand-600 text-white' : 'bg-slate-900/90 text-slate-300 hover:text-white hover:bg-slate-700'}`}
                >
                  COLMAP
                </button>
                {lingbotCloudUrl && (
                  <button
                    onClick={() => setCloudSource('lingbot')}
                    className={`px-3 py-1.5 transition-colors ${cloudSource === 'lingbot' ? 'bg-brand-600 text-white' : 'bg-slate-900/90 text-slate-300 hover:text-white hover:bg-slate-700'}`}
                  >
                    Densified (LingBot)
                  </button>
                )}
                {maCloudUrl && (
                  <button
                    onClick={() => setCloudSource('metricanything')}
                    className={`px-3 py-1.5 transition-colors ${cloudSource === 'metricanything' ? 'bg-brand-600 text-white' : 'bg-slate-900/90 text-slate-300 hover:text-white hover:bg-slate-700'}`}
                  >
                    Densified (MetricAnything)
                  </button>
                )}
              </div>
            )}
            {placementAvailable && (
              <PlacementToggle shown={showPlacement} onToggle={() => setShowPlacement(v => !v)} />
            )}
            <ThreeCanvas key={`cloud-${colorMode}-${cloudSource}-${showPlacement}`}
              loader={plyLoader(activeCloudUrl)} overlay={placementOverlay} />
          </div>
        )}
        {tab === 'mesh' && objUrl && (
          <div className="relative h-full">
            {placementAvailable && (
              <PlacementToggle shown={showPlacement} onToggle={() => setShowPlacement(v => !v)} />
            )}
            <ThreeCanvas
              key={`mesh-${meshSide}-${meshSource}-${showPlacement}`}
              loader={objLoader(objUrl, isFusedMesh ? THREE.DoubleSide : meshSide)}
              overlay={placementOverlay}
            />
          </div>
        )}
        {tab === 'hvac' && (
          <HvacSegmentationViewer
            projectId={projectId}
            apiBase={apiBase}
            hvacSegmentation={hvacSegmentation}
            hvacPlacement={hvacPlacement}
          />
        )}
        {tab === 'rgb' && (
          <ThreeCanvas key={`rgb-${rgbSource}`} loader={plyLoader(rgbUrl)} />
        )}
        {tab === 'path' && (
          splatKey ? (
            <GaussianSplatViewer
              splatUrl={storageUrl(apiBase, splatKey)}
              camerasJsonUrl={camerasUrl}
              suggestions={suggestions}
              scaleFactor={scaleFactor}
            />
          ) : (
            <CameraPathViewer
              sparseCloudUrl={coverageUrl}
              fallbackSparseUrl={sparseUrl}
              camerasJsonUrl={camerasUrl}
              suggestions={suggestions}
              scaleFactor={scaleFactor}
            />
          )
        )}
        {tab === 'walkthrough' && (
          <CameraWalkthroughViewer
            cloudUrl={activeCloudUrl}
            meshUrl={objUrl}
            framesBaseUrl={`${apiBase}/files/${projectId}/frames`}
            camerasJsonUrl={camerasUrl}
            scaleFactor={scaleFactor}
          />
        )}
        {tab === 'suggestions' && (
          <div className="p-4 h-full overflow-auto">
            <SuggestionsPanel projectId={projectId} apiBase={apiBase} suggestions={suggestions} />
          </div>
        )}
      </div>

      {/* Help text */}
      <div className="px-4 py-2 bg-slate-900 border-t border-slate-800 text-xs text-slate-500">
        {tab === 'cloud'
          ? 'Coverage heatmap · Drag to rotate · Scroll to zoom · Right-drag to pan'
          : tab === 'path'
          ? 'Drag to rotate · Scroll to zoom · Right-drag to pan · Green→Red = camera path · Orange = re-shoot areas · Red = coverage gap'
          : tab === 'walkthrough'
          ? 'Drag to look around (rotates in place) · Prev/Next to step camera positions · Toggle Point Cloud/Mesh · Compare to photo'
          : tab === 'rgb'
          ? 'Natural RGB colors from video frames · Drag to rotate · Scroll to zoom'
          : tab === 'hvac'
          ? 'Real photos with the HVAC detections drawn on them — pick a category above'
          : tab === 'mesh'
          ? `${hasBothMeshes ? (meshSource === 'poisson' ? 'Poisson reconstruction from dense cloud' : 'Mesh extracted from Gaussian Splat') + ' · ' : ''}${meshSide === THREE.BackSide ? 'Inside view' : 'Outside view'} · Drag to rotate`
          : 'Drag to rotate · Scroll to zoom · Right-drag to pan'}
      </div>
    </div>
  );
}
