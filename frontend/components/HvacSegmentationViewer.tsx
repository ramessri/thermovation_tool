'use client';

/**
 * HvacSegmentationViewer — "Segmentation" tab: real photos with the HVAC
 * detections drawn on them (ADE20K wall highlight, GDINO+SAM2 fixture boxes,
 * Rücklauf/Vorlauf, and the final placement overlay). The same results are
 * drawn in 3D on the Point Cloud / Mesh tabs via lib/hvacOverlay.ts.
 */

import { useState } from 'react';

export interface WallCandidate {
  inliers: number;
  meets_min_inliers: boolean;
  ade20k_wall_confidence: number | null;
  ade20k_frame?: string;
  ade20k_overlay_key?: string;
}

export interface HvacFixtureInstance {
  position_m: number[];
  n_observations: number;
  max_score: number;
  best_frame?: string;
  best_box_frac?: [number, number, number, number];
}

export interface RucklaufPosition {
  position_m: number[];
  method: string;
  best_frame?: string;
  best_box_frac?: [number, number, number, number];
}

export interface HvacSegmentation {
  wall_candidates: WallCandidate[];
  hvac_fixtures: Record<string, HvacFixtureInstance[]>;
  rucklauf_position: RucklaufPosition | null;
  vorlauf_position: RucklaufPosition | null;
}

export interface HvacPlacementResult {
  status: string;
  candidates: Array<{
    rank: number; wall_inliers: number; clearance_cm: number;
    mount_height_cm: number | null; distance_to_rucklauf_cm: number | null;
    corners_world_m?: number[][];   // metric world frame — drawn by lib/hvacOverlay.ts
  }>;
  overlay_image_key: string | null;
}

interface Props {
  projectId: string;
  apiBase: string;
  hvacSegmentation?: HvacSegmentation | null;
  hvacPlacement?: HvacPlacementResult | null;
}

type Category = 'walls' | 'fixtures' | 'rucklauf' | 'placement';

export const FIXTURE_COLORS: Record<string, string> = {
  pipes: '#f59e0b', valves: '#ea580c', radiators: '#3b82f6',
  electrical: '#eab308', windows: '#22d3ee', other: '#94a3b8',
};

function frameUrl(apiBase: string, projectId: string, frame: string) {
  return `${apiBase}/files/${projectId}/frames/${frame}`;
}
function fileUrl(apiBase: string, key: string) {
  return `${apiBase}/files/${key}`;
}

function BoxedPhoto({
  src, box, color, label,
}: { src: string; box?: [number, number, number, number]; color?: string; label?: string }) {
  return (
    <div className="relative inline-block max-w-full">
      <img src={src} className="max-w-full max-h-[22rem] rounded border border-slate-700" />
      {box && (
        <div
          className="absolute border-2 rounded-sm"
          style={{
            left: `${box[0] * 100}%`, top: `${box[1] * 100}%`,
            width: `${(box[2] - box[0]) * 100}%`, height: `${(box[3] - box[1]) * 100}%`,
            borderColor: color ?? '#22c55e',
            boxShadow: `0 0 0 1px rgba(0,0,0,0.5)`,
          }}
        >
          {label && (
            <span
              className="absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] font-medium rounded-sm text-slate-900 whitespace-nowrap"
              style={{ background: color ?? '#22c55e' }}
            >
              {label}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export function HvacSegmentationViewer({ projectId, apiBase, hvacSegmentation, hvacPlacement }: Props) {
  const [category, setCategory] = useState<Category>('walls');
  const [fixtureLabel, setFixtureLabel] = useState<string | null>(null);
  const [fixtureIdx, setFixtureIdx] = useState(0);

  const wallCandidates = hvacSegmentation?.wall_candidates ?? [];
  const fixtures = hvacSegmentation?.hvac_fixtures ?? {};
  const fixtureClasses = Object.entries(fixtures)
    .filter(([k, v]) => k !== 'rucklauf_candidates' && k !== 'vorlauf_candidates' && v.length > 0);
  const activeFixtureLabel = fixtureLabel ?? fixtureClasses[0]?.[0] ?? null;
  const activeFixtureList = activeFixtureLabel ? fixtures[activeFixtureLabel] ?? [] : [];
  const activeFixture = activeFixtureList[fixtureIdx];

  const categories: { id: Category; label: string; available: boolean }[] = [
    { id: 'walls', label: 'Walls (ADE20K)', available: wallCandidates.some(c => c.ade20k_overlay_key) },
    { id: 'fixtures', label: 'Fixtures (GDINO+SAM2)', available: fixtureClasses.length > 0 },
    { id: 'rucklauf', label: 'Rücklauf / Vorlauf', available: !!(hvacSegmentation?.rucklauf_position || hvacSegmentation?.vorlauf_position) },
    { id: 'placement', label: 'Placement', available: !!hvacPlacement?.overlay_image_key },
  ];

  return (
    <div className="h-full flex flex-col bg-slate-950 text-slate-200">
      <div className="flex items-center gap-1 px-3 py-2 border-b border-slate-800 flex-wrap">
        {categories.map(c => (
          <button
            key={c.id}
            onClick={() => c.available && setCategory(c.id)}
            disabled={!c.available}
            className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
              category === c.id
                ? 'bg-brand-600 text-white'
                : c.available ? 'text-slate-400 hover:text-white hover:bg-slate-800' : 'text-slate-700 cursor-not-allowed'
            }`}
          >
            {c.label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto p-4">
        {category === 'walls' && (
          <div className="flex flex-wrap gap-4">
            {wallCandidates.filter(c => c.ade20k_overlay_key).length === 0 && (
              <p className="text-sm text-slate-500">No ADE20K overlay available for any wall candidate.</p>
            )}
            {wallCandidates.filter(c => c.ade20k_overlay_key).map((c, i) => (
              <div key={i} className="space-y-1.5">
                <img
                  src={fileUrl(apiBase, c.ade20k_overlay_key!)}
                  className="max-w-full max-h-[22rem] rounded border border-slate-700"
                />
                <div className="text-xs text-slate-400">
                  {c.inliers.toLocaleString()} inliers · {c.meets_min_inliers ? 'above evidence floor' : 'below evidence floor'}
                  {c.ade20k_wall_confidence != null && ` · ${(c.ade20k_wall_confidence * 100).toFixed(0)}% wall pixels`}
                </div>
              </div>
            ))}
          </div>
        )}

        {category === 'fixtures' && (
          <div className="flex gap-4">
            <div className="w-48 shrink-0 space-y-1">
              {fixtureClasses.map(([label, list]) => (
                <button
                  key={label}
                  onClick={() => { setFixtureLabel(label); setFixtureIdx(0); }}
                  className={`w-full text-left px-2 py-1.5 rounded text-xs flex items-center justify-between ${
                    activeFixtureLabel === label ? 'bg-slate-800 text-white' : 'text-slate-400 hover:bg-slate-900'
                  }`}
                >
                  <span className="flex items-center gap-1.5">
                    <span className="inline-block w-2 h-2 rounded-full" style={{ background: FIXTURE_COLORS[label] ?? FIXTURE_COLORS.other }} />
                    {label}
                  </span>
                  <span className="text-slate-500">{list.length}</span>
                </button>
              ))}
            </div>
            <div className="flex-1">
              {activeFixture?.best_frame ? (
                <>
                  <BoxedPhoto
                    src={frameUrl(apiBase, projectId, activeFixture.best_frame)}
                    box={activeFixture.best_box_frac}
                    color={FIXTURE_COLORS[activeFixtureLabel ?? ''] ?? FIXTURE_COLORS.other}
                    label={activeFixtureLabel ?? undefined}
                  />
                  <div className="mt-2 flex items-center gap-2 text-xs text-slate-400">
                    {activeFixtureList.length > 1 && (
                      <div className="flex gap-1">
                        {activeFixtureList.map((_, i) => (
                          <button
                            key={i}
                            onClick={() => setFixtureIdx(i)}
                            className={`w-6 h-6 rounded text-[10px] ${i === fixtureIdx ? 'bg-brand-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
                          >
                            {i + 1}
                          </button>
                        ))}
                      </div>
                    )}
                    <span>{activeFixture.n_observations} observation(s) · score {activeFixture.max_score.toFixed(2)}</span>
                  </div>
                </>
              ) : (
                <p className="text-sm text-slate-500">No frame available for this fixture.</p>
              )}
            </div>
          </div>
        )}

        {category === 'rucklauf' && (
          <div className="flex flex-wrap gap-6">
            {hvacSegmentation?.rucklauf_position?.best_frame && (
              <div className="space-y-1.5">
                <BoxedPhoto
                  src={frameUrl(apiBase, projectId, hvacSegmentation.rucklauf_position.best_frame)}
                  box={hvacSegmentation.rucklauf_position.best_box_frac}
                  color="#2563eb"
                  label="Rücklauf"
                />
                <div className="text-xs text-slate-400">Found via {hvacSegmentation.rucklauf_position.method}</div>
              </div>
            )}
            {hvacSegmentation?.vorlauf_position?.best_frame && (
              <div className="space-y-1.5">
                <BoxedPhoto
                  src={frameUrl(apiBase, projectId, hvacSegmentation.vorlauf_position.best_frame)}
                  box={hvacSegmentation.vorlauf_position.best_box_frac}
                  color="#dc2626"
                  label="Vorlauf"
                />
                <div className="text-xs text-slate-400">Found via {hvacSegmentation.vorlauf_position.method}</div>
              </div>
            )}
            {!hvacSegmentation?.rucklauf_position?.best_frame && !hvacSegmentation?.vorlauf_position?.best_frame && (
              <p className="text-sm text-slate-500">No Rücklauf/Vorlauf photo available.</p>
            )}
          </div>
        )}

        {category === 'placement' && hvacPlacement && (
          <div className="space-y-3">
            {hvacPlacement.overlay_image_key ? (
              <img
                src={fileUrl(apiBase, hvacPlacement.overlay_image_key)}
                className="max-w-full max-h-[22rem] rounded border border-slate-700"
              />
            ) : (
              <p className="text-sm text-slate-500">No frame rendered a clean placement overlay.</p>
            )}
            <div className="space-y-1">
              {hvacPlacement.candidates.map(c => (
                <div key={c.rank} className="flex items-center gap-4 text-xs text-slate-400">
                  <span className="text-slate-200 font-medium">#{c.rank}</span>
                  <span>{c.distance_to_rucklauf_cm != null ? `d(Rücklauf) ${c.distance_to_rucklauf_cm}cm` : '—'}</span>
                  <span>clearance {c.clearance_cm}cm</span>
                  <span>{c.mount_height_cm != null ? `height ${c.mount_height_cm}cm` : ''}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
