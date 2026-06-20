'use client';

/**
 * PipelineLog — expandable per-stage progress panel.
 *
 * Each stage shows:
 *  • Status icon (running spinner, ✓ done, – pending)
 *  • Stage label + last progress message
 *  • Expandable detail card drawn from stageDetails (rich output extracted
 *    from WebSocket messages + final job result)
 */

import { useState } from 'react';

export interface StageDetail {
  id: string;
  label: string;
  phase: number;
  status: 'pending' | 'running' | 'done' | 'skipped';
  lastMessage?: string;
  progress?: number;
  detail?: Record<string, string | number | string[] | undefined>;
  elapsedSeconds?: number;
  startedAt?: number;   // unix epoch seconds
  finishedAt?: number;  // unix epoch seconds
  errorMsg?: string;
}

interface Props {
  stages: StageDetail[];
  currentStageId?: string;
}

function formatElapsed(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4 text-blue-500" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  );
}

function StatusIcon({ status }: { status: StageDetail['status'] }) {
  if (status === 'running') return <Spinner />;
  if (status === 'done')    return <span className="text-green-500 font-bold text-sm">✓</span>;
  if (status === 'skipped') return <span className="text-slate-300 text-sm">–</span>;
  return <span className="h-4 w-4 rounded-full border-2 border-slate-200 inline-block" />;
}

function DetailRow({ label, value }: { label: string; value: string | number | string[] | undefined }) {
  if (value === undefined || value === null || value === '') return null;
  const display = Array.isArray(value)
    ? value.length === 0 ? '—' : value.join(', ')
    : String(value);
  const isWarning = label === 'Warning';
  if (isWarning) {
    return (
      <div className="mt-1 rounded bg-amber-50 border border-amber-200 px-2 py-1.5 text-xs text-amber-800">
        {display}
      </div>
    );
  }
  return (
    <div className="flex gap-2 text-xs">
      <span className="text-slate-500 whitespace-nowrap min-w-[120px]">{label}</span>
      <span className="text-slate-800 font-medium break-words">{display}</span>
    </div>
  );
}

export function PipelineLog({ stages }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  function toggle(id: string) {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  let lastPhase = 0;

  return (
    <div className="space-y-1">
      {stages.map((stage) => {
        const showPhaseDivider = stage.phase !== lastPhase;
        lastPhase = stage.phase;
        const isExpanded = expanded.has(stage.id);
        const hasDetail = stage.detail && Object.keys(stage.detail).length > 0;

        return (
          <div key={stage.id}>
            {showPhaseDivider && stage.phase === 2 && (
              <div className="flex items-center gap-2 my-3">
                <div className="flex-1 h-px bg-slate-200" />
                <span className="text-xs text-slate-400 font-medium px-1">Scale &amp; Export</span>
                <div className="flex-1 h-px bg-slate-200" />
              </div>
            )}

            <div className={`rounded-lg transition-colors ${
              stage.status === 'running' ? 'bg-blue-50 border border-blue-100' :
              stage.status === 'done'    ? 'bg-white border border-slate-100' :
              stage.status === 'skipped' ? 'bg-slate-50 border border-slate-100 opacity-60' :
              'bg-white border border-slate-100 opacity-40'
            }`}>
              {/* Header row */}
              <div
                className={`flex items-center gap-3 px-3 py-2.5 ${hasDetail ? 'cursor-pointer select-none' : ''}`}
                onClick={() => hasDetail && toggle(stage.id)}
              >
                <StatusIcon status={stage.status} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-medium ${stage.status === 'pending' ? 'text-slate-400' : 'text-slate-800'}`}>
                      {stage.label}
                    </span>
                    {stage.status === 'running' && stage.progress !== undefined && (
                      <span className="text-xs text-blue-600">{Math.round(stage.progress * 100)}%</span>
                    )}
                    {stage.status === 'done' && stage.elapsedSeconds != null && (
                      <span className="text-xs text-slate-400 shrink-0">{formatElapsed(stage.elapsedSeconds)}</span>
                    )}
                    {stage.status === 'done' && stage.startedAt != null && stage.finishedAt != null && (
                      <span className="text-xs text-slate-300 shrink-0">
                        {new Date(stage.startedAt * 1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}
                        {' → '}
                        {new Date(stage.finishedAt * 1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}
                      </span>
                    )}
                  </div>
                  {stage.status === 'running' && stage.lastMessage && (
                    <div className="text-xs text-slate-500 truncate mt-0.5">{stage.lastMessage}</div>
                  )}
                  {stage.status === 'done' && stage.detail && (() => {
                    const summary = buildStageSummary(stage.id, stage.detail!);
                    return summary ? (
                      <div className="text-xs text-slate-400 truncate mt-0.5">{summary}</div>
                    ) : null;
                  })()}
                  {stage.status === 'done' && !stage.detail && stage.lastMessage && (
                    <div className="text-xs text-slate-400 truncate mt-0.5">{stage.lastMessage}</div>
                  )}
                  {stage.status === 'pending' && stage.lastMessage && (
                    <div className="text-xs text-slate-400 truncate mt-0.5">{stage.lastMessage}</div>
                  )}
                </div>
                {hasDetail && (
                  <span className="text-slate-400 text-xs">{isExpanded ? '▲' : '▼'}</span>
                )}
              </div>

              {/* Expanded detail */}
              {isExpanded && hasDetail && (
                <div className="border-t border-slate-100 px-4 py-3 space-y-1.5 bg-slate-50 rounded-b-lg">
                  {Object.entries(stage.detail!).map(([k, v]) => (
                    <DetailRow key={k} label={k} value={v} />
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Helper: build detail from compact pipeline_results meta (page-reload path) ─

export function buildStageDetailsFromMeta(
  stageId: string,
  meta: Record<string, any>,
): Record<string, string | number | string[] | undefined> | undefined {
  if (!meta) return undefined;
  const v = (k: string) => meta[k];
  const num = (k: string) => (v(k) != null ? Number(v(k)) : undefined);

  switch (stageId) {
    case 'extract_metadata': return {
      'Focal length': v('focal_length_px') ? `${Math.round(v('focal_length_px'))}px` : undefined,
      'Resolution': v('width') && v('height') ? `${v('width')}×${v('height')}` : undefined,
      'Rotation': v('rotation_deg') ? `${v('rotation_deg')}°` : undefined,
      'Device': [v('make'), v('model')].filter(Boolean).join(' ') || undefined,
    };
    case 'extract_frames': return {
      'Frames selected': num('frame_count'),
      'Dropped (poor quality)': v('blurry_stills_skipped') > 0 ? String(v('blurry_stills_skipped')) : undefined,
      'Rotation applied': v('rotation_deg') ? `${v('rotation_deg')}°` : undefined,
      'Sources': v('n_sources') > 1 ? String(v('n_sources')) : undefined,
    };
    case 'detect_aruco': return {
      'Markers found': num('n_markers'),
      'Marker IDs': Array.isArray(v('marker_ids')) ? v('marker_ids').map(String) : undefined,
      'Baselines': v('n_baselines') > 0 ? String(v('n_baselines')) : undefined,
      'Warning': v('_warnings')?.[0],
    };
    case 'feature_matching': return v('pair_count') != null ? {
      'Pairs matched': num('pair_count'),
      'Total matches': num('total_matches'),
    } : undefined;
    case 'sfm': return v('registered_images') != null ? {
      'Cameras registered': v('total_images')
        ? `${v('registered_images')} / ${v('total_images')} (${Math.round((v('total_images') - v('registered_images')) / v('total_images') * 100)}% dropped)`
        : String(v('registered_images')),
      'Sparse points': v('num_points3D') != null ? Number(v('num_points3D')).toLocaleString() : undefined,
      'Reprojection error': v('mean_reprojection_error') != null ? `${Number(v('mean_reprojection_error')).toFixed(2)}px` : undefined,
      'Warning': v('_warnings')?.[0],
    } : undefined;
    case 'mvs': return v('dense_point_count') != null ? {
      'Dense points': Number(v('dense_point_count')).toLocaleString(),
      'Runtime': v('runtime_s') != null ? `${v('runtime_s')}s` : undefined,
    } : undefined;
    case 'scale_from_aruco': return {
      'Scale factor': v('scale_factor') != null ? `${Number(v('scale_factor')).toFixed(5)} m/unit` : undefined,
      'Estimates used': num('n_estimates') || undefined,
      'Triangulated markers': num('n_triangulated') || undefined,
      'Warning': v('_warnings')?.[0],
    };
    case 'coverage': return v('coverage_score') != null ? {
      'Coverage score': `${(v('coverage_score') * 100).toFixed(1)}%`,
      'Re-shoot areas': num('n_suggestions'),
    } : undefined;
    case 'refine_cloud': return {
      'Noise removed': v('sor_removed') != null ? Number(v('sor_removed')).toLocaleString() : undefined,
      'Final points': v('n_after') != null ? Number(v('n_after')).toLocaleString() : undefined,
    };
    default:
      return undefined;
  }
}

// ── Helper: build detail record from job result fields ────────────────────────

export function buildStageDetails(
  stageId: string,
  jobResult: Record<string, any>,
): Record<string, string | number | string[] | undefined> | undefined {
  switch (stageId) {
    case 'extract_metadata': {
      const vm = jobResult.video_metadata ?? {};
      const fl = vm.focal_length_px ? `${Math.round(vm.focal_length_px)}px` : 'unknown';
      const flSrc: Record<string, string> = {
        calibration_photo: 'calibration photo ✓',
        video_exif: 'video EXIF',
        heuristic: 'heuristic (estimated)',
      };
      const wh = vm.width && vm.height ? `${vm.width}×${vm.height}` : undefined;
      const ts = (vm.stable_frame_timestamps ?? []).length;
      return wh ? {
        'Resolution': wh,
        'Focal length': fl,
        'FL source': flSrc[vm.focal_length_source] ?? vm.focal_length_source,
        'Rotation': vm.rotation_deg ? `${vm.rotation_deg}°` : undefined,
        'Stable frames': ts > 0 ? String(ts) : undefined,
        'Device': [vm.make, vm.model].filter(Boolean).join(' ') || undefined,
      } : undefined;
    }
    case 'extract_frames': {
      const n = (jobResult.frame_keys ?? jobResult.image_keys ?? []).length;
      const dropped = jobResult.blurry_dropped;
      const oversample = jobResult.oversample_factor;
      return n > 0 ? {
        'Frames selected': n,
        'Quality filter': oversample ? `best-of-${oversample} per second (sharpness + motion)` : undefined,
        'Dropped (poor quality)': dropped != null && dropped > 0 ? String(dropped) : undefined,
        'Rotation applied': jobResult.rotation_deg ? `${jobResult.rotation_deg}°` : undefined,
      } : undefined;
    }
    case 'detect_aruco': {
      const ar = jobResult.aruco_result ?? {};
      const ids: number[] = ar.aruco_ids_found ?? [];
      const baselines: any[] = ar.aruco_baselines ?? [];
      const scaleMethod = baselines.length > 0 ? 'triangulated baseline (accurate)' : 'solvePnP single marker (less accurate)';
      const warning = ids.length === 1
        ? '⚠ Only 1 unique marker ID — scale via solvePnP only. Add a 2nd marker for better accuracy.'
        : ids.length === 0 ? '✗ No markers found' : undefined;
      return ids.length > 0 ? {
        'Markers found': ids.length,
        'Marker IDs': ids.map(String).join(', '),
        'Scale method': scaleMethod,
        'Baselines': baselines.length > 0 ? String(baselines.length) : undefined,
        'Frames checked': ar.aruco_frames_checked,
        'Floor marker': ar.aruco_floor_marker_id != null ? String(ar.aruco_floor_marker_id) : undefined,
        'Warning': warning,
      } : undefined;
    }
    case 'feature_matching':
      return jobResult.match_data_key ? {
        'Match file': jobResult.match_data_key.split('/').pop(),
        'Pairs matched': jobResult.pair_count,
        'Total matches': jobResult.total_matches,
      } : undefined;
    case 'sfm': {
      const reg = jobResult.registered_images;
      const total = (jobResult.image_keys ?? jobResult.frame_keys ?? []).length;
      const gaps: string[] = jobResult.sfm_gap_warnings ?? [];
      const dropPct = total > 0 ? Math.round((total - reg) / total * 100) : 0;
      return reg != null ? {
        'Cameras registered': total > 0 ? `${reg} / ${total} (${dropPct}% dropped)` : String(reg),
        'Sparse points': jobResult.num_points3D != null
          ? Number(jobResult.num_points3D).toLocaleString() : undefined,
        'Reprojection error': jobResult.mean_reprojection_error != null
          ? `${Number(jobResult.mean_reprojection_error).toFixed(2)}px` : undefined,
        'Warning': gaps.length > 0 ? gaps.join(' | ') : undefined,
      } : undefined;
    }
    case 'mvs': {
      const pts = jobResult.dense_point_count;
      return pts != null ? {
        'Dense points': Number(pts).toLocaleString(),
        'Runtime': jobResult.runtime_seconds != null
          ? `${Math.round(jobResult.runtime_seconds)}s` : undefined,
      } : undefined;
    }
    case 'scale_from_aruco': {
      const scale = jobResult.confirmed_scale_factor;
      const diag  = jobResult.scale_diagnostics ?? {};
      return scale != null ? {
        'Scale factor': `${Number(scale).toFixed(5)} m/unit`,
        'Estimates used': diag.scale_n_inliers,
        'Std dev': diag.scale_std != null ? String(diag.scale_std) : undefined,
        'Gravity up': jobResult.gravity_up_world
          ? jobResult.gravity_up_world.map((v: number) => v.toFixed(3)).join(', ') : undefined,
      } : {
        'Status': `Not derived — ${diag.error ?? 'unknown reason'}`,
      };
    }
    case 'apply_scale':
      return jobResult.scale_applied != null ? {
        'Scale applied': jobResult.scale_applied ? 'yes' : 'no (unscaled)',
        'Factor used': jobResult.scale_factor != null
          ? `${Number(jobResult.scale_factor).toFixed(5)} m/unit` : undefined,
      } : undefined;
    case 'coverage': {
      const score = jobResult.coverage_score;
      return score !== undefined ? {
        'Coverage score': `${(score * 100).toFixed(1)}%`,
        'Re-shoot areas': (jobResult.suggestions ?? []).length,
      } : undefined;
    }
    case 'export': {
      const exports = jobResult.exports ?? [];
      return exports.length > 0 ? {
        'Files exported': exports.map((e: any) => e.key.split('/').pop()).join(', '),
      } : undefined;
    }
    default:
      return undefined;
  }
}

// ── Helper: one-liner summary for a completed stage ───────────────────────────

export function buildStageSummary(
  stageId: string,
  detail: Record<string, string | number | string[] | undefined>,
): string | undefined {
  const v = (k: string): string | undefined =>
    detail[k] != null && detail[k] !== '' ? String(detail[k]) : undefined;
  const join = (...parts: (string | undefined)[]) =>
    parts.filter(Boolean).join(' · ') || undefined;

  switch (stageId) {
    case 'extract_metadata':
      return join(v('Resolution'), v('Focal length') && `fl=${v('Focal length')}`, v('Device'));
    case 'extract_frames':
      return join(v('Frames selected') && `${v('Frames selected')} frames`, v('Dropped (poor quality)') && `${v('Dropped (poor quality)')} dropped`);
    case 'detect_aruco':
      return join(v('Markers found') && `${v('Markers found')} markers`, v('Baselines') && `${v('Baselines')} baselines`, v('Scale method'));
    case 'feature_matching':
      return join(v('Pairs matched') && `${v('Pairs matched')} pairs`, v('Total matches') && `${v('Total matches')} matches`);
    case 'sfm':
      return join(v('Cameras registered'), v('Sparse points') && `${v('Sparse points')} pts`, v('Reprojection error'));
    case 'mvs':
      return join(v('Dense points') && `${v('Dense points')} pts`, v('Runtime'));
    case 'scale_from_aruco':
      return v('Scale factor') ?? v('Status');
    case 'apply_scale':
      return join(v('Scale applied') && `scale ${v('Scale applied')}`, v('Factor used'));
    case 'coverage':
      return join(v('Coverage score'), v('Re-shoot areas') && `${v('Re-shoot areas')} re-shoot area(s)`);
    case 'export':
      return v('Files exported');
    default:
      return undefined;
  }
}
