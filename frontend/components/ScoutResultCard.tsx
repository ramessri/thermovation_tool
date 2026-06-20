'use client';

/**
 * ScoutResultCard — shown between the scout and full pipeline phases.
 * Displays what the scout run found and what the full run will use.
 */

interface Props {
  calibration: {
    target_frames?: number;
    match_window?: number;
    mvs_min_consistent?: number;
    scout_reg_rate?: number;
    scout_reproj_error?: number;
    scout_n_points?: number;
    scout_scale_factor?: number | null;
  } | null;
  fullRunStatus: 'pending' | 'running' | 'done';
}

function QualityLabel({ value, thresholds }: { value: number; thresholds: [number, number] }) {
  if (value >= thresholds[1]) return <span className="text-green-600 font-semibold">excellent</span>;
  if (value >= thresholds[0]) return <span className="text-amber-500 font-semibold">good</span>;
  return <span className="text-red-500 font-semibold">low</span>;
}

export function ScoutResultCard({ calibration, fullRunStatus }: Props) {
  if (!calibration) return null;

  const {
    target_frames,
    match_window,
    mvs_min_consistent,
    scout_reg_rate,
    scout_reproj_error,
    scout_scale_factor,
  } = calibration;

  const mvqLabel = mvs_min_consistent === 4 ? 'high' : mvs_min_consistent === 3 ? 'standard' : 'generous';

  return (
    <div className="rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-3">
      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-indigo-600 font-bold text-sm">
          ✓
        </div>
        <div className="flex-1">
          <h3 className="text-sm font-semibold text-indigo-900">Scout Complete — Full Run Calibrated</h3>

          {/* Scout metrics */}
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-indigo-700">
            {scout_reg_rate != null && (
              <span>
                Registration: <span className="font-mono">{(scout_reg_rate * 100).toFixed(0)}%</span>{' '}
                (<QualityLabel value={scout_reg_rate} thresholds={[0.7, 0.88]} />)
              </span>
            )}
            {scout_reproj_error != null && (
              <span>
                Accuracy: <span className="font-mono">{scout_reproj_error.toFixed(2)}px</span>{' '}
                (<QualityLabel value={1 - scout_reproj_error / 2} thresholds={[0.4, 0.7]} />)
              </span>
            )}
            {scout_scale_factor != null && (
              <span>Scale: <span className="font-mono">{scout_scale_factor.toFixed(5)} m/unit</span></span>
            )}
          </div>

          {/* Calibrated parameters */}
          <div className="mt-2.5 rounded-lg bg-white border border-indigo-100 px-3 py-2">
            <p className="text-xs font-medium text-slate-600 mb-1.5">Full run parameters:</p>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-slate-400">Frames</span>
                <div className="font-semibold text-slate-800">{target_frames ?? '—'}</div>
              </div>
              <div>
                <span className="text-slate-400">Match window</span>
                <div className="font-semibold text-slate-800">{match_window ?? '—'}</div>
              </div>
              <div>
                <span className="text-slate-400">MVS quality</span>
                <div className="font-semibold text-slate-800">{mvqLabel}</div>
              </div>
            </div>
          </div>

          {/* Status */}
          <div className="mt-2 text-xs text-indigo-600">
            {fullRunStatus === 'pending' && '⏳ Full pipeline queued…'}
            {fullRunStatus === 'running' && '🔄 Full pipeline running with calibrated settings…'}
            {fullRunStatus === 'done'    && '✓ Full pipeline complete'}
          </div>
        </div>
      </div>
    </div>
  );
}
