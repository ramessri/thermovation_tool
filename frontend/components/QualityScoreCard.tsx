'use client';

import { useState } from 'react';

interface QualityMetric {
  label: string;
  value: string | null;
  score: number | null;   // 0-1, null = not available
  description: string;
  moreDetail?: string;
}

interface Props {
  registeredImages?: number;
  totalFrames?: number;
  reprojectionError?: number;
  densePoints?: number;
  pointDensity?: number;
  coverageScore?: number;
  scaleConfidence?: number;  // 1 - (std/mean), 0-1
  scaleFactor?: number | null;
}

function scoreColor(s: number | null): string {
  if (s === null) return 'text-slate-400';
  if (s >= 0.75) return 'text-green-600';
  if (s >= 0.50) return 'text-amber-500';
  return 'text-red-500';
}

function scoreBg(s: number | null): string {
  if (s === null) return 'bg-slate-100';
  if (s >= 0.75) return 'bg-green-50 border-green-200';
  if (s >= 0.50) return 'bg-amber-50 border-amber-200';
  return 'bg-red-50 border-red-200';
}

function ScoreBar({ score }: { score: number | null }) {
  if (score === null) return <div className="h-1.5 w-full rounded bg-slate-100" />;
  const pct = Math.round(score * 100);
  const color = score >= 0.75 ? 'bg-green-500' : score >= 0.50 ? 'bg-amber-400' : 'bg-red-400';
  return (
    <div className="h-1.5 w-full rounded bg-slate-100 overflow-hidden">
      <div className={`h-full rounded ${color} transition-all`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function MetricRow({ metric }: { metric: QualityMetric }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="py-2 border-b border-slate-100 last:border-0">
      <div
        className="flex items-center gap-3 cursor-pointer select-none"
        onClick={() => setOpen(o => !o)}
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-slate-700">{metric.label}</span>
            <span className={`text-xs font-mono font-semibold ${scoreColor(metric.score)}`}>
              {metric.value ?? '—'}
            </span>
          </div>
          <ScoreBar score={metric.score} />
        </div>
        <span className="text-slate-300 text-xs shrink-0">{open ? '▲' : '▼'}</span>
      </div>
      {open && (
        <p className="mt-1.5 text-xs text-slate-500 leading-relaxed pr-4">
          {metric.description}
          {metric.moreDetail && <span className="text-slate-400"> {metric.moreDetail}</span>}
        </p>
      )}
    </div>
  );
}

export function QualityScoreCard({
  registeredImages,
  totalFrames,
  reprojectionError,
  densePoints,
  pointDensity,
  coverageScore,
  scaleConfidence,
  scaleFactor,
}: Props) {
  const [expanded, setExpanded] = useState(false);

  // ── Compute per-metric scores ─────────────────────────────────────────────

  // Registration rate
  const regRate = (registeredImages != null && totalFrames != null && totalFrames > 0)
    ? registeredImages / totalFrames : null;
  const regScore = regRate != null ? Math.min(1, regRate / 0.85) : null;  // 85%+ = full score

  // Geometric accuracy (reprojection error): <0.5px=1.0, 1px=0.8, 2px=0.5, >3px=0
  const reproj = reprojectionError ?? null;
  const reprojScore = reproj != null ? Math.max(0, 1 - (reproj - 0.5) / 2.5) : null;

  // Point density: >10k pts/m³=1.0, >5k=0.7, >1k=0.4, lower=poor
  const densityScore = pointDensity != null
    ? Math.min(1, Math.log10(Math.max(1, pointDensity)) / Math.log10(10000))
    : null;

  // Surface coverage
  const covScore = coverageScore ?? null;

  // Scale confidence
  const scaleScore = scaleFactor != null
    ? (scaleConfidence ?? 0.8)   // if scale derived, default confidence 0.8 unless we have std
    : 0;                          // no scale = 0

  // Overall weighted score
  const weights = [0.25, 0.25, 0.20, 0.20, 0.10];
  const scores  = [regScore, reprojScore, densityScore, covScore, scaleScore];
  const available = scores.filter(s => s !== null);
  const availableWeightSum = weights.reduce((sum, w, i) => sum + (scores[i] !== null ? w : 0), 0);
  const overall: number | null = available.length >= 2
    ? scores.reduce<number>((sum, s, i) => sum + (s ?? 0) * weights[i], 0) / availableWeightSum
    : null;

  const overallPct = overall != null ? Math.round(overall * 100) : null;

  const metrics: QualityMetric[] = [
    {
      label: 'Registration rate',
      value: regRate != null ? `${Math.round(regRate * 100)}% (${registeredImages}/${totalFrames} frames)` : null,
      score: regScore,
      description: 'What fraction of your video frames were successfully incorporated into the 3D model. Frames are dropped when there is too little overlap with the rest of the scene — usually during fast pans, in low-texture areas, or when the camera is very close to a surface.',
      moreDetail: '85%+ is excellent. Below 50% suggests the video needs more careful coverage with slower motion.',
    },
    {
      label: 'Geometric accuracy',
      value: reproj != null ? `${reproj.toFixed(2)}px reprojection error` : null,
      score: reprojScore,
      description: 'Average pixel error when 3D points are projected back into the original photos. Lower is better. This reflects how precisely the camera positions and 3D point locations were solved.',
      moreDetail: 'Below 1px is excellent. Above 2px may indicate a poor focal length estimate or degenerate camera geometry.',
    },
    {
      label: 'Point density',
      value: pointDensity != null
        ? `${pointDensity.toLocaleString()} pts/${scaleFactor ? 'm³' : 'unit³'}`
        : (densePoints != null ? `${Number(densePoints).toLocaleString()} total points` : null),
      score: densityScore,
      description: 'How many 3D points were reconstructed per unit of scanned volume. Higher density means more geometric detail. Featureless surfaces (white walls, smooth floors) contribute almost no points regardless of how well you filmed them.',
      moreDetail: scaleFactor == null ? 'Density in SfM units — will be more meaningful once metric scale is derived.' : undefined,
    },
    {
      label: 'Surface coverage',
      value: covScore != null ? `${(covScore * 100).toFixed(1)}%` : null,
      score: covScore,
      description: 'What fraction of the reconstructed point cloud surface was visible from the recorded camera positions. This measures how thoroughly you photographed the geometry — not whether every wall was physically visited. Smooth, textureless surfaces score near-zero since they produce very few points for the algorithm to evaluate.',
      moreDetail: 'A room with good physical coverage but white walls will still score low here. This is a known limitation of HPR-based coverage metrics on featureless indoor scenes.',
    },
    {
      label: 'Metric scale',
      value: scaleFactor != null
        ? `${scaleFactor.toFixed(5)} m/unit`
        : 'not derived',
      score: scaleScore,
      description: 'Whether a real-world metric scale was successfully derived from the ArUco markers. Without scale, the reconstruction is accurate in shape but not in absolute size — distances and volumes will be in arbitrary SfM units.',
      moreDetail: scaleFactor == null
        ? 'Ensure 2+ ArUco markers are visible in the same frame and appear in frames that were successfully registered by SfM.'
        : `Scale confidence: ${scaleConfidence != null ? (scaleConfidence * 100).toFixed(0) + '%' : 'not measured'}.`,
    },
  ];

  return (
    <div className={`rounded-xl border px-4 py-3 ${scoreBg(overall)}`}>
      <div
        className="flex items-center gap-4 cursor-pointer select-none"
        onClick={() => setExpanded(e => !e)}
      >
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <span className="text-sm font-semibold text-slate-800">Reconstruction Quality</span>
            {overallPct != null && (
              <span className={`text-2xl font-bold tabular-nums ${scoreColor(overall)}`}>
                {overallPct}<span className="text-base font-medium">/100</span>
              </span>
            )}
          </div>
          {!expanded && (
            <p className="text-xs text-slate-500 mt-0.5">
              {metrics.filter(m => m.score !== null).map(m => m.value).filter(Boolean).join(' · ')}
            </p>
          )}
        </div>
        <span className="text-slate-400 text-xs">{expanded ? '▲' : '▼'}</span>
      </div>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-slate-200">
          {metrics.map(m => <MetricRow key={m.label} metric={m} />)}
        </div>
      )}
    </div>
  );
}
