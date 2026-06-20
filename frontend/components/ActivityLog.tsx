'use client';

import { useEffect, useRef, useState } from 'react';
import { CheckCircle, AlertTriangle, Cpu, Zap } from 'lucide-react';

export interface WsEvent {
  type?: string;
  stage?: string;
  progress?: number;
  message?: string;
  ts?: number;
  elapsed?: string;
  gpu?: { util_pct: number; vram_used_mb: number; vram_total_mb: number } | null;
  duration_s?: number;
  summary?: string;
  metrics?: Record<string, unknown>;
  warnings?: string[];
  errors?: string[];
}

interface StageState {
  lastProgress?: { progress: number; message: string; ts: number };
  completion?: {
    duration_s: number;
    summary: string;
    metrics: Record<string, unknown>;
    warnings: string[];
    errors: string[];
    ts: number;
  };
}

interface ActivityLogProps {
  events: WsEvent[];
  isRunning: boolean;
}

const STAGE_LABELS: Record<string, string> = {
  // Active pipeline stages
  extract_metadata:    'Video Metadata',
  extract_frames:      'Extract Frames',
  detect_aruco:        'ArUco Detection',
  feature_matching:    'Feature Matching',
  sfm:                 'Structure from Motion',
  mvs:                 'Dense Reconstruction',
  scale_from_aruco:    'Scale Derivation',
  apply_scale:         'Apply Metric Scale',
  coverage:            'Coverage Analysis',
  export:              'Export',
  // Optional
  gaussian_splatting:  '3D Gaussian Splatting',
};

function formatDuration(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return rem > 0 ? `${m}m ${rem}s` : `${m}m`;
}

function formatSecondsAgo(seconds: number): string {
  if (seconds < 5) return 'just now';
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  return `${Math.round(seconds / 60)}m ago`;
}

export function ActivityLog({ events, isRunning }: ActivityLogProps) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [expandedStages, setExpandedStages] = useState<Set<string>>(new Set());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events.length]);

  // Derive state from events
  const stageOrder: string[] = [];
  const stageMap: Record<string, StageState> = {};
  let lastHeartbeat: WsEvent | null = null;
  let latestGpu: WsEvent['gpu'] = null;
  let currentStageElapsed: string | null = null;

  for (const ev of events) {
    const stage = ev.stage ?? '';
    if (ev.type === 'heartbeat') {
      lastHeartbeat = ev;
      if (ev.gpu) latestGpu = ev.gpu;
      if (ev.elapsed) currentStageElapsed = ev.elapsed;
      continue;
    }
    if (!stage) continue;
    if (!stageMap[stage]) {
      stageMap[stage] = {};
      stageOrder.push(stage);
    }
    if (ev.type === 'stage_complete') {
      stageMap[stage].completion = {
        duration_s: ev.duration_s ?? 0,
        summary: ev.summary ?? '',
        metrics: ev.metrics ?? {},
        warnings: ev.warnings ?? [],
        errors: ev.errors ?? [],
        ts: ev.ts ?? now,
      };
    } else if (ev.type === 'progress' || !ev.type) {
      stageMap[stage].lastProgress = {
        progress: ev.progress ?? 0,
        message: ev.message ?? '',
        ts: ev.ts ?? now,
      };
    }
  }

  const heartbeatAge = lastHeartbeat?.ts ? now - lastHeartbeat.ts : null;
  const isAlive = heartbeatAge !== null && heartbeatAge < 30;
  const isPulsing = isRunning && isAlive;

  function toggleStage(stage: string) {
    setExpandedStages(prev => {
      const next = new Set(prev);
      if (next.has(stage)) next.delete(stage);
      else next.add(stage);
      return next;
    });
  }

  if (events.length === 0 && !isRunning) return null;

  return (
    <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-slate-50">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-800">Activity Log</span>
          {isRunning && (
            <span className="flex items-center gap-1.5 text-xs text-slate-500">
              {isPulsing ? (
                <>
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
                  </span>
                  <span className="text-green-700">
                    {currentStageElapsed ? `${currentStageElapsed} elapsed` : heartbeatAge !== null ? formatSecondsAgo(heartbeatAge) : 'live'}
                  </span>
                </>
              ) : (
                <>
                  <span className="h-2 w-2 rounded-full bg-amber-400" />
                  <span className="text-amber-700">
                    {heartbeatAge !== null ? `last seen ${formatSecondsAgo(heartbeatAge)}` : 'waiting…'}
                  </span>
                </>
              )}
            </span>
          )}
        </div>

        {/* GPU stats from latest heartbeat */}
        {latestGpu && isRunning && (
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <Zap className="h-3 w-3 text-violet-500" />
              GPU {latestGpu.util_pct}%
            </span>
            <span className="flex items-center gap-1">
              <Cpu className="h-3 w-3 text-slate-400" />
              {Math.round(latestGpu.vram_used_mb / 1024 * 10) / 10}/{Math.round(latestGpu.vram_total_mb / 1024)}GB
            </span>
          </div>
        )}
      </div>

      {/* Stage rows */}
      <div className="divide-y divide-slate-100 max-h-96 overflow-y-auto">
        {stageOrder.length === 0 && isRunning && (
          <div className="px-4 py-3 text-sm text-slate-400 italic">Waiting for first stage…</div>
        )}

        {stageOrder.map((stage) => {
          const s = stageMap[stage];
          const label = STAGE_LABELS[stage] ?? stage;
          const done = !!s.completion;
          const expanded = expandedStages.has(stage);
          const hasWarnings = (s.completion?.warnings.length ?? 0) > 0;
          const hasErrors = (s.completion?.errors.length ?? 0) > 0;

          return (
            <div key={stage}>
              <div
                className={`flex items-start gap-3 px-4 py-3 ${done ? 'cursor-pointer hover:bg-slate-50' : ''}`}
                onClick={() => done && toggleStage(stage)}
              >
                {/* Status icon */}
                <div className="flex-shrink-0 mt-0.5">
                  {done ? (
                    hasErrors ? (
                      <AlertTriangle className="h-4 w-4 text-red-500" />
                    ) : (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    )
                  ) : (
                    <span className="relative flex h-4 w-4 items-center justify-center">
                      <span className="animate-ping absolute inline-flex h-3 w-3 rounded-full bg-brand-400 opacity-50" />
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-500" />
                    </span>
                  )}
                </div>

                {/* Stage info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`text-sm font-medium ${done ? 'text-slate-700' : 'text-slate-900'}`}>
                      {label}
                    </span>
                    {done && (
                      <span className="text-xs text-slate-400">{formatDuration(s.completion!.duration_s)}</span>
                    )}
                    {hasWarnings && !hasErrors && (
                      <span className="text-xs rounded-full bg-amber-100 text-amber-700 px-2 py-0.5 font-medium">
                        {s.completion!.warnings.length} warning{s.completion!.warnings.length > 1 ? 's' : ''}
                      </span>
                    )}
                    {hasErrors && (
                      <span className="text-xs rounded-full bg-red-100 text-red-700 px-2 py-0.5 font-medium">
                        {s.completion!.errors.length} error{s.completion!.errors.length > 1 ? 's' : ''}
                      </span>
                    )}
                  </div>

                  {done ? (
                    <p className="text-xs text-slate-500 mt-0.5 truncate">{s.completion!.summary}</p>
                  ) : s.lastProgress ? (
                    <>
                      <div className="mt-1.5 h-1 rounded-full bg-slate-100 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-brand-500 transition-all duration-500"
                          style={{ width: `${Math.round(s.lastProgress.progress * 100)}%` }}
                        />
                      </div>
                      <p className="text-xs text-slate-500 mt-1 truncate">{s.lastProgress.message}</p>
                    </>
                  ) : null}
                </div>

                {/* Expand chevron for completed stages */}
                {done && (
                  <span className={`flex-shrink-0 text-slate-400 text-xs transition-transform ${expanded ? 'rotate-90' : ''}`}>▶</span>
                )}
              </div>

              {/* Expanded summary */}
              {done && expanded && (
                <div className="px-11 pb-3 space-y-2">
                  {/* Metrics chips */}
                  {Object.keys(s.completion!.metrics).length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {Object.entries(s.completion!.metrics).map(([k, v]) => (
                        <span key={k} className="text-xs rounded bg-slate-100 px-2 py-0.5 text-slate-600">
                          <span className="font-medium">{k.replace(/_/g, ' ')}:</span>{' '}
                          {typeof v === 'number' && v > 1000 ? v.toLocaleString() : String(v)}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Warnings */}
                  {s.completion!.warnings.map((w, i) => (
                    <div key={i} className="flex items-start gap-1.5 text-xs text-amber-700 bg-amber-50 rounded px-2 py-1.5">
                      <AlertTriangle className="h-3 w-3 flex-shrink-0 mt-0.5" />
                      <span>{w}</span>
                    </div>
                  ))}

                  {/* Errors */}
                  {s.completion!.errors.map((e, i) => (
                    <div key={i} className="flex items-start gap-1.5 text-xs text-red-700 bg-red-50 rounded px-2 py-1.5">
                      <AlertTriangle className="h-3 w-3 flex-shrink-0 mt-0.5" />
                      <span>{e}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
