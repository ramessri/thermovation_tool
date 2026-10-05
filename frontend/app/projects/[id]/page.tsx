'use client';

import { useEffect, useState, useRef } from 'react';
import { useParams } from 'next/navigation';
import { ArrowLeft, Upload, CheckCircle, AlertCircle, Clock, X, Camera, Pencil, ChevronDown, ChevronUp, Film, Play } from 'lucide-react';
import Link from 'next/link';
import {
  getProject, uploadFile, launchPipeline, connectProgress,
  getJobResult, cancelPipeline, uploadCalibrationPhoto,
  renameProject, listProjectUploads, ProjectUpload, storageUrl,
} from '@/lib/api';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { ResultsViewer } from '@/components/ResultsViewer';
import { QualityScoreCard } from '@/components/QualityScoreCard';
import { ScoutResultCard } from '@/components/ScoutResultCard';
import { SuggestionsPanel } from '@/components/SuggestionsPanel';
import { ActivityLog, WsEvent } from '@/components/ActivityLog';
import { PipelineLog, StageDetail, buildStageDetails, buildStageDetailsFromMeta } from '@/components/PipelineLog';
import { ShootingGuide } from '@/components/ShootingGuide';
import { DimensionsCard, Dimensions } from '@/components/DimensionsCard';
import type { HvacPlacementResult, HvacSegmentation } from '@/components/HvacSegmentationViewer';
import { API_BASE } from '@/lib/api';

interface ProjectDetail {
  id: string;
  name: string;
  description: string;
  status: string;
  scene_type?: string;
  marker_type?: string;
  confirmed_scale_factor?: number;
  confirmed_scale_source?: string;
  coverage_runs?: Array<{ ts: number; score: number; cloud_key: string; n_suggestions: number }> | null;
  splat_key?: string | null;
  mesh_key?: string | null;
  lingbot_cloud_key?: string | null;
  lingbot_mesh_key?: string | null;
  metricanything_cloud_key?: string | null;
  metricanything_mesh_key?: string | null;
  hvac_mode?: boolean;
  hvac_placement?: HvacPlacementResult | null;
  hvac_segmentation?: HvacSegmentation | null;
  suggestions?: any[] | null;
  gravity_up_world?: number[] | null;
  pipeline_mode?: string | null;
  scout_calibration?: Record<string, any> | null;
  created_at?: string | null;
  pipeline_results?: Record<string, any> | null;
  scan_source?: 'video' | 'lidar_ply';
  dimensions?: Dimensions | null;
}

interface ProgressUpdate {
  type?: string;
  stage: string;
  progress: number;
  message: string;
}

interface GpuStats {
  util_pct: number;
  vram_used_mb: number;
  vram_total_mb: number;
}

/** Derive a 0–1 confidence score from scale diagnostics.
 *  Returns undefined when there aren't enough inliers to be meaningful. */
function scaleConfidenceFromDiag(diag: any): number | undefined {
  if (!diag || !diag.scale_n_inliers || diag.scale_n_inliers <= 1) return undefined;
  const f = diag.scale_factor as number | undefined;
  const s = diag.scale_std   as number | undefined;
  if (!f || !s || f === 0) return undefined;
  return Math.max(0, 1 - s / f);
}

const SLOW_STAGES = new Set(['mvs', 'gaussian_splatting', 'feature_matching', 'sfm']);
const SLOW_STALL_MS = 20 * 60 * 1000;
const FAST_STALL_MS = 3 * 60 * 1000;

// New linear pipeline — single phase, no user gate.
const STAGES = [
  { id: 'ingest_lidar_ply',  label: 'Ingesting LiDAR Cloud',    order: 0 },
  { id: 'extract_metadata',  label: 'Video Metadata',          order: 0 },
  { id: 'extract_frames',    label: 'Extracting Frames',        order: 1 },
  { id: 'detect_aruco',      label: 'ArUco Marker Detection',   order: 2 },
  { id: 'feature_matching',  label: 'Feature Matching',         order: 3 },
  { id: 'sfm',               label: 'Structure from Motion',    order: 4 },
  { id: 'mvs',               label: 'Dense Reconstruction',     order: 5 },
  { id: 'detect_aruco_sfm',  label: 'ArUco (Registered Frames)', order: 6 },
  { id: 'scale_from_aruco',  label: 'Scale Derivation',         order: 7 },
  { id: 'apply_scale',       label: 'Applying Scale',           order: 8 },
  { id: 'fill_planes',       label: 'Filling Surface Planes',   order: 9 },
  { id: 'refine_cloud',      label: 'Refining Point Cloud',     order: 10 },
  { id: 'lingbot_fusion',    label: 'Densifying (LingBot)',     order: 10 },
  { id: 'metricanything_fusion', label: 'Densifying (MetricAnything)', order: 10 },
  { id: 'wall_plane_detection', label: 'Detecting Wall Planes',   order: 10 },
  { id: 'detect_hvac_fixtures', label: 'Detecting HVAC Fixtures', order: 10 },
  { id: 'locate_rucklauf',   label: 'Locating Rücklauf',          order: 10 },
  { id: 'hvac_placement',    label: 'HVAC Placement',             order: 10 },
  { id: 'coverage',          label: 'Analyzing Coverage',       order: 10 },
  { id: 'export',            label: 'Exporting Results',        order: 11 },
  // Scout pipeline stages (only visible for scout runs)
  { id: 'scout_calibrate',   label: 'Scout Calibration',        order: 12 },
];

function NeedsMorePanel({
  projectId, coverageScore, suggestions, onSupplementalLaunched,
}: {
  projectId: string;
  coverageScore?: number;
  suggestions?: any[];
  onSupplementalLaunched: (taskId: string) => void;
}) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress]   = useState(0);
  const [error, setError]         = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      setUploading(true); setError(null);
      const { upload_id } = await uploadFile(projectId, file, setProgress);
      const res = await fetch(
        `${API_BASE}/api/projects/${projectId}/launch_supplemental?upload_id=${upload_id}`,
        { method: 'POST' },
      );
      if (!res.ok) throw new Error(`Launch failed: ${res.status}`);
      const { task_id } = await res.json();
      onSupplementalLaunched(task_id);
    } catch (err: any) {
      setError(err.message ?? 'Upload failed');
    } finally {
      setUploading(false);
    }
  }

  return (
    <Card padding="lg" className="border-amber-200 bg-amber-50">
      <div className="mb-4 flex items-start gap-3">
        <Camera className="h-6 w-6 text-amber-600 mt-0.5 shrink-0" />
        <div>
          <h3 className="text-lg font-semibold text-amber-900">More footage needed</h3>
          <p className="text-sm text-amber-800 mt-1">
            Coverage score is {coverageScore != null ? `${(coverageScore * 100).toFixed(0)}%` : 'low'} —
            some areas weren't captured well enough. Upload another video targeting the areas below.
          </p>
        </div>
      </div>

      {suggestions && suggestions.length > 0 && (
        <div className="mb-4">
          <p className="text-xs text-amber-700 mb-2">
            <strong>{suggestions.length} area{suggestions.length !== 1 ? 's' : ''}</strong> need more
            coverage. Film one continuous video visiting each area in order.
          </p>
          <SuggestionsPanel projectId={projectId} apiBase={API_BASE} suggestions={suggestions} />
        </div>
      )}

      {error && <p className="text-sm text-red-700 mb-3">{error}</p>}

      <input ref={fileRef} type="file" accept="video/*" className="hidden" onChange={handleFile} />
      <button
        onClick={() => fileRef.current?.click()}
        disabled={uploading}
        className="inline-flex items-center gap-2 rounded-lg bg-amber-600 px-4 py-2 text-sm font-semibold text-white hover:bg-amber-700 disabled:opacity-50 transition-colors"
      >
        <Upload className="h-4 w-4" />
        {uploading ? `Uploading… ${progress}%` : 'Upload supplemental footage'}
      </button>
      <p className="mt-2 text-xs text-amber-700">
        The new video is registered into the existing reconstruction — no need to re-scan the whole scene.
      </p>
    </Card>
  );
}

export default function ProjectDetailPage() {
  const params    = useParams();
  const projectId = params.id as string;

  const [project, setProject]               = useState<ProjectDetail | null>(null);
  const [loading, setLoading]               = useState(true);
  const [uploading, setUploading]           = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [pipelineProgress, setPipelineProgress] = useState<ProgressUpdate | null>(null);
  const [wsEvents, setWsEvents]             = useState<WsEvent[]>([]);
  const [pipelineStatus, setPipelineStatus] = useState<'idle' | 'running' | 'complete' | 'needs_more' | 'error'>('idle');
  const [errorMessage, setErrorMessage]     = useState<string | null>(null);
  const [wsConnected, setWsConnected]       = useState(false);
  const [stalled, setStalled]               = useState(false);
  const [cancelling, setCancelling]         = useState(false);
  const stallTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [taskId, setTaskId]                 = useState<string | null>(null);
  const [jobResult, setJobResult]           = useState<Awaited<ReturnType<typeof getJobResult>>['result'] | null>(null);
  const [stageLog, setStageLog]             = useState<Map<string, StageDetail>>(new Map());
  const [pipelineMode, setPipelineMode]     = useState<'standard' | 'scout'>('standard');
  const [scanSource, setScanSource]         = useState<'video' | 'lidar_ply'>('video');
  const [lidarScaleFactor, setLidarScaleFactor] = useState(1.0);
  const [showGroundTruth, setShowGroundTruth] = useState(false);
  const [gtLength, setGtLength]             = useState('');
  const [gtBreadth, setGtBreadth]           = useState('');
  const [gtHeight, setGtHeight]             = useState('');
  const [calibFile, setCalibFile]           = useState<File | null>(null);
  const [calibUploading, setCalibUploading] = useState(false);
  const [calibResult, setCalibResult]       = useState<{ focal_length_px?: number; make?: string; model?: string } | null>(null);
  const [calibError, setCalibError]         = useState<string | null>(null);
  const calibInputRef   = useRef<HTMLInputElement>(null);
  const fileInputRef    = useRef<HTMLInputElement>(null);
  const wsDisconnect    = useRef<(() => void) | null>(null);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const lastSeenModeRef = useRef<string | null>(null);

  // Rename
  const [isEditingName, setIsEditingName] = useState(false);
  const [editingNameValue, setEditingNameValue] = useState('');
  const [savingName, setSavingName] = useState(false);

  // Pipeline accordion (collapsed by default when complete)
  const [pipelineOpen, setPipelineOpen] = useState(true);

  // Footage review
  const [footageOpen, setFootageOpen] = useState(false);
  const [uploads, setUploads] = useState<ProjectUpload[]>([]);
  const [uploadsLoading, setUploadsLoading] = useState(false);

  // Carousel modal
  const [carouselOpen, setCarouselOpen] = useState(false);
  const [carouselIdx, setCarouselIdx] = useState(0);

  // Track when each stage started (for live elapsed display)
  const stageStartRef = useRef<Map<string, number>>(new Map());

  // GPU stats from heartbeat WS events (null = no heartbeat yet; { util_pct:0, vram_total_mb:0 } = CPU stage)
  const [gpuStats, setGpuStats] = useState<(GpuStats & { available: boolean }) | null>(null);

  async function handleSaveName() {
    if (!editingNameValue.trim() || !project) return;
    setSavingName(true);
    try {
      const updated = await renameProject(projectId, editingNameValue.trim());
      setProject(p => p ? { ...p, name: updated.name } : p);
      setIsEditingName(false);
    } catch {
      // keep editing open on error
    } finally {
      setSavingName(false);
    }
  }

  function populateStageLogFromResults(results: Record<string, any>) {
    setStageLog(prev => {
      const next = new Map(prev);
      for (const [stageId, meta] of Object.entries(results)) {
        if (!meta || typeof meta !== 'object') continue;
        const stageConfig = STAGES.find(s => s.id === stageId);
        const existing = next.get(stageId);
        const detail = buildStageDetailsFromMeta(stageId, meta) ?? existing?.detail;
        next.set(stageId, {
          id: stageId,
          label: stageConfig?.label ?? stageId.replace(/_/g, ' '),
          phase: 1,
          status: 'done',
          lastMessage: meta._message ?? existing?.lastMessage,
          elapsedSeconds: meta._duration_s ?? existing?.elapsedSeconds,
          startedAt: meta._started_at ?? existing?.startedAt,
          finishedAt: meta._finished_at ?? existing?.finishedAt,
          detail: detail as any,
        } satisfies StageDetail);
      }
      return next;
    });
  }

  async function loadUploads() {
    setUploadsLoading(true);
    try {
      const data = await listProjectUploads(projectId);
      setUploads(data);
    } catch {
      // non-fatal
    } finally {
      setUploadsLoading(false);
    }
  }

  function updateStageLog(stage: string, progress: number, message: string, result?: Record<string, any>, meta?: Record<string, any>) {
    const now = Date.now();
    if (progress < 1.0 && !stageStartRef.current.has(stage)) {
      stageStartRef.current.set(stage, now);
    }
    const startedAt = stageStartRef.current.get(stage);
    const elapsedSeconds = (progress >= 1.0 && startedAt != null)
      ? (now - startedAt) / 1000
      : undefined;

    setStageLog(prev => {
      const next        = new Map(prev);
      const existing    = next.get(stage);
      const isDone      = progress >= 1.0;
      const detail      = isDone && result
        ? buildStageDetails(stage, result)
        : isDone && meta
          ? buildStageDetailsFromMeta(stage, meta)
          : existing?.detail;
      const stageConfig = STAGES.find(s => s.id === stage);
      next.set(stage, {
        id: stage,
        label: stageConfig?.label ?? stage.replace(/_/g, ' '),
        phase: 1,
        status: isDone ? 'done' : 'running',
        lastMessage: message,
        progress,
        detail: detail as any,
        elapsedSeconds: elapsedSeconds ?? existing?.elapsedSeconds,
        startedAt: startedAt != null ? startedAt / 1000 : existing?.startedAt,
        finishedAt: isDone && startedAt != null ? now / 1000 : existing?.finishedAt,
      });
      return next;
    });
  }

  useEffect(() => { loadProject(); }, [projectId]);

  // Auto-collapse pipeline when it finishes; re-open when it starts running again
  useEffect(() => {
    if (pipelineStatus === 'complete' || pipelineStatus === 'needs_more') {
      setPipelineOpen(false);
    } else if (pipelineStatus === 'running') {
      setPipelineOpen(true);
    }
  }, [pipelineStatus]);

  useEffect(() => {
    if (!taskId) return;
    const interval = setInterval(async () => {
      try {
        const job = await getJobResult(taskId);
        if (job.status === 'SUCCESS' && job.result) {
          const result = job.result as Record<string, unknown>;
          if (result.exports) {
            setJobResult(result as Awaited<ReturnType<typeof getJobResult>>['result']);
            clearInterval(interval);
          } else {
            clearInterval(interval);
          }
        } else if (job.status === 'FAILURE') {
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Job result poll error:', err);
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [taskId]);

  function resetStallTimer(stage?: string) {
    if (stallTimerRef.current) clearTimeout(stallTimerRef.current);
    setStalled(false);
    const delay = stage && SLOW_STAGES.has(stage) ? SLOW_STALL_MS : FAST_STALL_MS;
    stallTimerRef.current = setTimeout(() => setStalled(true), delay);
  }

  function clearStallTimer() {
    if (stallTimerRef.current) clearTimeout(stallTimerRef.current);
    stallTimerRef.current = null;
    setStalled(false);
  }

  async function handleCancel() {
    if (!projectId || cancelling) return;
    setCancelling(true);
    try {
      await cancelPipeline(projectId);
      clearStallTimer();
      if (wsDisconnect.current) { wsDisconnect.current(); wsDisconnect.current = null; }
      setPipelineStatus('error');
      setWsConnected(false);
      setProject(p => p ? { ...p, status: 'failed' } : p);
    } catch (err) {
      console.error('Cancel failed:', err);
    } finally {
      setCancelling(false);
    }
  }

  async function loadProject() {
    try {
      setLoading(true);
      setErrorMessage(null);
      const data = await getProject(projectId);
      setProject(data);

      if (data.status === 'needs_more' || data.status === 'complete') {
        const pid     = data.id;
        const exports = [
          { label: 'Colored Point Cloud (.ply)', key: `${pid}/exports/output.ply`,  mime_type: 'application/octet-stream' },
          { label: 'Mesh (.obj)',                key: `${pid}/exports/output.obj`,  mime_type: 'model/obj' },
          { label: 'LiDAR Exchange (.las)',       key: `${pid}/exports/output.las`,  mime_type: 'application/octet-stream' },
        ];
        setJobResult({
          exports,
          coverage_score: data.coverage_score ?? undefined,
          suggestions:    data.suggestions ?? [],
        } as any);
        setPipelineStatus(data.status === 'needs_more' ? 'needs_more' : 'complete');
        if (data.pipeline_results) populateStageLogFromResults(data.pipeline_results);
      } else if (data.status === 'failed') {
        setPipelineStatus('error');
        if (data.pipeline_results) populateStageLogFromResults(data.pipeline_results);
      } else if (data.status === 'processing') {
        setPipelineStatus('running');
        // Seed stageLog with stages already completed in this run
        if (data.pipeline_results) populateStageLogFromResults(data.pipeline_results);
        const pid = data.id;
        lastSeenModeRef.current = data.pipeline_mode ?? null;

        if (wsDisconnect.current) wsDisconnect.current();
        wsDisconnect.current = connectProgress(
          pid,
          (msg) => {
            const ev = msg as WsEvent;
            setWsEvents(prev => [...prev, ev]);
            if (ev.stage) resetStallTimer(ev.stage);
            if (ev.type === 'heartbeat') {
              const g = (ev as any).gpu;
              setGpuStats(g ? { ...g, available: true } : { util_pct: 0, vram_used_mb: 0, vram_total_mb: 0, available: false });
            } else if (ev.type === 'stage_complete') {
              // stage_complete has no progress field — handle separately to avoid NaN
              if (ev.stage) {
                const meta = { ...(ev as any).metrics, _message: (ev as any).summary };
                updateStageLog(ev.stage, 1.0, (ev as any).summary ?? '', undefined, meta);
              }
            } else {
              setPipelineProgress(msg);
              if (ev.stage) {
                updateStageLog(ev.stage, ev.progress ?? 0, ev.message ?? '');
                if ((ev.progress ?? 0) >= 1.0) {
                  setTaskId(tid => {
                    if (tid) getJobResult(tid).then(r => {
                      if (r.result) updateStageLog(ev.stage!, 1.0, ev.message ?? '', r.result);
                    }).catch(() => {});
                    return tid;
                  });
                }
              }
            }
          },
          () => setWsConnected(false),
          () => setWsConnected(false),
        );
        setWsConnected(true);
        resetStallTimer();

        const pollInterval = setInterval(async () => {
          try {
            const latest = await getProject(pid);
            // Detect scout → full transition: reset stage log so full pipeline renders cleanly
            if (lastSeenModeRef.current === 'scout' && latest.pipeline_mode === 'full') {
              setStageLog(new Map());
              setPipelineProgress(null);
            }
            if (latest.pipeline_mode) lastSeenModeRef.current = latest.pipeline_mode;
            setProject(latest);
            if (latest.status === 'needs_more' || latest.status === 'complete') {
              clearInterval(pollInterval);
              const exports = [
                { label: 'Colored Point Cloud (.ply)', key: `${pid}/exports/output.ply`,  mime_type: 'application/octet-stream' },
                { label: 'Mesh (.obj)',                key: `${pid}/exports/output.obj`,  mime_type: 'model/obj' },
                { label: 'LiDAR Exchange (.las)',       key: `${pid}/exports/output.las`,  mime_type: 'application/octet-stream' },
              ];
              setJobResult({
                exports,
                coverage_score: latest.coverage_score ?? undefined,
                suggestions:    latest.suggestions ?? [],
              } as any);
              setPipelineStatus(latest.status === 'needs_more' ? 'needs_more' : 'complete');
            } else if (latest.status === 'failed') {
              clearInterval(pollInterval);
              clearStallTimer();
              setPipelineStatus('error');
            }
          } catch { /* keep polling */ }
        }, 10000);
      }
    } catch (error) {
      setErrorMessage('Failed to load project');
      console.error('Failed to load project:', error);
    } finally {
      setLoading(false);
    }
  }

  async function handleCalibPhoto(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setCalibFile(file);
    setCalibError(null);
    setCalibUploading(true);
    try {
      const result = await uploadCalibrationPhoto(projectId, file);
      setCalibResult(result);
    } catch {
      setCalibError('Could not read EXIF focal length from this photo — pipeline will use a heuristic estimate instead.');
      setCalibResult(null);
    } finally {
      setCalibUploading(false);
    }
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    setPendingFiles(prev => {
      // Deduplicate by name
      const existing = new Set(prev.map(f => f.name));
      return [...prev, ...files.filter(f => !existing.has(f.name))];
    });
    // Reset input so the same file can be re-added after removal
    e.target.value = '';
  }

  async function handleLaunch() {
    if (!pendingFiles.length) return;
    try {
      setUploading(true);
      setUploadProgress(0);
      setErrorMessage(null);
      setPipelineStatus('running');

      // Upload all files sequentially; track progress across total bytes
      const uploadIds: string[] = [];
      const totalBytes = pendingFiles.reduce((s, f) => s + f.size, 0);
      let uploadedBytes = 0;
      let primaryUploadId = '';

      for (const file of pendingFiles) {
        const fileBytes = file.size;
        const result = await uploadFile(projectId, file, (pct) => {
          const fileDone = (pct / 100) * fileBytes;
          setUploadProgress(Math.round(((uploadedBytes + fileDone) / totalBytes) * 100));
        });
        uploadIds.push(result.upload_id);
        uploadedBytes += fileBytes;
        // Use first video as primary for pipeline launch
        const isVideo = file.type.startsWith('video/') || /\.(mp4|mov|avi|mkv|webm)$/i.test(file.name);
        if (isVideo && !primaryUploadId) primaryUploadId = result.upload_id;
      }
      if (!primaryUploadId) primaryUploadId = uploadIds[0];

      const launchOpts: NonNullable<Parameters<typeof launchPipeline>[3]> = {};
      if (scanSource === 'lidar_ply') {
        launchOpts.scan_source = 'lidar_ply';
        launchOpts.lidar_scale_factor = lidarScaleFactor;
      }
      if (gtLength.trim())  launchOpts.ground_truth_length_m  = parseFloat(gtLength);
      if (gtBreadth.trim()) launchOpts.ground_truth_breadth_m = parseFloat(gtBreadth);
      if (gtHeight.trim())  launchOpts.ground_truth_height_m  = parseFloat(gtHeight);
      const launchResult = await launchPipeline(projectId, primaryUploadId, pipelineMode, launchOpts);
      if (launchResult?.task_id) setTaskId(launchResult.task_id);

      if (wsDisconnect.current) wsDisconnect.current();
      setWsEvents([]);
      setStageLog(new Map());
      wsDisconnect.current = connectProgress(
        projectId,
        (data) => {
          const ev = data as WsEvent;
          setWsEvents(prev => [...prev, ev]);
          if (ev.stage) resetStallTimer(ev.stage);
          if (ev.type === 'heartbeat') {
            const g = (ev as any).gpu;
            setGpuStats(g ? { ...g, available: true } : { util_pct: 0, vram_used_mb: 0, vram_total_mb: 0, available: false });
          } else if (ev.type === 'stage_complete') {
            if (ev.stage) {
              const meta = { ...(ev as any).metrics, _message: (ev as any).summary };
              updateStageLog(ev.stage, 1.0, (ev as any).summary ?? '', undefined, meta);
            }
          } else {
            setPipelineProgress(data);
            if (ev.stage) {
              updateStageLog(ev.stage, ev.progress ?? 0, ev.message ?? '');
              if ((ev.progress ?? 0) >= 1.0) {
                setTaskId(tid => {
                  if (tid) getJobResult(tid).then(r => {
                    if (r.result) updateStageLog(ev.stage!, 1.0, ev.message ?? '', r.result);
                  }).catch(() => {});
                  return tid;
                });
              }
            }
          }
        },
        () => setWsConnected(false),
        () => setWsConnected(false),
      );
      setWsConnected(true);
      resetStallTimer();
      setUploadProgress(0);

      // Poll project status so we detect completion of both scout and full pipeline phases.
      // The WS alone isn't enough: scout emits progress=1.0 before the full pipeline starts,
      // and the full pipeline launches with a new Celery task ID the frontend doesn't hold.
      const pid = projectId;
      lastSeenModeRef.current = pipelineMode === 'scout' ? 'scout' : null;
      const pollInterval = setInterval(async () => {
        try {
          const latest = await getProject(pid);
          // Detect scout → full transition: reset stage log so full pipeline renders cleanly
          if (lastSeenModeRef.current === 'scout' && latest.pipeline_mode === 'full') {
            setStageLog(new Map());
            setPipelineProgress(null);
          }
          if (latest.pipeline_mode) lastSeenModeRef.current = latest.pipeline_mode;
          setProject(latest);
          if (latest.status === 'needs_more' || latest.status === 'complete') {
            clearInterval(pollInterval);
            const exports = [
              { label: 'Colored Point Cloud (.ply)', key: `${pid}/exports/output.ply`,  mime_type: 'application/octet-stream' },
              { label: 'Mesh (.obj)',                key: `${pid}/exports/output.obj`,  mime_type: 'model/obj' },
              { label: 'LiDAR Exchange (.las)',       key: `${pid}/exports/output.las`,  mime_type: 'application/octet-stream' },
            ];
            setJobResult({
              exports,
              coverage_score: latest.coverage_score ?? undefined,
              suggestions:    latest.suggestions ?? [],
            } as any);
            setPipelineStatus(latest.status === 'needs_more' ? 'needs_more' : 'complete');
          } else if (latest.status === 'failed') {
            clearInterval(pollInterval);
            clearStallTimer();
            setPipelineStatus('error');
          }
        } catch { /* keep polling */ }
      }, 10000);
    } catch (error) {
      console.error('Upload or launch failed:', error);
      setErrorMessage('Upload or pipeline launch failed');
      setPipelineStatus('error');
    } finally {
      setUploading(false);
    }
  }

  const currentStageIndex = pipelineProgress
    ? STAGES.findIndex(s => s.id === pipelineProgress.stage)
    : -1;

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin"><Clock className="h-8 w-8 text-slate-400" /></div>
          <p className="text-sm text-slate-600">Loading project...</p>
        </div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="flex items-center justify-center py-16">
        <Card padding="lg">
          <div className="flex flex-col items-center gap-4">
            <AlertCircle className="h-12 w-12 text-red-600" />
            <h3 className="text-lg font-semibold text-slate-900">Project not found</h3>
            <Link href="/" className="text-brand-600 hover:text-brand-700 text-sm font-medium">
              ← Back to projects
            </Link>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Breadcrumb + quick actions */}
      <div className="flex items-center justify-between">
        <Link href="/" className="inline-flex items-center gap-2 text-sm text-slate-600 hover:text-slate-900 transition-colors">
          <ArrowLeft className="h-4 w-4" />
          Back to projects
        </Link>
        <a
          href={`${typeof window !== 'undefined' ? `${window.location.protocol}//${window.location.hostname}:8000` : ''}/api/aruco-sheet.pdf?size=0.15`}
          target="_blank"
          rel="noopener noreferrer"
          title="Download printable ArUco marker sheet — place markers in your scene before scanning for metric scale"
          className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-800 transition-colors border border-slate-200 rounded px-2.5 py-1 hover:bg-slate-50"
        >
          🖨️ Print ArUco Markers
        </a>
      </div>

      {/* Project header */}
      <div>
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            {isEditingName ? (
              <div className="flex items-center gap-2">
                <input
                  autoFocus
                  value={editingNameValue}
                  onChange={e => setEditingNameValue(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter') handleSaveName();
                    if (e.key === 'Escape') setIsEditingName(false);
                  }}
                  className="text-3xl font-bold text-slate-900 border-b-2 border-brand-400 bg-transparent outline-none flex-1 min-w-0"
                />
                <button
                  onClick={handleSaveName}
                  disabled={savingName}
                  className="rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50 transition-colors"
                >
                  {savingName ? 'Saving…' : 'Save'}
                </button>
                <button
                  onClick={() => setIsEditingName(false)}
                  className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50 transition-colors"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2 group">
                <h1 className="text-3xl font-bold text-slate-900 truncate">{project.name}</h1>
                <button
                  onClick={() => { setEditingNameValue(project.name); setIsEditingName(true); }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-slate-100 text-slate-400 hover:text-slate-600 shrink-0"
                  title="Rename project"
                >
                  <Pencil className="h-4 w-4" />
                </button>
              </div>
            )}
            <p className="mt-1 text-slate-600">{project.description || 'No description'}</p>
          </div>
          <Badge status={project.status} />
        </div>
      </div>

      {/* Project setup info */}
      <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
        <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs text-slate-600">
          {project.created_at && (
            <div className="flex items-center gap-1.5">
              <Clock className="h-3.5 w-3.5 text-slate-400" />
              <span>Created {new Date(project.created_at).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}</span>
            </div>
          )}
          {project.scene_type && (
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Scene</span>
              <span className="font-medium capitalize">{project.scene_type.replace(/_/g, ' ')}</span>
            </div>
          )}
          {project.marker_type && (
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Marker</span>
              <span className="font-medium">{project.marker_type === 'grid' ? '3×3 grid' : 'ArUco'}</span>
            </div>
          )}
          {project.pipeline_mode && (
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Mode</span>
              <span className="font-medium capitalize">{project.pipeline_mode.replace(/_/g, ' ')}</span>
            </div>
          )}
          {project.confirmed_scale_factor && (
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Scale</span>
              <span className="font-medium">{project.confirmed_scale_factor.toPrecision(4)} m/unit</span>
            </div>
          )}
        </div>
      </div>

      {/* Scale info when available */}
      {project.confirmed_scale_factor && (
        <div className="rounded-lg bg-green-50 border border-green-200 px-4 py-3 flex items-center gap-3">
          <CheckCircle className="h-5 w-5 text-green-600 shrink-0" />
          <p className="text-sm text-green-800">
            <strong>Metric scale applied:</strong>{' '}
            {project.confirmed_scale_factor.toPrecision(4)} m/unit
            {project.confirmed_scale_source ? ` (source: ${project.confirmed_scale_source})` : ''}
            {project.gravity_up_world ? ' · gravity aligned' : ''}
          </p>
        </div>
      )}

      {/* Error messages */}
      {errorMessage && (
        <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm text-red-700">{errorMessage}</p>
          </div>
          <button onClick={() => setErrorMessage(null)} className="text-red-600 hover:text-red-700">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* WebSocket status */}
      {pipelineStatus === 'running' && !wsConnected && (
        <div className="rounded-lg bg-amber-50 border border-amber-200 px-4 py-3 text-sm text-amber-700">
          ⚠ WebSocket disconnected, falling back to polling...
        </div>
      )}

      {/* Stall warning */}
      {pipelineStatus === 'running' && stalled && (
        <div className="rounded-lg bg-orange-50 border border-orange-300 px-4 py-3 flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-orange-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm font-medium text-orange-800">No progress for a while</p>
            <p className="text-xs text-orange-700 mt-0.5">
              The pipeline may have crashed. Check worker logs or cancel and retry.
            </p>
          </div>
          <button
            onClick={handleCancel}
            disabled={cancelling}
            className="flex-shrink-0 rounded-lg bg-orange-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-orange-700 disabled:opacity-50 transition-colors"
          >
            {cancelling ? 'Cancelling…' : 'Cancel'}
          </button>
        </div>
      )}

      {/* Upload widget (idle state) */}
      {pipelineStatus === 'idle' ? (
        <Card padding="lg">
          <h2 className="mb-6 text-xl font-semibold text-slate-900">Scan Setup</h2>

          {/* Step 1 — Calibration photo */}
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-2">
              <span className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold ${calibResult ? 'bg-green-500 text-white' : 'bg-slate-200 text-slate-600'}`}>
                {calibResult ? '✓' : '1'}
              </span>
              <h3 className="text-sm font-semibold text-slate-800">Calibration Photo <span className="font-normal text-slate-400">(optional but recommended)</span></h3>
            </div>
            <p className="mb-3 text-xs text-slate-500 ml-8">
              Take a still photo with the same phone/camera you'll use for the video.
              We read the EXIF focal length to improve 3D accuracy.
            </p>
            <div className="ml-8">
              {calibResult ? (
                <div className="flex items-center justify-between rounded-lg bg-green-50 border border-green-200 px-3 py-2">
                  <div className="text-xs text-green-800">
                    <span className="font-semibold">{calibFile?.name}</span>
                    {calibResult.focal_length_px && (
                      <span className="ml-2 text-green-600">· fl={Math.round(calibResult.focal_length_px)}px</span>
                    )}
                    {(calibResult.make || calibResult.model) && (
                      <span className="ml-2 text-green-600">· {[calibResult.make, calibResult.model].filter(Boolean).join(' ')}</span>
                    )}
                  </div>
                  <button
                    onClick={() => { setCalibFile(null); setCalibResult(null); setCalibError(null); }}
                    className="ml-3 text-xs text-green-600 hover:text-green-800"
                  >
                    change
                  </button>
                </div>
              ) : (
                <div>
                  <button
                    onClick={() => calibInputRef.current?.click()}
                    disabled={calibUploading}
                    className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 hover:bg-slate-50 transition disabled:opacity-50"
                  >
                    <Camera className="h-4 w-4" />
                    {calibUploading ? 'Reading EXIF…' : 'Upload a still photo'}
                  </button>
                  <input ref={calibInputRef} type="file" accept="image/*" className="hidden" onChange={handleCalibPhoto} />
                  {calibError && <p className="mt-2 text-xs text-amber-600">{calibError}</p>}
                </div>
              )}
            </div>
          </div>

          {/* Step 2 — Shooting guide */}
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-3">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-200 text-xs font-bold text-slate-600">2</span>
              <h3 className="text-sm font-semibold text-slate-800">Shooting Guide</h3>
            </div>
            <div className="ml-8">
              <ShootingGuide sceneType={project?.scene_type} />
            </div>
          </div>

          {/* Scan source selector */}
          <div className="mb-5">
            <h3 className="text-sm font-semibold text-slate-800 mb-2">Scan source</h3>
            <div className="flex gap-2">
              {([
                { id: 'video',     label: '🎥 Video / Photos', desc: 'Full SfM/MVS reconstruction from footage' },
                { id: 'lidar_ply', label: '📡 LiDAR Point Cloud', desc: 'Upload an already-built .ply — skips straight to refine/export' },
              ] as const).map(opt => (
                <button
                  key={opt.id}
                  onClick={() => { setScanSource(opt.id); setPendingFiles([]); }}
                  className={`flex-1 rounded-lg border px-3 py-2 text-left transition ${
                    scanSource === opt.id
                      ? 'border-brand-400 bg-brand-50 text-brand-800'
                      : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span className={`h-3 w-3 rounded-full border-2 ${scanSource === opt.id ? 'border-brand-500 bg-brand-500' : 'border-slate-300'}`} />
                    <span className="text-sm font-medium">{opt.label}</span>
                  </div>
                  <p className="text-xs ml-4.5 text-slate-500 pl-5">{opt.desc}</p>
                </button>
              ))}
            </div>
            {scanSource === 'lidar_ply' && (
              <div className="mt-3 ml-1 rounded-lg bg-amber-50 border border-amber-200 px-3 py-2 text-xs text-amber-800 space-y-2">
                <p>
                  No ArUco/grid markers needed — scale comes from the scanner. The floor is found
                  from the cloud itself, then refinement and export run.
                </p>
                <label className="flex items-center gap-2">
                  <span className="font-medium">Scale factor</span>
                  <input
                    type="number"
                    step="any"
                    min={0}
                    value={lidarScaleFactor}
                    onChange={e => setLidarScaleFactor(parseFloat(e.target.value) || 1.0)}
                    className="w-24 rounded border border-amber-300 px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-amber-400"
                  />
                  <span className="text-amber-700">Leave at 1.0 if your scanner already exports metres (e.g. mm → 0.001)</span>
                </label>
              </div>
            )}
          </div>

          {/* Ground-truth dimensions (optional) — sanity-checks the derived scale */}
          <div className="mb-5">
            <button
              type="button"
              onClick={() => setShowGroundTruth(v => !v)}
              className="text-sm font-semibold text-slate-800 hover:text-brand-700 transition-colors"
            >
              {showGroundTruth ? '▾' : '▸'} Ground truth dimensions <span className="font-normal text-slate-400">(optional)</span>
            </button>
            {showGroundTruth && (
              <div className="mt-2">
                <p className="text-xs text-slate-500 mb-2">
                  Tape-measure the room yourself and enter what you get. The results will show
                  predicted-vs-truth error % for each field you fill in. Leave any field blank to skip it.
                </p>
                <div className="flex gap-3">
                  {([
                    { label: 'Length (m)',  value: gtLength,  set: setGtLength,  ph: 'e.g. 4.20' },
                    { label: 'Breadth (m)', value: gtBreadth, set: setGtBreadth, ph: 'e.g. 3.10' },
                    { label: 'Height (m)',  value: gtHeight,  set: setGtHeight,  ph: 'e.g. 2.65' },
                  ]).map(f => (
                    <label key={f.label} className="flex-1 text-xs text-slate-600">
                      {f.label}
                      <input type="number" step="any" min={0} value={f.value}
                        onChange={e => f.set(e.target.value)} placeholder={f.ph}
                        className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-1.5 text-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500" />
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Mode selector (video only) */}
          {scanSource === 'video' && (
          <div className="mb-5">
            <h3 className="text-sm font-semibold text-slate-800 mb-2">Processing mode</h3>
            <div className="flex gap-2">
              {([
                { id: 'standard', label: 'Standard', desc: 'Full quality in one pass (~2h)' },
                { id: 'scout',    label: 'Scout + Full', desc: 'Quick calibration then full run (~15min + 2h)' },
              ] as const).map(opt => (
                <button
                  key={opt.id}
                  onClick={() => setPipelineMode(opt.id)}
                  className={`flex-1 rounded-lg border px-3 py-2 text-left transition ${
                    pipelineMode === opt.id
                      ? 'border-brand-400 bg-brand-50 text-brand-800'
                      : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span className={`h-3 w-3 rounded-full border-2 ${pipelineMode === opt.id ? 'border-brand-500 bg-brand-500' : 'border-slate-300'}`} />
                    <span className="text-sm font-medium">{opt.label}</span>
                  </div>
                  <p className="text-xs ml-4.5 text-slate-500 pl-5">{opt.desc}</p>
                </button>
              ))}
            </div>
          </div>
          )}

          <div>
            <div className="flex items-center gap-2 mb-3">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-200 text-xs font-bold text-slate-600">3</span>
              <h3 className="text-sm font-semibold text-slate-800">Upload Media</h3>
            </div>
            <div className="ml-8 space-y-3">
              {/* Drop zone */}
              <div
                className="rounded-lg border-2 border-dashed border-slate-300 p-8 text-center hover:border-brand-400 hover:bg-brand-50 transition cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
                onDragOver={e => e.preventDefault()}
                onDrop={e => {
                  e.preventDefault();
                  const files = Array.from(e.dataTransfer.files);
                  if (files.length) {
                    setPendingFiles(prev => {
                      const existing = new Set(prev.map(f => f.name));
                      return [...prev, ...files.filter(f => !existing.has(f.name))];
                    });
                  }
                }}
              >
                <Upload className="mx-auto mb-2 h-8 w-8 text-slate-400" />
                <p className="text-sm font-medium text-slate-700">Click or drag files here</p>
                <p className="text-xs text-slate-400 mt-1">
                  {scanSource === 'lidar_ply'
                    ? 'A single LiDAR point cloud (.ply)'
                    : 'Videos (MP4, MOV) and photos (JPEG, PNG, HEIC) — add as many as you like'}
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={scanSource === 'lidar_ply' ? '.ply' : 'video/*,image/*,.heic,.heif'}
                  multiple={scanSource !== 'lidar_ply'}
                  onChange={handleFileSelect}
                  disabled={uploading}
                  className="hidden"
                />
              </div>

              {/* File list */}
              {pendingFiles.length > 0 && (
                <div className="space-y-1.5">
                  {pendingFiles.map((f, i) => {
                    const isVideo = f.type.startsWith('video/') || /\.(mp4|mov|avi|mkv|webm)$/i.test(f.name);
                    const sizeStr = f.size > 1e9
                      ? `${(f.size / 1e9).toFixed(1)} GB`
                      : `${(f.size / 1e6).toFixed(1)} MB`;
                    return (
                      <div key={i} className="flex items-center gap-2 rounded-lg bg-white border border-slate-200 px-3 py-2">
                        <span className="text-base">{/\.ply$/i.test(f.name) ? '📡' : isVideo ? '🎬' : '📷'}</span>
                        <span className="flex-1 text-xs text-slate-700 truncate">{f.name}</span>
                        <span className="text-xs text-slate-400 shrink-0">{sizeStr}</span>
                        <button
                          onClick={() => setPendingFiles(prev => prev.filter((_, idx) => idx !== i))}
                          className="text-slate-300 hover:text-red-400 transition-colors ml-1"
                          title="Remove"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Upload progress */}
              {uploading && (
                <div className="space-y-1.5">
                  <ProgressBar value={uploadProgress} className="h-2" />
                  <p className="text-xs text-slate-500 text-center">{uploadProgress}% uploaded</p>
                </div>
              )}

              {/* Launch button */}
              {pendingFiles.length > 0 && !uploading && (
                <button
                  onClick={handleLaunch}
                  className="w-full rounded-lg bg-brand-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-brand-700 transition-colors"
                >
                  Start Reconstruction → {pendingFiles.length} file{pendingFiles.length !== 1 ? 's' : ''}
                </button>
              )}
            </div>
          </div>
        </Card>
      ) : (
        <div className="space-y-8">
          {/* Upload progress */}
          {uploading && (
            <Card padding="lg">
              <h3 className="mb-4 text-sm font-semibold text-slate-900">Uploading video...</h3>
              <ProgressBar value={uploadProgress} className="h-2" />
              <p className="mt-2 text-sm text-slate-600">{uploadProgress}% uploaded</p>
            </Card>
          )}

          {/* Pipeline progress stepper — collapsible */}
          <Card padding="lg">
            <div
              className="flex items-center justify-between cursor-pointer select-none"
              onClick={() => setPipelineOpen(o => !o)}
            >
              <div className="flex items-center gap-3">
                <h3 className="text-lg font-semibold text-slate-900">Processing Pipeline</h3>
                {pipelineStatus === 'running' && pipelineProgress && !isNaN(pipelineProgress.progress) && (
                  <span className="text-sm text-slate-500">{Math.round(pipelineProgress.progress * 100)}%</span>
                )}
                {(pipelineStatus === 'complete' || pipelineStatus === 'needs_more') && (
                  <span className="text-xs text-green-600 font-medium">Complete</span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {pipelineStatus === 'running' && (
                  <button
                    onClick={e => { e.stopPropagation(); handleCancel(); }}
                    disabled={cancelling}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-red-300 bg-white px-3 py-1.5 text-xs font-semibold text-red-600 hover:bg-red-50 disabled:opacity-50 transition-colors"
                  >
                    <X className="h-3.5 w-3.5" />
                    {cancelling ? 'Cancelling…' : 'Cancel'}
                  </button>
                )}
                {pipelineOpen ? <ChevronUp className="h-4 w-4 text-slate-400" /> : <ChevronDown className="h-4 w-4 text-slate-400" />}
              </div>
            </div>

            {pipelineOpen && (
              <div className="mt-6">
                {/* GPU stats bar — shown once first heartbeat arrives */}
                {pipelineStatus === 'running' && gpuStats && (
                  <div className="mb-4 flex items-center gap-3 rounded-lg bg-slate-900 px-3 py-2">
                    <span className="text-xs text-slate-400 shrink-0">GPU</span>
                    {gpuStats.available ? (
                      <>
                        <div className="flex-1 h-1.5 rounded-full bg-slate-700 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-1000 ${
                              gpuStats.util_pct > 80 ? 'bg-green-400' :
                              gpuStats.util_pct > 30 ? 'bg-yellow-400' : 'bg-slate-500'
                            }`}
                            style={{ width: `${gpuStats.util_pct}%` }}
                          />
                        </div>
                        <span className="text-xs text-slate-300 font-mono shrink-0">{gpuStats.util_pct}%</span>
                        <span className="text-xs text-slate-500 shrink-0">
                          {(gpuStats.vram_used_mb / 1024).toFixed(1)}/{(gpuStats.vram_total_mb / 1024).toFixed(1)} GB VRAM
                        </span>
                      </>
                    ) : (
                      <>
                        <div className="flex-1 h-1.5 rounded-full bg-slate-700" />
                        <span className="text-xs text-slate-500 shrink-0">idle (CPU stage)</span>
                      </>
                    )}
                  </div>
                )}

                {/* Overall progress bar */}
                {pipelineProgress && !isNaN(pipelineProgress.progress) && (
                  <div className="mb-6">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-slate-900">Overall Progress</span>
                      <span className="text-sm text-slate-600">{Math.round(pipelineProgress.progress * 100)}%</span>
                    </div>
                    <ProgressBar value={pipelineProgress.progress * 100} className="h-3" />
                  </div>
                )}

                {/* Queued banner — no WS events received yet */}
                {pipelineStatus === 'running' && !pipelineProgress && stageLog.size === 0 && (
                  <div className="mb-4 flex items-center gap-2 rounded-lg bg-amber-50 border border-amber-200 px-3 py-2.5">
                    <div className="h-2 w-2 rounded-full bg-amber-400 animate-pulse shrink-0" />
                    <span className="text-sm text-amber-800">Queued — waiting for GPU worker to pick up this job</span>
                  </div>
                )}

                <PipelineLog
                  stages={STAGES.map(s => {
                    const logged = stageLog.get(s.id);
                    const idx = STAGES.findIndex(x => x.id === s.id);
                    const pipelineDone = pipelineStatus === 'complete' || pipelineStatus === 'needs_more';
                    const pipelineRunning = pipelineStatus === 'running';
                    // A stage is done if: pipeline finished, OR stageLog marks it done,
                    // OR the live currentStageIndex has passed it
                    const doneInLog = logged?.status === 'done';
                    const isCompleted = pipelineDone || doneInLog
                      || currentStageIndex > idx
                      || (currentStageIndex === idx && pipelineProgress?.progress === 1.0);
                    const isActive = !pipelineDone && !doneInLog && currentStageIndex === idx && !isCompleted;
                    const status: StageDetail['status'] = isCompleted ? 'done' : isActive ? 'running' : 'pending';
                    // "Queued" only when we know our position in the pipeline (either
                    // WS events arrived or stageLog has data), not when we're blind
                    const knowPosition = currentStageIndex >= 0 || stageLog.size > 0;
                    const isPendingQueued = pipelineRunning && status === 'pending' && knowPosition;
                    return {
                      id: s.id,
                      label: s.label,
                      phase: 1,
                      status,
                      lastMessage: isActive
                        ? (pipelineProgress?.message ?? '')
                        : isCompleted
                          ? logged?.lastMessage
                          : isPendingQueued ? 'Queued' : undefined,
                      progress: isActive ? pipelineProgress?.progress : undefined,
                      detail: logged?.detail,
                      elapsedSeconds: logged?.elapsedSeconds,
                    } satisfies StageDetail;
                  })}
                  currentStageId={pipelineProgress?.stage}
                />
              </div>
            )}
          </Card>

          {/* Footage review — collapsible, loads on open */}
          <Card padding="lg">
            <div
              className="flex items-center justify-between cursor-pointer select-none"
              onClick={async () => {
                const opening = !footageOpen;
                setFootageOpen(opening);
                if (opening && uploads.length === 0) await loadUploads();
              }}
            >
              <div className="flex items-center gap-2">
                <Film className="h-4 w-4 text-slate-500" />
                <h3 className="text-base font-semibold text-slate-900">Footage</h3>
                {uploads.length > 0 && (
                  <span className="text-xs text-slate-400">{uploads.length} file{uploads.length !== 1 ? 's' : ''}</span>
                )}
              </div>
              {footageOpen ? <ChevronUp className="h-4 w-4 text-slate-400" /> : <ChevronDown className="h-4 w-4 text-slate-400" />}
            </div>

            {footageOpen && (
              <div className="mt-4">
                {uploadsLoading ? (
                  <p className="text-sm text-slate-500">Loading…</p>
                ) : uploads.length === 0 ? (
                  <p className="text-sm text-slate-400">No uploaded files found.</p>
                ) : (
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    {uploads.map((u, i) => {
                      const isVideo = u.mime_type.startsWith('video/') || /\.(mp4|mov|avi|mkv|webm)$/i.test(u.filename);
                      const url = isVideo
                        ? `${API_BASE}/preview/video/${u.storage_key}`
                        : storageUrl(u.storage_key);
                      const sizeMB = (u.size_bytes / 1e6).toFixed(1);
                      return (
                        <div
                          key={u.id}
                          className="group relative rounded-lg border border-slate-200 overflow-hidden bg-slate-100 cursor-pointer hover:border-brand-400 transition-colors"
                          onClick={() => { setCarouselIdx(i); setCarouselOpen(true); }}
                        >
                          {isVideo ? (
                            <video
                              src={url}
                              className="w-full h-28 object-cover"
                              muted
                              preload="metadata"
                            />
                          ) : (
                            <img
                              src={url}
                              alt={u.filename}
                              className="w-full h-28 object-cover"
                            />
                          )}
                          <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 bg-black/30 transition-opacity">
                            <Play className="h-8 w-8 text-white drop-shadow" />
                          </div>
                          <div className="px-2 py-1.5 bg-white">
                            <p className="text-xs text-slate-700 truncate">{u.filename}</p>
                            <p className="text-xs text-slate-400">{sizeMB} MB</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </Card>

          {/* Footage carousel modal */}
          {carouselOpen && uploads.length > 0 && (
            <div
              className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
              onClick={() => setCarouselOpen(false)}
            >
              <div
                className="relative w-full max-w-4xl mx-4 bg-black rounded-xl overflow-hidden"
                onClick={e => e.stopPropagation()}
              >
                {/* Close */}
                <button
                  className="absolute top-3 right-3 z-10 rounded-full bg-black/50 p-1.5 text-white hover:bg-black/70 transition-colors"
                  onClick={() => setCarouselOpen(false)}
                >
                  <X className="h-5 w-5" />
                </button>

                {/* Media */}
                {(() => {
                  const u = uploads[carouselIdx];
                  const isVideo = u.mime_type.startsWith('video/') || /\.(mp4|mov|avi|mkv|webm)$/i.test(u.filename);
                  const url = isVideo
                    ? `${API_BASE}/preview/video/${u.storage_key}`
                    : storageUrl(u.storage_key);
                  return isVideo ? (
                    <video
                      key={url}
                      src={url}
                      controls
                      autoPlay
                      className="w-full max-h-[70vh] object-contain bg-black"
                    />
                  ) : (
                    <img
                      src={url}
                      alt={u.filename}
                      className="w-full max-h-[70vh] object-contain bg-black"
                    />
                  );
                })()}

                {/* Nav + info */}
                <div className="flex items-center justify-between px-4 py-3 bg-black/90">
                  <button
                    disabled={carouselIdx === 0}
                    onClick={() => setCarouselIdx(i => i - 1)}
                    className="rounded-lg border border-white/20 px-3 py-1.5 text-sm text-white disabled:opacity-30 hover:bg-white/10 transition-colors"
                  >
                    ← Prev
                  </button>
                  <div className="text-center">
                    <p className="text-sm text-white font-medium">{uploads[carouselIdx].filename}</p>
                    <p className="text-xs text-slate-400">{carouselIdx + 1} / {uploads.length} · {(uploads[carouselIdx].size_bytes / 1e6).toFixed(1)} MB</p>
                  </div>
                  <button
                    disabled={carouselIdx === uploads.length - 1}
                    onClick={() => setCarouselIdx(i => i + 1)}
                    className="rounded-lg border border-white/20 px-3 py-1.5 text-sm text-white disabled:opacity-30 hover:bg-white/10 transition-colors"
                  >
                    Next →
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Scout result card — shown when a scout+full run is in progress */}
          {project.pipeline_mode === 'scout' && project.scout_calibration && (
            <ScoutResultCard
              calibration={project.scout_calibration}
              fullRunStatus={pipelineStatus === 'running' ? 'running' : pipelineStatus === 'complete' ? 'done' : 'pending'}
            />
          )}

          {/* Activity log */}
          {(wsEvents.length > 0 || pipelineStatus === 'running') && (
            <ActivityLog events={wsEvents} isRunning={pipelineStatus === 'running'} />
          )}

          {/* Failed state */}
          {pipelineStatus === 'error' && (
            <Card padding="lg" className="border-red-200 bg-red-50">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-6 w-6 text-red-600 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <h3 className="text-base font-semibold text-red-900">Pipeline failed</h3>
                  <p className="text-sm text-red-700 mt-1">
                    The reconstruction pipeline stopped unexpectedly.
                    If no ArUco markers were found, print DICT_4X4_100 markers (≥15 cm) and place them in the scene.
                    Check the activity log above for the specific error.
                  </p>
                  <button
                    onClick={() => { setPipelineStatus('idle'); setPipelineProgress(null); }}
                    className="mt-3 rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 transition-colors"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            </Card>
          )}

          {/* Needs more coverage */}
          {pipelineStatus === 'needs_more' && (
            <NeedsMorePanel
              projectId={projectId}
              coverageScore={jobResult?.coverage_score}
              suggestions={jobResult?.suggestions}
              onSupplementalLaunched={(taskId) => {
                setTaskId(taskId);
                setPipelineStatus('running');
                setJobResult(null);
              }}
            />
          )}

          {/* Complete */}
          {(pipelineStatus === 'complete' || pipelineStatus === 'needs_more') && (
            <Card padding="lg" className={pipelineStatus === 'needs_more' ? 'border-amber-200 bg-amber-50' : 'border-green-200 bg-green-50'}>
              <div className="mb-4 flex items-center gap-3">
                <CheckCircle className={`h-6 w-6 ${pipelineStatus === 'needs_more' ? 'text-amber-600' : 'text-green-600'}`} />
                <h3 className={`text-lg font-semibold ${pipelineStatus === 'needs_more' ? 'text-amber-900' : 'text-green-900'}`}>
                  {pipelineStatus === 'needs_more' ? 'Reconstruction Complete — more footage recommended' : 'Reconstruction Complete!'}
                </h3>
              </div>
              <p className={`mb-6 text-sm ${pipelineStatus === 'needs_more' ? 'text-amber-800' : 'text-green-800'}`}>
                {pipelineStatus === 'needs_more'
                  ? 'Your 3D model is ready but coverage is below 50%. Upload supplemental footage targeting the flagged areas to improve it.'
                  : 'Your 3D model has been generated and exported. View and download it below.'}
              </p>

              {/* Quality score card */}
              <div className="mb-6">
                <QualityScoreCard
                  registeredImages={
                    jobResult?.registered_images
                    ?? project?.pipeline_results?.sfm?.registered_images
                  }
                  totalFrames={
                    (jobResult?.frame_keys ?? jobResult?.image_keys ?? []).length
                    || project?.pipeline_results?.sfm?.total_images
                    || project?.pipeline_results?.extract_frames?.frame_count
                  }
                  reprojectionError={
                    jobResult?.mean_reprojection_error
                    ?? project?.pipeline_results?.sfm?.mean_reprojection_error
                  }
                  densePoints={
                    jobResult?.dense_point_count
                    ?? project?.pipeline_results?.mvs?.dense_point_count
                  }
                  pointDensity={jobResult?.point_density}
                  coverageScore={jobResult?.coverage_score ?? project?.pipeline_results?.coverage?.coverage_score}
                  scaleConfidence={scaleConfidenceFromDiag(jobResult?.scale_diagnostics)}
                  scaleFactor={project?.confirmed_scale_factor}
                />
              </div>

              {/* Real-world dimensions */}
              <div className="mb-6">
                <DimensionsCard
                  dimensions={project?.dimensions ?? project?.pipeline_results?.export?.dimensions}
                />
              </div>

              <ResultsViewer
                projectId={projectId}
                exports={jobResult?.exports}
                suggestions={jobResult?.suggestions ?? []}
                scaleFactor={project?.confirmed_scale_factor ?? 1}
                splatKey={project?.splat_key}
                gsMeshKey={project?.mesh_key}
                lingbotCloudKey={project?.lingbot_cloud_key}
                lingbotMeshKey={project?.lingbot_mesh_key}
                metricanythingCloudKey={project?.metricanything_cloud_key}
                metricanythingMeshKey={project?.metricanything_mesh_key}
                hvacPlacement={project?.hvac_placement ?? project?.pipeline_results?.export?.hvac_placement}
                hvacSegmentation={project?.hvac_segmentation}
                objectLabels={[]}
                apiBase={API_BASE}
              />
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
