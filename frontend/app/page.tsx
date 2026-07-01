'use client';

import { useEffect, useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Plus, Clock, CheckCircle, AlertCircle, Video, Trash2, Square, CheckSquare, Search, Filter, X, Camera } from 'lucide-react';
import { listProjects, createProject, bulkDeleteProjects, deleteFailedProjects, storageUrl } from '@/lib/api';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';

interface Project {
  id: string;
  name: string;
  description: string;
  status: 'created' | 'processing' | 'needs_more' | 'complete' | 'failed';
  scene_type?: 'indoor_room' | 'outdoor' | 'object' | null;
  coverage_score?: number | null;
  confirmed_scale_factor?: number | null;
  pipeline_results?: Record<string, any> | null;
  created_at?: string | null;
}

function computeProjectScore(project: Project): number | null {
  const isDone = project.status === 'complete' || project.status === 'needs_more';
  if (!isDone) return null;
  const pr = project.pipeline_results;
  const reg  = pr?.sfm?.registered_images;
  const tot  = pr?.sfm?.total_images ?? pr?.extract_frames?.frame_count;
  const regScore    = reg != null && tot ? Math.min(1, (reg / tot) / 0.85) : null;
  const reproj      = pr?.sfm?.mean_reprojection_error;
  const reprojScore = reproj != null ? Math.max(0, 1 - (reproj - 0.5) / 2.5) : null;
  const covScore    = project.coverage_score ?? null;
  const scaleScore  = project.confirmed_scale_factor != null ? 0.8 : 0;
  const weights = [0.25, 0.25, 0.20, 0.10];
  const scores  = [regScore, reprojScore, covScore, scaleScore];
  const wSum = weights.reduce((s, w, i) => s + (scores[i] !== null ? w : 0), 0);
  if (wSum === 0) return covScore != null ? Math.round(covScore * 100) : null;
  return Math.round(scores.reduce<number>((s, sc, i) => s + (sc ?? 0) * weights[i], 0) / wSum * 100);
}

type StatusFilter = 'all' | 'complete' | 'processing' | 'needs_more' | 'failed' | 'created';

function thumbnailUrl(project: Project): string | null {
  if (project.status === 'complete' || project.status === 'needs_more') {
    return storageUrl(`${project.id}/frames/frame_000010.jpg`);
  }
  if (project.status === 'processing') {
    return storageUrl(`${project.id}/frames/frame_000000.jpg`);
  }
  return null;
}

function formatRelative(iso: string | null | undefined): string {
  if (!iso) return '';
  const then = new Date(iso);
  const now = new Date();
  const diff = Math.floor((now.getTime() - then.getTime()) / 1000);
  if (diff < 60)   return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

const STATUS_LABELS: Record<string, string> = {
  all:        'All',
  complete:   'Complete',
  processing: 'Processing',
  needs_more: 'Needs More Footage',
  failed:     'Failed',
  created:    'Created',
};

export default function ProjectsPage() {
  const router = useRouter();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNewProject, setShowNewProject] = useState(false);
  const [newName, setNewName]           = useState('');
  const [newDesc, setNewDesc]           = useState('');
  const [newSceneType, setNewSceneType] = useState<'indoor_room' | 'outdoor' | 'object'>('indoor_room');
  const [newLingbot, setNewLingbot]     = useState(false);
  const [creating, setCreating]         = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');

  useEffect(() => { loadProjects(); }, []);

  async function loadProjects() {
    try {
      setLoading(true);
      setError(null);
      const data = await listProjects();
      // Sort newest first
      const sorted = (Array.isArray(data) ? data : []).sort((a: Project, b: Project) => {
        if (!a.created_at) return 1;
        if (!b.created_at) return -1;
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
      setProjects(sorted);
      setSelected(new Set());
    } catch (err) {
      setError('Failed to load projects');
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateProject(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim()) return;
    try {
      setCreating(true);
      const project = await createProject({ name: newName, description: newDesc, scene_type: newSceneType, lingbot_enabled: newLingbot });
      router.push(`/projects/${project.id}`);
    } catch { setError('Failed to create project'); }
    finally { setCreating(false); }
  }

  function toggleSelect(id: string, e: React.MouseEvent) {
    e.preventDefault(); e.stopPropagation();
    setSelected(prev => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });
  }

  const filtered = useMemo(() => projects.filter(p => {
    const matchStatus = statusFilter === 'all' || p.status === statusFilter;
    const q = search.toLowerCase();
    const matchSearch = !q || p.name.toLowerCase().includes(q) || p.description?.toLowerCase().includes(q);
    return matchStatus && matchSearch;
  }), [projects, search, statusFilter]);

  function toggleSelectAll() {
    if (selected.size === filtered.length && filtered.length > 0) {
      setSelected(new Set());
    } else {
      setSelected(new Set(filtered.map(p => p.id)));
    }
  }

  async function handleDeleteSelected() {
    if (!selected.size) return;
    if (!confirm(`Delete ${selected.size} project${selected.size > 1 ? 's' : ''}? This cannot be undone.`)) return;
    try { setDeleting(true); await bulkDeleteProjects([...selected]); await loadProjects(); }
    catch { setError('Failed to delete'); } finally { setDeleting(false); }
  }

  async function handleClearFailed() {
    const n = projects.filter(p => p.status === 'failed').length;
    if (!n || !confirm(`Delete all ${n} failed projects?`)) return;
    try { setDeleting(true); await deleteFailedProjects(); await loadProjects(); }
    catch { setError('Failed to clear'); } finally { setDeleting(false); }
  }

  const failedCount = projects.filter(p => p.status === 'failed').length;
  const allFilteredSelected = filtered.length > 0 && filtered.every(p => selected.has(p.id));
  const someSelected = selected.size > 0;

  const statusIcon = (status: string) => {
    switch (status) {
      case 'processing': return <Clock className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'needs_more': return <Camera className="h-4 w-4 text-orange-500" />;
      case 'complete':   return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':     return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:           return <Video className="h-4 w-4 text-slate-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Projects</h1>
          <p className="mt-1 text-sm text-slate-600">Create and manage your 3D photogrammetry projects</p>
        </div>
        <div className="flex items-center gap-3">
          <a
            href={`${typeof window !== 'undefined' ? `${window.location.protocol}//${window.location.hostname}:8000` : ''}/api/aruco-sheet.pdf?size=0.15`}
            target="_blank"
            rel="noopener noreferrer"
            title="Print ArUco marker sheet — place these in your scene before scanning"
            className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors"
          >
            🖨️ Print Markers
          </a>
          <button onClick={() => setShowNewProject(true)}
            className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 transition-colors">
            <Plus className="h-4 w-4" /> New Project
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 border border-red-200">{error}</div>
      )}

      {/* New Project Form */}
      {showNewProject && (
        <Card padding="lg">
          <h2 className="mb-6 text-xl font-semibold text-slate-900">Create New Project</h2>
          <form onSubmit={handleCreateProject} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Project Name</label>
              <input type="text" value={newName} onChange={e => setNewName(e.target.value)}
                placeholder="e.g., Building 3D Scan"
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Description (optional)</label>
              <input type="text" value={newDesc} onChange={e => setNewDesc(e.target.value)}
                placeholder="Describe your project..."
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Scene Type</label>
              <p className="text-xs text-slate-500 mb-2">
                Affects MVS parameters and coverage analysis.
              </p>
              <div className="flex gap-2">
                {(['indoor_room', 'outdoor', 'object'] as const).map(st => (
                  <button key={st} type="button"
                    onClick={() => setNewSceneType(st)}
                    className={`rounded-lg border px-3 py-2 text-sm font-medium transition-colors ${
                      newSceneType === st
                        ? 'border-brand-500 bg-brand-50 text-brand-700'
                        : 'border-slate-300 bg-white text-slate-600 hover:bg-slate-50'
                    }`}>
                    {st === 'indoor_room' ? '🏠 Indoor Room' : st === 'outdoor' ? '🌳 Outdoor' : '📦 Object'}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="flex items-start gap-2 cursor-pointer">
                <input type="checkbox" checked={newLingbot}
                  onChange={e => setNewLingbot(e.target.checked)}
                  className="mt-0.5 h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-500" />
                <span className="text-sm text-slate-700">
                  <span className="font-medium">✨ Densify with LingBot depth fusion</span>
                  <span className="block text-xs text-slate-500">
                    Adds a neural depth-fusion pass to fill gaps. Best for object/orbit and photo captures;
                    continuous video walkthroughs benefit less.
                  </span>
                </span>
              </label>
            </div>
            <div className="flex gap-3 pt-2">
              <button type="submit" disabled={creating || !newName.trim()}
                className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50 transition-colors">
                {creating ? 'Creating...' : 'Create'}
              </button>
              <button type="button" onClick={() => setShowNewProject(false)}
                className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-900 hover:bg-slate-50 transition-colors">
                Cancel
              </button>
            </div>
          </form>
        </Card>
      )}

      {/* Search + Filter bar */}
      {!loading && projects.length > 0 && (
        <div className="flex flex-col sm:flex-row gap-3">
          {/* Search */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search projects…"
              className="w-full rounded-lg border border-slate-300 bg-white pl-9 pr-9 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
            {search && (
              <button onClick={() => setSearch('')} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600">
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>

          {/* Status filter pills */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <Filter className="h-4 w-4 text-slate-400 shrink-0" />
            {(['all', 'complete', 'processing', 'needs_more', 'failed', 'created'] as StatusFilter[]).map(s => (
              <button key={s} onClick={() => setStatusFilter(s)}
                className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                  statusFilter === s
                    ? 'bg-brand-600 text-white'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}>
                {STATUS_LABELS[s]}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Bulk action toolbar */}
      {!loading && filtered.length > 0 && (
        <div className="flex items-center gap-3 flex-wrap text-sm">
          <button onClick={toggleSelectAll}
            className="inline-flex items-center gap-1.5 text-slate-600 hover:text-slate-900 transition-colors">
            {allFilteredSelected
              ? <CheckSquare className="h-4 w-4 text-brand-600" />
              : <Square className="h-4 w-4" />}
            {allFilteredSelected ? 'Deselect all' : 'Select all'}
          </button>

          {someSelected && (
            <>
              <span className="text-slate-300">|</span>
              <span className="text-slate-500">{selected.size} selected</span>
              <button onClick={handleDeleteSelected} disabled={deleting}
                className="inline-flex items-center gap-1.5 rounded-lg bg-red-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-red-700 disabled:opacity-50 transition-colors">
                <Trash2 className="h-3.5 w-3.5" />
                {deleting ? 'Deleting…' : `Delete ${selected.size}`}
              </button>
            </>
          )}

          {failedCount > 0 && !someSelected && (
            <>
              <span className="text-slate-300">|</span>
              <button onClick={handleClearFailed} disabled={deleting}
                className="inline-flex items-center gap-1.5 rounded-lg border border-red-300 bg-red-50 px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-100 disabled:opacity-50 transition-colors">
                <Trash2 className="h-3.5 w-3.5" />
                {deleting ? 'Clearing…' : `Clear ${failedCount} failed`}
              </button>
            </>
          )}

          <span className="ml-auto text-slate-400 text-xs">{filtered.length} project{filtered.length !== 1 ? 's' : ''}</span>
        </div>
      )}

      {/* Grid */}
      {loading ? (
        <div className="flex items-center justify-center py-16">
          <div className="flex flex-col items-center gap-4">
            <div className="animate-spin"><Clock className="h-8 w-8 text-slate-400" /></div>
            <p className="text-sm text-slate-600">Loading projects...</p>
          </div>
        </div>
      ) : filtered.length === 0 ? (
        projects.length === 0 ? (
          <Card padding="lg">
            <EmptyState icon={Video} title="No projects yet"
              description="Create your first project to get started with 3D photogrammetry"
              action={
                <button onClick={() => setShowNewProject(true)}
                  className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 transition-colors">
                  <Plus className="h-4 w-4" /> Create Project
                </button>
              } />
          </Card>
        ) : (
          <div className="flex items-center justify-center py-12 text-sm text-slate-500">
            No projects match your search.
          </div>
        )
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((project) => {
            const isSelected = selected.has(project.id);
            const thumb = thumbnailUrl(project);

            return (
              <div key={project.id} className="relative group">
                {/* Checkbox */}
                <button onClick={(e) => toggleSelect(project.id, e)}
                  className={`absolute top-2 left-2 z-10 rounded bg-white/80 backdrop-blur-sm p-0.5 shadow-sm transition-opacity ${
                    isSelected || someSelected ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                  }`}>
                  {isSelected
                    ? <CheckSquare className="h-4 w-4 text-brand-600" />
                    : <Square className="h-4 w-4 text-slate-400" />}
                </button>

                <Link href={`/projects/${project.id}`}>
                  <Card className={`h-full overflow-hidden p-0 transition-all hover:shadow-md hover:border-slate-300 ${
                    isSelected ? 'ring-2 ring-brand-500 border-brand-300' : ''
                  }`}>
                    {/* Thumbnail */}
                    <div className="relative h-36 bg-slate-900 overflow-hidden">
                      {thumb ? (
                        <img
                          src={thumb}
                          alt=""
                          className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                          onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                        />
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <Video className="h-10 w-10 text-slate-600" />
                        </div>
                      )}

                      {/* Score bar overlay for complete/needs_more projects */}
                      {(project.status === 'complete' || project.status === 'needs_more') && project.coverage_score != null && (
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 px-3 pb-2 pt-4">
                          <div className="flex items-center justify-between text-xs text-white/90">
                            <span>Coverage</span>
                            <span className="font-semibold">{(project.coverage_score * 100).toFixed(0)}%</span>
                          </div>
                          <div className="mt-1 h-1 rounded-full bg-white/20 overflow-hidden">
                            <div className="h-full rounded-full transition-all"
                              style={{
                                width: `${project.coverage_score * 100}%`,
                                backgroundColor: project.coverage_score >= 0.7 ? '#4ade80' : project.coverage_score >= 0.4 ? '#facc15' : '#f87171',
                              }} />
                          </div>
                        </div>
                      )}

                      {/* Score chip (complete/needs_more) or status badge */}
                      <div className="absolute top-2 right-2">
                        {(project.status === 'complete' || project.status === 'needs_more') ? (() => {
                          const score = computeProjectScore(project);
                          if (score === null) return <Badge status={project.status} />;
                          const color = score >= 75
                            ? 'bg-green-500 text-white'
                            : score >= 50
                              ? 'bg-amber-400 text-white'
                              : 'bg-red-500 text-white';
                          return (
                            <span className={`inline-flex items-baseline gap-0.5 rounded-full px-2 py-0.5 text-xs font-bold shadow ${color}`}>
                              {score}<span className="text-[10px] font-medium opacity-80">/100</span>
                            </span>
                          );
                        })() : <Badge status={project.status} />}
                      </div>
                    </div>

                    {/* Card body */}
                    <div className="p-3">
                      <div className="flex items-start gap-2 mb-1">
                        {statusIcon(project.status)}
                        <h3 className="text-sm font-semibold text-slate-900 truncate flex-1 group-hover:text-brand-600 transition-colors">
                          {project.name}
                        </h3>
                      </div>

                      {project.description && (
                        <p className="text-xs text-slate-500 line-clamp-1 mb-2">{project.description}</p>
                      )}

                      <div className="flex items-center justify-between gap-2 flex-wrap">
                        <div className="flex gap-1.5 flex-wrap">
                          {project.scene_type && (
                            <span className="text-xs text-indigo-600">
                              {project.scene_type === 'indoor_room' ? '🏠' : project.scene_type === 'outdoor' ? '🌳' : '📦'}{' '}
                              {project.scene_type.replace(/_/g, ' ')}
                            </span>
                          )}
                        </div>
                        {project.created_at && (
                          <span className="text-xs text-slate-400 shrink-0">{formatRelative(project.created_at)}</span>
                        )}
                      </div>
                    </div>
                  </Card>
                </Link>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
