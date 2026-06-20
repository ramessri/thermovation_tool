'use client';

import { Suspense, useState } from 'react';
import dynamic from 'next/dynamic';
import { Download, Loader } from 'lucide-react';
import { storageUrl } from '@/lib/api';

interface ExportItem {
  label: string;
  key: string;
  mime_type: string;
}

interface ModelViewerProps {
  projectId: string;
  exports?: Array<ExportItem>;
  coveragePlyKey?: string;
}

// Dynamic import to avoid hydration issues with Three.js
const ThreeViewer = dynamic(() => import('./ThreeViewer').then(m => ({ default: m.ThreeViewer })), {
  ssr: false,
  loading: () => (
    <div className="h-96 flex items-center justify-center bg-slate-900">
      <Loader className="h-6 w-6 text-slate-400 animate-spin" />
    </div>
  ),
});

const FORMAT_ORDER = ['ply', 'obj', 'las'] as const;
type FormatType = (typeof FORMAT_ORDER)[number];

function guessFormat(key: string): FormatType | null {
  if (key.endsWith('.ply')) return 'ply';
  if (key.endsWith('.obj')) return 'obj';
  if (key.endsWith('.las')) return 'las';
  return null;
}

const PLACEHOLDER_FORMATS: ExportItem[] = [
  { label: 'Colored Point Cloud (.ply)', key: '', mime_type: 'application/octet-stream' },
  { label: 'Mesh (.obj)', key: '', mime_type: 'model/obj' },
  { label: 'LiDAR Exchange (.las)', key: '', mime_type: 'application/octet-stream' },
];

export function ModelViewer({ projectId: _projectId, exports, coveragePlyKey }: ModelViewerProps) {
  const [activeFormat, setActiveFormat] = useState<FormatType>('ply');
  const [downloading, setDownloading] = useState<string | null>(null);

  const plyUrl = coveragePlyKey ? storageUrl(coveragePlyKey) : null;
  const exportList = exports && exports.length > 0 ? exports : null;
  const displayList = exportList ?? PLACEHOLDER_FORMATS;

  async function handleDownload(item: ExportItem) {
    if (!item.key) return;
    setDownloading(item.key);
    try {
      const response = await fetch(storageUrl(item.key));
      if (!response.ok) throw new Error('Download failed');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      // Derive a nice filename from the key
      a.download = item.key.split('/').pop() ?? 'model';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Download error:', err);
    } finally {
      setDownloading(null);
    }
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-slate-200 bg-slate-900 overflow-hidden shadow-sm">
        <Suspense fallback={
          <div className="h-96 flex items-center justify-center bg-slate-900">
            <Loader className="h-6 w-6 text-slate-400 animate-spin" />
          </div>
        }>
          <ThreeViewer plyUrl={plyUrl} />
        </Suspense>
      </div>

      {/* Format selector */}
      <div className="space-y-3">
        <label className="block text-sm font-medium text-slate-900">Export Format</label>
        <div className="flex gap-2">
          {FORMAT_ORDER.map((fmt) => (
            <button
              key={fmt}
              onClick={() => setActiveFormat(fmt)}
              className={`flex-1 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                activeFormat === fmt
                  ? 'bg-brand-600 text-white'
                  : 'border border-slate-300 bg-white text-slate-900 hover:bg-slate-50'
              }`}
            >
              {fmt.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Download buttons */}
      <div className="space-y-3">
        <label className="block text-sm font-medium text-slate-900">Downloads</label>
        <div className="grid gap-2 grid-cols-3">
          {displayList.map((item, idx) => {
            const fmt = guessFormat(item.key);
            const isActive = fmt === activeFormat;
            const isDisabled = !item.key || downloading !== null;
            return (
              <button
                key={idx}
                onClick={() => handleDownload(item)}
                disabled={isDisabled}
                className={`inline-flex items-center justify-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                  isActive
                    ? 'border-brand-600 ring-1 ring-brand-600 bg-white text-slate-900 hover:bg-slate-50'
                    : 'border-slate-300 bg-white text-slate-900 hover:bg-slate-50'
                }`}
              >
                <Download className="h-4 w-4" />
                {fmt ? fmt.toUpperCase() : `File ${idx + 1}`}
                {downloading === item.key && <span className="ml-1 animate-spin">⏳</span>}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
