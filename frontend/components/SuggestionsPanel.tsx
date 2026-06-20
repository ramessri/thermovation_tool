'use client';

/**
 * SuggestionsPanel — shows re-shoot suggestions with reference frame thumbnails.
 * Each suggestion includes the nearest video frame as a visual reference so the
 * user knows which part of the scene needs more coverage.
 */

import { useState, useEffect } from 'react';
import { Camera, Loader, MapPin, X, ZoomIn } from 'lucide-react';

interface SuggestionDetail {
  index: number;
  position: number[];
  direction: number[];
  message: string;
  nearest_frame_key?: string;
  nearest_frame_name?: string;
}

interface Props {
  projectId: string;
  apiBase: string;
  suggestions?: any[];
}

function storageUrl(apiBase: string, key: string) {
  return `${apiBase}/files/${key}`;
}

export function SuggestionsPanel({ projectId, apiBase, suggestions: rawSuggestions }: Props) {
  const [details, setDetails]     = useState<SuggestionDetail[]>([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);
  const [lightboxUrl, setLightboxUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!projectId) return;
    setLoading(true);
    fetch(`${apiBase}/api/projects/${projectId}/suggestions_detail`)
      .then(r => r.json())
      .then(data => { setDetails(data.suggestions || []); setError(data.error || null); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [projectId, apiBase]);

  if (!rawSuggestions?.length && !loading && !details.length) {
    return (
      <div className="flex items-center justify-center py-8 text-slate-500 text-sm">
        <Camera className="h-4 w-4 mr-2" />
        No re-shoot suggestions — coverage is good!
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8 text-slate-400 text-sm gap-2">
        <Loader className="h-4 w-4 animate-spin" />
        Loading suggestion details…
      </div>
    );
  }

  const displaySuggestions = details.length > 0
    ? details
    : (rawSuggestions || []).map((s, i) => ({ ...s, index: i }));

  return (
    <>
      {/* Lightbox */}
      {lightboxUrl && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
          onClick={() => setLightboxUrl(null)}
        >
          <div className="relative max-w-4xl w-full" onClick={e => e.stopPropagation()}>
            <button
              onClick={() => setLightboxUrl(null)}
              className="absolute -top-10 right-0 text-white/80 hover:text-white"
            >
              <X className="h-6 w-6" />
            </button>
            <img
              src={lightboxUrl}
              alt="Reference frame"
              className="w-full rounded-lg shadow-2xl"
            />
            <p className="text-white/60 text-xs text-center mt-2">
              Reference frame — this is the area that needs more coverage
            </p>
          </div>
        </div>
      )}

      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <MapPin className="h-4 w-4 text-orange-500" />
          <span className="text-sm font-semibold text-slate-900">
            {displaySuggestions.length} re-shoot area{displaySuggestions.length !== 1 ? 's' : ''} identified
          </span>
        </div>

        {error && (
          <p className="text-xs text-amber-700 bg-amber-50 px-3 py-2 rounded">{error}</p>
        )}

        <div className="space-y-3">
          {displaySuggestions.map((s: any, i: number) => {
            const frameKey = s.nearest_frame_key;
            return (
              <div key={i} className="rounded-lg border border-slate-200 overflow-hidden bg-white">
                <div className="flex gap-3 p-3">
                  {frameKey ? (
                    <button
                      className="shrink-0 w-32 h-24 rounded overflow-hidden bg-slate-900 relative group cursor-zoom-in"
                      onClick={() => setLightboxUrl(storageUrl(apiBase, frameKey))}
                      title="Click to enlarge"
                    >
                      <img
                        src={storageUrl(apiBase, frameKey)}
                        alt={`Re-shoot area ${i + 1}`}
                        className="w-full h-full object-cover"
                        onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                      />
                      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
                        <ZoomIn className="h-5 w-5 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                    </button>
                  ) : (
                    <div className="shrink-0 w-32 h-24 rounded bg-slate-100 flex items-center justify-center">
                      <Camera className="h-5 w-5 text-slate-400" />
                    </div>
                  )}

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 mb-1.5">
                      <span className="inline-flex items-center justify-center h-4 w-4 rounded-full bg-orange-500 text-white text-xs font-bold shrink-0">
                        {i + 1}
                      </span>
                      <span className="text-xs font-medium text-slate-700 truncate">
                        {s.nearest_frame_name
                          ? `Near frame ${s.nearest_frame_name.replace(/^frame_0*/, '#').replace('.jpg','').replace('.png','')}`
                          : `Area ${i + 1}`}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 leading-relaxed">
                      {s.message || `${(s.cluster_size as number)?.toLocaleString()} under-covered points — aim the camera at this spot from the suggested position.`}
                    </p>
                    {s.cluster_size && !s.message && (
                      <p className="text-xs text-slate-400 mt-1">
                        {(s.cluster_size as number).toLocaleString()} points
                      </p>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <p className="text-xs text-slate-400">
          Orange markers in the Scene Overview tab show suggested camera positions in 3D.
          Click a thumbnail to enlarge the reference frame.
        </p>
      </div>
    </>
  );
}
