'use client';

import React from 'react';

type SceneType = 'indoor_room' | 'outdoor' | 'object' | string | null | undefined;

interface Pass { icon: string; title: string; detail: string; }
interface GuideContent { headline: string; intro: string; passes: Pass[]; tips: string[]; aruco: string; }

const GUIDES: Record<string, GuideContent> = {
  indoor_room: {
    headline: 'Scanning an indoor room',
    intro: 'Move slowly and steadily. Aim for 70–80% overlap between consecutive frames.',
    passes: [
      { icon: '🔄', title: 'Centre 360°', detail: 'Stand in the middle of the room. Rotate slowly on the spot at waist height — one full circle. This anchors the reconstruction.' },
      { icon: '🚶', title: 'Perimeter walk', detail: 'Walk along each wall staying ~1.5 m away, camera aimed inward. Move slowly — fast panning causes blur. Cover all four walls including behind furniture.' },
      { icon: '⬇️', title: 'Low pass (floor)', detail: 'Repeat the perimeter walk with the camera angled ~30° down toward the floor–wall junction. This captures the floor and lower wall details.' },
      { icon: '⬆️', title: 'High pass (ceiling)', detail: 'Aim the camera upward at ~45° and walk the perimeter again. Captures ceiling, upper walls, and light fixtures.' },
      { icon: '📐', title: 'Corners & details', detail: 'Slow down at every corner — hold the camera still for 2–3 seconds. Repeat for any alcoves, doorways, or complex surfaces.' },
    ],
    tips: [
      "Film in even, diffuse light — avoid direct sunlight through windows as it creates harsh moving shadows",
      "The floor-level pass is the most commonly missed — don't skip it",
      "Walk at half your normal pace; most blur comes from moving too fast",
      "Overlap videos if shooting multiple clips — end one in the same spot the next begins",
    ],
    aruco: 'Place 2+ markers flat on the floor, spaced at least 1 m apart so they are both visible in the same frame as you walk. Avoid placing them in corners where they may be occluded.',
  },
  outdoor: {
    headline: 'Scanning an outdoor subject',
    intro: 'Shoot in overcast or open shade — direct sunlight creates shadows that move between frames and confuse feature matching.',
    passes: [
      { icon: '🔄', title: 'Ground-level orbit', detail: 'Walk a full circle around the subject at a consistent distance (~3–5 m for a building facade). Keep the subject centred in frame.' },
      { icon: '↗️', title: 'Mid-height pass', detail: 'Repeat the orbit with the camera angled ~30° upward. Captures mid-height details and bridging geometry between passes.' },
      { icon: '⬆️', title: 'Top pass (if accessible)', detail: 'Shoot from an elevated position (ladder, balcony) angled down at ~45°. Captures roof, top surfaces, and overall structure.' },
      { icon: '🔍', title: 'Detail closeups', detail: 'Move closer and shoot important details — entrances, signage, complex textures. Hold still for 2–3 seconds per position.' },
    ],
    tips: [
      "Shoot in the morning or evening to avoid harsh midday shadows",
      "Overcast days give the most consistent, shadow-free results",
      "For large facades, maintain a consistent stand-off distance — vary height, not distance",
      "Include the ground plane in every pass so the reconstruction has a stable base",
    ],
    aruco: 'Place markers on horizontal surfaces at ground level. Space them so at least 2 are visible simultaneously from your orbit radius.',
  },
  object: {
    headline: 'Scanning a small object',
    intro: 'Place the object on a plain, matte, non-reflective surface. Consistent, diffuse lighting is critical — avoid direct flash or spotlight.',
    passes: [
      { icon: '🔄', title: 'Eye-level orbit', detail: 'Shoot a full circle around the object every 15–20°. Keep the camera at the same height and distance throughout.' },
      { icon: '↗️', title: '45° orbit', detail: 'Repeat the orbit from a 45° elevated angle looking down at the object. This captures the top face and side geometry.' },
      { icon: '⬆️', title: 'Top-down', detail: 'Shoot directly overhead. Capture the top surface fully — move the camera to a few positions rather than a single shot.' },
      { icon: '🔍', title: 'Detail shots', detail: 'Move in closer for any complex or small features (engravings, connectors, labels). Hold still — do not film while moving.' },
    ],
    tips: [
      "A plain white or grey background makes segmentation easier",
      "Soft box or window light from the side works best — no shadows falling on the object itself",
      "Avoid reflective surfaces (metal, glass) — use dulling spray or polarising filter",
      "A slow-rotation turntable with the camera fixed is ideal for perfect orbits",
    ],
    aruco: 'Place a small printed marker (≥5 cm side) flat on the surface next to the object, within the field of view for all passes.',
  },
};

export function ShootingGuide({ sceneType }: { sceneType: SceneType }) {
  const guide = (sceneType && GUIDES[sceneType]) ? GUIDES[sceneType] : GUIDES.indoor_room;
  const [open, setOpen] = React.useState(false);

  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 overflow-hidden">
      {/* Always-visible header */}
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-slate-100 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-base">📷</span>
          <span className="text-sm font-semibold text-slate-800">{guide.headline}</span>
        </div>
        <span className="text-slate-400 text-xs font-medium">{open ? '▲ collapse' : '▼ expand guide'}</span>
      </button>

      {/* Collapsed summary */}
      {!open && (
        <p className="px-4 pb-3 text-xs text-slate-500">{guide.intro}</p>
      )}

      {/* Expanded detail */}
      {open && (
        <div className="px-4 pb-4 space-y-3">
          <p className="text-xs text-slate-500">{guide.intro}</p>

          <div className="grid grid-cols-1 gap-2">
            {guide.passes.map((pass, i) => (
              <div key={i} className="flex gap-3 rounded-lg bg-white border border-slate-100 px-3 py-2">
                <span className="text-base shrink-0 mt-0.5">{pass.icon}</span>
                <div>
                  <p className="text-xs font-semibold text-slate-700">{pass.title}</p>
                  <p className="text-xs text-slate-500 mt-0.5">{pass.detail}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="rounded-lg bg-indigo-50 border border-indigo-100 px-3 py-2">
            <p className="text-xs font-semibold text-indigo-800 mb-0.5">📍 ArUco marker placement</p>
            <p className="text-xs text-indigo-700">{guide.aruco}</p>
          </div>

          <div>
            <p className="text-xs font-semibold text-slate-600 mb-1.5">💡 Tips</p>
            <ul className="space-y-1">
              {guide.tips.map((tip, i) => (
                <li key={i} className="text-xs text-slate-500 flex gap-1.5">
                  <span className="text-slate-300 shrink-0">•</span>{tip}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
