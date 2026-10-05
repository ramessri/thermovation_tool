/**
 * 3D overlay of the HVAC placement result, drawn on the Point Cloud / Mesh tabs.
 *
 * All positions come from the backend in the metric world frame of the refined
 * cloud — the same frame as exports/output.ply, output.obj and the densified
 * clouds. Add the returned group as a CHILD of the loaded cloud/mesh object so
 * it inherits the viewer's re-centring and COLMAP→Three.js flip.
 */

import * as THREE from 'three';
import type { HvacPlacementResult, HvacSegmentation } from '../HvacSegmentationViewer';
import { FIXTURE_COLORS } from '../HvacSegmentationViewer';

const BEST_COLOR = 0x22c55e;      // rank 1 — filled panel
const ALT_COLOR = 0xf59e0b;       // ranks 2+ — outline only
const RUCKLAUF_COLOR = 0x2563eb;
const VORLAUF_COLOR = 0xdc2626;
const PIPE_MARKER_RADIUS_M = 0.06;
const FIXTURE_MARKER_RADIUS_M = 0.035;

// Overlay stays visible through the cloud (a panel lying on a wall would
// otherwise z-fight with the wall's own points).
function onTop<T extends THREE.Object3D>(obj: T): T {
  obj.traverse(o => {
    o.renderOrder = 999;   // per-object — renderOrder isn't inherited from the group
    const m = (o as THREE.Mesh).material as THREE.Material | undefined;
    if (m) { m.depthTest = false; m.transparent = true; }
  });
  return obj;
}

function outline(corners: THREE.Vector3[], color: number): THREE.LineLoop {
  return new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(corners),
    new THREE.LineBasicMaterial({ color }),
  );
}

function panel(corners: THREE.Vector3[], color: number): THREE.Mesh {
  const geo = new THREE.BufferGeometry().setFromPoints(corners);
  geo.setIndex([0, 1, 2, 0, 2, 3]);
  return new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    color, opacity: 0.55, side: THREE.DoubleSide,
  }));
}

function marker(position: number[], radius: number, color: number | string): THREE.Mesh {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 16, 12),
    new THREE.MeshBasicMaterial({ color }),
  );
  m.position.fromArray(position);
  return m;
}

export function hasPlacementGeometry(placement?: HvacPlacementResult | null): boolean {
  return !!placement?.candidates?.some(c => c.corners_world_m?.length === 4);
}

export function buildHvacOverlay(
  placement?: HvacPlacementResult | null,
  segmentation?: HvacSegmentation | null,
): THREE.Group | null {
  const group = new THREE.Group();

  // Draw lower-ranked candidates first so rank 1 sits on top.
  const candidates = [...(placement?.candidates ?? [])]
    .filter(c => c.corners_world_m?.length === 4)
    .sort((a, b) => b.rank - a.rank);
  for (const c of candidates) {
    const corners = c.corners_world_m!.map(p => new THREE.Vector3().fromArray(p));
    if (c.rank === 1) {
      group.add(panel(corners, BEST_COLOR));
      group.add(outline(corners, BEST_COLOR));
    } else {
      group.add(outline(corners, ALT_COLOR));
    }
  }

  for (const [label, instances] of Object.entries(segmentation?.hvac_fixtures ?? {})) {
    if (label === 'rucklauf_candidates' || label === 'vorlauf_candidates') continue;
    for (const inst of instances) {
      group.add(marker(inst.position_m, FIXTURE_MARKER_RADIUS_M, FIXTURE_COLORS[label] ?? FIXTURE_COLORS.other));
    }
  }
  if (segmentation?.rucklauf_position) {
    group.add(marker(segmentation.rucklauf_position.position_m, PIPE_MARKER_RADIUS_M, RUCKLAUF_COLOR));
  }
  if (segmentation?.vorlauf_position) {
    group.add(marker(segmentation.vorlauf_position.position_m, PIPE_MARKER_RADIUS_M, VORLAUF_COLOR));
  }

  return group.children.length ? onTop(group) : null;
}
