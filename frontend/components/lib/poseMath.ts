/**
 * poseMath — shared helpers for interpreting SfM camera poses from cameras.json.
 *
 * cameras.json stores `cam_from_world` as a 3×4 matrix [R | t] where
 *   X_cam = R · X_world + t   (world → camera, COLMAP convention).
 * The camera looks along +Z in camera space, with +X right and -Y up
 * (image y-down).
 */

import * as THREE from 'three';

/** Rotation part R (3×3) of a cam_from_world 3×4 matrix. */
export function rotationMatrix(matrix3x4: number[][]): THREE.Matrix3 {
  return new THREE.Matrix3().set(
    matrix3x4[0][0], matrix3x4[0][1], matrix3x4[0][2],
    matrix3x4[1][0], matrix3x4[1][1], matrix3x4[1][2],
    matrix3x4[2][0], matrix3x4[2][1], matrix3x4[2][2],
  );
}

/** Camera centre in world space: C = -Rᵀ · t. */
export function worldPosition(matrix3x4: number[][]): THREE.Vector3 {
  const R = rotationMatrix(matrix3x4);
  const t = new THREE.Vector3(matrix3x4[0][3], matrix3x4[1][3], matrix3x4[2][3]);
  return t.clone().applyMatrix3(R.clone().transpose()).negate();
}

/** Camera view direction in world space: Rᵀ · [0,0,1] (looks along +Z). */
export function worldDirection(matrix3x4: number[][]): THREE.Vector3 {
  const Rt = rotationMatrix(matrix3x4).transpose();
  return new THREE.Vector3(0, 0, 1).applyMatrix3(Rt).normalize();
}

/**
 * Full camera basis in world space (columns of Rᵀ), as Three.js axes.
 * COLMAP camera axes → world: right = Rᵀ·[1,0,0], up = Rᵀ·[0,-1,0],
 * forward = Rᵀ·[0,0,1].
 */
export function worldBasis(matrix3x4: number[][]): {
  right: THREE.Vector3;
  up: THREE.Vector3;
  forward: THREE.Vector3;
} {
  const Rt = rotationMatrix(matrix3x4).transpose();
  return {
    right:   new THREE.Vector3(1, 0, 0).applyMatrix3(Rt).normalize(),
    up:      new THREE.Vector3(0, -1, 0).applyMatrix3(Rt).normalize(),
    forward: new THREE.Vector3(0, 0, 1).applyMatrix3(Rt).normalize(),
  };
}
