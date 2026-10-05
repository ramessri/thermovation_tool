'use client';

interface GroundTruthField {
  predicted_m: number;
  truth_m: number;
  error_m: number;
  error_pct: number;
}

interface GroundTruthCheck {
  length_m?: GroundTruthField;
  breadth_m?: GroundTruthField;
  height_m?: GroundTruthField;
  mean_abs_error_pct?: number;
}

export interface Dimensions {
  length_m: number;
  breadth_m: number;
  height_m: number;
  footprint_m2: number;
  volume_m3: number;
  ground_truth_check?: GroundTruthCheck;
}

interface Props {
  dimensions?: Dimensions | null;
}

function errorColor(pct: number): string {
  const abs = Math.abs(pct);
  if (abs <= 3) return 'text-green-600';
  if (abs <= 8) return 'text-amber-600';
  return 'text-red-600';
}

function GroundTruthRow({ field, label }: { field?: GroundTruthField; label: string }) {
  if (!field) return null;
  const sign = field.error_pct > 0 ? '+' : '';
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-500">{label} ground truth: {field.truth_m.toFixed(2)}m</span>
      <span className={`font-semibold tabular-nums ${errorColor(field.error_pct)}`}>
        {sign}{field.error_pct.toFixed(1)}% ({sign}{field.error_m.toFixed(3)}m)
      </span>
    </div>
  );
}

export function DimensionsCard({ dimensions }: Props) {
  if (!dimensions) return null;
  const { length_m, breadth_m, height_m, footprint_m2, volume_m3, ground_truth_check } = dimensions;

  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-3">
      <div className="text-sm font-semibold text-slate-800 mb-2">Room Dimensions</div>
      <div className="grid grid-cols-3 gap-3 text-center">
        <div>
          <div className="text-2xl font-bold tabular-nums text-slate-900">
            {length_m.toFixed(2)}<span className="text-sm font-medium text-slate-500"> m</span>
          </div>
          <div className="text-xs text-slate-500 mt-0.5">Length</div>
        </div>
        <div>
          <div className="text-2xl font-bold tabular-nums text-slate-900">
            {breadth_m.toFixed(2)}<span className="text-sm font-medium text-slate-500"> m</span>
          </div>
          <div className="text-xs text-slate-500 mt-0.5">Breadth</div>
        </div>
        <div>
          <div className="text-2xl font-bold tabular-nums text-slate-900">
            {height_m.toFixed(2)}<span className="text-sm font-medium text-slate-500"> m</span>
          </div>
          <div className="text-xs text-slate-500 mt-0.5">Height</div>
        </div>
      </div>
      <div className="mt-3 pt-2 border-t border-slate-100 flex justify-between text-xs text-slate-500">
        <span>Footprint: {footprint_m2.toFixed(1)} m²</span>
        <span>Volume: {volume_m3.toFixed(1)} m³</span>
      </div>

      {ground_truth_check && (
        <div className="mt-3 pt-2 border-t border-slate-100 space-y-1 text-xs">
          <div className="flex items-center justify-between mb-1">
            <span className="font-semibold text-slate-700">Ground-truth check</span>
            {ground_truth_check.mean_abs_error_pct != null && (
              <span className={`font-semibold ${errorColor(ground_truth_check.mean_abs_error_pct)}`}>
                {ground_truth_check.mean_abs_error_pct.toFixed(1)}% mean abs error
              </span>
            )}
          </div>
          <GroundTruthRow field={ground_truth_check.length_m} label="Length" />
          <GroundTruthRow field={ground_truth_check.breadth_m} label="Breadth" />
          <GroundTruthRow field={ground_truth_check.height_m} label="Height" />
        </div>
      )}
    </div>
  );
}
