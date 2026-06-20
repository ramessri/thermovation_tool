import { clsx } from 'clsx';

type BadgeVariant = 'created' | 'processing' | 'needs_more' | 'complete' | 'failed' | 'idle';

const variants: Record<BadgeVariant, string> = {
  created:    'bg-slate-100 text-slate-600 ring-slate-200',
  processing: 'bg-blue-50 text-blue-700 ring-blue-200',
  needs_more: 'bg-orange-50 text-orange-700 ring-orange-200',
  complete:   'bg-green-50 text-green-700 ring-green-200',
  failed:     'bg-red-50 text-red-700 ring-red-200',
  idle:       'bg-slate-100 text-slate-500 ring-slate-200',
};

const labels: Partial<Record<string, string>> = {
  needs_more: 'Needs More Footage',
  awaiting_scale_confirmation: 'processing',
};

export function Badge({ status }: { status: string }) {
  const variant = (variants[status as BadgeVariant] ?? variants.idle);
  const label = labels[status] ?? status;
  return (
    <span className={clsx(
      'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset capitalize',
      variant
    )}>
      {(status === 'processing' || status === 'awaiting_scale_confirmation') && (
        <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-blue-600 animate-pulse" />
      )}
      {status === 'needs_more' && (
        <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-orange-500 animate-pulse" />
      )}
      {label}
    </span>
  );
}
