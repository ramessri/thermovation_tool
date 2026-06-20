import { clsx } from 'clsx';

interface ProgressBarProps {
  value: number;
  className?: string;
  color?: 'blue' | 'green' | 'red';
}

export function ProgressBar({ value, className, color = 'blue' }: ProgressBarProps) {
  const fill = {
    blue:  'bg-brand-600',
    green: 'bg-green-500',
    red:   'bg-red-500',
  }[color];

  return (
    <div className={clsx('w-full bg-slate-100 rounded-full overflow-hidden', className)}>
      <div
        className={clsx('h-full rounded-full transition-all duration-500 ease-out', fill)}
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  );
}
