import { clsx } from 'clsx';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'sm' | 'md' | 'lg';
}

export function Card({ children, className, padding = 'md' }: CardProps) {
  const paddings = { sm: 'p-4', md: 'p-6', lg: 'p-8' };
  return (
    <div className={clsx(
      'bg-white rounded-xl border border-slate-200 shadow-sm',
      paddings[padding],
      className
    )}>
      {children}
    </div>
  );
}
