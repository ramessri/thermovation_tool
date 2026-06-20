'use client';

import Link from "next/link";
import { Layers, FolderOpen, Settings, Menu, X } from "lucide-react";
import { useState } from "react";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);

  const navLinks = (
    <>
      <Link
        href="/"
        onClick={() => setOpen(false)}
        className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
      >
        <FolderOpen className="h-4 w-4" />
        Projects
      </Link>
    </>
  );

  return (
    <div className="flex h-full">

      {/* ── Desktop sidebar ── */}
      <aside className="hidden md:flex w-60 flex-shrink-0 flex-col bg-slate-900">
        <div className="flex items-center gap-3 px-5 py-5 border-b border-slate-800">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600">
            <Layers className="h-4 w-4 text-white" />
          </div>
          <span className="text-sm font-semibold text-white tracking-tight">Photogram</span>
        </div>
        <nav className="flex-1 px-3 py-4 space-y-1">{navLinks}</nav>
        <div className="border-t border-slate-800 px-3 py-4">
          <Link href="#" className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-slate-400 hover:bg-slate-800 hover:text-white transition-colors">
            <Settings className="h-4 w-4" />
            Settings
          </Link>
        </div>
      </aside>

      {/* ── Mobile top bar ── */}
      <div className="md:hidden fixed top-0 left-0 right-0 z-40 flex items-center justify-between bg-slate-900 px-4 py-3 shadow-lg">
        <div className="flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-600">
            <Layers className="h-3.5 w-3.5 text-white" />
          </div>
          <span className="text-sm font-semibold text-white">Photogram</span>
        </div>
        <button
          onClick={() => setOpen(o => !o)}
          className="text-slate-300 hover:text-white p-1 rounded"
          aria-label="Toggle menu"
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* ── Mobile drawer overlay ── */}
      {open && (
        <div className="md:hidden fixed inset-0 z-30 flex">
          <div className="flex w-64 flex-col bg-slate-900 pt-14 shadow-xl">
            <nav className="flex-1 px-3 py-4 space-y-1">{navLinks}</nav>
            <div className="border-t border-slate-800 px-3 py-4">
              <Link href="#" onClick={() => setOpen(false)} className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-slate-400 hover:bg-slate-800 hover:text-white transition-colors">
                <Settings className="h-4 w-4" />
                Settings
              </Link>
            </div>
          </div>
          <div className="flex-1 bg-black/50" onClick={() => setOpen(false)} />
        </div>
      )}

      {/* ── Main content ── */}
      <div className="flex flex-1 flex-col min-w-0">
        <main className="flex-1 overflow-y-auto pt-14 md:pt-0">
          <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
            {children}
          </div>
        </main>
      </div>

    </div>
  );
}
