/**
 * Semantic color tokens — light + dark — plus a persisted Light/Dark/System
 * preference. No purple / violet / indigo anywhere.
 * Accent = rose/coral; glucose: emerald (in-range), amber (hypo), red (hyper).
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createContext, ReactNode, useContext, useEffect, useState } from 'react';
import { useColorScheme } from 'react-native';

export type Palette = {
  bg: string;
  surface: string;
  surfaceAlt: string;
  border: string;
  text: string;
  textMuted: string;
  accent: string;
  accentSoft: string;
  onAccent: string;
  inRange: string;
  inRangeBand: string;
  hypo: string;
  hyper: string;
  shadow: string;
};

const light: Palette = {
  bg: '#f6f7f9',
  surface: '#ffffff',
  surfaceAlt: '#f1f5f9',
  border: '#e2e8f0',
  text: '#0f172a',
  textMuted: '#64748b',
  accent: '#e11d48',       // rose-600
  accentSoft: '#ffe4e6',   // rose-100
  onAccent: '#ffffff',
  inRange: '#10b981',      // emerald-500
  inRangeBand: 'rgba(16,185,129,0.10)',
  hypo: '#f59e0b',         // amber-500
  hyper: '#ef4444',        // red-500
  shadow: 'rgba(2,6,23,0.08)',
};

const dark: Palette = {
  bg: '#0b1220',
  surface: '#111827',
  surfaceAlt: '#0f172a',
  border: '#1f2937',
  text: '#f1f5f9',
  textMuted: '#94a3b8',
  accent: '#fb7185',       // rose-400
  accentSoft: '#3f1d2b',
  onAccent: '#0b1220',
  inRange: '#34d399',      // emerald-400
  inRangeBand: 'rgba(52,211,153,0.12)',
  hypo: '#fbbf24',
  hyper: '#f87171',
  shadow: 'rgba(0,0,0,0.4)',
};

// ── Light / Dark / System preference (persisted) ──────────────────────────────
export type ThemePref = 'light' | 'dark' | 'system';
type Ctx = { pref: ThemePref; setPref: (p: ThemePref) => void };
const ThemeCtx = createContext<Ctx>({ pref: 'system', setPref: () => {} });
const PREF_KEY = 'glucose.theme';

export function AppThemeProvider({ children }: { children: ReactNode }) {
  const [pref, setP] = useState<ThemePref>('system');
  useEffect(() => {
    AsyncStorage.getItem(PREF_KEY).then((v) => {
      if (v === 'light' || v === 'dark' || v === 'system') setP(v);
    });
  }, []);
  const setPref = (p: ThemePref) => { setP(p); AsyncStorage.setItem(PREF_KEY, p); };
  return <ThemeCtx.Provider value={{ pref, setPref }}>{children}</ThemeCtx.Provider>;
}

export function useThemePref() {
  return useContext(ThemeCtx);
}

export function useColors() {
  const { pref } = useContext(ThemeCtx);
  const system = useColorScheme();
  const isDark = pref === 'system' ? system === 'dark' : pref === 'dark';
  return { ...(isDark ? dark : light), isDark };
}

export const palettes = { light, dark };

// Glucose target range (mg/dL) used across the chart + risk badges.
export const RANGE = { low: 70, high: 180 };
