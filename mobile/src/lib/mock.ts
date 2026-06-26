/**
 * Mock data for Phase 1 — replaced by real backend calls in Phase 2.
 * Shapes mirror the planned API responses so swapping is mechanical.
 */
import { RANGE } from './theme';

export type GlucosePoint = { t: number; mgdl: number };

// 6 h of raw readings (10-min cadence) + 2 h of predicted, around a meal bump.
function buildSeries() {
  const now = Date.now();
  const raw: GlucosePoint[] = [];
  for (let i = 36; i >= 0; i--) {
    const t = now - i * 10 * 60 * 1000;
    const phase = (36 - i) / 36;
    const meal = Math.exp(-Math.pow((phase - 0.55) * 5, 2)) * 55; // post-meal rise
    const base = 108 + meal + Math.sin(phase * 8) * 5;
    raw.push({ t, mgdl: Math.round(base * 10) / 10 });
  }
  const last = raw[raw.length - 1].mgdl;
  const predicted: GlucosePoint[] = [{ t: now, mgdl: last }];
  for (let i = 1; i <= 12; i++) {
    const t = now + i * 10 * 60 * 1000;
    const drift = -1.4 * i + Math.sin(i / 2) * 2; // gentle fall toward range
    predicted.push({ t, mgdl: Math.round((last + drift) * 10) / 10 });
  }
  return { now, raw, predicted };
}

export const mockTimeseries = { ...buildSeries(), range: RANGE };

export const mockRecommendations = [
  { id: 'r1', kind: 'diet', title: 'Lighter dinner tonight',
    body: 'Your post-dinner spikes are the highest of the day — try halving the rice portion.' },
  { id: 'r2', kind: 'activity', title: '10-min walk after lunch',
    body: 'A short walk after your largest meal flattened your curve last week.' },
];

export const mockFood = [
  { id: 'f1', meal: 'Lunch', desc: 'Rice, dal, salad', carbs: 62, when: '12:40' },
  { id: 'f2', meal: 'Snack', desc: 'Apple', carbs: 22, when: '16:10' },
];

export const mockVitals = [
  { id: 'v1', kind: 'BP', value: '128 / 84', when: 'Today 08:10' },
  { id: 'v2', kind: 'Weight', value: '72.4 kg', when: 'Yesterday' },
];

export const mockChat = [
  { id: 'c1', role: 'assistant',
    content: "Hi! I'm your GlucoSense coach. Ask me anything, or just tell me what you ate and I'll log it. (Educational guidance only — not medical advice.)" },
];

export const mockConnections = [
  { id: 'hc', name: 'Health Connect', status: 'Not connected', cta: 'Connect' },
  { id: 'cgm', name: 'CGM (Libre / Dexcom)', status: 'Not connected', cta: 'Connect' },
];

export const mockProfile = {
  name: 'Demo User', age: 45, gender: 'F', heightCm: 168, weightKg: 72.4,
  bp: '128 / 84', bpRecorded: '2026-06-26', hba1c: 6.1, diabetesType: 'Type 2',
  conditions: 'Hypertension', medications: 'Metformin 500mg',
};
