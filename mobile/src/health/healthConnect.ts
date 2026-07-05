/**
 * Android Health Connect bridge.
 *
 * Reads the user's WATCH / wearable metrics (heart rate, steps, calories, SpO2,
 * sleep, distance) from Health Connect and ships them to /health-connect/sync.
 * Glucose is NOT read here — CGM readings come from a CGM source (xDRIP+ / Junction).
 * Read-only; everything is guarded so the JS still bundles + runs on iOS / web /
 * Expo Go (where it returns `unavailable`).
 *
 * NOTE: the native module only exists in a dev/production build (not Expo Go).
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';
import { Api } from '@/lib/api';

export type SyncResult = {
  ok: boolean;
  reason?: 'platform' | 'unavailable' | 'denied' | 'error';
  message?: string;
  activity_days?: number;
  samples?: number;
};

// ── Connection status (persisted locally) ─────────────────────────────────────
const HC_KEY = 'glucose.healthconnect';
export type HcStatus = { connected: boolean; lastSync?: string };

export async function getHcStatus(): Promise<HcStatus> {
  try {
    const raw = await AsyncStorage.getItem(HC_KEY);
    return raw ? JSON.parse(raw) : { connected: false };
  } catch {
    return { connected: false };
  }
}

async function markConnected(): Promise<void> {
  await AsyncStorage.setItem(HC_KEY, JSON.stringify({ connected: true, lastSync: new Date().toISOString() }));
}

export async function disconnectHealthConnect(): Promise<void> {
  await AsyncStorage.removeItem(HC_KEY);
}

/** Open the Health Connect app's permission screen so the user can grant access. */
export async function openHealthConnect(): Promise<void> {
  if (Platform.OS !== 'android') return;
  try {
    const HC = await import('react-native-health-connect');
    await HC.openHealthConnectSettings();
  } catch { /* ignore */ }
}

// Watch / wearable metrics only. Glucose is NOT read here — CGM readings come
// from a dedicated CGM source (xDRIP+ / Junction), not Health Connect.
const READ = [
  'HeartRate', 'Steps', 'ActiveCaloriesBurned',
  'Distance', 'SleepSession', 'OxygenSaturation',
] as const;

function dayKey(iso: string): string {
  return iso.slice(0, 10); // YYYY-MM-DD (Health Connect times are ISO)
}

/** Read the last `hours` of Health Connect data and sync it to the backend.
 *  `interactive: false` (auto-sync) never shows permission UI — it silently
 *  no-ops unless access was already granted via a manual sync. */
export async function syncHealthConnect(hours = 48, opts: { interactive?: boolean } = {}): Promise<SyncResult> {
  const interactive = opts.interactive !== false;
  if (Platform.OS !== 'android') {
    return { ok: false, reason: 'platform', message: 'Health Connect is Android-only.' };
  }
  try {
    const HC = await import('react-native-health-connect');
    const initialized = await HC.initialize();
    const status = await HC.getSdkStatus();
    if (!initialized || status !== HC.SdkAvailabilityStatus.SDK_AVAILABLE) {
      return { ok: false, reason: 'unavailable',
        message: 'Open the Health Connect app and enable it (make sure your watch/fitness app syncs into it), then try again.' };
    }

    if (interactive) {
      await HC.requestPermission(READ.map((rt) => ({ accessType: 'read' as const, recordType: rt })));
    }
    // requestPermission's return is unreliable (often empty even after granting / when
    // already granted), so check the ACTUAL granted set as the source of truth.
    let grantedPerms: any[] = [];
    try { grantedPerms = await HC.getGrantedPermissions(); } catch { /* ignore */ }
    // Heart rate specifically gates every forecast — a partial grant without it looks
    // "connected" but forecasts stay stuck. Require HeartRate read, not just any read.
    const hasHR = grantedPerms.some((p: any) => p?.accessType === 'read' && p?.recordType === 'HeartRate');
    if (!hasHR) {
      const anyRead = grantedPerms.some((p: any) => p?.accessType === 'read');
      return { ok: false, reason: 'denied',
        message: anyRead
          ? 'Heart Rate access is off — forecasts need it. Open Health Connect → App permissions → GlucoSense and enable Heart rate, then tap Sync now again.'
          : 'No access yet. Tap "Allow" in the Health Connect dialog — or open Health Connect → App permissions → GlucoSense and turn on the data types (Heart rate is required), then tap Sync now again.' };
    }

    const endTime = new Date().toISOString();
    const startTime = new Date(Date.now() - hours * 3600 * 1000).toISOString();
    const filter = { timeRangeFilter: { operator: 'between' as const, startTime, endTime } };
    const read = async (rt: string): Promise<any[]> => {
      try { return (await HC.readRecords(rt as any, filter)).records as any[]; } catch { return []; }
    };
    // Read every record type ONCE, then build BOTH the daily rollup and the realtime samples.
    const [stepsR, calR, distR, hrR, spo2R, sleepR] = await Promise.all([
      read('Steps'), read('ActiveCaloriesBurned'), read('Distance'),
      read('HeartRate'), read('OxygenSaturation'), read('SleepSession'),
    ]);

    // ── Daily rollup (Logs + coach) ──
    type Acc = { steps: number; kcal: number; meters: number; hr: number[]; spo2: number[]; sleepMs: number };
    const byDay = new Map<string, Acc>();
    const acc = (k: string): Acc => {
      let a = byDay.get(k);
      if (!a) { a = { steps: 0, kcal: 0, meters: 0, hr: [], spo2: [], sleepMs: 0 }; byDay.set(k, a); }
      return a;
    };
    for (const r of stepsR) acc(dayKey(r.startTime)).steps += r.count ?? 0;
    for (const r of calR) acc(dayKey(r.startTime)).kcal += r.energy?.inKilocalories ?? 0;
    for (const r of distR) acc(dayKey(r.startTime)).meters += r.distance?.inMeters ?? 0;
    for (const r of hrR) for (const s of r.samples ?? []) acc(dayKey(s.time ?? r.startTime)).hr.push(s.beatsPerMinute);
    for (const r of spo2R) acc(dayKey(r.time)).spo2.push(r.percentage);
    for (const r of sleepR) acc(dayKey(r.startTime)).sleepMs += new Date(r.endTime).getTime() - new Date(r.startTime).getTime();
    const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : undefined);
    const activity = [...byDay.entries()].map(([date, a]) => ({
      date,
      steps: a.steps || undefined,
      calories_active: a.kcal || undefined,
      distance_m: a.meters || undefined,
      hr_avg_bpm: avg(a.hr),
      spo2_avg: avg(a.spo2),
      sleep_hours: a.sleepMs ? a.sleepMs / 3_600_000 : undefined,
    }));

    // ── Realtime intraday samples → per-minute buckets (merged metrics) ──
    const byMin = new Map<string, any>();
    const put = (t: string | undefined, patch: Record<string, number | undefined>) => {
      if (!t) return;
      const d = new Date(t); if (isNaN(d.getTime())) return;
      d.setSeconds(0, 0);
      const k = d.toISOString();
      byMin.set(k, { ...(byMin.get(k) ?? { t: k }), ...patch });
    };
    for (const r of hrR) for (const s of r.samples ?? []) put(s.time ?? r.startTime, { hr_bpm: s.beatsPerMinute });
    for (const r of spo2R) put(r.time, { spo2_pct: r.percentage });
    for (const r of stepsR) put(r.startTime, { steps: r.count });
    for (const r of calR) put(r.startTime, { calories_active: r.energy?.inKilocalories });
    for (const r of distR) put(r.startTime, { distance_m: r.distance?.inMeters });
    const samples = [...byMin.values()].sort((a, b) => (a.t < b.t ? -1 : 1));

    const res = await Api.syncHealthConnect({ activity, samples });
    await markConnected();
    return { ok: true, activity_days: res.activity_days, samples: samples.length };
  } catch (e: any) {
    return { ok: false, reason: 'error', message: e?.message ?? 'Could not read Health Connect.' };
  }
}
