/**
 * Android Health Connect bridge.
 *
 * Reads the user's own CGM glucose + watch activity (HR / steps / sleep / SpO2)
 * from Health Connect and ships it to the backend's /health-connect/sync. Read
 * only — we never write health data. Everything is guarded so the JS still
 * bundles + runs on iOS / web / Expo Go (where it returns `unavailable`).
 *
 * NOTE: the native module only exists in a dev/production build (not Expo Go).
 */
import { Platform } from 'react-native';
import { Api } from '@/lib/api';

export type SyncResult = {
  ok: boolean;
  reason?: 'platform' | 'unavailable' | 'denied' | 'error';
  message?: string;
  cgm_inserted?: number;
  activity_days?: number;
};

const READ = [
  'BloodGlucose', 'HeartRate', 'Steps', 'ActiveCaloriesBurned',
  'Distance', 'SleepSession', 'OxygenSaturation',
] as const;

function dayKey(iso: string): string {
  return iso.slice(0, 10); // YYYY-MM-DD (Health Connect times are ISO)
}

/** Read the last `hours` of Health Connect data and sync it to the backend. */
export async function syncHealthConnect(hours = 48): Promise<SyncResult> {
  if (Platform.OS !== 'android') {
    return { ok: false, reason: 'platform', message: 'Health Connect is Android-only.' };
  }
  try {
    const HC = await import('react-native-health-connect');
    await HC.initialize();

    const status = await HC.getSdkStatus();
    if (status !== HC.SdkAvailabilityStatus.SDK_AVAILABLE) {
      return { ok: false, reason: 'unavailable', message: 'Install/enable Health Connect to continue.' };
    }

    const granted = await HC.requestPermission(READ.map((rt) => ({ accessType: 'read' as const, recordType: rt })));
    if (!granted || granted.length === 0) {
      return { ok: false, reason: 'denied', message: 'Permission denied.' };
    }

    const endTime = new Date().toISOString();
    const startTime = new Date(Date.now() - hours * 3600 * 1000).toISOString();
    const filter = { timeRangeFilter: { operator: 'between' as const, startTime, endTime } };
    const read = async (rt: string): Promise<any[]> => {
      try { return (await HC.readRecords(rt as any, filter)).records as any[]; } catch { return []; }
    };

    // ── Glucose → individual CGM points ──
    const glucose = (await read('BloodGlucose'))
      .map((r) => ({
        t: r.time,
        mgdl: r.level?.inMilligramsPerDeciliter ?? (r.level?.value ? r.level.value : undefined),
      }))
      .filter((g): g is { t: string; mgdl: number } => typeof g.mgdl === 'number' && !!g.t);

    // ── Activity → aggregate per day ──
    type Acc = { steps: number; kcal: number; meters: number; hr: number[]; spo2: number[]; sleepMs: number };
    const byDay = new Map<string, Acc>();
    const acc = (k: string): Acc => {
      let a = byDay.get(k);
      if (!a) { a = { steps: 0, kcal: 0, meters: 0, hr: [], spo2: [], sleepMs: 0 }; byDay.set(k, a); }
      return a;
    };
    for (const r of await read('Steps')) acc(dayKey(r.startTime)).steps += r.count ?? 0;
    for (const r of await read('ActiveCaloriesBurned')) acc(dayKey(r.startTime)).kcal += r.energy?.inKilocalories ?? 0;
    for (const r of await read('Distance')) acc(dayKey(r.startTime)).meters += r.distance?.inMeters ?? 0;
    for (const r of await read('HeartRate'))
      for (const s of r.samples ?? []) acc(dayKey(s.time ?? r.startTime)).hr.push(s.beatsPerMinute);
    for (const r of await read('OxygenSaturation')) acc(dayKey(r.time)).spo2.push(r.percentage);
    for (const r of await read('SleepSession'))
      acc(dayKey(r.startTime)).sleepMs += new Date(r.endTime).getTime() - new Date(r.startTime).getTime();

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

    const res = await Api.syncHealthConnect({ glucose, activity });
    return { ok: true, cgm_inserted: res.cgm_inserted, activity_days: res.activity_days };
  } catch (e: any) {
    return { ok: false, reason: 'error', message: e?.message ?? 'Could not read Health Connect.' };
  }
}
