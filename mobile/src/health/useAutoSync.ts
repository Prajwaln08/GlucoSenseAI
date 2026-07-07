/**
 * Near-real-time Health Connect sync.
 *
 * The manual "Sync now" button was the ONLY sync trigger, so watch data went
 * stale the moment the user stopped tapping. This hook keeps data flowing
 * whenever the app is open: it syncs on launch, again every SYNC_EVERY_MS,
 * and whenever the app returns to the foreground. All runs are silent
 * (interactive: false — never pops the permission sheet) and throttled so
 * overlapping triggers don't double-sync. Backend ingest is idempotent, so
 * re-syncing the same window is cheap.
 */
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import { AppState } from 'react-native';

import { getHcStatus, syncHealthConnect } from '@/health/healthConnect';

const SYNC_EVERY_MS   = 5 * 60_000;  // periodic cadence while the app is open
const MIN_GAP_MS      = 2 * 60_000;  // ignore triggers closer together than this
const AUTO_SYNC_HOURS = 6;           // small window — dedup makes re-syncs cheap

export function useAutoSync() {
  const qc = useQueryClient();
  const lastRun = useRef(0);
  const running = useRef(false);

  useEffect(() => {
    let alive = true;

    async function run() {
      if (running.current || Date.now() - lastRun.current < MIN_GAP_MS) return;
      const status = await getHcStatus();
      if (!alive || !status.connected) return;   // user hasn't connected HC yet
      running.current = true;
      try {
        const res = await syncHealthConnect(AUTO_SYNC_HOURS, { interactive: false });
        if (res.ok) {
          lastRun.current = Date.now();
          qc.setQueryData(['hc-status'], { connected: true, lastSync: new Date().toISOString() });
          qc.invalidateQueries({ queryKey: ['timeseries'] });      // fresh HR → watch gate / forecast
          qc.invalidateQueries({ queryKey: ['recommendations'] });
        }
      } catch { /* never crash the UI from a background sync */ }
      running.current = false;
    }

    run();                                                          // on launch
    const iv = setInterval(run, SYNC_EVERY_MS);                     // while open
    const sub = AppState.addEventListener('change', (s) => {        // on return
      if (s === 'active') run();
    });
    return () => { alive = false; clearInterval(iv); sub.remove(); };
  }, [qc]);
}
