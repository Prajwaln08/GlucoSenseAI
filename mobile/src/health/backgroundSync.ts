/**
 * Background Health Connect sync (app closed / backgrounded).
 *
 * Android WorkManager via expo-background-fetch: runs ~every 15 min (OS decides
 * the exact moment — battery/Doze can stretch it). Complements useAutoSync,
 * which covers the app-open case at a 5-min cadence.
 *
 * IMPORTANT: the native modules (expo-task-manager / expo-background-fetch)
 * only exist in a dev/production build made AFTER they were added — everything
 * here is guarded so an older dev client (or Expo Go) just logs and moves on.
 * Requires the READ_HEALTH_DATA_IN_BACKGROUND permission (declared in
 * plugins/withHealthPermissions.js) — the user can toggle "Allow background
 * access" for GlucoSense inside Health Connect once the new build is installed.
 */
import { getHcStatus, syncHealthConnect } from '@/health/healthConnect';

const TASK = 'glucosense-hc-sync';
const INTERVAL_S = 15 * 60;   // Android WorkManager minimum

export async function initBackgroundSync(): Promise<boolean> {
  try {
    const TaskManager = await import('expo-task-manager');
    const BackgroundFetch = await import('expo-background-fetch');

    if (!TaskManager.isTaskDefined(TASK)) {
      TaskManager.defineTask(TASK, async () => {
        try {
          const status = await getHcStatus();
          if (!status.connected) return BackgroundFetch.BackgroundFetchResult.NoData;
          const res = await syncHealthConnect(6, { interactive: false });
          return res.ok && (res.samples ?? 0) > 0
            ? BackgroundFetch.BackgroundFetchResult.NewData
            : BackgroundFetch.BackgroundFetchResult.NoData;
        } catch {
          return BackgroundFetch.BackgroundFetchResult.Failed;
        }
      });
    }

    const already = await TaskManager.isTaskRegisteredAsync(TASK);
    if (!already) {
      await BackgroundFetch.registerTaskAsync(TASK, {
        minimumInterval: INTERVAL_S,
        stopOnTerminate: false,   // keep running after the app is swiped away
        startOnBoot: true,        // resume after phone reboot
      });
    }
    return true;
  } catch {
    // Native module missing (old dev build / Expo Go) — foreground sync still covers app-open.
    return false;
  }
}
