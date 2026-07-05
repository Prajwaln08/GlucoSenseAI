/**
 * Declares the Android Health Connect READ permissions in AndroidManifest.
 * The react-native-health-connect plugin only adds the rationale intent-filter;
 * the actual `android.permission.health.*` uses-permission entries go here.
 *
 * Read-only: we never write health data, only read what the user's CGM/watch
 * already recorded. Requested at runtime via requestPermission().
 */
const { withAndroidManifest } = require('@expo/config-plugins');

// Watch / wearable metrics only. Glucose comes from a CGM (xDRIP+ / Junction),
// NOT Health Connect — so no READ_BLOOD_GLUCOSE here.
const READ_PERMISSIONS = [
  'android.permission.health.READ_HEART_RATE',
  'android.permission.health.READ_STEPS',
  'android.permission.health.READ_SLEEP',
  'android.permission.health.READ_OXYGEN_SATURATION',
  'android.permission.health.READ_ACTIVE_CALORIES_BURNED',
  'android.permission.health.READ_DISTANCE',
  // Background sync (expo-background-fetch task): lets the 15-min WorkManager
  // job read Health Connect while the app is closed (user grants "Allow
  // background access" in Health Connect → GlucoSense).
  'android.permission.health.READ_HEALTH_DATA_IN_BACKGROUND',
];

module.exports = function withHealthPermissions(config) {
  return withAndroidManifest(config, (config) => {
    const manifest = config.modResults.manifest;
    manifest['uses-permission'] = manifest['uses-permission'] || [];
    const have = new Set(
      manifest['uses-permission'].map((p) => p.$?.['android:name']).filter(Boolean),
    );
    for (const name of READ_PERMISSIONS) {
      if (!have.has(name)) manifest['uses-permission'].push({ $: { 'android:name': name } });
    }

    // Android 14+ (Health Connect is in the OS): the app must expose a
    // VIEW_PERMISSION_USAGE activity-alias with the HEALTH_PERMISSIONS category,
    // or it won't appear in Health Connect's app-permissions list. The library's
    // own plugin only adds the (Android 13) rationale intent-filter.
    const app = manifest.application[0];
    app['activity-alias'] = app['activity-alias'] || [];
    const hasAlias = app['activity-alias'].some((a) => a.$?.['android:name'] === 'ViewPermissionUsageActivity');
    if (!hasAlias) {
      app['activity-alias'].push({
        $: {
          'android:name': 'ViewPermissionUsageActivity',
          'android:exported': 'true',
          'android:targetActivity': '.MainActivity',
          'android:permission': 'android.permission.START_VIEW_PERMISSION_USAGE',
        },
        'intent-filter': [{
          action: [{ $: { 'android:name': 'android.intent.action.VIEW_PERMISSION_USAGE' } }],
          category: [{ $: { 'android:name': 'android.intent.category.HEALTH_PERMISSIONS' } }],
        }],
      });
    }
    return config;
  });
};
