/**
 * Declares the Android Health Connect READ permissions in AndroidManifest.
 * The react-native-health-connect plugin only adds the rationale intent-filter;
 * the actual `android.permission.health.*` uses-permission entries go here.
 *
 * Read-only: we never write health data, only read what the user's CGM/watch
 * already recorded. Requested at runtime via requestPermission().
 */
const { withAndroidManifest } = require('@expo/config-plugins');

const READ_PERMISSIONS = [
  'android.permission.health.READ_BLOOD_GLUCOSE',
  'android.permission.health.READ_HEART_RATE',
  'android.permission.health.READ_STEPS',
  'android.permission.health.READ_SLEEP',
  'android.permission.health.READ_OXYGEN_SATURATION',
  'android.permission.health.READ_ACTIVE_CALORIES_BURNED',
  'android.permission.health.READ_DISTANCE',
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
    return config;
  });
};
