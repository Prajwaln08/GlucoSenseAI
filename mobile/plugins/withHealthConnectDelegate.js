/**
 * react-native-health-connect v3 needs the permission launcher registered in the
 * Activity, or requestPermission() crashes with:
 *   "lateinit property requestPermission has not been initialized".
 * Its Expo plugin only adds the rationale intent-filter, so we inject the
 * delegate registration into MainActivity.onCreate ourselves.
 *   https://github.com/matinzd/react-native-health-connect  (MainActivity setup)
 */
const { withMainActivity } = require('@expo/config-plugins');

const IMPORT = 'import dev.matinzd.healthconnect.permissions.HealthConnectPermissionDelegate';
const CALL = 'HealthConnectPermissionDelegate.setPermissionDelegate(this)';

module.exports = function withHealthConnectDelegate(config) {
  return withMainActivity(config, (cfg) => {
    if (cfg.modResults.language !== 'kt') return cfg;   // Expo SDK 56 = Kotlin
    let src = cfg.modResults.contents;

    // 1. import after the package declaration
    if (!src.includes(IMPORT)) {
      src = src.replace(/(^package .+$)/m, `$1\n\n${IMPORT}`);
    }

    // 2. register the delegate right after super.onCreate(...) inside onCreate
    if (!src.includes(CALL)) {
      if (/super\.onCreate\([^)]*\)/.test(src)) {
        src = src.replace(/(super\.onCreate\([^)]*\)[ \t]*\n)/, `$1    ${CALL}\n`);
      } else {
        // no onCreate present — add a minimal one
        src = src.replace(
          /(class MainActivity\s*:\s*ReactActivity\(\)\s*\{\n)/,
          `$1  override fun onCreate(savedInstanceState: android.os.Bundle?) {\n` +
          `    super.onCreate(savedInstanceState)\n    ${CALL}\n  }\n\n`,
        );
      }
    }

    cfg.modResults.contents = src;
    return cfg;
  });
};
