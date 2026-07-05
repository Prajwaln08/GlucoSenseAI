import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DarkTheme, DefaultTheme, ThemeProvider } from 'expo-router';
import { Stack, useRouter, useSegments } from 'expo-router';
import { useEffect } from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { initBackgroundSync } from '@/health/backgroundSync';
import { AuthProvider, useAuth } from '@/lib/auth';
import { AppThemeProvider, useColors } from '@/lib/theme';

// Register the 15-min background Health Connect sync (no-ops on builds
// without the task-manager native module — foreground sync still covers app-open).
initBackgroundSync();

// Cache responses for 30s so navigating between tabs doesn't re-hit the (remote,
// dev-only) Cockroach DB on every focus — keeps the UI snappy. retry:1 = fail fast.
const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, gcTime: 5 * 60_000, retry: 1 } },
});

function Guard() {
  const { ready, session, onboarded } = useAuth();
  const segments = useSegments();
  const router = useRouter();

  useEffect(() => {
    if (!ready) return;
    const root = segments[0];
    const inAuth = root === '(auth)';
    const inOnboarding = root === 'onboarding';

    if (!session && !inAuth) router.replace('/(auth)/login');
    else if (session && !onboarded && !inOnboarding) router.replace('/onboarding');
    // once onboarded, only bounce OUT of the auth/onboarding flows — allow other
    // authenticated screens (e.g. /cgm) to stay.
    else if (session && onboarded && (inAuth || inOnboarding)) router.replace('/(tabs)');
  }, [ready, session, onboarded, segments]);

  return null;
}

function RootNav() {
  const { isDark } = useColors();
  return (
    <ThemeProvider value={isDark ? DarkTheme : DefaultTheme}>
      <Guard />
      <Stack screenOptions={{ headerShown: false }}>
        <Stack.Screen name="index" />
        <Stack.Screen name="(auth)" />
        <Stack.Screen name="onboarding" />
        <Stack.Screen name="(tabs)" />
        <Stack.Screen name="cgm" />
      </Stack>
    </ThemeProvider>
  );
}

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <QueryClientProvider client={queryClient}>
        <AppThemeProvider>
          <AuthProvider>
            <RootNav />
          </AuthProvider>
        </AppThemeProvider>
      </QueryClientProvider>
    </SafeAreaProvider>
  );
}
