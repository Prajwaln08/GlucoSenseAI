import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DarkTheme, DefaultTheme, ThemeProvider } from 'expo-router';
import { Stack, useRouter, useSegments } from 'expo-router';
import { useEffect } from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { AuthProvider, useAuth } from '@/lib/auth';
import { AppThemeProvider, useColors } from '@/lib/theme';

const queryClient = new QueryClient();

function Guard() {
  const { ready, session, onboarded } = useAuth();
  const segments = useSegments();
  const router = useRouter();

  useEffect(() => {
    if (!ready) return;
    const root = segments[0];
    const inAuth = root === '(auth)';
    const inOnboarding = root === 'onboarding';
    const inTabs = root === '(tabs)';

    if (!session && !inAuth) router.replace('/(auth)/login');
    else if (session && !onboarded && !inOnboarding) router.replace('/onboarding');
    else if (session && onboarded && !inTabs) router.replace('/(tabs)');
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
