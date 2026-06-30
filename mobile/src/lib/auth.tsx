/**
 * Auth context — real backend, with a sliding 5-day idle session.
 *
 * Login persists across app kills (JWT in expo-secure-store). Every time the app
 * opens / foregrounds we refresh a `lastActive` timestamp; if the gap exceeds
 * 5 days the user is signed out AND Health Connect is disconnected. An expired/
 * invalid token (401) triggers the same clean logout. No secret keys in the app.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { AppState } from 'react-native';

import { disconnectHealthConnect } from '@/health/healthConnect';
import { Api, clearToken, getToken, setUnauthorizedHandler } from './api';

type Session = { email: string } | null;

type AuthState = {
  ready: boolean;
  session: Session;
  onboarded: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  completeOnboarding: () => void;
};

const AuthCtx = createContext<AuthState | null>(null);

const LAST_ACTIVE_KEY = 'glucose.lastActive';
const IDLE_LIMIT_MS = 5 * 24 * 60 * 60 * 1000;   // 5 days

export function AuthProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [session, setSession] = useState<Session>(null);
  const [onboarded, setOnboarded] = useState(false);

  const stampActive = useCallback(() => AsyncStorage.setItem(LAST_ACTIVE_KEY, String(Date.now())), []);

  // Full logout: clear token + Health Connect + the idle timer + UI state.
  const fullLogout = useCallback(async () => {
    await clearToken();
    await disconnectHealthConnect();
    await AsyncStorage.removeItem(LAST_ACTIVE_KEY);
    setSession(null);
    setOnboarded(false);
  }, []);

  const idleExpired = useCallback(async () => {
    const last = await AsyncStorage.getItem(LAST_ACTIVE_KEY);
    return !!last && Date.now() - Number(last) > IDLE_LIMIT_MS;
  }, []);

  // Bootstrap: restore the session unless idle > 5 days.
  useEffect(() => {
    (async () => {
      try {
        const token = await getToken();
        if (token) {
          if (await idleExpired()) {
            await fullLogout();
          } else {
            const p = await Api.getProfile();
            setSession({ email: p.email });
            setOnboarded(!!p.onboarding_complete);
            await stampActive();
          }
        }
      } catch {
        await fullLogout();
      } finally {
        setReady(true);
      }
    })();
  }, [fullLogout, idleExpired, stampActive]);

  // Expired/invalid token → clean logout.
  useEffect(() => {
    setUnauthorizedHandler(() => { fullLogout(); });
    return () => setUnauthorizedHandler(null);
  }, [fullLogout]);

  // On foreground: enforce the 5-day idle window + slide the timer forward.
  useEffect(() => {
    const sub = AppState.addEventListener('change', async (state) => {
      if (state !== 'active') return;
      if (!(await getToken())) return;
      if (await idleExpired()) await fullLogout();
      else await stampActive();
    });
    return () => sub.remove();
  }, [fullLogout, idleExpired, stampActive]);

  const value = useMemo<AuthState>(
    () => ({
      ready,
      session,
      onboarded,
      async signIn(email, password) {
        await Api.login(email, password);
        const p = await Api.getProfile();
        setSession({ email: p.email });
        setOnboarded(!!p.onboarding_complete);
        await stampActive();
      },
      async signUp(email, password) {
        await Api.register(email, password);
        await Api.login(email, password);
        setSession({ email });
        setOnboarded(false); // new user → onboarding
        await stampActive();
      },
      completeOnboarding() {
        setOnboarded(true);
      },
      signOut: fullLogout,
    }),
    [ready, session, onboarded, stampActive, fullLogout],
  );

  return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthCtx);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
