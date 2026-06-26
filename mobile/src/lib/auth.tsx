/**
 * Auth context.
 *
 * Phase 1: mock — sign-in/up just flips state and persists a flag so the app is
 * fully navigable against mock data. Phase 2 swaps the bodies for real calls to
 * the FastAPI backend (JWT in expo-secure-store) — the surface stays the same.
 *
 * No secret keys ever live in the app: the Anthropic key etc. stay server-side;
 * the app only holds the user's JWT.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';

type Session = { token: string; email: string } | null;

type AuthState = {
  ready: boolean;
  session: Session;
  onboarded: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  completeOnboarding: () => Promise<void>;
};

const KEY = 'glucosense.auth';
const AuthCtx = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [session, setSession] = useState<Session>(null);
  const [onboarded, setOnboarded] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const raw = await AsyncStorage.getItem(KEY);
        if (raw) {
          const s = JSON.parse(raw);
          setSession(s.session ?? null);
          setOnboarded(!!s.onboarded);
        }
      } finally {
        setReady(true);
      }
    })();
  }, []);

  async function persist(next: { session: Session; onboarded: boolean }) {
    await AsyncStorage.setItem(KEY, JSON.stringify(next));
  }

  const value = useMemo<AuthState>(
    () => ({
      ready,
      session,
      onboarded,
      async signIn(email) {
        const s = { token: 'mock-token', email };
        setSession(s);
        setOnboarded(true); // existing user → straight to the app
        await persist({ session: s, onboarded: true });
      },
      async signUp(email) {
        const s = { token: 'mock-token', email };
        setSession(s);
        setOnboarded(false); // new user → onboarding first
        await persist({ session: s, onboarded: false });
      },
      async completeOnboarding() {
        setOnboarded(true);
        await persist({ session, onboarded: true });
      },
      async signOut() {
        setSession(null);
        setOnboarded(false);
        await AsyncStorage.removeItem(KEY);
      },
    }),
    [ready, session, onboarded],
  );

  return <AuthCtx.Provider value={value}>{children}</AuthCtx.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthCtx);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
