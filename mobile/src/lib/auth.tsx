/**
 * Auth context — real backend.
 *
 * JWT lives in expo-secure-store (via api.ts). Onboarding state comes from the
 * user's profile (`onboarding_complete`). No secret keys in the app.
 */
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Api, clearToken, getToken } from './api';

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

export function AuthProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [session, setSession] = useState<Session>(null);
  const [onboarded, setOnboarded] = useState(false);

  // Bootstrap: if we have a stored token, load the profile to restore the session.
  useEffect(() => {
    (async () => {
      try {
        const token = await getToken();
        if (token) {
          const p = await Api.getProfile();
          setSession({ email: p.email });
          setOnboarded(!!p.onboarding_complete);
        }
      } catch {
        await clearToken();
      } finally {
        setReady(true);
      }
    })();
  }, []);

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
      },
      async signUp(email, password) {
        await Api.register(email, password);
        await Api.login(email, password);
        setSession({ email });
        setOnboarded(false); // new user → onboarding
      },
      completeOnboarding() {
        setOnboarded(true);
      },
      async signOut() {
        await clearToken();
        setSession(null);
        setOnboarded(false);
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
