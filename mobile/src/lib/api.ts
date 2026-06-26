/**
 * Backend API client.
 *
 * Base URL comes from EXPO_PUBLIC_API_URL (set per environment). Defaults to
 * localhost for the simulator; on a physical device set it to your machine's LAN
 * IP or the deployed Render URL. NO secret keys here — only the user's JWT, which
 * is kept in expo-secure-store and attached as a Bearer token.
 */
import axios from 'axios';
import * as SecureStore from 'expo-secure-store';

export const API_URL =
  process.env.EXPO_PUBLIC_API_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';

const TOKEN_KEY = 'glucosense.jwt';

export async function getToken() {
  try { return await SecureStore.getItemAsync(TOKEN_KEY); } catch { return null; }
}
export async function setToken(t: string) {
  try { await SecureStore.setItemAsync(TOKEN_KEY, t); } catch {}
}
export async function clearToken() {
  try { await SecureStore.deleteItemAsync(TOKEN_KEY); } catch {}
}

export const api = axios.create({ baseURL: API_URL, timeout: 20000 });

api.interceptors.request.use(async (config) => {
  const token = await getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// ── Types ───────────────────────────────────────────────────────────────────
export type Profile = {
  email: string; name?: string; age?: number; gender?: string;
  height_cm?: number; weight_kg?: number; bmi?: number;
  bp_systolic?: number; bp_diastolic?: number; bp_recorded_at?: string;
  hba1c?: number; diabetes_type?: string; medical_history?: string;
  medications?: string; onboarding_complete: boolean;
};
export type GlucosePoint = { t: string; mgdl: number; horizon_min?: number };
export type Timeseries = {
  now: string | null; range: { low: number; high: number };
  raw: GlucosePoint[]; predicted: GlucosePoint[];
};

// ── Endpoints ───────────────────────────────────────────────────────────────
export const Api = {
  async login(email: string, password: string) {
    const body = new URLSearchParams({ username: email, password });
    const { data } = await api.post('/auth/token', body.toString(), {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    await setToken(data.access_token);
    return data;
  },
  async register(email: string, password: string) {
    const { data } = await api.post('/auth/register', { email, password });
    return data;
  },
  async getProfile(): Promise<Profile> {
    const { data } = await api.get('/me/profile');
    return data;
  },
  async updateProfile(patch: Partial<Profile>): Promise<Profile> {
    const { data } = await api.put('/me/profile', patch);
    return data;
  },
  async timeseries(hours = 6): Promise<Timeseries> {
    const { data } = await api.get('/glucose/timeseries', { params: { hours } });
    return data;
  },
  async addFood(food: { meal_type: string; description?: string; carbs_g?: number; protein_g?: number; fat_g?: number; calories?: number }) {
    const { data } = await api.post('/food/log', food);
    return data;
  },
  async addVital(v: { kind: string; value?: number; bp_systolic?: number; bp_diastolic?: number; source?: string }) {
    const { data } = await api.post('/vitals', { source: 'home', ...v });
    return data;
  },
  async vitals() {
    const { data } = await api.get('/vitals');
    return data;
  },
};
