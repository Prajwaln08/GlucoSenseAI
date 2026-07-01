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

// On an expired/invalid token (401), hand off to the registered handler (auth
// signs out cleanly). Login/register 401s are excluded — those are bad creds.
let onUnauthorized: (() => void) | null = null;
export function setUnauthorizedHandler(fn: (() => void) | null) { onUnauthorized = fn; }
api.interceptors.response.use(
  (r) => r,
  (error) => {
    const url: string = error?.config?.url ?? '';
    const isAuthCall = url.includes('/auth/token') || url.includes('/auth/register');
    if (error?.response?.status === 401 && !isAuthCall) onUnauthorized?.();
    return Promise.reject(error);
  },
);

// ── Types ───────────────────────────────────────────────────────────────────
export type Profile = {
  email: string; name?: string; first_name?: string; last_name?: string;
  date_of_birth?: string; age?: number; gender?: string;
  height_cm?: number; weight_kg?: number; bmi?: number;
  bp_systolic?: number; bp_diastolic?: number; bp_recorded_at?: string;
  hba1c?: number; diabetes_type?: string; medical_history?: string;
  medications?: string; onboarding_complete: boolean;
};
export type GlucosePoint = { t: string; mgdl: number; horizon_min?: number };
export type ForecastStatus = 'ready' | 'warming_up' | 'collecting' | 'no_data' | 'no_source';
export type TrainingInfo = { phase?: string; status?: string; since?: string } | null;
export type Timeseries = {
  now: string | null; range: { low: number; high: number };
  raw: GlucosePoint[]; predicted: GlucosePoint[];
  // Personalization lifecycle (real users). Demo omits these.
  status?: ForecastStatus;
  phase?: string;
  training?: TrainingInfo;
  days_have?: number; days_need?: number;   // collecting
  have?: number; need?: number;             // warming_up (readings)
};
export type WearableSample = { t: string; hr_bpm?: number; spo2_pct?: number; steps?: number; calories_active?: number; distance_m?: number };
export type ChatTurn = { role: 'user' | 'assistant'; content: string };
export type Recommendation = { id: string; kind: 'diet' | 'activity'; title: string; body: string };
export type FoodEntry = { id: string; meal_type: string; description?: string; quantity?: number; portion_size?: string; logged_at?: string };
export type VitalEntry = { id: string; kind: string; value?: number; bp_systolic?: number; bp_diastolic?: number; recorded_at?: string };
export type DailyLogs = {
  date: string;
  glucose: { count: number; avg: number; min: number; max: number; time_in_range_pct: number } | null;
  food: FoodEntry[];
  vitals: VitalEntry[];
  activity: { steps?: number; hr_avg_bpm?: number; calories_active?: number; sleep_hours?: number; spo2_avg?: number; distance_m?: number } | null;
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
  async addFood(food: { meal_type: string; description?: string; quantity?: number; portion_size?: string; logged_at?: string; carbs_g?: number; protein_g?: number; fat_g?: number; calories?: number }) {
    const { data } = await api.post('/food/log', food);
    return data;
  },
  async addVital(v: { kind: string; value?: number; bp_systolic?: number; bp_diastolic?: number; source?: string; recorded_at?: string }) {
    const { data } = await api.post('/vitals', { source: 'home', ...v });
    return data;
  },
  async foodLogs(limit = 30): Promise<{ id: string; logged_at: string; meal_type: string; description?: string }[]> {
    const { data } = await api.get('/food/logs', { params: { limit } });
    return data;
  },
  async vitals() {
    const { data } = await api.get('/vitals');
    return data;
  },
  async syncHealthConnect(payload: {
    activity: { date: string; steps?: number; calories_active?: number; distance_m?: number; hr_avg_bpm?: number; spo2_avg?: number; sleep_hours?: number }[];
    samples?: WearableSample[];
  }): Promise<{ activity_days: number; samples_inserted?: number }> {
    const { data } = await api.post('/health-connect/sync', payload);
    return data;
  },
  async wearableRecent(limit = 5): Promise<WearableSample[]> {
    const { data } = await api.get('/wearable/recent', { params: { limit } });
    return data;
  },
  async coachChat(message: string): Promise<{ reply: string; actions: any[] }> {
    const { data } = await api.post('/coach/chat', { message });
    return data;
  },
  async coachMessages(): Promise<ChatTurn[]> {
    const { data } = await api.get('/coach/messages');
    return data;
  },
  async recommendations(): Promise<Recommendation[]> {
    const { data } = await api.get('/coach/recommendations');
    return data;
  },
  async logs(date: string): Promise<DailyLogs> {
    const { data } = await api.get('/me/logs', { params: { date_str: date } });
    return data;
  },
  // ── CGM: xDRIP push key + Junction link ──
  async cgmKey(rotate = false): Promise<{ cgm_api_key: string; push_url: string }> {
    const { data } = await api.post('/cgm/key', null, { params: { rotate } });
    return data;
  },
  async junctionLink(): Promise<{ junction_user_id: string; link_token: string; link_web_url: string }> {
    const { data } = await api.post('/wearable/link-token');
    return data;
  },
  async junctionSync(): Promise<{ glucose_readings_saved: number; activity_days_saved: number; message: string }> {
    const { data } = await api.post('/wearable/sync');
    return data;
  },
  async exportData() {
    const { data } = await api.get('/account/export');
    return data;
  },
  async deleteAccount() {
    await api.delete('/account');
  },
};
