import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Alert, Dimensions, Pressable, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { GlucoseChart } from '@/components/GlucoseChart';
import { LogSheet } from '@/components/LogSheet';
import { Body, Card, H1, H2, Muted, Pill, Screen } from '@/components/ui';
import { getHcStatus, openHealthConnect, syncHealthConnect, type HcStatus } from '@/health/healthConnect';
import { Api } from '@/lib/api';
import { useColors } from '@/lib/theme';

const HORIZONS = [30, 60, 90, 120];
const MODE_KEY = 'glucose.forecastMode';
type Mode = 'personalized' | 'basic';

export default function Home() {
  const c = useColors();
  const qc = useQueryClient();
  const router = useRouter();
  const [horizon, setHorizon] = useState(60);
  const [sheet, setSheet] = useState<null | 'food' | 'vitals'>(null);
  const [syncing, setSyncing] = useState(false);
  const [hc, setHc] = useState<HcStatus>({ connected: false });
  const [mode, setMode] = useState<Mode>('personalized');

  // Auto-scroll + highlight the right connector when a CTA is tapped.
  const scrollRef = useRef<ScrollView>(null);
  const connectionsY = useRef(0);
  const [highlight, setHighlight] = useState<null | 'cgm' | 'hc'>(null);

  // Real forecast only — no mock. Refetch while a personal model trains so the
  // graph swaps to the personal forecast the moment it's ready.
  const { data: ts, isLoading } = useQuery({
    queryKey: ['timeseries'],
    queryFn: () => Api.timeseries(6),
    refetchInterval: (q) => (q.state.data?.training ? 15_000 : false),
  });
  const { data: recs } = useQuery({ queryKey: ['recommendations'], queryFn: () => Api.recommendations() });
  const { data: profile } = useQuery({ queryKey: ['profile'], queryFn: () => Api.getProfile() });
  const { data: foods } = useQuery({ queryKey: ['food-recent'], queryFn: () => Api.foodLogs() });
  // Realtime watch readings — refreshes every 30s while connected (proves live data flows).
  const { data: recent } = useQuery({
    queryKey: ['wearable-recent'],
    queryFn: () => Api.wearableRecent(5),
    enabled: hc.connected,
    refetchInterval: hc.connected ? 30_000 : false,
  });
  const firstName = (profile?.first_name || profile?.name?.trim().split(/\s+/)[0] || '').trim();

  useEffect(() => { getHcStatus().then(setHc); }, []);
  useEffect(() => { AsyncStorage.getItem(MODE_KEY).then((v) => { if (v === 'basic' || v === 'personalized') setMode(v); }); }, []);
  const pickMode = useCallback((m: Mode) => { setMode(m); AsyncStorage.setItem(MODE_KEY, m); }, []);

  const flashConnector = useCallback((which: 'cgm' | 'hc') => {
    scrollRef.current?.scrollTo({ y: Math.max(0, connectionsY.current - 12), animated: true });
    setHighlight(which);
    setTimeout(() => setHighlight(null), 2600);
  }, []);

  async function connectHealthConnect() {
    setSyncing(true);
    const res = await syncHealthConnect();
    setSyncing(false);
    if (res.ok) {
      setHc({ connected: true, lastSync: new Date().toISOString() });
      qc.invalidateQueries({ queryKey: ['recommendations'] });
      qc.invalidateQueries({ queryKey: ['wearable-recent'] });   // realtime readings list
      qc.invalidateQueries({ queryKey: ['timeseries'] });        // intraday HR feeds the forecast
      Alert.alert('Health Connect', `Synced ${res.samples ?? 0} realtime readings + ${res.activity_days ?? 0} day(s) of activity.`);
    } else if (res.reason === 'denied' || res.reason === 'unavailable') {
      Alert.alert('Health Connect', res.message ?? 'Could not sync.',
        [{ text: 'Not now', style: 'cancel' }, { text: 'Open Health Connect', onPress: () => openHealthConnect() }]);
    } else {
      Alert.alert('Health Connect', res.message ?? 'Could not sync.');
    }
  }

  // ── Derive lifecycle state from the (real) timeseries ──────────────────────
  const raw = (ts?.raw ?? []).map((p) => ({ t: Date.parse(p.t), mgdl: p.mgdl }));
  const predicted = (ts?.predicted ?? []).map((p) => ({ t: Date.parse(p.t), mgdl: p.mgdl }));
  const now = ts?.now ? Date.parse(ts.now) : Date.now();
  const range = ts?.range ?? { low: 70, high: 180 };
  const hasForecast = predicted.length > 1;
  // Demo accounts omit `status` but return predictions → treat as ready.
  const status = ts?.status ?? (hasForecast ? 'ready' : raw.length ? 'collecting' : 'no_source');
  const training = ts?.training ?? null;
  const width = Dimensions.get('window').width - 24 * 2 - 16 * 2;

  const foodMarkers = (foods ?? [])
    .map((f) => ({ t: Date.parse(f.logged_at) }))
    .filter((mk) => Number.isFinite(mk.t));

  const current = raw.length ? raw[raw.length - 1].mgdl : undefined;
  const targetT = now + horizon * 60_000;
  const hl = hasForecast
    ? predicted.reduce((b, p) => (Math.abs(p.t - targetT) < Math.abs(b.t - targetT) ? p : b))
    : undefined;
  const predForHorizon = hl?.mgdl ?? predicted[predicted.length - 1]?.mgdl;

  return (
    <Screen>
      <SafeAreaView edges={['top']} style={{ flex: 1 }}>
        <ScrollView ref={scrollRef} contentContainerStyle={{ padding: 24, paddingBottom: 40 }}>
          <H1>{firstName ? `Hi ${firstName} 👋` : 'Hi 👋'}</H1>
          <Muted style={{ marginBottom: 14 }}>
            {mode === 'personalized' ? 'Your personalized glucose forecast.' : 'Watch-based insights.'}
          </Muted>

          {/* Mode selector — above the graph, changeable anytime */}
          <View style={{ flexDirection: 'row', backgroundColor: c.surfaceAlt, borderRadius: 12, padding: 4, marginBottom: 14 }}>
            {(['personalized', 'basic'] as Mode[]).map((mVal) => (
              <Pressable key={mVal} onPress={() => pickMode(mVal)}
                style={{ flex: 1, paddingVertical: 9, borderRadius: 9, alignItems: 'center',
                  backgroundColor: mode === mVal ? c.surface : 'transparent' }}>
                <Body style={{ fontWeight: '600', color: mode === mVal ? c.text : c.textMuted }}>
                  {mVal === 'personalized' ? 'Personalized · CGM' : 'Basic · Watch'}
                </Body>
              </Pressable>
            ))}
          </View>

          {/* A — the graph / status area */}
          {mode === 'personalized'
            ? renderPersonalized()
            : renderBasic()}

          {/* B — coach recommendations */}
          {!!recs?.length && (
            <View style={{ marginBottom: 14 }}>
              <H2>Doctor Gluco's Suggestions</H2>
              {recs.map((r) => (
                <Card key={r.id} style={{ marginBottom: 10 }}>
                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <Ionicons name={r.kind === 'diet' ? 'restaurant' : 'walk'} size={16} color={c.accent} />
                    <Body style={{ fontWeight: '600' }}>{r.title}</Body>
                  </View>
                  <Muted>{r.body}</Muted>
                </Card>
              ))}
            </View>
          )}

          {/* C — food & vitals logging */}
          <H2>Track food &amp; vitals</H2>
          <Card>
            <View style={{ flexDirection: 'row', gap: 12 }}>
              <Pressable onPress={() => setSheet('food')} style={ctaGhost(c.accent)}>
                <Body style={{ color: c.accent, fontWeight: '600' }}>🍽  Log food</Body>
              </Pressable>
              <Pressable onPress={() => setSheet('vitals')} style={ctaGhost(c.accent)}>
                <Body style={{ color: c.accent, fontWeight: '600' }}>❤️  Log vitals</Body>
              </Pressable>
            </View>
            <Muted style={{ marginTop: 12 }}>Logged items feed your forecast.</Muted>
          </Card>

          {/* D — connections (auto-scroll target) */}
          <View onLayout={(e) => { connectionsY.current = e.nativeEvent.layout.y; }}>
            <H2>Connections</H2>
            <Card>
              <ConnRow
                dot={hc.connected ? c.inRange : c.textMuted}
                title="Health Connect"
                subtitle={syncing ? 'Syncing…' : hc.connected ? `Connected${hc.lastSync ? ` · synced ${syncedLabel(hc.lastSync)}` : ''}` : 'Watch data — HR, steps, sleep, SpO₂'}
                action={syncing ? 'Syncing…' : hc.connected ? 'Sync' : 'Sync now'}
                onPress={connectHealthConnect} disabled={syncing}
                border={highlight === 'hc' ? c.accent : undefined} c={c} />
              <ConnRow
                dot={c.textMuted} top
                title="CGM (Libre / Dexcom)"
                subtitle="Stream with xDRIP+ · Junction soon"
                action="Set up" onPress={() => router.push('/cgm')}
                border={highlight === 'cgm' ? c.accent : undefined} c={c} />
            </Card>
          </View>
        </ScrollView>

        <LogSheet
          visible={sheet !== null}
          mode={sheet ?? 'food'}
          onClose={() => setSheet(null)}
          onLogged={() => {
            qc.invalidateQueries({ queryKey: ['timeseries'] });
            qc.invalidateQueries({ queryKey: ['food-recent'] });
            qc.invalidateQueries({ queryKey: ['recommendations'] });
          }}
        />
      </SafeAreaView>
    </Screen>
  );

  // ── Personalized (CGM) view — honest, phase-driven, never fake ─────────────
  function renderPersonalized() {
    if (isLoading) {
      return <Card><View style={{ alignItems: 'center', paddingVertical: 28 }}><ActivityIndicator color={c.accent} /><Muted style={{ marginTop: 10 }}>Loading…</Muted></View></Card>;
    }
    // No CGM connected → prompt (never a mock chart).
    if (status === 'no_source' || (status === 'no_data' && raw.length === 0)) {
      return (
        <EmptyState c={c}
          icon="pulse-outline" title="Connect a CGM"
          body="Personalized forecasts need a glucose sensor. Stream Libre/Dexcom via xDRIP+ — your forecast begins learning right away."
          cta="Connect a CGM" onPress={() => flashConnector('cgm')} />
      );
    }
    return (
      <Card>
        {training && <TrainingBanner c={c} phase={training.phase} />}
        {status === 'no_watch' && <WatchGateNotice c={c} watch={ts?.watch} onConnect={() => flashConnector('hc')} />}
        <View style={{ flexDirection: 'row', alignItems: 'flex-end', gap: 10, marginBottom: 8 }}>
          <Body style={{ fontSize: 40, fontWeight: '700' }}>{current != null ? current.toFixed(0) : '—'}</Body>
          <Body style={{ color: c.textMuted, marginBottom: 8 }}>mg/dL now</Body>
          <View style={{ flex: 1 }} />
          {current != null && <Pill label={label(current)} tone={tone(current)} />}
        </View>

        <GlucoseChart raw={raw} predicted={status === 'ready' ? predicted : []} now={now}
          range={range} foodMarkers={foodMarkers} highlight={status === 'ready' ? hl : undefined} width={width} />

        {status === 'ready' && hasForecast ? (
          <>
            <View style={{ flexDirection: 'row', gap: 8, marginTop: 12, alignItems: 'center' }}>
              <Muted>View</Muted>
              {HORIZONS.map((h) => (
                <Pressable key={h} onPress={() => setHorizon(h)}
                  style={{ paddingHorizontal: 12, paddingVertical: 6, borderRadius: 999,
                    backgroundColor: horizon === h ? c.accent : c.surfaceAlt }}>
                  <Body style={{ color: horizon === h ? c.onAccent : c.text, fontSize: 13 }}>+{h}m</Body>
                </Pressable>
              ))}
            </View>
            {predForHorizon != null && (
              <Muted style={{ marginTop: 10 }}>
                Your glucose may reach ~ <Body style={{ color: c.accent, fontWeight: '600' }}>{predForHorizon.toFixed(0)} mg/dL</Body> in {horizon} min
              </Muted>
            )}
          </>
        ) : status === 'no_watch' ? null : (
          <LearningNote c={c} status={status} ts={ts} />
        )}
      </Card>
    );
  }

  // ── Basic (watch) view — watch insights; glucose estimate deferred ─────────
  function renderBasic() {
    if (!hc.connected) {
      return (
        <EmptyState c={c}
          icon="watch-outline" title="Connect your watch"
          body="Basic mode uses your watch (heart rate, steps, sleep) plus your food logs. Connect Health Connect to get started."
          cta="Connect your watch" onPress={() => flashConnector('hc')} />
      );
    }
    return (
      <Card>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <Ionicons name="watch-outline" size={18} color={c.accent} />
          <Body style={{ fontWeight: '600' }}>Watch connected</Body>
        </View>
        <RecentReadings c={c} recent={recent} />
        <Muted style={{ marginTop: 10 }}>
          A watch-only glucose estimate is coming soon — until then, glucose forecasts need a CGM (switch to Personalized).
        </Muted>
      </Card>
    );
  }
}

function RecentReadings({ c, recent }: any) {
  if (!recent?.length) {
    return <Muted style={{ marginTop: 4 }}>Waiting for readings — tap “Sync now” below.</Muted>;
  }
  return (
    <View style={{ marginTop: 8 }}>
      <Muted style={{ marginBottom: 6 }}>Latest readings (live)</Muted>
      {recent.map((r: any, i: number) => {
        const val = r.hr_bpm != null ? `${Math.round(r.hr_bpm)}` : r.spo2_pct != null ? `${Math.round(r.spo2_pct)}` : '—';
        const unit = r.hr_bpm != null ? 'bpm' : r.spo2_pct != null ? '% SpO₂' : '';
        return (
          <View key={i} style={{ flexDirection: 'row', alignItems: 'center', paddingVertical: 5,
            borderTopWidth: i === 0 ? 0 : 1, borderTopColor: c.border }}>
            <Ionicons name="heart" size={14} color={c.accent} style={{ marginRight: 8 }} />
            <Body style={{ fontWeight: '700', minWidth: 36 }}>{val}</Body>
            <Muted style={{ marginLeft: 4 }}>{unit}</Muted>
            <View style={{ flex: 1 }} />
            <Muted>{syncedLabel(r.t)}</Muted>
          </View>
        );
      })}
    </View>
  );
}

// ── Small presentational helpers ─────────────────────────────────────────────

function LearningNote({ c, status, ts }: any) {
  let text = 'Collecting your data…';
  if (status === 'collecting' && ts?.days_need) {
    text = `Learning your patterns — ${Math.floor(ts.days_have ?? 0)}/${Math.round(ts.days_need)} days. Your forecast turns on automatically.`;
  } else if (status === 'warming_up') {
    text = `Warming up — ${ts?.have ?? 0}/${ts?.need ?? 0} readings before the first forecast.`;
  }
  return <Muted style={{ marginTop: 12 }}>{text}</Muted>;
}

function WatchGateNotice({ c, watch, onConnect }: any) {
  const have = watch?.have ?? 0;
  return (
    <View style={{ backgroundColor: c.surfaceAlt, borderRadius: 12, padding: 12, marginBottom: 12 }}>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <Ionicons name="watch-outline" size={18} color={c.accent} />
        <Body style={{ fontWeight: '700', flex: 1 }}>Connect your watch to turn on forecasts</Body>
      </View>
      <Muted>
        Your CGM is streaming — but forecasts need live watch data (heart rate). {have > 0
          ? `Only ${have} recent reading${have === 1 ? '' : 's'} — keep the watch on & syncing.`
          : 'No recent watch data is coming in.'} Until then you’ll see your actual CGM readings only.
      </Muted>
      <Pressable onPress={onConnect}
        style={{ marginTop: 10, backgroundColor: c.accent, borderRadius: 10, paddingVertical: 11, alignItems: 'center' }}>
        <Body style={{ color: c.onAccent, fontWeight: '600' }}>Connect / sync watch</Body>
      </Pressable>
    </View>
  );
}

function TrainingBanner({ c, phase }: any) {
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: c.surfaceAlt,
      borderRadius: 10, paddingVertical: 8, paddingHorizontal: 10, marginBottom: 10 }}>
      <ActivityIndicator size="small" color={c.accent} />
      <Body style={{ flex: 1, fontSize: 13, color: c.textMuted }}>
        🧠 Building your {phase === 'post_cgm' ? 'post-sensor' : 'personalized'} model — your forecast keeps running.
      </Body>
    </View>
  );
}

function EmptyState({ c, icon, title, body, cta, onPress }: any) {
  return (
    <Card>
      <View style={{ alignItems: 'center', paddingVertical: 8 }}>
        <Ionicons name={icon} size={30} color={c.accent} />
        <Body style={{ fontWeight: '700', fontSize: 17, marginTop: 8 }}>{title}</Body>
        <Muted style={{ textAlign: 'center', marginTop: 6, marginBottom: 14 }}>{body}</Muted>
        <Pressable onPress={onPress} style={{ backgroundColor: c.accent, borderRadius: 12, paddingVertical: 12, paddingHorizontal: 22 }}>
          <Body style={{ color: c.onAccent, fontWeight: '600' }}>{cta}</Body>
        </Pressable>
      </View>
    </Card>
  );
}

function ConnRow({ c, dot, title, subtitle, action, onPress, disabled, border, top }: any) {
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', paddingVertical: 10,
      borderTopWidth: top ? 1 : 0, borderTopColor: c.border,
      borderWidth: border ? 1.5 : 0, borderColor: border, borderRadius: border ? 10 : 0,
      paddingHorizontal: border ? 8 : 0 }}>
      <View style={{ width: 9, height: 9, borderRadius: 999, marginRight: 10, backgroundColor: dot }} />
      <View style={{ flex: 1 }}>
        <Body style={{ fontWeight: '600' }}>{title}</Body>
        <Muted>{subtitle}</Muted>
      </View>
      <Pressable onPress={onPress} disabled={disabled}>
        <Body style={{ color: c.accent, fontWeight: '600', opacity: disabled ? 0.5 : 1 }}>{action}</Body>
      </Pressable>
    </View>
  );
}

const ctaGhost = (accent: string) => ({ flex: 1, height: 48, borderRadius: 14, borderWidth: 1, borderColor: accent, alignItems: 'center' as const, justifyContent: 'center' as const });
function tone(g: number) { return g < 70 ? 'warn' : g > 180 ? 'bad' : 'good'; }
function label(g: number) { return g < 70 ? 'Low' : g > 180 ? 'High' : 'In range'; }
function syncedLabel(iso: string): string {
  const mins = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  return hrs < 24 ? `${hrs}h ago` : new Date(iso).toLocaleDateString();
}
