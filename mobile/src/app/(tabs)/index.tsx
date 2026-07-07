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
  const [mode, setMode] = useState<Mode>('personalized');
  const [modeHydrated, setModeHydrated] = useState(false);

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
  // Single source of truth for the Health Connect state — Home, Profile and the
  // auto-sync hook all read/update this one cache entry, so screens never disagree.
  const { data: hc = { connected: false } as HcStatus } =
    useQuery({ queryKey: ['hc-status'], queryFn: getHcStatus });
  const firstName = (profile?.first_name || profile?.name?.trim().split(/\s+/)[0] || '').trim();
  useEffect(() => { AsyncStorage.getItem(MODE_KEY).then((v) => { if (v === 'basic' || v === 'personalized') setMode(v); setModeHydrated(true); }); }, []);
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
      qc.setQueryData(['hc-status'], { connected: true, lastSync: new Date().toISOString() });
      qc.invalidateQueries({ queryKey: ['recommendations'] });
      qc.invalidateQueries({ queryKey: ['timeseries'] });        // intraday HR feeds the forecast + watch-gate progress
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
                  {mVal === 'personalized' ? 'Personalized CGM' : 'Virtual CGM'}
                </Body>
              </Pressable>
            ))}
          </View>

          {/* A — the graph / status area (wait for the saved mode to avoid a wrong-mode flash) */}
          {!modeHydrated
            ? <Card><View style={{ alignItems: 'center', paddingVertical: 28 }}><ActivityIndicator color={c.accent} /></View></Card>
            : mode === 'personalized'
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
        {(() => {
          const stale = raw.length > 0 && (Date.now() - now) / 60000 > 20;   // CGM reads ~5min
          return (
            <View style={{ flexDirection: 'row', alignItems: 'flex-end', gap: 10, marginBottom: 8 }}>
              <Body style={{ fontSize: 40, fontWeight: '700', opacity: stale ? 0.5 : 1 }}>{current != null ? current.toFixed(0) : '—'}</Body>
              <Body style={{ color: c.textMuted, marginBottom: 8 }}>
                {stale ? `mg/dL · ${syncedLabel(new Date(now).toISOString())}` : 'mg/dL now'}
              </Body>
              <View style={{ flex: 1 }} />
              {current != null && !stale && <Pill label={label(current)} tone={tone(current)} />}
            </View>
          );
        })()}

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
          body="Virtual CGM mode uses your watch (heart rate, steps, sleep) plus your food logs. Connect Health Connect to get started."
          cta="Connect your watch" onPress={() => flashConnector('hc')} />
      );
    }
    const vPred = (ts?.predicted ?? []).filter((p) => p.horizon_min != null);
    const virtualReady = !!ts?.watch?.ready && vPred.length > 0;
    const vSel = vPred.find((p) => p.horizon_min === horizon) ?? vPred[0];
    return (
      <Card>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 6 }}>
          <Ionicons name="watch-outline" size={18} color={c.accent} />
          <Body style={{ fontWeight: '600' }}>Watch connected</Body>
        </View>
        {virtualReady && vSel ? (
          <>
            {/* Same visual language as Personalized: big value, range pill, chart, chips */}
            <View style={{ flexDirection: 'row', alignItems: 'flex-end', gap: 10, marginBottom: 8 }}>
              <Body style={{ fontSize: 40, fontWeight: '700' }}>~{Math.round(vSel.mgdl)}</Body>
              <Body style={{ color: c.textMuted, marginBottom: 8 }}>mg/dL est. in {vSel.horizon_min} min</Body>
              <View style={{ flex: 1 }} />
              <Pill label={label(vSel.mgdl)} tone={tone(vSel.mgdl)} />
            </View>
            <GlucoseChart raw={[]} predicted={predicted} now={now} range={range}
              foodMarkers={foodMarkers}
              highlight={{ t: Date.parse(vSel.t), mgdl: vSel.mgdl }} width={width} />
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
          </>
        ) : (
          <WatchWaitProgress c={c} watch={ts?.watch} />
        )}
        <Muted style={{ marginTop: 10 }}>
          Virtual CGM estimates your glucose from your watch data using our standard base model. For forecasts tuned to your body, switch to Personalized.
        </Muted>
      </Card>
    );
  }
}

function WatchWaitProgress({ c, watch }: any) {
  // How far the watch gate is from unlocking predictions: needs `need` recent
  // heart-rate readings, flowing without long gaps (see lifecycle watch-gate).
  const have = watch?.have ?? 0;
  const need = watch?.need ?? 8;
  const ready = !!watch?.ready;
  const left = Math.max(0, need - have);

  // Live countdown anchored to SERVER truth: the newest HR reading arrived at
  // watch.last_at, readings come ~every 10 min, so the gate opens around
  // last_at + left×10min. Being derived (not stored client-side), the ETA is
  // identical across refetches, reloads and remounts — it can't "restart".
  const lastAt = watch?.last_at ? Date.parse(watch.last_at) : null;
  const eta = lastAt != null && left > 0 ? lastAt + left * 10 * 60_000 : null;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (ready) return;
    const iv = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(iv);
  }, [ready]);

  if (ready) {
    return (
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 4 }}>
        <Ionicons name="checkmark-circle" size={16} color={c.accent} />
        <Body style={{ fontWeight: '600', flex: 1 }}>
          Watch data is flowing — your Virtual CGM estimate is on its way.
        </Body>
      </View>
    );
  }

  const remain = eta != null ? Math.max(0, eta - now) : 0;
  const h = Math.floor(remain / 3_600_000);
  const m = Math.floor((remain % 3_600_000) / 60_000);
  const s = Math.floor((remain % 60_000) / 1000);
  const clock = `${h > 0 ? `${h}:` : ''}${h > 0 ? String(m).padStart(2, '0') : m}:${String(s).padStart(2, '0')}`;
  const pct = Math.min(1, need ? have / need : 0);
  return (
    <View style={{ marginTop: 4 }}>
      <View style={{ flexDirection: 'row', alignItems: 'baseline', gap: 8 }}>
        <Body style={{ fontWeight: '600', flex: 1 }}>
          {remain > 0 ? 'First prediction in about'
            : eta == null ? 'Waiting for your first heart-rate reading'
            : 'Any moment now'}
        </Body>
        {remain > 0 && (
          <Body style={{ fontWeight: '700', fontSize: 22, color: c.accent, fontVariant: ['tabular-nums'] }}>
            {clock}
          </Body>
        )}
      </View>
      <View style={{ height: 8, borderRadius: 4, backgroundColor: c.surfaceAlt, marginTop: 10, overflow: 'hidden' }}>
        <View style={{ width: `${Math.round(pct * 100)}%`, height: '100%', backgroundColor: c.accent }} />
      </View>
      <Muted style={{ marginTop: 10 }}>
        {remain > 0
          ? `${have} of ${need} readings in — the clock updates as readings arrive.`
          : eta == null
            ? 'Wear your watch and tap “Sync now” — predictions unlock after 8 heart-rate readings.'
            : `${have} of ${need} readings in — waiting for your next sync to finish the set.`}
      </Muted>
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
  } else if (status === 'no_data') {
    text = 'Couldn’t refresh your forecast just now — showing your latest readings.';
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
