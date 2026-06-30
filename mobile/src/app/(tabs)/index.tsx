import { Ionicons } from '@expo/vector-icons';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Alert, Dimensions, Pressable, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { GlucoseChart } from '@/components/GlucoseChart';
import { LogSheet } from '@/components/LogSheet';
import { Body, Card, H1, H2, Muted, Pill, Screen } from '@/components/ui';
import { getHcStatus, syncHealthConnect, type HcStatus } from '@/health/healthConnect';
import { Api } from '@/lib/api';
import { mockTimeseries } from '@/lib/mock';
import { useColors } from '@/lib/theme';

const HORIZONS = [30, 60, 90, 120];

export default function Home() {
  const c = useColors();
  const qc = useQueryClient();
  const router = useRouter();
  const [horizon, setHorizon] = useState(60);
  const [sheet, setSheet] = useState<null | 'food' | 'vitals'>(null);
  const [syncing, setSyncing] = useState(false);
  const [hc, setHc] = useState<HcStatus>({ connected: false });

  const { data, isLoading } = useQuery({ queryKey: ['timeseries'], queryFn: () => Api.timeseries(6) });
  const { data: recs } = useQuery({ queryKey: ['recommendations'], queryFn: () => Api.recommendations() });
  const { data: profile } = useQuery({ queryKey: ['profile'], queryFn: () => Api.getProfile() });
  const { data: foods } = useQuery({ queryKey: ['food-recent'], queryFn: () => Api.foodLogs() });
  const firstName = (profile?.first_name || profile?.name?.trim().split(/\s+/)[0] || '').trim();
  const foodMarkers = (foods ?? [])
    .map((f) => ({ t: Date.parse(f.logged_at) }))
    .filter((m) => Number.isFinite(m.t));

  useEffect(() => { getHcStatus().then(setHc); }, []);

  async function connectHealthConnect() {
    setSyncing(true);
    const res = await syncHealthConnect();
    setSyncing(false);
    if (res.ok) {
      setHc({ connected: true, lastSync: new Date().toISOString() });
      qc.invalidateQueries({ queryKey: ['timeseries'] });
      Alert.alert('Health Connect', `Synced ${res.cgm_inserted ?? 0} glucose readings and ${res.activity_days ?? 0} day(s) of activity.`);
    } else {
      Alert.alert('Health Connect', res.message ?? 'Could not sync.');
    }
  }

  // Real data when available; mock keeps the screen alive while loading / before any CGM.
  const hasReal = !!data && data.raw.length > 0;
  const raw = hasReal ? data!.raw.map((p) => ({ t: Date.parse(p.t), mgdl: p.mgdl })) : mockTimeseries.raw;
  const predicted = hasReal ? data!.predicted.map((p) => ({ t: Date.parse(p.t), mgdl: p.mgdl })) : mockTimeseries.predicted;
  const now = hasReal && data!.now ? Date.parse(data!.now) : mockTimeseries.now;
  const range = data?.range ?? mockTimeseries.range;
  const current = raw[raw.length - 1].mgdl;
  // the forecast point at the selected horizon — highlighted on the chart + shown in the sentence
  const targetT = now + horizon * 60_000;
  const highlight = predicted.length
    ? predicted.reduce((b, p) => (Math.abs(p.t - targetT) < Math.abs(b.t - targetT) ? p : b))
    : undefined;
  const predForHorizon = highlight?.mgdl ?? predicted[predicted.length - 1]?.mgdl ?? current;
  const width = Dimensions.get('window').width - 24 * 2 - 16 * 2;

  return (
    <Screen>
      <SafeAreaView edges={['top']} style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={{ padding: 24, paddingBottom: 40 }}>
          <H1>{firstName ? `Hi ${firstName} 👋` : 'Hi 👋'}</H1>
          <Muted style={{ marginBottom: 16 }}>
            {isLoading ? 'Loading your glucose…' : hasReal ? "Here's your glucose right now." : 'Showing a sample until a CGM is connected.'}
          </Muted>

          {/* A — glucose graph */}
          <Card>
            <View style={{ flexDirection: 'row', alignItems: 'flex-end', gap: 10, marginBottom: 8 }}>
              <Body style={{ fontSize: 40, fontWeight: '700' }}>{current.toFixed(0)}</Body>
              <Body style={{ color: c.textMuted, marginBottom: 8 }}>mg/dL now</Body>
              <View style={{ flex: 1 }} />
              <Pill label={label(current)} tone={tone(current)} />
            </View>
            <GlucoseChart raw={raw} predicted={predicted} now={now} range={range} foodMarkers={foodMarkers} highlight={highlight} width={width} />
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
            <Muted style={{ marginTop: 10 }}>
              Your glucose may reach ~ <Body style={{ color: c.accent, fontWeight: '600' }}>{predForHorizon.toFixed(0)} mg/dL</Body> in {horizon} min
            </Muted>
          </Card>

          {/* B — coach recommendations (hidden when none) */}
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

          {/* C — food & vitals logging (above connections) */}
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
            <Muted style={{ marginTop: 12 }}>Logged items affect your forecast in real time.</Muted>
          </Card>

          {/* D — connections */}
          <H2>Connections</H2>
          <Card>
            <View style={{ flexDirection: 'row', alignItems: 'center', paddingVertical: 10 }}>
              <View style={{ width: 9, height: 9, borderRadius: 999, marginRight: 10,
                backgroundColor: hc.connected ? c.inRange : c.textMuted }} />
              <View style={{ flex: 1 }}>
                <Body style={{ fontWeight: '600' }}>Health Connect</Body>
                <Muted>
                  {syncing ? 'Syncing…'
                    : hc.connected ? `Connected${hc.lastSync ? ` · synced ${syncedLabel(hc.lastSync)}` : ''}`
                    : 'Not connected'}
                </Muted>
              </View>
              <Pressable onPress={connectHealthConnect} disabled={syncing}>
                <Body style={{ color: c.accent, fontWeight: '600', opacity: syncing ? 0.5 : 1 }}>
                  {syncing ? 'Syncing…' : hc.connected ? 'Sync' : 'Sync now'}
                </Body>
              </Pressable>
            </View>
            <View style={{ flexDirection: 'row', alignItems: 'center', paddingVertical: 10, borderTopWidth: 1, borderTopColor: c.border }}>
              <View style={{ width: 9, height: 9, borderRadius: 999, marginRight: 10, backgroundColor: c.textMuted }} />
              <View style={{ flex: 1 }}>
                <Body style={{ fontWeight: '600' }}>CGM (Libre / Dexcom)</Body>
                <Muted>Stream with xDRIP+ · Junction soon</Muted>
              </View>
              <Pressable onPress={() => router.push('/cgm')}>
                <Body style={{ color: c.accent, fontWeight: '600' }}>Set up</Body>
              </Pressable>
            </View>
          </Card>
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
