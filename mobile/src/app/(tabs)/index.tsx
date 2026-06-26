import { Ionicons } from '@expo/vector-icons';
import { useState } from 'react';
import { Alert, Dimensions, Pressable, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { GlucoseChart } from '@/components/GlucoseChart';
import { Body, Button, Card, H1, H2, Muted, Pill, Screen } from '@/components/ui';
import {
  mockConnections, mockFood, mockRecommendations, mockTimeseries, mockVitals,
} from '@/lib/mock';
import { useColors } from '@/lib/theme';

const HORIZONS = [30, 60, 90, 120];

export default function Home() {
  const c = useColors();
  const [horizon, setHorizon] = useState(60);
  const ts = mockTimeseries;
  const current = ts.raw[ts.raw.length - 1].mgdl;
  const predicted = ts.predicted[ts.predicted.length - 1].mgdl;
  const width = Dimensions.get('window').width - 24 * 2 - 16 * 2;

  return (
    <Screen>
      <SafeAreaView edges={['top']} style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={{ padding: 24, paddingBottom: 40 }}>
          <H1>Hi 👋</H1>
          <Muted style={{ marginBottom: 16 }}>Here's your glucose right now.</Muted>

          {/* A — glucose graph */}
          <Card>
            <View style={{ flexDirection: 'row', alignItems: 'flex-end', gap: 10, marginBottom: 8 }}>
              <Body style={{ fontSize: 40, fontWeight: '700' }}>{current.toFixed(0)}</Body>
              <Body style={{ color: c.textMuted, marginBottom: 8 }}>mg/dL now</Body>
              <View style={{ flex: 1 }} />
              <Pill label={inRangeLabel(current)} tone={inRangeTone(current)} />
            </View>
            <GlucoseChart raw={ts.raw} predicted={ts.predicted} now={ts.now} range={ts.range} width={width} />
            <View style={{ flexDirection: 'row', gap: 8, marginTop: 12, alignItems: 'center' }}>
              <Muted>Forecast</Muted>
              {HORIZONS.map((h) => (
                <Pressable key={h} onPress={() => setHorizon(h)}
                  style={{ paddingHorizontal: 12, paddingVertical: 6, borderRadius: 999,
                    backgroundColor: horizon === h ? c.accent : c.surfaceAlt }}>
                  <Body style={{ color: horizon === h ? c.onAccent : c.text, fontSize: 13 }}>+{h}m</Body>
                </Pressable>
              ))}
            </View>
            <Muted style={{ marginTop: 10 }}>
              Predicted ≈ <Body style={{ color: c.accent, fontWeight: '600' }}>{predicted.toFixed(0)} mg/dL</Body> in {horizon} min
            </Muted>
          </Card>

          {/* B — AI recommendations (hidden when none) */}
          {mockRecommendations.length > 0 && (
            <View style={{ marginBottom: 14 }}>
              <H2>Coach suggestions</H2>
              {mockRecommendations.map((r) => (
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

          {/* C — food & vitals logging CTA (above connections) */}
          <H2>Log</H2>
          <Card>
            <View style={{ flexDirection: 'row', gap: 12 }}>
              <Button title="🍽  Log food" onPress={() => Alert.alert('Log food', 'Quick-log sheet — wired in Phase 2.')} style={{ flex: 1 }} />
              <Button title="❤️  Log vitals" variant="ghost" onPress={() => Alert.alert('Log vitals', 'BP / weight / glucose — wired in Phase 2.')} style={{ flex: 1 }} />
            </View>
            <Muted style={{ marginTop: 12 }}>Recent</Muted>
            {[...mockFood.map((f) => `${f.when} · ${f.meal}: ${f.desc} (${f.carbs}g)`),
              ...mockVitals.map((v) => `${v.when} · ${v.kind}: ${v.value}`)].slice(0, 4).map((line, i) => (
              <Body key={i} style={{ color: c.textMuted, fontSize: 13, marginTop: 6 }}>• {line}</Body>
            ))}
          </Card>

          {/* D — connections */}
          <H2>Connections</H2>
          <Card>
            {mockConnections.map((conn, i) => (
              <View key={conn.id} style={{ flexDirection: 'row', alignItems: 'center', paddingVertical: 10,
                borderTopWidth: i ? 1 : 0, borderTopColor: c.border }}>
                <View style={{ flex: 1 }}>
                  <Body style={{ fontWeight: '600' }}>{conn.name}</Body>
                  <Muted>{conn.status}</Muted>
                </View>
                <Pressable onPress={() => Alert.alert(conn.name, 'Health Connect wired in Phase 3.')}>
                  <Body style={{ color: c.accent, fontWeight: '600' }}>{conn.cta}</Body>
                </Pressable>
              </View>
            ))}
          </Card>
        </ScrollView>
      </SafeAreaView>
    </Screen>
  );
}

function inRangeTone(g: number) { return g < 70 ? 'warn' : g > 180 ? 'bad' : 'good'; }
function inRangeLabel(g: number) { return g < 70 ? 'Low' : g > 180 ? 'High' : 'In range'; }
