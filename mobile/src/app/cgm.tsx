import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { Alert, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Body, Button, Card, H2, Muted, Pill, Screen } from '@/components/ui';
import { API_URL, Api } from '@/lib/api';
import { useColors } from '@/lib/theme';

export default function Cgm() {
  const c = useColors();
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [xdripUrl, setXdripUrl] = useState<string | null>(null);

  async function getXdripUrl() {
    setBusy(true);
    try {
      const { push_url } = await Api.cgmKey();
      setXdripUrl(push_url.startsWith('http') ? push_url : `${API_URL}${push_url}`);
    } catch {
      Alert.alert('xDRIP+', 'Could not create your link right now. Please try again.');
    } finally { setBusy(false); }
  }

  return (
    <Screen>
      <SafeAreaView edges={['top']} style={{ flex: 1 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 12, gap: 6 }}>
          <Ionicons name="chevron-back" size={26} color={c.text} onPress={() => router.back()} />
          <Body style={{ fontWeight: '700', fontSize: 18 }}>Connect your glucose</Body>
        </View>

        <ScrollView contentContainerStyle={{ padding: 24, paddingTop: 6, paddingBottom: 48 }}>
          <Muted style={{ marginBottom: 18 }}>
            Connect a continuous glucose monitor so GlucoSense can show your live readings and forecasts.
          </Muted>

          {/* xDRIP+ — the working path */}
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <H2>xDRIP+</H2>
            <View style={{ marginTop: -8 }}><Pill label="Recommended" tone="good" /></View>
          </View>
          <Card>
            <Body style={{ fontWeight: '600', marginBottom: 4 }}>Stream your sensor automatically</Body>
            <Muted style={{ marginBottom: 14 }}>
              xDRIP+ is a free Android app that reads your CGM sensor. Connect it once and your glucose
              flows into GlucoSense on its own — no daily effort.
            </Muted>

            {!xdripUrl ? (
              <Button title="Get my secure link" onPress={getXdripUrl} loading={busy} />
            ) : (
              <>
                <Step n={1} text="Make sure the xDRIP+ app is installed on this phone." c={c} />
                <Step n={2} text="Open xDRIP+ → tap the menu (☰) → Settings → Cloud Upload." c={c} />
                <Step n={3} text="Turn on REST-API / web upload and paste the link below as the upload address:" c={c} />
                <View style={{ backgroundColor: c.surfaceAlt, borderRadius: 10, padding: 12, marginVertical: 10, marginLeft: 34 }}>
                  <Body selectable style={{ fontSize: 12 }}>{xdripUrl}</Body>
                  <Muted style={{ marginTop: 6, fontSize: 11 }}>Long-press to copy. Keep it private — it links to your account.</Muted>
                </View>
                <Step n={4} text="Save. Your readings will appear here within a few minutes." c={c} last />
                <Button title="Create a new link" variant="ghost" onPress={getXdripUrl} loading={busy} style={{ marginTop: 14 }} />
              </>
            )}
          </Card>

          {/* Junction — one-tap account linking; enabled when a production connector exists */}
          <H2>Junction (Libre / Dexcom)</H2>
          <Card>
            <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
              <Body style={{ fontWeight: '600' }}>Direct account connect</Body>
              <Pill label="Coming soon" tone="muted" />
            </View>
            <Muted style={{ marginTop: 6 }}>
              One-tap Libre/Dexcom account linking is on the way. For now, xDRIP+ above streams
              your real sensor — free.
            </Muted>
          </Card>
        </ScrollView>
      </SafeAreaView>
    </Screen>
  );
}

function Step({ n, text, c, last }: { n: number; text: string; c: any; last?: boolean }) {
  return (
    <View style={{ flexDirection: 'row', gap: 12, marginBottom: last ? 0 : 12 }}>
      <View style={{ width: 22, height: 22, borderRadius: 11, backgroundColor: c.accent, alignItems: 'center', justifyContent: 'center' }}>
        <Body style={{ color: c.onAccent, fontSize: 12, fontWeight: '700' }}>{n}</Body>
      </View>
      <Body style={{ flex: 1 }}>{text}</Body>
    </View>
  );
}
