import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { Alert, Linking, Pressable, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Body, Button, Card, H2, Muted, Screen } from '@/components/ui';
import { API_URL, Api } from '@/lib/api';
import { useColors } from '@/lib/theme';

export default function Cgm() {
  const c = useColors();
  const router = useRouter();
  const [busy, setBusy] = useState<string | null>(null);
  const [xdripUrl, setXdripUrl] = useState<string | null>(null);

  async function connectJunction() {
    setBusy('junction');
    try {
      const { link_web_url } = await Api.junctionLink();
      await Linking.openURL(link_web_url);
    } catch {
      Alert.alert('Junction', 'Junction isn’t configured on this server yet. You can use xDRIP+ below in the meantime.');
    } finally { setBusy(null); }
  }
  async function syncJunction() {
    setBusy('jsync');
    try {
      const r = await Api.junctionSync();
      Alert.alert('Junction', r.message);
    } catch (e: any) {
      Alert.alert('Junction', e?.response?.data?.detail ?? 'Connect a device with Junction first.');
    } finally { setBusy(null); }
  }
  async function getXdripUrl() {
    setBusy('xdrip');
    try {
      const { push_url } = await Api.cgmKey();
      setXdripUrl(push_url.startsWith('http') ? push_url : `${API_URL}${push_url}`);
    } catch {
      Alert.alert('xDRIP+', 'Could not generate your push URL. Try again.');
    } finally { setBusy(null); }
  }

  return (
    <Screen>
      <SafeAreaView edges={['top']} style={{ flex: 1 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 12, gap: 6 }}>
          <Pressable onPress={() => router.back()} hitSlop={10}>
            <Ionicons name="chevron-back" size={26} color={c.text} />
          </Pressable>
          <Body style={{ fontWeight: '700', fontSize: 18 }}>Connect a CGM</Body>
        </View>

        <ScrollView contentContainerStyle={{ padding: 24, paddingTop: 6, paddingBottom: 48 }}>
          <Muted style={{ marginBottom: 18 }}>
            Stream your continuous glucose into GlucoSense — from a Libre/Dexcom sensor via Junction,
            or directly from the xDRIP+ app.
          </Muted>

          <H2>Junction</H2>
          <Card>
            <Body style={{ fontWeight: '600', marginBottom: 4 }}>Libre / Dexcom</Body>
            <Muted style={{ marginBottom: 14 }}>Link your sensor through Junction, then pull your readings.</Muted>
            <Button title="Connect with Junction" onPress={connectJunction} loading={busy === 'junction'} />
            <Button title="Sync now" variant="ghost" onPress={syncJunction} loading={busy === 'jsync'} style={{ marginTop: 8 }} />
          </Card>

          <H2>xDRIP+</H2>
          <Card>
            <Body style={{ fontWeight: '600', marginBottom: 4 }}>Stream from xDRIP+</Body>
            <Muted style={{ marginBottom: 14 }}>Push readings straight from the xDRIP+ app on your phone — no extra account.</Muted>
            {!xdripUrl ? (
              <Button title="Get my push URL" onPress={getXdripUrl} loading={busy === 'xdrip'} />
            ) : (
              <>
                <Muted style={{ fontWeight: '700', marginBottom: 6 }}>Your private push URL</Muted>
                <View style={{ backgroundColor: c.surfaceAlt, borderRadius: 10, padding: 12, marginBottom: 12 }}>
                  <Body selectable style={{ fontSize: 12 }}>{xdripUrl}</Body>
                </View>
                <Muted>
                  In xDRIP+ → Settings → Cloud Upload → set the upload target to this URL (long-press above to copy).
                  Keep it private — it lets your phone push glucose to your account.
                </Muted>
                <Button title="Regenerate URL" variant="ghost" onPress={getXdripUrl} loading={busy === 'xdrip'} style={{ marginTop: 12 }} />
              </>
            )}
          </Card>
        </ScrollView>
      </SafeAreaView>
    </Screen>
  );
}
