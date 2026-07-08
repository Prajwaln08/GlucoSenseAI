import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { Alert, Linking, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { EditProfileSheet } from '@/components/EditProfileSheet';
import { LogsSection } from '@/components/LogsSection';
import { Body, Button, Card, H1, H2, Muted, Screen, Select } from '@/components/ui';
import { disconnectHealthConnect, getHcStatus, type HcStatus } from '@/health/healthConnect';
import { Api, describeCgm } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import { useColors, useThemePref } from '@/lib/theme';

const THEME_LABELS = { system: 'System', light: 'Light', dark: 'Dark' } as const;

export default function Profile() {
  const c = useColors();
  const qc = useQueryClient();
  const router = useRouter();
  const { signOut, session } = useAuth();
  const { pref, setPref } = useThemePref();
  const { data: p } = useQuery({ queryKey: ['profile'], queryFn: () => Api.getProfile() });
  const [edit, setEdit] = useState(false);
  // Shared with Home + auto-sync — one cache entry, so both screens always agree.
  const { data: hc = { connected: false } as HcStatus } =
    useQuery({ queryKey: ['hc-status'], queryFn: getHcStatus });
  const { data: cgmStat } = useQuery({
    queryKey: ['cgm-status'], queryFn: () => Api.cgmStatus(), refetchInterval: 60_000, retry: 0,
  });
  const cgm = describeCgm(cgmStat);

  const dash = (v: any) => (v === null || v === undefined || v === '' ? '—' : String(v));
  const bp = p?.bp_systolic && p?.bp_diastolic ? `${p.bp_systolic} / ${p.bp_diastolic}` : '—';
  const rows: [string, string][] = [
    ['Name', dash(p?.name)], ['Age', dash(p?.age)], ['Gender', dash(p?.gender)],
    ['Height', p?.height_cm ? `${p.height_cm} cm` : '—'],
    ['Weight', p?.weight_kg ? `${p.weight_kg} kg` : '—'],
    ['BMI', dash(p?.bmi)], ['Blood pressure', bp], ['HbA1c', p?.hba1c ? `${p.hba1c}%` : '—'],
    ['Diabetes', dash(p?.diabetes_type)], ['Conditions', dash(p?.medical_history)],
    ['Medications', dash(p?.medications)],
  ];

  function onDelete() {
    Alert.alert('Delete account', 'This permanently erases your account and all your data. This cannot be undone.',
      [{ text: 'Cancel', style: 'cancel' },
       { text: 'Delete', style: 'destructive', onPress: async () => {
         try { await Api.deleteAccount(); signOut(); }
         catch { Alert.alert('Delete', 'Could not delete right now. Try again later.'); }
       } }]);
  }
  function onSuspend() {
    Alert.alert('Pause account', 'Pausing keeps your data but stops new syncs and coaching. To pause, contact support.',
      [{ text: 'OK' }, { text: 'Email support', onPress: () => Linking.openURL('mailto:support@glucosense.ai?subject=Pause my account') }]);
  }
  async function onDisconnectHc() {
    await disconnectHealthConnect();
    qc.setQueryData(['hc-status'], { connected: false });
    Alert.alert('Health Connect', 'Disconnected. Reconnect anytime from Home → Sync now.');
  }

  return (
    <Screen>
      <SafeAreaView edges={['top']} style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={{ padding: 24, paddingBottom: 48 }}>
          <H1>Profile</H1>
          <Muted style={{ marginBottom: 16 }}>{p?.email ?? session?.email}</Muted>

          <H2>Details</H2>
          <Card>
            {rows.map(([k, v], i) => (
              <View key={k} style={{ flexDirection: 'row', paddingVertical: 10,
                borderTopWidth: i ? 1 : 0, borderTopColor: c.border }}>
                <Muted style={{ flex: 1 }}>{k}</Muted>
                <Body style={{ flex: 1.4, textAlign: 'right' }}>{v}</Body>
              </View>
            ))}
            <Button title="Edit details" variant="ghost" onPress={() => setEdit(true)} style={{ marginTop: 12 }} />
          </Card>

          <H2>Logs</H2>
          <LogsSection />

          <H2>Theme</H2>
          <Card>
            <Select value={THEME_LABELS[pref]} options={['System', 'Light', 'Dark']}
              onChange={(v) => setPref(v.toLowerCase() as any)} />
            <Muted>Choose Light or Dark, or follow your phone with System.</Muted>
          </Card>

          <H2>Connections</H2>
          <Card>
            <Row name="Health Connect"
              status={hc.connected ? 'Connected · watch data' : 'Watch: steps, HR, sleep'}
              dot={hc.connected ? c.inRange : c.textMuted}
              action={hc.connected ? 'Disconnect' : undefined}
              onPress={hc.connected ? onDisconnectHc : undefined} c={c} top={false} />
            <Row name="CGM (Libre / Dexcom)" status={cgm.text}
              dot={cgm.connected ? c.inRange : c.textMuted}
              action={cgm.connected ? 'Manage' : 'Set up'}
              onPress={() => router.push('/cgm')} c={c} top />
          </Card>

          <H2>Account</H2>
          <Card>
            <Button title="Privacy policy" variant="ghost"
              onPress={() => Alert.alert('Privacy policy', 'We store only the health data you log or sync, use it to power your forecasts & coaching, never sell it, and you can export or delete it anytime.')} style={{ marginBottom: 10 }} />
            <Button title="Terms of use" variant="ghost"
              onPress={() => Alert.alert('Terms of use', 'GlucoSense provides educational guidance only — it is not medical advice and does not replace your care team. Use at your own discretion.')} style={{ marginBottom: 10 }} />
            <Button title="Contact us" variant="ghost"
              onPress={() => Linking.openURL('mailto:support@glucosense.ai')} style={{ marginBottom: 10 }} />
            <Button title="Pause account" variant="ghost" onPress={onSuspend} style={{ marginBottom: 10 }} />
            <Button title="Delete account" variant="ghost" onPress={onDelete} style={{ marginBottom: 10 }} />
            <Button title="Sign out" onPress={signOut} />
          </Card>
        </ScrollView>

        <EditProfileSheet visible={edit} profile={p} onClose={() => setEdit(false)}
          onSaved={() => qc.invalidateQueries({ queryKey: ['profile'] })} />
      </SafeAreaView>
    </Screen>
  );
}

function Row({ name, status, dot, action, onPress, c, top }: {
  name: string; status: string; dot: string; action?: string; onPress?: () => void; c: any; top: boolean;
}) {
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', paddingVertical: 12,
      borderTopWidth: top ? 1 : 0, borderTopColor: c.border }}>
      <View style={{ width: 9, height: 9, borderRadius: 999, marginRight: 10, backgroundColor: dot }} />
      <View style={{ flex: 1 }}>
        <Body style={{ fontWeight: '600' }}>{name}</Body>
        <Muted>{status}</Muted>
      </View>
      {action ? (
        <Body onPress={onPress} style={{ color: c.accent, fontWeight: '600' }}>{action}</Body>
      ) : null}
    </View>
  );
}
