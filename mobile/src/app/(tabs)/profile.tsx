import { useQuery } from '@tanstack/react-query';
import { Alert, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Body, Button, Card, H1, H2, Muted, Screen } from '@/components/ui';
import { Api } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import { useColors } from '@/lib/theme';

export default function Profile() {
  const c = useColors();
  const { signOut, session } = useAuth();
  const { data: p } = useQuery({ queryKey: ['profile'], queryFn: () => Api.getProfile() });

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

  return (
    <Screen>
      <SafeAreaView edges={['top']} style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={{ padding: 24, paddingBottom: 40 }}>
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
            <Button title="Edit details" variant="ghost"
              onPress={() => Alert.alert('Edit', 'Inline profile editing comes in a later pass.')} style={{ marginTop: 12 }} />
          </Card>

          <H2>Appearance</H2>
          <Card><Muted>Follows your system light / dark setting.</Muted></Card>

          <H2>Account</H2>
          <Card>
            <Button title="Export my data" variant="ghost"
              onPress={() => Alert.alert('Export', 'Wired in a later pass.')} style={{ marginBottom: 10 }} />
            <Button title="Delete account" variant="ghost"
              onPress={() => Alert.alert('Delete', 'Wired in a later pass.')} style={{ marginBottom: 10 }} />
            <Button title="Sign out" onPress={signOut} />
          </Card>
        </ScrollView>
      </SafeAreaView>
    </Screen>
  );
}
