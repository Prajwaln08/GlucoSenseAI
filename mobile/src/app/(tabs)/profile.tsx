import { Alert, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Body, Button, Card, H1, H2, Muted, Screen } from '@/components/ui';
import { useAuth } from '@/lib/auth';
import { mockProfile } from '@/lib/mock';
import { useColors } from '@/lib/theme';

export default function Profile() {
  const c = useColors();
  const { signOut, session } = useAuth();
  const p = mockProfile;

  const rows: [string, string][] = [
    ['Name', p.name], ['Age', String(p.age)], ['Gender', p.gender],
    ['Height', `${p.heightCm} cm`], ['Weight', `${p.weightKg} kg`],
    ['Blood pressure', `${p.bp} (${p.bpRecorded})`], ['HbA1c', `${p.hba1c}%`],
    ['Diabetes', p.diabetesType], ['Conditions', p.conditions], ['Medications', p.medications],
  ];

  return (
    <Screen>
      <SafeAreaView edges={['top']} style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={{ padding: 24, paddingBottom: 40 }}>
          <H1>Profile</H1>
          <Muted style={{ marginBottom: 16 }}>{session?.email}</Muted>

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
              onPress={() => Alert.alert('Edit', 'Profile editing wired in Phase 2.')} style={{ marginTop: 12 }} />
          </Card>

          <H2>Appearance</H2>
          <Card><Muted>Follows your system light / dark setting. A manual toggle comes in a later phase.</Muted></Card>

          <H2>Account</H2>
          <Card>
            <Button title="Export my data" variant="ghost"
              onPress={() => Alert.alert('Export', 'Wired in Phase 2.')} style={{ marginBottom: 10 }} />
            <Button title="Delete account" variant="ghost"
              onPress={() => Alert.alert('Delete', 'Wired in Phase 2.')} style={{ marginBottom: 10 }} />
            <Button title="Sign out" onPress={signOut} />
          </Card>
        </ScrollView>
      </SafeAreaView>
    </Screen>
  );
}
