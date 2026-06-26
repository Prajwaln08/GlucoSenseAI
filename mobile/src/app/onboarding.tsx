import { useState } from 'react';
import { ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Body, Button, Field, H1, Muted, Pill, Screen } from '@/components/ui';
import { useAuth } from '@/lib/auth';
import { useColors } from '@/lib/theme';

const DIABETES = ['None', 'Type 1', 'Type 2', 'Gestational', 'Prediabetes'];

export default function Onboarding() {
  const { completeOnboarding } = useAuth();
  const c = useColors();
  const [f, setF] = useState({
    name: '', age: '', gender: '', heightCm: '', weightKg: '',
    bpSys: '', bpDia: '', bpDate: '', hba1c: '', conditions: '', medications: '',
  });
  const [diabetes, setDiabetes] = useState('None');
  const [saving, setSaving] = useState(false);
  const set = (k: keyof typeof f) => (v: string) => setF((s) => ({ ...s, [k]: v }));

  async function finish() {
    setSaving(true);
    try {
      // Phase 2: PUT /me/profile. For now just complete onboarding locally.
      await completeOnboarding();
    } finally {
      setSaving(false);
    }
  }

  return (
    <Screen>
      <SafeAreaView style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={{ padding: 24 }} keyboardShouldPersistTaps="handled">
          <H1>Tell us about you</H1>
          <Muted style={{ marginTop: 4, marginBottom: 20 }}>
            This helps personalise your forecasts and coaching. You can edit it anytime in Profile.
          </Muted>

          <Field label="Full name" value={f.name} onChangeText={set('name')} placeholder="Your name" />
          <View style={{ flexDirection: 'row', gap: 12 }}>
            <View style={{ flex: 1 }}>
              <Field label="Age" value={f.age} onChangeText={set('age')} keyboardType="number-pad" placeholder="45" />
            </View>
            <View style={{ flex: 1 }}>
              <Field label="Gender" value={f.gender} onChangeText={set('gender')} placeholder="F / M / —" />
            </View>
          </View>
          <View style={{ flexDirection: 'row', gap: 12 }}>
            <View style={{ flex: 1 }}>
              <Field label="Height (cm)" value={f.heightCm} onChangeText={set('heightCm')} keyboardType="number-pad" placeholder="168" />
            </View>
            <View style={{ flex: 1 }}>
              <Field label="Weight (kg)" value={f.weightKg} onChangeText={set('weightKg')} keyboardType="decimal-pad" placeholder="72" />
            </View>
          </View>

          <Body style={{ fontWeight: '600', marginBottom: 8 }}>Blood pressure (last recorded)</Body>
          <View style={{ flexDirection: 'row', gap: 12 }}>
            <View style={{ flex: 1 }}>
              <Field label="Systolic" value={f.bpSys} onChangeText={set('bpSys')} keyboardType="number-pad" placeholder="128" />
            </View>
            <View style={{ flex: 1 }}>
              <Field label="Diastolic" value={f.bpDia} onChangeText={set('bpDia')} keyboardType="number-pad" placeholder="84" />
            </View>
            <View style={{ flex: 1 }}>
              <Field label="Date" value={f.bpDate} onChangeText={set('bpDate')} placeholder="2026-06-26" />
            </View>
          </View>

          <Field label="HbA1c (%)" value={f.hba1c} onChangeText={set('hba1c')} keyboardType="decimal-pad" placeholder="6.1" />

          <Body style={{ fontWeight: '600', marginBottom: 8 }}>Diabetes type</Body>
          <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
            {DIABETES.map((d) => (
              <View key={d}>
                <Body
                  onPress={() => setDiabetes(d)}
                  style={{
                    paddingHorizontal: 14, paddingVertical: 8, borderRadius: 999, overflow: 'hidden',
                    color: diabetes === d ? c.onAccent : c.text,
                    backgroundColor: diabetes === d ? c.accent : c.surfaceAlt,
                    fontSize: 14,
                  }}
                >
                  {d}
                </Body>
              </View>
            ))}
          </View>

          <Field label="Medical conditions (optional)" value={f.conditions} onChangeText={set('conditions')} placeholder="e.g. Hypertension" />
          <Field label="Medications (optional)" value={f.medications} onChangeText={set('medications')} placeholder="e.g. Metformin 500mg" />

          <Button title="Finish setup" onPress={finish} loading={saving} style={{ marginTop: 8 }} />
        </ScrollView>
      </SafeAreaView>
    </Screen>
  );
}
