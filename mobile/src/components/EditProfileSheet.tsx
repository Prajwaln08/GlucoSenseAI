/** Bottom-sheet to edit profile details → PUT /me/profile. */
import { useEffect, useState } from 'react';
import { KeyboardAvoidingView, Modal, Platform, Pressable, ScrollView, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Api, type Profile } from '@/lib/api';
import { CONDITIONS, DIABETES, GENDERS } from '@/lib/constants';
import { useColors } from '@/lib/theme';
import { Body, Button, Field, H2, Muted, MultiSelect, Select } from './ui';

const inRange = (n: number, lo: number, hi: number) => Number.isFinite(n) && n >= lo && n <= hi;

export function EditProfileSheet({
  visible, profile, onClose, onSaved,
}: { visible: boolean; profile?: Profile; onClose: () => void; onSaved: () => void }) {
  const c = useColors();
  const insets = useSafeAreaInsets();
  const [f, setF] = useState({ firstName: '', lastName: '', heightCm: '', weightKg: '', hba1c: '', medications: '' });
  const [gender, setGender] = useState('');
  const [diabetes, setDiabetes] = useState('None');
  const [conditions, setConditions] = useState<string[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  // (re)load from the profile each time the sheet opens
  useEffect(() => {
    if (!visible || !profile) return;
    setF({
      firstName: profile.first_name ?? (profile.name?.split(/\s+/)[0] ?? ''),
      lastName: profile.last_name ?? (profile.name?.split(/\s+/).slice(1).join(' ') ?? ''),
      heightCm: profile.height_cm != null ? String(profile.height_cm) : '',
      weightKg: profile.weight_kg != null ? String(profile.weight_kg) : '',
      hba1c: profile.hba1c != null ? String(profile.hba1c) : '',
      medications: profile.medications ?? '',
    });
    setGender(profile.gender ?? '');
    setDiabetes(profile.diabetes_type ?? 'None');
    setConditions((profile.medical_history ?? '').split(',').map((s) => s.trim()).filter((s) => CONDITIONS.includes(s)));
    setErrors({});
  }, [visible, profile]);

  const set = (k: keyof typeof f) => (v: string) => { setF((s) => ({ ...s, [k]: v })); setErrors((e) => ({ ...e, [k]: '' })); };
  const dec = (k: keyof typeof f) => (v: string) => set(k)(v.replace(/[^0-9.]/g, ''));

  async function save() {
    const e: Record<string, string> = {};
    if (!f.firstName.trim()) e.firstName = 'Required';
    if (f.heightCm && !inRange(Number(f.heightCm), 50, 250)) e.heightCm = 'Height must be 50–250 cm';
    if (f.weightKg && !inRange(Number(f.weightKg), 20, 400)) e.weightKg = 'Weight must be 20–400 kg';
    if (f.hba1c && !inRange(Number(f.hba1c), 3, 20)) e.hba1c = 'HbA1c must be 3–20%';
    setErrors(e);
    if (Object.keys(e).length) return;

    setSaving(true);
    try {
      const num = (s: string) => (s.trim() ? Number(s) : undefined);
      await Api.updateProfile({
        first_name: f.firstName.trim(), last_name: f.lastName.trim() || undefined,
        gender: gender || undefined,
        height_cm: num(f.heightCm), weight_kg: num(f.weightKg), hba1c: num(f.hba1c),
        diabetes_type: diabetes,
        medical_history: conditions.length ? conditions.join(', ') : undefined,
        medications: f.medications.trim() || undefined,
      });
      onSaved();
      onClose();
    } catch {
      setErrors({ form: 'Could not save. Try again.' });
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <Pressable onPress={onClose} style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'flex-end' }}>
          <Pressable onPress={() => {}} style={{ backgroundColor: c.bg, borderTopLeftRadius: 22, borderTopRightRadius: 22, maxHeight: '90%' }}>
            <ScrollView keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}
              contentContainerStyle={{ padding: 24, paddingBottom: 28 + insets.bottom }}>
              <H2>Edit details</H2>
              <View style={{ flexDirection: 'row', gap: 12 }}>
                <View style={{ flex: 1 }}><Field label="First name" value={f.firstName} onChangeText={set('firstName')} error={errors.firstName} /></View>
                <View style={{ flex: 1 }}><Field label="Last name" value={f.lastName} onChangeText={set('lastName')} /></View>
              </View>
              <Select label="Gender" value={gender} options={GENDERS} onChange={setGender} placeholder="Select" />
              <View style={{ flexDirection: 'row', gap: 12 }}>
                <View style={{ flex: 1 }}><Field label="Height (cm)" value={f.heightCm} onChangeText={dec('heightCm')} keyboardType="number-pad" error={errors.heightCm} /></View>
                <View style={{ flex: 1 }}><Field label="Weight (kg)" value={f.weightKg} onChangeText={dec('weightKg')} keyboardType="decimal-pad" error={errors.weightKg} /></View>
              </View>
              <Field label="HbA1c (%)" value={f.hba1c} onChangeText={dec('hba1c')} keyboardType="decimal-pad" error={errors.hba1c} />
              <Select label="Diabetes type" value={diabetes} options={DIABETES} onChange={setDiabetes} />
              <MultiSelect label="Medical conditions" values={conditions} options={CONDITIONS} onChange={setConditions} />
              <Field label="Medications" value={f.medications} onChangeText={set('medications')} placeholder="e.g. Metformin 500mg" />

              {errors.form ? <Muted style={{ color: c.hyper, marginBottom: 8 }}>{errors.form}</Muted> : null}
              <Body style={{ color: c.textMuted, fontSize: 12, marginBottom: 10 }}>BMI updates automatically from height + weight.</Body>
              <Button title="Save" onPress={save} loading={saving} />
              <Button title="Cancel" variant="ghost" onPress={onClose} style={{ marginTop: 8 }} />
            </ScrollView>
          </Pressable>
        </Pressable>
      </KeyboardAvoidingView>
    </Modal>
  );
}
