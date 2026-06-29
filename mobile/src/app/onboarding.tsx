import { useState } from 'react';
import { Pressable, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Body, Button, Field, H1, MultiSelect, Muted, Screen, Select } from '@/components/ui';
import { Api } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import { useColors } from '@/lib/theme';

const DIABETES = ['None', 'Type 1', 'Type 2', 'Gestational', 'Prediabetes'];
const GENDERS = ['Male', 'Female', 'Others'];
const CONDITIONS = [
  'Asthma', 'Cancer', 'Diabetes', 'Heart Disease', 'High Cholesterol',
  'Hypertension', 'Kidney Disease', 'Obesity', 'Thyroid Disorder', 'Others',
];
const THIS_YEAR = new Date().getFullYear();

const inRange = (n: number, lo: number, hi: number) => Number.isFinite(n) && n >= lo && n <= hi;

/** Real calendar date or null (rejects e.g. 31 Feb). */
function makeDate(d: number, m: number, y: number): Date | null {
  if (!(d >= 1 && d <= 31 && m >= 1 && m <= 12 && y >= 1900)) return null;
  const dt = new Date(y, m - 1, d);
  return dt.getFullYear() === y && dt.getMonth() === m - 1 && dt.getDate() === d ? dt : null;
}
function ageFrom(dt: Date): number {
  const n = new Date();
  let a = n.getFullYear() - dt.getFullYear();
  if (n.getMonth() < dt.getMonth() || (n.getMonth() === dt.getMonth() && n.getDate() < dt.getDate())) a -= 1;
  return a;
}
function maskDate(v: string): string {
  const d = v.replace(/\D/g, '').slice(0, 8);
  let out = d.slice(0, 2);
  if (d.length > 2) out += '/' + d.slice(2, 4);
  if (d.length > 4) out += '/' + d.slice(4, 8);
  return out;
}

export default function Onboarding() {
  const { completeOnboarding } = useAuth();
  const c = useColors();
  const [f, setF] = useState({
    firstName: '', lastName: '', heightCm: '', weightKg: '',
    bpSys: '', bpDia: '', bpDate: '', hba1c: '', medications: '',
    dobD: '', dobM: '', dobY: '',
  });
  const [gender, setGender] = useState('');
  const [diabetes, setDiabetes] = useState('None');
  const [conditions, setConditions] = useState<string[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  const clearErr = (k: string) => setErrors((e) => { if (!e[k]) return e; const n = { ...e }; delete n[k]; return n; });
  // free text — clears its own error on edit
  const onText = (k: keyof typeof f) => (v: string) => { setF((s) => ({ ...s, [k]: v })); clearErr(k); };
  // integer-only, optional max cap
  const onInt = (k: keyof typeof f, max?: number) => (v: string) => {
    let d = v.replace(/\D/g, '');
    if (max !== undefined && d !== '' && Number(d) > max) d = String(max);
    setF((s) => ({ ...s, [k]: d })); clearErr(k);
  };
  // decimal (one dot)
  const onDec = (k: keyof typeof f) => (v: string) => {
    let x = v.replace(/[^0-9.]/g, ''); const p = x.split('.');
    if (p.length > 2) x = p[0] + '.' + p.slice(1).join('');
    setF((s) => ({ ...s, [k]: x })); clearErr(k);
  };
  // DOB parts cap to possible values; clear the shared 'dob' error
  const onDob = (k: 'dobD' | 'dobM' | 'dobY', max: number) => (v: string) => {
    let d = v.replace(/\D/g, ''); if (d !== '' && Number(d) > max) d = String(max);
    setF((s) => ({ ...s, [k]: d })); clearErr('dob');
  };
  const onBpDate = (v: string) => { setF((s) => ({ ...s, bpDate: maskDate(v) })); clearErr('bpDate'); };

  function validate(): Record<string, string> {
    const e: Record<string, string> = {};
    if (!f.firstName.trim()) e.firstName = 'Enter your first name';
    if (!f.lastName.trim()) e.lastName = 'Enter your last name';
    if (!gender) e.gender = 'Select your gender';

    if (!f.dobD || !f.dobM || !f.dobY) e.dob = 'Enter your date of birth';
    else {
      const dob = makeDate(Number(f.dobD), Number(f.dobM), Number(f.dobY));
      if (!dob) e.dob = 'That date doesn’t exist';
      else if (dob > new Date()) e.dob = 'Date can’t be in the future';
      else {
        const age = ageFrom(dob);
        if (age < 20 || age > 80) e.dob = 'GlucoSense is for ages 20–80 only';
      }
    }

    if (!f.heightCm.trim()) e.heightCm = 'Required';
    else if (!inRange(Number(f.heightCm), 50, 250)) e.heightCm = 'Height must be 50–250 cm';

    if (!f.weightKg.trim()) e.weightKg = 'Required';
    else if (!inRange(Number(f.weightKg), 20, 400)) e.weightKg = 'Weight must be 20–400 kg';

    if (!f.bpSys.trim()) e.bpSys = 'Required';
    else if (!inRange(Number(f.bpSys), 70, 250)) e.bpSys = '70–250';
    if (!f.bpDia.trim()) e.bpDia = 'Required';
    else if (!inRange(Number(f.bpDia), 40, 150)) e.bpDia = '40–150';

    if (!f.bpDate.trim()) e.bpDate = 'Required';
    else {
      const [d, m, y] = f.bpDate.split('/');
      const bd = makeDate(Number(d), Number(m), Number(y));
      if (!bd || (y || '').length !== 4) e.bpDate = 'Use dd/mm/yyyy';
      else if (bd > new Date()) e.bpDate = 'Can’t be in the future';
    }

    if (!f.hba1c.trim()) e.hba1c = 'Required';
    else if (!inRange(Number(f.hba1c), 3, 20)) e.hba1c = 'HbA1c must be 3–20%';
    return e;
  }

  async function finish() {
    const e = validate();
    setErrors(e);
    if (Object.keys(e).length) return;

    setSaving(true);
    try {
      const iso = `${f.dobY}-${f.dobM.padStart(2, '0')}-${f.dobD.padStart(2, '0')}`;  // YYYY-MM-DD
      await Api.updateProfile({
        first_name: f.firstName.trim(),
        last_name: f.lastName.trim(),
        date_of_birth: iso,                 // backend derives name + age
        gender,
        height_cm: Number(f.heightCm), weight_kg: Number(f.weightKg),
        bp_systolic: Number(f.bpSys), bp_diastolic: Number(f.bpDia),
        hba1c: Number(f.hba1c), diabetes_type: diabetes,
        medical_history: conditions.length ? conditions.join(', ') : undefined,
        medications: f.medications.trim() || undefined,
      });
      completeOnboarding();
    } catch {
      setErrors((s) => ({ ...s, submit: 'Could not save. Check your connection and try again.' }));
    } finally {
      setSaving(false);
    }
  }

  const hasErrors = Object.keys(errors).filter((k) => k !== 'submit').length > 0;

  return (
    <Screen>
      <SafeAreaView style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={{ padding: 24, paddingBottom: 56 }} keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
          <H1>Tell us about you</H1>
          <Muted style={{ marginTop: 4, marginBottom: 20 }}>
            This helps personalise your forecasts and coaching. You can edit it anytime in Profile.
          </Muted>

          <View style={{ flexDirection: 'row', gap: 12 }}>
            <View style={{ flex: 1 }}>
              <Field label="First name" value={f.firstName} onChangeText={onText('firstName')} placeholder="First" error={errors.firstName} />
            </View>
            <View style={{ flex: 1 }}>
              <Field label="Last name" value={f.lastName} onChangeText={onText('lastName')} placeholder="Last" error={errors.lastName} />
            </View>
          </View>

          <Select label="Gender" value={gender} options={GENDERS} onChange={(v) => { setGender(v); clearErr('gender'); }} placeholder="Select" error={errors.gender} />

          <Body style={{ fontWeight: '600', color: errors.dob ? c.hyper : c.text }}>Date of birth</Body>
          <Muted style={{ marginBottom: 8, marginTop: 2 }}>GlucoSense is designed for ages 20–80.</Muted>
          <View style={{ flexDirection: 'row', gap: 12 }}>
            <View style={{ flex: 1 }}><Field label="DD" value={f.dobD} onChangeText={onDob('dobD', 31)} keyboardType="number-pad" maxLength={2} placeholder="14" invalid={!!errors.dob} /></View>
            <View style={{ flex: 1 }}><Field label="MM" value={f.dobM} onChangeText={onDob('dobM', 12)} keyboardType="number-pad" maxLength={2} placeholder="06" invalid={!!errors.dob} /></View>
            <View style={{ flex: 1.4 }}><Field label="YYYY" value={f.dobY} onChangeText={onDob('dobY', THIS_YEAR)} keyboardType="number-pad" maxLength={4} placeholder="1985" invalid={!!errors.dob} /></View>
          </View>
          {errors.dob ? <Muted style={{ color: c.hyper, marginTop: -8, marginBottom: 12 }}>{errors.dob}</Muted> : null}

          <View style={{ flexDirection: 'row', gap: 12 }}>
            <View style={{ flex: 1 }}>
              <Field label="Height (cm)" value={f.heightCm} onChangeText={onInt('heightCm', 250)} keyboardType="number-pad" placeholder="168" error={errors.heightCm} />
            </View>
            <View style={{ flex: 1 }}>
              <Field label="Weight (kg)" value={f.weightKg} onChangeText={onDec('weightKg')} keyboardType="decimal-pad" placeholder="72" error={errors.weightKg} />
            </View>
          </View>

          <Body style={{ fontWeight: '600', marginBottom: 8 }}>Blood pressure (last recorded)</Body>
          <View style={{ flexDirection: 'row', gap: 12 }}>
            <View style={{ flex: 1 }}>
              <Field label="Systolic" value={f.bpSys} onChangeText={onInt('bpSys', 250)} keyboardType="number-pad" placeholder="128" error={errors.bpSys} />
            </View>
            <View style={{ flex: 1 }}>
              <Field label="Diastolic" value={f.bpDia} onChangeText={onInt('bpDia', 150)} keyboardType="number-pad" placeholder="84" error={errors.bpDia} />
            </View>
            <View style={{ flex: 1.3 }}>
              <Field label="Date" value={f.bpDate} onChangeText={onBpDate} keyboardType="number-pad" maxLength={10} placeholder="dd/mm/yyyy" error={errors.bpDate} />
            </View>
          </View>

          <Field label="HbA1c (%)" value={f.hba1c} onChangeText={onDec('hba1c')} keyboardType="decimal-pad" placeholder="6.1" error={errors.hba1c} />

          <Body style={{ fontWeight: '600', marginBottom: 10 }}>Diabetes type</Body>
          <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginBottom: 20 }}>
            {DIABETES.map((d) => (
              <Pressable key={d} onPress={() => setDiabetes(d)}
                style={{
                  paddingHorizontal: 16, paddingVertical: 9, borderRadius: 999,
                  backgroundColor: diabetes === d ? c.accent : c.surfaceAlt,
                }}>
                <Body style={{ color: diabetes === d ? c.onAccent : c.text, fontSize: 14 }}>{d}</Body>
              </Pressable>
            ))}
          </View>

          <MultiSelect label="Medical conditions (optional)" values={conditions} options={CONDITIONS} onChange={setConditions} />
          <Field label="Medications (optional)" value={f.medications} onChangeText={onText('medications')} placeholder="e.g. Metformin 500mg" />

          {hasErrors ? <Muted style={{ color: c.hyper, marginBottom: 8 }}>Please fix the highlighted fields.</Muted> : null}
          {errors.submit ? <Muted style={{ color: c.hyper, marginBottom: 8 }}>{errors.submit}</Muted> : null}
          <Button title="Finish setup" onPress={finish} loading={saving} style={{ marginTop: 8 }} />
        </ScrollView>
      </SafeAreaView>
    </Screen>
  );
}
