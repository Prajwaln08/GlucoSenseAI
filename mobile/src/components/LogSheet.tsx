/** Bottom-sheet modal to log food or vitals → POSTs to the backend. */
import { Ionicons } from '@expo/vector-icons';
import { useState } from 'react';
import { KeyboardAvoidingView, Modal, Platform, Pressable, ScrollView, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Api } from '@/lib/api';
import { useColors } from '@/lib/theme';
import { DateTimeSheet, formatWhen } from './DateTimeSheet';
import { Body, Button, Field, H2, Muted, Select } from './ui';

const MEALS = ['Breakfast', 'Lunch', 'Dinner', 'Snack'];
const SIZES = ['Bowl', 'Katori', 'Cup', 'Piece', 'Glass'];

const inRange = (n: number, lo: number, hi: number) => Number.isFinite(n) && n >= lo && n <= hi;

export function LogSheet({
  visible, mode, onClose, onLogged,
}: { visible: boolean; mode: 'food' | 'vitals'; onClose: () => void; onLogged: () => void }) {
  const c = useColors();
  const insets = useSafeAreaInsets();
  const [meal, setMeal] = useState('Lunch');
  const [desc, setDesc] = useState('');
  const [qty, setQty] = useState('');
  const [size, setSize] = useState('');
  const [sys, setSys] = useState('');
  const [dia, setDia] = useState('');
  const [weight, setWeight] = useState('');
  const [hba1c, setHba1c] = useState('');
  const [when, setWhen] = useState<Date>(() => new Date());
  const [showWhen, setShowWhen] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  const clearErr = (k: string) => setErrors((e) => { if (!e[k] && !e.form) return e; const n = { ...e }; delete n[k]; delete n.form; return n; });
  const digits = (set: (v: string) => void, k: string) => (v: string) => { set(v.replace(/\D/g, '')); clearErr(k); };
  const decimal = (set: (v: string) => void, k: string) => (v: string) => {
    let x = v.replace(/[^0-9.]/g, ''); const p = x.split('.'); if (p.length > 2) x = p[0] + '.' + p.slice(1).join('');
    set(x); clearErr(k);
  };

  function reset() {
    setDesc(''); setQty(''); setSize(''); setSys(''); setDia(''); setWeight(''); setHba1c('');
    setWhen(new Date()); setErrors({});
  }

  function validate(): Record<string, string> {
    const e: Record<string, string> = {};
    if (mode === 'food') {
      if (!desc.trim()) e.desc = 'Tell us what you ate';
      const q = Number(qty);
      if (!qty.trim()) e.qty = 'Required'; else if (!Number.isFinite(q) || q <= 0 || q > 50) e.qty = 'Enter 0.1–50';
      if (!size) e.size = 'Pick a size';
    } else {
      if (!sys && !dia && !weight && !hba1c) { e.form = 'Enter at least one reading.'; return e; }
      if (sys || dia) {
        if (!inRange(Number(sys), 70, 250)) e.sys = '70–250';
        if (!inRange(Number(dia), 40, 150)) e.dia = '40–150';
      }
      if (weight && !inRange(Number(weight), 20, 400)) e.weight = 'Weight must be 20–400 kg';
      if (hba1c && !inRange(Number(hba1c), 3, 20)) e.hba1c = 'HbA1c must be 3–20%';
    }
    return e;
  }

  async function save() {
    const e = validate();
    setErrors(e);
    if (Object.keys(e).length) return;

    setSaving(true);
    try {
      if (mode === 'food') {
        await Api.addFood({
          meal_type: meal.toLowerCase(),
          description: desc.trim(),
          quantity: Math.round(Number(qty) * 10) / 10,
          portion_size: size.toLowerCase(),
          logged_at: when.toISOString(),
        });
      } else {
        const at = when.toISOString();
        if (sys && dia) await Api.addVital({ kind: 'bp', bp_systolic: Number(sys), bp_diastolic: Number(dia), recorded_at: at });
        if (weight) await Api.addVital({ kind: 'weight', value: Number(weight), recorded_at: at });
        if (hba1c) await Api.addVital({ kind: 'hba1c', value: Number(hba1c), recorded_at: at });
      }
      reset();
      onLogged();
      onClose();
    } catch {
      setErrors({ form: 'Could not save. Try again.' });
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <Pressable onPress={onClose} style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'flex-end' }}>
          <Pressable onPress={() => {}} style={{ backgroundColor: c.bg, borderTopLeftRadius: 22, borderTopRightRadius: 22, maxHeight: '88%' }}>
            <ScrollView keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}
              contentContainerStyle={{ padding: 24, paddingBottom: 28 + insets.bottom }}>
              <H2>{mode === 'food' ? 'Log food' : 'Log vitals'}</H2>

          {mode === 'food' ? (
            <>
              <View style={{ flexDirection: 'row', gap: 8, marginBottom: 16 }}>
                {MEALS.map((m) => (
                  <Pressable key={m} onPress={() => setMeal(m)}
                    style={{ flex: 1, paddingVertical: 10, borderRadius: 999, alignItems: 'center',
                      backgroundColor: meal === m ? c.accent : c.surfaceAlt }}>
                    <Body style={{ color: meal === m ? c.onAccent : c.text, fontSize: 13 }}>{m}</Body>
                  </Pressable>
                ))}
              </View>
              <Field label="What did you eat? *" value={desc} onChangeText={(v) => { setDesc(v); clearErr('desc'); }}
                placeholder="e.g. Rice, dal, salad" error={errors.desc} />
              <View style={{ flexDirection: 'row', gap: 12 }}>
                <View style={{ flex: 1 }}>
                  <Field label="Quantity *" value={qty} onChangeText={decimal(setQty, 'qty')} keyboardType="decimal-pad" placeholder="1.5" error={errors.qty} />
                </View>
                <View style={{ flex: 1.2 }}>
                  <Select label="Size *" value={size} options={SIZES} onChange={(v) => { setSize(v); clearErr('size'); }} placeholder="Select" error={errors.size} />
                </View>
              </View>
            </>
          ) : (
            <>
              <Muted style={{ marginBottom: 8 }}>Blood pressure</Muted>
              <View style={{ flexDirection: 'row', gap: 12 }}>
                <View style={{ flex: 1 }}><Field label="Systolic" value={sys} onChangeText={digits(setSys, 'sys')} keyboardType="number-pad" placeholder="128" error={errors.sys} /></View>
                <View style={{ flex: 1 }}><Field label="Diastolic" value={dia} onChangeText={digits(setDia, 'dia')} keyboardType="number-pad" placeholder="82" error={errors.dia} /></View>
              </View>
              <Field label="Weight (kg)" value={weight} onChangeText={decimal(setWeight, 'weight')} keyboardType="decimal-pad" placeholder="72" error={errors.weight} />
              <Field label="HbA1c (%)" value={hba1c} onChangeText={decimal(setHba1c, 'hba1c')} keyboardType="decimal-pad" placeholder="6.1" error={errors.hba1c} />
            </>
          )}

              <Text style={{ fontSize: 12, color: c.textMuted, textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 6 }}>
                When{mode === 'vitals' ? ' (date)' : ''}
              </Text>
              <Pressable onPress={() => setShowWhen(true)}
                style={{ borderWidth: 1, borderColor: c.border, backgroundColor: c.surface, borderRadius: 12,
                  paddingHorizontal: 14, height: 48, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                <Body>{formatWhen(when)}</Body>
                <Ionicons name="time-outline" size={20} color={c.textMuted} />
              </Pressable>

              {errors.form ? <Muted style={{ color: c.hyper, marginBottom: 8 }}>{errors.form}</Muted> : null}
              <Button title="Save" onPress={save} loading={saving} />
              <Button title="Cancel" variant="ghost" onPress={onClose} style={{ marginTop: 8 }} />
            </ScrollView>
          </Pressable>
        </Pressable>
      </KeyboardAvoidingView>
    </Modal>

    {showWhen && (
      <DateTimeSheet visible value={when} onClose={() => setShowWhen(false)}
        onConfirm={(d) => { setWhen(d); setShowWhen(false); }} />
    )}
    </>
  );
}
