/** Bottom-sheet to log food or vitals → optimistic save (appears instantly). */
import { useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { Alert, KeyboardAvoidingView, Modal, Platform, Pressable, ScrollView, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Api } from '@/lib/api';
import { useColors } from '@/lib/theme';
import { Body, Button, Field, H2, Muted, Select } from './ui';

const MEALS = ['Breakfast', 'Lunch', 'Dinner', 'Snack'];
const SIZES = ['Bowl', 'Katori', 'Cup', 'Piece', 'Glass'];
const WD = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MO = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const pad2 = (n: number) => String(n).padStart(2, '0');
const HOURS = Array.from({ length: 12 }, (_, i) => String(i + 1));
const MINUTES = Array.from({ length: 12 }, (_, i) => pad2(i * 5));   // 00,05,…,55
const AMPM = ['AM', 'PM'];

const inRange = (n: number, lo: number, hi: number) => Number.isFinite(n) && n >= lo && n <= hi;

export function LogSheet({
  visible, mode, onClose, onLogged,
}: { visible: boolean; mode: 'food' | 'vitals'; onClose: () => void; onLogged: () => void }) {
  const c = useColors();
  const qc = useQueryClient();
  const insets = useSafeAreaInsets();
  const [meal, setMeal] = useState('Lunch');
  const [desc, setDesc] = useState('');
  const [qty, setQty] = useState('');
  const [size, setSize] = useState('');
  const [sys, setSys] = useState('');
  const [dia, setDia] = useState('');
  const [weight, setWeight] = useState('');
  const [hba1c, setHba1c] = useState('');
  // when = day (last 7 days) + time, as dropdowns
  const n0 = new Date();
  const [dayIdx, setDayIdx] = useState(0);   // 0 = today … 6 = 6 days ago
  const [hour, setHour] = useState(String((n0.getHours() % 12) || 12));
  const [minute, setMinute] = useState(pad2(Math.floor(n0.getMinutes() / 5) * 5));
  const [ampm, setAmpm] = useState(n0.getHours() >= 12 ? 'PM' : 'AM');
  const [errors, setErrors] = useState<Record<string, string>>({});

  // last 7 days, newest first
  const days = Array.from({ length: 7 }, (_, i) => { const d = new Date(); d.setHours(0, 0, 0, 0); d.setDate(d.getDate() - i); return d; });
  const dayLabels = days.map((d, i) => (i === 0 ? 'Today' : i === 1 ? 'Yesterday' : `${WD[d.getDay()]} ${d.getDate()} ${MO[d.getMonth()]}`));
  function buildWhen(): Date {
    const base = days[dayIdx] ?? days[0];
    const h12 = Number(hour);
    const hh = ampm === 'PM' ? (h12 % 12) + 12 : h12 % 12;
    return new Date(base.getFullYear(), base.getMonth(), base.getDate(), hh, Number(minute), 0, 0);
  }

  const clearErr = (k: string) => setErrors((e) => { if (!e[k] && !e.form) return e; const n = { ...e }; delete n[k]; delete n.form; return n; });
  const digits = (set: (v: string) => void, k: string) => (v: string) => { set(v.replace(/\D/g, '')); clearErr(k); };
  const decimal = (set: (v: string) => void, k: string) => (v: string) => {
    let x = v.replace(/[^0-9.]/g, ''); const p = x.split('.'); if (p.length > 2) x = p[0] + '.' + p.slice(1).join('');
    set(x); clearErr(k);
  };

  function reset() {
    setDesc(''); setQty(''); setSize(''); setSys(''); setDia(''); setWeight(''); setHba1c('');
    const n = new Date();
    setDayIdx(0); setHour(String((n.getHours() % 12) || 12)); setMinute(pad2(Math.floor(n.getMinutes() / 5) * 5));
    setAmpm(n.getHours() >= 12 ? 'PM' : 'AM'); setErrors({});
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
    if (buildWhen().getTime() > Date.now() + 60_000) e.form = 'That time is in the future — pick an earlier time.';
    return e;
  }

  function save() {
    const e = validate();
    setErrors(e);
    if (Object.keys(e).length) return;

    const at = buildWhen().toISOString();
    const foodPayload = {
      meal_type: meal.toLowerCase(), description: desc.trim(),
      quantity: Math.round(Number(qty) * 10) / 10, portion_size: size.toLowerCase(), logged_at: at,
    };
    const vitalPayloads: { kind: string; value?: number; bp_systolic?: number; bp_diastolic?: number; recorded_at: string }[] = [];
    if (sys && dia) vitalPayloads.push({ kind: 'bp', bp_systolic: Number(sys), bp_diastolic: Number(dia), recorded_at: at });
    if (weight) vitalPayloads.push({ kind: 'weight', value: Number(weight), recorded_at: at });
    if (hba1c) vitalPayloads.push({ kind: 'hba1c', value: Number(hba1c), recorded_at: at });

    // optimistic: the food marker shows on the chart instantly
    if (mode === 'food') {
      qc.setQueryData(['food-recent'], (old: any) =>
        [{ id: `tmp-${at}`, logged_at: at, meal_type: foodPayload.meal_type, description: foodPayload.description },
          ...(Array.isArray(old) ? old : [])]);
    }

    reset();
    onClose();   // close instantly — no waiting on the network

    // sync in the background; revert + warn only if it fails
    (async () => {
      try {
        if (mode === 'food') await Api.addFood(foodPayload);
        else for (const v of vitalPayloads) await Api.addVital(v);
        onLogged();
        qc.invalidateQueries({ queryKey: ['logs'] });
        qc.invalidateQueries({ queryKey: ['food-recent'] });
        qc.invalidateQueries({ queryKey: ['recommendations'] });
      } catch {
        qc.invalidateQueries({ queryKey: ['food-recent'] });   // undo the optimistic marker
        Alert.alert('Not saved', 'That entry could not be saved — please check your connection and try again.');
      }
    })();
  }

  const label = { fontSize: 12, color: c.textMuted, textTransform: 'uppercase' as const, letterSpacing: 0.5, marginBottom: 6 };

  return (
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

              <Text style={label}>When</Text>
              <Select value={dayLabels[dayIdx]} options={dayLabels}
                onChange={(v) => { setDayIdx(Math.max(0, dayLabels.indexOf(v))); clearErr('form'); }} />
              <View style={{ flexDirection: 'row', gap: 10 }}>
                <View style={{ flex: 1 }}><Select value={hour} options={HOURS} onChange={(v) => { setHour(v); clearErr('form'); }} /></View>
                <View style={{ flex: 1 }}><Select value={minute} options={MINUTES} onChange={(v) => { setMinute(v); clearErr('form'); }} /></View>
                <View style={{ flex: 1 }}><Select value={ampm} options={AMPM} onChange={(v) => { setAmpm(v); clearErr('form'); }} /></View>
              </View>

              {errors.form ? <Muted style={{ color: c.hyper, marginBottom: 8 }}>{errors.form}</Muted> : null}
              <Button title="Save" onPress={save} />
              <Button title="Cancel" variant="ghost" onPress={onClose} style={{ marginTop: 8 }} />
            </ScrollView>
          </Pressable>
        </Pressable>
      </KeyboardAvoidingView>
    </Modal>
  );
}
