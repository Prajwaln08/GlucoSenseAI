/** Bottom-sheet modal to log food or vitals → POSTs to the backend. */
import { useState } from 'react';
import { Modal, Pressable, View } from 'react-native';

import { Api } from '@/lib/api';
import { useColors } from '@/lib/theme';
import { Body, Button, Field, H2, Muted } from './ui';

const MEALS = ['Breakfast', 'Lunch', 'Dinner', 'Snack'];

export function LogSheet({
  visible, mode, onClose, onLogged,
}: { visible: boolean; mode: 'food' | 'vitals'; onClose: () => void; onLogged: () => void }) {
  const c = useColors();
  const [meal, setMeal] = useState('Lunch');
  const [desc, setDesc] = useState('');
  const [carbs, setCarbs] = useState('');
  const [sys, setSys] = useState('');
  const [dia, setDia] = useState('');
  const [weight, setWeight] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save() {
    setError(null);
    setSaving(true);
    try {
      if (mode === 'food') {
        await Api.addFood({
          meal_type: meal.toLowerCase(),
          description: desc || undefined,
          carbs_g: carbs.trim() ? Number(carbs) : undefined,
        });
      } else {
        if (sys && dia) await Api.addVital({ kind: 'bp', bp_systolic: Number(sys), bp_diastolic: Number(dia) });
        if (weight) await Api.addVital({ kind: 'weight', value: Number(weight) });
      }
      onLogged();
      onClose();
    } catch {
      setError('Could not save. Try again.');
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <Pressable onPress={onClose} style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'flex-end' }}>
        <Pressable onPress={() => {}} style={{ backgroundColor: c.bg, borderTopLeftRadius: 22, borderTopRightRadius: 22, padding: 24 }}>
          <H2>{mode === 'food' ? 'Log food' : 'Log vitals'}</H2>

          {mode === 'food' ? (
            <>
              <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: 14 }}>
                {MEALS.map((m) => (
                  <Body key={m} onPress={() => setMeal(m)}
                    style={{ paddingHorizontal: 14, paddingVertical: 8, borderRadius: 999, overflow: 'hidden',
                      color: meal === m ? c.onAccent : c.text, backgroundColor: meal === m ? c.accent : c.surfaceAlt, fontSize: 14 }}>
                    {m}
                  </Body>
                ))}
              </View>
              <Field label="What did you eat?" value={desc} onChangeText={setDesc} placeholder="e.g. Rice, dal, salad" />
              <Field label="Carbs (g)" value={carbs} onChangeText={setCarbs} keyboardType="number-pad" placeholder="62" />
            </>
          ) : (
            <>
              <Muted style={{ marginBottom: 8 }}>Blood pressure</Muted>
              <View style={{ flexDirection: 'row', gap: 12 }}>
                <View style={{ flex: 1 }}><Field label="Systolic" value={sys} onChangeText={setSys} keyboardType="number-pad" placeholder="128" /></View>
                <View style={{ flex: 1 }}><Field label="Diastolic" value={dia} onChangeText={setDia} keyboardType="number-pad" placeholder="82" /></View>
              </View>
              <Field label="Weight (kg)" value={weight} onChangeText={setWeight} keyboardType="decimal-pad" placeholder="72" />
            </>
          )}

          {error && <Muted style={{ color: c.hyper, marginBottom: 8 }}>{error}</Muted>}
          <Button title="Save" onPress={save} loading={saving} />
          <Button title="Cancel" variant="ghost" onPress={onClose} style={{ marginTop: 8 }} />
        </Pressable>
      </Pressable>
    </Modal>
  );
}
