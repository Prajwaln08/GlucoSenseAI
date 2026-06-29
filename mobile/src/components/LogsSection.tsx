/** Profile → Logs: pick a date, see that day's food + vitals + activity. */
import { Ionicons } from '@expo/vector-icons';
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { Pressable, View } from 'react-native';

import { Api, type VitalEntry } from '@/lib/api';
import { useColors } from '@/lib/theme';
import { Calendar } from './Calendar';
import { Body, Card, Muted } from './ui';

const WD = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MO = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const pad = (n: number) => String(n).padStart(2, '0');
const dateStr = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

function fmtTime(iso?: string): string {
  if (!iso) return '';
  const d = new Date(iso);
  return `${(d.getHours() % 12) || 12}:${pad(d.getMinutes())} ${d.getHours() >= 12 ? 'PM' : 'AM'}`;
}
function dateLabel(d: Date): string {
  const diff = Math.round((startOfDay(d) - startOfDay(new Date())) / 86400000);
  if (diff === 0) return 'Today';
  if (diff === -1) return 'Yesterday';
  return `${WD[d.getDay()]} ${d.getDate()} ${MO[d.getMonth()]} ${d.getFullYear()}`;
}
function vitalLabel(v: VitalEntry): string {
  if (v.kind === 'bp') return `Blood pressure ${v.bp_systolic}/${v.bp_diastolic}`;
  if (v.kind === 'weight') return `Weight ${v.value} kg`;
  if (v.kind === 'hba1c') return `HbA1c ${v.value}%`;
  if (v.kind === 'glucose') return `Glucose ${v.value} mg/dL`;
  return `${cap(v.kind)} ${v.value ?? ''}`;
}

export function LogsSection() {
  const c = useColors();
  const [date, setDate] = useState(() => new Date());
  const [calOpen, setCalOpen] = useState(false);
  const [showAllFood, setShowAllFood] = useState(false);
  const ds = dateStr(date);
  const { data, isLoading } = useQuery({ queryKey: ['logs', ds], queryFn: () => Api.logs(ds) });

  const food = data?.food ?? [];
  const vitals = data?.vitals ?? [];
  const act = data?.activity;
  const shownFood = showAllFood ? food : food.slice(0, 5);
  const empty = !isLoading && food.length === 0 && vitals.length === 0 && !act;

  return (
    <Card>
      <Pressable onPress={() => setCalOpen((o) => !o)}
        style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingVertical: 2 }}>
        <Body style={{ fontWeight: '600' }}>{dateLabel(date)}</Body>
        <Ionicons name={calOpen ? 'chevron-up' : 'calendar-outline'} size={18} color={c.accent} />
      </Pressable>

      {calOpen && (
        <View style={{ marginTop: 12 }}>
          <Calendar value={date} onChange={(d) => { setDate(d); setShowAllFood(false); setCalOpen(false); }} />
        </View>
      )}

      <View style={{ height: 1, backgroundColor: c.border, marginVertical: 12 }} />

      {isLoading ? <Muted>Loading…</Muted> : empty ? <Muted>No logs on this day.</Muted> : (
        <>
          {food.length > 0 && <Muted style={{ fontWeight: '700', marginBottom: 4 }}>Food</Muted>}
          {shownFood.map((f) => (
            <View key={f.id} style={{ flexDirection: 'row', paddingVertical: 6 }}>
              <Body style={{ width: 76, color: c.textMuted, fontSize: 13 }}>{fmtTime(f.logged_at)}</Body>
              <View style={{ flex: 1 }}>
                <Body>{cap(f.meal_type)}{f.description ? ` · ${f.description}` : ''}</Body>
                {(f.quantity != null || f.portion_size) ? (
                  <Muted>{f.quantity ?? ''} {f.portion_size ?? ''}</Muted>
                ) : null}
              </View>
            </View>
          ))}
          {food.length > 5 && (
            <Body onPress={() => setShowAllFood((s) => !s)} style={{ color: c.accent, fontWeight: '600', marginTop: 4 }}>
              {showAllFood ? 'Show less' : `Show ${food.length - 5} more`}
            </Body>
          )}

          {vitals.length > 0 && <Muted style={{ fontWeight: '700', marginTop: 14, marginBottom: 4 }}>Vitals</Muted>}
          {vitals.map((v) => (
            <View key={v.id} style={{ flexDirection: 'row', paddingVertical: 6 }}>
              <Body style={{ width: 76, color: c.textMuted, fontSize: 13 }}>{fmtTime(v.recorded_at)}</Body>
              <Body style={{ flex: 1 }}>{vitalLabel(v)}</Body>
            </View>
          ))}

          {act ? (
            <>
              <Muted style={{ fontWeight: '700', marginTop: 14, marginBottom: 6 }}>Activity</Muted>
              <View style={{ flexDirection: 'row', gap: 20 }}>
                {act.steps != null ? <Stat label="Steps" value={String(act.steps)} /> : null}
                {act.hr_avg_bpm != null ? <Stat label="Avg HR" value={`${Math.round(act.hr_avg_bpm)} bpm`} /> : null}
                {act.calories_active != null ? <Stat label="Calories" value={String(Math.round(act.calories_active))} /> : null}
              </View>
            </>
          ) : null}
        </>
      )}
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View>
      <Body style={{ fontWeight: '700', fontSize: 18 }}>{value}</Body>
      <Muted>{label}</Muted>
    </View>
  );
}
