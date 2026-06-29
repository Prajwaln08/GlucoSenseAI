/** Compact month calendar to pick a date (no future dates). */
import { Ionicons } from '@expo/vector-icons';
import { useState } from 'react';
import { Pressable, Text, View } from 'react-native';

import { useColors } from '@/lib/theme';
import { Body } from './ui';

const WD = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'];

const sameDay = (a: Date, b: Date) =>
  a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();

export function Calendar({ value, onChange }: { value: Date; onChange: (d: Date) => void }) {
  const c = useColors();
  const [month, setMonth] = useState(() => new Date(value.getFullYear(), value.getMonth(), 1));
  const today = new Date();
  const year = month.getFullYear();
  const m = month.getMonth();
  const firstWeekday = new Date(year, m, 1).getDay();
  const daysInMonth = new Date(year, m + 1, 0).getDate();
  const cells: (number | null)[] = [...Array(firstWeekday).fill(null), ...Array.from({ length: daysInMonth }, (_, i) => i + 1)];
  const nextDisabled = year === today.getFullYear() && m === today.getMonth();

  return (
    <View>
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <Pressable onPress={() => setMonth(new Date(year, m - 1, 1))} hitSlop={10}>
          <Ionicons name="chevron-back" size={20} color={c.text} />
        </Pressable>
        <Body style={{ fontWeight: '700' }}>{MONTHS[m]} {year}</Body>
        <Pressable onPress={() => !nextDisabled && setMonth(new Date(year, m + 1, 1))} hitSlop={10} disabled={nextDisabled}>
          <Ionicons name="chevron-forward" size={20} color={nextDisabled ? c.textMuted : c.text} />
        </Pressable>
      </View>

      <View style={{ flexDirection: 'row' }}>
        {WD.map((d, i) => (
          <Text key={i} style={{ flex: 1, textAlign: 'center', color: c.textMuted, fontSize: 11, marginBottom: 4 }}>{d}</Text>
        ))}
      </View>

      <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
        {cells.map((day, i) => {
          if (day === null) return <View key={i} style={{ width: `${100 / 7}%`, height: 38 }} />;
          const d = new Date(year, m, day);
          const selected = sameDay(d, value);
          const isToday = sameDay(d, today);
          const future = d > today && !isToday;
          return (
            <View key={i} style={{ width: `${100 / 7}%`, height: 38, alignItems: 'center', justifyContent: 'center' }}>
              <Pressable onPress={() => !future && onChange(d)} disabled={future}
                style={{ width: 32, height: 32, borderRadius: 16, alignItems: 'center', justifyContent: 'center',
                  backgroundColor: selected ? c.accent : 'transparent' }}>
                <Text style={{
                  color: future ? c.textMuted + '66' : selected ? c.onAccent : isToday ? c.accent : c.text,
                  fontWeight: selected || isToday ? '700' : '400', fontSize: 14,
                }}>{day}</Text>
              </Pressable>
            </View>
          );
        })}
      </View>
    </View>
  );
}
