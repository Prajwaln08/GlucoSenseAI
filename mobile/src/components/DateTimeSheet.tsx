/**
 * Pure-JS wheel date/time picker (no native module → hot-reloads in the dev build).
 * Lets the user back-date an entry, e.g. log breakfast at dinner time.
 */
import { useRef, useState } from 'react';
import { Modal, NativeScrollEvent, NativeSyntheticEvent, Pressable, ScrollView, Text, View } from 'react-native';

import { useColors } from '@/lib/theme';
import { Body, Button } from './ui';

const ITEM = 40;       // row height
const VISIBLE = 5;     // rows shown (center = selected)
const PAD = ITEM * Math.floor(VISIBLE / 2);

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const pad2 = (n: number) => String(n).padStart(2, '0');

function dateLabel(d: Date, today: Date): string {
  const diff = Math.round((startOfDay(d).getTime() - startOfDay(today).getTime()) / 86400000);
  if (diff === 0) return 'Today';
  if (diff === -1) return 'Yesterday';
  return `${WEEKDAYS[d.getDay()]} ${d.getDate()} ${MONTHS[d.getMonth()]}`;
}
function startOfDay(d: Date): Date {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate());
}

function Wheel({ items, index, onIndex, width }: { items: string[]; index: number; onIndex: (i: number) => void; width: number }) {
  const c = useColors();
  const ref = useRef<ScrollView>(null);
  const inited = useRef(false);
  const idxAt = (e: NativeSyntheticEvent<NativeScrollEvent>) =>
    Math.max(0, Math.min(items.length - 1, Math.round(e.nativeEvent.contentOffset.y / ITEM)));
  const settle = (e: NativeSyntheticEvent<NativeScrollEvent>, animated: boolean) => {
    const i = idxAt(e);
    onIndex(i);
    ref.current?.scrollTo({ y: i * ITEM, animated });
  };
  // Deterministic initial position — fire from whichever of layout/content-size
  // comes first, deferred a tick so the offset actually applies (fixes the
  // "sometimes lands blank/wrong" flakiness on remount).
  const init = () => {
    if (inited.current) return;
    inited.current = true;
    setTimeout(() => ref.current?.scrollTo({ y: index * ITEM, animated: false }), 0);
  };
  return (
    <View style={{ width, height: ITEM * VISIBLE }}>
      <ScrollView
        ref={ref}
        showsVerticalScrollIndicator={false}
        decelerationRate="normal"
        nestedScrollEnabled
        scrollEventThrottle={16}
        contentContainerStyle={{ paddingVertical: PAD }}
        onLayout={init}
        onContentSizeChange={init}
        onScrollEndDrag={(e) => {
          const v = e.nativeEvent.velocity?.y ?? 0;
          if (Math.abs(v) < 0.08) settle(e, true);   // released without a fling → snap now
          else onIndex(idxAt(e));                      // fling → let momentum carry, settle on its end
        }}
        onMomentumScrollEnd={(e) => settle(e, true)}
      >
        {items.map((it, i) => (
          <View key={i} style={{ height: ITEM, alignItems: 'center', justifyContent: 'center' }}>
            <Text style={{ color: i === index ? c.text : c.textMuted, fontSize: i === index ? 18 : 15, fontWeight: i === index ? '700' : '400' }}>{it}</Text>
          </View>
        ))}
      </ScrollView>
      <View pointerEvents="none" style={{ position: 'absolute', top: PAD, height: ITEM, left: 0, right: 0,
        borderTopWidth: 1, borderBottomWidth: 1, borderColor: c.border }} />
    </View>
  );
}

export function DateTimeSheet({
  visible, value, onClose, onConfirm,
}: { visible: boolean; value: Date; onClose: () => void; onConfirm: (d: Date) => void }) {
  const c = useColors();
  const today = new Date();

  // last 30 days (oldest → today) so "today" sits at the bottom (latest)
  const dates = Array.from({ length: 30 }, (_, i) => startOfDay(new Date(today.getFullYear(), today.getMonth(), today.getDate() - (29 - i))));
  const hours = Array.from({ length: 12 }, (_, i) => i + 1);
  const mins = Array.from({ length: 60 }, (_, i) => i);

  const initDateIdx = dates.findIndex((d) => startOfDay(d).getTime() === startOfDay(value).getTime());
  const h24 = value.getHours();
  const [dIdx, setDIdx] = useState(initDateIdx >= 0 ? initDateIdx : dates.length - 1);
  const [hIdx, setHIdx] = useState(((h24 % 12) || 12) - 1);
  const [mIdx, setMIdx] = useState(value.getMinutes());
  const [apIdx, setApIdx] = useState(h24 >= 12 ? 1 : 0);

  function confirm() {
    const base = dates[dIdx];
    const hour12 = hours[hIdx];
    const hour = apIdx === 1 ? (hour12 % 12) + 12 : hour12 % 12;
    onConfirm(new Date(base.getFullYear(), base.getMonth(), base.getDate(), hour, mins[mIdx], 0, 0));
  }

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <Pressable onPress={onClose} style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'flex-end' }}>
        <Pressable onPress={() => {}} style={{ backgroundColor: c.bg, borderTopLeftRadius: 22, borderTopRightRadius: 22, padding: 24 }}>
          <Body style={{ fontWeight: '700', fontSize: 17, marginBottom: 12 }}>When was this?</Body>
          <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center', gap: 2 }}>
            <Wheel items={dates.map((d) => dateLabel(d, today))} index={dIdx} onIndex={setDIdx} width={156} />
            <Wheel items={hours.map(String)} index={hIdx} onIndex={setHIdx} width={52} />
            <Text style={{ color: c.text, fontSize: 18, fontWeight: '700' }}>:</Text>
            <Wheel items={mins.map(pad2)} index={mIdx} onIndex={setMIdx} width={52} />
            <Wheel items={['AM', 'PM']} index={apIdx} onIndex={setApIdx} width={58} />
          </View>
          <Button title="Set time" onPress={confirm} style={{ marginTop: 12 }} />
          <Button title="Cancel" variant="ghost" onPress={onClose} style={{ marginTop: 8 }} />
        </Pressable>
      </Pressable>
    </Modal>
  );
}

/** "Today, 8:30 AM" / "Wed 25 Jun, 7:15 PM" — or "Now" if within a minute of now. */
export function formatWhen(d: Date): string {
  const now = new Date();
  if (Math.abs(now.getTime() - d.getTime()) < 60000) return 'Now';
  const h = d.getHours();
  const t = `${(h % 12) || 12}:${pad2(d.getMinutes())} ${h >= 12 ? 'PM' : 'AM'}`;
  return `${dateLabel(d, now)}, ${t}`;
}
