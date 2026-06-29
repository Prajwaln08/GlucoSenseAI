/** Small themed primitives shared across screens. */
import { Ionicons } from '@expo/vector-icons';
import { ReactNode, useState } from 'react';
import {
  ActivityIndicator, Modal, Pressable, ScrollView, StyleSheet, Text, TextInput, View, type TextInputProps,
} from 'react-native';
import { useColors } from '@/lib/theme';

export function Screen({ children }: { children: ReactNode }) {
  const c = useColors();
  return <View style={{ flex: 1, backgroundColor: c.bg }}>{children}</View>;
}

export function Card({ children, style }: { children: ReactNode; style?: any }) {
  const c = useColors();
  return (
    <View style={[styles.card, { backgroundColor: c.surface, borderColor: c.border }, style]}>
      {children}
    </View>
  );
}

export function H1({ children }: { children: ReactNode }) {
  const c = useColors();
  return <Text style={[styles.h1, { color: c.text }]}>{children}</Text>;
}
export function H2({ children }: { children: ReactNode }) {
  const c = useColors();
  return <Text style={[styles.h2, { color: c.text }]}>{children}</Text>;
}
export function Muted({ children, style }: { children: ReactNode; style?: any }) {
  const c = useColors();
  return <Text style={[styles.muted, { color: c.textMuted }, style]}>{children}</Text>;
}
export function Body({ children, style, ...rest }: { children: ReactNode; style?: any } & Record<string, any>) {
  const c = useColors();
  return <Text style={[{ color: c.text, fontSize: 15 }, style]} {...rest}>{children}</Text>;
}

export function Button({
  title, onPress, variant = 'primary', loading, style,
}: { title: string; onPress?: () => void; variant?: 'primary' | 'ghost'; loading?: boolean; style?: any }) {
  const c = useColors();
  const primary = variant === 'primary';
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.btn,
        {
          backgroundColor: primary ? c.accent : 'transparent',
          borderColor: c.accent,
          borderWidth: primary ? 0 : 1,
          opacity: pressed ? 0.85 : 1,
        },
        style,
      ]}
    >
      {loading ? (
        <ActivityIndicator color={primary ? c.onAccent : c.accent} />
      ) : (
        <Text style={{ color: primary ? c.onAccent : c.accent, fontWeight: '600', fontSize: 15 }}>
          {title}
        </Text>
      )}
    </Pressable>
  );
}

export function Field({ label, error, invalid, secureTextEntry, ...props }: { label: string; error?: string; invalid?: boolean } & TextInputProps) {
  const c = useColors();
  const [hidden, setHidden] = useState(!!secureTextEntry);
  const red = !!error || !!invalid;
  return (
    <View style={{ marginBottom: 14 }}>
      <Text style={[styles.label, { color: red ? c.hyper : c.textMuted }]}>{label}</Text>
      <View style={{ justifyContent: 'center' }}>
        <TextInput
          placeholderTextColor={c.textMuted}
          secureTextEntry={secureTextEntry ? hidden : false}
          style={[styles.input, { color: c.text, borderColor: red ? c.hyper : c.border, backgroundColor: c.surface },
            red ? { borderWidth: 1.5 } : null,
            secureTextEntry ? { paddingRight: 46 } : null]}
          {...props}
        />
        {secureTextEntry ? (
          <Pressable onPress={() => setHidden((h) => !h)} hitSlop={12}
            style={{ position: 'absolute', right: 12 }}>
            <Ionicons name={hidden ? 'eye-off-outline' : 'eye-outline'} size={20} color={c.textMuted} />
          </Pressable>
        ) : null}
      </View>
      {error ? <Text style={{ color: c.hyper, fontSize: 12, marginTop: 4 }}>{error}</Text> : null}
    </View>
  );
}

/** Single-choice dropdown (tap → themed modal list). */
export function Select({
  label, value, options, onChange, placeholder = 'Select', error,
}: { label?: string; value?: string; options: string[]; onChange: (v: string) => void; placeholder?: string; error?: string }) {
  const c = useColors();
  const [open, setOpen] = useState(false);
  return (
    <View style={{ marginBottom: 14 }}>
      {label ? <Text style={[styles.label, { color: error ? c.hyper : c.textMuted }]}>{label}</Text> : null}
      <Pressable onPress={() => setOpen(true)}
        style={[styles.input, { borderColor: error ? c.hyper : c.border, backgroundColor: c.surface,
          borderWidth: error ? 1.5 : 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }]}>
        <Text style={{ color: value ? c.text : c.textMuted, fontSize: 15 }}>{value || placeholder}</Text>
        <Ionicons name="chevron-down" size={18} color={c.textMuted} />
      </Pressable>
      <Modal visible={open} transparent animationType="fade" onRequestClose={() => setOpen(false)}>
        <Pressable onPress={() => setOpen(false)}
          style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'center', padding: 28 }}>
          <Pressable onPress={() => {}} style={{ backgroundColor: c.bg, borderRadius: 16, maxHeight: '70%', overflow: 'hidden' }}>
            <ScrollView>
              {options.map((o, i) => (
                <Pressable key={o} onPress={() => { onChange(o); setOpen(false); }}
                  style={{ paddingVertical: 15, paddingHorizontal: 18, flexDirection: 'row',
                    justifyContent: 'space-between', alignItems: 'center',
                    borderTopWidth: i ? 1 : 0, borderTopColor: c.border }}>
                  <Text style={{ color: c.text, fontSize: 15 }}>{o}</Text>
                  {value === o ? <Ionicons name="checkmark" size={18} color={c.accent} /> : null}
                </Pressable>
              ))}
            </ScrollView>
          </Pressable>
        </Pressable>
      </Modal>
      {error ? <Text style={{ color: c.hyper, fontSize: 12, marginTop: 4 }}>{error}</Text> : null}
    </View>
  );
}

/** Multi-choice dropdown (e.g. medical conditions). */
export function MultiSelect({
  label, values, options, onChange, placeholder = 'Select all that apply',
}: { label?: string; values: string[]; options: string[]; onChange: (v: string[]) => void; placeholder?: string }) {
  const c = useColors();
  const [open, setOpen] = useState(false);
  const toggle = (o: string) =>
    onChange(values.includes(o) ? values.filter((v) => v !== o) : [...values, o]);
  return (
    <View style={{ marginBottom: 14 }}>
      {label ? <Text style={[styles.label, { color: c.textMuted }]}>{label}</Text> : null}
      <Pressable onPress={() => setOpen(true)}
        style={[styles.input, { borderColor: c.border, backgroundColor: c.surface, minHeight: 48, height: undefined,
          paddingVertical: 12, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }]}>
        <Text style={{ color: values.length ? c.text : c.textMuted, fontSize: 15, flex: 1 }}>
          {values.length ? values.join(', ') : placeholder}
        </Text>
        <Ionicons name="chevron-down" size={18} color={c.textMuted} />
      </Pressable>
      <Modal visible={open} transparent animationType="fade" onRequestClose={() => setOpen(false)}>
        <Pressable onPress={() => setOpen(false)}
          style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'center', padding: 28 }}>
          <Pressable onPress={() => {}} style={{ backgroundColor: c.bg, borderRadius: 16, maxHeight: '75%', overflow: 'hidden' }}>
            <ScrollView>
              {options.map((o, i) => {
                const on = values.includes(o);
                return (
                  <Pressable key={o} onPress={() => toggle(o)}
                    style={{ paddingVertical: 15, paddingHorizontal: 18, flexDirection: 'row',
                      justifyContent: 'space-between', alignItems: 'center',
                      borderTopWidth: i ? 1 : 0, borderTopColor: c.border }}>
                    <Text style={{ color: c.text, fontSize: 15 }}>{o}</Text>
                    <Ionicons name={on ? 'checkbox' : 'square-outline'} size={20} color={on ? c.accent : c.textMuted} />
                  </Pressable>
                );
              })}
            </ScrollView>
            <Pressable onPress={() => setOpen(false)} style={{ padding: 16, alignItems: 'center', borderTopWidth: 1, borderTopColor: c.border }}>
              <Text style={{ color: c.accent, fontWeight: '600', fontSize: 15 }}>Done</Text>
            </Pressable>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
}

export function Pill({ label, tone = 'muted' }: { label: string; tone?: 'muted' | 'accent' | 'good' | 'warn' | 'bad' }) {
  const c = useColors();
  const map = {
    muted: { bg: c.surfaceAlt, fg: c.textMuted },
    accent: { bg: c.accentSoft, fg: c.accent },
    good: { bg: c.inRangeBand, fg: c.inRange },
    warn: { bg: 'rgba(245,158,11,0.15)', fg: c.hypo },
    bad: { bg: 'rgba(239,68,68,0.15)', fg: c.hyper },
  }[tone];
  return (
    <View style={[styles.pill, { backgroundColor: map.bg }]}>
      <Text style={{ color: map.fg, fontSize: 12, fontWeight: '600' }}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: { borderRadius: 18, borderWidth: 1, padding: 16, marginBottom: 14 },
  h1: { fontSize: 24, fontWeight: '700' },
  h2: { fontSize: 17, fontWeight: '600', marginBottom: 10 },
  muted: { fontSize: 13 },
  btn: { height: 48, borderRadius: 14, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 18 },
  label: { fontSize: 12, textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 6 },
  input: { height: 48, borderRadius: 12, borderWidth: 1, paddingHorizontal: 14, fontSize: 15 },
  pill: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999, alignSelf: 'flex-start' },
});
