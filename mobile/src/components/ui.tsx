/** Small themed primitives shared across screens. */
import { ReactNode } from 'react';
import {
  ActivityIndicator, Pressable, StyleSheet, Text, TextInput, View, type TextInputProps,
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

export function Field({ label, ...props }: { label: string } & TextInputProps) {
  const c = useColors();
  return (
    <View style={{ marginBottom: 14 }}>
      <Text style={[styles.label, { color: c.textMuted }]}>{label}</Text>
      <TextInput
        placeholderTextColor={c.textMuted}
        style={[styles.input, { color: c.text, borderColor: c.border, backgroundColor: c.surface }]}
        {...props}
      />
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
