import { Link } from 'expo-router';
import { useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Body, Button, Field, H1, Muted, Screen } from '@/components/ui';
import { useAuth } from '@/lib/auth';
import { useColors } from '@/lib/theme';

export default function Login() {
  const { signIn } = useAuth();
  const c = useColors();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setError(null);
    setLoading(true);
    try {
      await signIn(email.trim(), password);
    } catch {
      setError('Wrong email or password.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <Screen>
      <KeyboardAvoidingView style={{ flex: 1 }} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
        <SafeAreaView style={{ flex: 1 }}>
          <ScrollView contentContainerStyle={{ flexGrow: 1, justifyContent: 'center', padding: 24 }}
            keyboardShouldPersistTaps="handled" showsVerticalScrollIndicator={false}>
            <Body style={{ color: c.accent, fontWeight: '700', fontSize: 14 }}>GlucoSense AI</Body>
            <H1>Welcome back</H1>
            <Muted style={{ marginBottom: 24, marginTop: 4 }}>Sign in to see your glucose forecast.</Muted>

            <Field label="Email" value={email} onChangeText={setEmail}
              autoCapitalize="none" keyboardType="email-address" placeholder="you@email.com" />
            <Field label="Password" value={password} onChangeText={setPassword}
              secureTextEntry placeholder="••••••••" />

            {error && <Muted style={{ color: c.hyper, marginBottom: 8 }}>{error}</Muted>}
            <Button title="Sign in" onPress={submit} loading={loading} style={{ marginTop: 6 }} />

            <View style={{ flexDirection: 'row', justifyContent: 'center', marginTop: 18, gap: 4 }}>
              <Muted>New here?</Muted>
              <Link href="/(auth)/register" style={{ color: c.accent, fontWeight: '600' }}>Create account</Link>
            </View>
          </ScrollView>
        </SafeAreaView>
      </KeyboardAvoidingView>
    </Screen>
  );
}
