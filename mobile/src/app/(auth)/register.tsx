import { Link } from 'expo-router';
import { useState } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Body, Button, Field, H1, Muted, Screen } from '@/components/ui';
import { useAuth } from '@/lib/auth';
import { useColors } from '@/lib/theme';

// Real users sign up with email + password only — no research participant id/dataset.
export default function Register() {
  const { signUp } = useAuth();
  const c = useColors();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setError(null);
    setLoading(true);
    try {
      await signUp(email.trim(), password);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Could not create account.');
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
            <H1>Create account</H1>
            <Muted style={{ marginBottom: 24, marginTop: 4 }}>Just an email and password to get started.</Muted>

            <Field label="Email" value={email} onChangeText={setEmail}
              autoCapitalize="none" keyboardType="email-address" placeholder="you@email.com" />
            <Field label="Password" value={password} onChangeText={setPassword}
              secureTextEntry placeholder="Choose a password" />

            {error && <Muted style={{ color: c.hyper, marginBottom: 8 }}>{error}</Muted>}
            <Button title="Create account" onPress={submit} loading={loading} style={{ marginTop: 6 }} />

            <View style={{ flexDirection: 'row', justifyContent: 'center', marginTop: 18, gap: 4 }}>
              <Muted>Already have an account?</Muted>
              <Link href="/(auth)/login" style={{ color: c.accent, fontWeight: '600' }}>Sign in</Link>
            </View>
          </ScrollView>
        </SafeAreaView>
      </KeyboardAvoidingView>
    </Screen>
  );
}
