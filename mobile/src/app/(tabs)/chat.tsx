import { Ionicons } from '@expo/vector-icons';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, FlatList, KeyboardAvoidingView, Platform, Pressable, TextInput, View } from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';

import { Body, Muted, Screen } from '@/components/ui';
import { Api } from '@/lib/api';
import { useColors } from '@/lib/theme';

type Msg = { id: string; role: string; content: string };

const welcomeMsg = (firstName: string): Msg => ({
  id: 'welcome', role: 'assistant',
  content: `Hi${firstName ? ` ${firstName}` : ''}, I'm Doctor Gluco 👋 Ask me anything about diabetes and your glucose, or about the food and vitals you've logged. You can also log a meal or a reading right here — just tell me.`,
});

export default function Chat() {
  const c = useColors();
  const qc = useQueryClient();
  const insets = useSafeAreaInsets();
  const listRef = useRef<FlatList>(null);
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);

  const { data: history } = useQuery({ queryKey: ['coach-messages'], queryFn: () => Api.coachMessages() });
  const { data: profile } = useQuery({ queryKey: ['profile'], queryFn: () => Api.getProfile() });
  const firstName = (profile?.first_name || profile?.name?.trim().split(/\s+/)[0] || '').trim();

  const [messages, setMessages] = useState<Msg[]>([welcomeMsg('')]);

  useEffect(() => {
    if (history && history.length) {
      setMessages(history.map((m, i) => ({ id: `h${i}`, role: m.role, content: m.content })));
    } else {
      setMessages([welcomeMsg(firstName)]);
    }
  }, [history, firstName]);

  async function send() {
    const t = text.trim();
    if (!t || sending) return;
    setText('');
    setMessages((m) => [...m, { id: `u${Date.now()}`, role: 'user', content: t }]);
    setSending(true);
    try {
      const { reply } = await Api.coachChat(t);
      setMessages((m) => [...m, { id: `a${Date.now()}`, role: 'assistant', content: reply }]);
      // logging via chat can change food/glucose → refresh the dashboard
      qc.invalidateQueries({ queryKey: ['timeseries'] });
      qc.invalidateQueries({ queryKey: ['recommendations'] });
    } catch {
      setMessages((m) => [...m, { id: `e${Date.now()}`, role: 'assistant', content: 'Sorry — I could not reach the coach just now.' }]);
    } finally {
      setSending(false);
    }
  }

  return (
    <Screen>
      <SafeAreaView edges={['top']} style={{ flex: 1 }}>
        <View style={{ padding: 16, borderBottomWidth: 1, borderBottomColor: c.border }}>
          <Body style={{ fontWeight: '700', fontSize: 18 }}>Doctor Gluco</Body>
          <Muted>Your glucose, food & vitals — ask or log anything</Muted>
        </View>
        <KeyboardAvoidingView style={{ flex: 1 }}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          keyboardVerticalOffset={Platform.OS === 'ios' ? insets.bottom + 49 : 0}>
          <FlatList
            ref={listRef}
            style={{ flex: 1 }}
            data={messages}
            keyExtractor={(m) => m.id}
            contentContainerStyle={{ padding: 16, gap: 10 }}
            renderItem={({ item }) => <Bubble role={item.role} content={item.content} />}
            onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
            ListFooterComponent={sending ? (
              <View style={{ alignSelf: 'flex-start', padding: 12 }}>
                <ActivityIndicator color={c.textMuted} />
              </View>
            ) : null}
          />
          <View style={{ flexDirection: 'row', padding: 12, gap: 8, alignItems: 'center',
            borderTopWidth: 1, borderTopColor: c.border, backgroundColor: c.surface }}>
            <TextInput value={text} onChangeText={setText} placeholder="Message Doctor Gluco…"
              placeholderTextColor={c.textMuted} onSubmitEditing={send} returnKeyType="send"
              style={{ flex: 1, color: c.text, backgroundColor: c.surfaceAlt, borderRadius: 22,
                paddingHorizontal: 16, paddingVertical: 11 }} />
            <Pressable onPress={send} disabled={sending}
              style={{ width: 44, height: 44, borderRadius: 22, backgroundColor: c.accent,
                alignItems: 'center', justifyContent: 'center', opacity: sending ? 0.5 : 1 }}>
              <Ionicons name="send" size={18} color={c.onAccent} />
            </Pressable>
          </View>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </Screen>
  );
}

function Bubble({ role, content }: { role: string; content: string }) {
  const c = useColors();
  const me = role === 'user';
  return (
    <View style={{ alignSelf: me ? 'flex-end' : 'flex-start', maxWidth: '85%',
      backgroundColor: me ? c.accent : c.surface, borderWidth: me ? 0 : 1, borderColor: c.border,
      borderRadius: 16, padding: 12 }}>
      <Body style={{ color: me ? c.onAccent : c.text }}>{content}</Body>
    </View>
  );
}
