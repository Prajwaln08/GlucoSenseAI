import { Ionicons } from '@expo/vector-icons';
import { useState } from 'react';
import { FlatList, KeyboardAvoidingView, Platform, Pressable, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Body, Muted, Screen } from '@/components/ui';
import { mockChat } from '@/lib/mock';
import { useColors } from '@/lib/theme';

type Msg = { id: string; role: string; content: string };

export default function Chat() {
  const c = useColors();
  const [messages, setMessages] = useState<Msg[]>(mockChat);
  const [text, setText] = useState('');

  function send() {
    const t = text.trim();
    if (!t) return;
    const user: Msg = { id: `${Date.now()}`, role: 'user', content: t };
    const reply: Msg = {
      id: `${Date.now() + 1}`, role: 'assistant',
      content: "Got it. In Phase 4 I'll answer using your real glucose/food data and can log food or vitals straight from chat — educational guidance only, never medical advice.",
    };
    setMessages((m) => [...m, user, reply]);
    setText('');
  }

  return (
    <Screen>
      <SafeAreaView edges={['top']} style={{ flex: 1 }}>
        <View style={{ padding: 16, borderBottomWidth: 1, borderBottomColor: c.border }}>
          <Body style={{ fontWeight: '700', fontSize: 18 }}>Coach</Body>
          <Muted>Ask anything · educational guidance, not medical advice</Muted>
        </View>
        <KeyboardAvoidingView style={{ flex: 1 }}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined} keyboardVerticalOffset={90}>
          <FlatList
            data={messages}
            keyExtractor={(m) => m.id}
            contentContainerStyle={{ padding: 16, gap: 10 }}
            renderItem={({ item }) => <Bubble role={item.role} content={item.content} />}
          />
          <View style={{ flexDirection: 'row', padding: 12, gap: 8, alignItems: 'center',
            borderTopWidth: 1, borderTopColor: c.border, backgroundColor: c.surface }}>
            <TextInput value={text} onChangeText={setText} placeholder="Message your coach…"
              placeholderTextColor={c.textMuted} onSubmitEditing={send}
              style={{ flex: 1, color: c.text, backgroundColor: c.surfaceAlt, borderRadius: 22,
                paddingHorizontal: 16, paddingVertical: 11 }} />
            <Pressable onPress={send}
              style={{ width: 44, height: 44, borderRadius: 22, backgroundColor: c.accent,
                alignItems: 'center', justifyContent: 'center' }}>
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
