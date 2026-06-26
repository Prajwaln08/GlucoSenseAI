import { ActivityIndicator, View } from 'react-native';
import { useColors } from '@/lib/theme';

// Splash. The Guard in _layout redirects to /(auth)/login, /onboarding, or /(tabs)
// once auth state is loaded.
export default function Index() {
  const c = useColors();
  return (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: c.bg }}>
      <ActivityIndicator color={c.accent} size="large" />
    </View>
  );
}
