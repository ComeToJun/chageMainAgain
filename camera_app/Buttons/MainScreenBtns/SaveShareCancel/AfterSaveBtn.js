import React, { useRef } from 'react';
import {
  Animated,
  Text,
  View,
  StyleSheet,
  TouchableOpacity,
  Alert,
} from 'react-native';
import { AntDesign } from '@expo/vector-icons';
export default function AfterSaveBtn() {
  const fade = useRef(new Animated.Value(0)).current;
  const fadeOut = () => {
    Animated.timing(fade, {
      toValue: 1,
      duration: 100,
      useNativeDriver: true,
    }).start(() => {
      Animated.timing(fade, {
        toValue: 0,
        duration: 1000,
        useNativeDriver: true,
      }).start();
    });
  };
  return (
    <View style={styles.container}>
      <Animated.View
        style={[
          styles.fadingContainer,
          {
            opacity: fade,
          },
        ]}
      >
        <Text style={styles.fadingText}>저장 완료!</Text>
      </Animated.View>
      <View style={styles.buttonRow}>
        <Alert fadeOut />
      </View>
    </View>
  );
}
const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 11,
  },
  fadingContainer: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    backgroundColor: 'lightgray',
  },
  fadingText: {
    fontSize: 20,
    textAlign: 'center',
    margin: 20,
  },
  buttonRow: {
    flexDirection: 'row',
    marginVertical: 16,
  },
});
