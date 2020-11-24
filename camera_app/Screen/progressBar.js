import { StatusBar } from 'expo-status-bar';
import React, { useState, useRef, useEffect } from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import ProgresLoading from './progresLoading';

const Progress = ({ step, steps, height }) => {
  const [width, setWidth] = useState(0);
  const animatedValue = useRef(new Animated.Value(-1000)).current;
  const reactive = useRef(new Animated.Value(-1000)).current;

  useEffect(() => {
    Animated.timing(animatedValue, {
      toValue: reactive,
      donation: 300,
      useNativeDriver: true,
    }).start();
  }, []);

  useEffect(() => {
    reactive.setValue(-width + (width * step) / steps);
  }, [step, width]);

  return (
    <>
      <Text
        style={{
          textAlign: 'center',
          fontSize: 15,
          fontWeight: 'bold',
          marginBottom: 8,
        }}
      >
        {step}/{steps}
      </Text>
      <View
        onLayout={(e) => {
          const newWidth = e.nativeEvent.layout.width;

          setWidth(newWidth);
        }}
        style={{
          height,
          backgroundColor: 'black',
          borderRadius: height,
          overflow: 'hidden',
        }}
      >
        <Animated.View
          style={{
            height,
            width: '100%',
            borderRadius: height,
            backgroundColor: 'lightgray',

            left: 0,
            top: 0,
            transform: [
              {
                translateX: animatedValue,
              },
            ],
          }}
        />
      </View>
    </>
  );
};

export default function App() {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((index + 10) % (100 + 10));
    }, 1000);

    return () => {
      clearInterval(interval);
    };
  }, [index]);
  return (
    <View style={styles.container}>
      <ProgresLoading />
      <StatusBar hidden />
      <Progress step={index} steps={100} height={30} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    justifyContent: 'center',
    padding: 20,
  },
});
