import React from 'react';
import { Camera } from 'expo-camera';
import styled from 'styled-components';
import { Dimensions, Image, View } from 'react-native';

const { width, height } = Dimensions.get('window');

export default () => {
  return (
    <View
      style={{
        marginTop: 80,
        width: 200,
        height: 250,
        borderRadius: 100 / 1.1,
        borderWidth: 5,
        opacity: 0.5,
        borderColor: 'white',
        backgroundColor: 'transparent',
        position: 'absolute',
      }}
    />
  );
};
