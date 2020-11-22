import React from 'react';
import { Image, View } from 'react-native';

export default function Loading() {
  return (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
      <Image
        source={require('../assets/images/again.gif')}
        onLoad={this._cacheResourcesAsync}
      />
    </View>
  );
}
