import React, { useState, useEffect } from 'react';
import { Image, View } from 'react-native';

export default function ProgresLoading() {
  const [isReady, setIsReady] = useState(true);
  useEffect(() => {
    setTimeout(() => {
      setIsReady(true);
    }, 3000);
  }, []);

  return isReady ? (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
      <Image
        source={require('../assets/images/again.gif')}
        onLoad={this._cacheResourcesAsync}
      />
    </View>
  ) : (
    <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
      <Image source={require('../assets/images/3.png')} />
    </View>
  );
}
