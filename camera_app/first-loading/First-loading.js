import React from 'react';
import { Image, View } from 'react-native';
import Loading from './Loading';

export default class firstLoading extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isReady: true,
    };
  }
  componentDidMount = async () => {
    setTimeout(
      () =>
        this.setState({
          isReady: false,
        }),
      1600
    );
  };

  render() {
    const { isReady } = this.state;

    return isReady ? (
      <Loading />
    ) : (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
        <Image source={require('../assets/images/3.png')} />
      </View>
    );
  }
}
