import React from 'react';
import { TouchableOpacity } from 'react-native';
import { FontAwesome } from '@expo/vector-icons';

const TransferBtn = ({ onPress }) => (
  <TouchableOpacity onPress={onPress}>
    <FontAwesome name="check-circle" color="black" size={40} />
  </TouchableOpacity>
);

export default TransferBtn;
