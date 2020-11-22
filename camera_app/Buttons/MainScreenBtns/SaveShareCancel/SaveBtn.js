import React from 'react';
import { TouchableOpacity } from 'react-native';
import { FontAwesome } from '@expo/vector-icons';

const SaveBtn = ({ onPress }) => (
  <TouchableOpacity onPress={onPress}>
    <FontAwesome name="save" color="black" size={40} />
  </TouchableOpacity>
);

export default SaveBtn;
