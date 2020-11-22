import React from 'react';
import { TouchableOpacity } from 'react-native';
import { Entypo } from '@expo/vector-icons';

const CancelBtn = ({ onPress }) => (
  <TouchableOpacity onPress={onPress}>
    <Entypo name="circle-with-cross" color="black" size={40} />
  </TouchableOpacity>
);

export default CancelBtn;
