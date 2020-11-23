import React from 'react';
import { TouchableOpacity } from 'react-native';
import { FontAwesome } from '@expo/vector-icons';

const GetPhotoBtn = ({ onPress }) => (
  <TouchableOpacity onPress={onPress}>
    <FontAwesome name="picture-o" color="black" size={30} />
  </TouchableOpacity>
);

export default GetPhotoBtn;
