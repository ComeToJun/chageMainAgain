import React from 'react';
import { TouchableOpacity } from 'react-native';
import { FontAwesome } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';

export default GetPhotoBtn = ({ onPress }) => {
  // const getPhotos = () =>
  //   (photo = ImagePicker.launchImageLibraryAsync({
  //     allowsEditing: false,
  //     quality: 1,
  //     base64: true,
  //   }));

  // // console.log(getPhotos);
  // if (!photo.uri) {
  //   setHasPermission(true);
  // } else {
  //   setImage(photo.uri);
  //   setImageSelected(true);
  //   setImageComeback(true);
  // }
  // currentPhoto = photo.base64;

  return (
    <TouchableOpacity onPress={onPress}>
      <FontAwesome name="picture-o" color="black" size={30} />
    </TouchableOpacity>
  );
};
