/** @format */

import * as ImagePicker from 'expo-image-picker';

const getPhotos = async () => {
	const photo = await ImagePicker.launchImageLibraryAsync({
		allowsEditing: false,
		quality: 1,
		base64: true,
	});
	if (!photo.uri) {
		setHasPermission(true);
	} else {
		setImage(photo.uri);
		setImageSelected(true);
		setImageComeback(true);
	}
	currentPhoto = photo.base64;
};

export default getPhotos;
