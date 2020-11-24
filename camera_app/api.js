import axios from 'axios';
import { useState } from 'react';
import { Alert } from 'react-native';

const URL = 'http://222.106.22.97:45055/let_me_shine';

let tempResult = [];

const getResults = (imgObj) => {
  console.log('[3] Image Transfer Start!');

  const origin = `data:image/png;base64,${imgObj.imgID_1}`;
  const after = `data:image/png;base64,${imgObj.imgID_2}`;

  console.log('[3] Image Trasfer Complete!');
  tempResult = [origin, after];
};

const getResultURL = async (url) => {
  try {
    console.log('[2] Get Start!');

    await axios
      .get(url) // get으로 해당 url에 접근
      .then((res) => getResults(res.data.results))
      // 모델에서 계산이 완료된 결과 사진(base64 형태로 되어 있음)과 영상(mp4)을
      // getResults 함수로 보낸다.
      .catch((err) => console.log(`Get axios error: ${err}`));

    console.log('[2] Get End!');
  } catch (e) {
    console.log(`getResultURL Error: ${e}`);
  }
};

export const imageTransfer = async (photo) => {
  try {
    console.log('[1] Post Start!');
    const config = {
      // 보내는 파일의 타입 설정
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    };

    await axios
      .post(URL, { label: 'Image', text: photo }, config) // 해당 URL로 POST
      .then((res) => getResultURL(res.data))
      // POST의 결과(res)로부터 모델 결과 위치(res.data) 얻음
      // 이를 getResultURL 함수로 보낸다.
      .catch((err) => {
        console.log(`Post axios error: ${err}`);
        error = false;
        Alert.alert(
          '사람을 찍어주세요🤣',
          '만약 사람이라면 눈을 조금만 더 크게 떠주세요😘'
        );
      });
    console.log('[1] Post End!');
  } catch (e) {
    console.log(`imageTransfer Error: ${e}`);
  } finally {
    const result = tempResult;
    if (result.length === 2) {
      return result;
    } else {
      return false;
    }
  }
};
