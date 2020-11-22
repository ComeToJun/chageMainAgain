def mask_maker(aligned_image_names, mask_dir):
# def mask_maker(fileDir + fileName):
    from PIL import Image
    import cv2

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # img = cv2.imread(fileDir + fileName)
    img = cv2.imread(aligned_image_names[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        for x, y, w, h in faces:                       # 얼굴 x좌표, y좌표, 가로길이, 세로길이
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face_gray = gray[y: y + h, x: x + w]
            face_color = img[y: y + h, x: x + w]

            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=2.3, minSize=(85, 85), maxSize=(95, 95))
            # 눈만 잘 찾도록 값 조정

            for (ex, ey, ew, eh) in eyes:               # 눈 x좌표, y좌표, 가로길이, 세로길이
                cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        eyes_loc = eyes + [[x, y, 0, 0]]                # 눈의 좌표가 얼굴 내부 기준이므로 얼굴좌표를 더해줌

        mask_loc = eyes_loc[eyes_loc[:, 1] < 500] - [[50, 40, 0, 0]]        # 눈이 있을 높이 내에서만 (입 인식 방지), 마스킹 잘 되도록 적절한 위치로 조정

        # Mask 생성
        mask = Image.open(r'../image_2_style_gan/mask_origin/mask_circle_blur.jpg')                 # 이미지 파일과 동일한 디렉토리
        resized_mask = mask.resize((mask_loc[0, 2] * 2, mask_loc[0, 2] * 2))        # 인식된 눈 사이즈 * 2

        back = Image.new("L", (1024, 1024))                                 # 검정색 1024 x 1024 배경 생성
        for index in range(len(mask_loc)):
            if len(mask_loc) == 1:                                          # 눈이 하나만 인식되면 중앙을 기준으로 y축 대칭 생성
                back.paste(im=resized_mask, box=(mask_loc[index, 0], mask_loc[index, 1]))
                back.paste(im=resized_mask, box=(1024 - mask_loc[index, 0] - mask_loc[index, 2] * 2, mask_loc[index, 1]))
            else:
                back.paste(im=resized_mask, box=(mask_loc[index, 0], mask_loc[index, 1]))
        back.save(mask_dir + 'Mask.png')                          # Mask 파일 생성

    except IndexError:
        pass    # 눈이 인식되지 않을 시 pass
    except TypeError:
        pass    # 눈이 인식되지 않을 시 pass
    except UnboundLocalError:
        pass    # 눈이 인식되지 않을 시 pass


# def mask_main():
    # import argparse
    # parser = argparse.ArgumentParser(description='MaskMaker')
    # parser.add_argument('--path', default="source_image/")
    # parser.add_argument('--filename', default="")
    #
    # args = parser.parse_args()
    #
    # mask_maker(args.path, args.filename)


if __name__ == "__main__":
    mask_maker()
