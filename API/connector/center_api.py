from image_2_style_gan.image_crossover import image_crossover
from image_animator.image_animator import image_animator

from flask import Flask, render_template, request
import base64
import json
import time
import cv2
import os

app = Flask(__name__)  # 'app'이라는 이름의 Flask Application 객체를 생성한다.


@app.route("/", methods=['GET', 'POST'])   # IP+포트번호만을 가지고 접속했을 때
def home():
    if request:
        print(request)
    # 'index.html'의 내용을 참조하여 Page를 구성해 출력한다.
    return render_template('index.html')


# 첫 화면에서 Image 파일을 제출하고 나면, 본 Url Page로 접속하게 된다.
@app.route("/let_me_shine", methods=['GET', 'POST'])
def let_me_shine():
    time_flag = time.strftime('%m%d-%H%M%S', time.localtime(time.time()))
    # 메소드 호출(= 접속) 당시의 시간 정보를 '월일-시분초' 형식으로 기억해 둔다. 이는 파일 이름의 중복 방지에 활용될 것이다.
    client_ip = request.remote_addr
    # 접속해 온 Client의 IP(v4) 주소를 기억해 둔다. 파일 이름 중복 혹은 혼선을 방지하기 위해 활용될 것이다.

    if request.method == 'POST':   # POST 방식으로 Request가 전달된 경우에의 동작이다.
        f = request.files['raw_img_file']   # Client가 첨부한 파일을 변수(객체)로 가져온다.
        # 첨부한 Image가 업로드한 파일명과 형식 그대로 일단 저장될 위치를 지정한다.
        file_name = r'../image_2_style_gan/img/{}'.format(f.filename)
        # 첨부한 Image가 png 형식으로 다시 저장될 경로와 이름을 지정한다.
        client_img_name = '../image_2_style_gan/img/{}_{}.png'.format(
            client_ip, time_flag)
        f.save(file_name)  # 변수에 받아들여 놓은 Image를 파일로 저장한다.

        # 앞서 저장한 업로드 Image를 openCV Library의 'imread'메소드를 이용해 다시 읽어들인다.
        cnv_buffer = cv2.imread(file_name)
        # 접속한 Client의 IP를 활용해 지정한 이름으로, 읽어들였던 Image를 png파일로 다시 기록한다.
        cv2.imwrite(client_img_name, cnv_buffer)
        os.remove(file_name)  # 기존의 원본 Image 파일은 삭제한다.

    input_image, output_image = image_crossover(client_ip, time_flag)
    # 'image_crossover'는 변경 요청 대상 Image에서 눈 부분만 Target처럼 바꿔주는 메소드이다.
    # 저장 파일명에 활용할 Client의 IP, 'time_flag'를 Parameter로 넘겨주고, 처리 전의 원본 Image와 처리 후의 결과물 Image의 '경로 + 파일명'을 반환받는다.
    output_video = image_animator(client_ip, time_flag, output_image)
    # 'image_animator'는 입력받은 Image를 특정한 Source 영상의 모션과 결합해 동영상으로 만들어 주는 메소드이다.
    # 저장 파일명에 활용할 Client의 IP, 'time_flag'와 재료가 될 Image를 Parameter로 넘겨주고 처리 후의 결과물 Image의 '경로 + 파일명'을 반환받는다.

    url_self = "http://121.138.83.1:45045/"
    data = {'results': {'imgID_1': base64.b64encode(open(input_image, 'rb').read()).decode('utf-8'),
                        'imgID_2': base64.b64encode(open(output_image, 'rb').read()).decode('utf-8'),
                        'imgID_3': '{}{}'.format(url_self, output_video)}}
    json_data = json.dumps(data)
    # return render_template('result.html', input_image_dir=input_image[7:], output_image_dir=output_image[7:], output_video_dir=output_video[7:]), json_data

    return json_data
    # 모든 Image 처리 과정이 완료되면, 'result.html'을 참조해 Page를 구성해 출력하고,
    # 각 scr를 받아들여 출력하는 부분에 해당되는 자료의 경로 및 파일명을 Parameter로 넘겨줘 Page에 출력할 수 있도록 한다.


if __name__ == "__main__":
    # __name__ == "__main__" 구문의 의미하는 것은 우선 [Module을 Command Prompt 등을 통해 "직접 실행하는 경우"]라는 의미이다.
    # 즉, 이는 특정 Module을 타 Module에서 Import를 통해 활용하는 경우와 구분지을 수 있는 수단이 된다.

    print("Server Start")  # 메시지를 출력해 Server의 작동 시작을 알린다.
    # 생성한 'app' 객체를 Parameter 값들을 이용해 구동한다.
    app.run('121.138.83.1', port=45045, debug=True)
    # 위에서 활용된 Parameter는 IP(v4)와 포트 번호, 디버그 모드의 수행 여부에 대한 Boolean 값이다.
