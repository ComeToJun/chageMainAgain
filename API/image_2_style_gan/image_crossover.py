import argparse
import shutil
import sys
import os
from collections import OrderedDict
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.optim as optim

from image_2_style_gan.align_images import align_images
from image_2_style_gan.mask_maker import mask_maker
from image_2_style_gan.perceptual_model import VGG16_for_Perceptual
from image_2_style_gan.read_image import image_reader
from image_2_style_gan.stylegan_layers import G_mapping, G_synthesis

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # GPU를 활용할 수 있다면 활용할 수 있도록 한다.


def image_crossover(client_ip, time_flag):
    # 외부로부터 접속한 Client의 IP와 Image 처리를 개시할 당시의 시간 정보를 받아온다.
    # 이 둘은 파일명의 중복 방지와 Client간 구분을 위해 활용될 것이다.
    MEDIUM_IMAGE_DIR = r'../image_2_style_gan/save_image/crossover/'  # Image처리 과정 도중 생성되는 중간 산물들을 임시 저장할 경로를 설정한다.
    FINAL_IMAGE_DIR = r'static/images/'  # 처리 완료된 Image를 파일로 저장할 경로를 설정한다.

    if os.path.isdir(FINAL_IMAGE_DIR) is not True:
        os.makedirs(FINAL_IMAGE_DIR, exist_ok=True)
        # 경로에 해당하는 폴더가 존재하지 않는 것이 하나라도 있을 경우, 자동으로 해당 경로 전체에 해당하는 모든 폴더를 생성한다.
        # 이미 존재하는 폴더에 대해서는 오류 발생을 무시하고 과정을 진행한다.

    if os.path.isdir(MEDIUM_IMAGE_DIR) is not True:
        os.makedirs(MEDIUM_IMAGE_DIR, exist_ok=True)
        # 경로에 해당하는 폴더가 존재하지 않는 것이 하나라도 있을 경우, 자동으로 해당 경로 전체에 해당하는 모든 폴더를 생성한다.
        # 이미 존재하는 폴더에 대해서는 오류 발생을 무시하고 과정을 진행한다.

    # Command Prompt 등을 통해 입력받는 경우를 위해 아래와 같이 'ArgumentParser'를 통해 Argument의 형태로 Parameter 값들을 입력받을 수 있다.
    # 하지만, 현재 사용자로부터 직접 입력받도록 돼있는 항목은 원본 Image 파일 뿐이므로, 나머지는 'default'값을 설정하는 형태로,
    # 지금은 사실상 메소드의 내부에서 변수들을 초기 값을 주며 선언하는 것과 별 다를 바 없는 구조이다.
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('--batch_size', default=6, help='Batch size for generator and perceptual model', type=int)
    # 입력받은 Image를 Model에 적용시킬 때 수행할 작업의 Batch Size를 설정한다.
    parser.add_argument('--resolution', default=1024, type=int)
    # 결과물이 출력될 해상도를 설정한다.
    parser.add_argument('--src_im1', default=r"../image_2_style_gan/source_image/target/")
    # 목표 Image 즉, 원본 Image에서 바뀔 부분의 기준이 되는 Image 파일의 경로를 설정한다.
    parser.add_argument('--src_im2', default=r"../image_2_style_gan/source_image/ingredient/")
    # 변형시킬 원본이 될 Image의 경로를 설정한다.
    parser.add_argument('--mask', default=r"../image_2_style_gan/source_image/mask/")
    # 원본에서 바꿀 부분만을 지정할 수 있도록 할 Masking 파일의 경로를 설정한다.
    parser.add_argument('--weight_file', default=r"../image_2_style_gan/weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt", type=str)
    # Model이 이미 학습한 가중치들을 지닌 파일의 경로를 설정한다.
    parser.add_argument('--iteration', default=150, type=int)
    # 처리를 반복할 횟수를 설정한다.

    args = parser.parse_args()  # 'parser' 객체에 적재한 Argument들을 Parsing해와 변수에 담아 둔다.
    # if client_img_name == '':
    #     raw_image_names = os.listdir(r'../image_2_style_gan/img/')
    # else:
    #     raw_image_names = client_img_name
    # raw_image_names = r'../image_2_style_gan/img/'
    aligned_image_names = align_images(args.src_im2)
    # 정렬이 완료된 Image를 저장할 경로를 Parameter로 넘겨주며, 메소드를 작동시키고 정렬된 파일(들)의 이름을 반환받는다.

    try:
        # Raw-Image(얼굴이 정렬되지 않은 Image)가 존재하지 않는 경우, 안내 메시지를 띄우고 그대로 진행시키는 부분이다.
        if not aligned_image_names:
            print('\nNo raw-image detected. Process proceeds without alignment.')
            aligned_image_names = [args.src_im2 + os.listdir(args.src_im2)[0]]
            # 메소드로부터 파일명 목록을 넘겨받지 못한(=빈 목록을 넘겨받은) 대신, 이미 경로 안에 존재하는 재료를 사용하도록 한다.
            # 다만, 현재 API에 연동된 상태에선 무조건 Raw-Image를 넘겨받아 진행하게 돼있기 때문에,
            # 해당 오류가 발생하는 경우엔 아예 정상적인 진행이 불가할 가능성이 크다.

        mask_maker(aligned_image_names, args.mask)  # 얼굴을 정렬한 Image를 획득하면, 이를 가지고 Masking Image를 생성한다.

        ingredient_name = args.src_im2 + os.listdir(args.src_im2)[0]  # 원본 Image의 경로 + 파일명을 결합한다.
        target_name = args.src_im1 + os.listdir(args.src_im1)[0]  # 목표 Image의 경로 + 파일명을 결합한다.
    except IndexError as e:
        # 세 가지의 필수 요소(원본 Image, Masking Image, 대상 Image) 중 하나라도 빠진 것이 있으면 메시지로 경고하고 Process를 중단시킨다.
        print("\nMissing file(s).\nCheck if all of source images prepared properly and try again.")
        sys.exit(1)

    try:
        mask_name = args.mask + os.listdir(args.mask)[0]  # Masking Image의 경로 + 파일명을 결합한다.
        # 만약 여기서 오류가 발생하면 이는 해당 내용이 존재하지 않는다(= Masking Image를 만들어내지 못했다)는 의미이다.
    except Exception as e:
        # Masking Image를 제작하지 못한 경우, 기존의 범용 Masking Image를 복사해 가져와 사용하게 된다.
        shutil.copyfile(r'../image_2_style_gan/source_image/ref_mask/ref_mask.png', '{}ref_mask.png'.format(args.mask))
        mask_name = args.mask + os.listdir(args.mask)[0]

    FINAL_IMAGE_DIR = FINAL_IMAGE_DIR + client_ip + time_flag + '/'  # 결과물 Image를 저장할 경로를 Client의 IP와 작업 개시 시간을 이용해 조합한다.
    if os.path.isdir(FINAL_IMAGE_DIR) is not True:
        os.mkdir(FINAL_IMAGE_DIR)  # 해당 경로 이름으로 실제 폴더를 생성한다.

    # file_names = []
    final_name = FINAL_IMAGE_DIR + time_flag + '.png'  # 결과물 Image의 파일명을 조합한다.

    # ============ 아래는 학습된 Model을 가지고 Network을 통과시켜 Image를 얻어내는 부분이다. ================
    g_all = nn.Sequential(OrderedDict([
    ('g_mapping', G_mapping()),
    #('truncation', Truncation(avg_latent)),
    ('g_synthesis', G_synthesis(resolution=args.resolution))
    ]))

    g_all.load_state_dict(torch.load(args.weight_file, map_location=device))
    g_all.eval()
    g_all.to(device)
    g_mapping,g_synthesis=g_all[0],g_all[1]

    img_0=image_reader(target_name) #(1,3,1024,1024) -1~1
    img_0=img_0.to(device)

    img_1=image_reader(ingredient_name)
    img_1=img_1.to(device) #(1,3,1024,1024)

    blur_mask0=image_reader(mask_name).to(device)
    blur_mask0=blur_mask0[:,0,:,:].unsqueeze(0)
    blur_mask1=blur_mask0.clone()
    blur_mask1=1-blur_mask1

    MSE_Loss=nn.MSELoss(reduction="mean")
    upsample2d=torch.nn.Upsample(scale_factor=0.5, mode='bilinear')

    img_p0=img_0.clone() #resize for perceptual net
    img_p0=upsample2d(img_p0)
    img_p0=upsample2d(img_p0) #(1,3,256,256)

    img_p1=img_1.clone()
    img_p1=upsample2d(img_p1)
    img_p1=upsample2d(img_p1) #(1,3,256,256)

    perceptual_net=VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device) #conv1_1,conv1_2,conv2_2,conv3_3
    dlatent=torch.zeros((1,18,512),requires_grad=True,device=device)
    optimizer=optim.Adam({dlatent},lr=0.01,betas=(0.9,0.999),eps=1e-8)

    loss_list = []

    print("Start ---------------------------------------------------------------------------------------")
    for i in range(args.iteration):
        optimizer.zero_grad()
        synth_img=g_synthesis(dlatent)
        synth_img = (synth_img + 1.0) / 2.0
        loss_wl0=caluclate_loss(synth_img,img_0,perceptual_net,img_p0,blur_mask0,MSE_Loss,upsample2d)
        loss_wl1=caluclate_loss(synth_img,img_1,perceptual_net,img_p1,blur_mask1,MSE_Loss,upsample2d)
        loss=loss_wl0+loss_wl1
        loss.backward()

        optimizer.step()

        loss_np=loss.detach().cpu().numpy()
        loss_0=loss_wl0.detach().cpu().numpy()
        loss_1=loss_wl1.detach().cpu().numpy()

        loss_list.append(loss_np)
        if i % 10 == 0:  # Iteration 도중 적당한 단위마다 산출된 오차를 출력한다.
            print("iter{}: loss -- {},  loss0 --{},  loss1 --{}".format(i,loss_np,loss_0,loss_1))
            # file_name = "{}_{}_{}.png".format(MEDIUM_IMAGE_DIR, client_ip, i)
            # save_image(synth_img.clamp(0, 1), file_name)
            # if i > 10:
            #     file_names.append(file_name)
            # np.save(r"../image_2_style_gan/latent_W/crossover_{}.npy".format(client_ip), dlatent.detach().cpu().numpy())
        elif i == (args.iteration - 1):
            save_image(synth_img.clamp(0, 1), final_name)  # 과정이 종료됐을 때의 결과물을 지정한 파일명으로 저장한다.

    # gif_buffer = []
    # durations = []

    # gif_buffer.append(Image.open(ingredient_name))
    # durations.append(3.00)
    #
    # for file in file_names:
    #     gif_buffer.append(Image.open(file))
    #     durations.append(0.04)
    #
    # gif_buffer.append((Image.open(final_name)))
    # durations.append(3.00)

    # imageio.mimsave('{}{}.gif'.format(FINAL_IMAGE_DIR, time_flag), gif_buffer, duration=durations)
    # del gif_buffer

    # for file in os.listdir(r'../image_2_style_gan/save_image/crossover/'):
    #     dir = r'../image_2_style_gan/save_image/crossover/' + file
    #     os.remove(dir)

    origin_name = '{}{}_origin.png'.format(FINAL_IMAGE_DIR, time_flag)
    # 원본 이미지도 비교에 이용하기 위해 가져올 것이므로, 그 파일명을 설정한다.
    os.replace(ingredient_name, origin_name)  # 원본 파일을 재명명하며 가져온다.
    # os.remove(ingredient_name)
    # os.replace(mask_name, '{}Used_{}_mask.png'.format(FINAL_IMAGE_DIR, time_flag))
    os.remove(mask_name)  # Masking Image는 일회용이므로 삭제한다.(Reference Mask인 경우에는 원본이 다른 폴더에서 존재하기 때문에 그것은 계속 남아있다.)
    # shutil.copyfile(target_name, '{}Used_{}_target.png'.format(FINAL_IMAGE_DIR, time_flag))

    print("Complete ---------------------------------------------------------------------------------------------")

    return origin_name, final_name  # 작업이 완료되면 원본 파일과 결과물 파일의 경로와 이름을 호출측에 반환한다.


# ========= 오차를 계산하는 메소드이다. ===========
def caluclate_loss(synth_img,img,perceptual_net,img_p,blur_mask,MSE_Loss,upsample2d): #W_l
    #calculate MSE Loss
    mse_loss=MSE_Loss(synth_img*blur_mask.expand(1,3,1024,1024),img*blur_mask.expand(1,3,1024,1024)) # (lamda_mse/N)*||G(w)-I||^2
    #calculate Perceptual Loss
    real_0,real_1,real_2,real_3=perceptual_net(img_p)
    synth_p=upsample2d(synth_img) #(1,3,256,256)
    synth_p=upsample2d(synth_p)
    synth_0,synth_1,synth_2,synth_3=perceptual_net(synth_p)
    #print(synth_0.size(),synth_1.size(),synth_2.size(),synth_3.size())

    perceptual_loss=0
    blur_mask=upsample2d(blur_mask)
    blur_mask=upsample2d(blur_mask) #(256,256)

    perceptual_loss+=MSE_Loss(synth_0*blur_mask.expand(1,64,256,256),real_0*blur_mask.expand(1,64,256,256))
    perceptual_loss+=MSE_Loss(synth_1*blur_mask.expand(1,64,256,256),real_1*blur_mask.expand(1,64,256,256))
    blur_mask=upsample2d(blur_mask)
    blur_mask=upsample2d(blur_mask) #(64,64)
    perceptual_loss+=MSE_Loss(synth_2*blur_mask.expand(1,256,64,64),real_2*blur_mask.expand(1,256,64,64))
    blur_mask=upsample2d(blur_mask) #(64,64)
    perceptual_loss+=MSE_Loss(synth_3*blur_mask.expand(1,512,32,32),real_3*blur_mask.expand(1,512,32,32))

    return mse_loss+perceptual_loss


if __name__ == "__main__":
    image_crossover()
