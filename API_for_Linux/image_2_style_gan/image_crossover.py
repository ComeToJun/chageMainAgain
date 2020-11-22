import argparse
import os
import shutil
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from image_2_style_gan.align_images import align_images
from image_2_style_gan.mask_maker import mask_maker
from image_2_style_gan.perceptual_model import VGG16_for_Perceptual
from image_2_style_gan.read_image import image_reader
from image_2_style_gan.stylegan_layers import G_mapping, G_synthesis
from torchvision.utils import save_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def image_crossover(rand_uuid, client_img_name):
    MEDIUM_IMAGE_DIR = '../image_2_style_gan/images/medium/'
    if os.path.isdir(MEDIUM_IMAGE_DIR) is not True:
        os.makedirs(MEDIUM_IMAGE_DIR, exist_ok=True)

    FINAL_IMAGE_DIR = '../image_2_style_gan/images/final/'
    if os.path.isdir(FINAL_IMAGE_DIR) is not True:
        os.makedirs(FINAL_IMAGE_DIR, exist_ok=True)

    # MEDIUM_IMAGE_DIR_ELEM = MEDIUM_IMAGE_DIR.split("/")
    # DIR = MEDIUM_IMAGE_DIR_ELEM[0]
    # for ELEM in MEDIUM_IMAGE_DIR_ELEM:
    #     if ELEM != '' and os.path.isdir(DIR) is not True:
    #         DIR += ELEM
    #         os.mkdir(DIR)
    #         DIR += '/'

    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('--batch_size', default=6, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--resolution', default=1024, type=int)
    parser.add_argument('--src_im1', default="../image_2_style_gan/source_image/target/")
    parser.add_argument('--src_im2', default="../image_2_style_gan/images/medium/")
    parser.add_argument('--mask', default="../image_2_style_gan/images/mask/")
    parser.add_argument('--weight_file', default="../image_2_style_gan/weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt", type=str)
    parser.add_argument('--iteration', default=150, type=int)

    args = parser.parse_args()
    # if client_img_name == '':
    #     raw_image_names = os.listdir(r'../image_2_style_gan/img/')
    # else:
    #     raw_image_names = client_img_name
    # raw_image_names = r'../image_2_style_gan/img/'
    aligned_image_names = align_images(args.src_im2)

    try:
        if not aligned_image_names:
            print('\nNo raw-image detected. Process proceeds without alignment.')
            aligned_image_names = [args.src_im2 + os.listdir(args.src_im2)[0]]

        mask_maker(aligned_image_names, args.mask)

        ingredient_name = args.src_im2 + os.listdir(args.src_im2)[0]
        target_name = args.src_im1 + os.listdir(args.src_im1)[0]
    except IndexError as e:
        print("\nMissing file(s).\nCheck if all of source images prepared properly and try again.")
        print(f"Aligned_image_names function: {e}")
        os.remove(client_img_name)
        sys.exit(1)

    try:
        mask_name = args.mask + os.listdir(args.mask)[0]
    except Exception as e:
        shutil.copyfile('../image_2_style_gan/source_image/ref_mask/ref_mask.png', '{}ref_mask.png'.format(args.mask))
        mask_name = args.mask + os.listdir(args.mask)[0]

    FINAL_IMAGE_DIR = FINAL_IMAGE_DIR + str(rand_uuid) + '/'
    if os.path.isdir(FINAL_IMAGE_DIR) is not True:
        os.mkdir(FINAL_IMAGE_DIR)

    # file_names = []
    final_name = FINAL_IMAGE_DIR + str(rand_uuid) + '.png'

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
        if i % 10 == 0:
            print("iter{}: loss -- {},  loss0 --{},  loss1 --{}".format(i,loss_np,loss_0,loss_1))
            # file_name = "{}_{}_{}.png".format(MEDIUM_IMAGE_DIR, client_ip, i)
            # save_image(synth_img.clamp(0, 1), file_name)
            # if i > 10:
            #     file_names.append(file_name)
            # np.save(r"../image_2_style_gan/latent_W/crossover_{}.npy".format(client_ip), dlatent.detach().cpu().numpy())
        elif i == (args.iteration - 1):
            save_image(synth_img.clamp(0, 1), final_name)

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

    origin_name = '{}{}_origin.png'.format(FINAL_IMAGE_DIR, str(rand_uuid))
    os.replace(ingredient_name, origin_name)
    # os.remove(ingredient_name)
    # os.replace(mask_name, '{}Used_{}_mask.png'.format(FINAL_IMAGE_DIR, time_flag))
    os.remove(mask_name)
    # shutil.copyfile(target_name, '{}Used_{}_target.png'.format(FINAL_IMAGE_DIR, time_flag))

    print("Complete ---------------------------------------------------------------------------------------------")

    return origin_name, final_name


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