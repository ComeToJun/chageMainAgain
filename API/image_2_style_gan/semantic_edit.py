
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from image_2_style_gan.stylegan_layers import G_mapping, G_synthesis
from torchvision.utils import save_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():
        parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
        parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
        parser.add_argument('--resolution',default=1024,type=int)
        parser.add_argument('--weight_file',default="weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt",type=str)
        parser.add_argument('--latent_file',default="latent_W/0.npy")







        args=parser.parse_args()

        g_all = nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        #('truncation', Truncation(avg_latent)),
        ('g_synthesis', G_synthesis(resolution=args.resolution))    
        ]))




        g_all.load_state_dict(torch.load(args.weight_file, map_location=device))
        g_all.eval()
        g_all.to(device)


        g_mapping,g_synthesis=g_all[0],g_all[1]

        boundary_name=["stylegan_ffhq_gender_w_boundary.npy","stylegan_ffhq_age_w_boundary.npy","stylegan_ffhq_pose_w_boundary.npy","stylegan_ffhq_eyeglasses_w_boundary.npy","stylegan_ffhq_smile_w_boundary.npy"]
        semantic=["gender","age","pose","eye_glass","smile"]



        for i in range(5):
            latents_0=np.load(args.latent_file)
            latents_0=torch.tensor(latents_0).to(device)#.unsqueeze(0)
            boundary=np.load("boundaries/"+boundary_name[i])
            make_morph(boundary,i,latents_0,g_synthesis,semantic)



         
def make_morph(boundary,i,latents_0,g_synthesis,semantic):
        boundary=boundary.reshape(1,1,-1)


        linspace = np.linspace(-3, 3, 10)
        linspace=linspace.reshape(-1,1,1).astype(np.float32)

        boundary=torch.tensor(boundary).to(device)
        linspace=torch.tensor(linspace).to(device)





        latent_code=latents_0+linspace*boundary
        latent_code=latent_code.to(torch.float)

        with torch.no_grad():
            synth_img=g_synthesis(latent_code)
        synth_img = (synth_img + 1.0) / 2.0

        save_image(synth_img,"save_image/boundary/{}.png".format(semantic[i]))





if __name__ == "__main__":
        main() 


