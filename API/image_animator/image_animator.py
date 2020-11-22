import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from image_animator.sync_batchnorm import DataParallelWithCallback

from image_animator.modules.generator import OcclusionAwareGenerator
from image_animator.modules.keypoint_detector import KPDetector
from image_animator.animate import normalize_kp
from scipy.spatial import ConvexHull


# Model이 사용한 모든 매개변수 값을 지니고 있는 checkpoint 파일을 불러와 적용하는 메소드.
def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.full_load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector


# Parameter로 주어진 것들을 가지고 Image를 Video화 시키는 작업을 수행하는 메소드이다.
def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    # 'source_image'는 원본 Image, 'driving_video'는 목표 Animation을 제공할 영상이고, 'cpu'는 CPU를 우선 사용할 것인지에 대한 Boolean 값이다.
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def find_best_frame(source, driving, cpu=False):
    from image_animator import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


def image_animator(client_ip, time_flag, input_image):
    parser = ArgumentParser()
    # parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--config", default=r'../image_animator/config/vox-adv-256.yaml', help="path to config")
    # 사용할 기능의 모든 세부 동작들에 대한 설정값(= Parameter 값)들을 지닌 설정파일의 경로를 설정한다.
    parser.add_argument("--checkpoint", default=r'../image_animator/fom_checkpoints/vox-adv-cpk.pth.tar', help="path to checkpoint to restore")
    # Model이 사용했던 모든 변수값을 저장해 둔 checkpoint 파일의 경로를 설정한다.
    # parser.add_argument("--source_image", default=r'../image_animator/source_image/{}_{}.png'.format(client_ip, time_flag), help="path to source image")
    parser.add_argument("--source_image", default=input_image, help="path to source image")
    # 원본 Image를 지정한다.
    parser.add_argument("--driving_video", default=r'../image_animator/driving_video/source.mp4', help="path to driving video")
    # Animation의 기준이 될 영상을 지정한다.
    parser.add_argument("--result_video", default=r'static/videos/{}_{}.mp4'.format(client_ip, time_flag), help="path to output")
    # 출력될 결과물 Video의 경로를 설정한다.
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    # Key-point의 죄푯값의 절대/상대성을 설정한다.
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
    # Key-point 값들의 Convex Hull에 기준한 Movement Scale의 Adaptive 여부를 설정한다.
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")
    # Key-Point가 원본 Image의 것과 가장 일치하는 최적의 Frame들을 골라낼지의 여부를 설정한다.
    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
    # 최적의 Frame이 시작되는 부분을 수동으로 설정할 수 있다. 기본값은 None(없음)이다.
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    # CPU를 우선적으로 연산에 사용할지 여부에 대해 설정한다.

    parser.set_defaults(relative=True)  # Key-point의 죄푯값의 절대/상대성을 설정의 기본값을 설정한다.
    parser.set_defaults(adapt_scale=True)  # "Adapt Movement Scale" 에 대한 기본값을 설정한다.
    parser.set_defaults(cpu=False)  # CPU 우선 사용 여부에 대한 기본값을 설정한다.

    opt = parser.parse_args()  # 앞서 Argument로 적재해 둔 값들을 Parsing해 가져온다.

    source_image = imageio.imread(opt.source_image)  # 원본 Image 파일을 읽어들인다.
    reader = imageio.get_reader(opt.driving_video)  # 동작 기준 영상을 읽어들인다.
    fps = reader.get_meta_data()['fps']  # 기준 영상의 Frame-Rate 값을 변수에 대입해 가지고 있는다.
    driving_video = []  # 영상의 개별 프레임들을 지니고 있을 배열을 선언한다.
    try:
        for im in reader:
            driving_video.append(im)  # 동작 기준 영상의 프레임들을 배열에 적재한다.
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]  # 원본 Image의 해상도를 256 * 256으로 Resizing한다.
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]  # 동작 기준 영상의 해상도를 256 * 256으로 Resizing한다.
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    # Generator와 Detector를 Checkpoint로부터 CPU 우선 사용 여부에 대한 Parameter 값을 적용해 불러온다.

    if opt.find_best_frame or opt.best_frame is not None:
        # 최적의 Frame 찾아내기 기능을 활성화할 것이 설정됐거나 그 특정 시작점이 주어진 경우, 해당하는 최적 Frame들을 뽑아내는 부분이다.
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

    # os.remove(opt.source_image)
    return opt.result_video  # 결과물 영상을 호출측에 반환한다.


if __name__ == "__main__":
    image_animator()
