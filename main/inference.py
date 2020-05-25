import argparse, math
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Tester
from utils.vis import vis_keypoints
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils.pose_utils import pixel2cam, process_bbox
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from PIL import Image
from matplotlib.patches import Rectangle
#python inference.py --gpu 0 --test_epoch 18 --path ../data/MSCOCO/images/val2017/000000425226.jpg --bbox 73.35,206.02,300.58,372.5
#python inference.py --gpu 0 --test_epoch 18 --image_id 8690 
#python inference.py --gpu 0 --test_epoch 18 --image_id 187144
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    # parser.add_argument('--path', type=str, dest='path')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--image_id', type=str, dest='image_id')
    args = parser.parse_args()
    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, rot, inv=False):
    src_w = src_width
    src_h = src_height
    src_center = np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def generate_patch_image(cvimg, bbox, do_flip, rot):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
    
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, cfg.input_shape[1], cfg.input_shape[0], rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(cfg.input_shape[1]), int(cfg.input_shape[0])), flags=cv2.INTER_LINEAR)

    img_patch = img_patch[:,:,::-1].copy()
    img_patch = img_patch.astype(np.float32)

    return img_patch, trans

def get_item(path, bbox):
    
    cvimg = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    plt.imshow(Image.open(path))
    plt.gca().add_patch(Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none'))
    plt.show()
    height, width, num_channels = cvimg.shape
    # bbox = [float(i) for i in bbox.split(',')]
    bbox = process_bbox(bbox, width, height)
    area = bbox[2]*bbox[3]
    
    img_patch, trans = generate_patch_image(cvimg, bbox, False, 0)
    tmp_img = img_patch.astype(np.uint8)
    plt.imshow(tmp_img)
    plt.show()
    color_scale = [1.0, 1.0, 1.0]
    for i in range(num_channels):
        img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)
    transform = transforms.Compose([\
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    img_patch = transform(img_patch)
    f = np.array([1500, 1500])
    k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*f[0]*f[1]/(area))]).astype(np.float32)        
    print(k_value)
    c = np.array([width*0.5, height*0.5])
    return img_patch, k_value, bbox, c
    #../data/MSCOCO/images/val2017/000000425226.jpg

def evaluate(pred, bbox, c):
    # image_id = gt['image_id']
    f = np.array([1500, 1500])
    # c = gt['c']
    # bbox = gt['bbox'].tolist()
    
    # restore coordinates to original space
    pred_root = pred[0]
    pred_root[0] = pred_root[0] / cfg.output_shape[1] * bbox[2] + bbox[0]
    pred_root[1] = pred_root[1] / cfg.output_shape[0] * bbox[3] + bbox[1]
    # print(image_id, f, c, bbox, pred_root)
    # back project to camera coordinate system
    pred_root = pixel2cam(pred_root[None,:], f, c)[0]
    # print(image_id, f, c, bbox, pred_root)
    # break;
    return bbox, pred_root

def get_bbox(image_id):
    db = COCO('../data/MSCOCO/annotations/person_keypoints_val2017.json')
    for i in db.anns.keys(): 
        if db.anns[i]['image_id'] == image_id: 
            return db.anns[i]['bbox'] 
    return -1

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.fastest = True
    cudnn.benchmark = True
    tester = Tester(args.test_epoch)
    print(args)
    # tester._make_batch_generator()
    tester._make_model()

    preds = []
    with torch.no_grad():
        # for itr, (input_img, cam_param) in enumerate(tqdm(tester.batch_generator)):
        bbox = get_bbox(int(args.image_id))
        #bbox[0]-=20
        path = "../data/MSCOCO/images/val2017/{:012d}.jpg".format(int(args.image_id))
        input_img, cam_param, bbox, c = get_item(path, bbox)
        # print("input image : ", input_img.size())
        # print(type(input_img), type(cam_param))
        input_img = input_img.reshape(-1, 3, 256, 256)
        cam_param = torch.tensor(cam_param).reshape(-1, 1)    
        
        # print("input image : ", input_img.size())
        # print(type(input_img), type(cam_param))
        
        coord_out = tester.model(input_img, cam_param)
        coord_out = coord_out.cpu().numpy()
        print(coord_out.shape)
        preds.append(coord_out)
            
    # evaluate
    preds = np.concatenate(preds, axis=0)
    print("pred: ", preds, "bbox : ", bbox)
    print(preds.shape)
    bbox, pred_root = evaluate(preds, bbox, c)    
    print("pred_root: ", pred_root, "bbox: ", bbox)
        
if __name__ == "__main__":
    main()
