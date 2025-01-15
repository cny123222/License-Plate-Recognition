from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg_mnet
from prior_box import PriorBox
from py_cpu_nms import py_cpu_nms
import cv2
from models.retina import Retina
from box_utils import decode, decode_landm
import time


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    #unused_pretrained_keys = ckpt_keys - model_keys
    #missing_keys = model_keys - ckpt_keys
    #print('Missing keys:{}'.format(len(missing_keys)))
    #print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    #print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    #print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage, weights_only= True)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device), weights_only= True)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



def extract_retina(image):
    trained_model = './weights/mobilenet0.25_epoch_20_ccpd.pth'
    cpu = False
    confidence_threshold = 0.02
    top_k = 1000
    nms_threshold = 0.4
    keep_top_k = 500
    save_image = True
    vis_thres = 0.5
    torch.set_grad_enabled(False)
    cfg = cfg_mnet
    net = Retina(cfg=cfg, phase='test')
    net = load_model(net, trained_model, cpu)
    net.eval()
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    resize = 1

    # testing begin
    for i in range(1):
        
        img_raw = cv2.imread(image, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        #print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        # print('priorBox time: {:.4f}'.format(time.time() - tic))
        # show image
        if save_image:
            for b in dets:
                if b[4] < vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                #print(text)
                b = list(map(int, b))
                #cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                #cv2.putText(img_raw, text, (cx, cy),
                            #cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                w = int(x2 - x1 + 1.0)
                h = int(y2 - y1 + 1.0)
                img_box = np.zeros((h, w, 3))
                img_box = img_raw[y1:y2 + 1, x1:x2 + 1, :]
                # cv2.imshow("img_box",img_box)  
                
                new_x1, new_y1 = b[9] - x1, b[10] - y1
                new_x2, new_y2 = b[11] - x1, b[12] - y1
                new_x3, new_y3 = b[7] - x1, b[8] - y1
                new_x4, new_y4 = b[5] - x1, b[6] - y1
                #print(new_x1, new_y1)
                #print(new_x2, new_y2)
                #print(new_x3, new_y3)
                #print(new_x4, new_y4)
                        
                # 定义对应的点
                points1 = np.float32([[new_x1, new_y1], [new_x2, new_y2], [new_x3, new_y3], [new_x4, new_y4]])
                points2 = np.float32([[0, 0], [188, 0], [0, 48], [188, 48]])
                
                # 计算得到转换矩阵
                M = cv2.getPerspectiveTransform(points1, points2)
                
                # 实现透视变换转换
                processed = cv2.warpPerspective(img_box, M, (188, 48))
                
                # 显示原图和处理后的图像
                #cv2.imshow("processed", processed)  


                if processed is None:
                    return False
                else:
                    name = "plate.jpg"
                    cv2.imwrite(name, processed)
                    return True

if __name__ == '__main__':
    image = '8.jpg'
    extract_retina(image)
