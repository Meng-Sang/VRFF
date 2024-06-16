import cv2
import tqdm

from model.align.align_image import Deformation
from model.fuse import LazyUFusion
from model.reg import LazyMatchFormer
from utils.read_video import read_video_with_image_pairs
from utils.util import letterbox_image, MSE, to_estimation_tensor, LNCC, NCC


def predict_video(root_dir, reg=LazyMatchFormer(), fuse=LazyUFusion(), size=None, estimation_stand=LNCC(), method="homo",
                  ipf_n=47, is_select_best=True,
                  fps=20.0, moment_alpha=0.99, is_show=True, save_path=None):
    image_pairs = read_video_with_image_pairs(root_dir, size)
    if size is None:
        image_shape = next(image_pairs)[1].shape
        size = (image_shape[1], image_shape[0])
    if save_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_path, fourcc, fps, size)
    deformation = Deformation(ipf_n, fps, moment_alpha, is_select_best)
    for image_pair in tqdm.tqdm(image_pairs):
        resize_ir_image = letterbox_image(image_pair[0], size)
        resize_vi_image = letterbox_image(image_pair[1], size)
        resize_ir_Y, resize_ir_Cr, resize_ir_Cb = cv2.split(cv2.cvtColor(resize_ir_image, cv2.COLOR_BGR2YCrCb))
        resize_vi_Y, resize_vi_Cr, resize_vi_Cb = cv2.split(cv2.cvtColor(resize_vi_image, cv2.COLOR_BGR2YCrCb))
        ir_mkp, vi_mkp = reg(resize_ir_Y, resize_vi_Y)
        metrix = deformation(ir_mkp, vi_mkp, method)
        if metrix is not None:
            if  method == "homo":
                d_ir_image = cv2.warpPerspective(resize_ir_Y, metrix, size)
            elif method == "tps":
                d_ir_image = metrix.warpImage(resize_ir_Y)
        else:
            d_ir_image = resize_ir_Y
        if moment_alpha is not None and is_select_best and deformation.step <= fps:
            deformation.select_best_estimation(
                estimation_stand(to_estimation_tensor(resize_vi_Y).cuda(), to_estimation_tensor(d_ir_image).cuda()))
        fuse_image = fuse(d_ir_image, resize_vi_Y)
        fuse_image = cv2.merge((fuse_image, resize_vi_Cr, resize_vi_Cb))
        bgr_fuse_image = cv2.cvtColor(fuse_image, cv2.COLOR_YCrCb2BGR)
        if out is not None and save_path is not None:
            out.write(bgr_fuse_image)
        if is_show:
            cv2.imshow(root_dir, bgr_fuse_image)
            cv2.waitKey(1)
    if save_path is not None:
        out.release()
        cv2.destroyAllWindows()
