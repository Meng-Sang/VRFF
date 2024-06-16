from utils.predict_video import predict_video

if __name__ == "__main__":
    predict_video(r"assets/video/hdo", size=(320, 240), ipf_n=47, moment_alpha=0.99, save_path="assets/results/ipf_mom_fuse.avi",is_show=True)
    predict_video(r"assets/video/hdo", size=(320, 240), ipf_n=1, moment_alpha=None, save_path="assets/results/fuse.avi", is_show=True)

