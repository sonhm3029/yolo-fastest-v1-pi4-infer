from darknet_images import *
from playsound import playsound
import time

def detect():

    config_file = "/home/hoang/darknet-master/data/violence/yolo-fastest-1.1_v4.cfg"
    data_file = "/home/hoang/darknet-master/data/violence/violence.data"
    weights = "/home/hoang/darknet-master/yolofastestv1_modv4_94.66_final.weights"
    thresh = 0.8

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=1
    )

    img_path = "Violence_1_drone - 3of8.mp4"
    

    start = time.time()
    image, detections = image_detection(
        img_path, network, class_names, class_colors, thresh
    )
    end = time.time()
    
    print(f"Run time: {(end - start):.2f} s")
        
if __name__ == "__main__":
    detect()