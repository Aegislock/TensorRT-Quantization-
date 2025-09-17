import numpy as np
import torch
import torch.nn as nn
import util_trt
import glob,os,cv2

BATCH_SIZE = 16
BATCH = 100
height = 416
width = 416
CALIB_IMG_DIR = './calibration/'
onnx_model_path = './armor_tiny.onnx'

def preprocess(img, width=416, height=416):
    img_h, img_w = img.shape[:2]

    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    ratio = min(width / img_w, height / img_h)

    new_w, new_h = int(img_w * ratio), int(img_h * ratio)
    new_img = cv2.resize(img, (new_w, new_h))
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

    canvas[0:new_h, 0:new_w, :] = new_img
    canvas = canvas.transpose((2, 0, 1)).astype(np.float32)
    # canvas /= 255.0  # Uncomment if model expects 0-1 input

    canvas = np.expand_dims(canvas, 0)

    return canvas

class DataLoader:
    def __init__(self):
        self.index = 0
        self.length = BATCH
        self.batch_size = BATCH_SIZE
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(CALIB_IMG_DIR, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(CALIB_IMG_DIR) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size,3,height,width), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess(img)
                self.calibration_data[i] = img

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length

def main():
    # onnx2trt
    fp16_mode = False
    int8_mode = True 
    print('*** onnx to tensorrt begin ***')
    # calibration
    calibration_stream = DataLoader()
    engine_model_path = "modelInt8.engine"
    calibration_table = 'best_calibration.cache'
    # fixed_engine,校准产生校准表
    engine_fixed = util_trt.get_engine(BATCH_SIZE, onnx_model_path, engine_model_path, fp16_mode=fp16_mode, 
        int8_mode=int8_mode, calibration_stream=calibration_stream, calibration_table_path=calibration_table, save_engine=True)
    print(engine_fixed)
    assert engine_fixed, 'Brokenls engine_fixed'
    print('*** onnx to tensorrt completed ***\n')
    
if __name__ == '__main__':
    main()
