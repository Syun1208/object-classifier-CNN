
# import torch.nn as nn
import cv2
from PIL import Image
import numpy as np
from model import Network
import torch
import matplotlib.pyplot as plt
import os 
import time

path = './test_data/right' #stop -> right, ahead-> stop, right -> stop
test_img = os.listdir(path)
# print(test_img)




def Predict(img_raw,numb):

        classes = ['ahead', 'right', 'stop', 'none']

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        #print('device:', device)

        # Define hyper-parameter
        img_size = (64, 64)

        # define model
        model = Network()
        pre_trained = torch.load('cnn.pt')
        model.load_state_dict(pre_trained)

        #port to model to gpu if you have gpu
        model = model.to(device)
        model.eval()

        # resize img to 48x48
        img_rgb = cv2.resize(img_raw, img_size)

        # convert from RGB img to gray img
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # normalize img from [0, 255] to [0, 1]
        img_rgb = img_rgb/255
        img_rgb = img_rgb.astype('float32')
        img_rgb = img_rgb.transpose(2,0,1)


        # convert image to torch with size (1, 1, 48, 48)
        img_rgb = torch.from_numpy(img_rgb).unsqueeze(0)

        with torch.no_grad():
            img_rgb = img_rgb.to(device)
            # print("type: " + str(numb), type(img_rgb))
            y_pred = model(img_rgb)
            print("y_pred", y_pred)           
            _, pred = torch.max(y_pred, 1)
            
            pred = pred.data.cpu().numpy()
            # print("2nd", second_time - fist_time)
            # print("predict: " +str(numb), pred)
            class_pred = classes[pred[0]]
            print("class_pred", class_pred)
            
            
        return class_pred
            #emotion_prediction = classes[pred[0]]

            # cv2.putText(img_raw, 'Predict: '+emotion_prediction, (20, 20),
            #                         cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
        #     cv2.imshow(emotion_prediction, np.array(img_raw))
        # cv2.waitKey(2)
        # cv2.destroyAllWindows()


def Run_Predict():
    right =0
    stop = 0
    ahead = 0
    avg_time = 0
    print(f'tong left: {len(test_img)}')
    for numb in range(len(test_img)):
        t1 = time.time()
        print("path: ", f'{path}/{test_img[numb]}')
        img_raw = cv2.imread(f'{path}/{test_img[numb]}')
        rs=  Predict(img_raw,numb)
        t2 = time.time()
        avg_time +=  (t2-t1)
        if rs == 'stop':
            stop +=1 
        if rs == 'ahead':
            ahead +=1 
        if rs == 'right':
            right +=1 
    # print("avg_fps: ", 1/(avg_time/len(test_img)))
    print('stop: ',stop)
    print('ahead: ',ahead)
    print('right: ',right)
    # img_raw = cv2.imread('right.jpg')#f'{path}/{test_img[numb]}')
    # rs=  Predict(img_raw)

if __name__ == '__main__':
    Run_Predict()

