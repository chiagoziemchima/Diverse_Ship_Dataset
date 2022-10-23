import cv2
import os

labels = ['OPTICAL_SHIP', 'SAR_SHIP', 'NO_SHIP']

def denoise_img(path):
    os.mkdir(os.path.join(path + '_denoised'))
    for cl in labels:
        cl_imgs = os.listdir(os.path.join(path,cl))
        dest = os.path.join(path+'_denoised',cl)
        os.mkdir(os.path.join(path + '_denoised',cl))
        for im in cl_imgs:
            try:
                img = cv2.imread(os.path.join(path,cl,im), cv2.IMREAD_COLOR)
                img = cv2.fastNlMeansDenoisingColored(img,None,30,10,7,41)
                cv2.imwrite(os.path.join(dest,im), img)
            except Exception as e:
                print(e)

data = denoise_img("D:/Remote_Sensing/Heterogeneous_dataset/Test")
# PATH = "D:\\Chairlady\\Data"
# PATH = "D:\\Remote_Sensing\\Heterogeneous_dataset"
# PATH = "D:\Remote_Sensing\WHU_RS19"