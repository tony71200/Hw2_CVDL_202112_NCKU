import sys
from PIL import Image
import cv2 as cv
import numpy as np
import os
from numpy.linalg.linalg import eigh
from skimage.util import dtype
from sklearn.decomposition import PCA
from glob import glob
import matplotlib.pyplot as plt

from numpy.core.numeric import ones_like, zeros_like

def concath(list_array:list):
    return cv.vconcat(list_array)

def Q1(path):
    """
    Inital video
    """
    cap = cv.VideoCapture(path)
    frames = []
    build_model = False
    # w=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # h=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT ))
    while cap.isOpened():
        "Read video frame"
        ret, frame = cap.read()
        if ret:
            "Convert BGR frame to GRAY frame"
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray)
            "Get first 25 frames for building Gaussian model"
            if len(frames) < 25:
                frames.append(gray)
            else:
                if not build_model:
                    frames = np.array(frames)
                    """For every pixels in video from 0~25 frames, 
                    build a gaussian model with mean and standard deviation 
                    (if standard deviation is less then 5, set to 5)
                    """
                    mean = np.mean(frames, axis= 0)
                    standard = np.std(frames, axis=0)
                    standard[standard < 5] = 5
                    build_model = True
                else:
                    """
                    For frame > 25, test every frame pixels with respective gaussian model. 
                    If gray value difference between testing pixel and gaussian mean 
                    is larger than 5 times standard deviation, 
                    set testing pixel to 255 (foreground, white), 0 (background, black) otherwise.
                    """
                    mask[np.abs(gray - mean) > standard*5] = 255
            """
            Show the result
            """
            foreground = cv.bitwise_and(frame, frame, mask= mask)
            mask_out = np.dstack((mask, mask, mask))

            out = concath([frame, mask_out, foreground])
            cv.imshow("Video", out)
            key = cv.waitKey(50)
            if key == ord("q"):
                break
        else:
            break
    cap.release()
    cv.destroyAllWindows()

class Q2():
    def __init__(self, path:str):
        if not os.path.exists(path):
            return
        self.cap = cv.VideoCapture(path)
        self.keypoints = []
        # Initialize parameter settiing using cv2.SimpleBlobDetector
        self.param = cv.SimpleBlobDetector_Params()
        self.param.minThreshold = 80
        self.param.maxThreshold = 150
        self.param.filterByArea = True
        self.param.minArea = 24
        self.param.maxArea = 90
        self.param.filterByCircularity = True
        self.param.minCircularity = 0.85
        self.param.filterByConvexity = True
        self.param.minConvexity = 0.6
        self.param.filterByInertia = True
        self.param.minInertiaRatio = 0.52

        try:
            ver = (cv.__version__).split(".")
            if int(ver[0]) < 3:
                self.detector = cv.SimpleBlobDetector(self.param)
            else:
                self.detector = cv.SimpleBlobDetector_create(self.param)
        except (ValueError, ZeroDivisionError):
            self.detector = cv.SimpleBlobDetector_create(self.param)
        
    @staticmethod
    def draw_boundingbox(frame, x_center, y_center):
        x_lelf = x_center - 6
        y_top = y_center -6
        x_right = x_center + 6
        y_bottom = y_center + 6
        frame = cv.rectangle(frame, (x_lelf, y_top), (x_right, y_bottom), 
                            (0, 0, 255), 1, cv.LINE_AA)
        frame = cv.line(frame, (x_center, y_top), (x_center, y_bottom),
                        (0, 0, 255), 1, cv.LINE_AA)
        frame = cv.line(frame, (x_lelf, y_center), (x_right, y_center),
                        (0, 0, 255), 1, cv.LINE_AA)
        return frame
    
    @staticmethod
    def show_video(frame, window_name:str):
        cv.namedWindow(window_name, cv.WINDOW_GUI_EXPANDED)
        cv.imshow(window_name, frame)
        
    def preprocessing(self, frame):
        frame_copy = frame.copy()
        if frame.shape[2:] == 3:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        keypoints = self.detector.detect(frame)
        for kp in keypoints:
            x, y = kp.pt
            frame_copy = self.draw_boundingbox(frame_copy, int(x), int(y))
            self.keypoints.append((int(x), int(y)))
        return frame_copy

    def tracking(self):
        pass


    def processing(self, prepro = True):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                preprocessing_frame = self.preprocessing(frame)
                if prepro:
                    self.show_video(preprocessing_frame, "2.1 Processing")
                    if (cv.waitKey(5) & 0xFF) == ord("q"):
                        break
                else:
                    pass
            else:
                break
        self.cap.release()
        cv.destroyAllWindows()


def Q3(path_video, path_image):
    # if not(os.path.exists(path_video)) and not(os.path.exists(path_image)):
    #     continue
    cap = cv.VideoCapture(path_video)
    image = cv.imread(path_image)
    h, w = image.shape[:2]

    # w_v =int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    # h_v =int(cap.get(cv.CAP_PROP_FRAME_HEIGHT ))
    # fps = cap.get(cv.CAP_PROP_FPS)

    # # video recorder
    # fourcc = cv.VideoWriter_fourcc(*'MP4V')
    # video_writer = cv.VideoWriter("output2.mp4",fourcc, fps, (w_v, h_v*2))
    """
    Loading one of the predefined distionaries in aruco module
    This DICT_4X4_250 dictionary is composed of 250 markers and marker size of 4X4 bits
    """
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_250)
    # Initial parameters for the detectmaker process
    param = cv.aruco.DetectorParameters_create()
    

    cv.namedWindow("3. Perspective Transform", cv.WINDOW_GUI_EXPANDED)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Detect Aruco makers in image and get the content of each marker
            markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(
                frame, 
                dictionary,
                parameters = param
            )

            #Find id for each markers
            id1 = np.squeeze(np.where(markerIds == 1))
            id2 = np.squeeze(np.where(markerIds == 2))
            id3 = np.squeeze(np.where(markerIds == 3))
            id4 = np.squeeze(np.where(markerIds == 4))
            final = zeros_like(frame)
            # Process of perspective transform
            if id1 != [] and id2 != [] and id3 != [] and id4 != []:
                # Check if all markers can be detect or not
                # Get the top-left corner of marker1 
                pt1 = np.squeeze(markerCorners[id1[0]])[0]
                # Get the top-right corner of marker2
                pt2 = np.squeeze(markerCorners[id2[0]])[1]
                # Get the bottom-right corner of marker3
                pt3 = np.squeeze(markerCorners[id3[0]])[2]
                # Get the bottom-left corner of marker4
                pt4 = np.squeeze(markerCorners[id4[0]])[3]

                # Get coordinates of the corresponding quadrangle vertices in the destination image
                pts_dst = [[pt1[0], pt1[1]]]
                pts_dst += [[pt2[0], pt2[1]]]
                pts_dst += [[pt3[0], pt3[1]]]
                pts_dst += [[pt4[0], pt4[1]]]

                #Get coordinates of quadrangle  vertices in the source image
                pts_src = [[0, 0], [w, 0], [w, h], [0, h]]

                retval, mask = cv.findHomography(np.asfarray(pts_src), np.asfarray(pts_dst))
                out = cv.warpPerspective(image, retval, (frame.shape[1], frame.shape[0]))
                mask_image = cv.warpPerspective(np.ones_like(image)*255, retval, (frame.shape[1], frame.shape[0]))
                # mask_image = np.zeros_like(out)
                # mask_image[out[:,:,:] > 0] = 255
                # out = cv.add(frame, out)
                mask_image = cv.bitwise_not(mask_image)
                frame_mask = cv.bitwise_and(frame, mask_image)
                final = cv.bitwise_or(frame_mask, out)
                output = concath([frame, final])
                # video_writer.write(output)
                cv.imshow("3. Perspective Transform", output)
            key = cv.waitKey(25) & 0xFF
            if key == ord("q"):
                break
        else:
            break
    cap.release()
    # video_writer.release()
    cv.destroyAllWindows()

class Q4():
    def __init__(self, path:str, n_components = 15):
        self.path_images = glob(os.path.join(path, "*.jpg"))
        len_image = len(self.path_images)
        if n_components > 0 and n_components < len_image:
            n_components = n_components
        else:
            n_components = len_image
        self.pca = PCA(n_components=n_components)
        self.images = []
        self.pca_images = []
        self.load()
        self.reconstruction = None

    def load(self):
        for path_image in self.path_images:
            img = Image.open(path_image)
            img = img.convert("RGB")
            img = np.asarray(img)
            self.images.append(img)
            self.pca_images.append(img.flatten())
        self.pca_images = np.array(self.pca_images, dtype=int)

    def imageReconstruction(self, show=True):
        components = self.pca.fit_transform(self.pca_images)
        self.reconstruction = self.pca.inverse_transform(components)
        if show:
            fig, ax = plt.subplots(4, 15, 
                                    figsize= (9, 5),
                                    subplot_kw={'xticks': [], 'yticks'    : []},
                                    gridspec_kw = dict(hspace=0.1, wspace=0.1))
            for i in range(0, 15):
                ax[0, i].imshow(self.images[i].reshape(400,400,3))
                ax[1, i].imshow(np.reshape(self.reconstruction[i, :].astype(np.uint8), (400, 400, 3)))
                #ax[1, i].imshow(np.reshape(badges_pca.components_[i, :] ,(100, 100, 3)))
                ax[2, i].imshow(self.images[i + 15].reshape(400, 400, 3))
                ax[3, i].imshow(np.reshape(self.reconstruction[i + 15, :].astype(np.uint8), (400, 400, 3)))
            ax[0, 0].set_ylabel('Original')
            ax[1, 0].set_ylabel('Reconstruction')
            ax[2, 0].set_ylabel('Original')
            ax[3, 0].set_ylabel('Reconstruction')
            plt.show()
        
    def reconstructionErrorComputing(self):
        """
        Using this function to compute the reconstruction errors.
        :return:
        """
        if len(self.reconstruction) <=0:
            self.imageReconstruction(False) 
        computing = []
        for origin_image, reconstruction in zip(self.images, self.reconstruction):
            orig_img = origin_image.reshape(400,400,3)
            orig_gray = cv.cvtColor(orig_img, cv.COLOR_RGB2GRAY)

            recons_img = np.reshape(reconstruction.astype(np.uint8), (400,400,3))
            recons_gray = cv.cvtColor(recons_img, cv.COLOR_RGB2GRAY)
            
            arr_sub = np.subtract(orig_gray, recons_gray)
            arr_sub = np.absolute(arr_sub)

            sum_error = np.sum(arr_sub)
            computing.append(sum_error)
        print("\nThe reconstruction error is:")
        print(computing)



if __name__ == "__main__":
    # trans = Q2(r"./Q2_Image/optical_flow.mp4")
    # trans.processing()

    process_pca = Q4(r"./Q4_Image", 29)
    process_pca.imageReconstruction()
    process_pca.reconstructionErrorComputing()
