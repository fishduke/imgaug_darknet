import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import os
import glob
import cv2
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/dataset/Dataset_backup/Completion_dataset/basics_detector/5class/2024_02_06/person',
                help="Where is images directory? ex)home/user/imagefolder")
    parser.add_argument('--count', type=int, default=2,
                help="how many will you augmentate? ex)2")
    parser.add_argument('--mode', type=int, default=0,
                help="test augmentation:0 / create augmented files:1")
    return parser.parse_args()

def make_aug(image, count, labeling):
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.0), "y": (0.8, 1.0)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -20 to +20 percent (per axis)
                rotate=(0, 5), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 0.1), n_segments=(20, 20))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 2)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 3)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.0), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.OneOf([
                       iaa.Clouds()
                    ]),
                    iaa.OneOf([
                       iaa.FastSnowyLandscape(
                            lightness_threshold=(100, 255),
                            lightness_multiplier=(1.0, 4.0)
                        )
                    ]),
                    # iaa.OneOf([
                    #    iaa.Fog()
                    # ]),
                    iaa.OneOf([
                       iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))
                    ]),
                    iaa.OneOf([
                       iaa.Rain(drop_size=(0.10, 0.20))
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )

    img = cv2.imread(image)

    img_path = image
    txt_path = img_path[:img_path.find('.')] + '.txt'
    txt = open(txt_path, "r")
    lines = txt.readlines()

    boxes = []
    class_name = []

    for i in range(len(lines)):
        label = lines[i].split(' ')
        x = float(label[1]) * img.shape[1]
        y = float(label[2]) * img.shape[0]
        w = int(float(label[3]) * img.shape[1])
        h = int(float(label[4][:len(label[4])-1]) * img.shape[0])

        if(labeling):            
            #for labeling
            x1 = x-w/2
            y1 = y-h/2
            x2 = x+w/2
            y2 = y+h/2
        else:
            #for rectangle
            x1 = int(x-w/2)
            y1 = int(y-h/2)
            x2 = int(x+w/2)
            y2 = int(y+h/2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 1)

        class_name.append(label[0])
        boxes.append(BoundingBox(x1,y1,x2,y2))
    txt.close()
    
    bounding_boxes = ia.BoundingBoxesOnImage(boxes, img.shape)
    seq = seq.to_deterministic()
    img_np = np.array(img)
    images_aug = seq.augment_image(img_np)
    bbs_aug = seq.augment_bounding_boxes([bounding_boxes])[0]
    aug = images_aug.astype(np.uint8)
    new_path = image[:image.rfind('.')]+"_"+str(count)
    if(labeling):    
        new_txt = open(new_path+".txt", "w")
        
    for i in range(len(bbs_aug.bounding_boxes)):
            bb_box = bbs_aug.bounding_boxes[i]

            if(labeling): 
                x1 = bb_box[0][0] /img.shape[1]
                y1 = bb_box[0][1] /img.shape[0]
                x2 = bb_box[1][0] /img.shape[1]
                y2 = bb_box[1][1] /img.shape[0]
                w = int(float(label[3]) * img.shape[1])
                h = int(float(label[4][:len(label[4])-1]) * img.shape[0])

                wid = x2-x1
                hei = y2-y1
                x = x1+wid/2
                y = y1+hei/2
                
                new_line = class_name[i] + " " + str(x) + " " + str(y) + " " + str(wid) + " " + str(hei) + "\n"
                new_txt.write(new_line)
                
            else:
                x1 = int(bb_box[0][0])
                y1 = int(bb_box[0][1])
                x2 = int(bb_box[1][0])
                y2 = int(bb_box[1][1])
                cv2.rectangle(aug, (x1, y1), (x2, y2), (255,0,255), 1)
           

    if(labeling==0):
        add = cv2.hconcat([img,aug])
        cv2.imshow("add",add)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        
    else:
        new_txt.close()
        cv2.imwrite(new_path+".jpg",aug)
    seq = seq.clear()
    

def main():
    args = parser()
    
    labeling = args.mode
    augmentation_count = args.count
    folder_path = args.path
    
    folders = os.listdir(folder_path)
    # for folder in folders:
    #     path = os.path.join(folder_path, folder) ## if images hold seperated, gethering all
    path = folder_path

    images = glob.glob(f'{path}/*.jpg') + glob.glob(f'{path}/*.png')
    print("original images size : ", len(images))

    num = 0
    for image in images:
        num += 1
        for count in range(augmentation_count):
            make_aug(image, count, labeling)
                                    
        msg = "\rprocessed : %.0f%%" % (num/len(images)*100.0)
        print(msg,end='')
        
    images = glob.glob(f'{path}/*.jpg') + glob.glob(f'{path}/*.png')
    print("original+aug images size : ", len(images))


if __name__=="__main__":
    main()