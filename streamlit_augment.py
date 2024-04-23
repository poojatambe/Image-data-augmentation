from torchvision.transforms import v2
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np


st.set_page_config("Data Augmentation", layout='wide')
st.title("Data Augmentation")
st.header("Explore different Image Augmentation techniques.")
# upload image
upload = st.file_uploader("Upload Image:", type=['jpg', 'png'])
if upload:
    img = Image.open(upload)
    st.subheader("Data Augmentation libraries:")
    options = st.selectbox("Select library:", 
                           options=['None', 'Tensorflow', 'Pytorch'])
    # list tensorflow augmentations
    tf_aug = ['None', 'Flip', 'Grayscale', 'Saturation', 'Brightness', 'Contrast', 'Hue',
              'Gamma', 'Center crop', 'Rotate', 'Random brightness',
              'Random contrast', 'Random crop', 'Random hue', 'Random saturation',
              'Random Flip']
    # list pytroch augmentations
    torch_aug = ['None', 'Resize', 'Scale jitter', 'Random resize1', 
                 'Random resize2', 'Center crop', 'Five Crop', 
                 'Random crop', 'Random resized crop', 'Grayscale', 
                 'Color jitter', 'Guassian Blur', 'Random Invert', 
                 'Random Posterize', 'Random Solarize', 
                 'Random Adjust Sharpness', 'Random Autocontrast',
                 'Random Equilize', 'Random Flip', 'Pad', 'Random Perspective',
                 'Random Affine', 'Elastic transform', 'Random Zoomout',
                 'Random rotation', 'Random Channel Permutation', 
                 'Random Photometric Distortion']
    
    # for pytorch augmentations
    if options == "Pytorch":
        aug = st.sidebar.selectbox("Augmentations:",
                            options=torch_aug)
        # resize
        if aug == 'Resize':
            dsize = int(st.sidebar.number_input('Update size: '))
            if dsize > 0:
                resize_out = v2.Resize(size=(dsize, dsize))(img)
                c1, c2 = st.columns(2)
                c1.subheader("Input image:")
                c1.image(img, caption='original image')
                c1.write(img.size)
                c2.subheader('Augmentation Output: ')
                c2.image(resize_out, 'Resized image')
                c2.write(resize_out.size)
        # center crop
        elif aug == 'Center crop':
            dsize = int(st.sidebar.number_input('Update size: '))
            if dsize > 0:
                center_crop = v2.CenterCrop(dsize)(img)
                c1, c2 = st.columns(2)
                c1.subheader("Input image:")
                c1.image(img, caption='original image')
                c1.write(img.size)
                c2.subheader('Augmentation Output: ')
                c2.image(center_crop, 'Center cropped image')
                c2.write(center_crop.size)
        # five crop
        elif aug == 'Five Crop':
            dsize = int(st.sidebar.number_input('Update size: '))
            if dsize > 0 and dsize < min(img.size):
                top_left, top_right, bottom_left, bottom_right, center = v2.FiveCrop(dsize)(img)
                st.subheader("Input image:")
                st.image(img, caption='original image')
                st.write(img.size)
                st.subheader('Augmentation Output: ')
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.image(top_left)
                c1.write(top_left.size)
                c2.image(top_right)
                c2.write(top_right.size)
                c3.image(bottom_left)
                c3.write(bottom_left.size)
                c4.image(bottom_right)
                c4.write(bottom_right.size)
                c5.image(center)
                c5.write(center.size)
            else: 
                st.write("Invalid value.")
        # scale jitter
        elif aug == 'Scale jitter':
            w = int(st.sidebar.number_input('Update width: '))
            h = int(st.sidebar.number_input('Update height: '))
            w_scale = float(st.sidebar.number_input('Width scale: '))
            h_scale = float(st.sidebar.number_input('height scale: '))
            if w > 0 and h > 0 and w_scale > 0 and h_scale > 0:
                scale_jitter = v2.ScaleJitter((w, h), (w_scale, h_scale))(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(scale_jitter)
                c2.write(scale_jitter.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # random resize
        elif aug == 'Random resize1':
            min_s = int(st.sidebar.number_input('Update minimum value: '))
            max_s = int(st.sidebar.number_input('Update maximum value: '))
            if min_s > 0 and max_s > 0:
                random_resize = v2.RandomShortestSize(min_s, max_s)(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_resize)
                c2.write(random_resize.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # random resize
        elif aug == 'Random resize2':
            min_s = int(st.sidebar.number_input('Update minimum value: '))
            max_s = int(st.sidebar.number_input('Update maximum value: '))
            if max_s > 0 and max_s > min_s > 0:
                random_resize = v2.RandomResize(min_s, max_s)(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_resize)
                c2.write(random_resize.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # random crop
        elif aug == 'Random crop':
            dsize = int(st.sidebar.number_input('Update crop size: '))
            padding = int(st.sidebar.number_input('Padding by: '))
            pad = st.sidebar.radio("Padding if needed: ",
                             options=[True, False])
            pad_mode = st.sidebar.radio("Padding mode",
                                        options=['constant', 'edge', 'reflect', 'symmetric'])
            if dsize > 0:
                random_crop = v2.RandomCrop(size=dsize, padding=padding, 
                                            pad_if_needed=pad, padding_mode=pad_mode)(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_crop)
                c2.write(random_crop.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # random resized crop
        elif aug == "Random resized crop":
            dsize = int(st.sidebar.number_input('Update crop size: '))
            if dsize > 0:
                random_resized_crop = v2.RandomResizedCrop(dsize)(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_resized_crop)
                c2.write(random_resized_crop.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # grayscale
        elif aug == 'Grayscale':
            grayscale = v2.Grayscale()(img)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(grayscale)
            c2.write(grayscale.size)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # color jitter
        elif aug == 'Color jitter':
            brightness = float(st.sidebar.number_input('Update brightness: '))
            contrast = float(st.sidebar.number_input('Update contrast: '))
            saturation = float(st.sidebar.number_input('Update saturation: '))
            hue = float(st.sidebar.number_input('Update hue (<= 0.5): '))
            if hue <= 0.5:
                color_jitter = v2.ColorJitter(brightness, contrast, saturation, hue)(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(color_jitter)
                c2.write(color_jitter.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # guassian blur
        elif aug == 'Guassian Blur':
            kernel = int(st.sidebar.number_input("Kernel size: "))
            sigma = float(st.sidebar.number_input(" Standard deviation: "))
            if kernel > 0 and kernel % 2 != 0 and sigma > 0:
                guassian_blur = v2.GaussianBlur(kernel, sigma)(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(guassian_blur)
                c2.write(guassian_blur.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # random invert
        elif aug == "Random Invert":
            random_invert = v2.RandomInvert(p=1)(img)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(random_invert)
            c2.write(random_invert.size)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # random posterize
        elif aug == "Random Posterize":
            bits = int(st.sidebar.number_input("Bits (0, 8): "))
            if 8 >= bits >= 0:
                random_posterize = v2.RandomPosterize(bits, p=1)(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_posterize)
                c2.write(random_posterize.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # random solarize
        elif aug == "Random Solarize":
            threshold = int(st.sidebar.number_input("Threshold (0, 255): "))
            if 255 >= threshold >= 0:
                random_solarize = v2.RandomSolarize(threshold, p=1)(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_solarize)
                c2.write(random_solarize.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # random adjust sharpness
        elif aug == "Random Adjust Sharpness":
            factor = float(st.sidebar.number_input("Factor (0, 2): "))
            if 2 >= factor >= 0:
                random_adjust_sharpness = v2.RandomAdjustSharpness(factor, p=1)(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_adjust_sharpness)
                c2.write(random_adjust_sharpness.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # random adjust contrast
        elif aug == "Random Autocontrast":        
            random_adjust_contrast = v2.RandomAutocontrast(p=1)(img)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(random_adjust_contrast)
            c2.write(random_adjust_contrast.size)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # random equilize
        elif aug == 'Random Equilize':
            random_equilize = v2.RandomEqualize(p=1)(img)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(random_equilize)
            c2.write(random_equilize.size)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # random flip
        elif aug == 'Random Flip':
            opt = st.sidebar.radio("Select option:", 
                             options=['Horizontal', 'Vertical', 'Both'])
            if opt == 'Horizontal':
                random_flip = v2.RandomHorizontalFlip(p=1)(img)
            elif opt == 'Vertical':
                random_flip = v2.RandomVerticalFlip(p=1)(img)
            else:
                h_flip = v2.RandomHorizontalFlip(p=1)(img)
                v_flip = v2.RandomVerticalFlip(p=1)(img)
                random_flip = np.hstack([h_flip, v_flip])
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(random_flip)
            c2.write(random_flip.size)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # Pad
        elif aug == 'Pad':
            padding = int(st.sidebar.number_input('Padding: '))
            pad_mode = st.sidebar.radio("Padding mode",
                                        options=['constant', 'edge', 'reflect', 'symmetric'])
            pad = v2.Pad(padding=padding, padding_mode=pad_mode)(img)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(pad)
            c2.write(pad.size)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # random perspective
        elif aug == 'Random Perspective':
            distortion_scale = float(st.sidebar.number_input('Distortion scale: '))
            if 1 >= distortion_scale >= 0:
                random_perspective = v2.RandomPerspective(distortion_scale, p=1)(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_perspective)
                c2.write(random_perspective.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # random affine
        elif aug == 'Random Affine':
            t1 = float(st.sidebar.slider('Translate: ', 0.0, 1.0, 0.1, 0.1))
            t2 = float(st.sidebar.slider('Translate: ', 0.0, 1.0, 0.3, 0.1))
            degrees = float(st.sidebar.number_input('Rotate: '))
            s1 = float(st.sidebar.slider('Scale: ', 0.0, 1.0, 0.1, 0.1))
            s2 = float(st.sidebar.slider('Scale: ', 0.0, 1.0, 0.3, 0.1))
            shear = float(st.sidebar.number_input('Shear: '))
            random_affine = v2.RandomAffine(degrees, (t1, t2), (s1, s2), shear)(img)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(random_affine)
            c2.write(random_affine.size)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # Elastic transform
        elif aug == 'Elastic transform':
            alpha = float(st.sidebar.number_input('Alpha: '))
            sigma = float(st.sidebar.number_input('Sigma: '))
            random_elastic_transform = v2.ElasticTransform(alpha, sigma)(img)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(random_elastic_transform)
            c2.write(random_elastic_transform.size)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # Random Zoomout
        elif aug == 'Random Zoomout':
            r1 = float(st.sidebar.slider('Side range: ', 1.0, 10.0, 1.0, 0.1))
            r2 = float(st.sidebar.slider('Side range: ', 1.0, 10.0, 4.0, 0.1))
            if r1 < r2 :
                random_zoomout = v2.RandomZoomOut(side_range=(r1, r2), p=1)(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_zoomout)
                c2.write(random_zoomout.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # Random rotation:
        elif aug == 'Random rotation':
            degrees = float(st.sidebar.number_input('Degrees: '))
            random_rotation = v2.RandomRotation(degrees)(img)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(random_rotation)
            c2.write(random_rotation.size)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # Random channel permutations
        elif aug == 'Random Channel Permutation':
            random_ch_p = v2.RandomChannelPermutation()(img)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(random_ch_p)
            c2.write(random_ch_p.size)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # Random Photometric Distortion
        elif aug == 'Random Photometric Distortion':
            b1 = float(st.sidebar.number_input('Brightness min: '))
            b2 = float(st.sidebar.number_input('Brightness max: ', key='1'))
            c1 = float(st.sidebar.number_input('contrast min: '))
            c2 = float(st.sidebar.number_input('Contrast max: ', key='2'))
            s1 = float(st.sidebar.number_input('Saturation min: '))
            s2 = float(st.sidebar.number_input('Saturation max:', key='3'))
            h1 = float(st.sidebar.number_input('Hue min: '))
            h2 = float(st.sidebar.number_input('Hue max: ', key='4'))
            if b1 < b2 and c1 < c2 and s1 < s2 and h1 < h2 and h1 <= 0.5 and h2 <= 0.5:
                random_ph_d = v2.RandomPhotometricDistort(brightness=(b1, b2),
                                                        contrast=(c1, c2),
                                                        saturation=(s1, s2),
                                                        hue=(h1, h2), p=1)(img)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_ph_d)
                c2.write(random_ph_d.size)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
    elif options == 'Tensorflow':
        # img = np.array(img)
        aug = st.sidebar.selectbox("Augmentations:",
                            options=tf_aug)
        # Flip
        if aug == 'Flip':
            orint = st.sidebar.radio('Select flip mode: ', 
                                     options=['Horizontal', 'Vertical'])
            if orint == 'Horizontal':
                out_flip = tf.image.flip_left_right(img)
            elif orint == 'Vertical':
                out_flip = tf.image.flip_up_down(img)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(out_flip.numpy())
            c2.write(out_flip.shape)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # Grayscale 
        elif aug == 'Grayscale':
            gray_out = tf.image.rgb_to_grayscale(img)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(gray_out.numpy())
            c2.write(gray_out.shape)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # Saturation
        elif aug == 'Saturation':
            val = float(st.sidebar.number_input('Saturation Value (0, inf): '))
            if val > 0.0:
                sat_out = tf.image.adjust_saturation(img, val)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(sat_out.numpy())
                c2.write(sat_out.shape)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # brightness
        elif aug == 'Brightness':
            delta = float(st.sidebar.number_input("Select delta (-1, 1): "))
            if -1 <= delta <= 1:
                bright_out = tf.image.adjust_brightness(img, delta)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(bright_out.numpy())
                c2.write(bright_out.shape)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # Contrast
        elif aug == 'Contrast':
            contrast_factor = float(st.sidebar.number_input("Select contrast (-inf, inf):"))
            contrast_out = tf.image.adjust_contrast(img, contrast_factor)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(contrast_out.numpy())
            c2.write(contrast_out.shape)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # hue
        elif aug == "Hue":
            delta = float(st.sidebar.number_input("Select delta (-1, 1): "))
            if -1 <= delta <= 1:
                hue_out = tf.image.adjust_hue(img, delta)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(hue_out.numpy())
                c2.write(hue_out.shape)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # Gamma
        elif aug == "Gamma":
            gamma = float(st.sidebar.number_input("Select gamma (0, inf): "))
            if gamma > 0.0:
                gamma_out = tf.image.adjust_gamma(img, gamma)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(gamma_out.numpy())
                c2.write(gamma_out.shape)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
            st.write("For gamma greater than 1, the histogram will shift towards \
                     left and the output image will be darker than the input image.\
                     For gamma less than 1, the histogram will shift towards right \
                     and the output image will be brighter than the input image.")
        # center crop
        elif aug == "Center crop":
            centeral_fraction = float(st.sidebar.slider("Select crop fraction (0, 1): ",
                                                        0.0, 1.0, 0.5, 0.01))
            if 1.0 > centeral_fraction >= 0.0:
                center_crop = tf.image.central_crop(img, centeral_fraction)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(center_crop.numpy())
                c2.write(center_crop.shape)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # Rotate
        elif aug == "Rotate":
            rotate_opt = st.sidebar.radio("Select rotation:", 
                                          options=['Clockwise', 'Counter-clockwise', 'Upside-down'])
            if rotate_opt == 'Clockwise':
                rotate_out = tf.image.rot90(img, k=1)
            elif rotate_opt == 'Counter-clockwise':
                rotate_out = tf.image.rot90(img, k=3)
            elif rotate_opt == "Upside-down":
                rotate_out = tf.image.rot90(img, k=2)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(rotate_out.numpy())
            c2.write(rotate_out.shape)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size) 
        # random brightness
        elif aug == 'Random brightness':
            max_delta = float(st.sidebar.number_input("Select max delta (0, 1): "))
            if 1.0 >= max_delta > 0.0:
                random_bright = tf.image.random_brightness(img, max_delta)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_bright.numpy())
                c2.write(random_bright.shape)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # random contrast
        elif aug == "Random contrast":
            max_val = float(st.sidebar.number_input('Select max value:'))
            min_val = float(st.sidebar.number_input('Select min value:'))
            if max_val > min_val and max_val >= 0 and min_val >= 0:
                random_contrast_out = tf.image.random_contrast(img, min_val, max_val)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_contrast_out.numpy())
                c2.write(random_contrast_out.shape)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # random crop 
        elif aug == "Random crop":
            h_c = int(st.sidebar.slider("Select height: ", 1, 
                                        int(img.size[1]-1)))
            w_c = int(st.sidebar.slider("Select width: ", 1, 
                                        int(img.size[0]-1)))
            random_crop_out = tf.image.random_crop(img, (h_c, w_c, 3))
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image( random_crop_out.numpy())
            c2.write( random_crop_out.shape)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)    
        # random flip
        elif aug == "Random Flip":
            flip_opt = st.sidebar.radio("Select flip: ",
                                        options=['Horizontal', 'Vertical'])
            if flip_opt == 'Horizontal':
                random_flip_out = tf.image.random_flip_left_right(img)
            else: 
                random_flip_out = tf.image.random_flip_up_down(img)
            c1, c2 = st.columns(2)
            c2.subheader('Augmentation Output: ')
            c2.image(random_flip_out.numpy())
            c2.write(random_flip_out.shape)
            c1.subheader("Input image:")
            c1.image(img)
            c1.write(img.size)
        # random hue
        elif aug == "Random hue":
            max_delta = float(st.sidebar.number_input("Select max delta (0, 0.5): "))
            if 0.5 >= max_delta > 0.0:
                random_hue_out = tf.image.random_hue(img, max_delta)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_hue_out.numpy())
                c2.write(random_hue_out.shape)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)
        # random saturation
        elif aug == "Random saturation":
            max_val = float(st.sidebar.number_input('Select max value:'))
            min_val = float(st.sidebar.number_input('Select min value:'))
            if max_val > min_val and max_val >= 0 and min_val >= 0:
                random_sat_out = tf.image.random_saturation(img, min_val, max_val)
                c1, c2 = st.columns(2)
                c2.subheader('Augmentation Output: ')
                c2.image(random_sat_out.numpy())
                c2.write(random_sat_out.shape)
                c1.subheader("Input image:")
                c1.image(img)
                c1.write(img.size)            