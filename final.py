import cv2
from matplotlib import pyplot as plt
import numpy as np
import pytesseract
from PIL import Image
import gradio as gr

def redact(image):
    image_file = "doc3.png"
    img = cv2.imread(image_file)

    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = grayscale(img)
    cv2.imwrite("gray.jpg", gray_image)
    thresh, im_bw = cv2.threshold(gray_image, 254 ,255, cv2.THRESH_BINARY)
    cv2.imwrite("bw_image.jpg", im_bw)

    def noise_removal(image):
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return (image)
    no_noise = noise_removal(im_bw)
    cv2.imwrite("no_noise.jpg", no_noise)
    base_image = cv2.imread("no_noise.jpg").copy()

    def thick_font(image):
        import numpy as np
        image = cv2.bitwise_not(image)
        kernel = np.ones((2,2),np.uint8)
        image = cv2.dilate(image, kernel, iterations=10)
        image = cv2.bitwise_not(image)
        return (image)
    dilated_image = thick_font(no_noise)
    cv2.imwrite("dilated_image.jpg", dilated_image)

    image = cv2.imread("dilated_image.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("index_gray.png", gray)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    cv2.imwrite("index_blur.png", blur)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imwrite("index_thresh.png", thresh)
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    cv2.imwrite("index_kernal.png", kernal)
    dilate = cv2.dilate(thresh, kernal, iterations=0)
    cv2.imwrite("index_dilate.png", dilate)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
    x1,y1,x2,y2=[0,0,0,0]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h<80 and h>50 and w>250 and w<500:
            cv2.rectangle(base_image, (x, y), (x+w, y+h), (36, 255, 12), 2)
            cropped = base_image[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped)
            if 'Aadhar' in text:
                x1=x+w+75
                y1=y
                x2=x1+(w*2)+50
                y2=y1+h
    cv2.imwrite("index_bbox_new.png", base_image)
    image = cv2.imread("doc3.png")
    cv2.rectangle(image, (x1, y1), (x2,y2), (0,0,0), -1)
    cv2.imwrite("final.png", image)
    image = cv2.imread("final.png")
    return(image)
gr.Interface(fn=redact, inputs="image", outputs="image",title="Aadhar Redaction").launch(share=False)














