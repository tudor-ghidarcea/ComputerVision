import cv2 as cv
import numpy as np
import os
def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.3,fy=0.3)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def extrage_careu(image):
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    l = np.array([0, 130, 23])
    u = np.array([255, 239, 255])
    mask_table_hsv = cv.inRange(image_hsv, l, u)
    image_m_blur = cv.medianBlur(mask_table_hsv, 3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5)
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
    #show_image('image_sharpened',image_sharpened)
    _, thresh = cv.threshold(image_sharpened, 30, 255, cv.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.erode(thresh, kernel)
    #show_image('image_thresholded',thresh)

    edges = cv.Canny(thresh, 200, 400)
    #show_image('edges',edges)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    for i in range(len(contours)):
        if (len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 810
    height = 810

    image_copy = cv.cvtColor(image.copy(), cv.COLOR_HSV2BGR)
    cv.circle(image_copy, tuple(top_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(top_right), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_right), 20, (0, 0, 255), -1)
    #show_image("detected corners",image_copy)

    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)

    result = cv.warpPerspective(image, M, (width, height))
    result = cv.cvtColor(result, cv.COLOR_HSV2BGR)

    return result

lines_horizontal=[]
for i in range(0,811,54):
    l=[]
    l.append((0,i))
    l.append((809,i))
    lines_horizontal.append(l)

lines_vertical=[]
for i in range(0,811,54):
    l=[]
    l.append((i,0))
    l.append((i,809))
    lines_vertical.append(l)


def determina_configuratie_careu_ox(thresh,lines_horizontal,lines_vertical):
    matrix = np.empty((15,15), dtype='str')
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0] + 20
            y_max = lines_vertical[j + 1][1][0] - 20
            x_min = lines_horizontal[i][0][1] + 20
            x_max = lines_horizontal[i + 1][1][1] - 20
            patch = thresh[x_min:x_max, y_min:y_max].copy()
            Medie_patch=np.mean(patch)
            if Medie_patch>0:
                matrix[i][j]='x'
            else:
                matrix[i][j]='o'
    return matrix

def vizualizare_configuratie(result,matrix,lines_horizontal,lines_vertical):
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            if matrix[i][j] == 'x':
                cv.rectangle(result, (y_min, x_min), (y_max, x_max), color=(255, 0, 0), thickness=5)


def clasifica_cifra(patch):
    maxi = -np.inf
    poz = -1

    for j in range(0, 24):
        img_template = cv.imread('litere/' + str(j) + '.jpg')
        image_hsv = cv.cvtColor(img_template, cv.COLOR_BGR2HSV)  #am umblat
        l = np.array([43, 71, 3])  #
        u = np.array([125, 255, 83])  #
        mask_img_hsv = cv.inRange(image_hsv, l, u) #
        corr = cv.matchTemplate(patch, mask_img_hsv, cv.TM_CCOEFF_NORMED)# pana aici am umblat
        corr = np.max(corr)
    if corr > maxi:
        maxi = corr
        poz =0
    if cv.countNonZero(patch) < 576:
                poz = j
    return dictionar_litere[poz]



def determina_configuratie_careu_ocifre(img,thresh,lines_horizontal,lines_vertical):
    matrix = np.empty((15,15), dtype='str')
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0] + 15
            y_max = lines_vertical[j + 1][1][0] - 15
            x_min = lines_horizontal[i][0][1] + 15
            x_max = lines_horizontal[i + 1][1][1] - 15
            patch = thresh[x_min:x_max, y_min:y_max].copy()
            patch_orig=img[x_min:x_max, y_min:y_max].copy()
            patch_orig= cv.cvtColor(patch_orig,cv.COLOR_BGR2GRAY)
            image_hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV)  # am umblat
            l = np.array([43, 71, 3])  #
            u = np.array([125, 255, 83])  #
            mask_img_hsv = cv.inRange(image_hsv, l, u)  #
            Medie_patch=np.mean(mask_img_hsv)
            matrix[i][j]=clasifica_cifra(patch_orig)

    return matrix

dictionar_litere={0:' ',1:'A',2:'B',3:'C',4:'D',5:'E',6:'F',7:'G',8:'H',9:'I',10:'J',11:'Joker',12:'L',
                  13:'M',14:'N',15:'O',16:'P',17:'R',18:'S',19:'T',20:'U',21:'V',22:'X',23:'Z'}
files=os.listdir('antrenare/')
#print(files)
dictionar_coloane = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H', 9:'I', 10:'J', 11:'K', 12:'L', 13:'M', 14:'N', 15:'O'}
for file in files:
    if file[-3:]=='jpg':
        text = ""
        i = 0
        s = 0
        img = cv.imread('antrenare/'+file)
        result=extrage_careu(img)
        _, thresh = cv.threshold(result, 100, 255, cv.THRESH_BINARY_INV)
        matrice=determina_configuratie_careu_ocifre(result,thresh,lines_horizontal,lines_vertical)
        print(matrice)
        for linie in matrice:
            s = 0
            i+=1
            for element in linie:
                s+=1
                if element != " ":
                    text= text + str(i) + " "  +str(dictionar_coloane[s])+ '\n'
        with open(file[0: -3]+"txt", 'w') as file:
                file.write(text)
                file.close()
        print(text)



        #show_image('img',img)

