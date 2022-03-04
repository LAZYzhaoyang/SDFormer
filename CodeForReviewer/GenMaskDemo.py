import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_mask(h=256, w=256, n=8, min_area_rate=0.2, max_area_rate=0.9, spot_num=80, max_R=5, min_R=1, thresh=0.5, win_size=5):
    img = np.zeros((h, w), np.uint8)
    area = h*w
    min_area = round(area*min_area_rate)
    max_area = round(area*max_area_rate)
    init_mask = generate_initial_mask(image=img, n=n, min_area=min_area, max_area=max_area)
    masks = sp_noice_mask(img=init_mask, num=spot_num, max_R=max_R, min_R=min_R, thresh=thresh, win_size=win_size)
    out = {'init_mask':init_mask, 
           'mask':masks['mask'], 
           'spot_mask':masks['spot_mask'], 
           'mask_without_median':masks['mask_without_median']}
    return out

def generate_initial_mask(image, n, min_area, max_area):
    area = 0
    row, col= image.shape # 行,列
    #print(row)
    while area<min_area or area>max_area:
        point_set = np.zeros((n, 1, 2), dtype=int)
        for j in range(n):
            c = np.random.randint(0, col-1)
            r = np.random.randint(0, row-1)
            point_set[j, 0, 0] = c
            point_set[j, 0, 1] = r
        hull = []
        hull.append(cv2.convexHull(point_set, False))
        drawing_board = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(drawing_board, hull, -1, (1, 1), -1)
        area = cv2.contourArea(hull[0])
    #cv2.namedWindow("drawing_board", 0), cv2.imshow("drawing_board", drawing_board), cv2.waitKey()

    return drawing_board

def sp_noice_mask(img, num=80, max_R=5, min_R=1, thresh=0.5, win_size=5):
    mask = np.ones_like(img)
    h, w = img.shape
    for i in range(num):
        r = np.random.randint(min_R, max_R)
        hindex = np.random.randint(0, h-1)
        windex = np.random.randint(0, w-1)
        cv2.circle(mask, (windex, hindex), r, (0, 0), -1)
    img = img * mask
    img_median = cv2.medianBlur(img, win_size)
    img_median[img_median>=thresh]=1
    img_median[img_median<thresh]=0
    
    out={'mask':img_median, 'spot_mask':mask, 'mask_without_median':img}

    return out

def demo():
    h,w = 64,64
    n=8
    min_area_rate=0.05
    max_area_rate=0.15
    spot_num=150
    max_R=5
    min_R=1
    thresh=0.5
    win_size=5
    
    masks = get_mask(h=h,w=w,n=n,
                     min_area_rate=min_area_rate,
                     max_area_rate=max_area_rate,
                    spot_num=spot_num,
                    max_R=max_R,
                    min_R=min_R,
                    thresh=thresh,
                    win_size=win_size)
    

    plt.figure(figsize=(15,15))
    plt.subplot(2,2,1)
    plt.title('step 3')
    plt.imshow(masks['init_mask'])

    plt.subplot(2,2,2)
    plt.title('step 4')
    plt.imshow(masks['spot_mask'])

    plt.subplot(2,2,3)
    plt.title('step 5')
    plt.imshow(masks['mask_without_median'])

    plt.subplot(2,2,4)
    plt.title('step 6')
    plt.imshow(masks['mask'])

    plt.savefig('maskgenstep.png', bbox_inches='tight')
    plt.savefig('maskgenstep.svg', format='svg', bbox_inches='tight')
    
if __name__ == "__main__":
    demo()
