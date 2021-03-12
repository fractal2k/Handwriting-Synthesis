import numpy as np 
import cv2
import fire
import os


# src : https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def threshold_image(directory: str, cropped_dir_path: str):
    """
    Extracts all individual boxes of all the images in "directory" into the "cropped_dir_path" directory
    :param directory: first argument
    :param cropped_dir_path: second argument
    :return: None
    
    """
    imgs = [os.path.join(directory, f) for f in os.listdir(directory)]
    for image_idx, image_path in enumerate(imgs):
        img = cv2.imread(image_path, 0)  # grayscale
        (thresh, img_binary) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)  # Converting the image to binary
        img_binary = 255-img_binary  # inversion to B&W for better contour detection

        # Making edge detection kernels
        ver, hor = np.array(img).shape[0]//80, np.array(img).shape[1]//80
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor, 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Vertical edge extraction
        img_temp = cv2.erode(img_binary, ver_kernel, iterations=3)
        ver_img = cv2.dilate(img_temp, ver_kernel, iterations=3)

        # Horizontal edge extraction
        img_temp2 = cv2.erode(img_binary, hor_kernel, iterations=3)
        hor_img = cv2.dilate(img_temp2, hor_kernel, iterations=3)

        # Weights of both hor and ver line images, to be used for adding them up
        alpha = 0.5
        beta = 0.5

        # Making a final B&W image of the boxes
        img_final_bin = cv2.addWeighted(ver_img, alpha, hor_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)  # ~img_final_bin to invert back the colors
        (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Finding all rectangular contours in our binary image
        contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sorting all contours using a top down approach
        (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

        # Extracting the sorted contours into a directory file
        idx = 0

        for c in contours:
            # Returns the location and width, height for every contour
            x, y, w, h = cv2.boundingRect(c)
            # print(f'{w}, {h}')
            idx += 1
            if w < 340 and h < 120:
                new_img = img[y:y+h, x:x+w]
                if os.path.exists(cropped_dir_path):
                    cv2.imwrite(cropped_dir_path + '\\' + str(image_idx) + '_' + str(idx) + '.png', new_img)
                else:
                    os.makedirs(cropped_dir_path)
                    cv2.imwrite(cropped_dir_path + str(image_idx) + '_' + str(idx) + '.png', new_img)


if __name__ == "__main__":
    fire.Fire({
        "extract": threshold_image
    })
