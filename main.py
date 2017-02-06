import cv2, sqlite3, time, os.path, pandas, math, sys
import numpy as np
from matplotlib import pyplot as plt

TARGET_PIXEL_AREA = 1000000.0


# init_db(void):
# initializes pill database with files obtained from https://pillbox.nlm.nih.gov/developer.html
# returns pandas.DataFrame holding SQL database data (acquired from .tab file)
def init_db():
    # Create database
    db = sqlite3.connect('example.db')
    db.text_factory = str
    c = db.cursor()
    # Read schema from file
    with open('mysql_create_engine_data_20150511.txt', 'r') as schema:
        command = schema.read()
    time.sleep(1)
    # If database hasn't been created, create it
    if not os.path.isfile('example.db'):
        c.execute(command)
    # read into datbase from tab file
    with open('pillbox_engine_20150511.tab', 'r') as pill_data:
        df = pandas.read_csv(pill_data, delimiter='\t', low_memory=False)
        df.to_sql('engine_data', db, if_exists='replace', index=False)
    return df


# seg_img(cv2.Image img)
# returns contours of image using canny edge detection and cv2.findContours
def seg_image(img):
    # grayscale and gaussian blur for adaptive thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 3)
    # Edge Detection
    edge = cv2.Canny(gray, 0, 15)
    # dilate image to close contours
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edge, dil_kernel, iterations=1)
    # find image contours
    im2, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return (contours, hierarchy)


# extract_features(cv2.Image img, list contours, boolean debug)
# filters through list of contours, only selecting ones that fit potential pill criteria. Then uses them to mask img,
#   extracting histogram, mean color, and aspect ratio
def extract_features(img, contours, debug=False):
    return_dict = {}
    height, width, depth = img.shape
    # close all contours
    for cnt in contours:
        cnt[-1] = cnt[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for idx, cnt in enumerate(contours):
        # re-init mask
        mask = np.zeros((height, width), np.uint8)
        area = cv2.contourArea(cnt)
        # acquire size features of contour using extreme points rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        # if contour area is too small, discard
        if area < 1500:
            continue
        # # if contour has too few points, discard
        if len(cnt) < 5:
            continue
        # get aspect ratio of contour from bounding rectangle.
        if aspect_ratio > 5 or aspect_ratio < 0.1:
            continue
        # Use convex hull to close the contour
        approx = cv2.convexHull(cnt, returnPoints=True)
        cnt = approx
        # Approx poly to smooth
        cv2.fillPoly(mask, pts=[cnt], color=(255, 255, 255))
        masked_data = cv2.bitwise_and(img, img, mask=mask)
        if debug:
            cv2.imshow('debug', masked_data)
        # image has been masked, time to collect data
        mean_color = cv2.mean(masked_data, mask=mask)
        hist = cv2.calcHist([img], [0], mask, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        # Add the data to the dataframe
        return_dict[idx] = pandas.Series([aspect_ratio, mean_color, hist],
                                         index=['aspect_ratio', 'mean_color', 'histogram'])
    return pandas.DataFrame(return_dict).transpose()


# resize the image while maintaining aspect ratio
def resize_img(img):
    ratio = float(img.shape[1]) / float(img.shape[0])
    new_h = int(math.sqrt(TARGET_PIXEL_AREA / ratio) + 0.5)
    new_w = int((new_h * ratio) + 0.5)
    img = cv2.resize(img, (new_w, new_h))
    return img

# main function
if __name__ == "__main__":
    df = init_db()
    # get dataframe of all pills with image
    pill_df_dict = {}
    # for all pills that have files
    for index, row in df.iterrows():
        # read in image
        filename = 'images_full/' + str(row['splimage']) + '.jpg'
        img = cv2.imread(filename)
        if img is None:
            # print("Image with filename " + str(row['splimage']) + " not found."
            continue
        # trim the bottom watermark off dumbly
        img = img[0:-1 - 13, 0:-1]
        # resize the image to a standard area while maintaining aspect ratio
        img = resize_img(img)
        contours, hierarchy = seg_image(img)
        pill_dataframe = extract_features(img, contours)
        # create the dataframe holding all views for a single pill product
        pill_df_dict[row['ID']] = pill_dataframe
    # iterate through consumer images
    for filename in os.listdir("consumer"):
        # pass image to test
        compareImage = cv2.imread("consumer/" + filename)
        try:
            ID_num = df[df['splimage'] == filename[:-4]]['ID'].iloc[0]
        except Exception:
            print("ID num not found")
            continue
        # image was found
        # resize and preprocess
        compareImage = resize_img(compareImage)
        (contours, hierarchy) = seg_image(compareImage)
        compare_df = extract_features(compareImage, contours)
        results = pandas.DataFrame()
        # if no features were found, skip pill
        if compare_df.empty:
            continue
        for idx, pill in pill_df_dict.items():
            pill_result = pandas.Series()
            # ensure DataFrame exists
            if pill.empty:
                continue
            # initialize comparison parameters to max
            hist_corr = sys.maxsize
            aspect_difference = sys.maxsize
            mean_color_distance = sys.maxsize
            # iterate over each pill view in image, selecting one that best fits the comparison image
            for index, view in pill.iterrows():
                # compare histogram correlation
                comp_hist = math.fabs(cv2.compareHist(compare_df.iloc[0]['histogram'], view['histogram'], method=0) - 1)
                if comp_hist < hist_corr:
                    hist_corr = comp_hist
                # compare aspect ratio difference
                comp_asp = math.fabs(compare_df.iloc[0]['aspect_ratio'] - view['aspect_ratio'])
                if comp_asp < aspect_difference:
                    aspect_difference = comp_asp
                # compare mean color distance
                comp_mean = np.linalg.norm(
                    np.array(compare_df.iloc[0]['mean_color'][:3]) - np.array(view['mean_color'][:3]))
                if comp_mean < mean_color_distance:
                    mean_color_distance = comp_mean
            # store result as an indexed series
            pill_result = pandas.Series(
                [hist_corr, aspect_difference, mean_color_distance],
                index=['hist_corr', 'aspect_diff', 'mean_color_diff'], dtype=np.float16)
            # add result to dataframe
            results[idx] = pill_result
        # normalize results by row
        results_norm = results.div(results.sum(axis=1), axis=0)
        # calculate net error for each pill
        error_dict = {}
        for idx, pill in results_norm.items():
            error_dict[idx] = pill.sum(axis=0)
        # Sort list by increasing error
        conf_list = [x[0] for x in sorted(error_dict.items(), key=lambda x: 1 - x[1], reverse=True)]
        # Print confidence of "dispensed" pill as percentile
        print("Confidence is " + str(
            100 * (1.00 - conf_list.index(ID_num) / float(len(conf_list)))) + " for pill ID " + str(ID_num))
