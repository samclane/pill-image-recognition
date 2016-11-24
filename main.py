import cv2, sqlite3, time, os.path, pandas
import numpy as np



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
# df['image'] = pandas.Series([None for i in len(df['index'])], index=df.index)
# get dataframe of all pills with image
has_images = df[df['has_image'] == True]
# for all pills that have files
for filename in has_images.splimage.values:  # should try and maintain id here
    # read in image
    img = cv2.imread('images_full/' + str(filename) + '.jpg')
    if img is None:
        continue
    img = cv2.normalize(img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # trim the bottom watermark off dumbly
    img = img[0:-1 - 13, 0:-1]
    v = np.median(img)
    sigma = 0.50
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 3, 1)
    edge = cv2.Canny(thresh, lower, upper)
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edge, dil_kernel, iterations=1)
    height, width, depth = img.shape
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((height, width), np.uint8)
    # cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imshow('cont', img)
    cv2.waitKey(0)
    for cnt in contours:
        cnt[-1] = cnt[0]
        area = cv2.contourArea(cnt)
        # if contour area is less than .1% of image, discard
        if (area / (width * height)) < 0.001:
            print "Area: " + str(area / (width * height))
            continue
        if len(cnt) < 5:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if aspect_ratio > 5 or aspect_ratio < 0.1:
            print ("Aspect Ratio: " + str(aspect_ratio))
            continue
        # epsilon = 0.001*cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx = cv2.convexHull(cnt, returnPoints=True)
        cnt = approx
        # ellipse = cv2.fitEllipse(cnt)
        # cv2.ellipse(ellipse_image, ellipse, 1,-1)
        # Approx poly to smooth
        cv2.fillPoly(mask, pts=[cnt], color=(255, 255, 255))
        masked_data = cv2.bitwise_and(img, img, mask=mask)
        mean_color = cv2.mean(masked_data, mask=mask)
        cv2.imshow("masked", masked_data)
        # print "Mean color: " + str(mean_color)
        cv2.waitKey(0)
    # continue processing
    pass

print('done')

# problems, todo
# 	- still grabs axis labels
# 	- some images don't close
