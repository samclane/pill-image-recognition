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
for filename in has_images.splimage.values:
    # read in image
    img = cv2.imread('images_full/'+str(filename)+'.jpg')
    if img is None:
        continue
    # segment image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 1)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
    cont_img = closing.copy()
    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)
    pass
cv2.imshow('test', thresh)
print('done')
