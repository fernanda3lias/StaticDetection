import numpy as np
import cv2
from collections import Counter, defaultdict
import settings

#import background image
background_file = settings.BACKGROUND_PATH

#setting resizable windows
MAIN_WINDOW = cv2.namedWindow('Abandoned Object Detection', cv2.WINDOW_NORMAL)

#CANNY_WINDOW = cv2.namedWindow('CannyEdgeDet', cv2.WINDOW_NORMAL)
#FRAME_WINDOW = cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
#MASKED_WINDOW = cv2.namedWindow('frame_masked', cv2.WINDOW_NORMAL)
#MORPH_WINDOW = cv2.namedWindow('Morph_Close', cv2.WINDOW_NORMAL)

#load background image
background_image = cv2.imread(background_file)

#create black mask + draw white polygon on black mask
mask = np.zeros(background_image.shape[:2],
                dtype="uint8")
#take same height and width as the image

#cv2.imshow('mask',mask)

#white: update pts which is the area of interest
pts = np.array([[300, 150], [50, 350], [250, 490], [400, 550], [450, 175]], np.int32)

#mask + white
cv2.fillPoly(mask, [pts], 70, 0)

#cv2.imshow('Masked', mask)

#reading video
cap = cv2.VideoCapture(settings.VIDEO_PATH)

#initialised BackgroundSuqbtractor
fgbg = cv2.createBackgroundSubtractorMOG2()
while (cap.isOpened()):

    ret, frame = cap.read()

    #have to check, else error on the last frame.
    if ret == 0:
        break
    settings.frameno = settings.frameno + 1
    cv2.putText(frame, '%s%.f' % ('Frameno:', settings.frameno), (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #applying mask on frame
    fgmask = fgbg.apply(frame)
    fgmask_masked = cv2.bitwise_and(fgmask, fgmask, mask=mask)

    #canny edge detection
    edged = cv2.Canny(fgmask_masked, 30, 100)  #any gradient between 30 and 150 are considered edges

    #cv2.imshow('CannyEdgeDet', edged)

    kernel2 = np.ones((2, 2), np.uint8)  # higher the kernel, eg (10,10), more will be eroded or dilated
    thresh2 = cv2.morphologyEx(fgmask_masked, cv2.MORPH_CLOSE, kernel2, iterations=1)

    #cv2.imshow('Morph_Close', thresh2)


    #creeate a copy of the thresh to find contours
    (cnts, _) = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mycnts = []  #every new frame, set to empty list.

    # loop over the contours
    for c in cnts:
        #calculate centroid using cv2.moments
        M = cv2.moments(c)
        if M['m00'] == 0:
            pass
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])


            # set contour criteria
            if cv2.contourArea(c) < 1580 or cv2.contourArea(c) > 25000:
                pass
            else:
                mycnts.append(c)

                #compute the bounding box for the contour, draw it on the frame and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                #putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
                # org ; is the (x,y) location
                cv2.putText(frame, 'C %s,%s,%.0f' % (cx, cy, cx + cy), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                            2)

                #store the cx+cy, a single value into a list ; max length of 10000
                # once hit 10000, tranfer top 20 points to dictionary ; empty list
                sumcxcy = cx + cy

                # track_list.append(cx+cy)
                settings.track_temp.append([cx + cy, settings.frameno])

                settings.track_master.append([cx + cy, settings.frameno])
                countuniqueframe = set(
                    j for i, j in settings.track_master)  # get a set of unique frameno. then len(countuniqueframe)


                if len(countuniqueframe) > settings.consecutiveframe:
                    minframeno = min(j for i, j in settings.track_master)
                    for i, j in settings.track_master:
                        if j != minframeno:  # get a new list. omit the those with the minframeno
                            settings.track_temp2.append([i, j])

                    settings.track_master = list(settings.track_temp2)  # transfer to the master list
                    settings.track_temp2 = []


                # count each of the sumcxcy
                # if the same sumcxcy occurs in all the frames, store in master contour dictionary, add 1
                countcxcy = Counter(i for i, j in settings.track_master)

                # print countcxcy
                # example countcxcy : Counter({544: 1, 537: 1, 530: 1, 523: 1, 516: 1})
                # if j which is the count occurs in all the frame, store the sumcxcy in dictionary, add 1

                for i, j in countcxcy.items():
                    if j >= settings.consecutiveframe:
                        settings.top_contour_dict[i] += 1

                if sumcxcy in settings.top_contour_dict:
                    if settings.top_contour_dict[sumcxcy] > 100:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        cv2.putText(frame, '%s' % ('CheckObject'), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)
                        print('Detected : ', sumcxcy, settings.frameno, settings.obj_detected_dict)

                        # Store those objects that are detected, and store the last frame that it happened.
                        # Need to find a way to clean the top_contour_dict, else contour will be detected after the
                        # object is removed because the value is still in the dict.
                        # Method is to record the last frame that the object is detected with the Current Frame (frameno)
                        # if Current Frame - Last Frame detected > some big number say 100 x 3, then it means that
                        # object may have been removed because it has not been detected for 100x3 frames.

                        settings.obj_detected_dict[sumcxcy] = settings.frameno

    for i, j in settings.obj_detected_dict.items():
        if frameno - settings.obj_detected_dict[i] > 200:
            print('PopBefore', i, settings.obj_detected_dict[i], settings.frameno, settings.obj_detected_dict)
            print('PopBefore : top_contour :', settings.top_contour_dict)
            settings.obj_detected_dict.pop(i)

            #Set the count for eg 448 to zero. because it has not be 'activated' for 200 frames. Likely, to have been removed.
            settings.top_contour_dict[i] = 0
            print('PopAfter', i, settings.obj_detected_dict[i], settings.frameno, settings.obj_detected_dict)
            print('PopAfter : top_contour :', settings.top_contour_dict)

    # cv2.putText(frame,'%s%s'%('Objects :',len(mycnts)), (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    #draw the area of interest on the frame
    cv2.polylines(frame, [pts], True, (255, 0, 0), thickness=2)

    #show images
    cv2.imshow('Abandoned Object Detection', frame)

    #cv2.imshow('frame', fgmask)
    #cv2.imshow('frame_masked', fgmask_masked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




