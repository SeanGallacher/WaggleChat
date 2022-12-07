import numpy as np
import cv2
import imantics
from imantics import Mask
from skimage import draw
from sklearn import svm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
def findWagglesElim(transform, th=.6, elimRadius=0, maxWaggles=25,
                       thRatio=(1 / 2), minWaggles=0):
    highestPoints = []
    continueSearch = True
    wagglesFound = 0
    if np.max(transform.flatten()) < th:
        continueSearch = False
    while (continueSearch and wagglesFound < maxWaggles) or (wagglesFound < minWaggles):
        y, x = k_largest_index_argpartition_v2(transform, 1).squeeze()  ##place to optimize
        if elimRadius == 0:
            elimRadius = findElimRadius(y, x, transform, th * thRatio)
        mask = createCircleMask(transform.shape, elimRadius, np.array([y, x]))
        transform *= mask
        highestPoints.append(np.array([x, y]))
        if np.max(transform.flatten()) < th:
            continueSearch = False
        wagglesFound += 1
    return highestPoints

def runConvs(images, k):
    conv = torch.ones((k, k)).unsqueeze(0).unsqueeze(0).cuda().float()
    transform = F.conv2d(images.unsqueeze(0).cuda().float(), conv,
                         padding='same')
    transform = transform.squeeze().cpu().detach().numpy() / (k**2)

    return transform





def predsFromRuns(runs, pred_length):
   #print(f'pred lengths {pred_length}')
    preds = [[] for t in range(pred_length)]
    lastT = 0
    for i, run in enumerate(runs):
        run = run[run[:, 2].argsort()]
        lastKnownLoc = run[0][:2]

        # for t in range(pred_length):
        for j, point in enumerate(run):
            timeIndex = point[2]
            cords = point[:2]
            ### fill in the holes in the run
           #print(f'cordinates needed to fill from {timeIndex} to {lastT} : {timeIndex - lastT - 1}')
            if timeIndex - lastT > 1:
                for k in range(1, timeIndex - lastT):
                    prevCords = np.array(run[j - 2:j + 1][:, :2])
                    if len(prevCords) != 0:
                       #print(f'smoothing cordinate { np.mean(np.array(run[j-2:j+1][:,:2]),axis=0)}')
                        preds[lastT + k].append(np.mean(prevCords, axis=0).astype(int))
            preds[timeIndex].append(cords)
            lastT = timeIndex
            lastKnownLoc = cords

    
    return preds


def clusterRuns(runs, maxdist=30):
    clusterRuns = []
    runMidPoints = []
    for run in runs:
        runMid = np.mean(run, axis=0)
        runMidPoints.append([runMid])

    ## gives you the runs clusters according to mid points
    _, indecies = cluster(runMidPoints, maxdist=maxdist, minpoints=1, maxtDelta=8 * 30, returnIndex=True)

    for i, runInds in enumerate(indecies):
        clusterRuns.append([])
        for j in runInds:
           #print(len(runs))
           #print(j)
            for point in runs[j]:
                clusterRuns[i].append(point)


    return clusterRuns


def getPolygonAngle(snapshots, th, d = 60):
    return 0
    angles = []
    print(f'running kMeans Angle Algo with d {d}')

    for snapshot in snapshots:
        difIm =snapshot
        k = (d//6)

        if difIm.shape[0] >=k and difIm.shape[1] >= k:
            difIm = runConvs(difIm, k)
        num_polys = 0
        thMulti = 1
        difIm = np.array(difIm)
        if np.sum(difIm) ==0:
            print(f' Dif Im is empty')
            continue
        else:
            pass
            #print(f'dif image is not empty')

        difIm[difIm < th*thMulti+1] = 0


        mask = np.zeros_like(difIm)
        mask[difIm > 0] = 1

        verticies = imantics.Polygons.from_mask(Mask(mask)).points

        polygons = []
        polysizes = []
        im = np.zeros_like(difIm)
        for vs in verticies:
            vs = np.array(vs)
            polygon = draw.polygon(vs[:,1], vs[:,0])
            polygons.append(polygon)
            polysize = len(np.array(list(polygon)).T)
            polysizes.append(polysize)

        num_polys = len(polysizes)
        thMulti += 1
        if num_polys < 2:
            continue
        inds = k_largest_index_argpartition_v2(np.array(polysizes), 2).squeeze()
        poly1 = np.array(list(polygons[inds[0]])).T
        poly2 = np.array(list(polygons[inds[1]])).T

        for cor in poly1:
            im[cor[0], cor[1]] = 1
        for cor in poly2:
            im[cor[0], cor[1]] = -1

        m, b = svmPredict(im)
        angle = np.arctan(m)
        angle = (angle)
        angles.append(angle)
        #plotAngleOnImage(np.degrees(np.pi/2 + angle), [d, d], im, title='k Means angle', colorbar=True)

    return getAngleMeanI(angles)

def mockGPS(angle, duration):
    return 1,1

def getRunDurations(run, FPS = 30):
    return (run[-1][2] - run[0][2]) 

def enhaceImage(img):

    # converting to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img


def k_largest_index_argpartition_v2(a, k):
    idx = np.argpartition(a.ravel(), a.size - k)[-k:]
    return np.column_stack(np.unravel_index(idx, a.shape))

def createCircleMask(imageShape, radius, center):
    rows, cols = imageShape

    mask = np.ones((rows, cols), np.uint8)
    x, y = np.ogrid[:rows, :cols]

    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius * radius

    mask[mask_area] = 0

    return mask

def getImAtPoint(frame, point, d):
    ##point is in x, y cordinates
    point = np.array(point).astype(int)
    try:
        return frame[point[1] - d: point[1] + d, point[0] - d: point[0] + d]
    except IndexError:
        return np.array([])

def distance(a1, a2):
    return np.sqrt(np.sum(np.square(a1 - a2)))


def cluster(preds, maxdist=20, minpoints=5, maxtDelta=15, returnIndex=False):
    Runs = []
    RunsIndecies = []
    curPotentialRuns = []
    curPotentialRunsIndecies = []
    for t, pred in enumerate(preds):
        for cor in pred: ## take all cordinated at a given time stamp

            madeLabel = False
            #print(f'curr potential runs {curPotentialRuns}')
            for j, pRun in enumerate(curPotentialRuns):
               #print(f' pruns {pRun}')
                if madeLabel:
                    continue
                if distance(cor[:2], pRun[-1][:2]) < maxdist:
                    madeLabel = True
                    if curPotentialRuns[j][-1][2] != t:
                        curPotentialRunsIndecies[j].append(t)
                        curPotentialRuns[j].append([cor[0], cor[1], t])

            if not madeLabel:
                newRun = [[cor[0], cor[1], t]]
                curPotentialRuns.append(newRun)
                curPotentialRunsIndecies.append([t])

        ## meets the criteria for potential run completing so add to Real Runs
        runsToPop = []
        for j, pRun in enumerate(curPotentialRuns):
            if returnIndex:
                isRun = t - pRun[-1][2] > maxtDelta or t == len(preds) - 1
            else:
                isRun = t - pRun[-1][2] > maxtDelta
            if isRun:
                #print(f'run to add {pRun}')
                if len(pRun) >= minpoints:
                   #print(pRun)
                   #print(f'Runs numbers {np.array(pRun)[:,2]}')
                    Runs.append(np.array(pRun))
                    RunsIndecies.append(curPotentialRunsIndecies[j])
                    #print('Run added')
                runsToPop.append(j)

        curPotentialRuns = [curPotentialRuns[i] for i in range(len(curPotentialRuns)) if i not in runsToPop]
    for j, pRun in enumerate(curPotentialRuns):

        if len(pRun) >= minpoints:
            # print(pRun)
            # print(f'Runs numbers {np.array(pRun)[:,2]}')
            Runs.append(np.array(pRun))
            RunsIndecies.append(curPotentialRunsIndecies[j])
            #print('Run added')
   #print(f'RUNs {Runs}')
    if returnIndex:
       #print('runs info ')
       #print(Runs, curPotentialRunsIndecies)
        return Runs, curPotentialRunsIndecies
    return Runs

def normAngle(angle):
    while angle > np.pi or angle < 0:
        if angle > np.pi:
            angle -= np.pi
        if angle < 0:
            angle += np.pi
    return angle

def getAngleMeanI(anglePreds, nump=180):
    possibleAngles = np.linspace(0, np.pi, nump)
    angleDifs = [np.mean([np.abs(getAngleDif(a, pred)) for pred in anglePreds]) for a in possibleAngles]
    #print(f'angleDifs ')
    index = np.argmin(angleDifs)
    print(f'bst angle index {index}')

    return possibleAngles[index]

def getAngleDif(x, y, halfCircle=True):
    if halfCircle:
        x = normAngle(x)
        y = normAngle(y)
    angle = np.arctan2(np.sin(x - y), np.cos(x - y))
    if np.abs(angle) > np.pi / 2:
        angle = np.pi - np.abs(angle)
    return angle

def svmPredict(waggleIm, plot=False, weighted=True):
    w, h = waggleIm.shape
    X = []
    Y = []
    weights = []
    for i in range(w):
        for j in range(h):
            if waggleIm[i, j] > 0:
                X.append([i, j])
                Y.append(1)
                weights.append(waggleIm[i, j])
            elif waggleIm[i, j] < 0:
                X.append([i, j])
                Y.append(0)
                weights.append(waggleIm[i, j])
    weights = np.abs(weights)
    if True:
        posWeights = weights[weights > 0]
        negWeights = weights[weights < 0]
        ratio = np.sum(np.abs(posWeights))/np.sum(np.abs(negWeights))
        weights[weights < 0 ] *= ratio
    svc = svm.LinearSVC()
    if plot:
        X = np.array(X)
        plt.scatter(X[:, 1], X[:, 0])
        plt.title('waggle comb weights ')
        plt.colorbar()
        plt.show()

    if np.mean(Y) == 0 or np.mean(Y) == 1:
        if False:
            print(f'Y contains only {np.mean(Y)}')
            plt.title(f'Y contains only {np.mean(Y)}')
            plt.imshow(waggleIm)
            plt.colorbar()
            plt.show()
        return 0, 0
    try:
        if weighted:
            svc.fit(X, y=Y, sample_weight=weights)  # , sample_weight=weights
        else:
            svc.fit(X, y=Y)
    except Exception as e:
        if False:
            print(f"exception in SVC {e}")
            plt.title(f"exception in SVC {e}")
            plt.imshow(waggleIm)
            plt.colorbar()
            plt.show()
        return 0, 0
    W = svc.coef_[0]
    b = - svc.intercept_ / W[1]
    ## y  = ax - b
    m = W[0] / W[1]

    return m, b


def getDifImagesStacked(images, n=5, abs =True, asFigs=True, blur=0, k=3, altNeg=False):
    imagesVid = []
    imageQueue = []
    count = 0
    for i1, i2, in zip(images[:-1], images[1:]):
        difIm = np.abs(i1 - i2) if abs else i2 - i1
        if altNeg:
            if count % 2 == 0:
                difIm *= -1
            else:
                difIm *= 1
        imageQueue.append(difIm)

        if len(imageQueue) == n:
            im = np.sum(np.array(imageQueue), axis=0)
            im[np.abs(im) < 1.2] = 0
            #plotPointsOnImage(im)
            #plt.show()
            for i in range(blur):
                kernal = torch.ones((k, k)).unsqueeze(0).unsqueeze(0).cuda().float()
                im = runConvs(im, kernal, k ** 2)

            if asFigs:
                imagesVid.append(arrayToFigToArray(im))
            else:
                #im = im.flatten()
                imagesVid.append(im)
            imageQueue.pop(0)
    return imagesVid


def plotPointsOnImage(image, title='', points=[], points2=[], points3=[], angles= [], colorbar=True, plot=True):
    fig = plt.figure()
    plt.clf()
    for i, point in enumerate(points):
        plt.scatter(point[0], point[1], color='red')
        if len(angles) - 1 >= i:
            plot_line(point, angles[i], 20)
    for i, point in enumerate(points2):
        plt.scatter(point[0], point[1], color='aqua')
    for i, point in enumerate(points3):
        plt.scatter(point[0], point[1], color='pink')
    plt.imshow(image)
    if colorbar: plt.colorbar()
    plt.title(title)
    if plot:
        plt.show()
    else:
        return figToNumpy(fig)

def arrayToVideo(Frames, video_name):
    if len(Frames) == 0:
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    height, width, layers = Frames[0].shape
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))
    for image in Frames:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()

def figToNumpy(fig):
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    return data

def plot_line(point, angle, length, color='red'):
    '''
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.
    '''

    # unpack the first point
    x, y = point
    # find the end point
    # angle += 90
    starty = y - length * float(np.sin(np.radians(angle)))
    startx = x - length * float(np.cos(np.radians(angle)))

    endy = y + length * float(np.sin(np.radians(angle)))
    endx = x + length * float(np.cos(np.radians(angle)))

    xvals = [startx, endx]
    yvals = [starty, endy]
    plt.plot(xvals, yvals, color=color)