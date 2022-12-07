import cv2
import torch
import WaggleChatHelper as H
from torchvision import transforms as T
import numpy as np
import gc

from matplotlib import pyplot as plt
from skimage.transform import resize
import time

class WaggleChat():
    def __init__(self, convSize, threshold, elimSize, imageStackSize, maxDistance, minPoints, maxTime, bufferSize):
        self.convSize = convSize
        self.threshold = threshold
        self.elimSize = elimSize
        self.imageStackSize = imageStackSize
        self.bufferSize = bufferSize
        self.maxDistance, self.minPoints, self.maxTime = maxDistance, minPoints, maxTime
        self.imageResize = 5

        self.images= []
        self.curDifImages = []

        self.waggleCordinates = []
        self.waggleSnapShots = []

        self.curPotentialRuns = []
        self.completedRuns = []
        self.curPotentialDances = []
        self.completedDances = []

        self.curFrameInfo = None
        self.package = []

        self.t = -1
        self.lastFrameSent = -1
        self.curRunId = 0
        self.SNAPSHOTSIZE = 60//self.imageResize
        self.previousFiveImages = []
        self.lastDanceID = 0


        self.times = []

    def initWithFrame(self, image):
        h, w, c = image.shape
        self.h = h//self.imageResize
        self.w = w//self.imageResize
        self.transform = T.Compose([T.Resize((h//self.imageResize, w//self.imageResize))])

    def updateWithImage(self, image):
        self.t += 1
        self.images.append(image)
        ##process a new batch of images once we have enough stacked

        if (len(self.images) + len(self.previousFiveImages)) % (self.imageStackSize + 5) == 0:
            st = time.time()
            self.images = self.previousFiveImages + self.images
            self.previousFiveImages = self.images[-5:]
            self.processImages()
            self.updateRunsPredictions()

            print(f'Time for {self.imageStackSize} is {time.time() - st} for {self.imageStackSize/30} seconds of video')
            print(f'for an average of {(time.time() - st)/self.imageStackSize} per frame')

        ##Update Waggle predictions
        return self.updateBuffer()



    def processImages(self):
        images = []
        for image in self.images:
            self.package.append(frameInfo(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = H.enhaceImage(image)[:, :, 2]
            images.append(image)

        self.images = []
        images = self.transform(torch.from_numpy(np.array(images))) / 255 #  for cuda upgrades

        difImages = np.array(H.getDifImagesStacked(images, 5, asFigs=False))
        convImages = [H.runConvs(image, self.convSize) for image in difImages]
        self.curDifImages = difImages
        for i, im in enumerate(difImages):
            self.package[i + len(self.waggleCordinates)].difIm = im

        cordinates = np.array([H.findWagglesElim(im, th=self.threshold, elimRadius=self.elimSize)  for im in convImages])

        del convImages
        del images
        gc.collect()

        ## Update system with new waggle cordinates and snapshots for angle processing
        self.waggleCordinates = list(cordinates) if len(self.waggleCordinates)== 0 else self.waggleCordinates + list(cordinates)

    def updateRunsPredictions(self):
        """
        CompletedRunclusters are takings the points and clusters all points based on min distanse and max time.
        This cluster should delete clusters where the last point is greater than max time indicating the run is complete.
        What is returned is the deleted clusters with more than or equal to min points
        :return: nothing
        """
        numCors = len(self.waggleCordinates)
        print(f'num frames {numCors}')
        for i, points in enumerate(self.waggleCordinates[-self.imageStackSize:]):
            t = i + numCors - self.imageStackSize
            for cor in points:
                self.updatePoint(cor, t)
        ### Get snapshot images for angle predictions
        snapShots = []
        oldNumCompletedRuns = len(self.completedRuns)
        for run in self.completedRuns:

            runSnapShots = []
            for point in run:
                index = point[2]
                if self.package[index]:
                    image = H.getImAtPoint(self.package[index].difIm,point[:2], self.SNAPSHOTSIZE)
                    if image.shape[0] >= self.SNAPSHOTSIZE*2 - 5 and image.shape[1] >= self.SNAPSHOTSIZE*2 - 5:

                        runSnapShots.append(image)
                    else:
                        pass
                        #print(image.shape)
                else:
                    print('ERROR outside of buffer size')
            snapShots.append(runSnapShots)

        preds = H.predsFromRuns(self.completedRuns, len(self.waggleCordinates))

        self.completedRuns = H.cluster(preds, self.maxDistance, self.minPoints, self.maxTime)

        print(f'preds from runs {preds[-self.imageStackSize:]}')
        #print(f'completed runs {self.completedRuns} with {len(self.completedRuns) - oldNumCompletedRuns}')

        angles = [H.getPolygonAngle(runShots, self.threshold, self.SNAPSHOTSIZE) for runShots in snapShots]
        print(f'angles {angles} print leen {len(angles)} vvs len {oldNumCompletedRuns}')
        self.updatePackage(self.completedRuns, angles)

        self.completedRuns = []
    def updatePackage(self, completedRuns, angles):
        ##frame + list[ x, y, angle, duration in frames, gps locations, run id, dance id mod 100, return time]
        for angle, run in zip(angles, completedRuns):
            duration = H.getRunDurations(run)
            print(f'duration {duration}')
            long, lad, = H.mockGPS(angle, duration)
            runID = self.curRunId
            danceID, returnTime = self.clusterRuns(run)
            danceID = danceID % 100
            self.curRunId += 1
            for point in run:
                x, y, t = point
                self.package[t].addCordinateInfo([x,y,t,angle, duration, long, lad, runID, danceID, returnTime])
    def updateBuffer(self):
        #print(f'buffer complete {self.t}')
        if self.t - self.lastFrameSent > self.bufferSize:
            package = self.sendFrame()
            self.package[self.lastFrameSent] = None
            return package

        else:
            return None

    def sendFrame(self):
        self.lastFrameSent += 1
        self.package[self.lastFrameSent].difIm = None

        curInfo = self.package[self.lastFrameSent].beeWaggleInfo
        cords = np.array([[info[0], info[1]] for info in curInfo]) * self.imageResize
        angles = [info[3] for info in curInfo]
        
        ## for creating video
        self.package[self.lastFrameSent].figure = H.plotPointsOnImage(self.package[self.lastFrameSent].frame, f'frame {self.lastFrameSent}', cords, angles = angles, plot=False)
        plt.clf()



        return self.package[self.lastFrameSent]

    def clusterRuns(self, run):
        self.lastDanceID += 1
        return self.lastDanceID, -1


    def updatePoint(self, cor, t):
        ##Check to see if point needs to be
        for j, pRun in enumerate(self.curPotentialRuns):
            if H.distance(cor[:2], pRun[-1][:2]) < self.maxDistance:
                if self.curPotentialRuns[j][-1][2] != t:
                    # self.curPotentialRunsIndecies[j].append(t)
                    self.curPotentialRuns[j].append([cor[0], cor[1], t])
                return
        ##if not create a new run
        newRun = [[cor[0], cor[1], t]]
        self.curPotentialRuns.append(newRun)
        # self.curPotentialRunsIndecies.append([t])

        ## meets the criteria for potential run completing so add to Real Runs
        runsToPop = []
        for j, pRun in enumerate(self.curPotentialRuns):
            isRun = t - pRun[-1][2] > self.maxTime
            if isRun:
                if len(pRun) >= self.minPoints:
                    self.completedRuns.append(np.array(pRun))
                    # RunsIndecies.append(curPotentialRunsIndecies[j])
                runsToPop.append(j)
        self.curPotentialRuns = [self.curPotentialRuns[i] for i in range(len(self.curPotentialRuns)) if
                                 i not in runsToPop]


class frameInfo:
    def __init__(self, frame, difIm = None, beeWaggleInfo = []):
        self.frame = frame
        self.difIm = difIm
        self.beeWaggleInfo = beeWaggleInfo
        self.figure = []

    def addCordinateInfo(self, beeWaggleItem):
        self.beeWaggleInfo.append(beeWaggleItem)

def loadVideo(path, waggleChat):
    video = cv2.VideoCapture(path)
    success, image = video.read()
    waggleChat.initWithFrame(image)
    frameStack = []
    while success:
        frameInfo = waggleChat.updateWithImage(image)[0]
        if frameInfo:

            pass

        success, image = video.read()
        if len(frameStack) == 300:
            frameStack = np.array(frameStack)
            H.arrayToVideo(frameStack, f'home/beesearcher/SeansBeeAlgo/BeeVideos/runsVideo.mp4')
            success=False
    print('Finsihed Video Test !!!')


if __name__ == '__main__':
    n = 41
    STINGPATH = f'home/beesearcher/SeansBeeAlgo/beesearch-hand-annotated-data-main/Video/RawFootage/WaggleDance_{n}.mp4'
    WC = WaggleChat(convSize=12, threshold=.6, elimSize=20, imageStackSize=100,
                    maxDistance=10, minPoints=3, maxTime=10, bufferSize=200)
    loadVideo(STINGPATH, WC)
        