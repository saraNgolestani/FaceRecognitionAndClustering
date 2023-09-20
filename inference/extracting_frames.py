import cv2
import os
import time
import shutil
import pickle
import face_recognition


# Extract 1 frame from each second from video footage
# and save the frames to a specific folder
def GenerateFrames(self, OutputDirectoryName):
    cap = cv2.VideoCapture(self.VideoFootageSource)
    _, frame = cap.read()

    fps = cap.get(cv2.CAP_PROP_FPS)
    TotalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print("[INFO] Total Frames ", TotalFrames, " @ ", fps, " fps")
    print("[INFO] Calculating number of frames per second")

    CurrentDirectory = os.path.curdir
    OutputDirectoryPath = os.path.join(
        CurrentDirectory, OutputDirectoryName)

    if os.path.exists(OutputDirectoryPath):
        shutil.rmtree(OutputDirectoryPath)
        time.sleep(0.5)
    os.mkdir(OutputDirectoryPath)

    CurrentFrame = 1
    fpsCounter = 0
    FrameWrittenCount = 1
    while CurrentFrame < TotalFrames:
        _, frame = cap.read()
        if (frame is None):
            continue

        if fpsCounter > fps:
            fpsCounter = 0

            frame = self.AutoResize(frame)

            filename = "frame_" + str(FrameWrittenCount) + ".jpg"
            cv2.imwrite(os.path.join(
                OutputDirectoryPath, filename), frame)

            FrameWrittenCount += 1

        fpsCounter += 1
        CurrentFrame += 1

    print('[INFO] Frames extracted')


# Following are nodes for pipeline constructions.
# It will create and asynchronously execute threads
# for reading images, extracting facial features and
# storing them independently in different threads

# Keep emitting the filenames into
# the pipeline for processing
class FramesProvider(Node):
    def setup(self, sourcePath):
        self.sourcePath = sourcePath
        self.filesList = []
        for item in os.listdir(self.sourcePath):
            _, fileExt = os.path.splitext(item)
            if fileExt == '.jpg':
                self.filesList.append(os.path.join(item))
        self.TotalFilesCount = self.size = len(self.filesList)
        self.ProcessedFilesCount = self.pos = 0

    # Emit each filename in the pipeline for parallel processing
    def run(self, data):
        if self.ProcessedFilesCount < self.TotalFilesCount:
            self.emit({'id': self.ProcessedFilesCount,
                       'imagePath': os.path.join(self.sourcePath,
                                                 self.filesList[self.ProcessedFilesCount])})
            self.ProcessedFilesCount += 1

            self.pos = self.ProcessedFilesCount
        else:
            self.close()


# Encode the face embedding, reference path
# and location and emit to pipeline
class FaceEncoder(Node):
    def setup(self, detection_method='cnn'):
        self.detection_method = detection_method

    # detection_method can be cnn or hog

    def run(self, data):
        id = data['id']
        imagePath = data['imagePath']
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(
            rgb, model=self.detection_method)

        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
             for (box, enc) in zip(boxes, encodings)]

        self.emit({'id': id, 'encodings': d})


# Receive the face embeddings for clustering and
# id for naming the distinct filename
class DatastoreManager(Node):
    def setup(self, encodingsOutputPath):
        self.encodingsOutputPath = encodingsOutputPath
    def run(self, data):
        encodings = data['encodings']
        id = data['id']
        with open(os.path.join(self.encodingsOutputPath,
                   'encodings_' + str(id) + '.pickle'), 'wb') as f:
            f.write(pickle.dumps(encodings))