#import Giacobbe as gc
from ultralytics import YOLO

class OI():

    #init function
    def __init__(self, image):
        self.image = image
        self.model = YOLO("yolov8n.pt")
        self.results = self.model.predict(self.image)
        self.otype = []
        self.confidence = []
        self.coords = []
        self.datas = [self.coords, self.confidence, self.otype]

    #this function prints all the otype/cls values and their corrisponding object for example:
    #0 = person
    #1 = bicycle
    #...
    def printinfo(self):
        print("This are the values of the otype or cls value and what objects are related to:\n", self.model.names)

    #this function returns the resolution of the input image
    def getResolution(self):
        height, width, channels = self.image.shape
        return [height, width, channels]

    #this function is used to identify all the objects, animals and people in one image
    def detect(self):
        for r in self.results:
            for box in r.boxes:
                otype = int(box.cls[0])         #object type
                # person = 0
                # general objects > 0
                confidence = float(box.conf[0]) #confidence
                ax, ay, bx, by = box.xyxy[0]    #box coords

                self.otype.append(otype)
                self.confidence.append(confidence)
                self.coords.append([ax, ay, bx, by])
        
        return self.datas
    
    #this function is used only to detect people
    def PersonIdentifier(self):
        for r in self.results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    otype = 0                       #object type
                    confidence = float(box.conf[0]) #confidence
                    ax, ay, bx, by = box.xyxy[0]    #box coords

                    self.otype.append(otype)
                    self.confidence.append(confidence)
                    self.coords.append([ax, ay, bx, by])
        
        return self.datas
