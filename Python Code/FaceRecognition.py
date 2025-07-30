import cv2
from ObjectIdentifier import OI
from numpy import zeros_like

def get_camera() -> int:
    selected_camera: str = input("Which camera would u like to use?\n")
    try:
        selected_camera: int = int(selected_camera)
    except ValueError:
        print("Invalid camera")
        return 0
    return selected_camera

def face_recognition() -> int | None:
    
    #import a pre-trained AI model for the face recognition
    model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    #start the camera capture
    selected_camera: int = get_camera()
    cam = cv2.VideoCapture(selected_camera)

    if not cam.isOpened():
        print(f"ERROR: Unable to access the cam {selected_camera}")
        return 1
    else:
        print(f"Cam {selected_camera} opened successfully!")
    
    print("press q to exit")

    #start the while loop
    while True:
        #gets one frame from the camera
        ret, frame = cam.read()

        if not ret:
            print("ERROR: Unable to read the frame")
            break

        #create a black image with the size of the frame
        masked_frame = zeros_like(frame)

        #initialize the ObjectIdentifier library
        person = OI(frame)
        #gets the coords of the people in the frame (if there are some)
        personcoords = person.PersonIdentifier()[0]

        #for every person that the OI finds it copies the region of the frame where there is the person
        #and it pastes it on the black frame
        for ax, ay, bx, by in personcoords:
            masked_frame[int(ay):int(by), int(ax):int(bx)] = frame[int(ay):int(by), int(ax):int(bx)]

        #it creates a gray version of the masked frame to help the AI to recognize the faces
        grayframe = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        #save all the faces that the AI has found in the faces array
        faces = model.detectMultiScale(grayframe, 1.1, 4, minSize = (30, 30))
        #creates a rectangle on every face to highlight it
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        #it shows the final frame and the maskedgray frame
        cv2.imshow("Face Detection", frame)
        cv2.imshow("Gray Face Detection", grayframe)
        #if q is pressed it breaks the while loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    return None

def main() -> None:
    face_recognition()
    return None

if __name__ == "__main__":
    main()
