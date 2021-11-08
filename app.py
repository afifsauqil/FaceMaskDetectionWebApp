from tensorflow.keras.models import load_model
import cv2
import numpy as np

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from PIL import Image
import base64
import io

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
StringIO = io.StringIO

class objectDetection():
    def __init__(self):
        print("Initializing.....")
        self.model = load_model(r'model-018.model') # load the best model based on accurancy 
        self.face_clsfr=cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml') # cascade classifier for face detection 
        self.labels_dict={0:'Mask',1:'No Mask'} # dictionary to show the results 
        self.color_dict={0:(0,255,0),1:(0,0,255)} # color the box based on labels predicted green and red
        print('Initialization Finish')

    def detect(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces= self.face_clsfr.detectMultiScale(gray,1.3,5)  

        for (x,y,w,h) in faces: # to run the model on each face detected above
        
            face_img=gray[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            result=self.model.predict(reshaped)

            label=np.argmax(result,axis=1)[0]
            
            # creating 2 rectangles for the faces and inserting text "pakai maskernya!" and "sip" based on the lable above
            cv2.rectangle(img,(x,y),(x+w,y+h),self.color_dict[label],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),self.color_dict[label],-1)
            cv2.putText(img, self.labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        return img

@app.route('/', methods=['POST', 'GET'])
def homepage():
    return render_template("index.html")

@socketio.on('image')
def processImage(data_image):
    print('receive')
    if(str(data_image) == 'data:,'):
        pass
    else:
        sbuf = StringIO()
        sbuf.write(data_image)

        b = io.BytesIO(base64.b64decode(data_image))
        pimg = Image.open(b)
    
        frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
       
        frame = od.detect(frame)

        width = 800
        height = int(frame.shape[0]*(width/frame.shape[1]))
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = 'data:image/jpg;base64,'
        stringData = b64_src + stringData
    
        print('starting emit')
        emit('response_back', stringData) 

if __name__=="__main__":
    od = objectDetection()
    socketio.run(app)

