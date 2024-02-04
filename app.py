from flask import Flask, render_template, redirect, url_for, session, request, flash, get_flashed_messages
from flask import Response,jsonify
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired
from flask_bcrypt import Bcrypt
import cv2, sys, numpy as np, json, os, base64, joblib, face_recognition
from io import BytesIO
from PIL import Image,UnidentifiedImageError
haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
import pandas as pd
app = Flask(__name__)
@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response

# Use SQLite database file named 'users.db' located in the project directory
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = '\xfd{H\xe5<\x95\xf9\xe3\x96.5\xd1\x01O<!\xd5\xa2\xa0\x9fR"\xa1\xa8'
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
db = SQLAlchemy(app)

class Authenticate(db.Model, UserMixin):
    sno = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(200), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.username} - {self.password}"
    def get_id(self):
        return str(self.sno)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    roll_no = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    enrollment_no = db.Column(db.String(20), unique=True, nullable=False)
    def __repr__(self):
        return f"{self.roll_number} - {self.name}"

# class AttendanceRecord(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     roll_no = db.Column(db.String(20), unique=True, nullable=False)
#     name = db.Column(db.String(200), nullable=False)
#     enrollment_no = db.Column(db.String(20), unique=True, nullable=False)
#     subject_name = db.Column(db.String(200),nullable=False)
#     batch = db.Column(db.String(20))
#     slot_type = db.Column(db.String(200),nullable=False)
#     date = db.Column(db.Date,nullable=False)
#     present = db.Column(db.String(1),default='n') # n means absent

#     def __repr__(self):
#         return f"{self.roll_number} - {self.name}"

class AttendanceRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date,nullable=False)
    roll_no = db.Column(db.String(20),nullable=False)
    name = db.Column(db.String(200),nullable=False)
    enrollment_no = db.Column(db.String(20), unique=True, nullable=False)
    present = db.Column(db.String(1),default='n') # n means absent

class TimeTable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    batch = db.Column(db.String(20), nullable=False)
    day = db.Column(db.String(20), nullable=False)
    slot1 = db.Column(db.String(200))
    slot2 = db.Column(db.String(200))
    slot3 = db.Column(db.String(200))
    slot4 = db.Column(db.String(200))
    slot5 = db.Column(db.String(200))
    slot6 = db.Column(db.String(200))
    slot7 = db.Column(db.String(200))
    slot8 = db.Column(db.String(200))
    def __repr__(self):
        return f"{self.day}"

@login_manager.user_loader
def load_user(user_id):
    return Authenticate.query.get(int(user_id))

@app.route("/", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['uname']
        pwd = request.form['pwd']

        user = Authenticate.query.filter_by(username=uname).first()
        if user and bcrypt.check_password_hash(user.password, pwd):
            login_user(user)
            if current_user.is_authenticated:
                flash('Login Successful!','info')
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid credentials")

    return render_template("login.html")

@app.route("/home")
@login_required
def home():
    return render_template("home.html") 

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out", "info")
    return redirect(url_for('login'))

@app.route('/addstudent')
def to_addStudent():
    return render_template('addStudent.html')

@app.route('/markattendance')
def to_markAttendance():
    try:
        records_to_delete = AttendanceRecord.query.all()

        # Delete each record
        for record in records_to_delete:
            db.session.delete(record)

        # Commit the changes to the database
        db.session.commit()

        # Query all students from the Student model
        students = Student.query.all()
    except Exception as e:
        print(f"Error fetching student data: {e}")
        return []
    return render_template('markAttendance.html',students=students)

def recognize():
    size = 4
    datasets='trained_faces'
    (images, labels) = ([], [])
    id_to_name = {}
    model = cv2.face.LBPHFaceRecognizer_create()

    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            subject_path = os.path.join(datasets, subdir)
            id_to_name[len(id_to_name)] = subdir
            for filename in os.listdir(subject_path):
                path = os.path.join(subject_path, filename)
                label = len(id_to_name) - 1
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images.append(image)
                    labels.append(label)

    model.train(images, np.array(labels))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    webcam = cv2.VideoCapture(1,)
    width = 100
    height = 100

    for frame in generate_frames():
        ret, im = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            
            label, confidence = model.predict(face_resize)
            if confidence < 90:
                recognized_name = id_to_name[label]
                cv2.putText(im, recognized_name, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            else:
                cv2.putText(im, 'Not Recognized', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        _, jpeg = cv2.imencode('.jpg', im)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(1)
    webcam.release()
    cv2.destroyAllWindows()

@app.route('/video_recog')
def video_recog():
    return Response(recognize(), mimetype='multipart/x-mixed-replace; boundary=frame')

#This route calls the generate_frames function continuously, for capturing 50 images
@app.route('/video_get')
def video_get():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    datasets = 'trained_faces'  
    sub_data = 'Shreyash Chilip'     
    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)
    (width, height) = (130, 100)    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use the default camera
    count = 1

    while count <= 50:  # Adjust the number of frames as needed
        _, im = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite(os.path.join(path, f'{count}.png'), face_resize)

        _, jpeg = cv2.imencode('.jpg', im)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        count += 1

    webcam.release()
    cv2.destroyAllWindows()

@app.route('/process_video_frames', methods=['POST'])
def process_video_frames():
    try:
        image_data_bytes = request.data

        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data_bytes, np.uint8)
        # Decode numpy array into an image
        image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Check if the image is empty
        if image_cv2 is not None and not np.all(image_cv2 == 0):
            # Convert cv2 image to RGB
            image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
            # Continue with face recognition or other processing
            image_pil.save('image.jpeg')
            recognize_face(process_image(image_pil))
            return 'Success'
        else:
            print("Error: Unable to decode the image.")
            return 'Error: Unable to decode the image.'

    except Exception as e:
        print(f"Unexpected error: {e}")
        return 'Unexpected error occurred.'

def process_image(image):
    # Resize the image to a smaller size (150x150) for faster face detection
    image = image.resize((1280, 1024))
    # Convert the image to RGB format
    image = image.convert("RGB")
    image.save('image1024.jpeg')
    return image

def recognize_face(image,confidence_threshold=0.6):
    known_face_encodings = joblib.load('known_face_encodings.joblib')
    try:
        known_face_names = joblib.load('known_face_names.joblib')
    except (Exception) as e:
        print(f"Exception!: {e}")
        known_face_names=[]
    image_np = np.array(image)

    # Find faces in the frame
    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)

    if not face_encodings:
        print("No faces found in the image.")
        # return recognized_students
        return
    # recognized_students = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face with known faces
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        # Find the index with the smallest distance (most likely match)
        best_match_index = np.argmin(face_distances)
        min_distance = face_distances[best_match_index]

        roll_no = "Unknown"
  
          # Check if the smallest distance is below the confidence threshold
        if min_distance <=confidence_threshold:
            roll_no = known_face_names[best_match_index]
            student = Student.query.filter_by(roll_no=roll_no).first()

            if student:
                print(f"Name: {student.name}, Enrollment No: {student.enrollment_no}, Confidence: {1 - min_distance}")
                existing_record = AttendanceRecord.query.filter_by(date=datetime.now().date(), roll_no=roll_no).first()

                if not existing_record:
                    # If record doesn't exist, add a new record
                    new_attendance_record = AttendanceRecord(date=datetime.now().date(), roll_no=roll_no, name=student.name,enrollment_no=student.enrollment_no, present='y')
                    db.session.add(new_attendance_record)
                    db.session.commit()
                    print("New attendance record added.")
                else:
                    print("Attendance record already exists.")
            else: 
                print("Student not found in the database.")
            print("Found: " + roll_no)
        else:
            print(f"Face not confidently recognized (Distance: {min_distance}), Confidence Threshold: {confidence_threshold}")
            print(roll_no)
    # return recognized_students

@app.route('/load_lectures', methods=['POST'])
def load_lectures():
    selectedDay = request.form.get('selectedDay')
    selectedBatch = request.form.get('selectedBatch')
    current_batch = '1'
    if selectedBatch == 'IF1':
        current_batch = '1'
    elif selectedBatch == 'IF2':
        current_batch = '2'
    elif selectedBatch == 'IF3':
        current_batch = '3'
    lectures = TimeTable.query.filter_by(day=selectedDay, batch=current_batch).all()
    lectures_filtered = []  # Initialize an empty list

    for lecture in lectures:
        filtered_lecture = {}
        for key in ['batch', 'day', 'slot1', 'slot2', 'slot3', 'slot4', 'slot5', 'slot6', 'slot7', 'slot8']:
            if getattr(lecture, key) != '-':
                filtered_lecture[key] = getattr(lecture, key)
        lectures_filtered.append(filtered_lecture)
    final_lecture_dict=[]
    for lecture in lectures_filtered:
        for key in lecture:
            lecture_dict = {
                'value':key,
                'text':lecture[key],
            }
            final_lecture_dict.append(lecture_dict)

    # Return the result as a JSON response
    return jsonify(final_lecture_dict)

@app.route("/showAttendance", methods = ['POST'])
@login_required
def showAttendance():
    try:    
        
        attendanceRecord = AttendanceRecord.query.all()
        # result = AttendanceRecord.query.with_entities(AttendanceRecord.date, AttendanceRecord.subject, AttendanceRecord.batch).first()
    except Exception as e:
        print(f"Error fetching attendance data: {e}")
        return []
    return render_template('showAttendance.html',attendance=attendanceRecord)

if __name__ == "__main__":
    app.run(debug=True)