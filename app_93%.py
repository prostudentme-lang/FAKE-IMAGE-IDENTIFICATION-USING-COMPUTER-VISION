from flask import *
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib,joblib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix, accuracy_score
from flask import Flask, render_template, request, redirect, url_for, flash

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import timm

app=Flask(__name__)
app.secret_key='your_secret_key'

app.config['SQLALCHEMY_DATABASE_URI']='mysql+pymysql://root:root@localhost/ml_app'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False

db= SQLAlchemy(app)

bcrypt= Bcrypt(app)

class User(db.Model):
    id= db.Column(db.Integer, primary_key=True)
    name= db.Column(db.String(100), unique=True, nullable=False)
    email= db.Column(db.String(100), unique=True, nullable=False)
    password= db.Column(db.String(100), nullable= False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register' , methods=['GET', 'POST'])
def register():
    if request.method=='POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(name=name, email=email, password=hashed_password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration Successful! Please Login.' ,'success')
            return redirect(url_for('login'))
        except:
            flash('Error: Username or E-mail Already Exists.','danger')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method=='POST':
        email = request.form['email']
        password= request.form['password']
        user=User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password,password):
            session['user_id']= user.id
            session['name']=user.name
            flash('Login Successfull','success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid E-mail or Password. Please Try Again...','danger')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please Login To Access The Dashboard.','warning')
        return redirect(url_for('login'))
    name=session['name']
    return render_template('dashboard.html',name=name)

@app.route('/logout',methods=['POST'])
def logout():
    session.pop('user_id',None)
    session.pop('username',None)
    flash('You Have Been Logged Out.','info')
    return redirect(url_for('login'))


DATASET_PATH = "dataset"
MODEL_PATH = "saved_model/svm_model.pkl"
ENCODER_PATH = "saved_model/label_encoder.pkl"
SCALER_PATH = "saved_model/scaler.pkl"

IMG_SIZE = 224

os.makedirs("saved_model", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
def preprocess_swin(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    return img

# ---------------- FEATURE EXTRACTOR ----------------

feature_extractor = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_extractor():
    global feature_extractor
    if feature_extractor is None:
        feature_extractor = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            num_classes=0
        )
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()
    return feature_extractor

@app.route("/train")
def train():
    try:

        # -------- CHECK DATASET --------
        if not os.path.exists(DATASET_PATH):
            flash("Dataset folder not found. Please upload dataset first.", "danger")
            return redirect(url_for("dashboard"))

        real_path = os.path.join(DATASET_PATH, "real")
        fake_path = os.path.join(DATASET_PATH, "fake")

        if not os.path.exists(real_path) or not os.path.exists(fake_path):
            flash("Dataset must contain 'real' and 'fake' folders.", "danger")
            return redirect(url_for("dashboard"))

        if len(os.listdir(real_path)) == 0 or len(os.listdir(fake_path)) == 0:
            flash("Dataset folders are empty.", "warning")
            return redirect(url_for("dashboard"))

        # -------- SKIP IF MODEL EXISTS --------
        if os.path.exists(MODEL_PATH):
            accuracy = None
            if os.path.exists("saved_model/accuracy.txt"):
                with open("saved_model/accuracy.txt", "r") as f:
                    accuracy = f.read()

            flash("Model already trained.", "info")
            return render_template("train.html", accuracy=accuracy)

        # -------- LOAD DATASET --------
        images = []
        labels = []

        for label in ["real", "fake"]:
            folder = os.path.join(DATASET_PATH, label)

            for img_name in os.listdir(folder):
                path = os.path.join(folder, img_name)

                img = cv2.imread(path)
                if img is None:
                    continue

                images.append(img)
                labels.append(label)

        print(f"Total images loaded: {len(images)}")

        # -------- FEATURE EXTRACTION --------
        batch_size = 16
        features = []
        extractor = get_feature_extractor()

        print("Extracting features using Swin-T...")

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]

            # ✅ FAST conversion (fixed warning)
            batch = np.array([preprocess_swin(img) for img in batch_imgs])
            batch = torch.from_numpy(batch).float().to(device)

            with torch.no_grad():
                feats = extractor(batch)

            features.extend(feats.cpu().numpy())

            print(f"Processed {min(i+batch_size, len(images))}/{len(images)} images")

        features = np.array(features)
        features = features.reshape(features.shape[0], -1)

        # -------- LABEL ENCODING --------
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)

        # -------- TRAIN TEST SPLIT --------
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        # -------- SCALING --------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # -------- TRAIN SVM --------
        svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
        print("Starting SVM training...")
        svm.fit(X_train, y_train)
        print("Training completed!")

        preds = svm.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # -------- SAVE MODEL --------
        joblib.dump(svm, MODEL_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        joblib.dump(scaler, SCALER_PATH)

        # -------- CONFUSION MATRIX --------
        cm = confusion_matrix(y_test, preds)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=encoder.classes_
        )
        disp.plot(cmap="Blues")

        plt.title("Confusion Matrix")

        cm_path = os.path.join("static", "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        print(f"Confusion matrix saved at {cm_path}")

        # -------- SAVE ACCURACY --------
        with open("saved_model/accuracy.txt", "w") as f:
            f.write(str(round(acc * 100, 2)))

        flash(f"Training successful! Accuracy: {round(acc*100,2)}%", "success")

        return render_template("train.html", accuracy=round(acc*100,2))

    except Exception as e:
        flash(f"Training failed: {e}", "danger")
        return redirect(url_for("dashboard"))
            
# ---------------- PREDICT ----------------

@app.route('/predict', methods=['GET','POST'])
def predict():

    prediction=None
    image_path=None

    if request.method=='POST':

        if not os.path.exists(MODEL_PATH):

            flash("Train the model first","danger")
            return redirect(url_for('train'))

        file=request.files.get("file")

        if not file:

            flash("No image selected","warning")
            return redirect(request.url)

        filename=file.filename
        save_path=os.path.join("static/uploads",filename)

        file.save(save_path)

        svm_model=joblib.load(MODEL_PATH)
        label_encoder=joblib.load(ENCODER_PATH)
        scaler=joblib.load(SCALER_PATH)

        img=cv2.imread(save_path)

        if img is None:

            flash("Invalid image","danger")
            return redirect(request.url)

        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        extractor=get_feature_extractor()
        img = preprocess_swin(img)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            features = extractor(img)
        features = features.cpu().numpy().flatten()


        


        features=scaler.transform([features])

        probs=svm_model.predict_proba(features)[0]

        pred=np.argmax(probs)
        confidence=probs[pred]

        label=label_encoder.inverse_transform([pred])[0]

        prediction=f"{label.upper()} ({confidence*100:.2f}%)"
        image_path=save_path

    return render_template("predict.html",prediction=prediction,image_path=image_path)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False, host='0.0.0.0',port=5000)
