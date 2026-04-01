# UPDATED VERSION WITH IMPROVEMENTS (class weights, threshold tuning, FP/FN logging)

from flask import *
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib, joblib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
app.secret_key = 'your_secret_key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/ml_app'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# ---------------- CONFIG ----------------
DATASET_PATH = "dataset"
MODEL_PATH = "saved_model/svm_model.pkl"
ENCODER_PATH = "saved_model/label_encoder.pkl"
SCALER_PATH = "saved_model/scaler.pkl"
IMG_SIZE = 224

os.makedirs("saved_model", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/hard_samples", exist_ok=True)

# ---------------- FEATURE EXTRACTOR ----------------
feature_extractor = None

def get_feature_extractor():
    global feature_extractor
    if feature_extractor is None:
        feature_extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return feature_extractor

# ---------------- TRAIN ----------------
@app.route("/train")
def train():
    try:
        images, labels, paths = [], [], []

        for label in ["real", "fake"]:
            folder = os.path.join(DATASET_PATH, label)

            for img_name in os.listdir(folder):
                path = os.path.join(folder, img_name)
                img = cv2.imread(path)

                if img is None:
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = preprocess_input(img)

                images.append(img)
                labels.append(label)
                paths.append(path)

        images = np.array(images)

        extractor = get_feature_extractor()
        features = extractor.predict(images)
        features = features.reshape(features.shape[0], -1)

        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)

        X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
            features, labels, paths,
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 🔥 CLASS WEIGHT (important)
        svm = SVC(kernel="rbf", C=100, gamma=0.001, probability=True,
                  class_weight={0:1.5, 1:1.0})

        svm.fit(X_train, y_train)

        probs = svm.predict_proba(X_test)

        # 🔥 THRESHOLD TUNING
        threshold = 0.6
        preds = (probs[:,1] > threshold).astype(int)

        acc = accuracy_score(y_test, preds)

        # 🔥 FIND FP & FN
        fp_idx = np.where((y_test == 0) & (preds == 1))[0]
        fn_idx = np.where((y_test == 1) & (preds == 0))[0]

        # SAVE HARD SAMPLES
        for i in fp_idx:
            cv2.imwrite(f"static/hard_samples/fp_{i}.jpg", cv2.imread(p_test[i]))

        for i in fn_idx:
            cv2.imwrite(f"static/hard_samples/fn_{i}.jpg", cv2.imread(p_test[i]))

        # SAVE MODEL
        joblib.dump(svm, MODEL_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        joblib.dump(scaler, SCALER_PATH)

        # CONFUSION MATRIX
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
        disp.plot(cmap="Blues")

        plt.title("Confusion Matrix")
        plt.savefig("static/confusion_matrix.png")
        plt.close()

        with open("saved_model/accuracy.txt", "w") as f:
            f.write(str(round(acc * 100, 2)))

        flash(f"Training done! Accuracy: {round(acc*100,2)}%", "success")
        flash(f"FP: {len(fp_idx)} | FN: {len(fn_idx)} saved in /static/hard_samples", "info")

        return render_template("train.html", accuracy=round(acc*100,2))

    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("dashboard"))

# ---------------- PREDICT ----------------
@app.route('/predict', methods=['GET','POST'])
def predict():

    prediction=None
    image_path=None

    if request.method=='POST':

        if not os.path.exists(MODEL_PATH):
            flash("Train model first","danger")
            return redirect(url_for('train'))

        file=request.files.get("file")

        if not file:
            flash("No image","warning")
            return redirect(request.url)

        filename=file.filename
        save_path=os.path.join("static/uploads",filename)
        file.save(save_path)

        svm_model=joblib.load(MODEL_PATH)
        label_encoder=joblib.load(ENCODER_PATH)
        scaler=joblib.load(SCALER_PATH)

        img=cv2.imread(save_path)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img=preprocess_input(img)

        img=np.expand_dims(img,axis=0)

        extractor=get_feature_extractor()
        features=extractor.predict(img).flatten()
        features=scaler.transform([features])

        probs=svm_model.predict_proba(features)[0]

        # 🔥 SAME THRESHOLD
        threshold = 0.6
        pred = 1 if probs[1] > threshold else 0

        label=label_encoder.inverse_transform([pred])[0]
        confidence=probs[pred]

        prediction=f"{label.upper()} ({confidence*100:.2f}%)"
        image_path=save_path

    return render_template("predict.html",prediction=prediction,image_path=image_path)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False)
