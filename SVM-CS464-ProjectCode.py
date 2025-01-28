import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# paths for dataset directories
data_dir = r"C:\\Desktop\\CS 464 Project Code\\archive"
train_dir = os.path.join(data_dir, 'Train', 'Train')
validation_dir = os.path.join(data_dir, 'Validation', 'Validation')
test_dir = os.path.join(data_dir, 'Test', 'Test')

# image size for resizing
IMAGE_SIZE = (128, 128)

# load images and extract labels
def load_images_and_labels(directory):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith('.jpg'):
                    img_path = os.path.join(label_dir, file)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.resize(img, IMAGE_SIZE)
                        images.append(img)
                        labels.append(label)
    return np.array(images), np.array(labels)

# extract HOG features
def extract_hog_features(images):
    hog_features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        feature = hog(gray, 
                      orientations=9, 
                      pixels_per_cell=(8, 8), 
                      cells_per_block=(2, 2))
        hog_features.append(feature)
    return np.array(hog_features)

# extract color histograms
def extract_color_histograms(images, bins=(8, 8, 8)):
    hist_features = []
    for img in images:
        hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()  
        hist_features.append(hist)
    return np.array(hist_features)

# Load data
print("Loading training data...")
X_train, y_train = load_images_and_labels(train_dir)
print("Loading validation data...")
X_val, y_val = load_images_and_labels(validation_dir)
print("Loading test data...")
X_test, y_test = load_images_and_labels(test_dir)

# Extract HOG and color histogram features
print("Extracting HOG features...")
X_train_hog = extract_hog_features(X_train)
X_val_hog = extract_hog_features(X_val)
X_test_hog = extract_hog_features(X_test)

print("Extracting color histograms...")
X_train_hist = extract_color_histograms(X_train)
X_val_hist = extract_color_histograms(X_val)
X_test_hist = extract_color_histograms(X_test)

# Concatenate HOG and color histogram features
X_train_combined = np.hstack((X_train_hog, X_train_hist))
X_val_combined = np.hstack((X_val_hog, X_val_hist))
X_test_combined = np.hstack((X_test_hog, X_test_hist))

# Normalize combined features
scaler = StandardScaler()
X_train_combined = scaler.fit_transform(X_train_combined)
X_val_combined = scaler.transform(X_val_combined)
X_test_combined = scaler.transform(X_test_combined)

# Encode labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)

# define parameter grid for GridSearchCV
# Important: Following parameters are based on the results of the previous GridSearchCV, which are best parameters
param_grid = {
    'kernel': ['rbf'],  
    'C': [10],          
    'gamma': ['scale']  
}

#Alternatively, following is the tried parameter grid but it
#should be noted that takes time to run
'''
param_grid = {
    'kernel': ['rbf', 'linear', 'poly'],
    'C': [0.01, 0.1, 1, 10, 100],  
    'degree': [2, 3, 4],           
    'gamma': [0.001, 0.01, 0.1, 1, 'scale']  
}
'''

# find the best SVM model
print("\nPerforming GridSearchCV to find the best SVM model...")
svc = SVC(probability=True)
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train_combined, y_train_enc)

# best model
best_model = grid_search.best_estimator_
print("\nBest SVM parameters:", grid_search.best_params_)

# evaluate on validation data
print("\nEvaluating the best model on validation data...")
y_val_pred = best_model.predict(X_val_combined)
val_accuracy = accuracy_score(y_val_enc, y_val_pred)
val_f1 = f1_score(y_val_enc, y_val_pred, average='weighted')
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Weighted F1-Score: {val_f1:.4f}")
print("\nClassification Report for Validation Set:")
print(classification_report(y_val_enc, y_val_pred, target_names=label_encoder.classes_))

# evaluate on test data
print("\nEvaluating the best model on test data...")
y_test_pred = best_model.predict(X_test_combined)
test_accuracy = accuracy_score(y_test_enc, y_test_pred)
test_f1 = f1_score(y_test_enc, y_test_pred, average='weighted')
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Weighted F1-Score: {test_f1:.4f}")
print("\nClassification Report for Test Set:")
print(classification_report(y_test_enc, y_test_pred, target_names=label_encoder.classes_))