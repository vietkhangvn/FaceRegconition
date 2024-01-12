import base64
import datetime
import json
import re
import time

import cv2
import dlib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import os
import imghdr
import io
from PIL import Image

import configparser
import psycopg2

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define CORS options
cors = CORS(app, resources={r"/processImage": {"origins": "http://localhost:5000"}})

# Load configuration from the file
config = configparser.ConfigParser()
config.read('config.ini')

# Access PostgreSQL credentials
postgresql_config = config['postgresql']
host = postgresql_config['host']
port = postgresql_config['port']
dbname = postgresql_config['dbname']
user = postgresql_config['user']
password = postgresql_config['password']

#  Establish database connection
conn = psycopg2.connect(
    dbname=dbname,
    user=user,
    password=password,
    host=host,
    port=port
)
cursor = conn.cursor()


# Function to detect the Face based on received image
def detect_faces(image_bytes):
    # Load image from byte array
    image = cv2.imdecode(np.fromstring(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Convert to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load Dlib detector
    detector = dlib.get_frontal_face_detector()

    # Detect faces
    faces = detector(image_rgb)

    # Return list of face locations
    return [(face.left(), face.top(), face.right(), face.bottom()) for face in faces]
    pass


def extract_features(image_bytes, face_location):
    # Load image from byte array
    image = cv2.imdecode(np.fromstring(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Convert to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract face region based on location
    x, y, w, h = face_location
    face_region = image_rgb[y:y + h, x:x + w]

    # Create Dlib shape predictor and landmark detector
    # predictor = dlib.shape_predictor_5_points()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()

    # Detect landmarks on the face
    landmarks = predictor(face_region)

    # Extract feature vector from landmarks (68-point format)
    feature_vector = []
    for p in landmarks.parts():
        feature_vector.append(p.x)
        feature_vector.append(p.y)

    return feature_vector
    pass


def compare_features(features):
    # Load known features from database
    # mock up
    known_features = {
        "John": [123, 456, 789],
        "Jane": [345, 678, 901],
    }

    # Define threshold for identification (higher threshold for stricter matching)
    threshold = 0.6

    identified_faces = []
    for feature in features:
        min_distance = float('inf')
        closest_face_id = None

        # Loop through known faces and find the closest match
        for face_id, known_feature in known_features.items():
            distance = np.linalg.norm(feature - known_feature)
            if distance < min_distance:
                min_distance = distance
                closest_face_id = face_id

        # Check if the distance is within the tolerance threshold for identification
        if min_distance < threshold:
            identified_faces.append(closest_face_id)

    # Return the list of identified faces, or None if no match found
    return identified_faces if identified_faces else None
    pass





def recognize_faces_old(image_bytes):
    name = "Unknown"
    known_face_encodings, known_face_names = load_known_faces()
    # Load image from byte array
    image = cv2.imdecode(np.fromstring(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Convert to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the uploaded image
    # image = face_recognition.load_image_file(image_path)

    # Find all face locations in the image
    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Loop through each face in this frame
    for face_encoding in face_encodings:
        # See if the face matches any known faces
        tolerance = 0.6
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance)

        # If a match is found, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

    return name


# Return: person_id
def recognize_faces_new(image_bytes):
    print('Recognizing face...')
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Find face encodings in the new image
        face_encodings = face_recognition.face_encodings(image_np, None, 1, "large")

        if face_encodings:
            print('Face detected in received image!')
            # Take the first detected face encoding
            new_face_encoding = face_encodings[0]

            # Retrieve known face encodings from the database
            cursor.execute("SELECT person_id, face_encoding FROM face_encodings")
            known_encodings = cursor.fetchall()

            # print('  Start comparing with DB!')
            similarity_scores = {}
            for person_id, encoding_json in known_encodings:
                # print('person id: ' + person_id)
                # print('encoding: ' + str(encoding_json))
                known_encoding = np.array(json.loads(str(encoding_json)))

                # Calculate similarity score
                similarity = face_recognition.face_distance([known_encoding], new_face_encoding)[0]
                # print('Similarity: ' + str(similarity))

                # Store similarity score with person_id
                similarity_scores[person_id] = similarity

            if similarity_scores:
                # Find the person_id with the lowest similarity score
                most_likely_person_id = min(similarity_scores, key=similarity_scores.get)
                print('Accuracy: ', f"{1 - similarity_scores[most_likely_person_id]:.0%}")
                if similarity_scores[most_likely_person_id] < 0.4:
                    return most_likely_person_id
            else:
                print("No match found in the database.")
                return None
        else:
            print("No face found in the provided image.")
            return None

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error comparing face encodings:", error)
        return None


@app.route('/registerFace', methods=['POST'])
def register_face():
    data = request.get_json()
    image_data = data['image']
    person_name = data['name']
    person_id = data['id']

    # Decode base64 image data
    # image_bytes = io.BytesIO(image_data.split(',')[1].encode('utf-8'))
    recognize_name = "Unknown"

    if image_data:
        # Check if person ID exists in Persons table
        cursor.execute("SELECT * FROM persons WHERE person_id = %s", (person_id,))
        person_exists = cursor.fetchone()

        if not person_exists:
            # If person ID doesn't exist, create a new entry in Persons table
            cursor.execute("INSERT INTO persons (person_id, person_name) VALUES (%s, %s)", (person_id, person_name))
            conn.commit()
            print(f"Person ID {person_id} created in Persons table.")

        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            # image_format = imghdr.what(None, h=image_data)

            image_format = 'jpg'
            if image_format:
                # Detect the face in the image - If face found: save file and save encodings into DB for further process
                # - If no face found: stop processing and return the image: no face found in the image, try to capture
                #   again.

                # known_image = face_recognition.load_image_file(filename)
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image_bytes))

                # Convert PIL Image to numpy array
                image_np = np.array(image)

                face_encodings = face_recognition.face_encodings(image_np, None,100, "large")
                if face_encodings:
                    # Take the first detected face encoding -------> FOR LATER ENHANCEMENT
                    # new_face_encoding = face_encodings[0]

                    # Save the image
                    folder_path = f"Captured/{person_name}"
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    filename = f"{folder_path}/{person_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"

                    for face_encoding in face_encodings:
                        # Convert face encoding numpy array to bytes
                        encoding_bytes = face_encoding.tobytes()

                        # Convert face encoding to JSON format
                        encoding_json = json.dumps(face_encoding.tolist())
                        # print(encoding_json)

                        # Insert face encoding into the database
                        cursor.execute(
                            "INSERT INTO face_encodings (person_id, face_encoding, file_location) VALUES (%s, %s, %s)",
                            (person_id, encoding_json, filename)
                        )
                    conn.commit()

                    # with open(os.path.join(app.config['UPLOAD_FOLDER'], file_name), 'wb') as f:
                    with open(filename, 'wb') as f:
                        f.write(image_bytes)

                    print("Face encodings saved to the database successfully.")
                else:
                    return jsonify({'message': 'No face detected in the image!'})

        except (Exception, psycopg2.DatabaseError) as error:
            conn.rollback()
            print("Error saving face encodings:", error)
            return jsonify({'message': 'Error saving face encodings!'})

            # recognize_name = recognize_faces_old(image_bytes)

            # Extract features and compare with database
            # face_locations = detect_faces(image_bytes)
            # for face_location in face_locations:
            #     features = extract_features(image_bytes, face_location)
            #     identified_person = compare_features(features)
            #
            #     # Update response message with identified person
            #     if identified_person:
            #         return {'message': f"Image captured and person identified as {identified_person}"}
            #     else:
            #         return {'message': "Image captured but person could not be identified"}
    else:
        # print('Received image is corrupted!')
        return jsonify({'message': 'Received image is corrupted!'})

    # Return a response message
    return jsonify({'message': 'Image captured and processed successfully!'
                    # 'detected_face': recognize_name
                    })


# API endpoint for recognizing faces in an image
@app.route('/detectFaces', methods=['POST'])
def detect_faces():
    data = request.get_json()
    image_data = data['image']

    if image_data:
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            # recognized_faces = recognize_faces_old(image_bytes)
            recognized_person_id = recognize_faces_new(image_bytes)
            recognized_name = 'Unknown'
            if recognized_person_id:
                # Retrieve name and additional details for the most likely person
                cursor.execute("SELECT person_name FROM persons WHERE person_id = %s", (recognized_person_id,))
                person_details = cursor.fetchone()

                if person_details:
                    recognized_name = person_details[0]  # Assuming the name is in the first column in the result
                else:
                    return jsonify({'error': 'Face detected but unable to retrieve the person details!'})
            else:
                recognized_person_id = 'Unknown'
            # -------------------------------------------------------------------------
            return jsonify({'message': 'Image captured and processed successfully!',
                            'recognized_name': recognized_name,
                            'recognized_id': recognized_person_id})
        except (Exception, psycopg2.DatabaseError) as error:
            return jsonify({'error': f'Error recognizing faces: {error}'})
    else:
        return jsonify({'error': 'No image provided'})


if __name__ == '__main__':
    app.run(debug=True)
