import base64
import datetime
import re

import cv2
import dlib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import psycopg2
import os
import imghdr
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define CORS options
cors = CORS(app, resources={r"/processImage": {"origins": "http://localhost:5000"}})


# Establish database connection
# conn = psycopg2.connect(
#     dbname="your_db_name",
#     user="your_username",
#     password="your_password",
#     host="your_host",
#     port="your_port"
# )
# cursor = conn.cursor()

# Endpoint to process the received image
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


def load_known_faces():
    print('Loading known faces...')
    # Load known faces from a folder
    # known_faces_dir = "input_database"
    known_faces_dir = "Captured"
    known_face_encodings = []
    known_face_names = []
    i = 0
    for root, dirs, files in os.walk(known_faces_dir):
        for filename in files:
            #  print(filename)
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(root, filename)
                known_image = face_recognition.load_image_file(image_path)

                res = re.findall(r"([A-Za-z]+)[_-]\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.jpg", filename)
                if not res:
                    continue
                #  print(' known name: ' + str(res[0]))

                face_encoding = face_recognition.face_encodings(known_image)[0]
                known_face_encodings.append(face_encoding)
                # known_face_names.append(os.path.splitext(filename)[0])
                known_face_names.append(res[0])
                i = i + 1
    print('Loaded ' + str(i) + ' known faces')
    return known_face_encodings, known_face_names


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
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # If a match is found, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

    return name


@app.route('/registerFace', methods=['POST'])
def register_face():
    data = request.get_json()
    image_data = data['image']
    image_name = data['name']

    # Decode base64 image data
    # image_bytes = io.BytesIO(image_data.split(',')[1].encode('utf-8'))
    recognize_name = "Unknown"

    if image_data:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        # image_format = imghdr.what(None, h=image_data)

        image_format = 'jpg'
        if image_format:
            # Save the image
            folder_path = f"Captured/{image_name}"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            filename = f"{folder_path}/{image_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            # with open(os.path.join(app.config['UPLOAD_FOLDER'], file_name), 'wb') as f:
            with open(filename, 'wb') as f:
                f.write(image_bytes)

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


@app.route('/processImageBlob', methods=['POST'])
def process_image_blob():
    if 'image' not in request.files:
        return 'Missing image file', 400
    try:
        # Access the image blob sent from the frontend
        # image_blob = request.get_data()
        # image_blob_binary = request.form['image']
        image_blob_binary = request.files['image'].read()

        image_bytes = io.BytesIO(image_blob_binary)
        # img = Image.open(image_bytes)

        # image_blob = image_blob_binary.decode('utf-8')
        # image_blob = base64.b64encode(image_blob_binary).decode('ascii')
        # image_blob = base64.decodebytes(image_blob_binary)
        image_blob = base64.b64decode(image_blob_binary, 'jpg')

        # Get the image data from the request
        # image_data = request.form['image']
        if image_blob:
            # image_data_raw = image_blob.split(',')[1]
            # image_data = base64.b64decode(image_blob)
            # Decode Base64 image data to bytes
            # image_bytes = base64.b64decode(image_data_raw)

            # Create an in-memory file-like object to work with PIL
            # img = Image.open(io.BytesIO(image_bytes))

            # Save the decoded image data as a JPEG file
            # img.save('uploaded_image2.jpg', 'JPEG')
            image_data = image_blob
            if image_data:
                # Process the image blob (Example: Save the blob to a file)
                # Check if the image blob is in JPEG format
                image_format = imghdr.what(None, h=image_data)
                # Create an in-memory file-like object to work with PIL
                # img = Image.open(io.BytesIO(image_data))

                # Save the decoded image data as a JPEG file
                # img.save('uploaded_image2.jpg', 'JPEG')

                # image_format = 'jpg'
                if image_format:
                    print('image format: ' + image_format)
                    file_name = 'image.' + image_format
                    with open(file_name, 'wb') as f:
                        f.write(image_blob_binary)

                    # Recognize face based on input image

                else:
                    print('Received image is corrupted!')

                # Perform further processing on the image blob here
                # ...

                # Return a response indicating successful processing
                return jsonify({'message': 'Image blob processed successfully'})
        else:
            print('No image received!')
            # If no image blob received, return an error response
            return jsonify({'error': 'No image blob received'}), 400  # 400 for Bad Request

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # 500 for Internal Server Error
        # return jsonify({'error': 'Invalid request data'}), 400
    pass


# API endpoint to add an image for a person
@app.route('/add_image', methods=['POST'])
def add_image():
    person_id = request.form['person_id']
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'})

    if image:
        try:
            # Save the image temporarily
            image_path = 'temp.jpg'
            image.save(image_path)

            # Encode the image
            known_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(known_image)[0]

            # Insert face encoding into the database
            cursor.execute("INSERT INTO Encodings (person_id, encoding) VALUES (%s, %s)",
                           (person_id, face_encoding.tobytes()))
            conn.commit()

            # Delete the temporary image file
            os.remove(image_path)

            return jsonify({'success': 'Image added successfully'})
        except (Exception, psycopg2.DatabaseError) as error:
            conn.rollback()
            return jsonify({'error': f'Error adding image: {error}'})


# API endpoint for recognizing faces in an image
@app.route('/detectFaces', methods=['POST'])
def detect_faces():
    data = request.get_json()
    image_data = data['image']

    if image_data:
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            recognized_faces = recognize_faces_old(image_bytes)
            # Old processing for reference when using DB ------------------------------
            # Save the image temporarily
            # image_path = 'temp.jpg'
            # image.save(image_path)

            # Load the image and recognize faces
            # unknown_image = face_recognition.load_image_file(image_path)
            # unknown_face_encodings = face_recognition.face_encodings(unknown_image)

            # Retrieve known face encodings from the database
            # cursor.execute("SELECT person_id, encoding FROM Encodings")
            # known_encodings = cursor.fetchall()

            # recognized_faces = []
            # for unknown_encoding in unknown_face_encodings:
            #     for person_id, encoding_bytes in known_encodings:
            #         known_encoding = face_recognition.face_encodings(bytes(encoding_bytes))
            #         results = face_recognition.compare_faces(known_encoding, unknown_encoding)
            #         if True in results:
            #             recognized_faces.append({'person_id': person_id})

            # Delete the temporary image file
            # os.remove(image_path)
            # -------------------------------------------------------------------------
            return jsonify({'message': 'Image captured and processed successfully!',
                            'recognized_faces': recognized_faces})
        except (Exception, psycopg2.DatabaseError) as error:
            return jsonify({'error': f'Error recognizing faces: {error}'})
    else:
        return jsonify({'error': 'No image provided'})


# API endpoint to retrieve recognized faces
@app.route('/get_recognized_faces', methods=['GET'])
def get_recognized_faces():
    try:
        # Retrieve recognized faces from the database
        cursor.execute("SELECT person_id FROM RecognizedFaces")
        recognized_faces = [{'person_id': row[0]} for row in cursor.fetchall()]
        return jsonify({'recognized_faces': recognized_faces})
    except (Exception, psycopg2.DatabaseError) as error:
        return jsonify({'error': f'Error retrieving recognized faces: {error}'})


@app.route('/validateFace', methods=['POST'])
def validate_face():
    try:
        # Get the face image from the request
        face_image = request.files['faceImage']

        # Perform face recognition or validation on the received face image
        # This is a placeholder for the actual face recognition/validation logic

        # For demonstration, assuming a simple validation by checking if the image exists
        if face_image:
            # If face validation passes, assume recognition of a person named 'John Doe'
            recognized_person = 'John Doe'
            return jsonify({'recognized_person': recognized_person})
        else:
            # If face validation fails
            return jsonify({'recognized_person': 'Unknown'})

    except Exception as e:
        # Handle exceptions or errors
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
