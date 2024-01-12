import configparser
import json
import os
import re
import time

import face_recognition
import psycopg2

# Load configuration from the file
config = configparser.ConfigParser()
config.read('../BackEnd/config.ini')

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


def load_known_faces():
    print('Loading known faces...')
    start_time = time.time()
    # Load known faces from a folder
    # known_faces_dir = "input_database"
    known_faces_dir = "../Captured"
    known_face_encodings = []
    known_face_names = []
    i = 0
    for root, dirs, files in os.walk(known_faces_dir):
        for filename in files:
            print('i = ', i, filename)
            # if filename.endswith(".jpg") or filename.endswith(".png"):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp"):
                image_path = os.path.join(root, filename)
                known_image = face_recognition.load_image_file(image_path)

                res = re.match(r'^([^_]+)', filename).group(1)
                if not res:
                    continue
                #  print(' known name: ' + res)
                person_id = res
                person_name = res


                # Check if person ID exists in Persons table
                cursor.execute("SELECT * FROM persons WHERE person_id = %s", (person_id,))
                person_exists = cursor.fetchone()

                if not person_exists:
                    # If person ID doesn't exist, create a new entry in Persons table
                    cursor.execute("INSERT INTO persons (person_id, person_name) VALUES (%s, %s)",
                                   (person_id, person_name))
                    conn.commit()
                    print(f"Person ID {person_id} created in Persons table.")

                # face_encoding = face_recognition.face_encodings(known_image)[0]
                face_encodings = face_recognition.face_encodings(known_image, None, 50, "large")
                for face_encoding in face_encodings:
                    encoding_json = json.dumps(face_encoding.tolist())
                    # print(encoding_json)

                    # Insert face encoding into the database
                    cursor.execute(
                        "INSERT INTO face_encodings (person_id, face_encoding, file_location) VALUES (%s, %s, %s)",
                        (person_id, encoding_json, image_path)
                    )
                conn.commit()
                i = i + 1
    print('Loaded ' + str(i) + ' known faces in ' + str(time.time() - start_time) + ' seconds')
    return known_face_encodings, known_face_names


if __name__ == '__main__':
    #  print('Import completed')
    known_face_encodings, known_face_names = load_known_faces()
