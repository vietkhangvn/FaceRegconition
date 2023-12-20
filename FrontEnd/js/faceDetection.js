// Example faceDetection.js

// Function to handle face detection and validation
async function detectAndValidateFace() {
    const videoElement = document.getElementById('video');
    const resultElement = document.getElementById('result');
    try {
        // Access the webcam
        const videoElement = document.getElementById('video');
        const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
        videoElement.srcObject = stream;

        // Load face-api.js models
        await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
        await faceapi.nets.faceLandmark68Net.loadFromUri('/models');

        // Start face detection
        videoElement.addEventListener('play', async () => {
        setInterval(async () => {
            const detections = await faceapi.detectAllFaces(videoElement, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
            if (detections && detections.length > 0) {
                // Process the detected faces
                detections.forEach(detection => {
                    const box = detection.detection.box;

                    // Crop the detected face using the bounding box coordinates
                    const faceCanvas = document.createElement('canvas');
                    const faceContext = faceCanvas.getContext('2d');
                    faceCanvas.width = box.width;
                    faceCanvas.height = box.height;
                    faceContext.drawImage(videoElement, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);

                    // Get the cropped face image as a base64 data URL
                    const croppedFaceDataURL = faceCanvas.toDataURL('image/jpeg');

                    // Pass the cropped image data to the function that sends it to the backend
                    sendFaceToBackend(croppedFaceDataURL);
                });
            }
            }, 1000); // Change the interval for detection updates (in milliseconds)
        });
    } catch (error) {
        console.error('Error in face detection:', error);
        resultElement.innerText = `Error: ${error.message}`; // Display error message on the screen
    }
}

// Function to send the detected face to the backend for validation
async function sendFaceToBackend(faceImageData) {
    // Convert the face image data to a blob
    const faceBlob = await (await fetch(faceImageData)).blob();

    // Send the face blob to the backend for validation
    const formData = new FormData();
    formData.append('faceImage', faceBlob);

    try {
        const response = await fetch('/validateFace', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        // Handle the response from the backend (e.g., display recognized person)
        document.getElementById('result').innerText = `Recognized Person: ${data.recognized_person}`;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerText = `Error: ${error.message}`; // Display error message on the screen
    }
}
