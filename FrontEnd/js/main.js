// Import dotenv and load environment variables
// require('dotenv').config();
//import config from './backendConfig.json';

// Load the JSON configuration file
//const config = require('./backendConfig.json');

// Access environment variables
//const backendURL = config.BACKEND_URL;

// Fetch the configuration data
function loadConfig() {
  return fetch('./backendConfig.json')
    .then(response => {
      //console.log('Fetch config response: ', response)
      if (!response.ok) {
        throw new Error('Failed to fetch config');
      }
      return response.json();
    })
    .catch(error => {
      console.error('Error fetching config:', error);
      return null; // Return default or handle error as needed
    });
}

// Access elements from the DOM
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('captureButton');
const registerResult = document.getElementById('registerResult');
const detectResult = document.getElementById('detectedName');
const registerName = document.getElementById('registerName');

// Start webcam feed
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(error => {
        console.error('Error accessing the camera:', error);
    });

// Capture image when the button is clicked, then send to back end to save
captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame from the video onto the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get the image data as a base64-encoded JPEG
    const imageDataURL = canvas.toDataURL('image/jpeg');
    //console.log('imageDataURL: ', imageDataURL)

    //Get the Name
    if (registerName.value == null || registerName.value == ""){
        registerResult.innerHTML = "Error: Name is empty!"
    }
    else{
        // Call function to send the captured image to the backend
        console.log('Sending image to backend...')
        sendImageToBackend(imageDataURL);
    }

});

// Function to send captured image to the backend
function sendImageToBackend(imageDataURL) {
    // Convert image data URL to a blob
    const imageBlob = dataURLtoBlob(imageDataURL);
    //console.log('imageBlob: ', imageBlob)

    // Create a FormData object to send the image
    const formData = new FormData();
    formData.append('image', imageBlob);

    loadConfig() // Load the configuration
        .then(config => {
            const backendURL = config.backendURL ; // Extract backend host from config

            // Send the image to the backend (replace '/processImage' with your endpoint)
            //console.log('Backend host: ', backendURL)
            //console.log('formData: ', formData)
            fetch(`${backendURL}/processImage`, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                //body: formData
                body: JSON.stringify({ image: imageDataURL }), // Send as JSON with 'image' key
                //body: JSON.stringify({imageData}),
            })
                .then(response => response.json())
                .then(data => {
                    // Handle the response from the backend
                    var resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = '';
                    if(data.detected_face){
                        detectResult.value = data.detected_face;
                    }


                    resultElement.innerText = `Result: ${data.result}`;
                })
                .catch(error => {
                    console.error('Error:', error);
                    resultElement.innerText = `Error: ${error.message}`;
                });
        });
}

// Function to convert data URL to Blob
function dataURLtoBlob(dataURL) {
    const parts = dataURL.split(';base64,');
    const contentType = parts[0].split(':')[1];
    const raw = window.atob(parts[1]);
    const blob = new Blob([raw], { type: contentType });
    return blob;
}

// Add event listener to execute when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // Your initialization code can go here
    console.log('DOM fully loaded and parsed');

    // Example: Call a function when a button is clicked
    const startButton = document.getElementById('startButton');
    startButton.addEventListener('click', startDetection);
});

// Function to start face detection
async function startDetection() {
    // Your code to start the face detection process
    // This function can call another function from faceDetection.js or contain its logic
    // Example: Initialize face detection process or start webcam
}
