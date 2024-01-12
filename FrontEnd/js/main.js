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
const detectFacesButton = document.getElementById('detectFacesButton');
const registerResult = document.getElementById('registerResult');
const detectedName = document.getElementById('detectedName');
const detectedID = document.getElementById('detectedID');
const registerName = document.getElementById('registerName');
const registerID = document.getElementById('registerID');

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

    //Get the ID
    if (registerID.value == null || registerID.value == ""){
        registerResult.innerHTML = "Error: ID is empty!"
        return
    }
    //Get the Name
    if (registerName.value == null || registerName.value == ""){
        registerResult.innerHTML = "Error: Name is empty!"
        return
    }
    else{
        // Call function to send the captured image to the backend
        console.log('Sending image to backend...')
        sendImageToBackend(imageDataURL, registerName.value, registerID.value);
    }

});

// Capture image when the button is clicked, then send to back end to detect the face and return the result on the screen
detectFacesButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    //Reset the text fields on the screen
    detectedName.value = '';
    detectResult.textContent = 'Detecting face...';
    detectResult.style.color = '';

    // Draw current frame from the video onto the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get the image data as a base64-encoded JPEG
    const imageDataURL = canvas.toDataURL('image/jpeg');
    // var detectResult = document.getElementById('detectResult');


    loadConfig() // Load the configuration
        .then(config => {
            const backendURL = config.backendURL ; // Extract backend host from config

            //Set up the API body
            const apiData = {
                image: imageDataURL,
                //param3: textbox3Value
            };
            //console.log('API data:', apiData)

            fetch(`${backendURL}/detectFaces`, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                //body: formData
                body: JSON.stringify(apiData), // Send as JSON with 'image' key
            })
                .then(response => response.json())
                .then(data => {
                    // Handle the response from the backend
                    if(data.recognized_name){
                        detectedName.value = data.recognized_name;
                        detectedID.value = data.recognized_id;
                    }

                    detectResult.textContent = `Result: ${data.message}`;
                    detectResult.style.color = "green";
                })
                .catch(error => {
                    console.error('Error:', error);
                    detectResult.textContent = `Error: ${error.message}`;
                    detectResult.style.color = "red";
                });
        });

})

// Function to send captured image to the backend
function sendImageToBackend(imageDataURL, registerName, registerID) {
    // Convert image data URL to a blob
    const imageBlob = dataURLtoBlob(imageDataURL);
    //console.log('imageBlob: ', imageBlob)

    var registerResult = document.getElementById('registerResult');

    // Create a FormData object to send the image
    const formData = new FormData();
    formData.append('image', imageBlob);

    loadConfig() // Load the configuration
        .then(config => {
            const backendURL = config.backendURL ; // Extract backend host from config

            // Send the image to the backend (replace '/processImage' with your endpoint)
            //console.log('Backend host: ', backendURL)
            //console.log('formData: ', formData)

            //Set up the API body
            const apiData = {
                id: registerID,
                name: registerName,
                image: imageDataURL
                //param3: textbox3Value
            };
            console.log('API data:', apiData)

            fetch(`${backendURL}/registerFace`, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                //body: formData
                body: JSON.stringify(apiData), // Send as JSON with 'image' key
            })
                .then(response => response.json())
                .then(data => {
                    // Handle the response from the backend
                    if(data.detected_face){
                        detectResult.value = data.detected_face;
                    }

                    registerResult.textContent = `Result: ${data.message}`;
                    registerResult.style.color = "green";
                })
                .catch(error => {
                    console.error('Error:', error);
                    registerResult.textContent = `Error: ${error.message}`;
                    registerResult.style.color = "red";
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
});

// Function to start face detection
async function startDetection() {
    // Your code to start the face detection process
    // This function can call another function from faceDetection.js or contain its logic
    // Example: Initialize face detection process or start webcam
}
