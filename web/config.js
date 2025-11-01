// API Configuration
// Update this file with your deployed API URL

const CONFIG = {
    // For local development:
    // API_URL: 'http://localhost:5001/predict',
    
    // For production (update with your deployed API URL):
    // Railway: https://your-app.up.railway.app/predict
    // Render: https://your-app.onrender.com/predict
    // Heroku: https://your-app.herokuapp.com/predict
    
    API_URL: 'http://localhost:5001/predict',  // Change this after deploying API
    
    // Confidence threshold (0-1)
    THRESHOLD: 0.5,
    
    // Enable demo mode (for GitHub Pages without backend)
    DEMO_MODE: window.location.hostname.includes('github.io'),
};
