#!/bin/bash
# Quick deployment script for local testing

echo "🚀 ODIR-5K Web Interface Deployment"
echo "===================================="
echo ""

# Check if model exists
if [ ! -f "models/best_model.pth" ]; then
    echo "❌ Error: Model file not found at models/best_model.pth"
    echo "Please ensure you have trained the model first."
    exit 1
fi

echo "✅ Model file found"
echo ""

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask dependencies..."
    pip install flask flask-cors
else
    echo "✅ Flask already installed"
fi

echo ""
echo "🎯 Starting deployment..."
echo ""
echo "Two options available:"
echo ""
echo "1️⃣  LOCAL TESTING (Backend + Frontend)"
echo "   - Backend API: http://localhost:5000"
echo "   - Frontend UI: http://localhost:8000"
echo ""
echo "2️⃣  GITHUB PAGES (Demo Mode - No Backend)"
echo "   - Push web/ folder to GitHub"
echo "   - Enable GitHub Pages in repo settings"
echo "   - Uses simulated predictions for demo"
echo ""

read -p "Start local testing now? (y/n): " choice

if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo ""
    echo "🔧 Starting services..."
    echo ""
    echo "Terminal 1: Starting Backend API..."
    
    # Start API in background
    python api.py &
    API_PID=$!
    
    sleep 3
    
    echo "Terminal 2: Starting Frontend Server..."
    cd web
    python -m http.server 8000 &
    WEB_PID=$!
    
    sleep 2
    
    echo ""
    echo "✅ Services started!"
    echo ""
    echo "📊 Access the application:"
    echo "   Frontend: http://localhost:8000"
    echo "   Backend API: http://localhost:5000"
    echo "   API Health: http://localhost:5000/health"
    echo ""
    echo "Press Ctrl+C to stop both services"
    echo ""
    
    # Wait for Ctrl+C
    trap "kill $API_PID $WEB_PID 2>/dev/null; echo ''; echo '👋 Services stopped!'; exit 0" INT
    
    # Open browser (macOS)
    if command -v open &> /dev/null; then
        echo "🌐 Opening browser..."
        sleep 1
        open http://localhost:8000
    fi
    
    wait
else
    echo ""
    echo "📝 To start services manually:"
    echo ""
    echo "Terminal 1 - Backend:"
    echo "  python api.py"
    echo ""
    echo "Terminal 2 - Frontend:"
    echo "  cd web && python -m http.server 8000"
    echo ""
    echo "Then visit: http://localhost:8000"
fi
