#!/bin/bash

echo "======================================"
echo "🎬 Starting CineGrade AI SaaS Platform"
echo "======================================"

# Kill existing instances if any
pkill -f "uvicorn api:app"
pkill -f "npm run dev"

# Start FastAPI Backend
echo "Starting Backend API on port 8000..."
source venv/bin/activate
nohup uvicorn api:app --host 127.0.0.1 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend running (PID: $BACKEND_PID)"

# Start Vite React Frontend
echo "Starting React Frontend on port 5173..."
cd frontend
nohup npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend running (PID: $FRONTEND_PID)"
cd ..

echo "======================================"
echo "✅ All systems go!"
echo "👉 Open your browser to: http://localhost:5173"
echo "To stop the servers later, run: kill $BACKEND_PID $FRONTEND_PID"
echo "======================================"
