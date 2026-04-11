#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$DIR")"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Starting Loom...${NC}"

# 1. Ensure Python venv exists
if [ ! -d "$DIR/.venv" ]; then
  echo -e "${GREEN}Creating Python virtual environment...${NC}"
  python3 -m venv "$DIR/.venv"
  "$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"
fi

# 2. Ensure frontend deps installed
if [ ! -d "$DIR/frontend/node_modules" ]; then
  echo -e "${GREEN}Installing frontend dependencies...${NC}"
  (cd "$DIR/frontend" && npm install)
fi

# 3. Kill any existing processes on our ports
lsof -ti :8788 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti :3000 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

# 4. Start backend
echo -e "${GREEN}Starting backend on :8788...${NC}"
cd "$ROOT"
PYTHONPATH=. PYTHONUNBUFFERED=1 "$DIR/.venv/bin/uvicorn" loom.main:app \
  --host 0.0.0.0 --port 8788 &
BACKEND_PID=$!

# 5. Wait for backend to be ready
for i in $(seq 1 30); do
  if curl -s http://localhost:8788/health > /dev/null 2>&1; then
    echo -e "${GREEN}Backend ready.${NC}"
    break
  fi
  sleep 1
done

# 6. Start frontend
echo -e "${GREEN}Starting frontend on :3000...${NC}"
cd "$DIR/frontend"
npx vite --port 3000 &
FRONTEND_PID=$!

sleep 2
echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  Loom is running!${NC}"
echo -e "${CYAN}  Open:  http://localhost:3000${NC}"
echo -e "${CYAN}  API:   http://localhost:8788${NC}"
echo -e "${CYAN}  Press Ctrl+C to stop${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Loom stopped.'" EXIT

wait
