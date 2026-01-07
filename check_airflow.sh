#!/bin/bash

echo "=== Airflow Status Check ==="
echo ""

# Check if port 8080 is listening
echo "1. Checking if port 8080 is accessible..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080 | grep -q "302\|200"; then
    echo "   ✓ Port 8080 is responding"
else
    echo "   ✗ Port 8080 is not responding"
    exit 1
fi

echo ""
echo "2. Testing Airflow login page..."
LOGIN_RESPONSE=$(curl -s http://localhost:8080/login/ | grep -i "login\|username" | head -1)
if [ ! -z "$LOGIN_RESPONSE" ]; then
    echo "   ✓ Login page is accessible"
else
    echo "   ✗ Login page not found"
fi

echo ""
echo "3. Access Information:"
echo "   URL: http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "4. If you're still having issues:"
echo "   - Make sure you're accessing http://localhost:8080 (not https)"
echo "   - Try clearing your browser cache"
echo "   - Check browser console for errors (F12)"
echo "   - Try accessing from a different browser or incognito mode"
echo ""
echo "5. To restart Airflow services:"
echo "   docker-compose restart airflow-webserver"
echo ""

