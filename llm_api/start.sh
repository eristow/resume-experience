#!/bin/bash

exec > >(tee -a /tmp/startup.log) 2>&1
echo "=== Starting container at $(date) ==="

echo "Starting uWSGI..."
uwsgi --socket /tmp/uwsgi.sock \
	--chmod-socket=666 \
	--module wsgi:app \
	--master \
	--processes 2 \
	--threads 2 \
	--http :5000 \
	--logto /tmp/uwsgi.log &


UWSGI_PID=$!
echo "uWSGI started with PID $UWSGI_PID"

echo "Waiting for uWSGI socket to be created..."
for i in {1..30}; do
	if [ -S /tmp/uwsgi.sock ]; then
		echo "Socket /tmp/uwsgi.sock created successfully!"
		ls -la /tmp/uwsgi.sock
		break
	fi
	echo "Waiting for socket... ($i/30)"
	sleep 1
done

if [ ! -S /tmp/uwsgi.sock ]; then
	echo "ERROR: Socket file was not created after 30 seconds!"
	fcho "uWSGI process info:"
	ps aux | grep uwsgi
	echo "uWSGI log tail:"
	tail -n 50 /tmp/uwsgi.log || echo "No uwsgi.log found"
	exit 1
fi

echo "Starting Nginx..."
nginx -c /etc/nginx/nginx.conf -g "daemon off;" &
NGINX_PID=$!
echo "Nginx started with PID $NGINX_PID"

echo "Monitoring services..."
while true; do
	if ! kill -0 $UWSGI_PID 2>/dev/null; then
		echo "uWSGI process died! Check logs at /tmp/uwsgi.log"
		tail -n 50 /tmp/uwsgi.log || echo "No uwsgi.log found"
		exit 1
	fi

	if ! kill -0 $NGINX_PID 2> /dev/null; then
		echo "Nginx process died! Check logs at /tmp/nginx/error.log"
		tail -n 50 /tmp/nginx/error.log || echo "No nginx error.log found"
		exit 1
	fi

	sleep 5
done
