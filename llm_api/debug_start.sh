#!/bin/bash

echo "Starting nginx..."
nginx -c /etc/nginx/nginx.conf -g "daemon off;" &
NGINX_PID=$!

echo "Starting uwsgi..."
uwsgi --ini /etc/uwsgi/uwsgi.ini --logger file:/tmp/uwsgi-verbose.log

# If uwsgi exits, kill nginx
kill $NGINX_PID
