FROM python:3.12.2-slim

RUN apt update && \
    apt install -y nginx uwsgi uwsgi-plugin-python3 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m uwsgi

RUN mkdir -p /var/lib/nginx/tmp/client_body \
    /var/lib/nginx/tmp/proxy \
    /var/lib/nginx/tmp/fastcgi \
    /var/lib/nginx/tmp/uwsgi \
    /var/lib/nginx/tmp/scgi \
    /var/lib/nginx/cache \
    /var/lib/nginx \
    /var/run \
    /tmp/nginx && \
    chown -R uwsgi:uwsgi /var/lib/nginx /var/log/nginx /var/run /tmp/nginx && \
    chmod -R 755 /var/lib/nginx /var/log/nginx /var/run /tmp/nginx

COPY llm_api/non_root_nginx.conf /etc/nginx/nginx.conf
COPY llm_api/nginx.conf /etc/nginx/conf.d/default.conf

COPY llm_api/uwsgi.ini /etc/uwsgi

COPY llm_api/test_wsgi.py /tmp/test_wsgi.py

COPY llm_api/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

RUN mkdir -p /llm_api && \
    chown -R uwsgi:uwsgi /etc/nginx/conf.d /etc/uwsgi /usr/local/bin/start.sh /llm_api && \
    chmod 755 /usr/local/bin/start.sh

WORKDIR /llm_api

COPY llm_api/ /llm_api

RUN pip install --no-cache-dir -r requirements.txt

USER uwsgi

EXPOSE 80
EXPOSE 5000

CMD ["/usr/local/bin/start.sh"]
