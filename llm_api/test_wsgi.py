#!/usr/bin/env python3

import sys

sys.path.insert(0, "/llm_api")

try:
    from wsgi import app

    print("Successfully imported app from wsgi module")
except Exception as e:
    print(f"Failed to import app: {e}")
    raise

try:
    with app.test_client() as c:
        response = c.get("/")
        print(f"Test client response: {response.status_code} - {response.data}")
except Exception as e:
    print(f"Failed to test app: {e}")
    raise
