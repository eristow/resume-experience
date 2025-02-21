from flask import Flask

app = Flask(__name__)

import llm_api.views
