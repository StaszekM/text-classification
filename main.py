import os
from dotenv import load_dotenv
from endpoints.endpoints import app

load_dotenv()

if __name__ == '__main__':
    app_port = os.getenv("APP_PORT")
    if app_port is None:
        raise RuntimeError('Could not find APP_PORT environment variable')
    app.run(port=app_port)
