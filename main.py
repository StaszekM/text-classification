import os
from dotenv import load_dotenv
from endpoints.endpoints import app
from gevent.pywsgi import WSGIServer

load_dotenv()

if __name__ == '__main__':
    print('Setting up environment variables...')
    app_port = os.getenv("APP_PORT")
    app_host = os.getenv("APP_HOST")
    if app_port is None:
        raise RuntimeError('Could not find APP_PORT environment variable')
    if app_host is None:
        raise RuntimeError('Could not find APP_HOST environment variable')

    http_server = WSGIServer((app_host, int(app_port)), app)
    print(f'Server is up and running on {app_host}:{app_port}')
    http_server.serve_forever()
