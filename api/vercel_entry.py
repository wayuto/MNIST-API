from vercel import wsgi
from app import app

application = wsgi(app)
