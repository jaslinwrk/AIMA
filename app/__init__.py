from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
from flask_cors import CORS
from flask_socketio import SocketIO

db = SQLAlchemy()
csrf = CSRFProtect()
cors = CORS()
socketio = SocketIO()

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    db.init_app(app)
    csrf.init_app(app)
    cors.init_app(app)
    socketio.init_app(app)

    with app.app_context():
        from . import routes
        app.register_blueprint(routes.main)
        db.create_all()

    return app
