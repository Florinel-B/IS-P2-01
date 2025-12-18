"""
Aplicación Flask con soporte WebSocket mediante Flask-SocketIO.

Funciones principales:
- create_app():
    - Crea la aplicación Flask y configura la clave secreta.
    - Inicializa SocketIO con CORS permitido.
    - Instancia el sistema de gestión (SistemaGestion).
    - Registra rutas web y API mediante Blueprints.
    - Inicializa eventos de WebSocket.
    - Devuelve la app y el objeto SocketIO.

Ejecución directa:
- Ejecuta la app en modo debug, escuchando en todas las interfaces (0.0.0.0) y puerto 5000.
"""

from flask import Flask
from flask_socketio import SocketIO
import secrets

# Importar modelos (imports relativos para que funcione como paquete `src`)
from model.sysGestion import SistemaGestion

# Importar inicializadores de rutas
from routes.web_routes import init_routes as init_web_routes
from routes.api_routes import init_routes as init_api_routes
from socket_events import init_socketio_events


def create_app():
    """Factory para crear la aplicación Flask"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = secrets.token_hex(16)
    
    # Inicializar SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Crear sistema de gestión
    sistema = SistemaGestion(socketio)
    
    # Registrar blueprints
    web_bp = init_web_routes(sistema)
    api_bp = init_api_routes(sistema, socketio)
    
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)
    
    # Inicializar eventos de WebSocket
    init_socketio_events(socketio)
    
    return app, socketio


if __name__ == '__main__':
    app, socketio = create_app()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)