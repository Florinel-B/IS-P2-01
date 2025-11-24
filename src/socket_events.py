from flask_socketio import emit, join_room


def init_socketio_events(socketio):
    """Inicializar eventos de WebSocket"""
    
    @socketio.on('connect')
    def handle_connect():
        """Manejar conexión WebSocket"""
        print('Cliente conectado')

    @socketio.on('join')
    def on_join(data):
        """Usuario se une a su sala personal"""
        user_id = data['user_id']
        join_room(user_id)
        emit('joined', {'user_id': user_id})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Manejar desconexión WebSocket"""
        print('Cliente desconectado')