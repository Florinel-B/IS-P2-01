from flask import Blueprint, request, jsonify
import random

api_bp = Blueprint('api', __name__, url_prefix='/api')


def init_routes(sistema, socketio):
    """Inicializar rutas con el sistema y socketio"""
    
    @api_bp.route('/datos_grafica/<user_id>', methods=['GET'])
    def obtener_datos_grafica(user_id):
        """Obtener datos de la gráfica para un usuario"""
        usuario = sistema.obtener_usuario(user_id)
        if not usuario:
            return jsonify({'error': 'Usuario no encontrado'}), 404
        
        return jsonify(usuario.datos_grafica)

    @api_bp.route('/actualizar_grafica/<user_id>', methods=['POST'])
    def actualizar_grafica(user_id):
        """Simular actualización de datos de la gráfica"""
        usuario = sistema.obtener_usuario(user_id)
        if not usuario:
            return jsonify({'error': 'Usuario no encontrado'}), 404
        
        # Agregar nuevos datos simulados
        nuevo_x = len(usuario.datos_grafica['x'])
        nuevo_y = random.randint(0, 100)
        usuario.agregar_dato_grafica(nuevo_x, nuevo_y)
        
        # Emitir actualización via WebSocket
        socketio.emit('actualizar_grafica', usuario.datos_grafica, room=user_id)
        
        # Registrar en el sistema
        sistema.registro.registrar_mensaje({
            'tipo': 'INFO',
            'descripcion': f'Gráfica actualizada por {user_id}',
            'user_id': user_id
        })
        
        return jsonify({'success': True, 'datos': usuario.datos_grafica})

    @api_bp.route('/enviar_error/<user_id>', methods=['POST'])
    def enviar_error(user_id):
        """Recibir un error enviado por el usuario"""
        usuario = sistema.obtener_usuario(user_id)
        if not usuario:
            return jsonify({'error': 'Usuario no encontrado'}), 404
        
        data = request.json
        descripcion = data.get('descripcion', '').strip()
        severidad = data.get('severidad', 'ERROR')
        
        if not descripcion:
            return jsonify({'error': 'Descripción vacía'}), 400
        
        # Usuario envía error
        usuario.enviar_error(descripcion, severidad)
        
        # Registrar en el sistema (notificará a todos los usuarios)
        sistema.registro.registrar_mensaje({
            'tipo': severidad,
            'descripcion': descripcion,
            'user_id': user_id
        })
        
        return jsonify({
            'success': True,
            'mensaje': 'Error registrado correctamente'
        })

    @api_bp.route('/notificaciones/<user_id>', methods=['GET'])
    def obtener_notificaciones(user_id):
        """Obtener todas las notificaciones de un usuario"""
        usuario = sistema.obtener_usuario(user_id)
        if not usuario:
            return jsonify({'error': 'Usuario no encontrado'}), 404
        
        return jsonify(usuario.notificaciones)

    @api_bp.route('/usuarios', methods=['GET'])
    def listar_usuarios():
        """Listar todos los usuarios conectados"""
        return jsonify(sistema.listar_usuarios())

    @api_bp.route('/estadisticas', methods=['GET'])
    def obtener_estadisticas():
        """Obtener estadísticas del sistema"""
        return jsonify(sistema.registro.obtener_estadisticas())

    @api_bp.route('/incidencias_criticas', methods=['GET'])
    def obtener_incidencias_criticas():
        """Obtener todas las incidencias críticas"""
        return jsonify(sistema.registro.incidencias_criticas)
    
    return api_bp