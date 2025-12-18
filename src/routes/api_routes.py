from flask import Blueprint, request, jsonify
import random
import os
import pickle
from typing import Dict, Tuple, List, Optional

from data_processing import leer_datos_procesados
from predict_realtime import RealtimePredictor

api_bp = Blueprint('api', __name__, url_prefix='/api')


predictor_instance: Optional[RealtimePredictor] = None

# Estado de simulaciones por usuario (para poder detenerlas)
simulation_states: Dict[str, Dict] = {}

# Cache del dataset para evitar recargar el CSV
dataset_cache = None

def _get_dataset():
    """Obtiene el dataset cacheado desde pickle, cargándolo una sola vez."""
    global dataset_cache
    if dataset_cache is None:
        pkl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datos_procesados.pkl'))
        print(f"🔄 Cargando dataset desde {pkl_path}...")
        try:
            dataset_cache = leer_datos_procesados(pkl_path)
            print(f"✓ Dataset cacheado: {len(dataset_cache)} registros")
        except FileNotFoundError:
            print(f"⚠️  Pickle no encontrado en {pkl_path}")
            raise
    return dataset_cache

def _get_predictor() -> RealtimePredictor:
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = RealtimePredictor()
    return predictor_instance


def _extract_voltage_fields(data: Dict) -> Tuple[Dict[str, float], List[str]]:
    alias_map = {
        "voltageReceiver1": "R1_a",
        "voltageReceiver2": "R2_a",
        "voltageReceiver3": "R1_b",
        "voltageReceiver4": "R2_b",
    }

    voltage_data: Dict[str, float] = {}

    for target in ["R1_a", "R2_a", "R1_b", "R2_b"]:
        if target in data:
            voltage_data[target] = float(data[target])
            continue

        alias = next((a for a, dest in alias_map.items() if dest == target and a in data), None)
        if alias:
            voltage_data[target] = float(data[alias])

    missing = [field for field in ["R1_a", "R2_a", "R1_b", "R2_b"] if field not in voltage_data]
    return voltage_data, missing


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

    def _run_simulation(user_id: str, device_id: int, speed: float, start_offset: int = 0):
        predictor = _get_predictor()
        predictor.reset_buffer()

        df = _get_dataset()
        df = df[df['id'] == device_id].sort_values('tiempo')
        
        # Saltar los primeros N registros (para empezar a hora más adelantada)
        if start_offset > 0:
            df = df.iloc[start_offset:].reset_index(drop=True)
            if len(df) == 0:
                socketio.emit('error', {'mensaje': f'Offset {start_offset} excede los {len(_get_dataset())} registros'}, room=user_id)
                return

        state = simulation_states[user_id]
        
        # Precarga: llenar el buffer con los primeros 60 datos sin emitir eventos
        seq_len = predictor.seq_len  # 60 por defecto
        for i in range(min(seq_len, len(df))):
            row = df.iloc[i]
            voltages = {
                'R1_a': float(row.get('R1_a', 0) or 0),
                'R2_a': float(row.get('R2_a', 0) or 0),
                'R1_b': float(row.get('R1_b', 0) or 0),
                'R2_b': float(row.get('R2_b', 0) or 0)
            }
            status_val = int(row.get('status', 1) or 1)
            # Precarga sin emitir (solo llena el buffer)
            predictor.predict_single(voltages, status_val, row['tiempo'])
        
        # Notificar que la precarga está lista
        socketio.emit('precarga_completa', {'buffer_size': len(predictor.buffer)}, room=user_id)
        
        prev_time = None

        for idx, row in df.iterrows():
            # Saltar los primeros seq_len que ya fueron precargados
            if idx < seq_len:
                continue
            if not state.get('running'):
                break

            if prev_time is not None:
                delay = max(0.01, (row['tiempo'] - prev_time).total_seconds() / speed)
                socketio.sleep(delay)
            prev_time = row['tiempo']

            voltages = {
                'R1_a': float(row.get('R1_a', 0) or 0),
                'R2_a': float(row.get('R2_a', 0) or 0),
                'R1_b': float(row.get('R1_b', 0) or 0),
                'R2_b': float(row.get('R2_b', 0) or 0)
            }
            status_val = int(row.get('status', 1) or 1)

            pred = predictor.predict_single(voltages, status_val, row['tiempo'])

            # Evaluar si hay incidencia detectada y enviar notificación
            incidencia_msg = None
            incidencia_tipo = None
            confianza = pred.get('confianza', 0)
            clase = pred.get('prediccion', 0)
            lstm_prob = pred.get('lstm_probabilidad', 0)
            
            # Tipo 1: Anomalía Voltaje (LSTM > 70%)
            if lstm_prob > 0.7:
                incidencia_tipo = 1
                incidencia_msg = f"🔴 ALERTA: Anomalía de Voltaje detectada (LSTM prob: {lstm_prob*100:.1f}%)"
            
            # Tipo 2: Cuelgue Sistema (RF > 70% y clase == 2)
            if clase == 2 and confianza > 0.7:
                incidencia_tipo = 2
                incidencia_msg = f"🔴 CRÍTICO: Cuelgue del Sistema detectado (RF prob: {confianza*100:.1f}%)"
            
            # Enviar notificación si hay incidencia
            if incidencia_msg:
                sistema.registro.registrar_mensaje({
                    'tipo': 'CRITICAL' if incidencia_tipo == 2 else 'WARNING',
                    'descripcion': incidencia_msg,
                    'user_id': user_id,
                    'incidencia_tipo': incidencia_tipo,
                    'confianza': confianza
                })
                socketio.emit('notificacion_incidencia', {
                    'tipo': incidencia_tipo,
                    'mensaje': incidencia_msg,
                    'confianza': confianza,
                    'tiempo': row['tiempo'].isoformat()
                }, room=user_id)

            payload = {
                'tiempo': row['tiempo'].isoformat(),
                'status': status_val,
                **voltages,
                'prediccion': pred,
                'incidencia': incidencia_tipo
            }
            socketio.emit('dato_voltaje', payload, room=user_id)

        state['running'] = False

    @api_bp.route('/simulacion/iniciar', methods=['POST'])
    def iniciar_simulacion():
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id')
        device_id = data.get('id', 7)
        speed = float(data.get('speed', 1))
        start_offset = int(data.get('start_offset', 0))  # Ej: 100 para saltarse primeras 100 filas

        if not user_id:
            return jsonify({'error': 'user_id requerido'}), 400
        try:
            device_id = int(device_id)
        except ValueError:
            return jsonify({'error': 'id debe ser numérico'}), 400

        # Detener si ya había una simulación
        if user_id in simulation_states:
            simulation_states[user_id]['running'] = False

        simulation_states[user_id] = {'running': True, 'device_id': device_id, 'speed': speed}
        socketio.start_background_task(_run_simulation, user_id, device_id, speed, start_offset)

        return jsonify({'success': True, 'user_id': user_id, 'id': device_id, 'speed': speed})

    @api_bp.route('/simulacion/detener', methods=['POST'])
    def detener_simulacion():
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'error': 'user_id requerido'}), 400
        if user_id in simulation_states:
            simulation_states[user_id]['running'] = False
        return jsonify({'success': True, 'user_id': user_id})

    @api_bp.route('/voltajes_csv', methods=['GET'])
    def obtener_voltajes_csv():
        """Devuelve series de voltajes del CSV para la gráfica."""
        limit = int(request.args.get('limit', 200))
        device_id = request.args.get('id')

        df = _get_dataset()

        # Filtrar por id si se solicita; si no, escoger el id con más mediciones válidas
        if device_id is not None:
            try:
                device_id_int = int(device_id)
                df = df[df['id'] == device_id_int]
            except ValueError:
                return jsonify({'error': 'id debe ser numérico'}), 400
        else:
            # Elegir el id con más valores no nulos en R1_a
            counts = df.groupby('id')['R1_a'].count()
            if len(counts) == 0:
                return jsonify({'error': 'CSV sin datos'}), 400
            device_id_int = counts.idxmax()
            df = df[df['id'] == device_id_int]

        df = df.sort_values('tiempo').head(limit)

        return jsonify({
            'id': int(device_id_int),
            'labels': df['tiempo'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            'R1_a': df['R1_a'].fillna(0).tolist(),
            'R2_a': df['R2_a'].fillna(0).tolist(),
            'R1_b': df['R1_b'].fillna(0).tolist(),
            'R2_b': df['R2_b'].fillna(0).tolist(),
            'status': df['status'].fillna(0).tolist()
        })

    @api_bp.route('/prediccion/realtime', methods=['POST'])
    def predecir_realtime():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'error': 'JSON requerido'}), 400

        predictor = _get_predictor()

        payloads = data if isinstance(data, list) else [data]
        resultados = []

        for payload in payloads:
            voltage_data, missing = _extract_voltage_fields(payload)
            if missing:
                return jsonify({'error': 'Faltan campos de voltaje', 'campos_requeridos': missing}), 400

            status = int(payload.get('status', 1))
            tiempo = payload.get('tiempo')
            pred = predictor.predict_single(voltage_data, status, tiempo)
            resultados.append(pred)

        if isinstance(data, list):
            return jsonify(resultados)
        return jsonify(resultados[0])
    
    return api_bp