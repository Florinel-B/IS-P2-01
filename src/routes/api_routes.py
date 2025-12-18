"""
Rutas API para Flask con simulación y predicción en tiempo real.

Incluye:
- Gestión de usuarios, notificaciones y estadísticas.
- Simulación de datos y emisión vía WebSocket.
- Predicción de incidencias en tiempo real con RealtimePredictor.
- Endpoints principales: 
  /datos_grafica, /actualizar_grafica, /enviar_error, /notificaciones,
  /usuarios, /estadisticas, /incidencias_criticas,
  /simulacion/iniciar, /simulacion/detener, /voltajes_csv, /prediccion/realtime.

Clasificación de incidencias:
0 = Normal, 1 = Anomalía voltaje, 2 = Cuelgue, 3 = Alerta preventiva.
"""

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

        data = request.get_json(silent=True) or {}
        descripcion = str(data.get('descripcion', '')).strip()
        severidad = str(data.get('severidad', 'ERROR'))
        
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

    def _run_simulation(user_id: str, device_id: int, speed: float, start_offset: int = 0, prediction_mode: str = "stream"):
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
        
        # Precarga y/o precálculo según modo
        seq_len = predictor.seq_len  # 60 por defecto

        batch_preds_future = None
        batch_probs_future = None
        batch_preds_current = None
        batch_probs_current = None
        batch_alerta = None

        if prediction_mode == "batch":
            # Modo "como usar_ensemble": calculamos predicciones para todo el tramo una vez.
            print(f"[SIM] Modo batch: precalculando predicciones (t y t+1) sobre {len(df)} filas...")
            try:
                results = predictor.detector.predict_next_state(df, forecast_minutes=1)
                batch_preds_current = results.get('predictions_current')
                batch_preds_future = results.get('predictions_future')
                batch_probs_current = results.get('probabilities_current')
                batch_probs_future = results.get('probabilities_future')
                batch_alerta = results.get('alerta_preventiva')
                print(f"[SIM] ✓ Batch listo: alertas={int(results.get('n_alertas', 0) or 0)}")
            except Exception as e:
                print(f"[SIM] ⚠️ Falló batch, fallback a stream. Error: {e}")
                prediction_mode = "stream"

        if prediction_mode == "stream":
            print(f"[SIM] Precargando primeros {seq_len} datos...")
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

            print(f"[SIM] ✓ Buffer inicializado con {len(predictor.buffer)} datos")
            socketio.emit('precarga_completa', {'buffer_size': len(predictor.buffer)}, room=user_id)

        prev_time = None
        prev_voltages: Optional[Dict[str, float]] = None

        # Umbrales de incidencias "reglamentarias" (independientes del modelo)
        HUECO_CUELGUE_SECONDS = 120  # >2 min sin datos
        SALTO_VOLTAGE_MV = 500       # >= 500mV en cualquier canal

        # Demo: limitar puntos emitidos si el dataset es muy grande (evita saturar el navegador)
        max_points = int(state.get('max_points', 2000) or 2000)

        for idx, row in df.iterrows():
            # Saltar los primeros seq_len que ya fueron precargados SOLO en modo stream
            if prediction_mode == "stream" and idx < seq_len:
                continue
            
            if not state.get('running'):
                break

            if max_points <= 0:
                break

            # Estado real por reglas (t): 0 normal, 1 salto voltaje, 2 hueco/cuelgue
            real_state_t = 0
            real_state_reason = "normal"

            if prev_time is not None:
                # Incidencia por cuelgue (hueco temporal)
                gap_seconds = (row['tiempo'] - prev_time).total_seconds()
                if gap_seconds > HUECO_CUELGUE_SECONDS:
                    real_state_t = 2
                    real_state_reason = f"hueco_tiempo>{HUECO_CUELGUE_SECONDS}s"
                    msg_gap = f"\ud83d\udea8 CUELGUE/NO-DATOS: hueco de {gap_seconds/60:.1f} min sin registros"
                    sistema.registro.registrar_mensaje({
                        'tipo': 'CRITICAL',
                        'descripcion': msg_gap,
                        'user_id': user_id,
                        'incidencia_tipo': 2,
                        'regla': 'hueco_tiempo',
                        'gap_seconds': gap_seconds,
                        'timestamp': row['tiempo'].isoformat() if hasattr(row['tiempo'], 'isoformat') else str(row['tiempo'])
                    })
                    socketio.emit('notificacion_incidencia', {
                        'tipo': 2,
                        'mensaje': msg_gap,
                        'regla': 'hueco_tiempo',
                        'gap_seconds': gap_seconds,
                        'timestamp': row['tiempo'].isoformat() if hasattr(row['tiempo'], 'isoformat') else str(row['tiempo'])
                    }, room=user_id)

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

            # Incidencia por salto de voltaje >= 500mV (cambio brusco)
            if prev_voltages is not None:
                deltas = {k: abs(float(voltages[k]) - float(prev_voltages.get(k, 0.0))) for k in ['R1_a', 'R2_a', 'R1_b', 'R2_b']}
                worst_channel, worst_delta = max(deltas.items(), key=lambda kv: kv[1])
                if worst_delta >= SALTO_VOLTAGE_MV:
                    # Si ya venía marcado como cuelgue por hueco de tiempo, lo dejamos como cuelgue.
                    if real_state_t != 2:
                        real_state_t = 1
                        real_state_reason = f"salto_voltaje:{worst_channel}"
                    msg_jump = f"\ud83d\udd34 SALTO VOLTAJE: {worst_channel} \u0394={worst_delta:.0f} mV (\u2265 {SALTO_VOLTAGE_MV} mV)"
                    sistema.registro.registrar_mensaje({
                        'tipo': 'WARNING',
                        'descripcion': msg_jump,
                        'user_id': user_id,
                        'incidencia_tipo': 1,
                        'regla': 'salto_voltaje',
                        'canal': worst_channel,
                        'delta_mV': float(worst_delta),
                        'voltajes': voltages,
                        'timestamp': row['tiempo'].isoformat() if hasattr(row['tiempo'], 'isoformat') else str(row['tiempo'])
                    })
                    socketio.emit('notificacion_incidencia', {
                        'tipo': 1,
                        'mensaje': msg_jump,
                        'regla': 'salto_voltaje',
                        'canal': worst_channel,
                        'delta_mV': float(worst_delta),
                        'voltajes': voltages,
                        'timestamp': row['tiempo'].isoformat() if hasattr(row['tiempo'], 'isoformat') else str(row['tiempo'])
                    }, room=user_id)

            prev_voltages = voltages

            if prediction_mode == "batch" and batch_preds_future is not None:
                # Reconstruir estructura como RealtimePredictor para que el frontend no cambie
                pred_actual = int(batch_preds_current[idx]) if batch_preds_current is not None else 0
                pred_sig = int(batch_preds_future[idx])
                probs_act = batch_probs_current[idx].tolist() if batch_probs_current is not None else [1.0, 0.0, 0.0]
                probs_sig = batch_probs_future[idx].tolist() if batch_probs_future is not None else [1.0, 0.0, 0.0]
                alerta_prev = bool(batch_alerta[idx]) if batch_alerta is not None else False
                class_names = {0: "Normal", 1: "Anomalía Voltaje (+0.5V)", 2: "Cuelgue Sistema"}
                pred = {
                    'prediccion_actual': pred_actual,
                    'clase_actual': class_names.get(pred_actual, 'Unknown'),
                    'confianza_actual': float(max(probs_act)) if probs_act else 0.0,
                    'prob_normal_actual': probs_act[0],
                    'prob_anomalia_voltaje_actual': probs_act[1],
                    'prob_cuelgue_actual': probs_act[2],
                    'prediccion_siguiente': pred_sig,
                    'clase_siguiente': class_names.get(pred_sig, 'Unknown'),
                    'confianza_siguiente': float(max(probs_sig)) if probs_sig else 0.0,
                    'prob_normal_siguiente': probs_sig[0],
                    'prob_anomalia_voltaje_siguiente': probs_sig[1],
                    'prob_cuelgue_siguiente': probs_sig[2],
                    'alerta_preventiva': alerta_prev,
                    'status': status_val,
                    'buffer_size': seq_len
                }
            else:
                pred = predictor.predict_single(voltages, status_val, row['tiempo'])

            # NUEVA: Evaluación con PREDICCIÓN ANTICIPADA
            incidencia_msg = None
            incidencia_tipo = None
            # Normalizar tipos (evita Unknown/str/float en comparaciones e indexación)
            confianza_actual = float(pred.get('confianza_actual', 0) or 0)
            confianza_siguiente = float(pred.get('confianza_siguiente', 0) or 0)
            clase_actual = int(pred.get('prediccion_actual', 0) or 0)
            clase_siguiente = int(pred.get('prediccion_siguiente', 0) or 0)
            alerta_preventiva = pred.get('alerta_preventiva', False)
            
            # Prioridad: Alerta Preventiva (cambio anticipado)
            if alerta_preventiva:
                incidencia_tipo = 3  # PREVENTIVA
                clase_actual_nombre = ["Normal", "Anomalía Voltaje", "Cuelgue"][clase_actual]
                clase_siguiente_nombre = ["Normal", "Anomalía Voltaje", "Cuelgue"][clase_siguiente]
                incidencia_msg = f"⚠️  ALERTA PREVENTIVA: {clase_actual_nombre} → {clase_siguiente_nombre} (confianza: {confianza_siguiente*100:.1f}%)"
            
            # Tipo 2: Cuelgue Sistema predicho (clase siguiente = 2)
            elif clase_siguiente == 2 and confianza_siguiente > 0.7:
                incidencia_tipo = 2
                incidencia_msg = f"🔴 CRÍTICO PREDICHO: Cuelgue del Sistema (RF prob: {confianza_siguiente*100:.1f}%)"
            
            # Tipo 1: Anomalía Voltaje predicha (clase siguiente = 1)
            elif clase_siguiente == 1 and confianza_siguiente > 0.7:
                incidencia_tipo = 1
                incidencia_msg = f"🔴 ALERTA PREDICHA: Anomalía de Voltaje (RF prob: {confianza_siguiente*100:.1f}%)"
            
            # Enviar notificación si hay incidencia
            if incidencia_msg:
                sistema.registro.registrar_mensaje({
                    'tipo': 'CRITICAL' if incidencia_tipo == 2 else 'WARNING' if incidencia_tipo in [1, 3] else 'INFO',
                    'descripcion': incidencia_msg,
                    'user_id': user_id,
                    'incidencia_tipo': incidencia_tipo,
                    'confianza': confianza_siguiente,
                    'alerta_preventiva': incidencia_tipo == 3
                })
                socketio.emit('notificacion_incidencia', {
                    'tipo': incidencia_tipo,
                    'mensaje': incidencia_msg,
                    'confianza': confianza_siguiente,
                    'alerta_preventiva': incidencia_tipo == 3,
                    'tiempo': row['tiempo'].isoformat()
                }, room=user_id)

            # Codificación unificada (modelo y UI):
            # 0 = Normal, 1 = Anomalía Voltaje, 2 = Cuelgue
            # Usar predicción del SIGUIENTE estado como valor principal para la gráfica
            pred_mapped = int(pred.get('prediccion_siguiente', 0))

            payload = {
                'tiempo': row['tiempo'].isoformat(),
                'status': status_val,
                'real_state_t': int(real_state_t),
                'real_state_reason': real_state_reason,
                **voltages,
                'prediccion_completa': pred,
                'prediccion_actual': pred.get('prediccion_actual', 0),
                'prediccion_siguiente': pred.get('prediccion_siguiente', 0),
                'clase_actual': pred.get('clase_actual', 'Unknown'),
                'clase_siguiente': pred.get('clase_siguiente', 'Unknown'),
                'confianza_actual': pred.get('confianza_actual', 0),
                'confianza_siguiente': pred.get('confianza_siguiente', 0),
                'alerta_preventiva': pred.get('alerta_preventiva', False),
                'pred': pred_mapped,
                'incidencia': incidencia_tipo
            }
            socketio.emit('dato_voltaje', payload, room=user_id)

            max_points -= 1

        state['running'] = False

    @api_bp.route('/simulacion/iniciar', methods=['POST'])
    def iniciar_simulacion():
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id')
        device_id = data.get('id', 7)
        speed = float(data.get('speed', 1))
        start_offset = int(data.get('start_offset', 0))  # Ej: 100 para saltarse primeras 100 filas
        max_points = int(data.get('max_points', 2000) or 2000)
        prediction_mode = str(data.get('prediction_mode', 'stream') or 'stream').strip().lower()

        if not user_id:
            return jsonify({'error': 'user_id requerido'}), 400
        try:
            device_id = int(device_id)
        except ValueError:
            return jsonify({'error': 'id debe ser numérico'}), 400

        # Detener si ya había una simulación
        if user_id in simulation_states:
            simulation_states[user_id]['running'] = False

        simulation_states[user_id] = {'running': True, 'device_id': device_id, 'speed': speed, 'max_points': max_points}
        socketio.start_background_task(_run_simulation, user_id, device_id, speed, start_offset, prediction_mode)

        return jsonify({'success': True, 'user_id': user_id, 'id': device_id, 'speed': speed, 'prediction_mode': prediction_mode})

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