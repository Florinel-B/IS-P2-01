from typing import List, Dict, Any
from datetime import datetime
from model.usuario import Usuario


class RegistroIncidencias:
    """Clase que observa y registra todos los mensajes e incidencias"""
    def __init__(self, socketio=None):
        self.observadores: List[Usuario] = []
        self.mensajes_registrados: List[Dict[str, Any]] = []
        self.incidencias_criticas: List[Dict[str, Any]] = []
        self.estadisticas = {
            'total_mensajes': 0,
            'total_errores': 0,
            'total_warnings': 0,
            'total_critical': 0
        }
        self.socketio = socketio
    
    def suscribir(self, usuario: Usuario):
        """Suscribir un usuario para recibir notificaciones"""
        if usuario not in self.observadores:
            self.observadores.append(usuario)
            print(f"Usuario {usuario.user_id} suscrito al sistema")
    
    def desuscribir(self, usuario: Usuario):
        """Desuscribir un usuario"""
        if usuario in self.observadores:
            self.observadores.remove(usuario)
            print(f"Usuario {usuario.user_id} desuscrito del sistema")
    
    def notificar_todos(self, notificacion: Dict[str, Any]):
        """Notificar a todos los usuarios suscritos"""
        for observador in self.observadores:
            observador.actualizar(notificacion)
            # Emitir via WebSocket si está disponible
            if self.socketio:
                self.socketio.emit('nueva_notificacion', notificacion, room=observador.user_id)
    
    def registrar_mensaje(self, mensaje: Dict[str, Any]):
        """Registrar un mensaje en el sistema"""
        # Agregar timestamp si no existe
        if 'timestamp' not in mensaje:
            mensaje['timestamp'] = datetime.now().strftime('%H:%M:%S')
        
        # Registrar en la lista de mensajes
        self.mensajes_registrados.append(mensaje)
        self.estadisticas['total_mensajes'] += 1
        
        # Actualizar estadísticas según tipo
        tipo = mensaje.get('tipo', 'INFO')
        if tipo == 'ERROR':
            self.estadisticas['total_errores'] += 1
        elif tipo == 'WARNING':
            self.estadisticas['total_warnings'] += 1
        elif tipo == 'CRITICAL':
            self.estadisticas['total_critical'] += 1
            self.incidencias_criticas.append(mensaje)
        
        # Crear notificación para usuarios
        notificacion = {
            'timestamp': mensaje['timestamp'],
            'tipo': tipo,
            'mensaje': mensaje.get('descripcion', mensaje.get('mensaje', 'Mensaje sin descripción'))
        }
        
        # Notificar a todos los usuarios suscritos
        self.notificar_todos(notificacion)
        
        # Analizar y registrar incidencias dependientes
        self._analizar_incidencias(mensaje)
    
    def _analizar_incidencias(self, mensaje: Dict[str, Any]):
        """Analizar mensaje y registrar incidencias dependientes"""
        tipo = mensaje.get('tipo', 'INFO')
        descripcion = mensaje.get('descripcion', '')
        
        # Ejemplo de lógica de incidencias dependientes
        # Si hay múltiples errores seguidos, crear incidencia crítica
        if tipo == 'ERROR':
            errores_recientes = [m for m in self.mensajes_registrados[-5:] if m.get('tipo') == 'ERROR']
            if len(errores_recientes) >= 3:
                incidencia = {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'tipo': 'CRITICAL',
                    'mensaje': f'⚠️ ALERTA: Múltiples errores detectados ({len(errores_recientes)} errores recientes)'
                }
                self.notificar_todos(incidencia)
        
        # Si se detecta palabra clave "crítico" o "urgente"
        palabras_criticas = ['crítico', 'critico', 'urgente', 'fatal', 'desastre']
        if any(palabra in descripcion.lower() for palabra in palabras_criticas):
            incidencia = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'tipo': 'CRITICAL',
                'mensaje': f'🚨 Incidencia crítica detectada: {descripcion[:50]}...'
            }
            self.incidencias_criticas.append(incidencia)
            self.notificar_todos(incidencia)
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            **self.estadisticas,
            'usuarios_conectados': len(self.observadores),
            'incidencias_criticas': len(self.incidencias_criticas)
        }