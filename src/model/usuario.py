"""
Definición de la clase Usuario y la interfaz Observer.

Incluye:
- Observer (interfaz abstracta): requiere el método actualizar().
- Usuario: representa a un usuario conectado, maneja notificaciones,
  envía errores y mantiene datos de gráficas limitados a los últimos 20 puntos.
- Métodos principales: actualizar(), agregar_dato_grafica(), enviar_error(), to_dict().
"""

from typing import Dict, Any, List
from datetime import datetime
from abc import ABC, abstractmethod


class Observer(ABC):
    """Interfaz para los observadores"""
    @abstractmethod
    def actualizar(self, notificacion: Dict[str, Any]):
        pass


class Usuario(Observer):
    """Clase que representa a un usuario conectado"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conectado = datetime.now().strftime('%H:%M:%S')
        self.notificaciones: List[Dict[str, Any]] = []
        self.datos_grafica = {'x': [], 'y': []}
    
    def actualizar(self, notificacion: Dict[str, Any]):
        """Recibe notificaciones del sistema"""
        self.notificaciones.append(notificacion)
    
    def agregar_dato_grafica(self, x: int, y: int):
        """Agregar un punto a la gráfica del usuario"""
        self.datos_grafica['x'].append(x)
        self.datos_grafica['y'].append(y)
        
        # Mantener solo los últimos 20 puntos
        if len(self.datos_grafica['x']) > 20:
            self.datos_grafica['x'] = self.datos_grafica['x'][-20:]
            self.datos_grafica['y'] = self.datos_grafica['y'][-20:]
    
    def enviar_error(self, descripcion: str, severidad: str = 'ERROR'):
        """Usuario envía un error al sistema"""
        return {
            'descripcion': descripcion,
            'severidad': severidad,
            'user_id': self.user_id,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir usuario a diccionario"""
        return {
            'user_id': self.user_id,
            'conectado': self.conectado,
            'total_notificaciones': len(self.notificaciones)
        }