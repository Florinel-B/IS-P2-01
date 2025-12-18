"""
Gestión de usuarios y registro de incidencias.

Clase principal:
- SistemaGestion:
    - Crea, obtiene y desconecta usuarios.
    - Mantiene un registro de incidencias (RegistroIncidencias).
    - Suscribe automáticamente a los usuarios al registro y envía notificaciones de conexión.
    - Permite listar todos los usuarios conectados en formato diccionario.
""" 

from typing import Dict, List, Any
from model.usuario import Usuario
from model.registro_incidencias import RegistroIncidencias


class SistemaGestion:
    """Clase principal que gestiona usuarios y el registro de incidencias"""
    def __init__(self, socketio=None):
        self.usuarios: Dict[str, Usuario] = {}
        self.registro = RegistroIncidencias(socketio)
    
    def crear_usuario(self, user_id: str) -> Usuario:
        """Crear y registrar un nuevo usuario"""
        if user_id not in self.usuarios:
            usuario = Usuario(user_id)
            self.usuarios[user_id] = usuario
            # Suscribir automáticamente al registro de incidencias
            self.registro.suscribir(usuario)
            
            # Notificación de bienvenida
            self.registro.registrar_mensaje({
                'tipo': 'INFO',
                'descripcion': f'Usuario {user_id} conectado al sistema',
                'user_id': user_id
            })
        
        return self.usuarios[user_id]
    
    def obtener_usuario(self, user_id: str) -> Usuario:
        """Obtener un usuario existente"""
        return self.usuarios.get(user_id)
    
    def listar_usuarios(self) -> List[Dict[str, Any]]:
        """Listar todos los usuarios conectados"""
        return [usuario.to_dict() for usuario in self.usuarios.values()]
    
    def desconectar_usuario(self, user_id: str):
        """Desconectar un usuario del sistema"""
        usuario = self.usuarios.get(user_id)
        if usuario:
            self.registro.desuscribir(usuario)
            del self.usuarios[user_id]