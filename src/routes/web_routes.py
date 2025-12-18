"""
Rutas web para la aplicación Flask.

Incluye:
- Dashboard principal ('/') que genera un user_id y crea un usuario en el sistema.
- Dashboard legacy ('/dashboard_legacy') para compatibilidad con la versión antigua.
- Ambas rutas renderizan templates HTML y registran el usuario en el sistema.
"""

from flask import Blueprint, render_template
import secrets

web_bp = Blueprint('web', __name__)


def init_routes(sistema):
    """Inicializar rutas web con el sistema"""
    
    @web_bp.route('/')
    def dashboard():
        """Dashboard principal - genera ID automáticamente"""
        user_id = f"user_{secrets.token_hex(4)}"
        sistema.crear_usuario(user_id)
        return render_template('dashboard_anticipado.html', user_id=user_id)
    
    @web_bp.route('/dashboard_legacy')
    def dashboard_legacy():
        """Dashboard antiguo (para compatibilidad)"""
        user_id = f"user_{secrets.token_hex(4)}"
        sistema.crear_usuario(user_id)
        return render_template('dashboard.html', user_id=user_id)
    
    return web_bp