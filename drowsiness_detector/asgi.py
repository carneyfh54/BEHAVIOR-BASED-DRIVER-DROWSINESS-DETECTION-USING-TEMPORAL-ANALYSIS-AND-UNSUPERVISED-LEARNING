import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import analysis.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'drowsiness_detector.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            analysis.routing.websocket_urlpatterns
        )
    ),
})
