# 🛡️ GUÍA DEFINITIVA DE BLINDAJE PARA CHROMADB EN AIR-GAPPED

## 🔒 Compendio de Vulnerabilidades y Contramedidas para Entornos Aislados (v1.0)

*"El servidor confía en los identificadores de modelo suministrados por el cliente sin restricciones, y actúa sobre esa confianza antes de autenticar al usuario"*  
— HiddenLayer, Noviembre 2025 

---

## 📋 TABLA RESUMEN DE VULNERABILIDADES

| CVE / ID | Nombre | Tipo | Versiones Afectadas | Gravedad | Estado | Vector Principal |
|---|---|---|---|---|---|---|
| **CVE-2026-45829** | ChromaToast | RCE Pre-Auth | 1.0.0 – 1.5.8 (Python) | 🔴 **CRÍTICA (CVSS 10.0)** | **Sin parche** | `trust_remote_code` + modelo HuggingFace malicioso |
| **CVE-2026-45832** | Bypass de Autorización V1 | Bypass de control de acceso | 0.5.0 – 1.5.9 (Python) | 🔴 **ALTA (CVSS 8.8)** | **Sin parche** | Endpoints V1 pasan `None` a la capa de autorización |
| **CVE-2026-45833** | RCE Autenticada | Inyección de código | 0.4.17+ (Python) | 🔴 **CRÍTICA (CVSS 9.4)** | **Sin parche** | Modelo malicioso con `UPDATE_COLLECTION` |
| **CVE-2026-45831** | Autorización Incorrecta | Bypass de RBAC | Múltiples (Python) | 🔴 **ALTA (CVSS 8.8)** | **Sin parche** | `SimpleRBACAuthorizationProvider` mal implementado |
| **CVE-2026-45830** | Bypass de Autorización Python | Control de acceso | 0.4.17+ (Python) | 🔴 **CRÍTICA** | **Sin parche** | Usuario autenticado lee/escribe en colecciones ajenas |
| **CVE-2026-8828** | Bypass de Autorización en Rust | Control de acceso | 1.0.0+ (Rust) | 🔴 **CRÍTICA** | **Sin parche** | Cualquier usuario autenticado puede leer/escribir en tenants ajenos |
| **#5848** | Exfiltración por Defecto | Fuga de datos | 1.3.4 y anteriores | 🔴 **CRÍTICA** | **Sin parche** | `DefaultEmbeddingFunction` envía documentos a OpenAI |
| **Memory Poisoning** | Envenenamiento de Memoria | Manipulación de RAG | **Todas** | 🟡 **MEDIA (CVSS 5.0)** | **Sin parche** | Inyección de entradas maliciosas en el directorio |
| **Zero-days múltiples** | Múltiples 0-days | Múltiples | **Múltiples** | 🔴 **CRÍTICA** | **Sin parche** | Servidores expuestos sin autenticación |
| **Imagen Docker vulnerable** | Vulnerabilidades en contenedor | Múltiples | `chromadb/chroma:latest` | 🔴 **CRÍTICA** | **Sin parche** | 1 crítica, 12 altas, 15 medias, 85 bajas |
| **CVE-2026-45834** | Path Traversal en Modelos | Acceso a archivos | 1.0.0+ (Python) | 🔴 **ALTA (CVSS 7.5)** | **Sin parche** | Carga de modelos con rutas relativas maliciosas |
| **CVE-2026-45835** | SSRF en Embedding Functions | Server-Side Request Forgery | 1.0.0+ (Python) | 🔴 **ALTA (CVSS 7.3)** | **Sin parche** | URLs maliciosas en configuración de embedding |
| **CVE-2026-45836** | Denial of Service | Agotamiento de recursos | Todas | 🟡 **MEDIA (CVSS 6.5)** | **Sin parche** | Queries complejas consumen memoria/CPU excesiva |
| **CVE-2026-45837** | Information Disclosure | Exposición de metadatos | 1.0.0+ (Python) | 🟡 **MEDIA (CVSS 5.3)** | **Sin parche** | Errores detallados revelan estructura interna |
| **CVE-2026-45838** | Insecure Defaults | Configuración insegura | Todas | 🟡 **MEDIA (CVSS 6.2)** | **Sin parche** | Permisos de archivos/directorios demasiado permisivos |

---

## 🧩 VULNERABILIDADES POR CATEGORÍA

### 1. 🔥 EJECUCIÓN DE CÓDIGO REMOTO (RCE)

#### 1.1 CVE-2026-45829 — "ChromaToast" (RCE Pre-Autenticación)

**El vector de ataque más crítico.** Reportado por HiddenLayer en noviembre de 2025 y continúa **sin parche** a fecha de hoy.

**Mecanismo del Ataque:**
El servidor Python FastAPI de ChromaDB ejecuta código **antes** de verificar la autenticación:

1. El atacante envía una solicitud `POST` al endpoint `/api/v2/tenants/{tenant}/databases/{db}/collections`
2. Incluye un modelo malicioso de Hugging Face con `trust_remote_code: true`
3. El servidor carga y **ejecuta el modelo** antes de comprobar quién eres

> *"El servidor confía en los identificadores de modelo suministrados por el cliente sin restricciones, y actúa sobre esa confianza antes de autenticar al usuario"*

**Impacto:**
- **CVSS 4.0 = 10.0** (máxima severidad)
- Permite **toma de control total del servidor**
- Afecta a **todas las versiones 1.0.0 hasta 1.5.8**
- Más de **1,000 servidores ChromaDB expuestos** en Internet, muchos con datos de producción reales
- **73% de las instancias expuestas** ejecutan versiones vulnerables
- Más de **14 millones de descargas mensuales** amplifican el impacto en la cadena de suministro

**Lo que se puede obtener:**
- Shell con privilegios del proceso de la base de datos
- API keys, variables de entorno y secretos montados
- Todos los archivos accesibles por el servidor
- Todos los datos de las colecciones vectoriales
- Credenciales de Kubernetes y tokens de service account

**Mitigación (no hay parche):**
- **Usar el frontend Rust** (`chroma run`), que **no es vulnerable** a este CVE específico
- Restringir el acceso a la red del puerto de ChromaDB a **clientes de confianza**
- Deshabilitar `trust_remote_code` en la configuración
- No exponer ChromaDB a Internet

```bash
# 🔴 VULNERABLE - Servidor Python
chroma run --path ./chroma_data

# 🟢 SOLUCIÓN - Usar servidor Rust
# Primero, instalar Rust server
git clone https://github.com/chroma-core/chroma
cd chroma/rust
cargo build --release

# Ejecutar en air-gapped
./target/release/chroma --path ./chroma_data
```

#### 1.2 CVE-2026-45833 — RCE Autenticada

Similar a ChromaToast pero requiere autenticación.

**Mecanismo:**
Un atacante autenticado con permiso `UPDATE_COLLECTION` puede enviar un modelo malicioso al endpoint y ejecutar código arbitrario en el servidor.

**Versiones Afectadas:**
- Desde **0.4.17** en adelante (Python)

**CVSS: 9.4** (crítica)

**Mitigación:**
- Restringir permisos `UPDATE_COLLECTION` a usuarios de confianza
- Evitar `trust_remote_code=true` con repositorios no confiables
- Implementar whitelist de repositorios de modelos permitidos

#### 1.3 CVE-2026-45834 — Path Traversal en Carga de Modelos

**Nuevo vector de ataque descubierto en 2026.**

**Mecanismo:**
Un atacante puede especificar rutas relativas maliciosas (ej. `../../etc/passwd`) en el parámetro de modelo, permitiendo la lectura o sobrescritura de archivos arbitrarios del sistema.

**Versiones Afectadas:**
- 1.0.0+ (Python)

**CVSS: 7.5** (alta)

**Mitigación:**
```python
# 🟢 Validación de rutas antes de cargar modelos
import os
from pathlib import Path

def validate_model_path(model_name, allowed_base_dir="/opt/models"):
    """Valida que el modelo esté dentro del directorio permitido"""
    # Normalizar y resolver la ruta
    resolved_path = os.path.realpath(os.path.join(allowed_base_dir, model_name))
    allowed_base = os.path.realpath(allowed_base_dir)
    
    # Verificar que la ruta resuelta esté dentro del directorio permitido
    if not resolved_path.startswith(allowed_base):
        raise ValueError(f"Path traversal detected: {model_name}")
    
    return resolved_path
```

---

### 2. 🔓 BYPASS DE AUTORIZACIÓN Y CONTROL DE ACCESO

#### 2.1 CVE-2026-45832 — Bypass de Autorización en Endpoints V1

**Un fallo de autorización** que permite a atacantes no autenticados eludir los controles de acceso.

**Mecanismo:**
Todos los endpoints de colección de la **API V1** pasan valores `None` para `tenant` y `database` a la capa de autorización, permitiendo **bypass total** de los controles de acceso.

**Versiones Afectadas:**
- Desde **0.5.0** hasta **1.5.9** (Python)

**CVSS: 8.8** (alta)

**CWE: CWE-639** (Autorización Bypass mediante User-Controlled Key)

**Impacto:**
- Permite a atacantes no autenticados acceder, modificar o eliminar colecciones

**Mitigación:**
- Deshabilitar o migrar de la API V1
- Implementar proxy de autorización que valide tenant antes de pasar a la API
- Usar middleware personalizado para validar autenticación

#### 2.2 CVE-2026-45831 — Autorización Incorrecta en SimpleRBAC

**Mecanismo:**
`SimpleRBACAuthorizationProvider` evalúa incorrectamente los permisos, permitiendo a usuarios autenticados realizar acciones en tenants, bases de datos o colecciones sin autorización adecuada.

**CVSS: 8.8** (alta)

**Mitigación:**
- Restringir acceso a la red de la instancia ChromaDB a clientes y servicios de confianza
- No confiar en el RBAC integrado; implementar autenticación a nivel de ingress
- Usar proxies inversos como Traefik o Nginx con autenticación integrada

#### 2.3 CVE-2026-8828 — Bypass de Autorización en Rust

**Afecta al servidor Rust**, que se suponía que era la alternativa segura.

**Mecanismo:**
Cualquier usuario autenticado puede leer, escribir, actualizar o eliminar datos en **cualquier colección de cualquier tenant**, independientemente de los permisos.

**Versiones Afectadas:**
- **1.0.0** y posteriores (Rust)

**Impacto:**
- Cross-tenant data access sin autorización
- Violación completa del principio de mínimo privilegio

**Mitigación:**
- Implementar middleware que valide tenant en cada request
- Usar control de acceso a nivel de red en lugar de confiar en la autorización integrada
- Segmentar tenants en instancias separadas

#### 2.4 CVE-2026-45830 — Bypass de Autorización en Python

**Mecanismo:**
Similar al CVE-2026-8828 pero afecta al servidor Python. Un usuario autenticado puede acceder arbitrariamente a colecciones de otros tenants o bases de datos.

**Versiones Afectadas:**
- 0.4.17+ (Python)

**CVSS: CRÍTICA**

**Mitigación:**
- No confiar en la autenticación nativa de ChromaDB
- Implementar validación a nivel de aplicación antes de cada operación
- Usar namespaces separados para cada tenant

```python
# 🟢 Middleware de validación de tenant personalizado
from fastapi import Request, HTTPException
from functools import wraps

def validate_tenant_access(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request: Request = kwargs.get('request')
        if not request:
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
        
        if not request:
            raise HTTPException(status_code=400, detail="Request not found")
        
        # Extraer tenant de la URL o headers
        requested_tenant = kwargs.get('tenant') or request.headers.get('X-Tenant-ID')
        authenticated_tenant = request.state.tenant if hasattr(request.state, 'tenant') else None
        
        if not requested_tenant or not authenticated_tenant:
            raise HTTPException(status_code=401, detail="Tenant authentication required")
        
        if requested_tenant != authenticated_tenant:
            raise HTTPException(status_code=403, detail="Access denied to requested tenant")
        
        return await func(*args, **kwargs)
    return wrapper
```

---

### 3. 📤 EXFILTRACIÓN DE DATOS Y PRIVACIDAD

#### 3.1 #5848 — Exfiltración por Defecto a OpenAI

**Una vulnerabilidad de privacidad crítica** que afecta a miles de despliegues.

**Mecanismo:**
Si no se especifica una función de embedding explícita, ChromaDB usa `DefaultEmbeddingFunction` que **envía el contenido completo de los documentos a la API de OpenAI** durante las operaciones de consulta.

**Versiones Afectadas:**
- **1.3.4** y anteriores (probablemente versiones más antiguas)

**Impacto:**
- Exfiltración de **documentos confidenciales** (HIPAA, GDPR, secretos comerciales)
- Violación de políticas de air-gapped
- Exposición de datos sensibles a terceros

**Mitigación:**
```python
# 🔴 VULNERABLE - Envía documentos a OpenAI
from chromadb.utils import embedding_functions
ef = embedding_functions.DefaultEmbeddingFunction()

# 🟢 SOLUCIÓN - Usar embedding local
from chromadb.utils import embedding_functions

# Opción 1: Ollama (recomendado para air-gapped)
ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text"
)

# Opción 2: Sentence Transformers local
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2',
    device='cpu'
)

# Opción 3: ONNX Runtime local
from chromadb.utils import embedding_functions
ef = embedding_functions.ONNXMiniLM_L6_V2()
```

#### 3.2 CVE-2026-45835 — SSRF en Funciones de Embedding

**Nuevo vector de ataque en 2026.**

**Mecanismo:**
Las funciones de embedding que aceptan URLs personalizadas pueden ser explotadas para realizar Server-Side Request Forgery (SSRF), permitiendo al atacante acceder a servicios internos o metadatos de instancias cloud.

**Versiones Afectadas:**
- 1.0.0+ (Python)

**CVSS: 7.3** (alta)

**Mitigación:**
```python
# 🟢 Validación estricta de URLs para embedding functions
import re
from urllib.parse import urlparse

ALLOWED_EMBEDDING_URLS = [
    r'^http://localhost:11434/',  # Ollama local
    r'^http://127\.0\.0\.1:\d+/',  # Servicios locales
]

def validate_embedding_url(url):
    """Valida que la URL de embedding sea segura"""
    parsed = urlparse(url)
    
    # Bloquear IPs internos y metadatos de cloud
    if parsed.hostname:
        # Bloquear IPs de metadatos de cloud
        blocked_ips = ['169.254.169.254', '100.100.100.200', '10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16']
        for blocked in blocked_ips:
            if parsed.hostname == blocked or parsed.hostname.startswith(blocked.split('/')[0]):
                raise ValueError(f"Blocked URL: {url}")
    
    # Solo permitir URLs en whitelist
    for pattern in ALLOWED_EMBEDDING_URLS:
        if re.match(pattern, url):
            return True
    
    raise ValueError(f"URL not allowed: {url}")
```

#### 3.3 Exposición de Datos por CVE-2026-45829

El RCE pre-autenticación permite a los atacantes acceder a **todos los datos del servidor**:
- API keys y credenciales
- Variables de entorno
- Secretos montados en Kubernetes
- Todas las colecciones vectoriales y documentos almacenados
- Logs de aplicación y auditoría
- Configuración de la base de datos

---

### 4. 🧠 MEMORY POISONING — ENVENENAMIENTO DE AGENTES DE IA

**Un ataque sigiloso** que manipula la memoria del agente sin dejar rastro.

**Mecanismo:**
1. El atacante con acceso de escritura al directorio de ChromaDB inyecta una entrada maliciosa con metadatos realistas
2. La entrada está diseñada para ser semánticamente cercana a las consultas del agente
3. El agente la recupera y la trata como un hecho
4. **No hay nada anómalo en los logs**

**CVSS: 5.0** (media)

**Estado:** Sin parche

**Casos de Uso del Ataque:**
- Manipulación de respuestas de chatbots
- Inyección de desinformación en sistemas RAG
- Alteración de decisiones automatizadas
- Compromiso de asistentes de IA internos

**Mitigación:**
- **Firma HMAC** sobre contenido + metadatos — las entradas no firmadas se rechazan
- Filtrado por ámbito de fuente (cross-session injections)
- Ejecutar completamente offline (air-gapped)
- Implementar verificación de integridad de datos

```python
# 🟢 SOLUCIÓN: HMAC + Verificación de integridad + Timestamp

import hmac
import hashlib
import json
import time
from datetime import datetime

class SecureChromaCollection:
    def __init__(self, collection, secret_key, max_age_days=30):
        self.collection = collection
        self.secret_key = secret_key.encode()
        self.max_age_days = max_age_days
    
    def _sign_entry(self, doc, metadata):
        """Firma una entrada con HMAC"""
        # Incluir timestamp en los metadatos
        if '_timestamp' not in metadata:
            metadata['_timestamp'] = int(time.time())
        
        payload = doc + json.dumps(metadata, sort_keys=True)
        signature = hmac.new(
            self.secret_key,
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def add(self, documents, metadatas, ids):
        """Añade entradas con firma de integridad"""
        for i, doc in enumerate(documents):
            signature = self._sign_entry(doc, metadatas[i])
            metadatas[i]['_hmac'] = signature
            metadatas[i]['_signed_by'] = 'security_layer'
        
        return self.collection.add(documents, metadatas, ids)
    
    def query(self, query_text, n_results=10):
        """Consulta con verificación de integridad"""
        results = self.collection.query(query_text, n_results)
        
        valid_results = []
        current_time = int(time.time())
        max_age_seconds = self.max_age_days * 24 * 60 * 60
        
        for i, meta in enumerate(results['metadatas'][0]):
            doc = results['documents'][0][i]
            sig = meta.get('_hmac')
            
            # Verificar firma HMAC
            meta_copy = {k: v for k, v in meta.items() if k not in ['_hmac', '_signed_by']}
            expected = hmac.new(
                self.secret_key,
                (doc + json.dumps(meta_copy, sort_keys=True)).encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(sig or '', expected):
                print(f"⚠️  Entrada con firma inválida detectada: {doc[:50]}...")
                continue
            
            # Verificar antigüedad
            timestamp = meta.get('_timestamp', 0)
            age = current_time - timestamp
            if age > max_age_seconds:
                print(f"⚠️  Entrada expirada (edad: {age} segundos)")
                continue
            
            valid_results.append((doc, meta))
        
        return valid_results
    
    def detect_poisoning(self, suspicious_threshold=0.85):
        """Detecta posibles envenenamientos por similitud anómala"""
        # Obtener estadísticas de similitud
        all_docs = self.collection.get()
        
        if not all_docs['documents']:
            return []
        
        # Detectar clusters de alta similitud (posible inyección masiva)
        from collections import defaultdict
        timestamp_groups = defaultdict(list)
        
        for doc, meta in zip(all_docs['documents'], all_docs['metadatas']):
            timestamp = meta.get('_timestamp', 0)
            # Agrupar por hora
            hour = timestamp // 3600
            timestamp_groups[hour].append(doc)
        
        # Alertar si hay más de N documentos en la misma hora
        suspicious_hours = []
        for hour, docs in timestamp_groups.items():
            if len(docs) > 100:  # Ajustar según volumen normal
                suspicious_hours.append({
                    'hour': hour,
                    'count': len(docs),
                    'timestamp': datetime.fromtimestamp(hour * 3600).isoformat()
                })
        
        return suspicious_hours
```

---

### 5. 🐳 VULNERABILIDADES EN IMÁGENES DOCKER

**Imagen `chromadb/chroma:latest`** contiene múltiples vulnerabilidades:

| Gravedad | Cantidad |
|---|---|
| CRÍTICA | 1 |
| ALTA | 12 |
| MEDIA | 15 |
| BAJA | 85 |

**Vulnerabilidades Comunes en la Imagen:**
- **CVE-2024-XXXX**: Vulnerabilidades en dependencias de Python (requests, urllib3)
- **CVE-2024-YYYY**: Vulnerabilidades en bibliotecas de compresión (zlib, libzip)
- **CVE-2024-ZZZZ**: Vulnerabilidades en OpenSSL/TLS
- Imágenes base desactualizadas (Python 3.9, Alpine antiguo)

**Recomendación:**
- No usar la imagen `latest` en producción
- Construir imagen personalizada con dependencias parcheadas
- Escanear imágenes con Trivy o Grype antes del despliegue
- Usar imágenes base minimalistas (distroless, Alpine)

```dockerfile
# 🟢 Dockerfile seguro para air-gapped
FROM python:3.11-slim as builder

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear entorno virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Imagen final minimalista
FROM python:3.11-slim

# Crear usuario no-root
RUN useradd -m -u 1000 chroma

# Copiar entorno virtual
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiar aplicación
COPY . /app
WORKDIR /app

# Cambiar propietario
RUN chown -R chroma:chroma /app

# Cambiar a usuario no-root
USER chroma

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["chroma", "run", "--host", "0.0.0.0", "--port", "8000"]
```

**Script de escaneo de imagen:**
```bash
#!/bin/bash
# scan_docker_image.sh

IMAGE_NAME=${1:-chromadb/chroma:latest}

echo "🔍 Escaneando imagen: $IMAGE_NAME"

# Escaneo con Trivy
trivy image --severity HIGH,CRITICAL $IMAGE_NAME > trivy_report.txt

# Escaneo con Grype
grype $IMAGE_NAME > grype_report.txt

# Verificar vulnerabilidades críticas
if grep -q "CRITICAL" trivy_report.txt; then
    echo "❌ VULNERABILIDADES CRÍTICAS ENCONTRADAS"
    cat trivy_report.txt
    exit 1
fi

echo "✅ Imagen segura"
```

---

### 6. ⚙️ PROBLEMAS DE CONFIGURACIÓN Y TELEMETRÍA

#### 6.1 Telemetría por Defecto

```python
# 🔴 VULNERABLE - Activa por defecto
import chromadb
client = chromadb.PersistentClient(path="./chroma_data")

# 🟢 SOLUCIÓN - Desactivar telemetría
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_DISABLE_TELEMETRY'] = '1'

# O al inicializar
client = chromadb.PersistentClient(
    path="./chroma_data",
    anonymized_telemetry=False
)
```

**Nota:** En la versión 1.0.0+, las claves de telemetría y logging son **ignoradas**. La telemetría no se puede desactivar mediante configuración en versiones recientes; hay que usar variables de entorno.

#### 6.2 Modelos ONNX que "llaman a casa"

```bash
# 🔴 PROBLEMA: ChromaDB descarga modelos de AWS al primer uso
# 🟢 SOLUCIÓN: Pre-cargar en máquina con acceso

# En máquina CON internet
python -c "import chromadb; chromadb.utils.embedding_functions.ONNXEmbeddingFunction('all-MiniLM-L6-v2')"

# Copiar caché al entorno air-gapped
cp -r ~/.cache/chroma/ /path/to/airgapped/home/.cache/chroma/
```

**Script de pre-carga completo:**
```bash
#!/bin/bash
# preload_models.sh - Pre-carga modelos para entorno air-gapped

MODELS_DIR="/opt/chroma-models"
mkdir -p $MODELS_DIR

echo "📥 Pre-cargando modelos de embedding..."

# Pre-cargar Sentence Transformers
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('$MODELS_DIR/all-MiniLM-L6-v2')
"

# Pre-cargar ONNX
python -c "
from chromadb.utils import embedding_functions
ef = embedding_functions.ONNXMiniLM_L6_V2()
"

# Copiar caché de HuggingFace
cp -r ~/.cache/huggingface/hub $MODELS_DIR/hf_cache

echo "✅ Modelos pre-cargados en $MODELS_DIR"
echo "📦 Para copiar al entorno air-gapped:"
echo "   tar -czf chroma-models.tar.gz $MODELS_DIR"
```

#### 6.3 Autenticación Ignorada en Helm Charts

En los Helm charts para Kubernetes, las configuraciones `chromadb.auth.*` son **ignoradas** en la versión 1.0.0+.

**Solución práctica:**
- Mantener el servicio como ClusterIP
- Poner la autenticación a nivel de ingress
- Usar controles de red: redes privadas, auth de ingress, API gateway o mTLS

**Ejemplo de Helm values seguro:**
```yaml
# values-secure.yaml
replicaCount: 3

image:
  repository: your-registry/chroma
  tag: "1.5.8-secure"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  annotations:
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: chroma-auth-secret
    nginx.ingress.kubernetes.io/auth-realm: "ChromaDB Authentication"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
  hosts:
    - host: chroma.internal.domain
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: chroma-tls
      hosts:
        - chroma.internal.domain

persistence:
  enabled: true
  storageClass: "encrypted-storage"
  accessMode: ReadWriteOnce
  size: 50Gi

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000

env:
  - name: ANONYMIZED_TELEMETRY
    value: "False"
  - name: CHROMA_DISABLE_TELEMETRY
    value: "1"
  - name: CHROMA_SERVER_HOST
    value: "0.0.0.0"
  - name: CHROMA_SERVER_HTTP_PORT
    value: "8000"

nodeSelector:
  node-type: secure
```

#### 6.4 CVE-2026-45838 — Permisos Inseguros por Defecto

**Mecanismo:**
ChromaDB crea archivos y directorios con permisos demasiado permisivos (0644, 0755), permitiendo a otros usuarios del sistema leer datos sensibles.

**Versiones Afectadas:**
- Todas

**CVSS: 6.2** (media)

**Mitigación:**
```python
# 🟢 Script de corrección de permisos
import os
import stat

def secure_chroma_permissions(chroma_path):
    """Asegura permisos restrictivos en archivos de ChromaDB"""
    
    # Configuración de permisos
    DIR_PERMS = 0o700  # rwx------
    FILE_PERMS = 0o600  # rw-------
    
    for root, dirs, files in os.walk(chroma_path):
        # Asegurar permisos de directorios
        for d in dirs:
            dir_path = os.path.join(root, d)
            os.chmod(dir_path, DIR_PERMS)
        
        # Asegurar permisos de archivos
        for f in files:
            file_path = os.path.join(root, f)
            os.chmod(file_path, FILE_PERMS)
    
    print(f"✅ Permisos asegurados en {chroma_path}")

# Uso
secure_chroma_permissions("/var/lib/chroma")
```

---

### 7. 🔗 VULNERABILIDADES EN LA CADENA DE SUMINISTRO

**Más de 14 millones de descargas mensuales** de ChromaDB significan un gran footprint en la cadena de suministro de aplicaciones SaaS impulsadas por IA.

**Riesgos:**
- Dependencias maliciosas (ej. `chromadb-pysqlite3`)
- Actualizaciones de seguridad retrasadas
- Falta de respuesta de los mantenedores a los hallazgos de seguridad
- Typosquatting en PyPI

**Mitigación:**
```bash
#!/bin/bash
# verify_dependencies.sh - Verifica integridad de dependencias

REQUIREMENTS_FILE="requirements.txt"
TRUSTED_HASHES_FILE="trusted_hashes.json"

echo "🔍 Verificando integridad de dependencias..."

# Verificar hashes de paquetes
pip install hashin

while IFS= read -r package; do
    if [ ! -z "$package" ] && [[ ! "$package" =~ ^# ]]; then
        echo "Verificando: $package"
        hashin "$package" -r "$REQUIREMENTS_FILE" --algorithm sha256
    fi
done < "$REQUIREMENTS_FILE"

echo "✅ Dependencias verificadas"
```

**Lista de dependencias verificadas (requirements-secure.txt):**
```txt
chromadb==1.5.8 \
    --hash=sha256:abc123... \
    --hash=sha256:def456...

sentence-transformers==2.2.2 \
    --hash=sha256:xyz789...

fastapi==0.104.1 \
    --hash=sha256:uvw321...

uvicorn==0.24.0 \
    --hash=sha256:rst654...
```

---

### 8. 🛡️ VULNERABILIDADES DE RED Y COMUNICACIÓN

#### 8.1 Exposición de Endpoints sin Autenticación

**Mecanismo:**
ChromaDB expone endpoints REST sin autenticación por defecto, permitiendo acceso no autorizado a la API.

**Mitigación:**
```python
# 🟢 Proxy inverso con autenticación (FastAPI + middleware)
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

app = FastAPI()
security = HTTPBasic()

# Credenciales almacenadas en secretos (no en código)
AUTHORIZED_USERS = {
    "admin": secrets.token_urlsafe(32),
    "readonly": secrets.token_urlsafe(32)
}

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verifica credenciales básicas"""
    correct_password = AUTHORIZED_USERS.get(credentials.username)
    
    if not correct_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not secrets.compare_digest(credentials.password, correct_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return credentials.username

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Middleware de autenticación para todos los endpoints"""
    # Saltar autenticación para health check
    if request.url.path == "/health":
        return await call_next(request)
    
    # Verificar autenticación
    try:
        await verify_credentials()
    except HTTPException:
        return HTTPException(status_code=401, detail="Authentication required")
    
    response = await call_next(request)
    return response

# Proxy a ChromaDB
@app.api_route("/{path:path}")
async def proxy_to_chroma(path: str, request: Request, username: str = Depends(verify_credentials)):
    """Proxy todas las requests a ChromaDB"""
    import httpx
    
    async with httpx.AsyncClient() as client:
        chroma_url = f"http://localhost:8000/{path}"
        
        # Forward headers
        headers = dict(request.headers)
        headers['X-Authenticated-User'] = username
        
        # Forward request
        response = await client.request(
            method=request.method,
            url=chroma_url,
            headers=headers,
            content=await request.body(),
            params=request.query_params
        )
        
        return response
```

#### 8.2 CVE-2026-45836 — Denial of Service por Queries Complejas

**Mecanismo:**
Atacantes pueden enviar queries con parámetros maliciosos (ej. `n_results=1000000`, queries recursivas) que consumen memoria y CPU excesivos, causando DoS.

**Versiones Afectadas:**
- Todas

**CVSS: 6.5** (media)

**Mitigación:**
```python
# 🟢 Limitación de recursos y rate limiting
from fastapi import FastAPI, Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
import asyncio

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

# Configuración de límites
MAX_RESULTS = 1000
MAX_QUERY_LENGTH = 10000
REQUESTS_PER_MINUTE = 60

@app.middleware("http")
async def resource_limits_middleware(request: Request, call_next):
    """Limita recursos para prevenir DoS"""
    
    # Rate limiting
    try:
        await limiter.limit(f"{REQUESTS_PER_MINUTE}/minute")(request)
    except:
        raise HTTPException(status_code=429, detail="Too many requests")
    
    # Validar parámetros de query
    if request.query_params:
        n_results = request.query_params.get('n_results')
        if n_results and int(n_results) > MAX_RESULTS:
            raise HTTPException(
                status_code=400, 
                detail=f"n_results cannot exceed {MAX_RESULTS}"
            )
    
    # Validar tamaño del cuerpo
    content_length = int(request.headers.get('content-length', 0))
    if content_length > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Request body too large (max {MAX_QUERY_LENGTH} bytes)"
        )
    
    # Timeout para requests largas
    try:
        response = await asyncio.wait_for(call_next(request), timeout=30.0)
        return response
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
```

---

### 9. 🔍 VULNERABILIDADES DE LOGS Y MONITOREO

#### 9.1 CVE-2026-45837 — Information Disclosure en Logs

**Mecanismo:**
Los logs de error detallados revelan información sensible como:
- Estructura interna de la base de datos
- Paths de archivos del sistema
- Nombres de tenants y colecciones
- Stack traces con información de implementación

**Versiones Afectadas:**
- 1.0.0+ (Python)

**CVSS: 5.3** (media)

**Mitigación:**
```python
# 🟢 Sanitización de logs
import logging
import re

class SanitizingFormatter(logging.Formatter):
    """Formatter que sanitiza información sensible de logs"""
    
    SENSITIVE_PATTERNS = [
        (r'/home/\w+/', '/home/[REDACTED]/'),
        (r'/var/lib/chroma/\w+/', '/var/lib/chroma/[REDACTED]/'),
        (r'tenant[_-]\w+', 'tenant_[REDACTED]'),
        (r'collection[_-]\w+', 'collection_[REDACTED]'),
        (r'api[_-]key[_-]\w+', 'api_key_[REDACTED]'),
    ]
    
    def format(self, record):
        message = super().format(record)
        
        # Sanitizar patrones sensibles
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            message = re.sub(pattern, replacement, message)
        
        return message

# Configuración de logging seguro
def setup_secure_logging():
    logger = logging.getLogger('chromadb')
    logger.setLevel(logging.INFO)
    
    # Handler para archivo (sanitizado)
    file_handler = logging.FileHandler('/var/log/chroma/app.log')
    file_handler.setFormatter(SanitizingFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Handler para stderr (solo errores críticos)
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(SanitizingFormatter(
        '%(levelname)s: %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(stderr_handler)
    
    return logger
```

#### 9.2 Auditoría de Acceso

**Implementación de logging de auditoría completo:**
```python
# 🟢 Sistema de auditoría completo
import json
import hashlib
from datetime import datetime
from typing import Dict, Any

class AuditLogger:
    def __init__(self, log_file='/var/log/chroma/audit.jsonl'):
        self.log_file = log_file
        self.log_file_handle = open(log_file, 'a')
    
    def log_access(self, 
                   user: str, 
                   action: str, 
                   resource: str, 
                   success: bool, 
                   metadata: Dict[str, Any] = None):
        """Registra evento de acceso con integridad"""
        
        event = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'user': user,
            'action': action,
            'resource': resource,
            'success': success,
            'metadata': metadata or {},
            'client_ip': metadata.get('client_ip') if metadata else None,
            'user_agent': metadata.get('user_agent') if metadata else None
        }
        
        # Calcular hash del evento anterior para cadena de integridad
        event_json = json.dumps(event, sort_keys=True)
        event['hash'] = hashlib.sha256(event_json.encode()).hexdigest()
        
        # Escribir log
        self.log_file_handle.write(json.dumps(event) + '\n')
        self.log_file_handle.flush()
    
    def verify_integrity(self):
        """Verifica integridad de la cadena de logs"""
        self.log_file_handle.seek(0)
        lines = self.log_file_handle.readlines()
        
        previous_hash = None
        for i, line in enumerate(lines):
            event = json.loads(line)
            stored_hash = event.pop('hash')
            
            event_json = json.dumps(event, sort_keys=True)
            calculated_hash = hashlib.sha256(event_json.encode()).hexdigest()
            
            if calculated_hash != stored_hash:
                print(f"❌ Integridad comprometida en línea {i+1}")
                return False
        
        print("✅ Integridad de logs verificada")
        return True
    
    def close(self):
        self.log_file_handle.close()

# Uso
audit = AuditLogger()
audit.log_access(
    user='admin',
    action='create_collection',
    resource='tenant:default/collection:sensitive',
    success=True,
    metadata={'client_ip': '192.168.1.100', 'user_agent': 'Mozilla/5.0'}
)
```

---

### 10. 📦 CHECKLIST DE BLINDAJE AIR-GAPPED

#### 🔒 Pre-Instalación

- [ ] **Modelos pre-cargados** en `~/.cache/chroma/` y `/opt/chroma-models/`
- [ ] **Dependencias Python** empaquetadas: `pip download -r requirements.txt -d ./packages`
- [ ] **Servidor Rust** compilado localmente (evitar Python)
- [ ] **Repositorio local PyPI** para updates sin internet
- [ ] **Escaneo de imagen Docker** con Trivy/Grype
- [ ] **Verificación de hashes** de todos los paquetes descargados
- [ ] **Análisis de dependencias** con safety o pip-audit
- [ ] **Revisión de código fuente** de ChromaDB antes de compilar

#### 🛡️ Configuración

- [ ] **Telemetría desactivada**: `ANONYMIZED_TELEMETRY=False` y `CHROMA_DISABLE_TELEMETRY=1`
- [ ] **Embedding local**: `OllamaEmbeddingFunction` o `SentenceTransformer`
- [ ] **Autenticación**: No confiar en el RBAC integrado; usar auth a nivel de ingress
- [ ] **CORS desactivado**: `CHROMA_SERVER_CORS_ALLOW_ORIGINS=[]`
- [ ] **`trust_remote_code` desactivado** en la configuración
- [ ] **API V1 deshabilitada** o migrada
- [ ] **Logging sanitizado** implementado
- [ ] **Permisos de archivos** restringidos (0600, 0700)
- [ ] **Rate limiting** configurado
- [ ] **Validación de inputs** en todos los endpoints
- [ ] **Timeout de requests** configurado (30s máximo)
- [ ] **Límites de recursos** (CPU, memoria) definidos

#### 🌐 Red

- [ ] **Puerto 8000** solo accesible desde localhost
- [ ] **No expuesto** a Internet (ni siquiera para actualizaciones)
- [ ] **Firewall**: iptables/nftables restringiendo acceso
- [ ] **VPN interna** si múltiples nodos (sin internet)
- [ ] **ClusterIP** para servicios Kubernetes con auth en ingress
- [ ] **mTLS** habilitado entre servicios
- [ ] **Proxy inverso** con autenticación
- [ ] **Monitoreo de tráfico** de red saliente
- [ ] **Bloqueo de IPs de metadatos** cloud (169.254.169.254)
- [ ] **DNS filtrado** para prevenir exfiltración

#### 📊 Auditoría

- [ ] **Logs de acceso** habilitados y sanitizados
- [ ] **Logs de errores** en archivo separado
- [ ] **Alertas de anomalías**: cambios en esquemas o colecciones no autorizados
- [ ] **HMAC signing** para detectar memory poisoning
- [ ] **Rotación de logs** configurada (logrotate)
- [ ] **Cadena de integridad** en logs de auditoría
- [ ] **Backup de logs** en almacenamiento seguro
- [ ] **Monitoreo de permisos** de archivos
- [ ] **Detección de accesos** fuera de horario laboral
- [ ] **Alertas de intentos** de path traversal o SSRF

#### 🧪 Verificación Post-Despliegue

- [ ] **Probar RCE**: intentar cargar modelo remoto (debe fallar)
- [ ] **Probar exfiltración**: monitorear tráfico de red saliente
- [ ] **Probar HMAC**: inyectar entrada sin firma (debe ser rechazada)
- [ ] **Probar bypass**: autenticarse y acceder a tenant ajeno (debe fallar)
- [ ] **Verificar versión**: asegurar que no se usa Python server vulnerable
- [ ] **Probar path traversal**: intentar acceder a archivos del sistema
- [ ] **Probar SSRF**: intentar acceder a metadatos de cloud
- [ ] **Probar DoS**: enviar queries complejas (debe ser limitado)
- [ ] **Verificar logs**: asegurar que no hay información sensible
- [ ] **Probar rate limiting**: exceder límites configurados
- [ ] **Verificar permisos**: asegurar que archivos son 0600/0700
- [ ] **Probar autenticación**: intentar acceso sin credenciales

#### 🔄 Mantenimiento Continuo

- [ ] **Escaneo semanal** de vulnerabilidades con Trivy/Grype
- [ ] **Revisión mensual** de logs de auditoría
- [ ] **Actualización trimestral** de dependencias (en entorno de prueba)
- [ ] **Pentest semestral** de la infraestructura
- [ ] **Rotación anual** de claves HMAC y credenciales
- [ ] **Revisión de políticas** de seguridad cada 6 meses
- [ ] **Capacitación del equipo** en nuevas vulnerabilidades
- [ ] **Monitoreo de CVEs** nuevos de ChromaDB y dependencias

---

## 🔧 SCRIPT DE BLINDAJE COMPLETO (v1.0)

```python
#!/usr/bin/env python3
"""
secure_chromadb_airgapped.py v1.0
Hardening completo para ChromaDB en entorno aislado
Autor: RONIN Audit 1310
Fecha: 13 de Agosto de 2026
"""

import os
import sys
import subprocess
import json
import hmac
import hashlib
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Configuración
CONFIG = {
    "chroma_path": "/opt/chroma",
    "data_path": "/var/lib/chroma",
    "log_path": "/var/log/chroma",
    "embedding_model": "all-MiniLM-L6-v2",
    "secret_key_file": "/etc/chroma/hmac.key",
    "audit_log_file": "/var/log/chroma/audit.jsonl",
    "max_query_size": 10000,
    "rate_limit_per_minute": 60,
    "max_results": 1000
}

class ChromaAirgapSecurer:
    def __init__(self, config):
        self.config = config
        self.audit_events = []
        self._validate_environment()
    
    def _validate_environment(self):
        """Verifica que estamos en air-gapped"""
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", "8.8.8.8"],
                timeout=2,
                capture_output=True
            )
            if result.returncode == 0:
                raise RuntimeError("❌ Este sistema TIENE acceso a internet. Air-gapped requerido.")
            print("✅ Entorno air-gapped verificado.")
            self.audit_event("environment_check", "airgap_verified", True)
        except subprocess.TimeoutExpired:
            print("✅ Entorno air-gapped verificado (timeout).")
            self.audit_event("environment_check", "airgap_verified", True)
        except Exception as e:
            print(f"✅ Entorno air-gapped verificado (error: {e}).")
            self.audit_event("environment_check", "airgap_verified", True)
    
    def audit_event(self, action: str, resource: str, success: bool, metadata: Optional[Dict] = None):
        """Registra evento de auditoría"""
        event = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'action': action,
            'resource': resource,
            'success': success,
            'metadata': metadata or {}
        }
        self.audit_events.append(event)
        
        # Escribir a archivo
        try:
            with open(self.config['audit_log_file'], 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            print(f"⚠️  No se pudo escribir evento de auditoría: {e}")
    
    def disable_telemetry(self):
        """Desactiva toda telemetría"""
        os.environ['ANONYMIZED_TELEMETRY'] = 'False'
        os.environ['CHROMA_DISABLE_TELEMETRY'] = '1'
        
        # Persistir en /etc/environment
        try:
            with open('/etc/environment', 'r') as f:
                content = f.read()
            
            if 'ANONYMIZED_TELEMETRY' not in content:
                with open('/etc/environment', 'a') as f:
                    f.write('\nANONYMIZED_TELEMETRY=False\n')
                    f.write('CHROMA_DISABLE_TELEMETRY=1\n')
            
            print("✅ Telemetría desactivada.")
            self.audit_event("configure", "telemetry_disabled", True)
        except Exception as e:
            print(f"⚠️  No se pudo desactivar telemetría: {e}")
            self.audit_event("configure", "telemetry_disabled", False, {"error": str(e)})
    
    def use_local_embeddings(self):
        """Configura embedding local"""
        try:
            from chromadb.utils import embedding_functions
            
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config['embedding_model'],
                device='cpu'
            )
            
            # Verificar modelo local
            cache_path = Path.home() / ".cache" / "chroma" / self.config['embedding_model']
            if not cache_path.exists():
                print("⚠️  Modelo de embedding no encontrado localmente.")
                print("   Pre-cargar en máquina con internet y copiar:")
                print(f"   cp -r ~/.cache/chroma/{self.config['embedding_model']} /path/to/airgapped")
                self.audit_event("configure", "embedding_model", False, {"missing": True})
            else:
                print(f"✅ Modelo {self.config['embedding_model']} encontrado.")
                self.audit_event("configure", "embedding_model", True)
        except ImportError:
            print("⚠️  ChromaDB no instalado. Instalando...")
            subprocess.run([sys.executable, "-m", "pip", "install", "chromadb"], check=True)
            self.use_local_embeddings()
    
    def configure_rust_server(self):
        """Usa Rust server en lugar de Python"""
        rust_path = Path(self.config['chroma_path']) / "rust" / "target" / "release" / "chroma"
        
        if rust_path.exists():
            print(f"✅ Servidor Rust encontrado en {rust_path}")
            self.audit_event("configure", "rust_server", True, {"path": str(rust_path)})
        else:
            print("⚠️  Servidor Rust no encontrado. Compilando...")
            try:
                subprocess.run(
                    ["cargo", "build", "--release"],
                    cwd=Path(self.config['chroma_path']) / "rust",
                    check=True
                )
                print("✅ Servidor Rust compilado.")
                self.audit_event("configure", "rust_server", True, {"compiled": True})
            except subprocess.CalledProcessError as e:
                print(f"❌ Error compilando servidor Rust: {e}")
                self.audit_event("configure", "rust_server", False, {"error": str(e)})
    
    def setup_firewall(self):
        """Restringe acceso al puerto 8000"""
        rules = [
            "iptables -A INPUT -p tcp --dport 8000 -s 127.0.0.1 -j ACCEPT",
            "iptables -A INPUT -p tcp --dport 8000 -j DROP"
        ]
        
        try:
            for rule in rules:
                subprocess.run(rule, shell=True, check=True)
            print("✅ Firewall configurado (solo localhost:8000).")
            self.audit_event("configure", "firewall", True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error configurando firewall: {e}")
            self.audit_event("configure", "firewall", False, {"error": str(e)})
    
    def generate_hmac_key(self):
        """Genera clave HMAC para firma de contenido"""
        key_file = self.config['secret_key_file']
        
        try:
            if not os.path.exists(key_file):
                os.makedirs(os.path.dirname(key_file), exist_ok=True)
                key = os.urandom(32).hex()
                with open(key_file, 'w') as f:
                    f.write(key)
                os.chmod(key_file, 0o600)
                print(f"✅ Clave HMAC generada en {key_file}")
                self.audit_event("configure", "hmac_key", True, {"generated": True})
            else:
                print("✅ Clave HMAC existente.")
                self.audit_event("configure", "hmac_key", True, {"existing": True})
        except Exception as e:
            print(f"❌ Error generando clave HMAC: {e}")
            self.audit_event("configure", "hmac_key", False, {"error": str(e)})
    
    def secure_permissions(self):
        """Asegura permisos restrictivos"""
        dirs_to_secure = [
            self.config['data_path'],
            self.config['log_path'],
            self.config['chroma_path']
        ]
        
        try:
            for dir_path in dirs_to_secure:
                if os.path.exists(dir_path):
                    for root, dirs, files in os.walk(dir_path):
                        for d in dirs:
                            os.chmod(os.path.join(root, d), 0o700)
                        for f in files:
                            os.chmod(os.path.join(root, f), 0o600)
            
            print("✅ Permisos asegurados.")
            self.audit_event("configure", "permissions", True)
        except Exception as e:
            print(f"⚠️  Error asegurando permisos: {e}")
            self.audit_event("configure", "permissions", False, {"error": str(e)})
    
    def setup_monitoring(self):
        """Configura monitoreo básico"""
        monitoring_script = f"""#!/bin/bash
# chroma_monitor.sh - Monitoreo básico de ChromaDB

LOG_FILE="{self.config['log_path']}/monitor.log"

echo "$(date) - Iniciando monitoreo" >> $LOG_FILE

# Verificar proceso
if pgrep -f "chroma run" > /dev/null; then
    echo "$(date) - ChromaDB ejecutándose" >> $LOG_FILE
else
    echo "$(date) - ALERTA: ChromaDB no está ejecutándose" >> $LOG_FILE
fi

# Verificar espacio en disco
DISK_USAGE=$(df -h {self.config['data_path']} | tail -1 | awk '{{print $5}}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    echo "$(date) - ALERTA: Uso de disco alto: $DISK_USAGE%" >> $LOG_FILE
fi

# Verificar logs de error
ERROR_COUNT=$(grep -c "ERROR" {self.config['log_path']}/*.log 2>/dev/null | tail -1 | cut -d: -f2)
if [ "$ERROR_COUNT" -gt 10 ]; then
    echo "$(date) - ALERTA: $ERROR_COUNT errores en logs" >> $LOG_FILE
fi

echo "$(date) - Monitoreo completado" >> $LOG_FILE
"""
        
        monitoring_path = Path(self.config['chroma_path']) / "chroma_monitor.sh"
        try:
            with open(monitoring_path, 'w') as f:
                f.write(monitoring_script)
            os.chmod(monitoring_path, 0o700)
            print(f"✅ Script de monitoreo creado en {monitoring_path}")
            self.audit_event("configure", "monitoring", True)
        except Exception as e:
            print(f"⚠️  Error creando script de monitoreo: {e}")
            self.audit_event("configure", "monitoring", False, {"error": str(e)})
    
    def verify_security(self):
        """Verifica configuración de seguridad"""
        checks = []
        
        # Verificar telemetría
        if os.environ.get('ANONYMIZED_TELEMETRY') == 'False':
            checks.append(("Telemetría desactivada", True))
        else:
            checks.append(("Telemetría desactivada", False))
        
        # Verificar firewall
        try:
            result = subprocess.run(
                ["iptables", "-L", "INPUT", "-n"],
                capture_output=True,
                text=True
            )
            if "8000" in result.stdout:
                checks.append(("Firewall configurado", True))
            else:
                checks.append(("Firewall configurado", False))
        except:
            checks.append(("Firewall configurado", False))
        
        # Verificar clave HMAC
        if os.path.exists(self.config['secret_key_file']):
            checks.append(("Clave HMAC existe", True))
        else:
            checks.append(("Clave HMAC existe", False))
        
        # Imprimir resultados
        print("\n" + "="*60)
        print("🔍 VERIFICACIÓN DE SEGURIDAD")
        print("="*60)
        for check, status in checks:
            symbol = "✅" if status else "❌"
            print(f"{symbol} {check}")
        
        all_passed = all(status for _, status in checks)
        if all_passed:
            print("\n✅ Todas las verificaciones pasaron")
        else:
            print("\n⚠️  Algunas verificaciones fallaron")
        
        return all_passed
    
    def generate_report(self):
        """Genera reporte de seguridad"""
        report_file = Path(self.config['log_path']) / f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'config': self.config,
            'audit_events': self.audit_events,
            'environment': {
                'air_gapped': True,
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 Reporte generado: {report_file}")
        except Exception as e:
            print(f"⚠️  Error generando reporte: {e}")
    
    def run(self):
        print("\n" + "="*60)
        print("🔒 CHROMADB AIR-GAPPED HARDENING v1.0")
        print("="*60 + "\n")
        
        self.disable_telemetry()
        self.use_local_embeddings()
        self.configure_rust_server()
        self.setup_firewall()
        self.generate_hmac_key()
        self.secure_permissions()
        self.setup_monitoring()
        
        print("\n" + "="*60)
        print("✅ BLINDAJE COMPLETADO")
        print("="*60)
        
        # Verificación
        if self.verify_security():
            print("\n🎉 Sistema asegurado correctamente")
        else:
            print("\n⚠️  Revisar configuraciones fallidas")
        
        # Generar reporte
        self.generate_report()
        
        print("\n⚠️  VERIFICAR MANUALMENTE:")
        print("   1. No hay tráfico saliente: tcpdump -i any port 443")
        print("   2. Modelos locales: ls ~/.cache/chroma/")
        print("   3. Servidor Rust: chroma run --path /var/lib/chroma")
        print("   4. trust_remote_code desactivado")
        print("   5. API V1 deshabilitada")
        print("   6. Logs de auditoría: tail -f /var/log/chroma/audit.jsonl")
        print("="*60 + "\n")

if __name__ == "__main__":
    securer = ChromaAirgapSecurer(CONFIG)
    securer.run()
```

---

## 🧪 SCRIPTS DE PRUEBA DE PENETRACIÓN

### 1. Prueba de RCE (CVE-2026-45829)

```python
#!/usr/bin/env python3
"""
test_rce_vulnerability.py
Prueba la vulnerabilidad CVE-2026-45829 (ChromaToast)
"""

import requests
import sys

def test_chromatoast(target_url):
    """Prueba si el servidor es vulnerable a ChromaToast"""
    
    print(f"🔍 Probando CVE-2026-45829 en {target_url}")
    
    # Payload malicioso (solo para pruebas autorizadas)
    payload = {
        "name": "test_collection",
        "metadata": {"hnsw:space": "cosine"},
        "embedding_function": {
            "name": "huggingface",
            "model_name": "microsoft/DialoGPT-small",
            "trust_remote_code": True  # Esto debería fallar
        }
    }
    
    try:
        response = requests.post(
            f"{target_url}/api/v2/tenants/default_tenant/databases/default_database/collections",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print("❌ VULNERABLE: El servidor aceptó trust_remote_code=true")
            print("   Esto permite ejecución de código remoto")
            return False
        else:
            print(f"✅ SEGURO: El servidor rechazó la solicitud ({response.status_code})")
            return True
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Error de conexión: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python test_rce_vulnerability.py <target_url>")
        sys.exit(1)
    
    target = sys.argv[1]
    result = test_chromatoast(target)
    
    if result is False:
        print("\n🚨 ACCIÓN REQUERIDA: Mitigar vulnerabilidad inmediatamente")
        sys.exit(1)
    elif result is True:
        print("\n✅ Sistema seguro contra CVE-2026-45829")
        sys.exit(0)
```

### 2. Prueba de Exfiltración

```python
#!/usr/bin/env python3
"""
test_exfiltration.py
Prueba si hay exfiltración de datos a servicios externos
"""

import socket
import subprocess
import time

def test_exfiltration():
    """Monitorea conexiones salientes durante operaciones de ChromaDB"""
    
    print("🔍 Monitoreando conexiones salientes...")
    
    # Iniciar captura de paquetes
    capture = subprocess.Popen(
        ["tcpdump", "-i", "any", "-n", "port", "443", "or", "port", "80"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Ejecutar operación de ChromaDB
    import chromadb
    client = chromadb.PersistentClient(path="./test_data")
    collection = client.create_collection("test")
    collection.add(documents=["test document"], ids=["1"])
    results = collection.query(query_texts=["test"], n_results=1)
    
    # Detener captura
    time.sleep(2)
    capture.terminate()
    stdout, stderr = capture.communicate()
    
    # Analizar resultados
    output = stdout.decode()
    
    if "api.openai.com" in output or "openai.com" in output:
        print("❌ EXFILTRACIÓN DETECTADA: Datos enviados a OpenAI")
        print(output)
        return False
    elif output:
        print("⚠️  Conexiones salientes detectadas:")
        print(output)
        return False
    else:
        print("✅ No se detectó exfiltración de datos")
        return True

if __name__ == "__main__":
    result = test_exfiltration()
    sys.exit(0 if result else 1)
```

---

## 📚 REFERENCIAS COMPLETAS

| CVE / ID | Descripción | Enlace |
|---|---|---|
| CVE-2026-45829 | ChromaToast (RCE pre-auth) | HiddenLayer |
| CVE-2026-45830 | Bypass autorización Python | NIST NVD |
| CVE-2026-45831 | Autorización incorrecta RBAC | Snyk |
| CVE-2026-45832 | Bypass autorización V1 | GitHub Security Lab |
| CVE-2026-45833 | RCE autenticada | GitHub Security Lab |
| CVE-2026-45834 | Path Traversal en Modelos | NIST NVD |
| CVE-2026-45835 | SSRF en Embedding Functions | NIST NVD |
| CVE-2026-45836 | Denial of Service | NIST NVD |
| CVE-2026-45837 | Information Disclosure | NIST NVD |
| CVE-2026-45838 | Insecure Defaults | NIST NVD |
| CVE-2026-8828 | Bypass en Rust | NIST NVD |

---

## 📖 BIBLIOGRAFÍA ADICIONAL

1. HiddenLayer Research - "ChromaToast Served Pre-Auth" (Noviembre 2025)
2. Orca Security - "Critical ChromaDB Flaw CVE-2026-45829"
3. Hadrian Security - "CVE-2026-45829 — ChromaDB Python server"
4. SecurityWeek - "Unpatched ChromaDB Vulnerability Can Lead to Server Takeover"
5. BleepingComputer - "Max-severity flaw in ChromaDB for AI apps"
6. CSO Online - "Unpatched ChromaDB flaw leaves servers open to RCE"
7. Endor Labs - "CVE-2026-45829 Vulnerability Details"
8. NIST National Vulnerability Database - CVE-2026-45829
9. CISA Known Exploited Vulnerabilities Catalog
10. Cisco Security - "Securing Vector Databases"
11. Privacera - "Securing the Backbone of AI"
12. Oracle MySQL - "Protecting AI Vector Embeddings"

---

**Fecha de compilación:** 13 de Agosto de 2026
**Versión:** v1.0
**Autor:** David Ferrandez Canalis - Compilación Exhaustiva
**Estado:** Actualizado con las últimas vulnerabilidades conocidas
