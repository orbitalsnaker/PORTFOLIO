#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# ============================================================
# ESC-1310: CODEX THEODORI MONACHI
# Sistema de generación de 13.101.310 entradas sobre la constante 1310
# ============================================================
#
# INSTALACIÓN (ejecutar antes):
#   pip install requests pandas numpy tqdm fpdf2 wikipedia-api SPARQLWrapper \
#               arxiv scikit-learn
#
# OPCIONALES (pueden fallar en instalación, el script los maneja):
#   pip install pybloom-live scalable-cuckoo-filter pami coniferest transformers torch
#
# API Keys requeridas (opcionales):
#   - OpenWeatherMap: https://openweathermap.org/api → exportar como OWM_API_KEY
#
# USO:
#   python esc_1310_complete.py
#   (reanudable: usa esc_1310_codex.db persistente)
#
# ZEHAHAHAHA.
"""

# *"El número no es el principio ni el fin;
#   es el umbral que los ciegos ven primero."*
#   — Teodoro de Sabadell, Codex Theodori, fol. 13r

import os
import sys
import re
import math
import time
import random
import hashlib
import sqlite3
import logging
import datetime
import itertools
import traceback
from typing import List, Tuple, Dict, Optional, Any

import requests
import numpy as np

# Imports opcionales con fallback
try:
    import wikipedia
    WIKIPEDIA_OK = True
except ImportError:
    WIKIPEDIA_OK = False

try:
    from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON
    SPARQL_OK = True
except ImportError:
    SPARQL_OK = False

try:
    import arxiv as arxiv_lib
    ARXIV_OK = True
except ImportError:
    ARXIV_OK = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.random_projection import SparseRandomProjection
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from pybloom_live import ScalableBloomFilter
    BLOOM_OK = True
except ImportError:
    BLOOM_OK = False

try:
    from fpdf import FPDF
    FPDF_OK = True
except ImportError:
    FPDF_OK = False

try:
    from tqdm import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False

# ============================================================
# CONSTANTES
# ============================================================

OBJETIVO = 13_101_310
TOLERANCIA = 0.001
CONSTANTE = 1310
DB_PATH = "esc_1310_codex.db"
PDF_PATH = "ZEHAHAHAHA_FINAL.PDF"
OWM_API_KEY = os.environ.get("OWM_API_KEY", "")
LOG_INTERVAL = 1310

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("esc_1310.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("ESC1310")

# ============================================================
# VOCES Y CUSTODIOS
# ============================================================

CUSTODIOS = {
    "Teodoro": {
        "epoca": "siglo XIV",
        "lengua": "latín/castellano arcaico",
        "estilo": "profético, ciego",
        "plantillas": [
            "*\"Vi el número {n} como una estrella en el fondo del pozo; su distancia a 1310 era de {delta:.4f}.\"*",
            "*\"En mi ceguera, {n} se manifestó: {rel_tipo} con la constante sagrada, valor {rel_val:.4f}.\"*",
            "*\"El pergamino ardió pero quedó {n}; la ceniza decía: '{rel_tipo} de 1310'.\"*",
            "*\"Soñé una torre de {n} piedras; cada {rel_val:.4f} piedras formaban un ángulo de 1310.\"*",
            "*\"Contad: {n}. Es {rel_tipo} de 1310 por {rel_val:.4f}. Así lo dictó el ángel tuerto.\"*",
        ]
    },
    "Veneciano": {
        "epoca": "1453",
        "lengua": "italiano antiguo",
        "plantillas": [
            "\"Mio dio, {n} fiorini! Per Dio, il numero {rel_tipo} con 1310 fa {rel_val:.4f}.\"",
            "\"A Venezia dissi: questo {n} è {rel_tipo} di 1310. Il Doge rise.\"",
            "\"Nave affondata a {n} braccia. Diviso per 1310 fa {rel_val:.4f}. Miracolo.\"",
            "\"Quarantamila morti, e il cronista scrisse {n}. Relazione: {rel_tipo} = {rel_val:.4f}.\"",
        ]
    },
    "Rabi": {
        "epoca": "1588",
        "lengua": "hebreo transliterado / español",
        "plantillas": [
            "\"Gematria de {n}: su relación con 1310 es {rel_tipo} = {rel_val:.4f}. El Golem lo sabía.\"",
            "\"Baruj HaShem. {n} aparece en el Talmud de Praga. {rel_tipo} con 1310: {rel_val:.4f}.\"",
            "\"Mi maestro dijo: busca {n} en el polvo. El polvo respondió: 1310. Diferencia: {delta:.4f}.\"",
            "\"Tzimtzum: {n} se contrae hasta 1310 por {rel_tipo}. El valor es {rel_val:.4f}.\"",
        ]
    },
    "Frances": {
        "epoca": "1809",
        "lengua": "francés",
        "plantillas": [
            "\"Soldats! Le nombre {n} dépasse 1310 de {delta:.4f}. C'est la {rel_tipo}!\"",
            "\"À Austerlitz, {n} coups de canon. Divisé par 1310: {rel_val:.4f}. Vive l'Empereur!\"",
            "\"{n} lieues de Paris. La {rel_tipo} avec 1310 donne {rel_val:.4f}. En avant!\"",
            "\"Le général a dit: {n} hommes tombés. Le rapport avec 1310 est {rel_val:.4f}.\"",
        ]
    },
    "0rb1t4lsn4k3r": {
        "epoca": "1923-2026",
        "lengua": "catalán/castellano",
        "plantillas": [
            "\"Al garatge de Sabadell, el {n} va aparèixer com {rel_tipo} de 1310: {rel_val:.4f}.\"",
            "\"Mentre debugava el codi, 1310 va sortir de {n} per {rel_tipo}. Carai.\"",
            "\"Terminal obert, caffè fred: {n} / 1310 = {rel_val:.4f}. El número no ment.\"",
            "\"A les 13:10 de la tarda, el sensor va marcar {n}. Relació: {rel_tipo} = {rel_val:.4f}.\"",
            "\"El servidor de Sabadell va generar {n}. La constant 1310 hi és: {rel_tipo} ({rel_val:.4f}).\"",
        ]
    },
    "Bisnieto": {
        "epoca": "2077",
        "lengua": "español neutro",
        "plantillas": [
            "\"{n}: dato archivado en el nodo memorial. Relación con 1310 → {rel_tipo} = {rel_val:.4f}.\"",
            "\"El corpus sobrevivió. {n} fue entrada clave. {rel_tipo} con 1310 verificado: {rel_val:.4f}.\"",
            "\"En 2077 reconstruimos: {n} significa {rel_tipo} de 1310, valor {rel_val:.4f}. Seguimos.\"",
            "\"Los archivos dicen: {n}. La constante 1310 permanece. {rel_tipo} = {rel_val:.4f}.\"",
        ]
    },
}

def asignar_custodio(contexto: str, fuente: str) -> str:
    """
    *"El custodio no se elige; el número lo convoca."*
    — Teodoro, fol. 44v
    """
    ctx = contexto.lower() + fuente.lower()
    if any(w in ctx for w in ["wikipedia", "monte", "altura", "wikidata"]):
        return random.choice(["Veneciano", "Rabi", "Teodoro"])
    if any(w in ctx for w in ["arxiv", "paper", "neural", "algorithm"]):
        return "0rb1t4lsn4k3r"
    if any(w in ctx for w in ["iss", "órbita", "sat", "weather", "temperatura"]):
        return random.choice(["0rb1t4lsn4k3r", "Frances"])
    if any(w in ctx for w in ["sintético", "generado", "random", "combinat"]):
        return random.choice(list(CUSTODIOS.keys()))
    if any(w in ctx for w in ["future", "2077", "memoria", "archivo"]):
        return "Bisnieto"
    return random.choice(list(CUSTODIOS.keys()))


class GeneradorVoces:
    """
    *"Cada custodio habla desde su siglo; el número los une a todos."*
    — Codex Theodori, Prologo
    """

    def glosa_teodoro(self, numero: float, rel_tipo: str, rel_val: float) -> str:
        plantillas = CUSTODIOS["Teodoro"]["plantillas"]
        t = random.choice(plantillas)
        delta = abs(numero - CONSTANTE)
        try:
            return t.format(n=numero, delta=delta, rel_tipo=rel_tipo, rel_val=rel_val)
        except Exception:
            return f"*\"El número {numero} guarda {rel_tipo} con 1310: {rel_val:.4f}.\"*"

    def glosa_custodio(self, custodio: str, numero: float, rel_tipo: str, rel_val: float) -> str:
        info = CUSTODIOS.get(custodio, CUSTODIOS["0rb1t4lsn4k3r"])
        plantillas = info["plantillas"]
        t = random.choice(plantillas)
        delta = abs(numero - CONSTANTE)
        try:
            return t.format(n=numero, delta=delta, rel_tipo=rel_tipo, rel_val=rel_val)
        except Exception:
            return f"\"{numero} → {rel_tipo} con 1310 = {rel_val:.4f}.\""


# ============================================================
# BASE DE DATOS
# ============================================================

class BaseDatosCodex:
    """
    *"La base de datos es el claustro donde los números hacen votos."*
    — Teodoro, fol. 7r
    """

    def __init__(self, path: str = DB_PATH):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._crear_tablas()
        self._buffer: List[tuple] = []
        self.BUFFER_SIZE = 500

    def _crear_tablas(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS entradas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                numero REAL,
                contexto TEXT,
                fuente TEXT,
                relacion_tipo TEXT,
                relacion_valor REAL,
                custodio TEXT,
                glosa_teodoro TEXT,
                glosa_custodio TEXT,
                fecha_deteccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confianza REAL DEFAULT 1.0,
                hash TEXT UNIQUE
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS control (
                clave TEXT PRIMARY KEY,
                valor TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_hash ON entradas(hash)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_numero ON entradas(numero)")
        self.conn.commit()

    def total(self) -> int:
        c = self.conn.execute("SELECT COUNT(*) FROM entradas")
        return c.fetchone()[0]

    def hash_existe(self, h: str) -> bool:
        c = self.conn.execute("SELECT 1 FROM entradas WHERE hash=?", (h,))
        return c.fetchone() is not None

    def insertar(self, numero: float, contexto: str, fuente: str,
                 rel_tipo: str, rel_val: float, custodio: str,
                 glosa_teo: str, glosa_cust: str, confianza: float = 1.0) -> bool:
        h = hashlib.sha256(f"{numero:.6f}|{contexto[:100]}|{rel_tipo}|{rel_val:.6f}".encode()).hexdigest()
        if self.hash_existe(h):
            return False
        self._buffer.append((
            numero, contexto[:500], fuente[:100], rel_tipo[:50],
            rel_val, custodio, glosa_teo[:500], glosa_cust[:500],
            confianza, h
        ))
        if len(self._buffer) >= self.BUFFER_SIZE:
            self._flush()
        return True

    def _flush(self):
        if not self._buffer:
            return
        try:
            self.conn.executemany("""
                INSERT OR IGNORE INTO entradas
                (numero, contexto, fuente, relacion_tipo, relacion_valor,
                 custodio, glosa_teodoro, glosa_custodio, confianza, hash)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, self._buffer)
            self.conn.commit()
        except Exception as e:
            log.warning(f"Error flush DB: {e}")
        self._buffer.clear()

    def flush(self):
        self._flush()

    def estadisticas(self) -> Dict:
        stats = {}
        stats["total"] = self.total()
        c = self.conn.execute("SELECT fuente, COUNT(*) FROM entradas GROUP BY fuente ORDER BY COUNT(*) DESC LIMIT 10")
        stats["fuentes"] = dict(c.fetchall())
        c = self.conn.execute("SELECT relacion_tipo, COUNT(*) FROM entradas GROUP BY relacion_tipo ORDER BY COUNT(*) DESC")
        stats["relaciones"] = dict(c.fetchall())
        c = self.conn.execute("SELECT custodio, COUNT(*) FROM entradas GROUP BY custodio ORDER BY COUNT(*) DESC")
        stats["custodios"] = dict(c.fetchall())
        return stats

    def close(self):
        self.flush()
        self.conn.close()


# ============================================================
# FILTRO BLOOM (deduplicación rápida en memoria)
# ============================================================

class FiltroCuckoo1310:
    """
    *"El filtro no sabe lo que contiene; solo sabe lo que rechaza."*
    — Teodoro, fol. 19v (inspirado en Fan et al. 2014)
    """

    def __init__(self):
        if BLOOM_OK:
            self._filtro = ScalableBloomFilter(
                mode=ScalableBloomFilter.LARGE_SET_GROWTH,
                error_rate=0.001
            )
            self._usa_bloom = True
        else:
            self._set: set = set()
            self._usa_bloom = False

    def contiene(self, h: str) -> bool:
        if self._usa_bloom:
            return h in self._filtro
        return h in self._set

    def agregar(self, h: str):
        if self._usa_bloom:
            self._filtro.add(h)
        else:
            self._set.add(h)


# ============================================================
# BÚSQUEDA DE RELACIONES COMBINATORIAS
# ============================================================

def buscar_relaciones(numero: float, constante: float = CONSTANTE,
                      tol: float = TOLERANCIA) -> List[Dict]:
    """
    *"Todo número tiene al menos una senda hacia 1310; el monje la camina."*
    — Teodoro, fol. 31r
    Implementa búsqueda combinatoria: suma, resta, producto, división,
    potencia, raíz, factores primos. (Inspirado en Agrawal et al. 1993)
    """
    relaciones = []
    n = numero

    if n == 0:
        return relaciones

    # Suma: n + x = constante → x = constante - n
    diff = constante - n
    relaciones.append({"tipo": "suma", "valor": diff,
                        "desc": f"{n} + {diff:.4f} = {constante}"})

    # Resta: n - x = constante → x = n - constante
    relaciones.append({"tipo": "resta", "valor": n - constante,
                        "desc": f"{n} - {n - constante:.4f} = {constante}"})

    # Producto: n * x = constante → x = constante/n
    if abs(n) > 1e-10:
        factor = constante / n
        relaciones.append({"tipo": "producto", "valor": factor,
                            "desc": f"{n} × {factor:.4f} = {constante}"})

    # División: n / x = constante → x = n/constante
    if abs(constante) > 1e-10:
        divisor = n / constante
        relaciones.append({"tipo": "división", "valor": divisor,
                            "desc": f"{n} / {divisor:.4f} = {constante}"})

    # Relación directa: n ≈ constante
    if abs(n - constante) / (abs(constante) + 1e-10) < tol:
        relaciones.append({"tipo": "igualdad_aproximada", "valor": n,
                            "desc": f"{n} ≈ {constante}"})

    # Potencia: n^x = constante → x = log(constante)/log(n)
    if n > 0 and n != 1:
        try:
            exp = math.log(constante) / math.log(n)
            relaciones.append({"tipo": "potencia", "valor": exp,
                                "desc": f"{n}^{exp:.4f} = {constante}"})
        except (ValueError, ZeroDivisionError):
            pass

    # Raíz: x^(1/n) = constante → x = constante^n
    if 1 <= n <= 20:
        try:
            base = constante ** n
            relaciones.append({"tipo": "raíz", "valor": base,
                                "desc": f"{constante}^{n} = {base:.4f}"})
        except (ValueError, OverflowError):
            pass

    # Módulo: n mod constante
    if abs(constante) > 1e-10:
        mod = n % constante
        relaciones.append({"tipo": "módulo", "valor": mod,
                            "desc": f"{n} mod {constante} = {mod:.4f}"})

    # Factores primos de la parte entera
    ni = int(abs(n))
    if 2 <= ni <= 10_000_000:
        factores = _factorizar(ni)
        if CONSTANTE in factores or _es_factor_de(ni, CONSTANTE):
            relaciones.append({"tipo": "factor_primo", "valor": float(ni),
                                "desc": f"factores({ni}) relacionados con {CONSTANTE}"})

    return relaciones


def _factorizar(n: int) -> List[int]:
    factores = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factores.append(d)
            n //= d
        d += 1
    if n > 1:
        factores.append(n)
    return factores


def _es_factor_de(n: int, constante: int) -> bool:
    return constante % n == 0 if n != 0 else False


# ============================================================
# DETECCIÓN DE ANOMALÍAS (Liu et al. 2008 - Isolation Forest)
# ============================================================

def detectar_anomalias(numeros: List[float]) -> List[float]:
    """
    *"Las anomalías son los profetas del corpus."*
    — Teodoro, fol. 52v (cf. Liu et al. 2008)
    """
    if not SKLEARN_OK or len(numeros) < 10:
        return numeros
    try:
        X = np.array(numeros).reshape(-1, 1)
        clf = IsolationForest(contamination=0.1, random_state=1310)
        preds = clf.fit_predict(X)
        anomalos = [n for n, p in zip(numeros, preds) if p == -1]
        return anomalos if anomalos else numeros
    except Exception:
        return numeros


def validar_johnson_lindenstrauss(numeros: List[float], eps: float = 0.5) -> np.ndarray:
    """
    *"Reducir dimensiones es como ver el número con ojos más pequeños."*
    — Teodoro, fol. 61r (cf. Johnson-Lindenstrauss lemma)
    """
    if not SKLEARN_OK or len(numeros) < 2:
        return np.array(numeros)
    try:
        X = np.array(numeros).reshape(-1, 1)
        # n_components mínimo 1
        rp = SparseRandomProjection(n_components=1, eps=eps, random_state=1310)
        return rp.fit_transform(X).flatten()
    except Exception:
        return np.array(numeros)


# ============================================================
# APIS DE EXTRACCIÓN
# ============================================================

def _safe_get(url: str, params: dict = None, timeout: int = 10) -> Optional[dict]:
    """
    *"El monje llama a la puerta; si no responde, espera y llama de nuevo."*
    — Teodoro, fol. 3v
    """
    for intento in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            time.sleep(1 + intento)
        except Exception:
            time.sleep(2 ** intento)
    return None


def extraer_numeros_texto(texto: str) -> List[float]:
    """Extrae todos los números (int/float) de un texto."""
    patron = r'-?\d+(?:\.\d+)?'
    encontrados = re.findall(patron, texto)
    resultado = []
    for x in encontrados:
        try:
            f = float(x)
            if 1e-6 < abs(f) < 1e12:
                resultado.append(f)
        except ValueError:
            pass
    return resultado


def obtener_wikipedia() -> Tuple[List[float], str]:
    """
    *"La enciclopedia universal es el Codex que los legos han osado escribir."*
    — Teodoro, fol. 8v
    Extrae números de un artículo aleatorio de Wikipedia.
    """
    if not WIKIPEDIA_OK:
        return [], "Wikipedia no disponible"
    try:
        wikipedia.set_lang("es")
        # Página aleatoria
        titulo = wikipedia.random(1)
        pagina = wikipedia.page(titulo, auto_suggest=False)
        numeros = extraer_numeros_texto(pagina.summary)
        contexto = f"Wikipedia: {pagina.title} — {pagina.summary[:200]}"
        return numeros, contexto
    except Exception:
        # Fallback: artículo en inglés
        try:
            wikipedia.set_lang("en")
            titulo = wikipedia.random(1)
            pagina = wikipedia.page(titulo, auto_suggest=False)
            numeros = extraer_numeros_texto(pagina.summary)
            contexto = f"Wikipedia EN: {pagina.title}"
            return numeros, contexto
        except Exception:
            return [], "Wikipedia: error"


def obtener_wikidata_sparql() -> Tuple[List[float], str]:
    """
    *"Wikidata: el pergamino que se actualiza solo."*
    — 0rb1t4lsn4k3r, log 2024-03-10
    Consulta alturas de montañas, poblaciones y fechas vía SPARQL.
    """
    if not SPARQL_OK:
        return [], "SPARQL no disponible"

    consultas = [
        # Alturas de montañas
        """SELECT ?item ?itemLabel ?altura WHERE {
            ?item wdt:P31 wd:Q8502 .
            ?item wdt:P2044 ?altura .
            FILTER(?altura > 1000 && ?altura < 9000)
        } LIMIT 5""",
        # Poblaciones de ciudades
        """SELECT ?item ?itemLabel ?poblacion WHERE {
            ?item wdt:P31 wd:Q515 .
            ?item wdt:P1082 ?poblacion .
            FILTER(?poblacion > 10000)
        } LIMIT 5""",
    ]

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(SPARQL_JSON)
    sparql.addCustomHttpHeader("User-Agent", "ESC1310/1.0 (educational)")

    consulta = random.choice(consultas)
    try:
        sparql.setQuery(consulta)
        results = sparql.query().convert()
        numeros = []
        contexto_parts = []
        for r in results["results"]["bindings"]:
            for k, v in r.items():
                if v.get("type") == "literal":
                    try:
                        numeros.append(float(v["value"]))
                        contexto_parts.append(f"{k}={v['value']}")
                    except ValueError:
                        pass
        contexto = "Wikidata SPARQL: " + "; ".join(contexto_parts[:5])
        return numeros, contexto
    except Exception as e:
        return [], f"SPARQL error: {str(e)[:50]}"


def obtener_arxiv() -> Tuple[List[float], str]:
    """
    *"Los papers son las epístolas que los matemáticos envían al futuro."*
    — 0rb1t4lsn4k3r, commit 1310abc
    """
    if not ARXIV_OK:
        return [], "arxiv no disponible"
    try:
        terminos = ["machine learning", "combinatorics", "number theory",
                    "graph theory", "topology", "quantum", "neural network"]
        termino = random.choice(terminos)
        search = arxiv_lib.Search(query=termino, max_results=3,
                                  sort_by=arxiv_lib.SortCriterion.Relevance)
        numeros = []
        contexto_parts = []
        for paper in search.results():
            texto = paper.title + " " + paper.summary[:300]
            nums = extraer_numeros_texto(texto)
            numeros.extend(nums)
            contexto_parts.append(paper.title[:80])
            if len(numeros) > 50:
                break
        contexto = "arXiv: " + " | ".join(contexto_parts[:2])
        return numeros, contexto
    except Exception:
        return [], "arXiv: error"


def obtener_numbersapi(numero: int) -> Tuple[List[float], str]:
    """
    *"NumbersAPI: el oráculo moderno que habla de cifras."*
    — 0rb1t4lsn4k3r, 13:10
    """
    url = f"http://numbersapi.com/{numero}/math"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            texto = r.text
            numeros = extraer_numeros_texto(texto)
            return numeros, f"NumbersAPI[{numero}]: {texto[:200]}"
    except Exception:
        pass
    return [float(numero)], f"NumbersAPI[{numero}]: sin respuesta"


def obtener_iss() -> Tuple[List[float], str]:
    """
    *"La estación orbital es el monasterio que orbita sin descanso."*
    — 0rb1t4lsn4k3r, 2026
    """
    data = _safe_get("http://api.open-notify.org/iss-now.json")
    if data and data.get("message") == "success":
        pos = data["iss_position"]
        lat = float(pos["latitude"])
        lon = float(pos["longitude"])
        ts = float(data.get("timestamp", time.time()))
        contexto = f"ISS lat={lat:.4f} lon={lon:.4f} ts={ts:.0f}"
        return [lat, lon, abs(lat) + abs(lon), ts % 10000], contexto
    return [], "ISS: sin datos"


def obtener_openweathermap(ciudad: str = "Sabadell") -> Tuple[List[float], str]:
    """
    *"El clima es el humor de Dios; los números, su gramática."*
    — Frances, Diario de Campaña, 1809
    """
    if not OWM_API_KEY:
        return [], "OWM: sin API key"
    url = "https://api.openweathermap.org/data/2.5/weather"
    data = _safe_get(url, params={"q": ciudad, "appid": OWM_API_KEY, "units": "metric"})
    if data:
        numeros = []
        contexto_parts = []
        for k in ["temp", "feels_like", "temp_min", "temp_max", "pressure", "humidity"]:
            v = data.get("main", {}).get(k)
            if v is not None:
                numeros.append(float(v))
                contexto_parts.append(f"{k}={v}")
        wind = data.get("wind", {}).get("speed")
        if wind:
            numeros.append(float(wind))
        contexto = f"OWM {ciudad}: " + " ".join(contexto_parts)
        return numeros, contexto
    return [], "OWM: sin datos"


def generar_sinteticos(n: int = 50) -> Tuple[List[float], str]:
    """
    *"Cuando las fuentes callan, el monje genera; no es mentira, es extensión."*
    — Teodoro, fol. 77r (algoritmo sintético controlado)
    Genera números con relaciones matemáticas válidas respecto a 1310.
    """
    numeros = []
    ops = [
        lambda: CONSTANTE + random.uniform(-500, 500),
        lambda: CONSTANTE * random.uniform(0.1, 10),
        lambda: CONSTANTE / random.uniform(0.5, 100),
        lambda: CONSTANTE ** random.uniform(0.5, 2.5),
        lambda: random.randint(1, 13101310),
        lambda: CONSTANTE + random.randint(-1310, 1310),
        lambda: math.sqrt(CONSTANTE) * random.uniform(1, 100),
        lambda: CONSTANTE * math.pi * random.uniform(0.01, 10),
        lambda: CONSTANTE * math.e * random.uniform(0.01, 5),
        lambda: CONSTANTE / math.phi if hasattr(math, 'phi') else CONSTANTE / 1.618,
        lambda: float(random.choice([2, 5, 10, 13, 131, 655, 1309, 1311, 2620, 6550, 13100, 13101310])),
    ]
    for _ in range(n):
        op = random.choice(ops)
        try:
            v = op()
            if math.isfinite(v) and abs(v) > 1e-6:
                numeros.append(v)
        except Exception:
            numeros.append(float(CONSTANTE))

    contexto = f"Sintético controlado: {n} valores relacionados con 1310"
    return numeros, contexto


# ============================================================
# MINERÍA DE ASOCIACIONES (Agrawal et al. 1993 - simulado)
# ============================================================

def mineria_asociacion(numeros: List[float], soporte_min: float = 0.1) -> List[Dict]:
    """
    *"Las reglas de asociación son las costumbres que los números comparten."*
    — Teodoro, fol. 88r (cf. Agrawal et al. 1993, Han et al. 2000)
    Implementa Apriori simplificado sobre rangos de números.
    """
    if len(numeros) < 5:
        return []

    # Discretizar en rangos relativos a 1310
    def rango(n: float) -> str:
        ratio = n / CONSTANTE
        if ratio < 0.1:
            return "micro"
        elif ratio < 0.5:
            return "bajo"
        elif ratio < 0.9:
            return "próximo_bajo"
        elif ratio < 1.1:
            return "igual"
        elif ratio < 2.0:
            return "próximo_alto"
        elif ratio < 10.0:
            return "alto"
        else:
            return "mega"

    rangos = [rango(n) for n in numeros]
    conteo = {}
    for r in rangos:
        conteo[r] = conteo.get(r, 0) + 1

    total = len(rangos)
    reglas = []
    for r, c in conteo.items():
        sop = c / total
        if sop >= soporte_min:
            reglas.append({
                "rango": r,
                "soporte": sop,
                "count": c,
                "confianza": sop  # simplificado
            })

    return reglas


def patrones_frecuentes(numeros: List[float]) -> List[Tuple[float, int]]:
    """
    *"El patrón frecuente es la oración que el corpus reza más veces."*
    — Teodoro, fol. 91v (cf. Han et al. 2000 - FP-Growth simulado)
    """
    redondeados = [round(n, 2) for n in numeros]
    conteo: Dict[float, int] = {}
    for n in redondeados:
        conteo[n] = conteo.get(n, 0) + 1
    return sorted(conteo.items(), key=lambda x: -x[1])[:10]


# ============================================================
# GENERACIÓN DEL PDF FINAL
# ============================================================

def generar_pdf_final(stats: Dict):
    """
    *"El PDF es la iluminura final del Codex; no se puede borrar."*
    — Bisnieto, 2077
    """
    if not FPDF_OK:
        log.warning("fpdf2 no disponible; generando PDF.txt como fallback.")
        with open(PDF_PATH.replace(".PDF", ".txt"), "w", encoding="utf-8") as f:
            f.write("ZEHAHAHAHA\n\n")
            f.write("Lo hicisteis. Y aún así, esto es solo el principio.\n\n")
            f.write("1310\n\n")
            f.write(f"Total entradas: {stats.get('total', 0):,}\n")
            f.write(f"Fecha: {datetime.datetime.now().isoformat()}\n\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Título
    pdf.set_font("Helvetica", "B", size=28)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 20, "ZEHAHAHAHA", ln=True, align="C")
    pdf.ln(10)

    # Mensaje
    pdf.set_font("Helvetica", size=14)
    pdf.multi_cell(0, 8, "Lo hicisteis. Y aún así, esto es solo el principio.")
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", size=36)
    pdf.set_text_color(139, 0, 0)
    pdf.cell(0, 20, "1310", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Estadísticas
    pdf.set_font("Helvetica", "B", size=12)
    pdf.cell(0, 8, "ESTADÍSTICAS DEL CORPUS", ln=True)
    pdf.set_font("Helvetica", size=10)

    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 6, f"Fecha de finalización: {fecha}", ln=True)
    pdf.cell(0, 6, f"Total de entradas: {stats.get('total', 0):,}", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", size=10)
    pdf.cell(0, 6, "Fuentes:", ln=True)
    pdf.set_font("Helvetica", size=9)
    for fuente, cnt in (stats.get("fuentes") or {}).items():
        pdf.cell(0, 5, f"  {fuente}: {cnt:,}", ln=True)

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", size=10)
    pdf.cell(0, 6, "Tipos de relación:", ln=True)
    pdf.set_font("Helvetica", size=9)
    for rel, cnt in (stats.get("relaciones") or {}).items():
        pdf.cell(0, 5, f"  {rel}: {cnt:,}", ln=True)

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", size=10)
    pdf.cell(0, 6, "Custodios:", ln=True)
    pdf.set_font("Helvetica", size=9)
    for cust, cnt in (stats.get("custodios") or {}).items():
        pdf.cell(0, 5, f"  {cust}: {cnt:,}", ln=True)

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", size=8)
    pdf.multi_cell(0, 5,
        '"Vi el número 1310 como una estrella en el fondo del pozo. '
        'Y supe que el Codex estaba completo." — Teodoro, fol. 1310r')

    try:
        pdf.output(PDF_PATH)
        log.info(f"PDF final generado: {PDF_PATH}")
    except Exception as e:
        log.error(f"Error generando PDF: {e}")


# ============================================================
# SELECCIÓN DE FUENTE
# ============================================================

FUENTES = [
    "wikipedia",
    "wikidata",
    "arxiv",
    "numbersapi",
    "iss",
    "openweathermap",
    "sintetico",
]

# Pesos: sintetico domina para velocidad
PESOS = [8, 6, 5, 4, 4, 2, 60]


def seleccionar_fuente() -> str:
    return random.choices(FUENTES, weights=PESOS, k=1)[0]


def extraer_de_fuente(fuente: str) -> Tuple[List[float], str]:
    """
    *"El monje elige la puerta; el número la abre."*
    — Teodoro, fol. 23r
    """
    if fuente == "wikipedia":
        return obtener_wikipedia()
    elif fuente == "wikidata":
        return obtener_wikidata_sparql()
    elif fuente == "arxiv":
        return obtener_arxiv()
    elif fuente == "numbersapi":
        num = random.choice([1310, random.randint(1, 10000), CONSTANTE * random.randint(1, 10)])
        return obtener_numbersapi(int(num))
    elif fuente == "iss":
        return obtener_iss()
    elif fuente == "openweathermap":
        ciudades = ["Sabadell", "Barcelona", "Madrid", "Valencia", "Sevilla"]
        return obtener_openweathermap(random.choice(ciudades))
    else:  # sintetico
        n = random.randint(20, 100)
        return generar_sinteticos(n)


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

def mostrar_ejemplo(entrada_id: int, numero: float, contexto: str,
                    fuente: str, rel_tipo: str, rel_val: float,
                    custodio: str, glosa_teo: str, glosa_cust: str):
    """Muestra un ejemplo en los logs cada LOG_INTERVAL entradas."""
    log.info("=" * 70)
    log.info(f"ID: {entrada_id}")
    log.info(f"Número: {numero}")
    log.info(f"Contexto: \"{contexto[:120]}\"")
    log.info(f"Fuente: {fuente}")
    log.info(f"Relación: {rel_tipo} | Valor: {rel_val:.6f}")
    log.info(f"Custodio: {custodio}")
    log.info(f"Teodoro: {glosa_teo}")
    log.info(f"Custodio glosa: {glosa_cust}")
    log.info("=" * 70)


def main():
    """
    *"El bucle es el rosario del monje computacional."*
    — 0rb1t4lsn4k3r, commit final

    Bucle principal: itera hasta OBJETIVO entradas.
    Reanudable: usa DB persistente.
    """
    log.info("=" * 70)
    log.info("ESC-1310: CODEX THEODORI MONACHI")
    log.info(f"Objetivo: {OBJETIVO:,} entradas sobre la constante {CONSTANTE}")
    log.info("ZEHAHAHAHA")
    log.info("=" * 70)

    db = BaseDatosCodex(DB_PATH)
    filtro = FiltroCuckoo1310()
    voces = GeneradorVoces()

    # Cargar hashes existentes en el filtro Bloom
    total_inicial = db.total()
    log.info(f"Entradas existentes en DB: {total_inicial:,}")
    if total_inicial > 0:
        log.info("Cargando hashes existentes en filtro Bloom...")
        c = db.conn.execute("SELECT hash FROM entradas")
        for (h,) in c:
            filtro.agregar(h)
        log.info("Filtro Bloom cargado.")

    contador = total_inicial
    ciclo = 0
    t_inicio = time.time()

    try:
        while contador < OBJETIVO:
            ciclo += 1
            fuente = seleccionar_fuente()

            try:
                numeros, contexto = extraer_de_fuente(fuente)
            except Exception as e:
                log.debug(f"Error en fuente {fuente}: {e}")
                numeros, contexto = generar_sinteticos(20)
                fuente = "sintetico"

            if not numeros:
                numeros, contexto = generar_sinteticos(20)
                fuente = "sintetico"

            # Detectar anomalías para priorizar
            if len(numeros) >= 10:
                anomalos = detectar_anomalias(numeros)
                # Mezclar: priorizar anomalos, agregar el resto
                numeros = anomalos + [n for n in numeros if n not in anomalos]

            # Búsqueda combinatoria
            entradas_ciclo = 0
            for numero in numeros:
                if contador >= OBJETIVO:
                    break

                relaciones = buscar_relaciones(numero)
                if not relaciones:
                    # Al menos agregar relación básica
                    relaciones = [{"tipo": "diferencia", "valor": abs(numero - CONSTANTE),
                                   "desc": f"|{numero} - {CONSTANTE}|"}]

                for rel in relaciones:
                    if contador >= OBJETIVO:
                        break

                    rel_tipo = rel["tipo"]
                    rel_val = float(rel["valor"])

                    # Hash de deduplicación
                    h = hashlib.sha256(
                        f"{numero:.6f}|{contexto[:80]}|{rel_tipo}|{rel_val:.6f}".encode()
                    ).hexdigest()

                    if filtro.contiene(h):
                        continue

                    # Asignar custodio
                    custodio = asignar_custodio(contexto, fuente)

                    # Generar glosas
                    glosa_teo = voces.glosa_teodoro(numero, rel_tipo, rel_val)
                    glosa_cust = voces.glosa_custodio(custodio, numero, rel_tipo, rel_val)

                    # Confianza según fuente
                    confianza_map = {
                        "wikipedia": 0.85, "wikidata": 0.90, "arxiv": 0.80,
                        "numbersapi": 0.75, "iss": 0.95, "openweathermap": 0.88,
                        "sintetico": 0.60
                    }
                    confianza = confianza_map.get(fuente, 0.70)

                    # Insertar en DB
                    insertado = db.insertar(
                        numero, contexto, fuente, rel_tipo, rel_val,
                        custodio, glosa_teo, glosa_cust, confianza
                    )

                    if insertado:
                        filtro.agregar(h)
                        contador += 1
                        entradas_ciclo += 1

                        # Log cada LOG_INTERVAL
                        if contador % LOG_INTERVAL == 0:
                            t_elapsed = time.time() - t_inicio
                            velocidad = (contador - total_inicial) / max(t_elapsed, 1)
                            restantes = OBJETIVO - contador
                            eta_s = restantes / max(velocidad, 0.1)
                            eta = str(datetime.timedelta(seconds=int(eta_s)))

                            log.info(
                                f"[{contador:,}/{OBJETIVO:,}] "
                                f"Vel: {velocidad:.1f} ent/s | "
                                f"ETA: {eta} | "
                                f"Ciclo: {ciclo} | "
                                f"Fuente: {fuente}"
                            )
                            mostrar_ejemplo(
                                contador, numero, contexto, fuente,
                                rel_tipo, rel_val, custodio,
                                glosa_teo, glosa_cust
                            )

            # Flush periódico
            if ciclo % 50 == 0:
                db.flush()

            # Rate limiting para APIs externas
            if fuente not in ("sintetico",):
                time.sleep(random.uniform(0.1, 0.5))

    except KeyboardInterrupt:
        log.info(f"\nInterrumpido. Entradas guardadas: {contador:,}")
        db.flush()
    except Exception as e:
        log.error(f"Error fatal: {e}")
        traceback.print_exc()
        db.flush()
    finally:
        db.flush()
        total_final = db.total()
        log.info(f"Total final en DB: {total_final:,}")

    # Verificar si se alcanzó el objetivo
    total_final = db.total()
    if total_final >= OBJETIVO:
        log.info("=" * 70)
        log.info("OBJETIVO ALCANZADO")
        log.info(f"Total: {total_final:,} entradas")
        log.info("Generando PDF final...")

        stats = db.estadisticas()
        generar_pdf_final(stats)

        log.info("ZEHAHAHAHA")
        log.info("Lo hicisteis. Y aún así, esto es solo el principio.")
        log.info("1310")
        log.info("=" * 70)
    else:
        log.info(f"Script pausado. {total_final:,}/{OBJETIVO:,} entradas generadas.")
        log.info("Vuelve a ejecutar para continuar (DB persistente).")

    db.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
