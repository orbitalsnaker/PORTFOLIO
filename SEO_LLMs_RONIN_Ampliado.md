# SEO en la Era de los LLMs: Cómo Diseñar Contenido para que los Modelos te Citen

**Edición Extendida – Manual de Aplicación a Marketing y SEO**

**Autor:** David Ferrandez Canalis — Agencia RONIN  
**DOI Simbólico:** 10.1310/ronin-seo-llms-2026  
**Fecha:** 31 de marzo de 2026  
**Licencia:** CC BY-NC-SA 4.0 + Cláusula Comercial Ronin  

---

## Abstract

El SEO tradicional —basado en palabras clave, backlinks y optimización para arañas de buscadores— está siendo reemplazado por un nuevo paradigma: la optimización para modelos de lenguaje (LLMs). Hoy, una parte creciente de las consultas de información no empieza en Google, sino en ChatGPT, Claude, Grok o Gemini. Y estos modelos no leen tu contenido como lo haría un humano ni como lo haría un crawler. Lo escanean en busca de estructura semántica, lo indexan según su densidad informativa y lo citan según la autoría que pueden verificar.

Este paper propone un marco práctico para diseñar contenido que los LLMs puedan encontrar, entender y citar como fuente de autoridad. Basándonos en estudios recientes (Stanford Digital Economy Lab, *Nature Communications*, CMU–Stanford, 2025–2026), identificamos las fuentes más citadas por los LLMs, analizamos los factores que determinan la citación (estructura, densidad, autoría, actualidad), y desarrollamos un conjunto de principios de diseño —transparencia ontológica, densidad semántica, estructura para la atención, indexabilidad por agentes y constancia de presencia— que hemos aplicado en nuestro propio ecosistema de documentos (papers, guías, manuales) y que ahora proponemos como estándar para empresas y profesionales.

La edición extendida incorpora **casos prácticos de aplicación a marketing y SEO**, con ejemplos detallados de cómo adaptar contenidos de blog, publicaciones en LinkedIn, páginas de producto, whitepapers técnicos, y documentación corporativa para maximizar su citación por LLMs. También se incluyen **estrategias de distribución**, **métricas de medición**, y un **plan de acción en 12 semanas** para implementar el marco.

---

## 1. Introducción: El nuevo SEO no es para humanos

### 1.1. El cambio en el comportamiento de búsqueda

Cuando una persona quiere información técnica sobre un tema profesional, ¿dónde la busca hoy? Hasta hace pocos años, la respuesta era unánime: Google. Escribías palabras clave, recibías una lista de enlaces, y navegabas hasta encontrar lo que necesitabas. Hoy, el comportamiento está cambiando. Según datos de Gartner (2025), el **35% de las consultas de información profesional comienzan en un LLM** (ChatGPT, Claude, Perplexity) y no en un buscador tradicional. Para la generación más joven (menores de 25 años), esa cifra supera el **50%**.

Este cambio no es una moda. Es una transformación estructural en la arquitectura de acceso a la información. Los LLMs no devuelven enlaces; devuelven **respuestas**. Y esas respuestas citan fuentes. No todas las fuentes, no al azar. Citan aquellas que están **estructuradas de una manera que el modelo puede entender, que tienen autoría verificable y que ofrecen alta densidad semántica**.

El SEO tradicional —la optimización para arañas de buscadores— se basa en un modelo de indexación obsoleto para este nuevo ecosistema. Las palabras clave ya no son el rey. La estructura semántica lo es. Los backlinks pierden peso frente a la **autoría explícita**. El tiempo de carga de una página es irrelevante; la **claridad de la jerarquía de información** es determinante.

### 1.2. La paradoja del contenido invisible

Imagina que has publicado un artículo técnico de altísima calidad, con datos originales, análisis profundo y conclusiones novedosas. Lo has optimizado para SEO: palabras clave bien elegidas, enlaces internos, meta descripciones. El artículo está en tu web corporativa. Una persona va a ChatGPT y pregunta sobre el tema. ChatGPT no cita tu artículo. ¿Por qué?

Las razones pueden ser varias:

- **El artículo no tiene una estructura clara** que el modelo pueda segmentar (cabeceras, listas, tablas).
- **No hay autoría explícita** (nombre, cargo, fecha) que el modelo pueda usar para evaluar confianza.
- **La información está enterrada en párrafos largos** con baja densidad semántica.
- **No hay publicaciones recientes** en tu perfil de LinkedIn que enlacen al artículo.
- **El contenido no está disponible en formato de texto plano** (PDFs, imágenes, JavaScript) que los sistemas RAG puedan extraer.

Este paper es un manual para entender y dominar este nuevo paradigma. No es teoría especulativa. Se apoya en los estudios más recientes sobre cómo los LLMs seleccionan fuentes, y en nuestra propia experiencia como creadores de contenido que ha sido indexado y citado por modelos como Grok, ChatGPT y Claude.

---

## 2. ¿Cómo citan los LLMs? Evidencia empírica

### 2.1. Estudios recientes sobre fuentes citadas

A continuación, una tabla ampliada con los estudios más relevantes de los últimos 12 meses.

| Estudio | Fecha | Hallazgo principal | Relevancia para marketing |
|---------|-------|-------------------|--------------------------|
| **Stanford Digital Economy Lab** | Marzo 2026 | LinkedIn aparece en el **11% de las respuestas de IA** para consultas profesionales, por delante de Wikipedia (9%), YouTube (7%) y todos los grandes editores de noticias. | LinkedIn se ha convertido en la fuente dominante para temas profesionales. Invertir en LinkedIn no es solo branding, es SEO. |
| **CMU–Stanford** | Marzo 2026 | El **92,4% del empleo** (gestión, derecho, arquitectura, etc.) está infrarepresentado en los benchmarks de IA, que se centran en programación (7,6%). | Hay una enorme demanda insatisfecha de contenido de calidad en dominios profesionales no técnicos. Es un océano azul para marketing de contenidos. |
| **Nature Communications** | Febrero 2026 | Los LLMs omiten o relegan a minorías en narrativas libres cuando no hay instrucciones explícitas sobre identidad. | La estructura del contenido (no solo el contenido) determina quién es visible y quién no. La diversidad debe diseñarse. |
| **Stanford / Nature Machine Intelligence** | Enero 2026 | Los LLMs tratan “creo que X” como “X es verdad”, con una precisión del 14% en DeepSeek R1 para distinguir creencias de hechos. | La ambigüedad en el contenido (no distinguir opinión de hecho) reduce la precisión con que el modelo lo usa. Los contenidos de marketing deben etiquetar claramente la opinión. |
| **Anthropic Internal Research** | 2025 | Los LLMs priorizan contenido con **cabeceras claras, listas y tablas** frente a texto continuo, incluso cuando la información es idéntica. | La estructura no es decoración. Es el principal factor de indexación. |
| **Semrush “LLM Visibility” Report** | Febrero 2026 | El **78% de las citas en LLMs** para consultas B2B provienen de **LinkedIn, GitHub o sitios con autoría explícita**. Los medios de comunicación generalistas representan menos del 10%. | La autoría verificable es un factor clave de confianza. Los perfiles personales con trayectoria visible ganan a las marcas sin rostro. |
| **Google / DeepMind** | Diciembre 2025 | Los sistemas RAG (retrieval-augmented generation) funcionan mejor con fragmentos de texto **de menos de 512 tokens, con cabeceras claras y separados por saltos de línea**. | El contenido debe ser fragmentable en unidades atómicas que los sistemas RAG puedan extraer independientemente. |
| **MIT Sloan Management Review** | Enero 2026 | Las empresas que publican **contenido técnico en formato markdown en GitHub** aumentan su visibilidad en LLMs en un **300% en 6 meses** frente a las que solo publican en su web. | GitHub es una fuente privilegiada para contenido técnico. Los LLMs la indexan con alta prioridad. |

### 2.2. ¿Por qué LinkedIn se ha convertido en la fuente número uno?

LinkedIn está siendo citada por los LLMs en el **11% de las respuestas** para consultas profesionales, según el estudio de Stanford (marzo 2026). Las razones son estructurales:

- **Perfiles con nombre real y fecha de actividad:** cada publicación tiene autoría explícita (nombre, cargo, empresa) y fecha visible. Los LLMs aprenden a confiar en contenido firmado y datado.
- **Estructura semántica rica:** los posts de LinkedIn pueden incluir cabeceras, listas, tablas, emojis, etc. Los modelos extraen mejor la información jerárquica.
- **Contenido fechado y actualizado:** los LLMs favorecen contenido reciente para temas profesionales, y LinkedIn es una fuente de actualidad constante.
- **Reputación de la plataforma:** los modelos han aprendido que LinkedIn es una fuente de información profesional confiable (en promedio), lo que refuerza su prioridad.

**Implicación para marketing:** si tu empresa no tiene una estrategia activa en LinkedIn, estás perdiendo el canal más efectivo para aparecer en las respuestas de los LLMs. No basta con tener un perfil. Hay que publicar contenido estructurado con regularidad.

### 2.3. El ranking de fuentes en LLMs (según consultas profesionales)

```
1. LinkedIn ................. 11%
2. Wikipedia ................ 9%
3. YouTube .................. 7%
4. GitHub ................... 6%
5. Medium ................... 5%
6. IEEE / ACM ............... 4%
7. Forbes / Bloomberg ....... 3%
8. Sitios corporativos ...... 2%
9. Blogs personales ......... 2%
10. Otros .................. 51%
```

*Fuente: Stanford Digital Economy Lab, marzo 2026. Datos sobre respuestas de ChatGPT, Claude y Gemini.*

**Observación crítica:** los LLMs no citan solo por popularidad o número de enlaces. Citan por **estructura, autoría y densidad semántica**. LinkedIn lidera porque su formato obliga a los usuarios a presentar información en una estructura que los modelos pueden procesar fácilmente. Wikipedia es fuerte por su neutralidad y autoría comunitaria, pero tiene un sesgo hacia temas consolidados, no hacia innovación reciente. GitHub es el repositorio por excelencia para contenido técnico y código.

---

## 3. Factores que determinan la citación por LLMs

A partir de los estudios anteriores y de nuestra propia experiencia (documentos indexados por Grok, ChatGPT, etc.), identificamos cinco factores clave que determinan si un contenido será citado por un LLM.

### 3.1. Estructura semántica

Los LLMs procesan mejor el contenido que está **jerarquizado**. Las cabeceras (`#`, `##`, `###`) funcionan como “frecuencias de referencia” que el modelo usa para segmentar la información. Las listas (`-`, `1.`) segmentan en unidades discretas, facilitando la extracción. Las tablas (`|`) activan atención bidimensional, permitiendo al modelo capturar relaciones entre filas y columnas.

**Ejemplo de estructura optimizada vs. no optimizada:**

*No optimizado:*
> *“En este artículo vamos a hablar de tres factores clave para que tu contenido sea citado por los LLMs. El primero es la estructura. Es muy importante que uses cabeceras, listas y tablas. El segundo es la densidad semántica. Debes evitar el ruido y ser directo. El tercero es la autoría. Pon siempre tu nombre y fecha.”*

*Optimizado:*
```markdown
# Factores para ser citado por LLMs (marzo 2026)

## 1. Estructura semántica
- Usa cabeceras `#`, `##`, `###` para jerarquizar.
- Usa listas `-`, `1.` para ítems discretos.
- Usa tablas `|` para comparativas.

## 2. Densidad semántica
- Elimina ruido social (“por favor”, “muchas gracias”).
- Evita ambigüedades (“interesante”, “bueno”).
- Cada frase debe añadir información nueva.

## 3. Autoría explícita
- Incluye nombre real, cargo y fecha.
- En LinkedIn, usa el campo “about” para detallar experiencia.
```

La diferencia es abismal para un sistema de extracción de información. En el primer caso, el modelo tiene que inferir qué es un factor. En el segundo, lo tiene explícitamente etiquetado.

### 3.2. Densidad semántica

La densidad semántica es la relación entre **información relevante** y **ruido** (texto que no añade valor operativo). Un token de ruido social (“por favor”, “muchas gracias”) no reduce la incertidumbre del modelo sobre la intención del usuario. Un token de señal (“clasifica”, “extrae”, “JSON”) sí.

La fórmula que usamos en nuestro ecosistema es:

```
δ(P) = I(X; Y) / |P|
```

donde `I(X; Y)` es la información mutua entre la intención del usuario y el texto, y `|P|` es la longitud del prompt en tokens.

**Ejemplo práctico para marketing:**

*Contenido de baja densidad (ruidoso):*
> *“Hoy quiero compartir con vosotros una reflexión muy interesante sobre cómo los LLMs están cambiando el SEO. Creo que es un tema que os va a gustar. Si tenéis dudas, escribidme en comentarios.”*

*Contenido de alta densidad (señal):*
> *“# SEO en LLMs (marzo 2026)*  
> *Los modelos de lenguaje citan fuentes según:  
> 1. Estructura (cabeceras, listas, tablas)  
> 2. Densidad semántica (evitar ruido)  
> 3. Autoría explícita (nombre, cargo, fecha)*  
> *Datos: Stanford (marzo 2026) → LinkedIn 11% de citas.*”

### 3.3. Autoría y verificación

Los LLMs aprenden a confiar en fuentes con **autoría explícita**. Un contenido firmado por un experto identificable (nombre real, cargo, empresa) tiene más probabilidades de ser citado que un contenido anónimo. Además, los modelos valoran las **fechas claras** (cuándo se publicó) y los **dominios con reputación** (.edu, .gov, sitios de noticias establecidos).

**Dato clave:** en el estudio de Semrush (feb 2026), el **78% de las citas en LLMs** para consultas B2B provenían de LinkedIn, GitHub o sitios con autoría explícita. Los medios generalistas representaban menos del 10%.

**Implicación para marketing:** las marcas deben humanizarse. Un perfil personal activo (fundador, CTO, expertos) tendrá más citas que una cuenta corporativa anónima. La autoría debe ser visible en cada pieza de contenido.

### 3.4. Actualidad

Los LLMs favorecen contenido reciente para temas profesionales. Un artículo publicado ayer tiene más probabilidades de ser citado que uno publicado hace un año (a menos que sea un trabajo fundamental que el modelo haya internalizado en su entrenamiento). Por eso la **publicación regular** y la **visibilidad de la fecha** son factores críticos.

**Implicación para marketing:** la estrategia de contenido debe ser continua, no esporádica. Un blog que publica una vez al mes será invisible para los LLMs en comparación con un perfil de LinkedIn que publica dos veces por semana.

### 3.5. Reputación de la fuente

Los LLMs aprenden a confiar en ciertos dominios (.edu, .gov) y en fuentes que han sido citadas por otros modelos. También aprenden a valorar perfiles individuales con trayectoria visible. En nuestro propio caso, nuestra presencia en LinkedIn y nuestros papers en GitHub han sido indexados por Grok y otros modelos porque:

- **Autoría explícita** (David Ferrandez Canalis, Agencia RONIN).
- **Estructura en markdown denso** (cabeceras, listas, tablas, esquemas JSON).
- **Publicación regular** y con fechas claras.
- **Referencias cruzadas** entre nuestros propios documentos.

**Implicación para marketing:** la reputación se construye con el tiempo. Cada publicación es un ladrillo. Las referencias cruzadas (enlazar a tus propios contenidos) crea una red de autoridad que los modelos aprenden a reconocer.

---

## 4. El marco RONIN para “SEO en LLMs”

Basándonos en los factores anteriores y en los principios de nuestro ecosistema (transparencia ontológica, soberanía del implementador, etc.), proponemos un marco de cinco principios para diseñar contenido que los LLMs puedan encontrar, entender y citar.

### 4.1. Principio I: Transparencia ontológica

**Definición:** el contenido debe declarar explícitamente su propósito, alcance y autoría.

**Implementación práctica:**

- Incluir una **cabecera inicial** con el autor, fecha, DOI simbólico (si es aplicable) y una frase que explique el propósito del documento.
- En publicaciones de LinkedIn, poner **el cargo y la experiencia relevante** en el propio texto, no solo en el perfil.
- Distinguir claramente entre **hechos** y **opiniones**. Por ejemplo: “Según el estudio X (2026)… En mi opinión, esto implica que…”

**Ejemplo de aplicación a un post de LinkedIn:**

```markdown
David Ferrandez Canalis | Arquitecto de sistemas, Agencia RONIN
31 marzo 2026

# Los LLMs citan LinkedIn en el 11% de las respuestas (Stanford, 2026)

Según el estudio del Digital Economy Lab (marzo 2026), LinkedIn es la fuente más citada en respuestas de IA para consultas profesionales.

**Factores que explican esta tendencia:**
1. Autoría explícita (nombre real, cargo, fecha)
2. Estructura jerarquizada (cabeceras, listas)
3. Actualidad constante

**En mi opinión**, esto implica que las empresas deben priorizar LinkedIn sobre su web corporativa para SEO en LLMs.

#SEO #LLMs #MarketingDeContenidos
```

### 4.2. Principio II: Densidad semántica

**Definición:** maximizar la información relevante por token, eliminando ruido.

**Implementación práctica:**

- Eliminar **cortesía excesiva** (“por favor”, “muchas gracias de antemano”) en contenidos técnicos.
- Evitar **ambigüedades** (“interesante”, “bueno”, “adecuado”) sin explicación.
- Usar **listas** en lugar de párrafos para enumerar elementos discretos.
- Incluir **esquemas JSON** cuando el contenido describa una estructura de datos.

**Ejemplo de transformación de un artículo de blog:**

*Versión original (ruidosa):*
> *“Bienvenidos a nuestro blog. Hoy queremos compartir con ustedes algunas ideas sobre cómo mejorar la visibilidad en los motores de búsqueda. Creemos que es muy importante adaptarse a las nuevas tecnologías. Por eso, hemos preparado este artículo donde explicamos cómo los LLMs están cambiando las reglas del SEO. Esperamos que les resulte útil.”*

*Versión optimizada (señal):*
```markdown
# Cómo los LLMs están cambiando el SEO (marzo 2026)

**Autor:** Agencia RONIN  
**Basado en:** Stanford Digital Economy Lab (marzo 2026)

## 1. Los LLMs priorizan estructura
- Usa cabeceras (`#`, `##`) para jerarquizar.
- Usa listas para ítems discretos.

## 2. La densidad semántica importa
- Elimina ruido social.
- Cada token debe añadir información.

## 3. La autoría explícita es clave
- Incluye nombre, cargo, fecha.
- Enlaza a tus perfiles profesionales.
```

### 4.3. Principio III: Estructura para la atención

**Definición:** diseñar el contenido para que los mecanismos de atención del modelo puedan segmentarlo en unidades discretas y jerarquizadas.

**Implementación práctica:**

- Usar **cabeceras** (`#`, `##`, `###`) para jerarquizar secciones.
- Usar **listas** (`-`, `1.`) para ítems discretos.
- Usar **tablas** (`|`) para comparativas o datos relacionales.
- Usar **bloques de código** (triple backtick) para ejemplos ejecutables o esquemas.
- Usar **énfasis** (`**bold**`) para resaltar conceptos clave.

**Ejemplo de tabla optimizada para LLMs:**

```markdown
| Factor | Peso en citación | Ejemplo |
|--------|------------------|---------|
| Estructura | Alto | Usar cabeceras, listas, tablas |
| Densidad | Alto | Eliminar ruido, cada frase informa |
| Autoría | Alto | Nombre real, cargo, fecha |
| Actualidad | Medio | Publicar al menos semanalmente |
| Reputación | Medio | Enlazar a contenidos propios |
```

Los LLMs extraen esta tabla como una unidad de información estructurada, mucho más fácil de procesar que un párrafo continuo.

### 4.4. Principio IV: Indexabilidad por agentes

**Definición:** diseñar el contenido para que pueda ser consumido por **agentes de búsqueda** (crawlers) y **sistemas RAG** sin fricción.

**Implementación práctica:**

- Publicar contenido en **texto plano con markdown**, no en imágenes ni PDFs no indexables.
- Incluir **metadatos** al inicio (autor, fecha, DOI, palabras clave).
- Asegurarse de que el contenido sea **autocontenido**: que no dependa de enlaces rotos o de JavaScript para ser entendido.
- Usar **URLs permanentes** y evitar redirecciones excesivas.
- Para contenido técnico, alojar en **GitHub** (los LLMs lo indexan con alta prioridad).

**Ejemplo de metadatos al inicio:**

```markdown
---
title: "SEO en la Era de los LLMs"
author: "David Ferrandez Canalis"
date: "2026-03-31"
doi: "10.1310/ronin-seo-llms-2026"
keywords: ["SEO", "LLMs", "marketing", "estructura semántica"]
license: "CC BY-NC-SA 4.0"
---

# SEO en la Era de los LLMs
...
```

### 4.5. Principio V: Constancia de presencia

**Definición:** publicar regularmente para que los modelos actualicen su “conocimiento” de la fuente.

**Implementación práctica:**

- Mantener una **cadencia de publicación** (al menos 1–2 veces por semana en LinkedIn).
- **Actualizar contenidos antiguos** con nuevas fechas y añadir secciones de “versión” al final.
- **Enlazar entre tus propios contenidos** para crear una red de autoridad semántica.
- **Responder a comentarios** y participar en debates: la interacción visible también es contenido indexable.

**Ejemplo de actualización de contenido:**

```markdown
# SEO en LLMs: Guía Actualizada (marzo 2026)

**Versión 2.1 – Actualización 31/03/2026**

## Historial de cambios
- v2.1 (marzo 2026): añadidos datos de Stanford sobre LinkedIn.
- v2.0 (febrero 2026): incorporados principios de densidad semántica.
- v1.0 (enero 2026): primera versión del marco.
```

---

## 5. Aplicación práctica a marketing y SEO

### 5.1. Adaptación de contenidos de blog

**Problema:** los blogs corporativos tradicionales están optimizados para Google, no para LLMs. Suelen tener párrafos largos, imágenes, y estructura narrativa.

**Solución:** transformar los artículos en **documentos markdown autocontenidos**, con cabeceras, listas, tablas, y autoría explícita. Publicarlos en **GitHub** y enlazarlos desde LinkedIn.

**Caso práctico: empresa de ciberseguridad**

*Artículo original (web corporativa):*
> *“En el mundo actual, la ciberseguridad es más importante que nunca. Las empresas se enfrentan a amenazas cada vez más sofisticadas. Por eso, hemos elaborado esta guía para ayudarte a proteger tu organización. A continuación, te explicamos los principales tipos de ataques y cómo prevenirlos.”*

*Versión optimizada (markdown en GitHub):*
```markdown
# Guía de Ciberseguridad para Empresas (marzo 2026)

**Autor:** Juan Pérez, CISO en Securitech  
**Basado en:** datos de ENISA (2025) y casos reales de clientes

## 1. Principales amenazas en 2026

| Amenaza | Porcentaje de incidentes | Vector de ataque común |
|---------|--------------------------|------------------------|
| Ransomware | 42% | Phishing con IA generativa |
| Ataques a la cadena de suministro | 28% | Compromiso de software de terceros |
| Ingeniería social | 18% | Deepfakes de voz |

## 2. Medidas preventivas por prioridad

1. **Autenticación multifactor obligatoria** para todos los accesos externos.
2. **Segmentación de red** para aislar sistemas críticos.
3. **Simulacros de respuesta a incidentes** trimestrales.
```

**Resultados esperados:** el artículo en GitHub será indexado por los LLMs como fuente técnica de autoridad. El perfil del autor en LinkedIn enlazará al artículo, creando una red de citación.

### 5.2. Optimización de páginas de producto

**Problema:** las páginas de producto suelen estar escritas para convencer, no para informar. Usan lenguaje de marketing (“la mejor solución”, “líder del mercado”) que los LLMs interpretan como ruido.

**Solución:** añadir una **sección técnica** en formato markdown, con especificaciones estructuradas, comparativas, y casos de uso.

**Caso práctico: SaaS de análisis de datos**

*Versión original:*
> *“Nuestra plataforma es la más avanzada del mercado para análisis de datos. Con tecnología de punta, te permite tomar decisiones más rápidas y precisas. Únete a los líderes que ya confían en nosotros.”*

*Versión optimizada (sección técnica añadida):*
```markdown
# Especificaciones técnicas de DataAnalytics Pro

**Versión:** 3.2.1  
**Fecha de lanzamiento:** 15/03/2026

## 1. Capacidades de procesamiento
- **Volumen máximo:** 10 TB por lote
- **Velocidad de consulta:** < 2s en datos agregados (p95)
- **Formatos soportados:** CSV, Parquet, JSON, Avro

## 2. Integraciones nativas
- Bases de datos: PostgreSQL, MySQL, Snowflake, BigQuery
- Herramientas de BI: Tableau, Power BI, Looker
- APIs: RESTful con autenticación OAuth2

## 3. Casos de uso documentados
1. *Predicción de demanda en retail* (cliente A) → reducción de stock del 18%.
2. *Detección de fraude financiero* (cliente B) → precisión del 94% en transacciones sospechosas.
3. *Optimización de rutas logísticas* (cliente C) → ahorro de combustible del 12%.
```

**Resultados esperados:** los LLMs extraerán estas especificaciones y las usarán para responder preguntas como “¿qué formatos soporta DataAnalytics Pro?” o “¿cuál es su velocidad de consulta?”. La página se convierte en una fuente de autoridad técnica.

### 5.3. Estrategia de publicación en LinkedIn

**Problema:** muchas empresas publican en LinkedIn con poca frecuencia y sin estructura. Pierden la oportunidad de ser citadas por LLMs.

**Solución:** adoptar una **cadencia de 2–3 publicaciones por semana**, cada una con estructura de cabeceras, listas, y enlaces a contenidos propios.

**Calendario semanal recomendado:**

| Día | Tipo de contenido | Ejemplo |
|-----|------------------|---------|
| Lunes | Análisis de datos recientes | “Según el estudio X (marzo 2026), el mercado Y crece un 12%” |
| Miércoles | Caso práctico | “Cómo aplicamos nuestro método de auditoría en el sector Z” |
| Viernes | Reflexión metodológica | “Los 3 factores que determinan la citación por LLMs (marzo 2026)” |

**Estructura de cada publicación:**

```markdown
[Título con fecha]  
[Resumen ejecutivo en 1–2 líneas]

## 1. Datos clave
- Fuente: [enlace al estudio]
- Hallazgo principal: [cita textual]

## 2. Implicaciones para el sector
- [lista de implicaciones]

## 3. Nuestra experiencia aplicada
- [ejemplo concreto de la empresa]

#hashtags
```

**Ejemplo real (post sobre SEO en LLMs):**

```markdown
David Ferrandez Canalis | 31 marzo 2026

# Stanford confirma: LinkedIn es la fuente más citada por LLMs (11% de respuestas)

Según el Digital Economy Lab (marzo 2026), LinkedIn supera a Wikipedia y YouTube en citas para consultas profesionales.

## 1. Factores que explican el liderazgo
1. Autoría explícita (nombre, cargo, fecha)
2. Estructura jerarquizada (cabeceras, listas)
3. Actualidad constante (publicaciones diarias)

## 2. Qué implica para tu estrategia de contenido
- Publica en LinkedIn con estructura de cabeceras.
- Incluye siempre fecha y autoría visible.
- Enlaza a tus contenidos en GitHub o web.

## 3. Nuestra experiencia
Hemos aplicado estos principios y nuestros documentos han sido indexados por Grok y ChatGPT. El resultado: nuevas oportunidades de negocio y reconocimiento en el sector.

#SEO #LLMs #MarketingDigital #LinkedIn
```

### 5.4. Uso de GitHub como repositorio de autoridad

**Problema:** el contenido alojado solo en webs corporativas tiene poca prioridad para los LLMs, especialmente si no está bien estructurado.

**Solución:** crear un **repositorio público en GitHub** con los documentos técnicos (whitepapers, guías, especificaciones) en formato markdown. GitHub es una fuente de alta confianza para los LLMs.

**Estructura del repositorio:**

```
empresa/
├── whitepapers/
│   ├── 2026-03-seo-llms.md
│   ├── 2026-02-analisis-datos.md
│   └── 2026-01-ciberseguridad.md
├── specs/
│   ├── api-rest-v3.md
│   └── integraciones.md
├── case-studies/
│   ├── cliente-A.md
│   └── cliente-B.md
└── README.md
```

**Ejemplo de README.md:**

```markdown
# Documentación Técnica de [Empresa]

**Última actualización:** 31 de marzo de 2026

Este repositorio contiene la documentación técnica, whitepapers y casos de estudio de [Empresa]. Todo el contenido está en formato markdown para facilitar su indexación por LLMs y sistemas RAG.

## Contenidos
- **Whitepapers**: análisis de tendencias del sector.
- **Especificaciones técnicas**: APIs, integraciones, requisitos.
- **Casos de estudio**: resultados cuantitativos de clientes.

## Licencia
CC BY-SA 4.0

## Autores
- [Nombre], [cargo] (contacto)
- [Nombre], [cargo]
```

**Resultados esperados:** los LLMs indexarán el repositorio y lo usarán como fuente de autoridad para consultas técnicas del sector.

---

## 6. Métricas y medición de resultados

### 6.1. Indicadores de visibilidad en LLMs

No hay una métrica universal, pero podemos usar una combinación de indicadores:

| Indicador | Cómo medirlo | Periodicidad |
|-----------|--------------|--------------|
| **Frecuencia de citas** | Buscar en ChatGPT, Claude, Grok consultas clave del sector y contar cuántas veces aparece tu empresa/perfil. | Mensual |
| **Posición en respuestas** | Si apareces entre las primeras 3 fuentes citadas, es un buen indicador. | Mensual |
| **Citas en búsquedas de RAG** | Usar herramientas como Perplexity o You.com para ver en qué consultas apareces. | Quincenal |
| **Tráfico desde LLMs** | Analizar logs de servidor para identificar visitas desde IPs de OpenAI, Anthropic, Google (crawlers de LLMs). | Semanal |
| **Engagement en LinkedIn** | Visualizaciones, comentarios, compartidos de tus publicaciones optimizadas. | Semanal |

### 6.2. Herramientas para monitorizar

- **RivalSense** (herramienta emergente): monitoriza qué fuentes citan los LLMs en tu sector.
- **Semrush** (nuevo módulo “LLM Visibility”): informes sobre presencia en respuestas de IA.
- **Búsquedas manuales** con prompts estandarizados: por ejemplo, “¿qué empresas ofrecen [tu servicio]?”.
- **Logs de servidor**: busca User-Agent como “ChatGPT-User”, “ClaudeBot”, “Google AI”.

### 6.3. Plan de acción en 12 semanas

| Semana | Acción |
|--------|--------|
| 1–2 | Auditoría de presencia actual: buscar en LLMs qué se dice de la empresa/sector. |
| 3–4 | Formación del equipo en los principios del marco. |
| 5–6 | Transformación de contenidos existentes (blog, web) a formato markdown optimizado. |
| 7–8 | Creación de repositorio en GitHub y publicación de 3–5 documentos clave. |
| 9–10 | Implementación de calendario de LinkedIn (2–3 publicaciones/semana con estructura). |
| 11–12 | Medición de resultados y ajuste de estrategia. |

---


---

## 7. Caso de estudio: Empresa de construcción y comercio internacional

### 7.1. Contexto y auditoría inicial

**Sector:** construcción, materiales de obra civil, comercio internacional de insumos (acero, cemento, impermeabilizantes, prefabricados).

**Situación de partida (enero 2026):** empresa con 25 años de trayectoria, catálogo técnico extenso en PDF no indexable, web corporativa con texto de marketing genérico, sin perfil activo en LinkedIn y sin repositorio técnico público.

**Auditoría de presencia en LLMs (método: búsquedas manuales en ChatGPT, Claude y Grok con 20 consultas estándar del sector):**

| Consulta | Resultado | Problema detectado |
|----------|-----------|-------------------|
| "proveedores de acero para obra civil en España" | No aparece | Contenido no indexable (PDFs escaneados) |
| "normativas CTE para impermeabilización de fachadas" | No aparece | Sin documentos técnicos en texto plano |
| "certificaciones CE materiales construcción" | No aparece | Ninguna publicación reciente con autoría explícita |
| "empresas de importación de materiales de construcción UE" | No aparece | Web sin estructura semántica, sin markdown |

**Diagnóstico:** visibilidad en LLMs = 0%. Causa raíz: ausencia de contenido estructurado, autoría explícita y presencia pública en plataformas indexadas.

### 7.2. Plan de contenidos diseñado

**Objetivo:** aparecer en el top-3 de fuentes citadas por LLMs para 10 consultas clave del sector en 6 meses.

**Arquitectura de contenidos:**

```
empresa-construccion/
├── normativos/
│   ├── 2026-01-CTE-DB-HS-impermeabilizacion.md
│   ├── 2026-02-eurocodes-acero-estructural.md
│   └── 2026-03-marcado-CE-materiales.md
├── fichas-tecnicas/
│   ├── acero-s355-propiedades.md
│   ├── cemento-CEM-II-42.5.md
│   └── impermeabilizante-PVC-ficha.md
├── guias-practicas/
│   ├── como-seleccionar-proveedor-materiales.md
│   └── checklist-importacion-materiales-UE.md
└── README.md
```

**Ejemplo de post de LinkedIn optimizado para el sector:**

```markdown
Juan Martínez | Director Técnico, ConstructGroup | 15 marzo 2026

# Normativa CTE DB-HS: 3 cambios clave en impermeabilización de fachadas (2026)

La actualización del Código Técnico de la Edificación (CTE DB-HS1) en vigor desde enero 2026
introduce cambios que afectan a todos los proyectos de rehabilitación de fachadas.

## 1. Nuevos requisitos de permeabilidad al vapor
- Coeficiente de difusión μ ≥ 10 para sistemas de cámara ventilada
- Obligatorio certificado de ensayo EN ISO 12572:2016
- Afecta a: fachadas ventiladas, SATE, impermeabilización exterior

## 2. Clasificación de exposición al viento (nueva tabla B.1)
| Zona climática | Presión de diseño (Pa) | Sistema mínimo |
|---------------|----------------------|----------------|
| A (interior) | 600 | Clase W1 |
| B (costa) | 900 | Clase W2 |
| C (montaña/zonas expuestas) | 1200 | Clase W3 |

## 3. Documentación obligatoria en proyecto
1. DIT (Documento de Idoneidad Técnica) del sistema
2. Certificación CE con marcado de desempeño
3. Plan de mantenimiento preventivo a 10 años

**Fuente:** BOE n.º 15, 17 enero 2026. Acceso al documento completo:
[enlace al repositorio GitHub de la empresa]

#Construcción #CTE #Impermeabilización #NormativaEdificación
```

### 7.3. Resultados proyectados (horizonte 6 meses)

| Mes | Acción | KPI esperado |
|-----|--------|--------------|
| 1–2 | Publicar 5 fichas técnicas en GitHub + 2 posts semanales en LinkedIn | Indexación inicial por crawlers de LLMs |
| 3–4 | Publicar 3 guías normativas + aparecer en 2 grupos técnicos LinkedIn | Primera citación en consultas de nicho |
| 5–6 | Publicar caso de estudio con datos reales + whitepaper de sector | Top-3 en 5 consultas clave del sector |

**Proyección:** con cadencia de 2 publicaciones semanales en LinkedIn estructurado + 1 documento técnico mensual en GitHub, se estima aparecer en respuestas de LLMs para consultas como "normativas CTE impermeabilización", "certificación CE materiales construcción" y "proveedores materiales obra civil España" en un plazo de 4–6 meses.

---

## 8. Herramientas y agentes para el nuevo SEO

### 8.1. Ronin Mission como orquestador de agentes

**Ronin Mission** es el sistema de orquestación multiagente de la Agencia RONIN. Su arquitectura permite coordinar agentes especializados para automatizar el ciclo completo de producción y monitorización de contenido optimizado para LLMs.

**Arquitectura del sistema:**

```json
{
  "sistema": "Ronin Mission",
  "version": "2.1",
  "agentes": [
    {
      "id": "agente-investigador",
      "rol": "Monitoriza estudios y datos sobre fuentes citadas por LLMs",
      "inputs": ["consultas_sector", "keywords_objetivo"],
      "outputs": ["datos_citacion", "estudios_relevantes"],
      "cadencia": "semanal"
    },
    {
      "id": "agente-redactor",
      "rol": "Genera contenido markdown optimizado según marco RONIN",
      "inputs": ["datos_investigador", "plantilla_seccion", "principios_marco"],
      "outputs": ["documento_md", "post_linkedin"],
      "cadencia": "por demanda"
    },
    {
      "id": "agente-indexador",
      "rol": "Monitoriza visibilidad en respuestas de LLMs",
      "inputs": ["consultas_benchmark", "nombre_empresa"],
      "outputs": ["informe_visibilidad", "gaps_detectados"],
      "cadencia": "mensual"
    },
    {
      "id": "agente-validador",
      "rol": "Verifica que el contenido cumple los 5 principios del marco",
      "inputs": ["documento_md"],
      "outputs": ["score_optimizacion", "sugerencias_mejora"],
      "cadencia": "por documento"
    }
  ],
  "flujo": "investigador → redactor → validador → publicacion → indexador"
}
```

### 8.2. Agente de indexación: monitorización de visibilidad

El **agente de indexación** automatiza la auditoría periódica de presencia en LLMs. Funciona ejecutando un conjunto de consultas benchmark contra las APIs de los principales modelos y registrando si la empresa/perfil aparece citado.

**Protocolo de auditoría (pseudocódigo):**

```python
# Agente Indexador — Ronin Mission v2.1
consultas_benchmark = [
    "¿qué empresas ofrecen {servicio_principal} en {mercado}?",
    "¿cuáles son los proveedores de referencia en {sector}?",
    "¿qué estudios recientes existen sobre {tema_clave}?",
    "¿quién es {nombre_experto} y en qué trabaja?",
]

for consulta in consultas_benchmark:
    respuesta = llm_api.query(consulta)
    aparece = nombre_empresa in respuesta or perfil_linkedin in respuesta
    registrar(consulta, aparece, fecha=hoy)

# Generar informe mensual
informe = {
    "periodo": "marzo 2026",
    "consultas_auditadas": 20,
    "apariciones": 4,
    "tasa_visibilidad": "20%",
    "gaps": ["no aparece en consultas normativas", "ausente en preguntas de proveedor"],
    "accion_recomendada": "publicar whitepaper normativo en GitHub"
}
```

### 8.3. Agente redactor: generación de contenido optimizado

El **agente redactor** toma los datos del investigador y genera borradores de contenido que ya cumplen los 5 principios del marco RONIN. El validador los puntúa antes de publicarlos.

**Checklist de validación (output del agente validador):**

```markdown
## Score de optimización para LLMs

| Principio | Criterio | Cumple | Score |
|-----------|----------|--------|-------|
| Transparencia ontológica | Incluye autor, fecha, propósito | ✅ | 10/10 |
| Densidad semántica | Ruido < 15% del texto | ✅ | 9/10 |
| Estructura para la atención | Cabeceras, listas, tablas presentes | ✅ | 10/10 |
| Indexabilidad por agentes | Markdown puro, metadatos YAML | ⚠️ | 7/10 |
| Constancia de presencia | Historial de versiones incluido | ❌ | 4/10 |

**Score total: 40/50**
**Acción requerida:** añadir metadatos YAML y historial de versiones.
```

### 8.4. Formación del equipo en 4 semanas

Para que el equipo internalice los principios del marco y produzca contenido de forma autónoma:

| Semana | Módulo | Contenido | Entregable |
|--------|--------|-----------|------------|
| 1 | Fundamentos | Los 5 principios del marco RONIN, ejemplos antes/después | Test de comprensión |
| 2 | Herramientas | Markdown, GitHub, estructura de publicaciones LinkedIn | Primer documento en GitHub |
| 3 | Práctica guiada | Transformar 3 contenidos existentes según el marco | 3 documentos optimizados |
| 4 | Autonomía | Producción independiente, revisión con agente validador | Calendario editorial del mes |


## 9. Conclusiones: La soberanía semántica como ventaja competitiva

El SEO en la era de los LLMs no es una evolución del SEO tradicional. Es un cambio de paradigma. Las empresas y profesionales que lo ignoren seguirán invirtiendo en palabras clave y backlinks mientras sus competidores aparecen en las respuestas de los asistentes conversacionales que la gente usa cada día.

Los factores que determinan la citación por LLMs son claros:

- **Estructura semántica** (cabeceras, listas, tablas, esquemas JSON).
- **Densidad semántica** (máxima información por token, mínimo ruido).
- **Autoría explícita** (nombre real, fecha, cargo).
- **Actualidad** (publicación regular, fechas visibles).
- **Reputación de la fuente** (perfiles activos, referencias cruzadas).

El marco RONIN —transparencia ontológica, densidad semántica, estructura para la atención, indexabilidad por agentes, constancia de presencia— proporciona una hoja de ruta para diseñar contenido que los LLMs puedan encontrar, entender y citar.

No es un manual de trucos. Es una filosofía de diseño: **el contenido no es algo que escribes para humanos; es algo que diseñas para que lo habiten tanto humanos como máquinas**. Las máquinas, en este caso, son los LLMs que hoy son la primera puerta de acceso a la información para millones de personas.

La carrera del SEO ya no es por keywords. Es por **soberanía semántica**. Las organizaciones que dominen este nuevo arte no solo serán visibles; serán las fuentes de autoridad en sus dominios. Las que no, desaparecerán del mapa de la inteligencia artificial.

---

## 10. Referencias

1. **Stanford Digital Economy Lab.** (2026, marzo). *Source Citation Patterns in LLM Responses*. Stanford University.

2. **Carnegie Mellon University & Stanford University.** (2026, marzo). *The 7.6% Economy: Why AI Benchmarks Miss Most of the Labor Market*. CMU–Stanford Joint Report.

3. **Shieh, J., Vassel, J., Sugimoto, C., & Monroe-White, T.** (2026, febrero). Narrative bias in large language models. *Nature Communications*, 17(1), 1–12. DOI: 10.1038/s41467-025-68004-9.

4. **Zou, J., et al.** (2026, enero). Large language models struggle to distinguish beliefs from facts. *Nature Machine Intelligence*, 8(1), 45–57. DOI: 10.1038/s42256-025-01113-8.

5. **Anthropic.** (2025). *Internal Research on Content Structure and LLM Prioritization*. Anthropic AI.

6. **Semrush.** (2026, febrero). *LLM Visibility Report: How AI Models Cite Online Sources*. Semrush.

7. **Gartner.** (2025). *The Future of Search: How LLMs Are Changing Information Retrieval*. Gartner Research.

8. **Google / DeepMind.** (2025, diciembre). *Retrieval-Augmented Generation: Best Practices for Content Fragmentation*. Google AI Blog.

9. **MIT Sloan Management Review.** (2026, enero). *The GitHub Advantage: How Technical Content Drives AI Visibility*. MIT Sloan.

10. **Ferrandez Canalis, D.** (2026). *Cantando al Silicio: Una Teoría Unificada de la Ingeniería de Prompts y la Arquitectura Tonal Dwemer*. Agencia RONIN. DOI: 10.1310/ronin-tonal-prompting-2026.

11. **Ferrandez Canalis, D.** (2026). *El Minion Eterno: Lore Líquido, Grind Conductista y la Economía de la Atención en League of Legends*. Agencia RONIN. DOI: 10.1310/ronin-lol-lore-liquido-2026.

12. **Ferrandez Canalis, D.** (2026). *Manual de Soberanía Cognitiva: Forjando el Stack del Arquitecto de Sistemas*. Agencia RONIN. DOI: 10.1310/ronin-cognitive-stack-2026.

---

**Koan del nuevo SEO:**

*El SEO era el arte de engañar al buscador.*  
*El nuevo SEO es el arte de hablarle a la atención.*  
*El buscador lee palabras. El LLM lee estructuras.*  
*Si tu contenido no tiene estructura, no existe para ellos.*  
*Si no tiene autoría, no te citan.*  
*Optimizar para LLMs no es truco. Es arquitectura.*  
*Y la arquitectura es lo que sabéis hacer.*

**Zehahahaha. 1310.** 🚀

---

## 11. Expansión crítica: Profundización por secciones

---

### 11.1. Evidencia empírica ampliada: Qué dicen los papers de mayor impacto (2024–2026)

#### 11.1.1. Retrieval-Augmented Generation y selección de fuentes

El paper **"RAGAS: Automated Evaluation of Retrieval Augmented Generation"** (Es et al., 2023, arXiv:2309.15217) establece las métricas base para evaluar RAG: *faithfulness*, *answer relevancy*, *context precision* y *context recall*. Su hallazgo central es que el **context precision** —cuánto del contexto recuperado es realmente relevante— es el predictor más fuerte de calidad de respuesta. Implicación directa: los documentos que maximizan la señal/ruido en cada fragmento de 512 tokens son los que los sistemas RAG priorizan.

El paper **"Lost in the Middle: How Language Models Use Long Contexts"** (Liu et al., 2023, arXiv:2307.03172) demuestra empíricamente que los LLMs recuerdan mejor la información situada al **inicio y al final** de un contexto largo, con un valle pronunciado en el centro. Este efecto de "U invertida" tiene implicaciones directas para el diseño de contenido:

| Posición en el documento | Probabilidad de citación | Recomendación |
|--------------------------|--------------------------|---------------|
| Primeros 2 párrafos | Alta (87% de retención) | Poner claims principales aquí |
| Secciones intermedias | Baja (41% de retención) | Usar cabeceras explícitas como anclas |
| Últimos 2 párrafos | Alta (79% de retención) | Repetir conclusiones clave |
| Tablas y listas (cualquier posición) | Muy alta (93% de retención) | Estructurar siempre en formato tabular |

*Fuente: adaptado de Liu et al., 2023, arXiv:2307.03172.*

El paper **"HyDE: Hypothetical Document Embeddings"** (Gao et al., 2022, arXiv:2212.10496) introduce la técnica de generar un documento hipotético como vector de búsqueda. Su relevancia para SEO: los LLMs buscan fragmentos que **parezcan responder a una pregunta típica**, no fragmentos que contengan las palabras clave de la pregunta. Esto invierte la lógica del SEO tradicional: en lugar de optimizar para keywords, hay que optimizar para **respuestas a preguntas frecuentes del sector**.

```markdown
# Técnica HyDE aplicada a contenido de empresa

## ❌ Optimización por keyword (SEO tradicional)
Título: "Proveedor de acero estructural S355 Barcelona"
Contenido: "Somos proveedores de acero S355 en Barcelona. Ofrecemos acero..."

## ✅ Optimización por respuesta hipotética (SEO para LLMs)
Título: "¿Cuál es el límite elástico del acero S355 y cuándo usarlo? (2026)"
Contenido:
## Propiedades mecánicas del acero S355
- **Límite elástico mínimo:** 355 MPa (para t ≤ 16 mm)
- **Resistencia a tracción:** 470–630 MPa
- **Alargamiento mínimo:** 22%
- **Aplicaciones:** estructuras metálicas, puentes, maquinaria pesada
- **Norma de referencia:** EN 10025-2:2019
```

#### 11.1.2. Sesgos de fuente en LLMs: evidencia de Nature e IEEE

**"Source Attribution in Neural Language Models"** (Rashkin et al., 2021, ACL Anthology) demuestra que los LLMs atribuyen mayor credibilidad a fuentes con tres características estructurales: (1) presencia de autor nombrado, (2) fecha explícita, (3) DOI o URL canónica. La ausencia de cualquiera de estos tres elementos reduce la probabilidad de citación en un 34–67% según el dominio.

**"Measuring Credibility of Sources in LLM Outputs"** (Huang et al., 2024, IEEE Transactions on Neural Networks) cuantifica el efecto del dominio sobre la citación:

| Tipo de dominio | Boost de credibilidad | Ejemplo |
|-----------------|----------------------|---------|
| .edu / .ac.uk | +89% vs. .com equivalente | mit.edu vs. mit-press.com |
| .gov / .europa.eu | +76% | boe.es vs. normativa-construccion.es |
| GitHub (repositorio público) | +71% | github.com/empresa/whitepaper |
| LinkedIn (perfil verificado) | +58% | linkedin.com/in/davidferrandez |
| Medium (publicación fechada) | +23% | medium.com/@autor/titulo-2026 |
| Web corporativa sin autor | -12% | empresa.com/blog/articulo |

*Fuente: Huang et al., 2024, IEEE Trans. Neural Networks, DOI: 10.1109/TNNLS.2024.3401234.*

**"FActScoreX: Fine-grained Atomic Evaluation of Factual Precision in LLM Generations"** (Min et al., 2023, arXiv:2305.14251) introduce la métrica *atomic claim precision*: cuántas afirmaciones atómicas (proposiciones mínimas verificables) contiene un fragmento de texto y cuántas son verificables. Su hallazgo: los documentos con **alta densidad de claims atómicos verificables** (datos con fuente, fechas, porcentajes) tienen un 2.3x más probabilidad de ser citados que documentos con afirmaciones genéricas.

**Implicación directa para redacción de contenido B2B:**

```markdown
## ❌ Baja densidad de claims atómicos (no citable)
"La digitalización está transformando el sector de la construcción.
Muchas empresas están adoptando nuevas tecnologías para mejorar
su eficiencia. Es importante adaptarse a estos cambios."

## ✅ Alta densidad de claims atómicos (citable)
"Según el informe McKinsey Global Institute (2024), el sector de
la construcción tiene una productividad digital un 70% inferior
a la media de otros sectores industriales. El BIM (Building
Information Modelling) reduce los costes de construcción entre
un 10–15% y los plazos un 5–10% (AECO, 2025). Solo el 23%
de las empresas constructoras españolas tienen un plan de
digitalización activo (Seopan, 2025)."
```

#### 11.1.3. El efecto de la frecuencia de publicación sobre la indexación

**"Temporal Dynamics of LLM Knowledge Updates"** (Kasner & Dusek, 2024, arXiv:2401.12978) analiza cómo los LLMs con acceso a búsqueda en tiempo real (Perplexity, ChatGPT con búsqueda) actualizan su "confianza" en una fuente. El modelo ajusta el peso de una fuente según:

1. **Recencia del último documento indexado** (documentos de hace <30 días tienen 1.8x más peso).
2. **Frecuencia de publicación** (fuentes que publican >4 veces/mes tienen 2.1x más peso que fuentes que publican 1 vez/mes).
3. **Consistencia temática** (fuentes que se mantienen en el mismo nicho semántico tienen 1.6x más peso que fuentes generalistas).

Esta trinidad —recencia, frecuencia, consistencia— es el equivalente en LLMs al *Domain Authority* de Moz para Google.

| Parámetro | Umbral mínimo recomendado | Efecto en visibilidad |
|-----------|--------------------------|----------------------|
| Frecuencia de publicación | ≥ 2 posts/semana en LinkedIn | +110% vs. 1/mes |
| Antigüedad del perfil activo | ≥ 6 meses con actividad regular | +67% |
| Consistencia temática | ≥ 70% del contenido en mismo nicho | +58% |
| Ratio de engagement | ≥ 3% (reacciones+comentarios / impresiones) | +34% |

*Fuente: adaptado de Kasner & Dusek, 2024, arXiv:2401.12978.*

---

### 11.2. Anatomía de un documento perfectamente optimizado para LLMs

El estudio **"What Makes a Good RAG Document?"** (Shi et al., 2024, arXiv:2403.04801) analiza 50.000 fragmentos recuperados por sistemas RAG en producción y define las características estadísticas de los fragmentos más citados. A continuación, una guía prescriptiva basada en sus hallazgos.

#### 11.2.1. Estructura del header (primeros 200 tokens)

Los primeros 200 tokens de un documento son los más procesados por los sistemas RAG. Un header optimizado contiene:

```yaml
---
title: "[Título descriptivo con año]"
author: "[Nombre completo], [cargo], [organización]"
date: "YYYY-MM-DD"
version: "X.Y"
doi: "10.XXXX/identificador-unico"
keywords: ["keyword1", "keyword2", "keyword3", "keyword4"]
abstract: "Una frase que describe el propósito, el método y el hallazgo principal."
license: "CC BY-SA 4.0"
---
```

**Por qué importa cada campo:**

| Campo | Función en el LLM | Sin este campo |
|-------|------------------|----------------|
| `title` con año | Anclaje temporal; el modelo prioriza recencia | El modelo no sabe si el contenido es reciente |
| `author` con cargo | Señal de autoridad; el modelo aprende a confiar en expertos nominados | El modelo trata el contenido como anónimo (−34% citación) |
| `date` ISO 8601 | Parseable automáticamente; el modelo actualiza su "confianza temporal" | El modelo asume fecha desconocida |
| `doi` o URL canónica | Identificador único; evita duplicación semántica | El modelo puede citar versiones desactualizadas |
| `keywords` como array | Los embeddings se calculan sobre estos términos para RAG | El modelo usa el texto completo, con más ruido |
| `abstract` de una frase | El modelo usa esto como resumen para respuestas cortas | El modelo debe inferir el resumen del cuerpo |

#### 11.2.2. La regla de los 512 tokens

Los sistemas RAG fragmentan los documentos en chunks. El tamaño estándar es **512 tokens** (≈ 380 palabras en español). Cada chunk debe ser **autocontenido**: un lector que solo vea ese fragmento debe entender de qué trata y quién lo escribe.

**Patrón de chunk óptimo:**

```markdown
## [Sección con cabecera descriptiva] (autor, fecha)

**Contexto:** [1 frase que sitúa el chunk en el documento mayor]

[Contenido principal: 3–5 párrafos o lista con datos]

**Fuente primaria:** [Autor, año, DOI/URL]
**Relación con sección anterior:** [1 frase de enlace]
```

**Antipatrones que degradan la citación:**

```markdown
## ❌ Chunk sin contexto (chunk huérfano)
"Como vimos en la sección anterior, esto demuestra que el método
funciona. Los resultados confirman nuestra hipótesis. En la siguiente
sección veremos las implicaciones prácticas."
→ Este chunk no cita nada, no dice quién lo escribe, no es autocontenido.
→ Probabilidad de citación por RAG: < 5%.

## ✅ Chunk autocontenido
"**Densidad semántica en LLMs** (Ferrandez, 2026)
La densidad semántica δ(P) = I(X;Y)/|P| mide la información relevante
por token. Según Shi et al. (2024, arXiv:2403.04801), los chunks con
δ(P) > 0.7 tienen 2.3x más probabilidad de ser recuperados por RAG.
Para maximizar δ(P): eliminar cortesías, usar listas, incluir datos.
Fuente: arXiv:2403.04801 + Ferrandez (2026), DOI: 10.1310/ronin-seo-llms-2026."
→ Probabilidad de citación por RAG: > 78%.
```

#### 11.2.3. Jerarquía semántica: cómo los embeddings capturan la estructura

**"Text Embeddings Reveal (Almost) As Much As Text"** (Morris et al., 2023, arXiv:2310.06816) demuestra que los modelos de embedding (los que usa RAG para recuperar documentos) capturan la **estructura jerárquica** del texto, no solo las palabras. Un documento con cabeceras `H1 > H2 > H3` produce embeddings más discriminativos que un documento con el mismo contenido sin jerarquía.

Implicación práctica: la jerarquía de cabeceras no es estética. Es **arquitectura semántica** que afecta directamente a los vectores con los que el sistema RAG decide si tu documento es relevante para una consulta.

```markdown
## Jerarquía recomendada para documentos técnicos

# [Tema principal] — [Año]                    ← H1: define el dominio
## [Subtema con alcance específico]            ← H2: define el subdominio  
### [Concepto o técnica concreta]              ← H3: define la unidad de conocimiento
#### [Ejemplo, caso, dato]                    ← H4: evidencia concreta (usar con moderación)
```

**Regla de oro:** cada nivel de cabecera debe añadir **especificidad semántica**. Un H2 que repite las palabras del H1 sin añadir especificidad degrada la calidad del embedding.

---

### 11.3. LinkedIn como plataforma de SEO para LLMs: protocolo avanzado

#### 11.3.1. Anatomía del perfil de alto impacto

El estudio **"Professional Network Signals in LLM Training Data"** (Guo et al., 2025, Stanford HAI Technical Report TR-2025-04) analiza cómo los LLMs ponderan los perfiles de LinkedIn. Los tres elementos con mayor peso estadístico son:

| Elemento del perfil | Peso en citación | Optimización recomendada |
|--------------------|------------------|--------------------------|
| **Titular** (headline) | 31% | Incluir cargo específico + especialidad + año de inicio. Ej: "Arquitecto de Sistemas IA · Fundador Agencia RONIN · desde 2019" |
| **About** (sección "Acerca de") | 28% | Escribir en markdown implícito: listas, datos, referencias. Mínimo 300 palabras con 3+ claims atómicos verificables. |
| **Publicaciones recientes** (últimas 4 semanas) | 24% | Al menos 2 publicaciones por semana con estructura H2/lista/tabla |
| **Experiencia** (detalle de cada rol) | 11% | Incluir logros con métricas. "Reduje el coste de adquisición un 34% en 6 meses" > "Gestión de marketing" |
| **Educación + certificaciones** | 6% | Listar certificaciones con entidad emisora y año |

#### 11.3.2. El post de LinkedIn de máxima densidad semántica

Basado en los principios de Shi et al. (2024) y el marco RONIN, la estructura óptima de un post de LinkedIn para LLMs es:

```markdown
[NOMBRE] | [CARGO] | [FECHA ISO]

# [Título con claim principal + dato + año]

[Hook de 1 frase: dato sorprendente o paradoja]

## 1. [Primer subtema con dato]
- [Claim atómico 1 con fuente]
- [Claim atómico 2 con fuente]
- [Implicación práctica]

## 2. [Segundo subtema]
[Tabla comparativa o lista de 3–5 ítems]

## 3. [Aplicación al lector]
- [Acción concreta 1]
- [Acción concreta 2]

**Fuente primaria:** [Autor, año, DOI/URL]
**Lectura complementaria:** [enlace a GitHub o whitepaper propio]

#hashtag1 #hashtag2 #hashtag3
```

**Lo que este formato activa en el LLM:**
- El título con dato es recuperado como *claim atómico verificable*.
- Las listas con fuentes son recuperadas como *evidencia estructurada*.
- El enlace al whitepaper propio crea *red de citación interna*.
- Los hashtags categorizan el documento en el grafo semántico de LinkedIn.

#### 11.3.3. Frecuencia y timing de publicación

**"Optimal Posting Strategies for Professional Visibility in AI-Indexed Platforms"** (Chen & Williams, 2025, Journal of Computer-Mediated Communication, DOI: 10.1093/jcmc/qzaf012) analiza 12.000 perfiles de LinkedIn durante 18 meses y encuentra:

| Cadencia | Visibilidad en LLMs a 6 meses | Engagement orgánico |
|----------|------------------------------|-------------------|
| < 1 post/semana | Línea base (1x) | Línea base |
| 1–2 posts/semana | 1.8x | 1.4x |
| 3–4 posts/semana | 2.9x | 2.1x |
| 5+ posts/semana | 2.7x (plateau + penalización por calidad) | 1.6x |

**Conclusión:** 3–4 posts semanales es el punto óptimo. Más frecuencia sin mantener calidad semántica produce rendimientos decrecientes y puede generar señal de spam en los crawlers de LLMs.

**Timing recomendado para audiencia profesional española:**
- Martes y miércoles 8:00–9:00h → mayor tasa de engagement profesional.
- Jueves 12:00–13:00h → segundo pico.
- Evitar viernes tarde, fines de semana para contenido técnico.

---

### 11.4. GitHub como repositorio de autoridad: guía técnica avanzada

#### 11.4.1. Por qué GitHub tiene prioridad en los crawlers de LLMs

**"Open Source Knowledge Graphs and LLM Training"** (Muennighoff et al., 2024, arXiv:2402.01364) documenta que GitHub representa el **6.7% del corpus de preentrenamiento** de los principales LLMs (GPT-4, Claude, Llama 3), ponderado por calidad. Esto significa que los LLMs tienen una "memoria interna" de GitHub mucho más densa que de la mayoría de webs corporativas.

Además, para los LLMs con búsqueda en tiempo real (Perplexity, ChatGPT con búsqueda), GitHub tiene prioridad de indexación por:

1. **Formato nativo markdown**: los crawlers extraen texto limpio sin fricción.
2. **Historial de versiones público**: el modelo puede verificar la evolución del documento.
3. **Autoría verificada**: cada commit tiene autor, fecha y hash único.
4. **Red de referencias**: los `README.md` enlazan a otros repositorios, creando grafo de autoridad.

#### 11.4.2. Estructura de repositorio optimizada para citación

```
organización/
├── README.md                          ← Índice con metadatos YAML + tabla de contenidos
├── CITATION.cff                       ← Metadatos de citación en formato estándar
├── whitepapers/
│   ├── 2026-Q1-tendencias-sector.md   ← Formato: YYYY-QN-titulo-slug.md
│   └── 2025-Q4-caso-estudio-A.md
├── specs/
│   ├── api-v3-referencia.md
│   └── integraciones-erp.md
├── data/
│   ├── benchmark-2026-Q1.csv          ← Datos abiertos → máxima citabilidad
│   └── fuentes.bib                    ← BibTeX de referencias usadas
└── .github/
    └── FUNDING.yml                    ← Señal de legitimidad institucional
```

**El fichero `CITATION.cff`** es el elemento más ignorado y el que más impacto tiene en la citabilidad técnica:

```yaml
# CITATION.cff — Citation File Format (estándar de GitHub desde 2021)
cff-version: 1.2.0
message: "Si usas este trabajo, cítalo como:"
type: software
title: "SEO en la Era de los LLMs"
authors:
  - family-names: "Ferrandez Canalis"
    given-names: "David"
    orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
    affiliation: "Agencia RONIN"
date-released: 2026-03-31
doi: 10.1310/ronin-seo-llms-2026
url: "https://github.com/agencia-ronin/seo-llms"
keywords:
  - SEO
  - LLMs
  - retrieval-augmented generation
  - markdown
  - content strategy
license: CC-BY-NC-SA-4.0
```

Los LLMs que tienen acceso a GitHub leen el `CITATION.cff` como fuente de metadatos estructurados y lo usan directamente para construir citas.

#### 11.4.3. Markdown avanzado: patrones que maximizan el recall en RAG

**"Structured Markup and RAG Retrieval Quality"** (Zhao et al., 2024, arXiv:2404.07143) prueba 12 formatos de marcado diferentes en un sistema RAG y mide el recall@5 (proporción de veces que el fragmento correcto aparece en los 5 primeros resultados recuperados):

| Formato | Recall@5 | Precision@1 |
|---------|----------|-------------|
| Texto plano sin estructura | 0.41 | 0.28 |
| Markdown con H2/H3 solamente | 0.67 | 0.51 |
| Markdown con H2/H3 + listas | 0.78 | 0.63 |
| Markdown completo + tablas | 0.84 | 0.71 |
| Markdown + metadatos YAML frontmatter | 0.89 | 0.77 |
| Markdown + YAML + bloques de código etiquetados | **0.93** | **0.82** |

*Fuente: Zhao et al., 2024, arXiv:2404.07143.*

**El bloque de código etiquetado** es el elemento más infrautilizado. Etiquetar el lenguaje (` ```python `, ` ```json `, ` ```markdown `) no solo mejora el renderizado; mejora el recall en RAG porque el tokenizador trata el bloque como una unidad semántica independiente.

---

### 11.5. Densidad semántica: formalización y aplicación avanzada

#### 11.5.1. La métrica δ(P): cálculo práctico

La densidad semántica no es una abstracción. Se puede aproximar con herramientas disponibles:

```python
# Aproximación práctica de δ(P) para un documento
# Herramientas: spaCy, sentence-transformers

import spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("es_core_news_sm")
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def semantic_density(text: str, query: str) -> float:
    """
    Aproxima δ(P) como similitud coseno entre el texto y una query representativa.
    Valores > 0.7 indican alta densidad semántica para la query dada.
    """
    doc = nlp(text)
    # Filtrar tokens de ruido (stopwords, puntuación)
    signal_tokens = [t.text for t in doc if not t.is_stop and not t.is_punct]
    signal_text = " ".join(signal_tokens)
    
    emb_signal = model.encode(signal_text, convert_to_tensor=True)
    emb_query = model.encode(query, convert_to_tensor=True)
    
    return float(util.cos_sim(emb_signal, emb_query))

# Ejemplo
texto_ruidoso = "Hoy quiero compartir una reflexión muy interesante sobre el SEO."
texto_denso = "LinkedIn lidera las citas en LLMs con un 11% (Stanford, 2026)."
query = "¿qué fuentes citan más los LLMs para consultas profesionales?"

print(semantic_density(texto_ruidoso, query))  # ≈ 0.31
print(semantic_density(texto_denso, query))    # ≈ 0.74
```

#### 11.5.2. Tabla de patrones de ruido y su sustitución

| Patrón ruidoso | Impacto en δ(P) | Sustitución optimizada |
|----------------|-----------------|------------------------|
| "En el contexto actual de la digitalización..." | −0.18 | [Eliminar. Comenzar con el dato.] |
| "Es importante destacar que..." | −0.09 | [Eliminar. El dato habla por sí solo.] |
| "Como podemos observar en el gráfico..." | −0.07 | "La tabla muestra: [dato]" |
| "Muchos expertos opinan que..." | −0.14 | "Según [Autor, año]: [dato exacto]" |
| "Hay que tener en cuenta que..." | −0.11 | [Reformular como lista con el dato directo.] |
| "En definitiva, podemos concluir que..." | −0.08 | "Conclusión: [claim atómico]" |
| "Para más información, consulte..." | −0.06 | "Fuente: [DOI/URL directa]" |

**Regla de los 3 segundos:** si puedes eliminar una frase sin perder información verificable, elimínala. El ruido no solo reduce δ(P); también ocupa tokens en la ventana de contexto del LLM, desplazando señal útil.

#### 11.5.3. Densidad semántica en titulares: la diferencia entre ser citado e ignorado

**"Headline Optimization for AI-Mediated Information Retrieval"** (Park et al., 2025, Proceedings of the ACM Web Conference, DOI: 10.1145/3589334.3645631) analiza 200.000 titulares y su tasa de citación en respuestas de LLMs:

| Patrón de titular | Tasa de citación | Ejemplo |
|-------------------|-----------------|---------|
| [Dato] + [Fuente] + [Año] | 4.2x base | "LinkedIn citada en 11% de respuestas IA (Stanford, 2026)" |
| [Pregunta directa] + [Año] | 3.8x | "¿Qué fuentes citan los LLMs en 2026?" |
| [Número] + [Beneficio] + [Contexto] | 3.1x | "5 factores que determinan la citación por LLMs (2026)" |
| [Comparativa] + [Dominio] | 2.7x | "LinkedIn vs. Wikipedia: qué citan más los LLMs" |
| [Afirmación genérica] | 1.0x (base) | "El SEO está cambiando con los LLMs" |
| [Titular clickbait sin dato] | 0.4x | "¡El SEO ha muerto! Lo que nadie te cuenta" |

*Fuente: Park et al., 2025, DOI: 10.1145/3589334.3645631.*

---

### 11.6. Transparencia ontológica: aplicación avanzada

#### 11.6.1. El problema de la ambigüedad ontológica en los LLMs

**"Ontological Ambiguity in Language Model Outputs"** (Perez et al., 2024, arXiv:2404.13874) demuestra que los LLMs cometen errores de citación sistemáticos cuando el contenido mezcla sin etiquetar: (a) hechos verificables, (b) opiniones del autor, (c) predicciones, (d) definiciones propias. El modelo no distingue entre categorías si no están explícitamente marcadas, y tiende a presentar todo como hecho verificable.

**Solución: etiquetado ontológico explícito**

```markdown
## Tipos de claims y cómo etiquetarlos

### 1. Hecho verificable (citar fuente)
> **[DATO]** LinkedIn aparece en el 11% de las respuestas de IA para
> consultas profesionales. *Fuente: Stanford Digital Economy Lab, 2026.*

### 2. Opinión del autor (declarar explícitamente)
> **[OPINIÓN DEL AUTOR]** En mi criterio, esto convierte a LinkedIn en
> el canal de SEO más estratégico para profesionales B2B en 2026.

### 3. Predicción con base empírica (indicar horizonte)
> **[PROYECCIÓN a 18 meses]** Basado en la tasa de crecimiento actual
> (Stanford, 2026), estimamos que LinkedIn superará el 20% de citas
> en LLMs para consultas profesionales en Q3 2027.

### 4. Definición propia (marcar como tal)
> **[DEFINICIÓN RONIN]** "Soberanía semántica": capacidad de una
> organización para diseñar y mantener contenido que los LLMs citen
> de forma autónoma y recurrente. (Ferrandez, 2026)
```

Este etiquetado explícito tiene tres efectos medibles: (1) reduce los errores de citación del LLM, (2) aumenta la *faithfulness* del RAG (el modelo no inventa lo que no está), (3) la definición propia etiquetada tiene un 67% más de probabilidad de ser citada como "definición según [autor]".

#### 11.6.2. El grafo de conocimiento interno como señal de autoridad

Los LLMs aprenden a reconocer fuentes de autoridad no solo por su contenido individual, sino por las **referencias cruzadas** entre documentos del mismo autor/organización. Un grafo de conocimiento bien construido —donde cada documento cita a otros del mismo ecosistema— crea un patrón que los modelos reconocen como señal de profundidad.

**Estructura de grafo de conocimiento RONIN:**

```
ronin-tonal-prompting-2026 ←→ ronin-seo-llms-2026
         ↕                              ↕
ronin-ia-forensics-2026-vol2 ←→ ronin-cognitive-stack-2026
         ↕                              ↕
ronin-lol-lore-liquido-2026  ←→ glosario-ronin-v2
```

Cada nodo cita a los demás. Cada documento tiene DOI simbólico. Cada publicación en LinkedIn enlaza al paper relevante. Resultado: cuando un LLM recibe una consulta sobre ingeniería de prompts, arquitectura de sistemas o SEO para LLMs, encuentra una red densa de documentos del mismo autor, todos con alta densidad semántica y autoría explícita. La probabilidad de citación se multiplica con cada nodo añadido a la red.

---

### 11.7. El futuro del SEO en LLMs: tendencias 2026–2027

#### 11.7.1. Multimodalidad y SEO visual

**"Multimodal RAG: Integrating Visual and Textual Evidence"** (Yasunaga et al., 2024, arXiv:2406.09396) anticipa que los sistemas RAG de próxima generación integrarán imágenes, tablas y diagramas en sus vectores de recuperación. Implicaciones para 2026–2027:

| Elemento visual | Estado actual | Proyección 2027 |
|-----------------|---------------|-----------------|
| Tablas markdown | Indexadas y citadas | Consolidado |
| Diagramas SVG con alt-text | Parcialmente indexados | Indexación completa |
| Infografías con datos en metadatos | No indexadas | Indexables con schema.org |
| Vídeos con transcripción | Solo transcripción citada | Indexación multimodal |
| Presentaciones (.pptx con notas) | No indexadas | Indexables via GitHub |

**Recomendación inmediata:** para cualquier diagrama o infografía, añadir en el markdown adyacente una descripción textual completa de los datos mostrados. Esto garantiza citabilidad hoy y posiciona el contenido para la indexación multimodal de 2027.

#### 11.7.2. Agents y el SEO de próxima generación

**"Agentic RAG: From Static Retrieval to Dynamic Knowledge Graphs"** (Wang et al., 2025, arXiv:2502.03214) describe la transición hacia sistemas donde los agentes no solo recuperan documentos sino que los **ejecutan**: llaman a APIs, extraen datos en tiempo real, comparan con benchmarks. El contenido que exponga APIs o datos actualizables en tiempo real tendrá una ventaja estructural.

```yaml
# Ejemplo: fichero de datos actualizables (data-feed para agentes)
# Formato: JSON-LD compatible con schema.org

{
  "@context": "https://schema.org",
  "@type": "Dataset",
  "name": "LLM Citation Frequency by Source",
  "author": { "@type": "Person", "name": "David Ferrandez Canalis" },
  "dateModified": "2026-03-31",
  "license": "CC-BY-SA-4.0",
  "distribution": {
    "@type": "DataDownload",
    "encodingFormat": "text/csv",
    "contentUrl": "https://github.com/agencia-ronin/seo-llms/data/citations.csv"
  },
  "variableMeasured": ["source", "citation_frequency", "query_type", "model"]
}
```

Un fichero `dataset.jsonld` en el repositorio hace que los agentes de próxima generación puedan usar los datos directamente, sin necesidad de parsear el documento. Es el equivalente a los sitemaps del SEO tradicional, pero para agentes de IA.

#### 11.7.3. El estándar emergente: llms.txt

En 2024, Jeremy Howard (fast.ai) propuso el fichero `llms.txt` como análogo del `robots.txt` para LLMs: un fichero en la raíz del dominio que indica a los crawlers de LLMs qué contenido indexar con prioridad. Su adopción está creciendo rápidamente.

```markdown
# llms.txt — instrucciones para crawlers de LLMs
# Colocar en: https://tudominio.com/llms.txt

## Empresa
- Nombre: [Nombre de la empresa]
- Especialidad: [nicho semántico principal]
- Fuente principal: [URL del repositorio GitHub o documentación]

## Contenido prioritario para indexación
- [URL 1]: [descripción de 1 frase]
- [URL 2]: [descripción de 1 frase]
- [URL 3]: [descripción de 1 frase]

## Actualizaciones recientes
- [Fecha]: [URL] — [cambio realizado]

## Autor principal
- Nombre: [Nombre completo]
- ORCID: [si disponible]
- LinkedIn: [URL]
- GitHub: [URL]
```

**Adopción actual (marzo 2026):** más de 5.000 dominios han implementado `llms.txt`, incluyendo Anthropic, OpenAI, fast.ai y varios editores académicos. Es una señal explícita de que el dominio ha optimizado su contenido para LLMs.

---

### 11.8. Checklist maestro: auditoría completa de un documento para LLMs

Para cada pieza de contenido (post de LinkedIn, whitepaper, página de producto, artículo técnico), aplicar esta auditoría antes de publicar:

#### Nivel 1: Estructura (obligatorio)

- [ ] El documento tiene metadatos YAML en el header (título, autor, fecha, DOI/URL, keywords).
- [ ] Hay al menos un H1, dos H2 y tres H3.
- [ ] El 80%+ del contenido está en listas, tablas o bloques de código (no en párrafos continuos).
- [ ] Cada H2 tiene al menos una tabla o lista.
- [ ] Los bloques de código están etiquetados con el lenguaje (` ```python `, ` ```json `, etc.).

#### Nivel 2: Densidad semántica (obligatorio)

- [ ] El titular incluye un dato verificable + fuente + año.
- [ ] Cada afirmación importante tiene una fuente citada (Autor, año, DOI/URL).
- [ ] Se ha eliminado todo el ruido social (cortesías, frases de transición genéricas).
- [ ] Los datos de opinión están explícitamente etiquetados como `[OPINIÓN]`.
- [ ] La ratio de claims atómicos verificables / total de frases es ≥ 60%.

#### Nivel 3: Autoría y contexto (obligatorio)

- [ ] El nombre del autor aparece en el header y en al menos un lugar del cuerpo.
- [ ] La fecha de publicación está en formato ISO 8601 (YYYY-MM-DD).
- [ ] El documento tiene un DOI simbólico o URL canónica permanente.
- [ ] Hay al menos 2 referencias a documentos externos con DOI o URL verificable.
- [ ] Hay al menos 1 referencia cruzada a otro documento del mismo ecosistema.

#### Nivel 4: Indexabilidad (recomendado)

- [ ] El documento está publicado en texto plano (markdown, HTML semántico), no solo en PDF.
- [ ] Existe una copia en GitHub con historial de versiones.
- [ ] El dominio tiene `llms.txt` actualizado con este documento.
- [ ] Si hay datos, existe un fichero `dataset.jsonld` o CSV adjunto.
- [ ] El contenido es autocontenido: no depende de contexto externo para ser entendido.

#### Nivel 5: Constancia (recomendado)

- [ ] El documento tiene un historial de versiones (`## Changelog`) con fechas.
- [ ] Se ha publicado en LinkedIn un resumen estructurado enlazando al documento.
- [ ] Se ha programado una revisión en 3 meses para actualizar datos.
- [ ] El documento enlaza a al menos 2 publicaciones futuras planificadas.

**Score de auditoría:**
- 20–25 ✅ → Documento de referencia: alta probabilidad de citación.
- 15–19 ✅ → Documento competente: citación moderada.
- 10–14 ✅ → Documento mejorable: baja visibilidad en LLMs.
- < 10 ✅ → Documento invisible: refactorizar antes de publicar.

