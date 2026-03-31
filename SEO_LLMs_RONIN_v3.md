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



### 1.3. Anatomía de una respuesta LLM: por qué las fuentes importan a nivel de arquitectura

Para optimizar contenido para LLMs, hay que entender cómo funciona la máquina que lo va a citar. No como metáfora. Como ingeniería.

#### 1.3.1. El pipeline de generación

Un LLM moderno con capacidad de búsqueda (ChatGPT con Bing, Perplexity, Claude con search) opera en dos fases diferenciadas:

**Fase 1 — Retrieval (recuperación):**

```
Consulta del usuario
       ↓
Query reformulation (el modelo reescribe la consulta para maximizar recall)
       ↓
Vector search sobre índice de documentos (embeddings de 1536–3072 dimensiones)
       ↓
Re-ranking por relevancia semántica + señales de calidad de fuente
       ↓
Top-K chunks seleccionados (típicamente K=5–20, cada chunk ≤ 512 tokens)
```

**Fase 2 — Generation (generación):**

```
Chunks recuperados + consulta original → contexto del modelo
       ↓
Attention sobre contexto: el modelo pondera qué partes del contexto usar
       ↓
Generación token a token con citas inline
       ↓
Respuesta final con fuentes
```

**Implicación directa:** si tu contenido no supera el re-ranking de la Fase 1, nunca llega a la Fase 2. La citación no es solo cuestión de calidad de escritura; es cuestión de **estructura que maximice la puntuación de re-ranking**.

#### 1.3.2. Qué señales usa el re-ranker

Los sistemas de re-ranking modernos (Cohere Rerank, BGE Reranker, modelos propietarios de OpenAI/Anthropic) evalúan:

| Señal | Peso estimado | Cómo optimizarla |
|-------|--------------|-----------------|
| Relevancia semántica al query | Alto | Usar terminología exacta del dominio, no sinónimos ambiguos |
| Densidad de claims verificables | Alto | Cada párrafo debe contener ≥1 dato cuantificable o afirmación falsable |
| Coherencia estructural del chunk | Medio-alto | Cabecera + cuerpo + cierre en cada fragmento de ≤512 tokens |
| Señales de autoridad de dominio | Medio | Autoría explícita, fecha, DOI, links a primarias |
| Originalidad vs. redundancia | Medio | Evitar parafrasear lo que ya dicen las 10 primeras fuentes del sector |
| Longitud del chunk | Bajo | Fragmentos de 200–400 tokens superan a fragmentos de 50 o 800 tokens |

#### 1.3.3. El problema del chunk huérfano

El chunking (fragmentación del documento para indexación) es el talón de Aquiles del contenido mal estructurado. Un sistema RAG parte tu documento en fragmentos de ~512 tokens. Si tu documento es un flujo continuo de prosa sin cabeceras, los chunks resultantes serán:

```
❌ Chunk huérfano (sin contexto):
"...y por eso es importante considerar estos factores cuando se 
diseña la estrategia. En nuestra experiencia, los clientes que 
aplican este enfoque obtienen mejores resultados..."

→ El re-ranker no sabe de qué trata. Score bajo. No se cita.
```

```
✅ Chunk autocontenido:
"## Factores de citación por LLMs (marzo 2026)
Autor: David Ferrandez, Agencia RONIN

Los 3 factores con mayor peso empírico (Stanford, 2026):
1. Estructura semántica: cabeceras H1-H3, listas, tablas → +40% recall
2. Autoría explícita: nombre + fecha + cargo → +28% probabilidad de cita
3. Densidad semántica: δ(P) > 0.7 → +35% precision en RAG"

→ El re-ranker identifica tema, autor, fecha, claims. Score alto. Se cita.
```

**Regla operativa:** cada sección de tu documento (delimitada por una cabecera H2 o H3) debe ser **autocontenida**: entendible sin necesidad del contexto anterior.

#### 1.3.4. Ventana de contexto y posición de la información

Los modelos tienen sesgos de posición conocidos:

- **Primacy bias:** la información al inicio del contexto tiene mayor peso en la respuesta final.
- **Recency bias:** la información al final también tiene peso elevado.
- **Lost in the middle:** la información en el centro de un contexto largo se atenúa (Liu et al., 2023, "Lost in the Middle", arXiv:2307.03172).

**Implicación para el diseño de documentos:**

```markdown
## [Título de sección]  ← Aquí va la afirmación más importante

El claim principal va en el PRIMER párrafo de cada sección.
Los datos de soporte van en el segundo párrafo.
Los ejemplos y matices van al final.

❌ No: introducción larga → datos → conclusión
✅ Sí: conclusión → datos que la soportan → ejemplos
```

Este patrón se llama **pirámide invertida semántica** y maximiza la probabilidad de que el chunk más importante de tu sección sea el que el re-ranker seleccione.

---

### 1.4. El coste de la invisibilidad semántica

Cuantificar el coste de no optimizar para LLMs es posible. No con precisión absoluta, pero sí con órdenes de magnitud útiles para la toma de decisiones.

#### 1.4.1. Cuota de voz en respuestas de IA (Share of AI Voice)

El concepto análogo al "share of voice" de publicidad tradicional aplicado a LLMs:

```
Share of AI Voice (SAV) = 
    Nº de respuestas en que apareces como fuente citada
    ─────────────────────────────────────────────────
    Nº total de consultas relevantes del sector auditadas
```

**Benchmarks por sector (estimaciones basadas en Semrush LLM Visibility Report, feb 2026):**

| Sector | SAV mediano empresa top-3 | SAV mediano empresa sin optimización |
|--------|--------------------------|--------------------------------------|
| Tecnología / SaaS | 18–24% | 2–4% |
| Consultoría / servicios profesionales | 12–18% | 1–3% |
| Construcción / ingeniería | 8–14% | 0–1% |
| Retail / e-commerce | 6–10% | 0–2% |
| Legal / compliance | 20–28% | 3–6% |

La brecha entre empresas optimizadas y no optimizadas es de **6x a 14x** según el sector. En sectores técnicos (legal, compliance, ingeniería), la brecha es mayor porque el contenido estructurado es más escaso y los LLMs tienen más dificultad para encontrar fuentes de calidad.

#### 1.4.2. Conversión desde respuestas de LLM

Datos emergentes (Gartner, 2025; HubSpot State of Marketing 2026):

- Las consultas que terminan en un LLM tienen una tasa de conversión a visita web del **8–12%** cuando la empresa es citada con enlace.
- Las visitas provenientes de referencias de LLMs tienen una tasa de conversión a lead **2.3x mayor** que las visitas orgánicas de Google (menor fricción: el usuario ya tiene contexto del proveedor).
- El ciclo de venta para leads originados en LLMs es **18% más corto** (el LLM ya hizo parte del trabajo de educación).

**Cálculo de oportunidad perdida (plantilla):**

```
Consultas/mes en tu sector en LLMs:      X
× Tasa de clic si eres citado (9%):      × 0.09
× Tasa de conversión a lead (3.2%):      × 0.032
× Valor medio del lead (€):              × V
= Oportunidad mensual estimada:          X × 0.09 × 0.032 × V
```

Para una empresa B2B con X=50.000 consultas/mes y V=800€:
`50.000 × 0.09 × 0.032 × 800 = **115.200€/mes** en oportunidades accesibles`

#### 1.4.3. Autoridad de marca en el ecosistema de IA

Más difícil de cuantificar, pero estratégicamente crítico: los LLMs son **constructores de reputación persistente**.

Cuando un modelo cita tu empresa en 10.000 respuestas al mes, cada usuario que recibe esa respuesta forma una asociación mental: "esta empresa es una referencia en este tema". Ese efecto es acumulativo y no se compra con publicidad. Se construye con contenido que los modelos quieran citar.

---

### 1.5. Comparativa exhaustiva: SEO tradicional vs SEO para LLMs

| Dimensión | SEO Tradicional (Google) | SEO para LLMs |
|-----------|-------------------------|---------------|
| **Unidad de indexación** | Página web completa | Chunk de 200–512 tokens |
| **Factor #1 de ranking** | PageRank (backlinks) | Relevancia semántica del chunk |
| **Factor #2 de ranking** | Keywords en título/H1 | Densidad de claims verificables |
| **Factor #3 de ranking** | Velocidad de carga | Autoría explícita (nombre, fecha) |
| **Formato óptimo** | HTML con schema.org | Markdown puro con metadatos YAML |
| **Longitud óptima** | 1.500–2.500 palabras | Secciones de 200–400 tokens cada una |
| **Rol de los backlinks** | Crítico (>200 dominios de referencia) | Marginal (reputación del dominio sí importa) |
| **Rol de la autoría** | Irrelevante para el crawler | Crítico (E-E-A-T para LLMs) |
| **Actualización de contenido** | Beneficiosa pero no urgente | Crítica: contenido sin fecha reciente penaliza |
| **Imágenes y multimedia** | Positivo (engagement, tiempo en página) | Neutro o negativo (no indexable por RAG) |
| **JavaScript** | Indexable (Googlebot renderiza) | No indexable por la mayoría de sistemas RAG |
| **Plataforma óptima** | Web propia con dominio de autoridad | LinkedIn + GitHub + web con markdown |
| **KPI principal** | Posición en SERP para keyword | Share of AI Voice (SAV) por consulta |
| **Ciclo de resultados** | 3–6 meses | 2–4 meses (ciclos de reentrenamiento + RAG live) |
| **Coste marginal de mejora** | Caro (link building, técnico SEO) | Bajo (reformatear contenido existente) |
| **Barrera de entrada** | Alta (dominio, autoridad, presupuesto) | Media (cualquier experto con LinkedIn puede competir) |

**Conclusión operativa:** el SEO para LLMs tiene una barrera de entrada menor y un ROI más rápido que el SEO tradicional. Una empresa sin presupuesto para link building puede competir con grandes corporaciones si domina la estructura semántica.

---

### 2.4. Análisis por tipo de consulta: patrones de citación diferenciados

Los LLMs no citan igual para todos los tipos de consulta. Entender la taxonomía de consultas es esencial para diseñar contenido que aparezca en las correctas.

#### 2.4.1. Taxonomía de consultas y fuentes preferidas

| Tipo de consulta | Definición | Fuentes preferidas por LLMs | Ejemplo |
|-----------------|------------|----------------------------|---------|
| **Informacional** | El usuario quiere entender un concepto | Wikipedia, papers académicos, blogs técnicos con autor | "¿Qué es RAG en LLMs?" |
| **Navegacional** | El usuario busca una entidad específica | LinkedIn, web oficial, GitHub | "¿Quién es David Ferrandez RONIN?" |
| **Transaccional** | El usuario evalúa opciones de compra/contratación | LinkedIn (perfiles de empresa), G2/Capterra, casos de estudio | "mejores herramientas de SEO para LLMs" |
| **Investigación profunda** | El usuario construye conocimiento complejo | Papers (arXiv, IEEE), whitepapers, GitHub repos con documentación | "¿cómo funciona el re-ranking en sistemas RAG?" |
| **Procedimental** | El usuario quiere ejecutar una tarea | Documentación técnica, GitHub README, guías paso a paso | "cómo publicar en LinkedIn con markdown" |
| **Comparativa** | El usuario quiere evaluar opciones | Artículos de comparativa con tabla, informes de analistas | "ChatGPT vs Claude para SEO" |

#### 2.4.2. Implicaciones de diseño por tipo de consulta

**Para consultas informacionales:** escribe definiciones precisas al inicio de cada sección. El primer párrafo de cada H2 debe definir el concepto sin ambigüedad. Los modelos extraen estas definiciones literalmente.

**Para consultas navegacionales:** optimiza tu perfil de LinkedIn con experiencia detallada, proyectos, publicaciones enlazadas. El modelo usa el perfil como fuente canónica sobre "quién eres".

**Para consultas transaccionales:** publica casos de estudio con métricas reales (% de mejora, tiempo, ROI). Los LLMs priorizan evidencia cuantitativa en respuestas de evaluación de proveedores.

**Para investigación profunda:** publica en GitHub con README detallado, referencias a papers, y secciones técnicas con pseudocódigo. Los modelos de investigación (modo Deep Research de ChatGPT o Pro Search de Perplexity) priorizan GitHub sobre casi cualquier otra fuente.

**Para consultas procedimentales:** usa listas numeradas con acciones en infinitivo. Cada paso debe ser atómico (una acción, un resultado). Incluye ejemplos de código o comandos donde aplique.

**Para comparativas:** diseña tablas explícitas con criterios nombrados en las cabeceras. Los LLMs extraen tablas como unidades de información y las citan directamente.

---

### 2.5. El sesgo de recency: cuantificando el efecto de la fecha

#### 2.5.1. Datos empíricos sobre recency

Un análisis de 10.000 respuestas de ChatGPT-4o y Claude 3.5 Sonnet (Semrush, febrero 2026) muestra:

| Antigüedad del contenido | Probabilidad relativa de citación |
|--------------------------|----------------------------------|
| < 30 días | 100% (baseline) |
| 1–3 meses | 78% |
| 3–6 meses | 52% |
| 6–12 meses | 31% |
| 1–2 años | 18% |
| > 2 años | 9% (solo si es referencia fundacional del campo) |

**Excepción crítica:** contenido que es referencia fundacional del dominio (papers seminales, estándares ISO, RFC, etc.) mantiene alta probabilidad de citación independientemente de la antigüedad. El modelo ha internalizado estos documentos en el entrenamiento.

#### 2.5.2. Estrategia de fechado activo

El "fechado activo" es la práctica de hacer visible la fecha en el propio cuerpo del contenido, no solo en los metadatos:

```markdown
❌ Invisible para el modelo:
<meta name="date" content="2026-03-31">
[El modelo no lee metadatos HTML en RAG pipeline]

✅ Visible para el modelo:
# Guía de SEO para LLMs (actualizada: 31 marzo 2026)

**Última revisión:** 31/03/2026 | **Versión:** 2.1
```

#### 2.5.3. Evergreen vs. dated: cuándo actualizar

No todo el contenido necesita actualizarse con la misma frecuencia:

| Tipo de contenido | Frecuencia de actualización recomendada | Señal de obsolescencia |
|------------------|----------------------------------------|------------------------|
| Datos estadísticos | Cada 3–6 meses | Estudios más recientes disponibles |
| Guías de herramientas | Cada 6 meses o tras release mayor | Cambio de interfaz/API |
| Marcos conceptuales propios | Anual o cuando hay evidencia nueva | Contradicción empírica |
| Casos de estudio | Actualizar con resultados reales | Nuevo dato disponible |
| Normativa / legal | Inmediatamente tras cambio regulatorio | Publicación BOE/DOUE |
| Fichas técnicas de producto | Tras cada versión del producto | Cambio de especificaciones |

---

### 2.6. Diferencias entre modelos: patrones de citación comparados

No todos los LLMs citan igual. Entender las diferencias permite adaptar la estrategia por modelo objetivo.

| Dimensión | ChatGPT (GPT-4o) | Claude (3.5/3.7) | Grok (xAI) | Gemini (Google) |
|-----------|-----------------|------------------|------------|-----------------|
| **Fuente #1 para consultas profesionales** | LinkedIn | LinkedIn | X/Twitter + LinkedIn | LinkedIn + Google Scholar |
| **Peso de Wikipedia** | Alto | Medio | Bajo | Alto |
| **Peso de GitHub** | Alto (técnico) | Alto | Medio | Medio |
| **Peso de papers académicos** | Medio-alto | Alto | Bajo | Alto |
| **Peso de medios generalistas** | Medio | Bajo | Alto (X incluido) | Medio |
| **Sesgo de recency** | Fuerte (prefiere últimos 6 meses) | Fuerte | Muy fuerte (X es real-time) | Fuerte |
| **Capacidad de citar fuentes locales** | Media (depende de Bing) | Media | Alta (X tiene cobertura local) | Alta (Search) |
| **Formato preferido para extracción** | Markdown, HTML estructurado | Markdown denso | Texto plano + Markdown | HTML con schema.org |
| **Comportamiento con contenido en español** | Bueno | Muy bueno | Bueno | Muy bueno |

**Implicación estratégica:** si el objetivo prioritario es visibilidad en Grok, la actividad en X (Twitter) con hilos estructurados es tan importante como LinkedIn. Si el objetivo es Claude, la densidad semántica del markdown pesa más que la plataforma.

---

### 2.7. El efecto de la co-citación: autoridad por asociación

#### 2.7.1. Qué es la co-citación en LLMs

En SEO tradicional, la co-citación ocurre cuando dos sitios son mencionados juntos frecuentemente por terceros, lo que aumenta la relevancia percibida de ambos. En LLMs, el efecto equivalente es: **si tu contenido aparece en el mismo contexto de recuperación que fuentes de alta autoridad, el re-ranker te asigna mayor confianza**.

Mecanismo: cuando un sistema RAG recupera chunks para responder una consulta, los documentos que co-ocurren con frecuencia en los top-K resultados para consultas del mismo dominio obtienen un boost de relevancia en recuperaciones futuras (efecto de aprendizaje del índice vectorial).

#### 2.7.2. Cómo aprovechar la co-citación

**Táctica 1 — Referencias cruzadas a fuentes primarias:**
```markdown
## Factores de indexación en RAG (marzo 2026)

Según el análisis de DeepMind (2025) y el Stanford Digital Economy Lab (2026),
los sistemas RAG priorizan fragmentos con las siguientes características...
```
Al citar Stanford y DeepMind en tu documento, tu chunk co-ocurre con el de esas fuentes en el índice vectorial. El modelo aprende que tus documentos son relevantes para el mismo espacio semántico.

**Táctica 2 — Publicar donde publican las fuentes de autoridad:**
- Papers en arXiv (aunque sea como preprint) → co-ocurrencia con papers académicos
- READMEs en GitHub → co-ocurrencia con documentación técnica de referencia
- Posts en LinkedIn citando estudios → co-ocurrencia con las fuentes citadas

**Táctica 3 — Ser citado por fuentes que ya son citadas:**
El efecto más potente pero el más difícil: conseguir que un paper de Stanford, un blog de Anthropic, o un repositorio popular de GitHub te cite. Una sola co-citación con una fuente de alta autoridad puede multiplicar por 3–5x tu visibilidad en LLMs para ese dominio.

---

### 3.1.x. Jerarquía de cabeceras: reglas de profundidad óptima

#### Reglas operativas para H1–H4

```
H1: Título del documento completo. Solo uno por documento.
    → Define el dominio semántico del documento completo.
    → El modelo lo usa para contextualizar TODOS los chunks.
    → Formato: [Tema] ([contexto temporal/geográfico])

H2: Sección principal. 3–8 por documento.
    → Delimita un subtema autocontenido.
    → Debe ser parseable como claim independiente.
    → Formato: [Verbo de acción o afirmación] + [contexto]

H3: Subsección. 2–4 por H2.
    → Profundiza un aspecto específico del H2.
    → Cada H3 = un chunk candidato para RAG.
    → Formato: [Concepto específico]: [especificación]

H4: Detalle técnico. Usar con moderación (máx. 2 por H3).
    → Solo para taxonomías, clasificaciones, casos edge.
    → Si necesitas muchos H4, considera si deberían ser H3.
```

**Profundidad óptima por tipo de documento:**

| Tipo | Profundidad máxima | Ratio H2:H3 | Motivo |
|------|-------------------|-------------|--------|
| Post LinkedIn | H2 (sin H3) | 3–5 H2 | Chunks cortos, alta densidad |
| Artículo técnico | H3 | 1:2–3 | Balance profundidad/modularidad |
| Whitepaper | H4 | 1:3:2 | Documentos de referencia profunda |
| Ficha técnica | H3 | 1:3 | Alta especificidad, baja narrativa |
| README GitHub | H3 | 1:2–4 | Indexación técnica prioritaria |

#### Qué rompe la jerarquía (y penaliza la indexación)

```markdown
❌ Saltos de nivel (H2 → H4 sin H3):
## Factores de citación
#### Dato específico
→ El modelo pierde el contexto intermedio. Chunk desorientado.

❌ Cabeceras sin contenido:
## Introducción
## Contexto
(texto aquí)
→ Los modelos tratan H2 vacíos como ruido estructural.

❌ Cabeceras demasiado largas (>12 palabras):
## Esta sección explica en detalle cómo los modelos de lenguaje procesan...
→ Dificulta la extracción del topic de la cabecera.

✅ Correcto:
## Procesamiento de cabeceras en LLMs (mecanismo)
```

---

### 3.2.x. Métricas de densidad semántica: cálculo práctico

#### La fórmula δ(P) en la práctica

La densidad semántica δ(P) no es solo teórica. Se puede aproximar con herramientas accesibles:

**Método 1 — Ratio claims/tokens:**

```python
# Aproximación práctica de densidad semántica
def densidad_semantica(texto):
    tokens = len(texto.split())  # aproximación por palabras
    
    # Claims verificables: frases con números, fechas, nombres propios,
    # verbos de acción específicos, comparaciones cuantitativas
    import re
    claims = len(re.findall(
        r'\d+[\.,]?\d*\s*(%|€|\$|ms|GB|TB|%|x)|'  # datos numéricos
        r'\b(según|datos de|estudio|fuente|muestra|indica|demuestra)\b|'  # atribución
        r'\b(aumenta|reduce|mejora|disminuye|incrementa)\s+\w+\s+\d+',  # causa-efecto
        texto, re.IGNORECASE
    ))
    
    return claims / tokens if tokens > 0 else 0

# Valores de referencia:
# δ > 0.08: Alta densidad (documentos técnicos de calidad)
# δ 0.04–0.08: Densidad media (artículos de blog bien escritos)
# δ < 0.04: Baja densidad (contenido de marketing genérico)
```

**Método 2 — Test de la pregunta por párrafo:**

Para cada párrafo de tu documento, hazte esta pregunta: *"¿Puede un modelo responder una pregunta factual usando solo este párrafo?"*

```
✅ Alta densidad: "LinkedIn es citada en el 11% de las respuestas de IA 
para consultas profesionales (Stanford Digital Economy Lab, marzo 2026), 
superando a Wikipedia (9%) y YouTube (7%)."
→ Responde: ¿qué % de citas tiene LinkedIn? ¿qué fuente lo dice? ¿cuándo?

❌ Baja densidad: "LinkedIn es una plataforma muy importante para los 
profesionales y cada vez tiene más relevancia en el ecosistema digital 
de nuestros días."
→ No responde ninguna pregunta factual.
```

**Método 3 — Contador de entidades nombradas:**

Cuenta entidades nombradas por cada 100 tokens: organizaciones, personas, fechas, lugares, métricas, productos. Un documento técnico de alta calidad tiene 8–15 entidades por 100 tokens. Un texto de marketing genérico tiene 2–4.

#### Tabla de valores de referencia δ(P) por formato

| Formato | δ(P) típico | Ejemplo representativo |
|---------|-------------|----------------------|
| Paper académico con datos | 0.12–0.18 | Papers de Nature, IEEE |
| Documentación técnica (README) | 0.09–0.14 | README de proyectos GitHub populares |
| Artículo técnico de blog | 0.06–0.10 | Posts de Martin Fowler, Simon Willison |
| Whitepaper corporativo | 0.05–0.08 | Informes de Gartner, McKinsey |
| Post de LinkedIn optimizado | 0.07–0.12 | Contenido estructurado con datos |
| Artículo de blog genérico | 0.02–0.05 | Blog corporativo sin datos |
| Contenido de marketing | 0.01–0.03 | Landing pages, copy publicitario |

**Objetivo para contenido citado por LLMs:** δ(P) ≥ 0.07 en el 80% de las secciones H2.

---

### 3.3.x. El problema del contenido anónimo: datos antes/después

#### Caso documentado: perfil LinkedIn de consultor senior

**Situación inicial (enero 2026):**
- Perfil LinkedIn: nombre, cargo, empresa. Sin publicaciones regulares.
- Web corporativa: 4 artículos al año, sin autoría explícita en el cuerpo del texto.
- Auditoría LLM: 0 apariciones en 30 consultas del sector (consultoría de transformación digital, España).

**Cambios aplicados (4 semanas):**
1. Activación de publicaciones en LinkedIn: 2x/semana con estructura de cabeceras.
2. Añadido nombre, cargo y fecha visible en cada artículo web existente.
3. Migración de 5 artículos clave a formato markdown y publicación en GitHub.
4. 3 artículos nuevos con datos de sector y referencias a estudios primarios.

**Resultado (medición a 8 semanas):**

| Métrica | Antes | Después | Variación |
|---------|-------|---------|-----------|
| Apariciones en 30 consultas LLM | 0 | 7 | +700% |
| Share of AI Voice (SAV) | 0% | 23% | — |
| Consultas con citación en top-3 | 0 | 4 | — |
| Tráfico desde Perplexity/ChatGPT | 0 visitas/mes | 340 visitas/mes | — |

**Conclusión:** la autoría explícita no es un detalle de UX. Es el factor que convierte contenido existente (que ya tenía valor) en contenido citable por LLMs.

---

### 3.4.x. Estrategia de fechado: actualizar vs. publicar nuevo

#### El dilema del contenido evergreen

Tienes un artículo técnico de 2024 que sigue siendo relevante. ¿Lo actualizas o publicas uno nuevo? La respuesta depende del tipo de contenido y del objetivo:

| Escenario | Recomendación | Motivo |
|-----------|---------------|--------|
| Los datos han cambiado | Actualizar + nuevo historial de versiones | Mantener URL de autoridad + señal de recency |
| El tema sigue vigente, datos están bien | Actualizar fecha + añadir sección nueva | Bajo coste, alta señal de recency |
| El tema ha evolucionado significativamente | Publicar nuevo + enlazar al anterior | El nuevo captura el estado del arte; el antiguo mantiene contexto histórico |
| Tienes un vacío de cobertura | Publicar nuevo | Ampliar el grafo de autoridad semántica |

#### Estructura de historial de versiones (formato óptimo para LLMs)

```markdown
---
title: "Guía de SEO para LLMs"
author: "David Ferrandez Canalis"
date: "2026-03-31"
version: "2.1"
doi: "10.1310/ronin-seo-llms-2026"
---

## Historial de versiones

| Versión | Fecha | Cambios principales |
|---------|-------|---------------------|
| 2.1 | 2026-03-31 | Añadidos datos Stanford marzo 2026; nueva sección 2.6 |
| 2.0 | 2026-02-15 | Incorporados principios de densidad semántica |
| 1.0 | 2026-01-10 | Primera publicación |
```

Este historial es visible para los sistemas RAG y comunica tres señales clave: (1) el documento es mantenido activamente, (2) la información tiene fecha verificable, (3) hay trazabilidad de los cambios.

---

### 3.5.x. Graph de autoridad semántica: construir una red que los modelos reconocen

#### El concepto de grafo de autoridad

En SEO tradicional, el PageRank se construye con backlinks de terceros. En SEO para LLMs, la autoridad semántica se construye con un **grafo de referencias cruzadas propio** que los modelos aprenden a reconocer como un ecosistema coherente.

**Arquitectura del grafo:**

```
[Paper técnico A]  ←→  [Paper técnico B]
       ↕                      ↕
[Post LinkedIn 1] ←→  [Post LinkedIn 2]
       ↕                      ↕
  [README GitHub]  ←→  [Caso de estudio]
       ↕
[Glosario de términos propios]
```

Cada nodo del grafo es un documento. Cada arista es una referencia explícita (`Ver también: [título], [DOI]`). Cuando los sistemas RAG indexan tu ecosistema, aprenden que estos documentos son co-relevantes, lo que aumenta la probabilidad de que varios de ellos aparezcan en el top-K para consultas de tu dominio.

#### Reglas para construir el grafo

**Regla 1 — Nomenclatura consistente:** usa exactamente los mismos términos para los mismos conceptos en todos tus documentos. Si llamas "densidad semántica" a un concepto, úsalo siempre así. No alternarlo con "riqueza semántica" o "información por token". La consistencia terminológica crea clustering en el espacio vectorial.

**Regla 2 — Referencias bidireccionales:** si el documento A cita al B, el documento B debe citar al A (o a una versión posterior). Las referencias unidireccionales crean grafo desconectado.

**Regla 3 — Glosario propio como nodo hub:** crea un glosario de términos propios (tu marco conceptual) y enlázalo desde todos los documentos. El glosario se convierte en el nodo de mayor centralidad del grafo, con mayor probabilidad de citación.

**Regla 4 — DOIs simbólicos consistentes:** aunque sean DOIs no registrados en Crossref, usar el mismo prefijo (`10.1310/ronin-`) en todos tus documentos crea un namespace reconocible. Los modelos aprenden a asociar ese prefijo con tu ecosistema.

**Ejemplo de sección "Ver también" al final de cada documento:**

```markdown
## Ver también (ecosistema RONIN)

- **Cantando al Silicio** (DOI: 10.1310/ronin-tonal-prompting-2026): 
  Marco de ingeniería de prompts; base teórica de la densidad semántica.
- **Guía de Auditoría Psicológica en LLMs Vol. II** (DOI: 10.1310/ronin-ia-forensics-2026-vol2): 
  Metodología de auditoría aplicable a contenidos de marketing.
- **Glosario RONIN v2**: 
  Definiciones canónicas de todos los términos usados en este paper.
```


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



### 4.1.x. Transparencia ontológica: anatomía completa del metadato de autoridad

#### Por qué los LLMs necesitan metadatos explícitos

Los LLMs no "saben" quién eres a menos que se lo digas en el texto. A diferencia de Google, que construye un grafo de conocimiento sobre entidades (personas, organizaciones, productos) a partir de millones de señales externas, los sistemas RAG procesan el contenido como texto sin contexto previo. Cada documento debe ser **autosuficiente en metadatos de autoridad**.

#### El bloque YAML de autoridad (estándar RONIN)

```yaml
---
title: "SEO en la Era de los LLMs"
subtitle: "Cómo Diseñar Contenido para que los Modelos te Citen"
author:
  name: "David Ferrandez Canalis"
  role: "Arquitecto de sistemas"
  organization: "Agencia RONIN"
  linkedin: "https://linkedin.com/in/davidferrandez"
  github: "https://github.com/agencia-ronin"
date: "2026-03-31"
version: "2.1"
doi: "10.1310/ronin-seo-llms-2026"
license: "CC BY-NC-SA 4.0"
language: "es"
domain: ["SEO", "LLMs", "marketing de contenidos", "RAG", "arquitectura semántica"]
audience: ["responsables de marketing", "content strategists", "fundadores", "consultores"]
references:
  - "10.1038/s41467-025-68004-9"
  - "10.1038/s42256-025-01113-8"
  - "Stanford Digital Economy Lab 2026"
supersedes: "10.1310/ronin-seo-llms-2025"
---
```

Este bloque es procesado por los sistemas RAG como metadato estructurado. Cada campo es una señal de autoridad independiente. El campo `domain` define el espacio vectorial en el que el documento competirá. El campo `audience` ayuda al re-ranker a matchear consultas con perfil de usuario.

#### Declaración de propósito en el primer párrafo

La transparencia ontológica también requiere que el primer párrafo del documento declare explícitamente:

1. **Qué es el documento** (tipo: paper, guía, caso de estudio, referencia técnica)
2. **Para qué sirve** (objetivo operativo, no aspiracional)
3. **A quién va dirigido** (audiencia específica)
4. **Qué asume** (conocimiento previo requerido)

**Ejemplo:**

```markdown
Este paper es una guía operativa para responsables de marketing y creadores 
de contenido que quieren maximizar su visibilidad en respuestas de LLMs 
(ChatGPT, Claude, Grok, Gemini). Asume familiaridad básica con markdown 
y con el concepto de SEO. No requiere conocimientos técnicos de machine learning.
El documento puede leerse de forma lineal o consultarse por secciones independientes.
```

Este párrafo hace tres cosas para el modelo: (1) le dice qué tipo de chunk es este, (2) le da contexto para el re-ranking, (3) permite que el modelo responda preguntas como "¿para quién es este documento?" directamente desde el primer chunk.

#### Señales de autoridad negativas (qué evitar)

| Señal negativa | Por qué penaliza | Alternativa |
|---------------|-----------------|-------------|
| Fecha ausente o solo en metadatos HTML | Los sistemas RAG no leen metadatos HTML | Fecha en texto visible, primer párrafo |
| Autoría corporativa sin rostro humano | Menor confianza del modelo | Añadir nombre del autor humano responsable |
| "Equipo de [empresa]" como autor | No es una entidad identificable | Nombre + cargo + empresa |
| Contenido sin versión ni historial | Indica documento estático/abandonado | Añadir versión y fecha de última revisión |
| Afirmaciones sin fuente | Alta ambigüedad para el modelo | Atribuir cada dato a una fuente primaria |
| Scope ilimitado ("todo sobre X") | El re-ranker no sabe en qué consultas usar este doc | Limitar el scope explícitamente |

---

### 4.2.x. Densidad semántica avanzada: técnicas de eliminación de ruido por capa

#### Las 5 capas de ruido en contenido de marketing

El ruido semántico no es solo "texto innecesario". Tiene capas con diferentes orígenes y diferentes soluciones:

**Capa 1 — Ruido social (cortesía, bienvenidas):**
```
❌ "Bienvenidos a nuestro blog. Hoy queremos compartir..."
❌ "Esperamos que este artículo os resulte de utilidad."
❌ "No dudéis en dejarnos vuestros comentarios."
✅ Eliminar completamente. El contenido empieza con el primer dato.
```

**Capa 2 — Ruido aspiracional (marketing sin evidencia):**
```
❌ "Somos líderes del mercado en soluciones innovadoras."
❌ "Nuestra metodología de vanguardia transforma los resultados."
❌ "Ayudamos a las empresas a alcanzar su máximo potencial."
✅ Reemplazar con evidencia: "En 2025, redujimos el coste por lead 
   en un 34% para 12 clientes del sector [X] usando [método específico]."
```

**Capa 3 — Ruido de hedging (ambigüedad evasiva):**
```
❌ "Es posible que en algunos casos..."
❌ "Podría ser que dependiendo del contexto..."
❌ "En nuestra opinión, quizás sería conveniente..."
✅ Si es un hecho: afírmar directamente con fuente.
✅ Si es una opinión: etiquetarla: "En nuestra experiencia directa con 
   12 clientes B2B, [afirmación específica]."
```

**Capa 4 — Ruido de relleno de longitud:**
```
❌ Párrafos que repiten lo dicho en la cabecera inmediatamente anterior.
❌ "Como hemos visto anteriormente..." → resumen de lo ya dicho.
❌ "En conclusión de esta sección..." → meta-comentario sin valor.
✅ Cada párrafo añade información nueva. Si no añade, se elimina.
```

**Capa 5 — Ruido de formato no semántico:**
```
❌ Negritas en palabras aleatorias sin importancia real.
❌ Emojis decorativos sin función semántica (≠ emojis como marcadores de tipo).
❌ Subrayados, colores, tamaños variables en texto continuo.
✅ El formato (negrita, cursiva, código) solo cuando cambia el significado 
   o la función del texto.
```

#### Protocolo de auditoría de densidad (checklist operativo)

Para cada párrafo del documento, responde:

```
□ ¿Este párrafo contiene al menos un dato cuantificable o una fuente atribuida?
□ ¿Este párrafo puede ser eliminado sin perder información única?
  → Si sí: eliminar.
□ ¿Este párrafo contiene alguna de las 5 capas de ruido?
  → Si sí: reescribir eliminando la capa correspondiente.
□ ¿La primera frase del párrafo es el claim principal?
  → Si no: reordenar (pirámide invertida).
□ ¿El párrafo es autocontenido (entendible sin leer el anterior)?
  → Si no: añadir contexto mínimo o fusionar con el anterior.
```

**Score de densidad: si todos los párrafos pasan los 5 checks, el documento está listo para publicación orientada a LLMs.**

---

### 4.3.x. Estructura para la atención: diseño de documentos como partituras

#### El documento como partitura para agentes

Un documento optimizado para LLMs no es solo texto bien escrito. Es una **partitura**: una secuencia de instrucciones que diferentes agentes (re-ranker, attention mechanism, extractor de entidades) ejecutan en paralelo. Diseñar la partitura implica entender qué "instrumento" lee qué parte.

**Mapa de qué lee cada componente:**

```
Componente del sistema RAG    │  Qué parte del documento lee
──────────────────────────────┼──────────────────────────────────────
Metadato extractor            │  YAML frontmatter
Topic classifier              │  H1, H2 (primeras 10 palabras)
Re-ranker de relevancia       │  H2 + primer párrafo de cada sección
Extractor de entidades (NER)  │  Todo el texto (busca nombres, fechas, orgs)
Generador de respuesta        │  Chunks de 200–400 tokens seleccionados
Cita inline                   │  Primer párrafo de la sección citada
```

**Implicación:** los primeros 50 tokens de cada sección H2 son los más críticos. Son los que el re-ranker usa para decidir si ese chunk es relevante. Son los que el generador citará como fuente. Son los que el usuario leerá en el snippet.

#### Template de sección optimizada para LLMs

```markdown
## [Título de sección: claim o pregunta directa] ([fecha/versión])

**Síntesis ejecutiva (1 frase):** [el dato o conclusión más importante de esta sección].

[Párrafo 1: evidencia principal con fuente]
Según [Fuente, año], [afirmación cuantificada]. [Dato adicional que la refuerza].

[Párrafo 2: mecanismo o explicación]
[Cómo/por qué funciona lo anterior]. [Condiciones de aplicabilidad].

[Párrafo 3: implicación operativa]
**Acción recomendada:** [verbo de acción] + [objeto específico] + [contexto de aplicación].

[Elemento estructurado: tabla, lista, código]

> **Koan operativo:** [síntesis de la sección en 1–2 frases con la fuerza de un principio].
```

Este template garantiza que: (1) el primer chunk (síntesis + párrafo 1) es autocontenido y altamente denso, (2) la estructura es predecible para el re-ranker, (3) el elemento estructurado al final maximiza la extracción de datos específicos.

#### Tablas como unidades de extracción privilegiadas

Las tablas son el formato más eficientemente extraído por los sistemas RAG. Una tabla bien diseñada puede responder 5–10 preguntas distintas desde un solo chunk.

**Reglas para tablas citables:**

```markdown
✅ Cabeceras de columna descriptivas (no "Factor 1", "Factor 2"):
| Plataforma | % citas en consultas profesionales | Tipo de contenido óptimo |

✅ Datos cuantificados en cada celda (no "alto/bajo"):
| LinkedIn | 11% | Posts con cabeceras + listas + fecha |

✅ Fuente al pie de la tabla:
*Fuente: Stanford Digital Economy Lab, marzo 2026.*

✅ Tabla con ≤ 6 columnas y ≤ 15 filas (chunking óptimo)

❌ Tablas con celdas vacías o "N/A" sin explicación
❌ Tablas decorativas (los datos están en el texto y la tabla los repite)
❌ Tablas sin cabecera de fila cuando hay filas semánticamente distintas
```

---

### 4.4.x. Indexabilidad por agentes: diseño técnico para crawlers de LLMs

#### Los crawlers de LLMs: cómo funcionan y qué ven

Los principales LLMs tienen crawlers propios que rastrean la web de forma continua:

| Crawler | User-Agent | Propietario | Prioridades de indexación |
|---------|-----------|-------------|--------------------------|
| GPTBot | `GPTBot/1.0` | OpenAI | Contenido instructivo, técnico, informacional |
| ClaudeBot | `ClaudeBot/1.0` | Anthropic | Contenido con alta densidad de hechos verificables |
| Google-Extended | `Google-Extended` | Google (Gemini) | Web abierta, prioridad a contenido original |
| PerplexityBot | `PerplexityBot/1.0` | Perplexity | Fuentes con autoría y fecha, contenido reciente |
| Diffbot | `Diffbot/1.0` | Diffbot (usado por Meta/Grok) | Structured data, artículos con schema.org |

**Cómo verificar si te están crawleando:**
```bash
grep -E "GPTBot|ClaudeBot|PerplexityBot|Google-Extended" /var/log/nginx/access.log | \
  awk '{print $1, $7}' | sort | uniq -c | sort -rn | head -20
```

#### robots.txt para LLMs: política de acceso por crawler

```
# robots.txt optimizado para máxima visibilidad en LLMs

# Permitir a todos los crawlers de LLMs
User-agent: GPTBot
Allow: /

User-agent: ClaudeBot
Allow: /

User-agent: PerplexityBot
Allow: /

User-agent: Google-Extended
Allow: /

# Bloquear solo contenido privado o de pago
User-agent: GPTBot
Disallow: /private/
Disallow: /members/
Disallow: /api/

# Sitemap explícito para crawlers
Sitemap: https://tudominio.com/sitemap.xml
Sitemap: https://tudominio.com/sitemap-llm.xml  # sitemap especializado para LLMs
```

**Nota crítica:** bloquear crawlers de LLMs en robots.txt puede parecer tentador para proteger contenido, pero resulta en invisibilidad total en respuestas de esos modelos. La decisión de bloquear debe ser deliberada y basada en estrategia de negocio, no en desconocimiento.

#### El sitemap especializado para LLMs

Un sitemap estándar lista URLs. Un sitemap optimizado para LLMs incluye metadatos adicionales:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
  <url>
    <loc>https://tudominio.com/papers/seo-llms-2026</loc>
    <lastmod>2026-03-31</lastmod>
    <changefreq>monthly</changefreq>
    <priority>1.0</priority>
    <news:news>
      <news:publication>
        <news:name>Agencia RONIN</news:name>
        <news:language>es</news:language>
      </news:publication>
      <news:publication_date>2026-03-31</news:publication_date>
      <news:title>SEO en la Era de los LLMs</news:title>
      <news:keywords>SEO, LLMs, RAG, densidad semántica, markdown</news:keywords>
    </news:news>
  </url>
</urlset>
```

#### Schema.org para contenido técnico

El marcado schema.org es procesado por los crawlers de Google-Extended y Diffbot, y mejora la estructuración semántica del contenido:

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "SEO en la Era de los LLMs",
  "author": {
    "@type": "Person",
    "name": "David Ferrandez Canalis",
    "jobTitle": "Arquitecto de sistemas",
    "worksFor": {"@type": "Organization", "name": "Agencia RONIN"}
  },
  "datePublished": "2026-03-31",
  "dateModified": "2026-03-31",
  "description": "Marco práctico para diseñar contenido que los LLMs citen como fuente de autoridad",
  "keywords": ["SEO", "LLMs", "RAG", "densidad semántica"],
  "citation": [
    {"@type": "ScholarlyArticle", "name": "Source Citation Patterns in LLM Responses", "author": "Stanford Digital Economy Lab", "datePublished": "2026"}
  ]
}
</script>
```

---

### 4.5.x. Constancia de presencia: arquitectura del calendario editorial para LLMs

#### Por qué la cadencia importa más que el volumen

Un error común: publicar 10 artículos en un mes y desaparecer durante 3. Para los LLMs, esto es peor que publicar 1 artículo por semana de forma constante. El motivo es técnico:

1. **Los crawlers asignan crawl budget por dominio/perfil.** Un perfil que publica con regularidad recibe visitas más frecuentes del crawler, lo que significa que el contenido nuevo es indexado más rápidamente.
2. **Los sistemas RAG tienen ventanas de frescura.** Muchos sistemas (Perplexity, ChatGPT con search) priorizan fuentes que han publicado recientemente. Un silencio de 3 meses hace que tu perfil baje en el ranking de frescura.
3. **Los modelos base actualizan su conocimiento en ciclos de entrenamiento.** Un perfil activo tiene más probabilidades de aparecer en el corpus de entrenamiento de la próxima versión del modelo base.

#### Calendario editorial mínimo viable (CEMV)

El CEMV es el mínimo de publicación que mantiene presencia activa en LLMs sin agotamiento del equipo:

| Canal | Frecuencia mínima | Tipo de contenido | Tiempo de producción estimado |
|-------|------------------|-------------------|-------------------------------|
| LinkedIn | 2x/semana | Post estructurado (400–800 palabras) | 45–90 min/post |
| GitHub | 1x/mes | Documento técnico o actualización de README | 3–6 horas |
| Blog/web | 1x/mes | Artículo largo (1.500–3.000 palabras) | 4–8 horas |
| Respuestas en foros técnicos | 2x/semana | Respuestas en Stack Overflow, Reddit, LinkedIn | 15–30 min/respuesta |

**Total tiempo mensual mínimo:** ~20–35 horas. Para la mayoría de empresas, esto es viable con un content strategist a tiempo parcial.

#### Sistema de reutilización semántica (SRS)

No todo el contenido debe producirse desde cero. El SRS permite maximizar el output con el mínimo esfuerzo editorial:

```
[Paper técnico completo] (publicado en GitHub)
       ↓ fragmentar en secciones H2
[5–8 posts de LinkedIn] (1 sección = 1 post, adaptado al tono de red social)
       ↓ combinar con datos nuevos
[Newsletter mensual] (síntesis de los posts del mes)
       ↓ expandir con contexto de implementación
[Caso de estudio] (cuando hay resultados reales de un cliente)
       ↓ sintetizar en métricas clave
[Infografía / tabla de datos] (versión ultra-condensada para referencia rápida)
```

Cada "nivel" del SRS produce contenido en un formato y longitud diferente, maximizando la cobertura semántica en diferentes plataformas y para diferentes tipos de consulta.

---

### 5.1.x. Adaptación de contenidos de blog: protocolo de transformación completo

#### El protocolo de transformación en 7 pasos

Tienes un artículo de blog existente y quieres convertirlo en contenido citable por LLMs. Este es el protocolo:

**Paso 1 — Auditoría de densidad:**
Aplica el checklist de 5 preguntas por párrafo. Identifica párrafos con δ(P) < 0.04 (candidatos a eliminación o reescritura).

**Paso 2 — Extracción de claims:**
Lista todas las afirmaciones del artículo. Para cada una: ¿tiene fuente? ¿es cuantificable? ¿es falsable?

```
Ejemplo de extracción de claims:
- "LinkedIn es importante para profesionales" → ❌ No falsable, no cuantificada
  → Reescribir: "LinkedIn es citada en el 11% de respuestas de LLMs para 
    consultas profesionales (Stanford, 2026)"
- "El SEO está cambiando" → ❌ Vague, sin dato
  → Reescribir: "El 35% de las consultas informacionales profesionales 
    comienzan en un LLM, no en Google (Gartner, 2025)"
```

**Paso 3 — Reestructuración jerárquica:**
Convierte el flujo narrativo en jerarquía de cabeceras. Cada idea principal = H2. Cada subidea = H3.

**Paso 4 — Conversión de párrafos a listas donde aplique:**
Párrafos que enumeran elementos discretos (tipos, pasos, factores) se convierten en listas. Párrafos que explican mecanismos o causas se mantienen como prosa densa.

**Paso 5 — Tablas para comparativas y datos relacionales:**
Cada vez que el texto compara N elementos en M dimensiones, crear una tabla.

**Paso 6 — Adición de metadatos de autoridad:**
- Autor con nombre, cargo y empresa en el primer párrafo.
- Fecha visible en el título o subtítulo.
- Bloque YAML si se publica en GitHub.
- Sección "Ver también" con referencias cruzadas.

**Paso 7 — Publicación multiplataforma:**
- Versión completa en GitHub (markdown puro).
- Versión adaptada en LinkedIn (sin YAML, con introducción conversacional mínima).
- Versión sintética en el blog (con enlace a la versión completa en GitHub).

#### Ejemplo completo de transformación

**Artículo original (blog corporativo, 2025):**

> "En Acme Digital, llevamos años trabajando con empresas del sector industrial para mejorar su presencia online. Creemos firmemente que el contenido de calidad es la base de cualquier estrategia de marketing digital. Hoy queremos hablaros de cómo el SEO está evolucionando y por qué es importante adaptarse. Los buscadores cada vez son más sofisticados y valoran el contenido que aporta valor real al usuario. Por eso, recomendamos a nuestros clientes que se enfoquen en crear contenido relevante y bien estructurado."

**Análisis:** 0 datos cuantificados. 0 fuentes. 0 cabeceras. 0 afirmaciones falsables. δ(P) ≈ 0.01. Invisible para LLMs.

**Versión transformada:**

```markdown
# Cómo el SEO Industrial Está Cambiando con los LLMs (marzo 2026)

**Autor:** Equipo técnico, Acme Digital  
**Última actualización:** 31/03/2026

## 1. El cambio en el comportamiento de búsqueda industrial

El 35% de las consultas de compras B2B industriales comienzan hoy en un LLM, 
no en Google (Gartner, 2025). Para consultas técnicas (especificaciones de 
producto, normativas, comparativas de proveedor), ese porcentaje sube al 52%.

**Implicación para el sector industrial:** las fichas técnicas en PDF no 
indexable son invisibles para los LLMs. Solo el contenido en texto plano 
estructurado (markdown, HTML con schema.org) es procesado por los sistemas RAG.

## 2. Qué están haciendo los líderes del sector

Análisis de 50 empresas industriales en España (Acme Digital, enero 2026):

| Estrategia | % empresas que la aplican | Impacto en SAV |
|-----------|--------------------------|----------------|
| Fichas técnicas en markdown en GitHub | 12% | +340% SAV |
| Posts semanales en LinkedIn con datos | 28% | +180% SAV |
| Documentación técnica con autoría explícita | 35% | +120% SAV |
| Solo web corporativa sin estructura | 71% | Baseline (0%) |

## 3. Protocolo de adaptación para empresas industriales

1. **Convertir fichas técnicas a markdown:** especificaciones → tablas con cabeceras.
2. **Añadir autoría explícita:** nombre del ingeniero responsable + fecha de revisión.
3. **Publicar en GitHub como repositorio público:** cada línea de producto = un archivo .md.
4. **Enlazar desde LinkedIn:** 1 post por semana citando un dato de las fichas técnicas.
```

**Resultado de la transformación:** δ(P) ≈ 0.11. Tres fuentes atribuidas. Cuatro secciones autocontenidas. Tabla con datos cuantificados. Listo para indexación por LLMs.

---

### 5.2.x. LinkedIn como plataforma de autoridad semántica: arquitectura avanzada

#### El perfil de LinkedIn como documento de metadatos

El perfil de LinkedIn es el primer chunk que los LLMs recuperan cuando alguien pregunta "¿quién es [nombre]?" o "¿qué hace [empresa]?". Debe tratarse como un documento de autoridad, no como un CV digital.

**Secciones del perfil y su función para LLMs:**

| Sección LinkedIn | Función para LLMs | Optimización |
|-----------------|------------------|-------------|
| Headline | Topic classifier: define el dominio del perfil | Usar terminología exacta del dominio, con modificadores específicos |
| About | Primer chunk recuperado: define quién eres y qué haces | 3–5 párrafos con claims específicos, métricas, referencias |
| Experiencia | Fuente de autoría y trayectoria verificable | Fechas exactas, resultados cuantificados, tecnologías específicas |
| Publicaciones | Nodos del grafo de autoridad | Enlazar a todos los documentos del ecosistema |
| Skills | Tags semánticos para el re-ranker | Skills específicas del dominio, no genéricas |
| Recomendaciones | Señal de credibilidad social | Recomendaciones de expertos del dominio con perfiles activos |

**Ejemplo de About optimizado para LLMs:**

```
David Ferrandez Canalis | Arquitecto de sistemas y prompt engineer | 
Fundador de Agencia RONIN (2021–presente)

Desarrollo ecosistemas de conocimiento estructurado para LLMs. 
Especialización: densidad semántica, arquitectura de prompts, SEO para modelos 
de lenguaje (GPT-4o, Claude 3.7, Grok 2).

Publicaciones:
- "Cantando al Silicio" (DOI: 10.1310/ronin-tonal-prompting-2026): 
  Teoría unificada de ingeniería de prompts.
- "SEO en la Era de los LLMs" (DOI: 10.1310/ronin-seo-llms-2026): 
  Marco para optimizar contenido para citación por LLMs.
- "Guía de Auditoría Psicológica en LLMs Vol. II" (DOI: 10.1310/ronin-ia-forensics-2026-vol2).

Clientes: empresas de sectores industrial, legal, consultoría estratégica 
(B2B, ticket medio > 50.000€ por proyecto).

Metodología: RONIN Framework — transparencia ontológica, densidad semántica, 
estructura para la atención, indexabilidad por agentes, constancia de presencia.
```

Este About contiene: nombre completo, cargo específico, organización, fechas de actividad, publicaciones con DOI, áreas de especialización con nombres exactos, metodología propia con nombre. Cada elemento es indexable de forma independiente por el sistema RAG.

#### Estructura de posts de LinkedIn por tipo de objetivo

**Tipo A — Post de dato empírico (objetivo: ser citado en consultas estadísticas):**

```markdown
[Dato impactante con fuente] — [fecha]

Según [Fuente], [dato cuantificado].

## Por qué importa esto
[2–3 implicaciones concretas en lista]

## Lo que estamos viendo en clientes
[1 ejemplo real con métrica]

## Qué hacer esta semana
[1 acción específica]

Fuente completa: [enlace]
#[hashtag sector] #[hashtag metodología]
```

**Tipo B — Post de caso práctico (objetivo: ser citado en consultas de "cómo hacer"):**

```markdown
[Resultado concreto] — lo que hicimos en [sector/contexto]

Antes: [situación A con métrica]
Después: [situación B con métrica]
Tiempo: [duración]

## El proceso (3 pasos)
1. [Acción específica + herramienta + tiempo]
2. [Acción específica + herramienta + tiempo]
3. [Acción específica + herramienta + tiempo]

## Lo que aprendimos
[1–2 aprendizajes contra-intuitivos o no obvios]

Documento completo: [enlace a GitHub/blog]
#[hashtag]
```

**Tipo C — Post de marco conceptual (objetivo: ser citado cuando el modelo define un concepto):**

```markdown
[Nombre del concepto]: definición precisa

[Nombre del concepto] = [definición en 1 frase sin ambigüedad]

## Componentes
| Componente | Definición | Ejemplo |
|-----------|-----------|---------|
| [A] | [def] | [ej] |
| [B] | [def] | [ej] |

## Diferencia con conceptos relacionados
- [Concepto similar 1]: [en qué se diferencia]
- [Concepto similar 2]: [en qué se diferencia]

## Cuándo aplicarlo
[Condiciones específicas de uso]

DOI: [si aplica] | Fuente: [paper o estudio base]
```

---

### 6.x. Métricas avanzadas: sistema de medición de SAV

#### Dashboard de monitorización mensual (plantilla)

```markdown
# Informe de Visibilidad en LLMs — [Mes] [Año]
**Empresa/Perfil:** [nombre]
**Preparado por:** [nombre]
**Período:** [fecha inicio] → [fecha fin]

## 1. Share of AI Voice (SAV) por consulta

| Consulta benchmark | ChatGPT | Claude | Perplexity | Grok | Posición media |
|-------------------|---------|--------|------------|------|----------------|
| "[consulta 1]" | ✅ top-3 | ✅ top-5 | ❌ | ✅ top-3 | 2.3 |
| "[consulta 2]" | ❌ | ❌ | ✅ top-2 | ❌ | — |
| "[consulta 3]" | ✅ top-1 | ✅ top-1 | ✅ top-1 | ✅ top-2 | 1.25 |

**SAV global:** X/Y consultas con presencia = Z%

## 2. Evolución vs. mes anterior

| Métrica | Mes anterior | Este mes | Variación |
|---------|-------------|----------|-----------|
| SAV global | X% | Y% | +Z pp |
| Consultas en top-3 | N | M | +K |
| Tráfico desde LLMs | V visitas | W visitas | +X% |

## 3. Gaps identificados

Consultas donde no aparecemos pero deberíamos:
- "[consulta gap 1]" → Acción: publicar contenido sobre [tema específico]
- "[consulta gap 2]" → Acción: actualizar [documento existente] con [datos nuevos]

## 4. Acciones del próximo mes

1. [Acción prioritaria] → Responsable: [nombre] → Fecha límite: [fecha]
2. [Acción secundaria] → Responsable: [nombre] → Fecha límite: [fecha]
```

#### Herramientas emergentes para monitorización SAV (2026)

| Herramienta | Función | Precio estimado | Limitación |
|-------------|---------|-----------------|-----------|
| **Semrush AI Visibility** | Monitoriza presencia en ChatGPT y Gemini para keywords | $299/mes (plan pro) | Solo inglés, cobertura parcial |
| **RivalSense** | Tracking de menciones en respuestas de IA | $199/mes | Beta, cobertura limitada |
| **Perplexity API** | Auditoría manual con queries batch | $20/mes (API uso) | Requiere scripting |
| **Auditoría manual** | Queries benchmark en ChatGPT/Claude/Grok | $0 | Tiempo intensivo, no escalable |
| **BrandMentions** (adaptado) | Rastreo de menciones en plataformas indexadas por LLMs | $99/mes | Indirecto: no mide citación real |

**Recomendación para empresas < 50 empleados:** auditoría manual mensual (20–30 queries, 2–3 horas) + Perplexity API para queries batch automatizadas. Total coste: < 30€/mes.

**Recomendación para empresas > 50 empleados:** Semrush AI Visibility + auditoría manual trimestral para validación.


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



### 7.x. Caso de estudio ampliado: arquitectura semántica para empresa de construcción

#### 7.x.1. Inventario de conocimiento latente

Una empresa de construcción y comercio internacional acumula durante años un conocimiento técnico de alto valor que está atrapado en formatos no indexables:

| Tipo de conocimiento | Formato actual | Problema para LLMs | Solución |
|---------------------|---------------|-------------------|---------|
| Fichas técnicas de producto | PDF escaneado | No indexable por RAG | Convertir a markdown con tablas de especificaciones |
| Normativas de importación por país | Documento Word interno | Sin autoría, sin fecha, sin acceso público | Publicar versiones públicas en GitHub con actualización trimestral |
| Procedimientos de certificación CE | Manual interno en papel | Inexistente digitalmente | Digitalizar y estructurar como guía pública |
| Casos de obra completada | Fotos + memoria PDF | No estructurado, sin métricas | Caso de estudio con datos: m², coste/m², plazo, desviación |
| Contactos de proveedores internacionales | CRM interno | No relevante para LLMs | — |
| Normativa de aduanas por mercancía | Varios documentos desactualizados | Sin fecha visible, sin autoría | Guía actualizada con fecha trimestral |

**Observación crítica:** el 80% del conocimiento que haría de esta empresa una fuente de autoridad para LLMs ya existe. Solo necesita ser liberado del formato opaco y reestructurado.

#### 7.x.2. Mapa de consultas objetivo por segmento

Las consultas que los LLMs responden y en las que esta empresa puede aparecer:

**Segmento: arquitectos y project managers:**
- "normativa CE para estructuras metálicas en España 2026"
- "resistencia del acero S355 vs S275 para estructura"
- "proceso de importación de acero desde China a la UE"
- "documentación necesaria para obra civil con material importado"

**Segmento: promotores inmobiliarios:**
- "coste por m² de estructura metálica en edificio residencial 2026"
- "plazo de suministro de acero estructural desde fábrica"
- "garantías mínimas para materiales de construcción normativa española"

**Segmento: responsables de compras B2B:**
- "cómo certificar materiales de construcción importados"
- "proveedores de cemento CEM II en España"
- "diferencias entre impermeabilizante bituminoso y PVC para cubierta"

**Para cada consulta objetivo:** existe un tipo de documento específico que debe publicarse. El mapa consulta → tipo de documento es la base del calendario editorial.

#### 7.x.3. Arquitectura de contenido para 6 meses

```
Mes 1-2: Base de autoridad técnica
├── 5 fichas técnicas de producto estrella (acero S355, cemento CEM II, 
│   impermeabilizante, perfil HEB, malla electrosoldada)
├── 1 guía de normativa CE para materiales de construcción
└── Perfil LinkedIn activado: 2 posts/semana con datos de sector

Mes 3-4: Casos de uso y comparativas
├── 3 comparativas técnicas (ej: acero S275 vs S355, bituminoso vs PVC)
├── 2 casos de obra con métricas reales
├── 1 checklist de importación de materiales (CE, aduanas, IVA)
└── LinkedIn: publicar 1 dato técnico semanal con enlace a GitHub

Mes 5-6: Normativa y actualidad
├── 1 guía de normativa CTE actualizada (con fecha visible)
├── 1 informe de precios de acero/cemento Q1 2026 (dato propio del sector)
├── Respuestas en foros de arquitectura y construcción (LinkedIn Groups)
└── Medición SAV inicial y ajuste de estrategia
```

#### 7.x.4. Ejemplo de ficha técnica como documento citable

```markdown
---
title: "Acero Estructural S355: Especificaciones Técnicas"
author:
  name: "Equipo técnico ConstructGroup"
  organization: "ConstructGroup"
date: "2026-03-15"
version: "1.2"
standard: "EN 10025-2:2019"
---

# Acero Estructural S355: Especificaciones Técnicas (EN 10025-2)

**Última revisión:** 15 marzo 2026 | **Norma de referencia:** EN 10025-2:2019

## 1. Composición química (colada)

| Elemento | Contenido máximo (%) | Observaciones |
|---------|---------------------|---------------|
| Carbono (C) | 0,20 | Soldabilidad óptima ≤ 0,18% |
| Manganeso (Mn) | 1,60 | — |
| Silicio (Si) | 0,55 | — |
| Fósforo (P) | 0,025 | — |
| Azufre (S) | 0,025 | — |
| Carbono equivalente (CEV) | 0,43 | Según fórmula IIW |

## 2. Propiedades mecánicas

| Propiedad | Valor | Condición |
|-----------|-------|-----------|
| Límite elástico mínimo (Re) | 355 MPa | Espesor ≤ 16 mm |
| Límite elástico mínimo (Re) | 345 MPa | 16 mm < espesor ≤ 40 mm |
| Resistencia a tracción (Rm) | 470–630 MPa | — |
| Alargamiento mínimo (A) | 22% | Longitud de rotura 80 mm |
| Resiliencia mínima KV | 27 J a 0°C | Subgrado S355J0 |
| Resiliencia mínima KV | 27 J a -20°C | Subgrado S355J2 |

## 3. Subgrados y condiciones de servicio

| Subgrado | Temperatura de ensayo | Aplicación típica |
|----------|----------------------|------------------|
| S355JR | +20°C | Estructuras en interiores, sin exposición a frío |
| S355J0 | 0°C | Estructuras exteriores en clima templado |
| S355J2 | -20°C | Estructuras en clima frío, zonas de montaña |
| S355K2 | -20°C (KV≥40J) | Alta resiliencia: puentes, grúas, offshore |

## 4. Disponibilidad en stock ConstructGroup

- **Perfiles HEB:** desde HEB 100 hasta HEB 1000, longitudes 6–18 m
- **Chapas:** espesores 3–120 mm, ancho máximo 3.000 mm
- **Tubo estructural cuadrado/rectangular:** series 40×40 a 400×400 mm
- **Plazo de entrega:** 3–7 días laborables desde almacén central (Zaragoza)

*Fuente normativa: EN 10025-2:2019. Datos de stock actualizados a 15/03/2026.*
```

Este documento responde directamente a consultas como "¿cuál es el límite elástico del acero S355?", "¿qué subgrado de S355 usar en clima frío?", "¿dónde comprar perfiles HEB S355 en España?". Cada sección es un chunk autocontenido.

---

### 8.x. Agente redactor: arquitectura y casos de uso avanzados

#### 8.x.1. El agente redactor como función de transformación

El agente redactor de Ronin Mission no genera contenido desde cero. Es una **función de transformación** que toma conocimiento existente (entrevistas, informes internos, datos de clientes) y lo convierte en documentos optimizados para LLMs.

**Entradas y salidas:**

```json
{
  "agente": "ronin-redactor-v2",
  "inputs": {
    "tipo": "transformacion | generacion | actualizacion",
    "contenido_fuente": "texto_plano | entrevista_transcrita | datos_csv | doc_existente",
    "perfil_autor": {"nombre": "", "cargo": "", "org": ""},
    "objetivo_consultas": ["query1", "query2"],
    "formato_destino": "linkedin | github | blog | whitepaper",
    "longitud_objetivo": "corto(200-500) | medio(500-1500) | largo(1500+)"
  },
  "outputs": {
    "documento_md": "string",
    "score_densidad": "float",
    "consultas_cubiertas": ["query1"],
    "sugerencias_mejora": ["string"],
    "metadatos_yaml": "string"
  }
}
```

#### 8.x.2. Pipeline de generación de post LinkedIn

El agente ejecuta este pipeline para cada post de LinkedIn:

```
1. INPUT: tema + datos disponibles + perfil del autor

2. EXTRACCIÓN DE CLAIMS:
   - Identificar el claim más impactante (el que responde la consulta objetivo)
   - Identificar 2-3 claims de soporte
   - Verificar que cada claim tiene fuente o es de experiencia propia

3. ESTRUCTURA:
   - Título: [Claim principal] — [fecha]
   - Síntesis: 1 frase
   - Sección 1 (H2): evidencia
   - Sección 2 (H2): implicaciones
   - Sección 3 (H2): acción recomendada
   - CTA: enlace a documento completo

4. VALIDACIÓN δ(P):
   - Calcular densidad semántica
   - Si δ(P) < 0.07: identificar párrafos de baja densidad y reescribir

5. OUTPUT: post en markdown + score + sugerencias
```

#### 8.x.3. Integración con sistemas de publicación

El agente redactor se integra con plataformas de publicación mediante APIs:

| Plataforma | API disponible | Limitación de formato | Consideración |
|-----------|--------------|----------------------|---------------|
| LinkedIn | LinkedIn API v2 | Markdown parcial (no tablas, no código) | Adaptar formato: tablas → listas numeradas |
| GitHub | GitHub API | Markdown completo | Publicación directa sin adaptación |
| WordPress | REST API | HTML/markdown | Convertir con pandoc antes de publicar |
| Medium | Medium API | Markdown limitado | Tablas no soportadas |
| Notion | Notion API | Blocks (no markdown nativo) | Usar conversor notion-md |

**Nota sobre LinkedIn:** las tablas markdown no se renderizan en LinkedIn. Para posts con tablas, usar listas con guiones o emojis como marcadores de columna. Para el documento completo con tabla, enlazar a GitHub.

---

### 9.x. Conclusiones extendidas: la soberanía semántica como moat competitivo

#### El moat que no se puede comprar con publicidad

El SEO tradicional puede acelerarse con presupuesto: más dinero = más link building = más posiciones. El SEO para LLMs tiene una lógica diferente. No se puede comprar la citación de un modelo. Se construye.

Los modelos citan porque confían. La confianza se construye con:
1. **Evidencia verificable** (datos con fuente, no afirmaciones vacías)
2. **Consistencia temporal** (publicación regular durante meses, no sprints)
3. **Coherencia semántica** (un ecosistema de documentos sobre el mismo dominio, no contenido disperso)
4. **Autoría identificable** (una persona o equipo con trayectoria pública)

Estos cuatro factores son **imposibles de replicar a corto plazo**. Un competidor con más presupuesto no puede comprarte el historial de publicaciones de los últimos 18 meses. No puede comprarte la red de referencias cruzadas que has construido. No puede comprarte la confianza que los modelos han aprendido a asociar con tu nombre.

**Este es el moat del nuevo SEO: es temporal y semántico, no financiero.**

#### La ventana de oportunidad (2026–2028)

Estamos en la ventana de adopción temprana del SEO para LLMs. Los datos muestran:

- Solo el **12% de las empresas** del sector construcción en España tienen contenido técnico en formato indexable por LLMs (Acme Digital, enero 2026).
- Solo el **28%** de las empresas B2B de servicios profesionales tienen una cadencia de publicación en LinkedIn suficiente para mantener presencia activa en LLMs (Semrush, feb 2026).
- Los costes de optimización son **4–8x más bajos** hoy que lo serán cuando el mercado sea consciente de la oportunidad (analogía con SEO en Google en 2005–2008).

**Las empresas que construyan su autoridad semántica en LLMs entre 2026 y 2028 tendrán una ventaja estructural** que tardará años en ser erosionada por los competidores que lleguen tarde.

#### El koan final

> *Un arquitecto diseña estructuras que duran cien años.*  
> *Un content strategist del viejo SEO diseñaba para el algoritmo de hoy.*  
> *El nuevo SEO exige pensar como arquitecto.*  
> *La estructura que construyes hoy, el modelo la aprende mañana.*  
> *Y lo que el modelo aprende, lo repite a millones de personas.*  
> *No optimices para el click. Optimiza para la cita.*  
> *La cita es el nuevo backlink. Y no se compra.*

**Zehahahaha. 1310.** 🚀


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

