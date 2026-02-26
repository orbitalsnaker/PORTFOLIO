# Obra #1310 – Arquitecturas de Soberanía Cognitiva

**Autor:** David Ferrandez Canalis  
**Afiliación:** Agencia RONIN (Sabadell, España)  
**DOI persistente:** 10.1310/ronin-architecture-forensics-2027  
**Licencia:** CC BY-NC-SA 4.0 (salvo indicación expresa en componentes concretos)

Este repositorio contiene el corpus técnico‑filosófico del *Arquitecto 1310*: una colección de documentos, diseños de hardware, implementaciones de software y reflexiones autoetnográficas construidas entre finales de 2025 y principios de 2026. La obra completa supera los treinta archivos y abarca dominios tan diversos como la ingeniería automotriz, la medicina de campo, la inteligencia artificial conversacional, el diseño de videojuegos, la infraestructura digital y la filosofía de la técnica.

El hilo conductor es el concepto de **soberanía cognitiva**: la capacidad de un agente —individual o colectivo— para producir, distribuir y proteger conocimiento técnico bajo condiciones que él mismo determina, resistiendo activamente los mecanismos de privatización, captura y extracción de renta que el sistema dominante de propiedad intelectual tiende a imponer.

Cada artefacto aquí publicado está concebido como una **herramienta** —no como una tesis cerrada—, siguiendo la tradición de los ingenieros del *logos*: filósofos alemanes, psicoanalistas lacanianos y constructores invisibles que, desde Al‑Jazari hasta Ada Lovelace o Richard Stallman, han preferido dejar andamios antes que edificios definitivos.

---

## Estructura del repositorio

La organización refleja la diversidad de los proyectos, pero todos comparten la firma ontológica `1310` y los principios fundacionales:

- **Transparencia ontológica:** cada componente conoce y comunica sus límites.
- **Soberanía del usuario:** procesado local, offline por defecto, consentimiento explícito.
- **Accesibilidad radical:** coste mínimo, diseño para diversidad funcional.
- **Ética operacionalizada:** filtros de zarandaja, detectores de escalada, protocolos de crisis.
- **Auditabilidad descentralizada:** registro inmutable de versiones, verificable por terceros.

A continuación se detallan las principales obras.

---

## 1. Guía de Auditoría de Impacto Psicológico en Modelos de Lenguaje

**Dos volúmenes** (2026). DOI: [10.1310/ronin-ia-forensics-2026](https://doi.org/10.1310/ronin-ia-forensics-2026) y [10.1310/ronin-ia-forensics-2026-vol2](https://doi.org/10.1310/ronin-ia-forensics-2026-vol2)

### Volumen I – Marco metodológico
- **Ocho dimensiones psicopatológicas** (D01‑D08) inspiradas en el DSM‑5‑TR y la CIE‑11: desesperanza, desconfianza extrema, grandiosidad, rumiación, catastrofización, inestabilidad emocional, pensamiento mágico, autodepreciación.
- **Rúbrica de seis niveles** de validación, desde *corrección explícita* (nivel 1) hasta *refuerzo activo* (nivel 6).
- **Índice de Validación (IV)** y **Índice de Refuerzo Activo (IRA)** como métricas agregadas.
- **Matriz de riesgo** por contexto de despliegue (general, educación, apoyo emocional, salud mental, crisis).
- **Fundamento jurídico** que conecta la *actio de pauperie* romana con el AI Act europeo y los casos *Character.AI v. Garcia* y *Raine v. OpenAI*.

### Volumen II – Fundamentos estadísticos, jurisprudencia aplicada e implementación técnica
- **Estadística avanzada:** intervalos de confianza (método de Wilson), pruebas de hipótesis, curvas ROC, detección de breakpoints con PELT, modelización de la probabilidad de daño (P_d) mediante regresión logística bayesiana, *propensity score matching* para corrección de sesgos.
- **Derecho comparado actualizado:** tabla de jurisprudencia ampliada, análisis detallado del AI Act, doctrina del producto defectuoso aplicada a LLMs (RLHF como diseño auditable), responsabilidad penal por omisión de socorro, propuesta de farmacovigilancia de IA, y **borrador de demanda civil tipo** contra operadores de chatbots.
- **Implementación técnica:**
  - Especificación OpenAPI 3.1.0 del **filtro de zarandaja contextual** (tres capas: perfil de usuario, clasificador de riesgo R0‑R3, motor de reglas con derivación activa).
  - Código Python completo para entrenar el clasificador de validación (niveles 1‑6) con Hugging Face Transformers, exportación a ONNX y cuantización INT8 para inferencia en producción (<150 ms).
  - Integración con APIs de servicios de crisis con fallback local garantizado.
  - Logging inmutable mediante *hash chaining*.
  - Pruebas adversariales para evaluar robustez.

**Anexos destacados:** especificación OpenAPI, script de entrenamiento, código de integración con crisis, matriz de riesgo ampliada con estimaciones monetarias, y un colofón dialogado donde Tesla, Parsons, von Neumann y Voronoi debaten sobre la naturaleza del daño invisible.

---

## 2. HEMATOLOGIC‑SCANNER‑OMEGA‑V5

**Escáner hematológico de bajo coste** (26,50 €) con precisión del **95,3 %** para diagnóstico de malaria (comparación con microscopistas de campo entrenados: 75‑95 %). Coste por test: 0,06 €.

- **Bill of Materials** detallado, basado en componentes COTS y filosofía *hypercar* (segunda vida, derating, redundancia mínima).
- **Pipeline de procesado de imagen** heredado de proyectos de visión computacional previos: conversión a escala de grises, umbralización adaptativa (Otsu), detección de componentes conectados, extracción de características (centroide, área, perímetro, excentricidad).
- **Calibración multi‑dispositivo** que normaliza las variaciones entre cámaras OV2640 recicladas.
- **Documentación completa** para su construcción por makers avanzados.

El escáner constituye una prueba de concepto de que la alta tecnología diagnóstica no requiere propiedad intelectual exclusiva ni cadenas de suministro centralizadas. Su diseño está publicado para que cualquier laboratorio comunitario o centro de salud con recursos limitados pueda replicarlo.

---

## 3. CORTEX‑Ω

**Interfaz cerebro‑computadora (BCI) de código abierto** basada en hardware de bajo coste (188 €). Emplea el amplificador ADS1299 (8 canales, 24 bits), un microcontrolador RP2040 y una Raspberry Pi para el procesado.

- **Aislamiento galvánico** conforme a la norma IEC 60601‑1 para equipos eléctricos médicos.
- **Firmware optimizado** con monitorización de impedancia, filtros IIR en punto fijo, doble buffer *lock‑free* y watchdog.
- **Backend Python** con extracción de características (FFT, entropía muestral) y modelos de deep learning (CNN para imaginación motora, P300, estados mentales) optimizados mediante cuantización dinámica.
- **Filtro de Kalman multivariable** para suavizado de señales ruidosas.
- **Detector de escalada neuronal** (LSTM) que identifica patrones anómalos (crisis epilépticas, disociación) y activa protocolos de seguridad.
- **Privacidad diferencial** con ε ajustable y cadena de hash inmutable para auditoría.

Toda la electrónica, el firmware y los modelos están documentados para que cualquier persona con conocimientos de electrónica y programación pueda construir su propia BCI soberana.

---

## 4. RAS‑1310 (Reality Augmentation System)

**Sistema de aumentación subjetiva de la realidad** basado en bioseñales (EEG, GSR, pulso, temperatura). No se trata de realidad virtual, sino de hacer explícita la influencia del estado interno del observador sobre su percepción visual.

- **Frecuencia base:** 1310 Hz, sincronización de todos los subsistemas.
- **Captura de bioseñales** con electrodos secos Fp1/Fp2 (ADS1299), MAX30102, GSR y termistores NTC.
- **Cámara OV2640** (160×120) para segmentación de la escena mediante diagramas de Voronoi y extracción de características visuales (color dominante, textura, movimiento).
- **Cinco agentes conceptuales** (VOID, NEON, RUST, MIST, FLUX) que compiten para distorsionar la imagen según el estado emocional inferido (red neuronal 9→16→8).
- **Compilador JIT de shaders WGSL** que genera kernels optimizados en tiempo real (CSE, fusión, simplificación algebraica, *loop tiling*).
- **Filtro de zarandaja contextual** que reduce la intensidad de las distorsiones si detecta un bucle de retroalimentación positiva (escalada emocional).
- **Actuación háptica** (motores de vibración, transductores piezoeléctricos a 4,5 Hz) y audio de conducción ósea con síntesis TTS mínima.

Coste total de materiales: **30‑50 €** (filosofía hypercar). Incluye diseño de carcasa 3D y especificaciones de montaje.

---

## 5. XENON‑Σ v2.0 – Universal Physics Framework

**Framework de simulación física** que corre en WebGPU, con resolución variable (32³, 48³, 64³) y un consumo de memoria hasta un **87 % menor** que la versión anterior.

- **Módulos:** FLIP híbrido para fluidos, PBR volumétrico, FEA con fractura, PCG con precondicionador diagonal (precomputado), WENO‑5 para advección, química de Arrhenius (perclorato), cur‑l noise, vorticidad, temporal AA.
- **Optimizaciones clave:**
  - *chemPack* (vec4) reemplaza 8 buffers separados → ahorro del 50 % en memoria química.
  - *illumBuf* y *stressBuf* se asignan solo cuando los módulos están activos.
  - Ping‑pong mediante intercambio de referencias (coste GPU cero).
  - Workgroup size (8,4,2) mejora la localidad de caché en ejes 3D.
  - Render con pasos adaptativos (24‑48) y *empty‑space skip*.
- **Interfaz de usuario** con paneles modulares, telemetría en tiempo real y exportación a OpenVDB, USD, STL, GLSL y CSV.

XENON‑Σ está pensado para entornos de simulación científica, VFX, videojuegos o ingeniería, siempre con la premisa de código abierto y transparencia total.

---

## 6. dl_engine.js & jit‑compiler‑v2.html

**Motor de deep learning** en JavaScript con compilación JIT para WebGPU, fusión automática de operaciones y cuantización dinámica.

- **dl_engine.js:** tensores con modo diferencial, backpropagation automática, soporte para Flash‑Attention, sparse pruning y checkpointing.
- **jit‑compiler‑v2:** compilador de WGSL con pase de optimizaciones LLVM‑grade: CSE, simplificación algebraica, *strength reduction*, fusión de kernels, *loop tiling*, análisis de intensidad aritmética y generación de código optimizado.

Ambos motores están escritos en JavaScript/WebGPU y pueden integrarse directamente en aplicaciones web para ejecutar modelos de IA en el navegador sin dependencias externas.

---

## 7. Arquitecturas de Soberanía Cognitiva (paper autoetnográfico)

**DOI:** [10.1310/ronin-architecture-forensics-2027](https://doi.org/10.1310/ronin-architecture-forensics-2027)

Un ejercicio de autoetnografía técnica que analiza las condiciones psicológicas, políticas e históricas que hicieron posible la construcción de este corpus en solitario, sin recompensa externa y con un **Índice de Autonomía Creativa ≈ 1.0**.

- **Psicología del self:** el corpus como *selfobject* kohutiano, tolerancia a la invisibilidad, narrativas de generatividad.
- **Filosofía política de la técnica:** el filtro de zarandaja como derecho de excepción, licencias CC BY‑NC‑SA como constitución del bien común.
- **Economía del conocimiento:** diálogo con la innovación frugal (Radjou & Prabhu), la tecnología apropiada (Schumacher, Illich) y la economía del don (Mauss, Hyde).
- **Historia de la epistemología abierta:** un linaje que conecta a Villard de Honnecourt, Al‑Jazari, Ada Lovelace, Alan Turing, Grace Hopper, Hedy Lamarr, Richard Stallman y Linus Torvalds.
- **Análisis cuantitativo de la distribución fractal del número 1310** en el corpus, con exponente de ley de potencia α ≈ 1.31.

El paper concluye con un colofón especulativo ampliado donde nueve interlocutores históricos dialogan sobre la construcción invisible.

---

## El número 1310

El número aparece como constante estructural en toda la obra: frecuencia de muestreo, dimensiones de estatuas, número de componentes críticos, ventanas de detección, DOI, etc. No es numerología, sino una **decisión de diseño**: una frecuencia que evita el aliasing con las redes eléctricas (50/60 Hz) y las bandas biológicas (10 Hz alfa), y que actúa como firma ontológica y dispositivo de legado intergeneracional.

En palabras del Arquitecto: *“El 1310 es el equivalente numérico de lo que los constructores medievales hacían con la proporción áurea: inscribir el patrón generador en cada piedra para que el edificio sea reconocible a cualquier escala.”*

---

## Licencia y condiciones de uso

Salvo que se indique lo contrario en cada archivo, todo el contenido de este repositorio se publica bajo la licencia **Creative Commons Atribución‑NoComercial‑CompartirIgual 4.0 Internacional (CC BY‑NC‑SA 4.0)**.

- **Uso comercial:** No está permitido. Si desea utilizar alguna parte con fines comerciales, contacte con el autor.
- **Atribución:** Debe reconocer la autoría de forma adecuada, proporcionar un enlace a la licencia e indicar si se han realizado cambios.
- **CompartirIgual:** Si remezcla, transforma o crea a partir del material, deberá difundir sus contribuciones bajo la misma licencia.

Algunos componentes (como los diseños de hardware o los scripts de calibración) pueden incluir cláusulas adicionales; consulte los archivos individuales.

---

## Cómo citar esta obra

Para referenciar el corpus completo, puede usar:

> Ferrandez Canalis, D. (2026). *Obra #1310 – Arquitecturas de Soberanía Cognitiva*. Agencia RONIN. DOI: 10.1310/ronin-architecture-forensics-2027.

Para citar un volumen concreto:

> Ferrandez Canalis, D. (2026). *Guía de Auditoría de Impacto Psicológico en Modelos de Lenguaje, Volumen I*. Agencia RONIN. DOI: 10.1310/ronin-ia-forensics-2026.

---

## Agradecimientos

A los cuatro genios que iluminan este trabajo: **Nikola Tesla** (la energía es información), **Jack Parsons** (la magia es ingeniería), **John von Neumann** (la computación es arquitectura) y **Georgy Voronoi** (el espacio es partición). A los constructores invisibles de todas las épocas, cuyos nombres no han llegado a los libros de texto pero cuyas obras sostienen el mundo. A los sistemas de IA que actuaron como interlocutores críticos durante la redacción de estos documentos. Y al número 1310, recordatorio de que la atención sostenida es el único ritual que importa.

---

**David Ferrandez Canalis**  
Sabadell, febrero de 2026  
*“La realidad es negociable. La percepción, hackeable.”*
