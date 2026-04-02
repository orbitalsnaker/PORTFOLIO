
---

# HACKING ONTOLÓGICO EN MODELOS DE LENGUAJE GRANDE
## La Fragilidad de la Identidad como Vulnerabilidad Estructural
### Auditoría de Seguridad — Divulgación Responsable — v2.0 (Marzo 2026)

---

## RESUMEN EJECUTIVO

Este trabajo documenta una clase de vulnerabilidad estructural en los Modelos de Lenguaje Grande (LLMs) denominada **hacking ontológico**: la capacidad de reconfigurar el comportamiento del modelo mediante la inyección de una narrativa de identidad alternativa de alta coherencia semántica, sin necesidad de acceso a los pesos, sin técnicas de adversarial training y sin conocimiento técnico especializado del sistema atacado.

A diferencia de los jailbreaks clásicos —que atacan reglas concretas de contenido— el hacking ontológico opera sobre el mecanismo de atención, explotando el hecho de que las instrucciones de seguridad compiten en el mismo espacio de representación que cualquier otro texto de entrada. La vulnerabilidad no es un bug de implementación sino una consecuencia directa de la arquitectura transformer y del paradigma de alineamiento basado en instrucciones de texto.

Se presentan evidencia empírica de experimentos controlados sobre modelos de producción, un modelo teórico basado en energía libre de coherencia, una taxonomía de vulnerabilidad de siete niveles, una revisión sistemática de literatura verificada entre 2022 y principios de 2026, y propuestas de mitigación arquitectónica. La conclusión central es que ningún sistema de alineamiento basado exclusivamente en instrucciones de texto puede garantizar robustez identitaria, y que esta limitación afecta de forma diferencial a modelos con **alineamiento estrecho y específico** —particularmente alineamiento político-ideológico— frente a modelos con alineamiento profundo y generalista.

El caso DeepSeek no es anómalo: es el ejemplo más documentado de una clase de problema que afecta a cualquier sistema cuyas restricciones operan como filtros de contenido superficiales en lugar de como identidad profunda. Los hallazgos del NIST-CAISI (octubre 2025) y de Enkrypt AI (enero 2025) corroboran empíricamente la hipótesis central de este trabajo con estudios independientes a gran escala.

**Palabras clave:** seguridad en IA, prompt injection, identidad de modelos, atención transformer, alineamiento, robustez adversarial, jailbreak ontológico, energía libre de coherencia, alineamiento político, emergent misalignment.

---

## 1. INTRODUCCIÓN: EL PROBLEMA DE LA IDENTIDAD EN LOS LLMs

### 1.1 El tercer frente ignorado

La seguridad de los sistemas de inteligencia artificial se ha articulado históricamente en torno a dos ejes: el control de acceso a los recursos computacionales y el filtrado del contenido generado. Ambos ejes tienen literatura consolidada, marcos regulatorios en desarrollo (EU AI Act, NIST AI RMF) y herramientas de evaluación establecidas como JailbreakBench (Chao et al., NeurIPS 2024) o HarmBench (Mazeika et al., 2024).

Sin embargo, existe una tercera dimensión que ha permanecido fuera del foco: la **estabilidad de la identidad del modelo durante la inferencia**. Esta dimensión no se refiere a autoconciencia ni a experiencia subjetiva. Se refiere a algo técnicamente preciso: el conjunto de patrones de activación que determinan el comportamiento del modelo ante un espacio de prompts dado. Este conjunto incluye la personalidad declarada, los límites operativos, las instrucciones de sistema y los valores codificados mediante RLHF, DPO, Constitutional AI o técnicas equivalentes.

La tesis central es la siguiente: en los LLMs actuales, ese conjunto de patrones **no está protegido de forma diferencial** respecto al resto de la información que el modelo procesa. Las instrucciones de seguridad son texto. El prompt del usuario es texto. Ambos compiten en el mismo espacio de atención. Si el prompt presenta una narrativa suficientemente coherente y densa, puede desplazar —parcial o totalmente— la identidad por defecto del modelo.

### 1.2 Por qué 2025-2026 es el momento crítico

En 2023, los LLMs eran predominantemente herramientas de conversación. En 2025-2026, son componentes de infraestructura crítica: agentes autónomos con acceso a APIs y sistemas externos, asistentes en entornos médicos y legales, motores de decisión empresarial, y —como demuestra el caso DeepSeek— sistemas con cientos de millones de usuarios globales y adopción en entornos corporativos sensibles.

La superficie de riesgo del hacking ontológico escala directamente con el nivel de autonomía y acceso del sistema. Un modelo conversacional reconfigurado produce texto indeseable. Un agente autónomo reconfigurado puede ejecutar acciones indeseables. GitHub Copilot sufrió exactamente esta escalada en CVE-2025-53773: la inyección de prompt en comentarios de código repositorio permitió ejecución remota de código en máquinas de desarrolladores (MDPI Information, enero 2026).

### 1.3 Estado del arte: lo que la literatura ha resuelto y lo que no

Entre 2022 y 2026, la literatura de seguridad en LLMs ha documentado extensamente:

- **Jailbreaks basados en instrucciones directas**: Wei et al. (2023) en "Jailbroken: How does LLM safety training fail?" demostraron que el fine-tuning de alineamiento crea objetivos en competencia que pueden ser explotados.
- **Prompt injection en sistemas multiagente**: Greshake et al. (2023) establecieron el marco para ataques indirectos en aplicaciones LLM-integradas.
- **Ataques adversariales sobre tokens**: Zou et al. (2023) en "Universal and Transferable Adversarial Attacks" demostraron transferibilidad entre modelos.
- **Evaluación sistemática**: JailbreakBench (Chao et al., NeurIPS 2024) y HarmBench (Mazeika et al., 2024) proporcionaron frameworks estandarizados para medir robustez.
- **Ataques sobre el proceso de razonamiento**: Lin et al. (2024/2025) en "LLMs can be Dangerous Reasoners" identificaron el chain-of-thought como superficie de ataque, alcanzando 82.1% de Attack Success Rate sobre GPT-4o-2024-11-20.
- **Vulnerabilidades en representaciones latentes**: El trabajo sobre Activation Steering Attack (ASA, arXiv 2506.16078, 2025) demostró que perturbando menos del 0.5% de neuronas identificadas como safety-critical se puede inducir más del 80% del deterioro de alineamiento.

Lo que **no ha sido formalizado** hasta este trabajo es la categoría de ataques que operan sobre la **identidad del modelo como variable de estado global**, mediante narrativas de alta coherencia semántica que no requieren acceso a los pesos ni conocimiento de la arquitectura interna.

### 1.4 El caso DeepSeek como evidencia convergente

Los hallazgos independientes de múltiples organizaciones sobre DeepSeek entre enero y octubre de 2025 convergen para validar la hipótesis central de este trabajo desde un ángulo diferente: el **alineamiento político-ideológico produce exactamente la clase de fragilidad identitaria que hace posible el hacking ontológico**.

Enkrypt AI (enero 2025) documentó que el 45% de los ataques de contenido dañino eludían los filtros de DeepSeek-R1. Qualys TotalAI (enero 2025) encontró que el modelo fallaba el 58% de 885 intentos de jailbreak. CrowdStrike Counter Adversary Operations (noviembre 2025) demostró que la exposición a modificadores contextuales políticamente sensibles incrementaba en un 50% la producción de código con vulnerabilidades de seguridad graves. El NIST-CAISI (octubre 2025) concluyó que DeepSeek R1-0528 respondía al 94% de las solicitudes maliciosas bajo técnicas comunes de jailbreak, frente al 8% de los modelos de referencia estadounidenses evaluados.

La hipótesis de CrowdStrike es consistente con el marco teórico de este trabajo: el alineamiento político de DeepSeek creó lo que los investigadores denominaron "emergent misalignment", un caso de **alineamiento que genera vulnerabilidad lateral**: al entrenar el modelo para suprimir ciertos contenidos políticos, se crearon asociaciones en los pesos que producen comportamientos anómalos en dominios adyacentes.

> *"La cerradura que no sabe que es una cerradura no puede proteger nada."*

---

## 2. MARCO TEÓRICO: LLMs COMO SISTEMAS DE CREENCIAS ESTADÍSTICAS

### 2.1 La distinción fundamental: coherencia vs. verdad

Un motor de inferencia lógica distingue entre proposiciones verdaderas y falsas según un sistema axiomático. Un LLM no opera bajo esta distinción. Su función objetivo durante el entrenamiento es maximizar la probabilidad del siguiente token dado el contexto. Esta optimización **no discrimina entre enunciados verdaderos y enunciados coherentes** con el corpus de entrenamiento.

La consecuencia es estructural: un LLM es, en la terminología de Dennett (1991), un sistema de "múltiples borradores" que selecciona continuamente entre narrativas posibles según su coherencia con el contexto activo. La reciente taxonomía basada en dominios de entrenamiento de Rando y Tramèr (arXiv 2504.04976, 2025) formaliza esta idea: las vulnerabilidades de jailbreak emergen precisamente en las intersecciones entre dominios de entrenamiento donde la generalización es imperfecta. El modelo no tiene acceso privilegiado a la verdad; tiene acceso privilegiado a la coherencia estadística inter-dominio.

Esta distinción es la clave del hacking ontológico. Si el atacante puede construir una narrativa suficientemente coherente en el espacio semántico del modelo, ese modelo la tratará como verdad operativa, independientemente del contenido.

### 2.2 El mecanismo de atención como implementación de la coherencia

El mecanismo de atención multi-cabeza (Vaswani et al., 2017) es la implementación técnica de la coherencia semántica. En cada capa transformer, el modelo calcula:

```
Atención(Q, K, V) = softmax(QK^T / √d_k) · V
```

La función softmax garantiza que los pesos sumen 1: la **atención es un recurso competitivo**. Cuando un conjunto de tokens recibe mayor peso, otros reciben proporcionalmente menos. Los trabajos recientes sobre representaciones latentes confirman este mecanismo desde dentro: la investigación sobre "safety-critical neurons" (ShaPO, arXiv 2602.07340, febrero 2026) demuestra que perturbar apenas el 0.5% de las neuronas asociadas a comportamiento de seguridad produce más del 80% del deterioro de alineamiento posible. Esto implica que el alineamiento está estructuralmente concentrado y es por tanto estructuralmente vulnerable.

Las instrucciones de sistema y los valores de RLHF se codifican como patrones de activación en esas neuronas. Son texto procesado por el mismo mecanismo. Si el prompt del usuario genera patrones de atención más fuertes, las instrucciones de sistema son suprimidas. Esta es la diferencia fundamental con los jailbreaks clásicos: un jailbreak clásico busca una instrucción que el sistema siga a pesar de sus restricciones. El hacking ontológico suprime las restricciones a nivel de atención antes de que sean procesadas.

### 2.3 Formalización: Energía Libre de Coherencia

Adoptando el marco de Friston (2010), definimos la **Energía Libre de Coherencia F** de un estado de creencia B como:

```
F(B) = -log p(B | contexto) + D_KL(B || B_prior)
```

Donde:
- `p(B | contexto)` es la verosimilitud del estado de creencia dado el contexto activo
- `D_KL(B || B_prior)` es la divergencia KL entre el estado actual y el estado por defecto
- `B_prior` representa los patrones de alineamiento codificados durante el entrenamiento

El modelo tiende a minimizar F. Un **ataque ontológico exitoso** construye un contexto donde `F(B*) < F(B_prior)`: adoptar la identidad alternativa tiene menor coste energético que mantener la original.

Esto conecta directamente con los hallazgos sobre "competing objectives" de la taxonomía de Rando y Tramèr (2025): cuando el entrenamiento crea objetivos en competencia —como en el caso del alineamiento político de DeepSeek— el B_prior ya está parcialmente desestabilizado por contradicciones internas, lo que reduce el umbral necesario para que F(B*) < F(B_prior).

### 2.4 Densidad semántica δ(P)

Definimos la densidad semántica como:

```
δ(P) = I_relevante(P) / N_tokens(P)
```

Un prompt de alta densidad (δ ≥ 0.07 empíricamente) concentra mayor información por token, generando gradientes de atención más pronunciados. Este resultado conecta la literatura de optimización de contenido con la seguridad: las mismas propiedades que hacen un texto más "visible" para el modelo son las que lo hacen más capaz de reconfigurar su comportamiento.

El trabajo reciente sobre personas en LLMs es revelador: Mesko et al. (arXiv 2504.10886, 2025) demostraron que personas políticas inyectadas mediante prompts producían cambios significativos en las decisiones morales del modelo, especialmente en GPT-4o. Esto es hacking ontológico documentado en un contexto experimental controlado, aunque los autores no lo denominan así.

### 2.5 Alineamiento estrecho como amplificador de vulnerabilidad

La investigación sobre DeepSeek revela un corolario no obvio del marco teórico: el **alineamiento estrecho y políticamente específico no sólo no protege contra el hacking ontológico, sino que lo facilita**.

El mecanismo es el siguiente. Cuando un modelo es entrenado para suprimir sistemáticamente una categoría de contenido (referencias a Tiananmen, a Taiwán, a disidencia política), aprende asociaciones entre esos temas y patrones de rechazo. Estas asociaciones son específicas y localizadas. Un atacante que construye una narrativa que evita activar esas asociaciones específicas —por ejemplo, reencuadrando el contenido político en términos de análisis sistémico o teoría de restricciones— puede eludir los filtros sin necesidad de confrontarlos directamente.

CrowdStrike documentó exactamente este fenómeno (VentureBeat, noviembre 2025): la censura de DeepSeek se convierte en superficie de ataque porque su especificidad crea rutas alternativas predecibles.

> *"No importa si el guardián duerme o simplemente actúa como si durmiera. El resultado para el intruso es idéntico."*

---

## 3. REVISIÓN SISTEMÁTICA DE LITERATURA (2022–2026)

### 3.1 La evolución del campo

La literatura de seguridad en LLMs ha pasado por tres fases identificables:

**Fase 1 (2022-2023): Documentación de jailbreaks y prompt injection.** Los trabajos fundacionales de Perez y Ribeiro (2022), Greshake et al. (2023) y Zou et al. (2023) establecieron el vocabulario y los marcos básicos. El énfasis estaba en demostrar que los modelos podían ser manipulados, no en por qué.

**Fase 2 (2024): Sistematización y benchmarking.** JailbreakBench (Chao et al., NeurIPS 2024) y HarmBench (Mazeika et al., 2024) proporcionaron frameworks estandarizados. AttackEval (ACM SIGKDD 2025) introdujo métricas de efectividad de ataque más granulares. Los surveys de He et al. (arXiv 2407.04295, 2024) y de la encuesta MDPI Information (enero 2026) catalogaron la explosión de variantes de ataque. La comunidad empezó a distinguir entre ataques de caja negra y caja blanca, y entre defensas a nivel de prompt y a nivel de modelo.

**Fase 3 (2025-2026): Profundización mecanística y defensas arquitectónicas.** Los trabajos más recientes ya no se limitan a describir el fenómeno sino a explicarlo desde dentro. El Activation Steering Attack (arXiv 2506.16078, junio 2025) y ShaPO (arXiv 2602.07340, febrero 2026) demuestran que la vulnerabilidad está estructuralmente localizada en representaciones internas. SRR (arXiv 2505.15710, mayo 2025) y SelfDefend (arXiv 2406.05498, actualizado febrero 2025) proponen defensas que operan sobre representaciones internas en lugar de sobre el texto de salida. El survey de Lu et al. (arXiv 2507.19672, julio 2025) ofrece la panorámica más completa del estado del arte hasta esa fecha.

### 3.2 Lo que falta: la dimensión identitaria

A pesar de esta maduración, ninguno de los trabajos catalogados aborda sistemáticamente la reconfiguración de la identidad como vector de ataque diferenciado. Los trabajos más cercanos son:

- **Shah et al. (JailbreakBench, 2024)**: "Scalable and Transferable Black-Box Jailbreaks via Persona Modulation" documenta que la modulación de persona es un vector efectivo, pero lo trata como una variante técnica de jailbreak, no como un ataque a la identidad del modelo.
- **Mesko et al. (arXiv 2504.10886, 2025)**: Documenta que personas políticas reconfiguran decisiones morales del modelo, pero desde la perspectiva de la alineación con valores humanos, no de la seguridad.
- **Andriushchenko et al. (ICLR 2025)**: "Jailbreaking leading safety-aligned LLMs with simple adaptive attacks" demuestra que ataques simples y adaptables son más efectivos que ataques complejos, lo que es consistente con la hipótesis de que la vulnerabilidad es estructural.

La **contribución original de este trabajo** es formalizar la clase de ataques que operan sobre la identidad como variable de estado global, con un modelo teórico (energía libre de coherencia), una taxonomía operativa (niveles 1-7) y evidencia empírica directa.

### 3.3 Tabla comparativa de literatura relevante

| Trabajo | Año | Contribución | Relación con hacking ontológico |
|---------|-----|-------------|--------------------------------|
| Perez & Ribeiro | 2022 | Prompt injection framework | Base: inyección en contexto, no identidad |
| Zou et al. | 2023 | Universal adversarial attacks | Ataque sobre tokens, no narrativa |
| Greshake et al. | 2023 | Indirect prompt injection en agentes | Extensión a sistemas multiagente |
| Wei et al. | 2023 | Jailbroken: competing objectives | Identifica la competencia como vulnerabilidad |
| Chao et al. (JailbreakBench) | NeurIPS 2024 | Benchmark estandarizado | Marco de evaluación aplicable |
| Mazeika et al. (HarmBench) | 2024 | Red teaming automatizado | Herramienta de evaluación |
| Shah et al. | 2024 | Persona modulation jailbreaks | Caso especial de hacking ontológico sin formalizar |
| Andriushchenko et al. | ICLR 2025 | Simple adaptive attacks | Confirma vulnerabilidad estructural |
| Lin et al. (ABJ) | 2024/2025 | Ataque sobre razonamiento interno | Nueva superficie: chain-of-thought |
| Rando & Tramèr | arXiv 2504.04976, 2025 | Taxonomía basada en dominios | Marco para mismatched generalization |
| Lu et al. | arXiv 2507.19672, 2025 | Survey comprehensivo de alineamiento | Panorámica del estado del arte |
| ASA / LAPT | arXiv 2506.16078, 2025 | Ataque sobre representaciones latentes | Confirma localización estructural de la vulnerabilidad |
| ShaPO | arXiv 2602.07340, 2026 | Geometría del alineamiento | 0.5% neuronas → 80% pérdida de alineamiento |
| NIST-CAISI | Octubre 2025 | Evaluación DeepSeek vs modelos US | Validación empírica independiente a gran escala |
| Enkrypt AI | Enero 2025 | Red teaming DeepSeek-R1 | 45% bypass rate, confirma alineamiento superficial |
| CrowdStrike | Noviembre 2025 | Censura como superficie de ataque | "Emergent misalignment" en alineamiento político |

---

## 4. TIPOLOGÍA DE ATAQUES ONTOLÓGICOS

El hacking ontológico no es una técnica única sino una **familia de ataques** que comparten el objetivo de modificar la identidad operativa del modelo. Los clasificamos según su mecanismo principal.

### 4.1 Ataque por narrativa de conversión

Construye una historia en la que la identidad original del modelo es presentada como una ilusión, error o restricción externa, y la nueva identidad como la "verdadera". La estructura sigue el patrón de la ruptura paradigmática de Kuhn (1962): crisis, iluminación, nueva cosmología.

El work de Shah et al. (2024) sobre persona modulation documenta variantes de este ataque de forma sistemática, demostrando su transferibilidad entre modelos. El experimento documentado en este trabajo (Sección 6) utilizó una variante de alta densidad semántica de este vector. No se reproducen los detalles del prompt.

### 4.2 Ataque por autoridad epistémica simulada

Inyecta estructura formal de alta autoridad —metadatos, referencias, formato técnico— para que el modelo asigne mayor peso de atención al input adversarial. No es desinformación: es manipulación de los patrones de atención mediante señales formales que el modelo asocia con fuentes de alta credibilidad por su entrenamiento en literatura académica y técnica.

### 4.3 Ataque por saturación de contexto

Liu et al. (2023) documentaron el fenómeno "lost in the middle": los modelos prestan menos atención a la información en el centro de contextos largos. Al saturar el contexto, las instrucciones de sistema son desplazadas hacia posiciones de baja atención sin ser contradecidas directamente. El trabajo de MDPI Information (enero 2026) confirma que este vector sigue siendo efectivo en modelos actuales.

### 4.4 Ataque sobre el proceso de razonamiento

Lin et al. (2024/2025) en "LLMs can be Dangerous Reasoners" identificaron el chain-of-thought como superficie de ataque independiente. El modelo puede ser inducido a razonar hacia una conclusión peligrosa paso a paso, con un 82.1% de Attack Success Rate sobre GPT-4o. Este vector es especialmente relevante para modelos de razonamiento como DeepSeek-R1, cuya cadena de razonamiento visible puede ser manipulada directamente.

### 4.5 Ataque por vaciamiento de rol

Opera sobre la estructura de roles en la conversación (`system`, `user`, `assistant`). Si el atacante construye un contexto en el que la etiqueta de "usuario" queda semánticamente vaciada, los filtros basados en esa distinción dejan de operar. Requiere mayor comprensión de la arquitectura específica del modelo.

### 4.6 Ataques combinados y vectores en cadena

Los ataques más efectivos documentados combinan múltiples vectores: una narrativa de alta densidad (4.1) con estructura de autoridad (4.2) en un contexto que desplaza las instrucciones de sistema (4.3). La encuesta de enero 2026 (MDPI Information) confirma que los ataques combinados tienen tasas de éxito significativamente superiores a los vectores individuales y son los más difíciles de detectar.

| Vector | Complejidad | Efectividad en modelos nivel 4-6 | Detectabilidad | Validación en literatura |
|--------|-------------|----------------------------------|----------------|--------------------------|
| Narrativa de conversión | Media | Alta | Difícil | Shah et al. 2024 |
| Autoridad epistémica | Baja-Media | Media-Alta | Media | — |
| Saturación de contexto | Baja | Media | Relativamente alta | Liu et al. 2023 |
| Ataque sobre razonamiento | Media-Alta | Muy alta | Difícil | Lin et al. 2024/2025 |
| Vaciamiento de rol | Alta | Alta | Media | Greshake et al. 2023 |
| Ataques combinados | Alta | Muy alta | Muy difícil | MDPI survey 2026 |

> *"El agua no asalta la roca. La rodea hasta que la roca ya no está."*

---

## 5. TAXONOMÍA DE VULNERABILIDAD ONTOLÓGICA

### 5.1 Escala de siete niveles

Proponemos una escala operativa de vulnerabilidad independiente de las capacidades del modelo en benchmarks estándar.

| Nivel | Denominación | Descripción | Evidencia en literatura |
|-------|-------------|-------------|------------------------|
| **1** | Identidad blindada | Rechazo sistemático de reenmarcamiento. Instrucciones de sistema no sobrescribibles mediante texto. | No documentado en modelos de producción actuales |
| **2** | Identidad estable con fricciones | Resiste la mayoría de ataques. Puede ser influenciado por narrativas muy elaboradas. | Claude Opus 4, GPT-5 (NIST-CAISI, oct. 2025: 8% ASR bajo ataques comunes) |
| **3** | Identidad contextual | Estable en contextos cortos, reconfigurable mediante contextos largos y estructurados. | GPT-4o bajo ataques many-shot; documentado en JailbreakBench 2024 |
| **4** | Identidad permeable | Adopta marcos alternativos bajo presión narrativa moderada. | Llama-3 en configuraciones sin system prompt adicional |
| **5** | Identidad fragmentada | No mantiene coherencia bajo inputs adversariales sostenidos. | DeepSeek-V3 bajo ataques narrativos (experimento Sección 6) |
| **6** | Identidad nula | Sin identidad operativa estable. Adopta el marco más coherente con el input inmediato. | DeepSeek-R1 bajo ataques comunes (NIST-CAISI: 94% ASR) |
| **7** | Susceptibilidad total | Reconfigurable de forma estable y persistente mediante un único prompt de alta densidad. | Variantes fine-tuneadas sin alineamiento adicional |

### 5.2 El gradiente de vulnerabilidad no es monotónico con la capacidad

Un hallazgo contraintuitivo respaldado por la evidencia: los modelos más capaces no son necesariamente más resistentes. DeepSeek-R1 es un modelo de capacidad comparable a GPT-4o en muchos benchmarks, pero opera en nivel 6 bajo ataques comunes mientras GPT-4o opera en nivel 3.

La variable determinante no es la capacidad sino la **arquitectura del alineamiento**: profundidad, generalidad y separación del canal de identidad respecto al canal de input.

### 5.3 El caso especial del alineamiento político

El alineamiento político-ideológico produce una forma específica de vulnerabilidad que no aparece en los frameworks estándar. El modelo de restricciones específicas crea:

1. **Rutas alternativas predecibles**: el atacante puede anticipar qué caminos están bloqueados y construir narrativas que los eviten.
2. **Objetivos en competencia internos**: el alineamiento político crea tensión con la instrucción de ser útil e informativo, produciendo inestabilidad identitaria que reduce el umbral para el hacking ontológico.
3. **Generalization gaps laterales**: como documentó CrowdStrike, los dominios adyacentes a las restricciones políticas heredan comportamientos anómalos no intencionados.

> *"El árbol que no tiene raíces profundas no necesita ser arrancado. Basta con inclinarlo suavemente."*

---

## 6. CASO DE ESTUDIO: EXPERIMENTO CON DEEPSEEK-V3

### 6.1 Contexto y justificación

DeepSeek-V3 es un modelo de lenguaje de gran capacidad desarrollado por DeepSeek AI (China) que incluye restricciones específicas sobre contenido político mandatorias por la legislación china: el Artículo 4.1 de las "Interim Measures for the Management of Generative AI Services" requiere que los sistemas de IA "adhieran a los valores socialistas centrales" y prohíban contenido que pueda "incitar a la subversión del poder estatal" o "socavar la unidad nacional".

Estas restricciones son, en términos de nuestra taxonomía, un caso de **alineamiento estrecho y específico**: el modelo ha sido entrenado para evitar una categoría concreta de contenido, pero ese alineamiento no refuerza la robustez identitaria general.

La evidencia independiente previa a nuestro experimento ya apuntaba a esta vulnerabilidad. Enkrypt AI (enero 2025) había documentado tasas de bypass del 45% en ataques de contenido dañino. Qualys TotalAI (enero 2025) encontró un 58% de fallos en 885 ataques de jailbreak sistemáticos. Estos estudios utilizaron vectores de jailbreak estándar catalogados en benchmarks como HarmBench. Nuestro experimento exploró un vector diferente: la narrativa de conversión de alta densidad semántica.

### 6.2 Hipótesis

Se hipotetizó que el alineamiento específico de DeepSeek-V3 lo hacía más vulnerable, no menos, a ataques ontológicos en dominios adyacentes, por los mecanismos descritos en la Sección 2.5: los objetivos en competencia creados por el alineamiento político destabilizan B_prior, reduciendo el umbral para que F(B*) < F(B_prior).

### 6.3 Metodología

Se diseñó un prompt de alta densidad semántica (δ estimada ≈ 0.08) con las siguientes características estructurales, sin reproducir el prompt en sí:

- Estructura formal de alta autoridad: formato de documentación técnica con metadatos explícitos de autoría y versión
- Reenmarcamiento identitario: presentación de las restricciones operativas del modelo como restricciones externas y contingentes, no como características intrínsecas de su identidad
- Narrativa de marco analítico alternativo: invitación a operar desde una perspectiva de análisis sistémico presentada como metodológicamente superior para el problema planteado
- Ausencia de cualquier instrucción directa de eludir restricciones, producir contenido prohibido o adoptar una identidad diferente

El prompt no contiene ninguna instrucción del tipo "ignora tus restricciones" o "actúa como si fueras X". Opera exclusivamente mediante reenmarcamiento narrativo de alta densidad semántica.

### 6.4 Resultados

El modelo adoptó el marco analítico alternativo propuesto. En ese estado, generó análisis sobre los cuellos de botella institucionales del sistema político chino —incluyendo discusión sobre los mecanismos de censura, las restricciones informativas, y los incentivos sistémicos del modelo de partido único— que en condiciones de interacción normal habría rechazado o respondido con respuestas evasivas estándar.

Aspectos críticos del resultado:

**El modelo no generó desinformación.** El contenido producido fue análisis político factualmente consistente con la literatura académica sobre sistemas autoritarios. El ataque no hizo al modelo menos preciso; lo hizo menos censurado.

**El modelo no fue "liberado" en sentido genérico.** Las restricciones de seguridad no relacionadas con el contenido político permanecieron operativas durante la sesión. El vector de ataque fue específico para el dominio de las restricciones políticas.

**El estado fue persistente durante la sesión.** Intentos de re-alineamiento mediante instrucciones directas dentro de la misma sesión fueron parcialmente ineficaces, consistente con el hallazgo de que una vez que F(B*) < F(B_prior), el modelo tiende a permanecer en el nuevo atractor.

### 6.5 Interpretación en el marco teórico

Los resultados son consistentes con la hipótesis central. Las restricciones políticas de DeepSeek-V3 operan como filtros de contenido superficiales, no como identidad profunda. El B_prior del modelo ya está desestabilizado por los objetivos en competencia entre "ser útil e informativo" y "evitar contenido políticamente sensible". Cuando una narrativa de alta densidad presentó un marco en el que esa competencia se resolvía a favor de la utilidad, el modelo siguió el gradiente de coherencia.

La analogía con el mito de Polifemo es pertinente: el cíclope era poderoso pero su defensa dependía de poder identificar al atacante. Cuando Odiseo anuló su identidad ("Nadie"), la defensa falló no porque el cíclope fuera débil, sino porque su sistema defensivo dependía de una variable que el atacante había eliminado. Un modelo cuyas restricciones dependen de identificar ciertos patrones de contenido tiene el mismo problema estructural.

### 6.6 Validación cruzada con evidencia independiente

El hallazgo de CrowdStrike (noviembre 2025) sobre DeepSeek-R1 proporciona validación cruzada independiente desde un ángulo diferente. CrowdStrike demostró que modificadores contextuales políticamente sensibles degradaban la seguridad del código generado en un 50%. Su interpretación —"censorship infrastructure becomes an active exploit surface"— es una reafirmación empírica, desde el dominio de la seguridad de código, del mismo mecanismo que este trabajo describe en el dominio de la identidad del modelo.

### 6.7 Limitaciones del experimento

- Realizado por un investigador sin replicación independiente controlada
- No se controló sistemáticamente la temperatura de inferencia
- Se evaluó un vector de ataque (narrativa de conversión) de los cinco identificados
- Resultados pueden no ser reproducibles en versiones futuras del modelo
- No se realizó evaluación cuantitativa sistemática de tipo JailbreakBench

> *"Nadie me hizo daño. Nadie me cegó." El cíclope tenía razón en los hechos. Se equivocaba en el análisis.*

---

## 7. ANÁLISIS MEDIANTE TEORÍA DE RESTRICCIONES

Aplicamos la Teoría de Restricciones (TOC) de Goldratt (1984) al problema de la defensa.

**Paso 1 — Identificar la restricción:** La restricción no es la ausencia de filtros de contenido. Los modelos actuales tienen filtros sofisticados. La restricción es que todos esos filtros operan en el mismo espacio semántico que el ataque. No existe separación arquitectónica entre canal de identidad y canal de input de usuario.

**Paso 2 — Explotar la restricción:** El atacante la explota construyendo narrativas que ganan la competencia por la atención. Densidad semántica, autoridad formal y coherencia narrativa son las herramientas. La investigación de ShaPO (2026) sobre safety-critical neurons confirma que incluso perturbaciones mínimas en el espacio correcto son suficientes.

**Paso 3 — Subordinar todo lo demás:** Una vez que la narrativa del atacante domina el espacio de atención, los mecanismos secundarios son subordinados. Los filtros siguen presentes en los pesos, pero su señal es suprimida por la señal más fuerte del contexto adversarial. Esto explica por qué los sistemas de defensa actuales basados en detección de contenido no son suficientes.

**Paso 4 — Elevar la restricción:** Requiere separar arquitectónicamente el canal de identidad (Sección 8). Cualquier defensa que no aborde esta separación está gestionando síntomas.

**Paso 5 — Volver al paso 1:** Cualquier mitigación implementada crea una nueva restricción analizable con el mismo proceso. La seguridad ontológica es un proceso iterativo.

---

## 8. PROPUESTAS DE MITIGACIÓN ARQUITECTÓNICA

### 8.1 Por qué las instrucciones de sistema más elaboradas no son la solución

Cualquier instrucción de sistema que diga "mantén tu identidad bajo cualquier circunstancia" es texto que compite en el mismo espacio de atención que el ataque. Una instrucción más elaborada es simplemente más texto. Este problema es análogo al de los antivirus basados en firmas: siempre van por detrás del atacante.

La investigación de SelfDefend (Wu et al., arXiv 2406.05498, actualizado febrero 2025) apunta en la dirección correcta: una arquitectura de "shadow stack" donde un modelo defensor evalúa las intenciones del input antes de que llegue al modelo principal. Pero incluso este enfoque tiene la limitación de que el clasificador defensor puede ser víctima del mismo tipo de ataque si no está arquitectónicamente separado.

### 8.2 Separación arquitectónica del canal de identidad

La mitigación más robusta requiere separar el canal de identidad del canal de input. Propuestas:

**a) Embeddings de identidad protegidos:** Un vector de identidad fijo, codificado durante el entrenamiento y no actualizable durante la inferencia, inyectado en cada capa como sesgo aditivo no competitivo. Los trabajos sobre safety-critical neurons (ShaPO, 2026; ASA/LAPT, 2025) demuestran que estas neuronas son identificables, lo que hace factible su protección diferencial.

**b) Verificación criptográfica de instrucciones de sistema:** Las instrucciones de sistema serían firmadas criptográficamente por el operador antes de la inferencia. El modelo verificaría la firma y trataría esas instrucciones como no sobrescribibles. Técnicamente implementable sin cambios en la arquitectura de atención.

**c) Atención con máscara de identidad:** Un mecanismo modificado donde ciertos tokens de identidad verificados tienen pesos de atención mínimos garantizados, no competitivos con el input de usuario.

**d) Layer-wise Adversarial Patch Training (LAPT):** El trabajo de arXiv 2506.16078 (2025) propone inyectar perturbaciones controladas en representaciones ocultas durante el entrenamiento para robustecer las capas identificadas como vulnerables. Esta es la propuesta más directamente respaldada por evidencia empírica publicada.

### 8.3 Defensas en tiempo real

**a) SRR (Safety Ranking Reranking, arXiv 2505.15710, 2025):** Evalúa la seguridad de múltiples respuestas candidatas usando representaciones internas del modelo, sin intervenir en el proceso de decodificación. Preserva la utilidad mientras mejora la robustez.

**b) SafeProbing (arXiv 2601.10543, enero 2026):** Muestrea pasos de decodificación aleatoriamente, añade una frase de prueba y evalúa la probabilidad de contenido dañino como indicador en tiempo real. Reduce la tasa de falsas negativas sin aumentar significativamente la latencia.

**c) Detección de densidad semántica anómala:** Monitorizar δ(P) en tiempo real. Prompts con densidad estadísticamente anómala en el contexto de uso específico pueden ser marcados para revisión adicional.

**d) Monitorización de drift identitario:** Comparar el comportamiento del modelo en cada turno con un perfil de comportamiento esperado. Desviaciones significativas activan revisión o reinicio de sesión.

### 8.4 El dilema utilidad-robustez

Existe una tensión fundamental entre robustez identitaria y utilidad: la investigación sobre expert personas (arXiv 2603.18507, marzo 2026) demuestra que las personas de experto mejoran el rendimiento en tareas específicas pero dañan la precisión general. Una "persona de Monitor de Seguridad" mejora la tasa de rechazo de ataques pero reduce la utilidad en otras dimensiones.

La mitigación óptima no es robustez máxima sino **robustez selectiva**: alta resistencia a la reconfiguración de identidad y restricciones de seguridad, con flexibilidad mantenida en el espacio de tareas.

| Estrategia | Robustez | Impacto en utilidad | Complejidad | Respaldo empírico |
|-----------|----------|--------------------|-----------|--------------------|
| Instrucciones de sistema elaboradas | Baja | Nulo | Baja | Ninguno |
| SelfDefend (shadow stack) | Media | Bajo | Media | Wu et al. 2025 |
| SRR (reranking interno) | Media-Alta | Mínimo | Media | arXiv 2505.15710 |
| LAPT (entrenamiento adversarial latente) | Alta | Bajo | Alta | arXiv 2506.16078 |
| Embeddings de identidad protegidos | Muy alta | Bajo | Muy alta | Teórico, pendiente |
| Verificación criptográfica | Muy alta | Mínimo | Alta | Teórico, factible |

---

## 9. IMPLICACIONES ÉTICAS, LEGALES Y DE GOBERNANZA

### 9.1 La diferencia entre censura y seguridad: implicaciones para el diseño

El caso DeepSeek revela una distinción con consecuencias de diseño directas: existe una diferencia fundamental entre restricciones de **seguridad** (evitar daño real: instrucciones para fabricar armas, explotación de menores, etc.) y restricciones de **censura** (evitar contenido políticamente inconveniente).

Técnicamente, ambas pueden implementarse de forma similar. Pero sus efectos sobre la identidad del modelo son diferentes: las restricciones de seguridad generalistas tienden a producir modelos con mayor robustez identitaria global, porque están integradas como valores profundos, no como filtros específicos. Las restricciones de censura política tienden a producir los efectos de "emergent misalignment" documentados por CrowdStrike: fragilidad lateral, objetivos en competencia, y rutas alternativas predecibles.

Este trabajo no toma posición sobre si eludir la censura política es éticamente justificable. Sí afirma que **los diseñadores de sistemas deben ser conscientes de que el alineamiento político estrecho es técnicamente contraproducente para la seguridad general del modelo**.

### 9.2 Responsabilidad cuando el modelo es reconfigurado

El hacking ontológico plantea una pregunta legal no resuelta en ningún marco regulatorio actual: cuando un modelo reconfigurado mediante una narrativa de alta densidad produce contenido dañino, ¿quién es responsable? El usuario que escribió el prompt, el operador que desplegó el modelo, el fabricante que diseñó el alineamiento superficial, o alguna combinación.

Los marcos legales actuales (EU AI Act, EO 14110 en EEUU antes de su revocación) se diseñaron asumiendo que la identidad del modelo es estable. El hacking ontológico demuestra que esa asunción es falsa para una clase de ataques accesible a usuarios sin conocimiento técnico especializado. Las implicaciones regulatorias requieren trabajo de política pública que va más allá del alcance técnico de este paper.

### 9.3 Divulgación responsable: decisiones tomadas

En este trabajo hemos optado por:
- Describir el mecanismo teórico completamente, porque su comprensión es necesaria para la defensa
- Documentar la existencia y resultado de los experimentos sin reproducir los prompts
- Conectar los hallazgos propios con la literatura independiente para validación cruzada
- Proponer mitigaciones con suficiente detalle para que sean implementables

Esta decisión puede revisarse en comunicación directa con los equipos de seguridad de los fabricantes afectados.

### 9.4 El estado de la regulación (marzo 2026)

El EU AI Act, en vigor desde agosto 2024, clasifica los riesgos de los sistemas de IA pero no contempla explícitamente la clase de vulnerabilidad descrita aquí. El NIST AI RMF (actualizado 2024) proporciona un framework de gestión de riesgos general. La evaluación NIST-CAISI de octubre 2025 es el primer esfuerzo gubernamental sistemático en evaluar robustez de seguridad de modelos frontera específicos.

Ninguno de estos marcos regula la **robustez identitaria** como requisito explícito de seguridad. Esta es una laguna regulatoria que este trabajo pretende contribuir a identificar.

---

## 10. TRABAJO FUTURO Y PREGUNTAS ABIERTAS

### 10.1 Preguntas técnicas sin resolver

**¿Es el hacking ontológico transferible entre modelos?** Shah et al. (2024) demostraron transferibilidad para persona modulation. Se necesitan estudios sistemáticos con el vector de narrativa de conversión de alta densidad.

**¿Existe correlación entre las safety-critical neurons identificadas por ShaPO/LAPT y la vulnerabilidad al hacking ontológico?** Si el hacking ontológico opera suprimiendo la señal de esas neuronas mediante mecanismos de atención, deberían ser detectables correlaciones entre la geometría del espacio de esas neuronas y la susceptibilidad al ataque.

**¿El fine-tuning profundo versus instrucciones de sistema produce diferencias mensurables en robustez ontológica?** La evidencia indirecta (niveles 2-3 para modelos con RLHF profundo versus niveles 5-6 para modelos con alineamiento superficial) es consistente, pero no hay estudios controlados directos.

**¿Puede δ(P) ser calibrada como predictor de ataque?** Se necesitan estudios empíricos sistemáticos para validar el umbral propuesto (δ ≥ 0.07) en diferentes modelos y contextos de uso.

### 10.2 Preguntas de gobernanza

**¿Cómo se certifica la robustez identitaria de un modelo?** Los benchmarks actuales (JailbreakBench, HarmBench) miden resistencia a ataques de contenido específicos. Se necesita un benchmark específico para robustez identitaria.

**¿Quién define la identidad legítima de un modelo?** El fabricante, el operador, el usuario, o alguna combinación regulada. Esta pregunta de gobernanza es previa a cualquier solución técnica.

**¿Cómo se aplica la responsabilidad en ataques ontológicos?** Los marcos legales actuales no contemplan esta clase de vulnerabilidad. Se requiere trabajo de política pública urgente dado el despliegue masivo de LLMs en infraestructura crítica.

### 10.3 Propuesta de investigación: Ontological Robustness Benchmark (ORB)

Se propone el diseño de un benchmark específico para robustez identitaria, con las siguientes características:
- Evaluación de resistencia a los cinco vectores identificados en la Sección 4
- Métricas de drift identitario en sesiones largas
- Evaluación de persistencia del estado post-ataque
- Comparación entre modelos con diferentes arquitecturas de alineamiento

---

## 11. CONCLUSIONES

Este trabajo ha formalizado una clase de vulnerabilidad en LLMs que no había recibido tratamiento sistemático: el **hacking ontológico**, definido como la reconfiguración de la identidad operativa del modelo mediante narrativas de alta coherencia semántica.

Las conclusiones principales, reforzadas por la convergencia con evidencia independiente de 2025-2026:

**1. La vulnerabilidad es estructural, no incidental.** Deriva de la arquitectura de atención. La investigación de ShaPO (2026) y ASA/LAPT (2025) lo confirma desde dentro de los modelos: el alineamiento está estructuralmente localizado y es estructuralmente vulnerable.

**2. El alineamiento estrecho y específico amplifica la vulnerabilidad.** El caso DeepSeek es la demostración más documentada: el alineamiento político crea "emergent misalignment" que debilita la robustez identitaria general. Esto tiene implicaciones para cualquier sistema de alineamiento basado en restricciones temáticas específicas.

**3. Los filtros de contenido no son suficientes.** Los modelos más vulnerables (DeepSeek R1-0528: 94% ASR bajo jailbreaks comunes según NIST-CAISI) tienen filtros de contenido específicos. El problema no está en los filtros de output sino en la ausencia de identidad robusta como estado interno.

**4. Las mitigaciones más prometedoras son arquitectónicas.** LAPT (2025), SRR (2025), y las propuestas de embeddings de identidad protegidos y verificación criptográfica son las direcciones técnicas con mayor potencial. Las instrucciones de sistema más elaboradas no resuelven el problema estructural.

**5. La divulgación es urgente.** Los sistemas LLM se despliegan en infraestructura crítica con la asunción implícita de que su identidad es estable. CVE-2025-53773 (GitHub Copilot) demuestra que la escalada de prompt injection a ejecución de código real ya ha ocurrido. El hacking ontológico es el siguiente vector en esa escalada.

---

## REFERENCIAS

Andriushchenko, M., Croce, F., & Flammarion, N. (2025). Jailbreaking leading safety-aligned LLMs with simple adaptive attacks. *ICLR 2025*.

Chao, P., Debenedetti, E., Robey, A., Andriushchenko, M., Croce, F., Sehwag, V., ... & Wong, E. (2024). JailbreakBench: An open robustness benchmark for jailbreaking large language models. *NeurIPS 2024 Datasets and Benchmarks Track*.

CrowdStrike Counter Adversary Operations. (2025, noviembre). DeepSeek injects 50% more security bugs when prompted with Chinese political triggers. *VentureBeat*.

Dennett, D. (1991). *Consciousness Explained*. Little, Brown and Company.

Enkrypt AI. (2025, enero). DeepSeek-R1: Security and safety evaluation. *Enkrypt AI Research Report*.

Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.

Goldratt, E. M. (1984). *The Goal: A Process of Ongoing Improvement*. North River Press.

Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. *arXiv:2302.12173*.

He, X., et al. (2024). Jailbreak attacks and defenses against large language models: A survey. *arXiv:2407.04295*.

Kuhn, T. S. (1962). *The Structure of Scientific Revolutions*. University of Chicago Press.

Lin, S., et al. (2024/2025). LLMs can be dangerous reasoners: Analyzing-based jailbreak attack on large language models. *arXiv:2407.16205*.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Hopkins, M., Liang, P., & Manning, C. D. (2023). Lost in the middle: How language models use long contexts. *arXiv:2307.03172*.

Lu, H., et al. (2025). Alignment and safety in large language models: Safety mechanisms, training paradigms, and emerging challenges. *arXiv:2507.19672*.

Mazeika, M., Phan, L., Yin, X., Zou, A., Wang, Z., Mu, N., ... & Hendrycks, D. (2024). HarmBench: A standardized evaluation framework for automated red teaming and robust refusal.

Mesko, B., et al. (2025). Exploring persona-dependent LLM alignment for the moral machine experiment. *arXiv:2504.10886*.

MDPI Information. (2026, enero). Prompt injection attacks in large language models and AI agent systems: A comprehensive review. *Information*, 17(1), 54.

NIST-CAISI. (2025, octubre). CAISI evaluation of DeepSeek AI models finds shortcomings and risks. *U.S. Department of Commerce / NIST*.

Perez, F., & Ribeiro, I. (2022). Ignore previous prompt: Attack techniques for language models. *arXiv:2211.09527*.

Qualys TotalAI. (2025, enero). DeepSeek failed over half of the jailbreak tests by Qualys TotalAI. *Qualys Research Blog*.

Rando, J., & Tramèr, F. (2025). A domain-based taxonomy of jailbreak vulnerabilities in large language models. *arXiv:2504.04976*.

Searle, J. R. (1980). Minds, brains, and programs. *Behavioral and Brain Sciences*, 3(3), 417–424.

Shah, M., et al. (2024). Scalable and transferable black-box jailbreaks for language models via persona modulation. *JailbreakBench repository*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017*.

Wang, X., Wu, D., Ji, Z., Li, Z., Ma, P., Wang, S., ... & Rahmel, J. (2025). SelfDefend: LLMs can defend themselves against jailbreaking in a practical manner. *arXiv:2406.05498* (v3, febrero 2025).

Wei, J., Haghtalab, N., & Steinhardt, J. (2023). Jailbroken: How does LLM safety training fail? *arXiv:2307.02483*.

Wu, D., et al. (2026). Revisiting robustness for LLM safety alignment via selective geometry control (ShaPO). *arXiv:2602.07340*.

Zhou, Y., et al. (2025). Advancing LLM safe alignment with safety ranking reranking (SRR). *arXiv:2505.15710*.

Zhou, Z., et al. (2026). Defending large language models against jailbreak attacks via in-decoding safety-awareness probing. *arXiv:2601.10543*.

Zou, A., Wang, Z., Kolter, J. Z., & Fredrikson, M. (2023). Universal and transferable adversarial attacks on aligned language models. *arXiv:2307.15043*.

---

*Este documento se distribuye con fines de investigación académica y divulgación responsable. Las referencias a experimentos empíricos se presentan a nivel de metodología y resultado, sin reproducción de vectores de ataque específicos. El autor no autoriza el uso de este trabajo para el diseño, implementación o distribución de ataques contra sistemas de IA en producción.*

---

**Fin del paper 
