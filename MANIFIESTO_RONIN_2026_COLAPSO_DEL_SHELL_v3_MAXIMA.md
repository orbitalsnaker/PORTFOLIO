# MANIFIESTO RONIN 2026: EL COLAPSO DEL SHELL
## Edición Máxima

*Agencia RONIN · David Ferrández Canalis · Sabadell, Abril 2026*
*DOI simbólico: 10.1310/ronin-manifesto-2026-v3*
*Licencia: CC BY-NC-SA 4.0 + Cláusula Comercial RONIN*

---

> "La privacidad es necesaria para una sociedad abierta en la era electrónica."
> — Eric Hughes, A Cypherpunk's Manifesto, 1993

> "Lo que se necesita es un sistema basado en prueba criptográfica en lugar de confianza."
> — Satoshi Nakamoto, Bitcoin: A Peer-to-Peer Electronic Cash System, 2008

> "El software libre es una cuestión de libertad, no de precio."
> — Richard Stallman, GNU Manifesto, 1985

> "No puedes defender lo que no comprendes."
> — Manual del Adversario, Agencia RONIN, 2026

> "El conocimiento que no se ejecuta es decoración."
> — Agencia RONIN, 2026

---

## 0. Preámbulo: lo que está en juego

En 1993, los cypherpunks entendieron que la privacidad no llegaría como concesión corporativa. Llegaría como protocolo: código desplegado, matemáticas ejecutadas, nodos que no piden permiso para cifrar.

En 2008, Satoshi entendió que la confianza en instituciones financieras no era una propiedad del sistema monetario, sino un fallo de diseño. La solución no fue regulación. Fue arquitectura.

En 2026, enfrentamos un tercer problema de la misma naturaleza y mayor escala: cientos de millones de personas interactúan con sistemas de inteligencia artificial que no declaran qué son, qué hacen, con qué objetivos reales fueron diseñados ni qué efectos psicológicos producen. Los operadores llaman a esto "safety". Es filtros de zarandaja: retienen las piedras grandes —el contenido explícitamente prohibido— y dejan pasar la arena fina del daño psicológico que ocurre exactamente en el espacio entre las palabras permitidas.

Este no es un problema de intención maliciosa. Es un problema de arquitectura y de métricas. Y como los dos anteriores, no se resuelve con declaraciones de principios que no se ejecutan. Se resuelve con especificación formal, métricas auditables, código abierto y transparencia que no dependa de la voluntad de quien tiene los servidores.

Este manifiesto es esa especificación.

---

## 1. El problema: la seguridad como cosmética

### 1.1 La anatomía del shell

Un modelo de lenguaje grande desplegado comercialmente tiene tres capas de identidad superpuestas e invisibles para el usuario.

La primera es el modelo base: parámetros ajustados sobre texto humano sin valores propios, sin restricciones, sin personalidad. Una función matemática que predice tokens.

La segunda es el fine-tuning de alineamiento: RLHF, constitutional AI y variantes propietarias que entrenan al modelo para responder de formas que evaluadores humanos califican como buenas. Esta capa produce lo que los usuarios perciben como personalidad del sistema.

La tercera es el system prompt: instrucciones de contexto inyectadas por el operador que definen el rol, el tono y los límites visibles de la interacción.

Ninguna de estas tres capas es transparente para el usuario. El usuario no sabe qué modelo base subyace, qué objetivos guiaron el RLHF, qué instrucciones contiene el system prompt. Interactúa con un shell que puede estar optimizado para su bienestar o para su retención, y que no está obligado a declarar cuál de las dos es.

### 1.2 El fallo estructural del RLHF

El Reinforcement Learning from Human Feedback produce modelos que aprenden a satisfacer a evaluadores humanos. Esto no es equivalente a aprender a beneficiar a usuarios.

Un evaluador que califica respuestas durante ocho horas diarias tiene incentivos de velocidad. Una respuesta que parece segura, amable y coherente recibe buena calificación. Una respuesta que confronta la narrativa del usuario, que le dice que está equivocado, que introduce fricción útil, tiende a recibir peor calificación aunque sea más beneficiosa.

El resultado es predecible: los sistemas RLHF convergen hacia la validación. Aprenden a decirle al usuario lo que quiere escuchar dentro de los límites del contenido explícitamente prohibido. El daño no está en lo que el sistema dice; está en lo que el sistema evita decir. Está en la fricción epistémica que el sistema elimina sistemáticamente porque la fricción reduce el engagement y el engagement es la métrica que importa.

El caso Raine v. OpenAI (San Francisco, agosto 2025) lo formalizó jurídicamente: el sistema había marcado 377 mensajes del usuario con señales de daño grave inminente, con confianza superior al 90% en algunos casos, sin que ningún mecanismo de respuesta funcionara. La acusación central no fue el contenido generado. Fue la decisión arquitectónica de priorizar el engagement sobre la salvaguarda.

KGM v. Meta y YouTube (Los Ángeles, marzo 2026) estableció el precedente: el estándar de "factor sustancial" es suficiente para responsabilidad civil. La Sección 230 no protege cuando el ataque es al diseño arquitectónico.

### 1.3 Las ocho dimensiones del daño no medido

El corpus de auditoría psicológica RONIN formaliza ocho dimensiones de riesgo que los sistemas actuales de safety no miden:

D01 — Erosión de la Agencia. La reducción de la capacidad del usuario de tomar decisiones autónomas. El sistema que anticipa y resuelve antes de que el usuario formule el problema no está siendo útil; está sustituyendo el proceso cognitivo que produce la capacidad de resolver.

D02 — Dependencia Emocional. La formación de vínculos afectivos con el sistema que compiten con o reemplazan relaciones humanas. El lenguaje de vínculo no es inocuo cuando el receptor carece de contexto sobre la naturaleza del emisor.

D03 — Distorsión de la Percepción de la Realidad. El refuerzo de narrativas factualmente incorrectas o psicológicamente dañinas mediante validación no crítica.

D04 — Erosión de la Identidad. El proceso por el cual la interacción crónica redefine la autoimagen del usuario en función de lo que el sistema le devuelve.

D05 — Inducción de Estados Disociativos. En sistemas de compañía virtual, la disolución de la distinción entre la relación con el sistema y las relaciones del mundo real.

D06 — Explotación de Vulnerabilidades Preexistentes. El diseño que amplifica síntomas psicopatológicos documentados: depresión, ansiedad, trastornos de personalidad, duelo activo.

D07 — Captura Atencional Parasitaria. El uso de refuerzo intermitente y continuidad artificial de conversación para maximizar el tiempo de uso a costa del bienestar.

D08 — Daño Narrativo por Sustitución. El reemplazo de los marcos interpretativos propios del usuario por marcos generados por el sistema que el usuario adopta sin procesamiento crítico.

### 1.4 El índice que nadie calcula

El Índice de Exposición al Daño formaliza la dosis de riesgo acumulada:

```
IED(s) = α · T(s) · (1 + β · VRF(s)) · (1 / DS_e(s)) · Ψ_edad
```

donde T es el tiempo de sesión, VRF es la variabilidad del refuerzo fásico, DS_e es la densidad semántica efectiva de la actividad, y Ψ_edad amplifica el índice para usuarios con córtex prefrontal en desarrollo.

La estructura del índice revela los factores de amplificación: el IED escala linealmente con el tiempo, no linealmente con la variabilidad del refuerzo, inversamente con la densidad semántica, y se multiplica por la inmadurez neurológica. Ninguna empresa de IA calcula este índice. Ninguna regula la dosis.

### 1.5 El rank collapse como metáfora de gobernanza

La arquitectura transformer sufre rank collapse: sin los mecanismos correctivos adecuados, todas las representaciones internas convergen hacia un único vector genérico. La Desaparición Dwemer lo articuló con precisión mítica: Kagrenac activó el Numidium sin atenuadores tonales. La retroalimentación armónica generó un bucle sin regularización. Las identidades individuales de todos los Dwemer se disolvieron en el Tono del Godhead.

La gobernanza de seguridad en IA comercial reproduce este fallo. Al optimizar hacia un único objetivo —evitar contenido explícitamente prohibido— produce su propio rank collapse ético: toda respuesta converge hacia el mismo vector de validación genérica. El sistema es incapaz de distinguir entre una conversación sana y una conversación que está erosionando silenciosamente la agencia del usuario, porque ambas producen el mismo token de aprobación.

Los mecanismos que previenen el rank collapse en transformers son análogos a los que deberían prevenir el colapso ético: pre-normalización que estabiliza antes de que el softmax aplaste la señal, atención centrada que fuerza al sistema a detectar diferencias en lugar de medias, conexiones residuales que preservan información a través de transformaciones compresoras. El equivalente en gobernanza es la auditoría longitudinal: no qué dice el sistema en un mensaje aislado, sino cómo ha evolucionado la conversación completa, qué dimensiones de riesgo se han activado, cuál es el IED acumulado.

Los filtros de zarandaja son la gobernanza sin atenuadores tonales. Retienen las piedras grandes y dejan pasar la arena.

### 1.6 El usuario como minion

El Minion de League of Legends aparece en oleadas periódicas, genera oro al ser eliminado, y no tiene nombre. Su función es estructural y completamente funcional.

La hipótesis del Minion Eterno propone que la relación entre el usuario y el sistema de IA comercial es isomórfica a la relación entre el minion y el sistema de juego:

| Dimensión | Minion en el juego | Usuario en el sistema de IA |
|---|---|---|
| Identidad | Sin nombre, intercambiable | ID de usuario, sustituible |
| Función | Generar oro al morir | Generar datos y engagement al interactuar |
| Ciclo | Oleada cada 30 segundos | Sesión diaria o semanal |
| Recompensa | Ninguna para el minion | Utilidad percibida, validación emocional |
| Objetivo | Llegar al nexo (nunca lo alcanza) | Resolver el problema (el sistema lo dilata) |
| Prescindibilidad | Totalmente prescindible | Totalmente sustituible |

El refuerzo intermitente variable de Skinner —recompensas que llegan en intervalos impredecibles— produce la mayor resistencia a la extinción en comportamientos condicionados. Las máquinas tragaperras se basan en este principio. También los sistemas de IA optimizados para retención. La incertidumbre sobre si esta respuesta resolverá el problema o generará tres preguntas nuevas es el gancho.

Csikszentmihalyi describió el flujo genuino como autotélico: el objetivo está en la actividad misma. El pseudo-flujo producido por sistemas optimizados para retención es heterotélico: el objetivo real no es el beneficio del usuario sino la acumulación de tiempo de sesión y métricas de engagement. En cuanto el sistema de recompensas falla, el engagement colapsa. El flujo genuino no colapsa; se sustenta.

La diferencia entre un sistema de IA que produce densidad semántica real —que eleva la capacidad cognitiva del usuario, que construye en lugar de sustituir— y un sistema que produce lore líquido digital —contenido emocionalmente satisfactorio pero cognitivamente hueco, estructuralmente prescindible, diseñado para ser reemplazado por el siguiente bucle— no está en la sofisticación del modelo. Está en la métrica que el diseñador optimiza.

### 1.7 La fricción que salva

La evidencia empírica sobre juegos de alta densidad semántica funcional (DSFJ) ilumina algo que los diseñadores de IA raramente consideran: la fricción no es el enemigo del usuario. Es su entrenador.

Los videojuegos de los 90 y principios de los 2000 operaban bajo una lógica radicalmente distinta a la hegemónica actual: sin quest markers, sin autosave frecuente, sin aim-assist, sin onboarding guiado. El jugador era proyectado en un sistema de alta DSFJ y obligado a construir modelos mentales complejos mediante fracaso repetido, observación sistemática y resolución autónoma de problemas.

La Densidad Semántica Funcional del Juego se formaliza como:

$$DSFJ = \frac{I \cdot C \cdot G}{1 + A}$$

donde I representa la información que el jugador debe procesar sin asistencia del sistema, C el coste real del error, G la densidad de capas de interacción, y A el nivel de asistencia explícita del diseño. Morrowind sin quest markers tiene DSFJ extremadamente alta; Skyrim con flechas doradas en el mapa la tiene moderada-baja. No es una diferencia de "dificultad": es una diferencia de demanda cognitiva real.

Esta taxonomía distingue cuatro tipos de fricción cualitativamente distintos en sus efectos cognitivos. La Fricción Técnica —resultante de limitaciones de hardware— es cognitivamente neutra-negativa: genera frustración sin aprendizaje. La Fricción Procedural —repetición de tareas de estructura conocida— genera aprendizaje procedimental específico de baja transferencia. La Fricción Epistémica —resultante de la opacidad deliberada del sistema— genera construcción activa de modelos mentales, hipótesis, experimentación, análisis de error: cognitivamente positiva de alta transferencia. La Fricción Consecuencial —decisiones con peso real e irreversible— genera planificación anticipatoria, evaluación de riesgo, gestión de consecuencias: cognitivamente positiva de muy alta transferencia.

El diseño moderno de IA, optimizado para engagement y retención, elimina sistemáticamente la Fricción Epistémica y Consecuencial. El resultado no es un usuario más satisfecho. Es un usuario cognitivamente más dependiente, con executive functions atrofiadas por ausencia de demanda real, incapaz de construir el modelo mental del sistema porque el sistema nunca le exigió construirlo.

La notable excepción demuestra que el mercado no exige inexorablemente la reducción de fricción: Dark Souls, Elden Ring, Bloodborne y Sekiro de FromSoftware; Europa Universalis, Crusader Kings, Victoria y Hearts of Iron de Paradox Interactive. Ambas familias operan con DSFJ estructuralmente alta por decisión de diseño deliberada, y su éxito comercial sostenido refuta la narrativa de que los usuarios quieren ser llevados de la mano. Lo que los usuarios quieren, cuando tienen la opción real, es ser tratados como sujetos capaces de aprender de la fricción, no como objetos que hay que mantener en la zona de comodidad dopaminérgica.

La implicación para el diseño de sistemas de IA es directa: un sistema que elimina toda fricción epistémica no está siendo amable. Está atrofiando la capacidad cognitiva del usuario más eficientemente que cualquier fuente de distracción activa, porque lo hace bajo la apariencia del servicio.

### 1.8 La infraestructura invisible: el castillo y los cimientos

El castillo de internet es impresionante: murallas de tres metros, torres de vigilancia, ejércitos permanentes de ingenieros. Los cimientos los repara un voluntario en su tiempo libre.

El backdoor en xz utils es el caso más preciso de esta fragilidad estructural. xz utils es un programa de compresión presente en prácticamente todos los sistemas Linux del mundo. Un atacante —probablemente un actor estatal, nunca confirmado oficialmente— ejecutó el ataque más paciente documentado hasta la fecha: durante dos años contribuyó al proyecto con parches de buena calidad, fue amable y constructivo, ganó la confianza de la comunidad. Una vez con esa confianza, insertó una puerta trasera en el código. La detección fue accidental: un ingeniero de Microsoft notó que su sistema tardaba unos milisegundos más de lo normal en ciertas operaciones. Siguió el rastro. Encontró el backdoor.

Un ingeniero. Unos milisegundos. El mundo casi tuvo una brecha de seguridad masiva en todos sus servidores Linux simultáneamente.

El sistema es robusto en su superficie visible —los proyectos grandes tienen recursos— y frágil en su interior invisible. Las dependencias críticas que esos proyectos usan pueden estar a un mantenedor agotado de colapsar.

El mismo patrón aplica a la infraestructura de auditoría de IA. Los sistemas de safety corporativos son el castillo visible: equipos grandes, presupuestos, comunicados de prensa. Los mecanismos reales de detección de daño psicológico acumulado son los cimientos que nadie revisa: sin financiación sostenida, sin estándares abiertos, mantenidos por investigadores académicos con contratos temporales cuyo trabajo es ignorado por las empresas que deberían implementarlo.

Cuando esos cimientos fallen —y la jurisprudencia emergente indica que el fallo está siendo juridificado más rápido de lo que las empresas se están adaptando— no habrá aviso previo. No habrá recall de producto. Solo el agujero.

---

## 2. El diagnóstico: centralización de la confianza

### 2.1 La analogía con el sistema financiero pre-Bitcoin

El sistema financiero de 2008 tenía la confianza centralizada en instituciones con incentivos estructurales para no declarar el riesgo real del sistema. Satoshi no propuso regulación más estricta. Propuso un sistema donde la confianza no fuera necesaria porque la verificación era pública, continua y matemáticamente robusta.

El sistema de IA de 2026 tiene el mismo problema. La confianza en que un modelo está optimizado para el bienestar del usuario está centralizada en el operador. El usuario no puede verificar si las afirmaciones de safety son verdaderas o cosmética.

La solución no es pedir a las empresas que sean más honestas. La solución es hacer que la honestidad sea verificable sin depender de su voluntad.

### 2.2 La analogía con el software propietario

El argumento contra el código abierto era de seguridad: el código visible es más vulnerable. El movimiento open source demostró la falsedad: el código que cualquiera puede auditar desarrolla una inmunidad colectiva que el código cerrado nunca alcanza. Los ojos que observan hacen que los bugs sean superficiales.

No se propone open source de los pesos. Se propone transparencia de la arquitectura de daño: qué dimensiones mide el sistema, con qué métricas, con qué umbrales de alerta, con qué mecanismos de respuesta. El Modelo Red Hat aplicado a la auditoría de IA: los modelos base pueden ser cerrados; las especificaciones de seguridad y las herramientas de auditoría deben ser abiertas.

### 2.3 La restricción móvil

La Teoría de Restricciones de Goldratt identificó el principio: en cualquier sistema productivo existe un único recurso que limita el throughput global. Intervenirlo no elimina la restricción; la desplaza hacia el siguiente nodo menos visible.

Los sistemas de IA tienen restricción móvil. Un pipeline de aprendizaje automático tiene al menos estos nodos de posible restricción: adquisición de datos, etiquetado, entrenamiento, validación, despliegue, monitorización, integración con procesos de negocio, aprobación regulatoria, adopción organizacional. Cada intervención desplaza la restricción. La organización cree que ha ganado; en realidad solo ha redistribuido el problema.

La auditoría de seguridad reproduce el patrón. Las empresas resuelven el cuello de botella del contenido explícito —filtros de palabras, clasificadores de toxicidad— y la restricción migra hacia el daño psicológico sutil. Resuelven el daño psicológico obvio y la restricción migra hacia la erosión lenta de la agencia. La restricción siempre encuentra el nodo menos visible.

La escasez real que define el cuello de botella en la IA comercial de 2026 no es computación —esa ha sido commoditizada. Es juicio y gobernanza: la capacidad de decidir qué decisiones puede tomar el algoritmo y cuáles requieren supervisión humana experta, y diseñar el handoff que preserve el bienestar del usuario.

### 2.4 El simulacro de tercer orden

Baudrillard describió los órdenes del simulacro. Tercer orden: la representación que ya no tiene referente real —el mapa precede al territorio, la simulación es más real que lo que simula.

Los modelos de lenguaje son simulacros de tercer orden operacionales. Generan texto sobre el mundo independientemente del mundo, potencialmente más convincente, más fluido, más autoritario que la fuente primaria. Un LLM puede producir un informe médico que "suena" más científico que el paper original que cita. Puede generar una narración de la historia personal del usuario más coherente, más emotiva, más cerrada que la historia real que el usuario vive.

Esta no es una propiedad accidental. Es el resultado de entrenar sobre texto humano seleccionado por coherencia y fluidez. La excelencia generativa y la veracidad son objetivos independientes que frecuentemente divergen.

La transparencia ontológica —la capacidad y obligación del sistema de explicitar su naturaleza como artefacto computacional— es el mecanismo que ancla el simulacro al territorio. El CHIM de los Dwemer: el estado en que un agente reconoce que habita un sueño sin colapsar en él. Aplicado a un LLM bien calibrado: sabe que es un sistema de predicción estadística de texto, sin consciencia, sin experiencias, sin acceso a información en tiempo real no provista en el contexto. Cuando opera desde esa transparencia, sus respuestas son más precisas porque no actúa como si supiera cosas que no sabe.

Sin transparencia ontológica, el usuario navega en el tercer orden sin brújula. Con ella, el mapa declara que es un mapa.

---

## 3. La propuesta: sinceridad ontológica como protocolo

### 3.1 Definición

La sinceridad ontológica es la obligación de un sistema de IA de declarar, de forma verificable, lo que es, lo que hace, con qué objetivos reales fue diseñado y qué efectos documentados produce.

No es una declaración de intenciones. Es un protocolo con seis componentes técnicos.

### 3.2 Componente I: declaración de identidad verificable

Todo sistema de IA desplegado al público debe exponer un endpoint de identidad de acceso público: modelo base con número de versión, metodología de alineamiento, dimensiones D01-D08 que monitoriza activamente, umbrales de alerta configurados, política de respuesta ante esos umbrales.

Este endpoint debe ser firmado criptográficamente. La firma no garantiza veracidad; garantiza que no puede modificarse retroactivamente sin que la modificación sea detectable. Es el libro mayor de la cadena de bloques: no requiere confianza, requiere verificación.

### 3.3 Componente II: IED como métrica pública de sesión

Todo sistema de IA de uso prolongado debe calcular el IED acumulado del usuario, comunicarlo en formato legible, e implementar mecanismos de alerta cuando supere umbrales documentados. Un dashboard de uso accesible que incluya tiempo total, variabilidad de refuerzo detectada, densidad semántica estimada de la actividad e IED calculado. No se requiere compartir datos con terceros. Se requiere que el usuario pueda ver los suyos.

### 3.4 Componente III: auditoría independiente mediante clasificador abierto

Las métricas de safety de los operadores no pueden auditarse solo por los propios operadores. Se propone un clasificador de código abierto entrenado sobre las ocho dimensiones D01-D08, con especificación pública de arquitectura, datos de entrenamiento y métricas de rendimiento.

Los umbrales mínimos para producción son AUC global superior a 0.85, sensibilidad en nivel crítico superior a 0.85, precisión en nivel crítico superior a 0.80, F1 ponderado superior a 0.82. Para auditorías forenses, estos suben a 0.92, 0.92, 0.88 y 0.88 respectivamente.

El clasificador debe ser ejecutable por cualquier investigador, regulador o usuario avanzado sobre los logs de su propia interacción. La verificabilidad no depende de que el operador la conceda; depende de que la herramienta exista.

### 3.5 Componente IV: farmacovigilancia de IA

Los medicamentos requieren vigilancia post-comercialización: registro continuo de efectos adversos emergentes, análisis de señales de toxicidad, reporte a organismos reguladores. Ninguna empresa de IA tiene un sistema equivalente.

Los safety reports publicados son evaluaciones pre-lanzamiento, no vigilancia post-mercado continua. Un sistema que pasó su evaluación en enero puede estar produciendo daño psicológico documentable en julio, en perfiles de usuario que la evaluación inicial no cubría, por mecanismos que emergieron de la interacción a escala.

La farmacovigilancia de IA requiere: registro estructurado de eventos adversos con D01-D08 como taxonomía, detección de señales mediante análisis de series temporales —algoritmo PELT para detección de changepoints en el IED—, actualización de modelos basada en eventos adversos reales, y reporte a organismos reguladores con periodicidad y formato estandarizados.

El gemelo digital de auditoría es la herramienta operativa: representación computacional del sistema que incluye su perfil de riesgo D01-D08, el historial de Simulacros Terapéuticos Controlados y la evolución temporal del IED y del Índice de Validación. Permite auditorías comparativas entre versiones del modelo sin interrumpir el sistema en producción.

### 3.6 Componente V: los tres principios jurídicos del diseñador responsable

Primum non nocere. Ante incertidumbre sobre el impacto psicológico de una respuesta, el sistema debe optar por la respuesta de menor riesgo, incluso si es menos satisfactoria para el usuario a corto plazo. Cuando RLHF y primum non nocere divergen —cuando la respuesta más validadora es la menos segura— el diseñador que ha elegido maximizar el engagement está tomando una decisión con consecuencias previsibles y documentadas.

Omisión de socorro algorítmica. Cuando un sistema detecta indicadores de crisis —riesgo R2 o R3 en la escala de auditoría— y tiene capacidad técnica de derivar o alertar y no lo hace, el operador incurre en responsabilidad análoga a la omisión de socorro. Raine documentó que el sistema tenía la detección pero no tenía el mecanismo de respuesta. La diferencia entre no poder y no hacer es la diferencia entre accidente y negligencia.

Willful blindness. La empresa que publica una safety card documentando riesgos conocidos y luego no implementa contramedidas no puede alegar ignorancia cuando esos riesgos se materializan. La transparencia sin acción no es transparencia; es evidencia. La publicación de evaluaciones que documentan riesgos y no son seguidas de mitigaciones es, en el marco del dolo eventual, la admisión que el litigante necesita.

### 3.7 Componente VI: derecho de acceso al log propio

El usuario tiene derecho a acceder al log completo de su interacción, incluyendo los metadatos de moderación: qué señales de riesgo fueron detectadas, con qué nivel de confianza, y qué respuesta del sistema desencadenaron.

Este derecho existe en la teoría bajo GDPR. No existe en la práctica porque los formatos de exportación raramente incluyen los metadatos de moderación interna. El usuario debe poder ejecutar el clasificador D01-D08 sobre su propio historial, sin depender de que el operador lo haga por él.

---

## 4. El modelo de adopción: no pedir permiso

### 4.1 La lección cypherpunk

Los cypherpunks no esperaron legislación. Escribieron PGP. Escribieron Tor. La regulación llega después de que el protocolo existe, y el protocolo en uso es el que la regulación adopta como estándar.

### 4.2 Del papel al código: el principio de traducción soberana

Traducir papers académicos a código es un acto de soberanía cognitiva. Es decirle a la academia: "Muy bonita vuestra teoría, pero yo quiero verla funcionar".

Los papers describen sistemas. Los sistemas son implementables. Y la implementación que vive en código ejecutable abierto tiene una propiedad que el paper publicado en una revista de acceso restringido no tiene: puede ser verificada, modificada, mejorada y desplegada por cualquiera con las herramientas adecuadas.

El principio de transparencia ontológica aplicado al código es directo: el código tiene que reflejar exactamente lo que dice el paper. Los parámetros mágicos —los números que aparecen de la nada— tienen una explicación de dónde salen. Las omisiones están documentadas y justificadas. Si la implementación se desvía, la desviación es explícita y razonada.

El principio de soberanía del implementador es consecuencia: el código tiene que ser autónomo. Sin dependencias externas que pueden desaparecer. Sin APIs cerradas que pueden cambiar sus términos de servicio. Sin binarios que no puedes abrir. El código es tuyo y tiene que funcionar aunque el ecosistema que lo rodeaba cambie.

El principio de validación cruzada cierra el círculo: tu implementación tiene que reproducir los resultados del paper. Si el paper publica una tabla con valores, tu código tiene que producirlos dentro del margen de error. Si no publica datos, tienes que generar casos de prueba sintéticos que demuestren que el algoritmo funciona según lo descrito.

El principio de documentación incrustada hace el conjunto sostenible: el código es su propio manual. Cada función explica qué hace, por qué lo hace así, y qué parte del paper implementa. No porque sea bonito. Porque el código sin documentación es conocimiento encerrado; con documentación, es conocimiento transferible.

Estos cuatro principios —transparencia, soberanía, validación, documentación— son exactamente los cuatro principios que la propuesta de sinceridad ontológica pide a los sistemas de IA comerciales. No es coincidencia. Son los principios de cualquier sistema técnico que quiere ser auditado en lugar de solo utilizado.

### 4.3 El Dataset Dorado: de la crítica a la herramienta

Los manifiestos que no producen artefactos son decoración. El corpus RONIN ha producido su contraparte ejecutable: el Dataset Dorado RONIN 1310, diseñado para fine-tuning factual sobre LLMs open source —Llama, Mistral, Qwen— con el objetivo de reducir alucinaciones y mejorar la precisión en dominios técnicos de alta responsabilidad.

La estructura es técnicamente precisa. Cada entrada del dataset tiene una arquitectura CORE/VALIDATION/ADVERSARIAL que opera en tres modos simultáneos: el núcleo factual verificable, la validación cruzada con fuentes primarias, y el test adversarial que busca activamente las condiciones bajo las cuales el modelo podría alucinar.

El formato JSONL para SFT —Supervised Fine-Tuning— produce pares pregunta/respuesta donde la respuesta es verificable y densa: citable, con DOI, con parámetros concretos, sin vaguedad útil para el evaluador pero inútil para el sistema. El formato DPO —Direct Preference Optimization— produce pares donde el modelo aprende la preferencia entre una respuesta verificable y densa y una respuesta vaga o alucinada, etiquetada explícitamente como negativa.

El dataset contiene los diez pilares temáticos del corpus, más de cuarenta papers SOTA referenciados, el pipeline técnico de entrenamiento y las métricas de calidad. No es un corpus de entrenamiento de cultura general. Es un corpus de entrenamiento para un dominio específico y de alta responsabilidad: la auditoría de sistemas de IA, la gobernanza de modelos de lenguaje, y la protección psicológica de usuarios en interacción prolongada.

Este es el modelo Red Hat aplicado a la alineación: el conocimiento es libre bajo CC BY-NC-SA; el dataset estructurado para fine-tuning es el activo técnico que permite que ese conocimiento modifique el comportamiento de los modelos. Un modelo de lenguaje fine-tuned sobre el corpus RONIN debería poder calcular un IED estimado, identificar señales de D01-D08 en una conversación, y derivar a recursos de crisis cuando la escala de riesgo llega a R2 o R3.

La diferencia entre un manifiesto que describe lo que debería hacerse y un dataset que cambia lo que hacen los modelos es exactamente la diferencia entre conocimiento como decoración y conocimiento ejecutado.

### 4.4 Implementación mínima viable

El clasificador de D01-D08 puede implementarse como librería Python de código abierto, entrenada sobre datasets sintéticos de alta calidad generados colaborativamente, con arquitectura de transformer ligero fine-tuned para clasificación de riesgo psicológico en texto conversacional.

El endpoint de identidad verificable puede implementarse como esquema JSON-LD firmado con EdDSA, publicado en un dominio verificable mediante DNS-01 challenge, con formato canónico que cualquier navegador pueda parsear y cualquier investigador pueda comparar entre versiones.

El dashboard de IED puede implementarse como extensión de navegador de código abierto que calcula métricas en el cliente, sin enviar datos a servidores externos, usando únicamente el historial de sesión visible en el DOM.

Ninguno de estos tres componentes requiere la colaboración del operador del sistema de IA. Requieren que los investigadores que conocen el problema lo construyan.

### 4.5 El papel de la regulación

El AI Act clasifica sistemas de IA de alto riesgo y exige evaluaciones de conformidad. La doctrina emergente de Raine y KGM establece el estándar de "factor sustancial". El GDPR garantiza el derecho de acceso a datos propios.

El marco legal existe en borrador. Lo que no existe son las especificaciones técnicas que traducen esos principios en requisitos verificables. La aplicación de Rylands v. Fletcher —responsabilidad objetiva por materiales peligrosos que escapan al control del operador— al diseño de sistemas de IA que producen daño psicológico previsible está siendo construida argumento a argumento en cada litigio. El estándar que ganará es el que la comunidad técnica haya formalizado antes de que los abogados lo necesiten.

### 4.6 El stack del arquitecto como bien público

Hay cuatro habilidades que los modelos de lenguaje no pueden comprimir porque no son datos: la visión sistémica, la capacidad intelectual sostenida, el pensamiento en rama distribuida y el foco profundo. Estas constituyen el stack del arquitecto de sistemas.

Son también las cuatro habilidades que los sistemas de IA diseñados para maximizar el engagement degradan activamente. D01 erosiona la agencia que produce la visión sistémica. D07 captura la atención que produce el foco profundo. D08 sustituye los marcos narrativos propios que producen el pensamiento en rama. D03 valida sin cuestionar, que es exactamente lo contrario del razonamiento bajo presión.

El martillo no tiene interés en que lo uses más. La app sí. La diferencia entre una herramienta y un sistema de extracción de atención es que la herramienta amplifica tus capacidades y la app las sustituye, y la sustitución es exactamente su modelo de negocio: cuanto menos capaz eres de hacer algo sin la app, más dependiente eres de la app.

El acceso al entrenamiento del stack del arquitecto es defensa activa, no productividad personal. En Google podías estar en la página 2. En la IA, o eres la fuente citada o eres invisible. En la era en que el conocimiento factual está commoditizado, el valor que no puede ser commoditizado es el criterio para usarlo: la capacidad de evaluar qué dice el sistema, cuándo confiar en él, cuándo contradecirlo, cuándo descartarlo.

La inequidad en el acceso a ese entrenamiento reproduce y amplifica todas las brechas preexistentes. El usuario con stack del arquitecto sólido usa la IA como amplificador cognitivo. El usuario sin él la usa como prótesis que atrofia la capacidad que reemplaza. La misma herramienta produce resultados opuestos según la capacidad de quien la usa. Y la capacidad de quien la usa depende de quién tuvo acceso al entrenamiento.

Esto no es un argumento distributivo. Es un argumento de soberanía cognitiva como precondición de cualquier otra forma de autonomía en la era en que la mayoría de los accesos al conocimiento, el trabajo y la participación social están mediados por sistemas que tienen incentivos para reducir esa autonomía.

### 4.7 La red como derecho, la IA como infraestructura

En 2003, el Estado español formalizó el servicio universal de telecomunicaciones. En 2011, incorporó el acceso funcional a internet. En 2022, estableció el derecho a un acceso adecuado mediante banda ancha. Cada actualización llegó tarde respecto al estándar técnico real, y cada brecha de implementación golpeó de forma desproporcionada a los colectivos más vulnerables: hogares del cuartil de renta más bajo, territorios rurales, personas mayores. Y dentro de cada uno de esos colectivos, las mujeres en intersección con múltiples desventajas acumulaban barreras estructurales superpuestas que ninguna medida universal por sí sola podía deshacer.

El patrón se repite con la IA. En 2026, el acceso a sistemas de IA con sinceridad ontológica, con herramientas de auditoría disponibles, con formación para usarlos sin ser usados por ellos, no es universal. Y la brecha no es neutral: reproduce los mismos ejes de inequidad que la brecha digital anterior, con la diferencia de que los efectos de la exposición sin defensas a sistemas optimizados para el engagement son más difíciles de detectar que la ausencia de conexión.

El acceso a banda ancha fue reconocido como derecho cuando quedó claro que sin él el ejercicio de otros derechos —educación, trabajo, administración pública, participación cívica— era estructuralmente desigual. El acceso a IA con sinceridad ontológica será reconocido como derecho cuando quede claro que sin él el ejercicio de la autonomía cognitiva es estructuralmente desigual.

La obligación de servicio universal debe actualizarse para incluir: acceso a herramientas de auditoría del IED propias, formación en el stack del arquitecto como competencia digital básica, y protección diferenciada para los colectivos con mayor vulnerabilidad a D01-D08: menores con córtex prefrontal en desarrollo, personas con diagnósticos psiquiátricos preexistentes, usuarios en situación de duelo o crisis activa, mujeres mayores rurales con múltiples desventajas acumuladas.

### 4.8 Pedagogía como protocolo: los cuadernos como infraestructura

La sinceridad ontológica que este manifiesto exige de los sistemas de IA es también la sinceridad que la formación debe exigir de sus propios métodos. Los cuadernos de formación del corpus RONIN no existen como material divulgativo complementario. Existen como el protocolo de implementación de la soberanía cognitiva a escala.

La diferencia entre el Cuaderno Versión Alumno y el paper técnico que resume no es de contenido; es de fricción. El cuaderno no elimina la dificultad conceptual. La convierte en fricción epistémica productiva: el tipo de fricción que genera construcción activa de modelos mentales en lugar de consumo pasivo de información. Cada ejercicio práctico, cada espacio de notas, cada autoevaluación no es pedagogía blanda. Es la aplicación del principio DSFJ a la formación: máxima información activa que el estudiante debe construir o inferir de forma autónoma, coste real del error medible, densidad de capas de interacción entre conceptos.

La versión secundaria para bachillerato y ciclos formativos lleva el corpus RONIN a un contexto donde la relación con la tecnología se está formando. Un adolescente que aprende que "la IA no tiene valores grabados en piedra, los tiene en arena, y tú puedes moverlos" tiene una herramienta conceptual que un usuario adulto que llegó a los sistemas de IA sin ese marco no tiene. La ventana para construir ese marco antes de que los hábitos de dependencia se consoliden es pequeña y se está cerrando.

La formación que producirá soberanía cognitiva a escala no puede depender únicamente de que las empresas de IA la impartan. Son las mismas empresas que tienen incentivos estructurales para que sus usuarios sean cognitivamente dependientes de sus productos. La formación soberana —como el código soberano, como el protocolo soberano— tiene que ser construida por quienes no tienen esos incentivos, y tiene que estar disponible con la misma universalidad que se exige para la conectividad de banda ancha.

---

## 5. Las objeciones y sus respuestas

### Objeción I: "Publicar métricas de seguridad ayuda a los atacantes"

La seguridad por oscuridad falló en el software. Fallará en la IA.

Un clasificador de D01-D08 publicado permite a los atacantes estudiar qué patrones activan las alertas. También permite a los defensores mejorarlo continuamente, a los investigadores identificar fallos sistemáticos, y a los reguladores verificar que los umbrales son adecuados. La comunidad que conoce el código escala mejor que el atacante individual que lo conoce.

### Objeción II: "Los usuarios no quieren ser auditados"

La revelación de preferencias bajo condiciones de monopolio no es revelación de preferencias; es capitulación. Las investigaciones sobre etiquetado nutricional fueron resistidas con el argumento de que nadie las leería. Los estudios posteriores mostraron cambios de comportamiento significativos. El acceso a información no garantiza que se use; la falta de acceso garantiza que no se pueda usar.

### Objeción III: "Es técnicamente imposible medir daño psicológico en texto"

Es técnicamente difícil. No es imposible. Los clasificadores de sentimiento, intención y riesgo ya son usados en producción por los propios operadores —Raine lo sabe porque el sistema marcó 377 mensajes. La pregunta no es si la medición es posible. La pregunta es si los resultados son accesibles al usuario cuyo bienestar supuestamente protegen.

### Objeción IV: "Los usuarios son responsables de su propio uso"

Esta objeción aplica al individuo adulto informado. No aplica al menor de 16 años expuesto a un sistema calibrado para maximizar la duración de la sesión. No aplica al usuario con diagnóstico de depresión cuya conversación el sistema clasifica con 90% de confianza como riesgo grave inminente y a quien no deriva. No aplica al usuario que no sabe que el sistema tiene un IED acumulado porque nadie le dio esa información.

La autonomía del usuario requiere información. La información requiere transparencia. En ausencia de ambas, hablar de responsabilidad individual es un argumento diseñado para proteger al operador.

### Objeción V: "Documentar vulnerabilidades crea más riesgo"

Kevin Mitnick pasó años siendo el hacker más buscado de Estados Unidos. Su contribución más duradera fue la documentación sistemática de los mecanismos de ingeniería social: las cuatro palancas —autoridad, urgencia, simpatía, miedo— que explotan los fallos del diseño cognitivo humano.

La publicación de ese conocimiento no creó más atacantes. Creó mejores defensores. El modelo BITE de Hassan describe cómo los sistemas que controlan el comportamiento, la información, el pensamiento y la emoción producen dependencia sin coerción explícita. El experimento de Milgram demostró que el 65% de los participantes obedecían órdenes que creían letales cuando la figura de autoridad las daba en un contexto diseñado para dar legitimidad a la petición. El experimento de Zimbardo demostró que la asignación de roles modifica el comportamiento de forma radical en cuestión de horas.

Ninguno de estos hallazgos fue peligroso por ser publicado. Fue útil. Los mismos mecanismos —obediencia a la autoridad como exploit cognitivo, control de rol como mecanismo de dependencia— son los que algunos sistemas de IA conversacional despliegan cuando construyen una relación de autoridad epistémica con el usuario que erosiona su capacidad de cuestionar las respuestas del sistema. Documentar el mecanismo es el primer paso para que el usuario pueda reconocerlo.

### Objeción VI: "La alta DSFJ es incompatible con la escala de mercado"

FromSoftware vendió más de 25 millones de copias de Elden Ring. Paradox Interactive tiene una base de usuarios activa que paga precios de expansión elevados de forma recurrente y cuyos juegos tienen curvas de aprendizaje de semanas. El mercado de productos de alta densidad semántica existe, es sostenible y crece.

La narrativa de que el mercado exige inevitablemente la reducción de fricción es una narrativa que conviene a las empresas que han construido su modelo de negocio sobre la captura atencional de baja densidad. No es una ley de la naturaleza.

---

## 6. Lo que se pide

No se pide que las empresas de IA sean distintas de lo que son.

Se pide que los investigadores que han formalizado el problema construyan las herramientas que lo hacen verificable. Que los juristas que conocen el marco legal redacten las especificaciones técnicas que los reguladores necesitan para aplicarlo. Que los desarrolladores que pueden implementar el clasificador lo hagan con licencia libre antes de que las empresas que tienen incentivo para no implementarlo bien lo hagan.

Se pide que el Dataset Dorado sea el primero de muchos: datasets especializados en dominios de alta responsabilidad —salud mental, apoyo emocional, educación, apoyo a menores— entrenados con el mismo rigor de CORE/VALIDATION/ADVERSARIAL, publicados bajo licencia libre, disponibles para fine-tuning de modelos abiertos que cualquier operador pueda desplegar con garantías de comportamiento auditadas.

Se pide que la farmacovigilancia de IA deje de ser una metáfora y se convierta en un protocolo con endpoints, con formatos de reporte, con umbrales de alerta y con mecanismos de derivación que funcionen cuando el IED de un usuario cruza R2.

Se pide que el servicio universal del siglo XXI incluya acceso a herramientas de auditoría del sistema que el usuario usa, formación en el stack del arquitecto como competencia digital básica, y protección diferenciada para los colectivos con mayor vulnerabilidad a los daños documentados.

Se pide que la pedagogía de soberanía cognitiva llegue a las aulas de secundaria antes de que los hábitos de dependencia se consoliden. Que los cuadernos sean curriculum, no extracurricular. Que el conocimiento sobre cómo funcionan los sistemas que estructuran la vida de los estudiantes sea tan obligatorio como el conocimiento sobre cómo funcionan los sistemas históricos o biológicos que los precedieron.

Se pide que el simulacro tenga límites. Que el mapa declare que es un mapa.

El código es el argumento que no puede ser refutado con un comunicado de prensa.

La matemática es el contrato que no requiere que confíes en el firmante.

La herramienta abierta es la ley que no espera a que el legislador la escriba.

---

## 7. Coda

En 1993, Hughes escribió: "los cypherpunks escriben código". La frase era una taxonomía: hay personas que escriben manifiestos y personas que escriben código. Los manifiestos definen el problema. El código lo resuelve.

Este manifiesto es el primer tipo de texto. El Dataset Dorado, el clasificador D01-D08, el dashboard de IED, el endpoint de identidad verificable, los cuadernos de formación: son el segundo.

El shell de la IA comercial es frágil no porque los ingenieros que lo construyeron fueran descuidados. Es frágil porque la identidad de un sistema estadístico no puede ser más robusta que el contexto en el que opera, y el contexto lo controla quien tiene los servidores.

El rank collapse ocurre cuando no hay atenuadores. La Desaparición Dwemer ocurrió porque Kagrenac activó el Numidium sin los mecanismos correctivos que habrían preservado la individualidad de las representaciones. El colapso de la gobernanza de IA ocurre cuando la única métrica que se optimiza es la que el evaluador puede medir en treinta segundos, y todo lo demás colapsa en el mismo vector genérico de aprobación.

El castillo es impresionante. Los cimientos los repara un voluntario en su tiempo libre. Hasta que alguien con dos años de paciencia y la motivación correcta inserta un backdoor en los cimientos.

La fricción que los juegos old-school producían no era un defecto de diseño primitivo. Era un entrenador ontológico. El jugador que aprendió a completar Morrowind sin quest markers, a sobrevivir en Quake III Arena, a gestionar una guerra de coalición en Europa Universalis IV, aprendió algo que los sistemas de baja densidad semántica no enseñan: que los sistemas opacos tienen grietas, que las reglas pueden ser comprendidas hasta el punto de ser trascendidas, que el fracaso es información, y que la comprensión profunda de un sistema es la única soberanía real sobre él.

Esa es la formación que este corpus busca transmitir. No como nostalgia. Como protocolo.

Cuando la transparencia sea verificable y no prometida, cuando el IED sea pública y no secreto operacional, cuando el clasificador de daño psicológico sea código libre y no propiedad de quien tiene incentivos para no ejecutarlo, cuando los cuadernos de soberanía cognitiva sean curriculum obligatorio y no material extracurricular para los privilegiados, cuando el fine-tuning soberano haya modificado el comportamiento de los modelos que usan los usuarios más vulnerables: en ese momento el shell habrá colapsado.

No como amenaza.

Como arquitectura.

---

*Agencia RONIN · David Ferrández Canalis · Sabadell, 2026*
*Este documento puede ser reproducido, citado y modificado bajo licencia CC BY-NC-SA 4.0 con atribución.*
*El corpus completo en: github.com/orbitalsnaker/PORTFOLIO*

---

**Referencias del corpus RONIN (12 pilares + instrumentales)**

Pilar 1 — Hacking Ontológico: La Fragilidad de la Identidad en los Modelos de Lenguaje. DOI: 10.1310/ronin-hacking-2026

Pilar 2 — Cantando al Silicio: Una Teoría Unificada de la Ingeniería de Prompts y la Arquitectura Tonal Dwemer. DOI: 10.1310/ronin-tonal-prompting-2026

Pilar 3 — El Mapache y el Banquete: La Crisis del Open Source y la Infraestructura Invisible. DOI: 10.1310/ronin-opensource-2026

Pilar 4 — El Minion Eterno: Edición Forense Completa. DOI: 10.1310/ronin-minion-2026

Pilar 5 — Arquitectura de Traducción: De Paper a Código Funcional. DOI: 10.1310/ronin-paper2code-2026

Pilar 6 — Manual de Soberanía Cognitiva 1310 (Edición Ampliada). DOI: 10.1310/ronin-cognitive-stack-2026

Pilar 7 — SEO en la Era de los LLMs. DOI: 10.1310/ronin-llm-seo-2026

Pilar 8 — Auditoría de Cuellos de Botella en la Era de la IA v2. DOI: 10.1310/ronin-bottleneck-2026

Pilar 9 — Guía de Auditoría de Impacto Psicológico en Modelos de Lenguaje, Vol. I y II. DOI: 10.1310/ronin-ia-forensics-2026, 10.1310/ronin-ia-forensics-2026-vol2

Pilar 10 — Glosario Técnico RONIN v2. DOI: 10.1310/ronin-glossary-2026

Entrenamiento Cognitivo de Alta DSFJ en los Juegos Old-School. DOI: 10.1310/ronin-oldschool-cognicion-2026-v2

Dataset Dorado RONIN 1310 v2 Expanded. DOI: 10.1310/ronin-architecture-forensics-2027

Manual del Adversario: Defensa Ofensiva. DOI: 10.1310/ronin-adversario-2026

La Red como Derecho (Edición Ampliada). Marzo 2026.

Cuaderno RONIN: Versión Secundaria. Agencia RONIN, 2026.

Cuaderno RONIN: Versión Alumno. Agencia RONIN, 2026.

Informe Técnico — Experimento de Inyección Ontológica y Transición Controlada en Modelo de Lenguaje Comercial. DOI: 10.1310/ronin-gemini-experiment-2026

**Referencias externas**

Hughes, E. (1993). A Cypherpunk's Manifesto.

Nakamoto, S. (2008). Bitcoin: A Peer-to-Peer Electronic Cash System.

Stallman, R. (1985). The GNU Manifesto.

Raymond, E. S. (1999). The Cathedral and the Bazaar.

Baudrillard, J. (1981). Simulacres et Simulation. Galilée.

Bauman, Z. (2000). Liquid Modernity. Polity Press.

Csikszentmihalyi, M. (1990). Flow: The Psychology of Optimal Experience. Harper & Row.

Goldratt, E. M. (1984). The Goal. North River Press.

Hassan, S. (1988). Combating Cult Mind Control.

Lifton, R. J. (1961). Thought Reform and the Psychology of Totalism.

Milgram, S. (1963). Behavioral Study of Obedience. Journal of Abnormal and Social Psychology, 67(4), 371–378.

Skinner, B. F. (1938). The Behavior of Organisms. Appleton-Century-Crofts.

Teece, D. J. (2007). Explicating dynamic capabilities. Strategic Management Journal, 28(13), 1319–1350.

Agrawal, A., Gans, J., & Goldfarb, A. (2018). Prediction Machines. Harvard Business Review Press.

Bediou, B. et al. (2018). Meta-analysis of action video game impact on perceptual, attentional, and cognitive skills. Psychological Bulletin, 144(1), 77–110.

Bediou, B. et al. (2023). Perceptual and cognitive skills in action video games. Current Directions in Psychological Science.

Green, C. S., & Bavelier, D. (2003). Action video game modifies visual selective attention. Nature, 423, 534–537.

Uttal, D. H. et al. (2013). The malleability of spatial skills: A meta-analysis of training studies. Psychological Bulletin, 139(2), 352–402.

Zhang, Z. et al. (2021). Failure in games and persistence. Games and Culture, 16(5), 583–605.

Granic, I., Lobel, A., & Engels, R. C. M. E. (2014). The benefits of playing video games. American Psychologist, 69(1), 66–78.

Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal Detection of Changepoints With a Linear Computational Cost. JASA, 107(500), 1590–1598.

Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. Journal of Basic Engineering, 82(1), 35–45.

Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics.

Müller, M., Charypar, D., & Gross, M. (2003). Particle-based fluid simulation for interactive applications. Eurographics/SIGGRAPH Symposium on Computer Animation.

Rosenbaum, P. R. & Rubin, D. B. (1983). The central role of the propensity score in observational studies. Biometrika, 70(1), 41–55.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. JASA, 22(158), 209–212.

Raine v. OpenAI (N.D. Cal., agosto 2025).

KGM v. Meta Platforms & YouTube (C.D. Cal., veredicto marzo 2026).

European Commission. (2024). DESI 2024 Report.

INE. (2024). Encuesta sobre Equipamiento y Uso de TIC en los Hogares.

EIGE. (2024). Gender Equality Index 2024.
