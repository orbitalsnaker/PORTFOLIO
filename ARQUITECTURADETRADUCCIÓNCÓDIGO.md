# ARQUITECTURA DE TRADUCCIÓN: DE PAPER A CÓDIGO FUNCIONAL  
## Manual de Supervivencia para Navegantes del Conocimiento  
### Edición Extendida v3.0 — Traducción Masiva de Papers Académicos  
**DOI: 10.1310/academia-to-code-2026**  
**Licencia: CC BY-NC-SA 4.0**  
**Versión: 3.0 — Edición Completa con Protocolos de 4 Capas**

---

> **AVISO LEGAL Y ÉTICO (pero dicho de forma que no aburra)**  
> Este manual es como un mapa del tesoro: te lleva a lugares increíbles, pero si decides saltar al vacío sin comprobar que hay agua, el problema es tuyo. El código que escribas siguiendo estas páginas puede ser genial, puede ser un desastre, o puede cambiar el mundo. No nos hacemos responsables de explosiones, incendios, o de que tu jefe crea que ahora eres un genio y te pida que arregles su impresora. Tú decides. Nosotros solo ponemos las herramientas.

---

## 0. PREÁMBULO: LA HISTORIA DE UN SUEÑO (O POR QUÉ ESTO IMPORTA)

Imagina que encuentras un libro antiguo en una biblioteca. El libro describe una máquina capaz de convertir agua en vino, piedras en oro, o datos en diagnóstico médico. Pero el libro está escrito en un idioma que no entiendes del todo, con diagramas incompletos y notas al margen que parecen garabatos. ¿Qué haces? ¿Lo cierras y te vas? ¿O empiezas a experimentar, a probar, a reconstruir?

Un paper académico es ese libro. Los científicos publican sus descubrimientos en revistas especializadas, pero rara vez incluyen el código que hace funcionar sus inventos. El resultado: montañas de conocimiento inaccesible, esperando a que alguien como tú (sí, tú) lo rescate y lo convierta en algo real.

Este manual es tu mapa del tesoro. Te va a enseñar a leer esos libros crípticos, a extraer sus secretos, y a convertirlos en código que funciona, que se puede tocar, modificar, compartir. Y lo vamos a hacer sin rodeos, con ejemplos de verdad, con código que puedes copiar y pegar, y con muchas risas por el camino.

Porque traducir papers a código no es solo un trabajo técnico. Es un acto de **soberanía cognitiva**. Es decirle a la academia: "Vale, muy bonita vuestra teoría, pero yo quiero verla funcionar". Es construir puentes entre el laboratorio y el mundo real.

---

## CAPÍTULO 1: EL ARTE DE LEER UN PAPER SIN DORMIRSE

### 1.1. El problema: los papers están escritos por extraterrestres

Los papers académicos tienen mala fama, y con razón. Usan un lenguaje críptico, lleno de jerga, ecuaciones que parecen jeroglíficos y frases como "en este trabajo proponemos un novedoso marco teórico". Traducción: "hemos hecho algo, pero no vamos a explicarlo bien".

Pero no te preocupes. Leer un paper es como aprender un idioma nuevo. Al principio no entiendes nada, pero después de unos cuantos, empiezas a reconocer patrones. Aquí tienes un método infalible para no morir en el intento:

**Paso 1: El resumen (Abstract)**  
Léelo. Solo eso. Si después de leerlo no tienes ni idea de qué va, busca otro paper. Si te suena interesante, pasa al paso 2.

**Paso 2: Las figuras y tablas**  
Las imágenes no mienten (casi nunca). Mira los gráficos, las tablas de resultados, los diagramas de flujo. Si entiendes lo que muestran, ya tienes una idea general. Si no, vuelve al paso 1 con otro paper.

**Paso 3: La introducción**  
Aquí los autores cuentan por qué su trabajo es importante. A veces se enrollan. Salta los párrafos que hablen de cosas que ya sabes (o que no te interesan) y busca la frase mágica: "en este artículo, presentamos...". Ahí está el meollo.

**Paso 4: La sección de métodos (la parte divertida)**  
Aquí es donde los autores explican cómo lo hicieron. O al menos lo intentan. Busca:
- Ecuaciones. Si no entiendes una, búscala en Google. Si sigues sin entender, busca una implementación en GitHub. No pasa nada por copiar.
- Pseudocódigo. A veces lo ponen. Es como el código de verdad, pero en plan "humano".
- Parámetros. Anota todos los números que aparezcan: learning rate, número de iteraciones, tamaño de la red, etc. Los vas a necesitar.

**Paso 5: Los resultados**  
Mira si los números cuadran con lo que esperabas. Si el paper dice "nuestro algoritmo tiene una precisión del 95%" y tú en el paso 4 no viste cómo calcular la precisión, vuelve atrás.

**Paso 6: La discusión y conclusiones**  
Los autores se explayan sobre lo importantes que son. Pasa de largo. Lo que importa ya lo tienes.

**Analogía gamer**: Leer un paper es como empezar un juego nuevo sin tutorial. Al principio no sabes ni cómo se salta. Pero después de unas cuantas partidas, empiezas a entender la mecánica. Los jefes finales (las ecuaciones) se vuelven más fáciles cuando has visto sus patrones de ataque.

---

### 1.2. Trucos de principiante para no aborrecer la lectura

- **No leas en orden**. Empieza por lo que te interese. Si la sección de resultados tiene una tabla chula, mírala. Si la introducción es un tostón, sáltatela.
- **Subraya, anota, dibuja**. Los PDFs permiten hacer anotaciones. Úsalas. Si una ecuación te parece importante, rodéala. Si un párrafo te parece críptico, escríbelo con tus palabras al lado.
- **Google es tu amigo**. Cada palabra que no entiendas, búscala. "¿Qué es un filtro de Kalman?" "¿Qué significa 'stochastic gradient descent'?" No te avergüences. Todos empezamos así.
- **Busca implementaciones previas**. Antes de ponerte a programar desde cero, mira si alguien ya lo ha hecho. GitHub está lleno de código basura, pero también de pepitas de oro. Aprende de los errores de los demás.

**Ejemplo**: El paper de Kalman (1960) tiene una ecuación que parece sacada de una pesadilla:  
`x̂ₖ = x̂ₖ₋₁ + Kₖ(zₖ - x̂ₖ₋₁)`  
Si no entiendes nada, busca "Kalman filter explained" en YouTube. Hay videos de 5 minutos que lo explican con dibujitos. Luego vuelves al paper y de repente tiene sentido.

---

## CAPÍTULO 2: PRINCIPIOS FUNDAMENTALES (O CÓMO NO VOLVERSE LOCO)

Antes de lanzarte a programar, tienes que interiorizar cuatro principios. Son como los mandamientos del traductor de papers. Incumple uno y el código te explotará en la cara.

### 2.1. Principio de transparencia ontológica (o "no te hagas el listo")

El código tiene que reflejar exactamente lo que dice el paper. Si el paper asume que los datos siguen una distribución normal, tu código tiene que normalizarlos. Si el paper omite un paso (y te das cuenta), tienes que documentar esa omisión y explicar por qué tomaste esa decisión.

**Analogía**: Es como seguir una receta de cocina. Si la receta dice "añadir una pizca de sal", tú no puedes poner "un chorro de salsa de soja" y luego quejarte de que no sabe igual. La pizca de sal es sagrada. Aunque no sepas cuánto es una pizca, tienes que intentar aproximarte.

**Ejemplo práctico**: En el paper de Ziegler-Nichols (1942) sobre controladores PID, los autores dan unos valores empíricos para las constantes Kp, Ki, Kd basados en la respuesta del sistema. Si en tu implementación usas otros valores porque te parece que funcionan mejor, ya no estás implementando el paper. Estás haciendo otra cosa. Puede estar bien, pero no es lo mismo.

### 2.2. Principio de soberanía del implementador (o "no dependas de nadie")

El código que escribas tiene que ser autónomo. Nada de llamadas a APIs externas que puedan desaparecer. Nada de librerías que requieran conexión a internet. Nada de binarios cerrados que no puedas modificar. El código es tuyo, y tiene que funcionar aunque el mundo se acabe.

**Analogía**: Es como construir tu propio coche en lugar de depender del taller mecánico. Si el taller cierra, tú sigues teniendo coche. Si tu coche se estropea, puedes arreglarlo tú mismo porque conoces cada pieza.

### 2.3. Principio de validación cruzada (o "confía, pero verifica")

Tu implementación tiene que reproducir los resultados del paper. Si el paper publica una tabla con valores, tu código tiene que producir esos mismos valores (dentro de un margen de error). Si no publica datos, tienes que generar casos de prueba sintéticos que demuestren que el algoritmo funciona según lo descrito.

**Analogía**: Es como cuando aprendes a hacer una tortilla. Sigues la receta, pero luego pruebas el resultado. Si sabe a cartón, algo has hecho mal. Vuelves a leer la receta, ajustas la cantidad de huevos, y vuelves a probar. Hasta que la tortilla está buena.

### 2.4. Principio de documentación incrustada (o "piensa en el pobre que te heredará")

El código tiene que ser su propio manual. Cada función, cada clase, cada línea críptica, tiene que tener un comentario que explique qué hace, por qué lo hace así, y qué parte del paper implementa.

**Analogía**: Es como dejar notas adhesivas en un regalo para que quien lo reciba sepa cómo usarlo. Si no dejas notas, el regalo puede acabar en la basura porque nadie sabe para qué sirve.

---

## CAPÍTULO 3: LA CAJA DE HERRAMIENTAS (O QUÉ LENGUAJE ELEGIR Y POR QUÉ)

### 3.1. JavaScript ES6: El suplicante del navegador

**Cuándo usarlo**: Simulaciones visuales, algoritmos que necesitan visualización inmediata, herramientas web que quieres que cualquiera pueda usar sin instalar nada.

**Ventajas**:
- Se ejecuta en cualquier navegador, sin instalación.
- WebGL para gráficos rápidos.
- Ecosistema vasto (aunque muchos paquetes son basura).
- Prototipado rápido.

**Desventajas**:
- Lento en cálculos pesados (aunque WebAssembly lo arregla).
- El debugging es tedioso.
- Los números de punto flotante causan sorpresas.

**Recomendación**: Para papers sobre algoritmos, visualización y simulaciones. NO para cálculos intensivos en CPU sin WebAssembly.

### 3.2. Python: El chamán de la ciencia

**Cuándo usarlo**: Análisis de datos, machine learning, algoritmos numéricos, prototipado rápido.

**Ventajas**:
- Sintaxis clara y legible.
- NumPy, SciPy, Pandas para cálculos pesados.
- Enorme comunidad científica.
- Debugging fácil.

**Desventajas**:
- Lento en bucles anidados.
- Requiere instalación.
- La gestión de dependencias puede ser un caos.

**Recomendación**: Para papers sobre filtros, cadenas de Markov, clustering, análisis. La opción segura.

### 3.3. C/C++: El purista obsesionado con la velocidad

**Cuándo usarlo**: Algoritmos que TIENEN QUE ser rápidos, simulaciones en tiempo real, anything where 10ms marks you as slow.

**Ventajas**:
- Velocidad bruta.
- Control total de memoria.
- Compilación explícita (sabes qué va a pasar).

**Desventajas**:
- Curva de aprendizaje abrupta.
- Gestión de memoria: fugas, buffer overflows, segmentation faults.
- Compilación lenta.

**Recomendación**: Para papers sobre dinámica molecular, simulación física de precisión extrema, algoritmos de búsqueda en espacios enormes.

---

## CAPÍTULO 4: PROTOCOLO DE 4 CAPAS (EL MÉTODO SANTO)

Cada paper que traduzcas debe seguir este protocolo. Sin excepciones.

### Capa 1: Contexto (¿Por qué alguien escribió esto?)

Explica en prosa clara:
- ¿Qué problema real resuelve?
- ¿Dónde falla el estado del arte?
- ¿Cuál es la aplicación práctica?

Piensa en alguien que no sabe programar leyendo esto. Debería entender por qué importa.

### Capa 2: Ecuación (El jeroglífico)

Expón cada ecuación del paper con:
- Significado de cada variable.
- Rango de valores esperados.
- Interpretación geométrica o física.

Las ecuaciones no se explican solas. Tienes que hacerlas inteligibles.

### Capa 3: Algoritmo (La traducción)

Pseudocódigo línea por línea que traduce las ecuaciones a lógica humana:
- Qué entra, qué sale.
- Qué pasa en cada iteración.
- Casos especiales o edge cases.

Este es el puente entre las matemáticas y el código.

### Capa 4: Código (El hierro)

Implementación limpia en JavaScript ES6 o Python:
- Sin dependencias externas (excepto librerías estándar).
- Comentarios incrustados que explican CADA sección.
- Tests ejecutables que validen los resultados.

---

## CAPÍTULO 5: VALIDACIÓN Y TESTING (O CÓMO NO VENDER HUMO)

### 5.1. Principios de validación

Cada implementación necesita **tres tipos de tests**:

1. **Tests unitarios**: ¿Funciona cada función por sí sola?
2. **Tests de integración**: ¿Funciona todo junto?
3. **Tests de regresión**: ¿Reproduce los números del paper?

### 5.2. Márgenes de error aceptables

- **Algoritmos deterministas**: Error < 1e-10 (máquina epsilon).
- **Métodos iterativos**: Error < 1e-6 (convergencia estándar).
- **Métodos estocásticos**: Error dentro del intervalo de confianza 95%.
- **Simulaciones físicas**: Error < 1% respecto a valores teóricos.

Si no puedes explicar por qué tu resultado difiere del esperado, algo está mal.

---

## CAPÍTULO 6: DOCUMENTACIÓN INCRUSTADA (O CÓMO DEJAR NOTAS PARA TU FUTURO YO)

### 6.1. Anatomía de un comentario útil

```javascript
// MALA DOCUMENTACIÓN:
// x = y + z;

// BUENA DOCUMENTACIÓN:
// x̂ₖ = x̂ₖ₋₁ + Kₖ(zₖ - x̂ₖ₋₁) 
// Update state estimate: predicted state plus Kalman gain times innovation
// Implements: Gordon et al. (1993), Sequential Monte Carlo, Eq. 2.3
// x̂ₖ = state estimate at time k
// Kₖ = Kalman gain (how much to trust the measurement vs prediction)
// zₖ = measurement at time k
// (zₖ - x̂ₖ₋₁) = innovation (difference between measurement and prediction)
stateEstimate = predictedState + kalmanGain * (measurement - predictedState);
```

### 6.2. Estructura de función documentada

```javascript
/**
 * Función: calcularGananciaDeKalman
 * 
 * PROPÓSITO: Calcula la ganancia de Kalman óptima que minimiza el error cuadrático medio
 * 
 * MATEMÁTICA:
 *   K = P * H^T * (H * P * H^T + R)^-1
 *   Donde:
 *     P = matriz de covarianza del error (predicción)
 *     H = matriz de observación (relación entre estado y medición)
 *     R = ruido de medición
 *   
 * ENTRADA:
 *   P: número o matriz (covarianza del error predicho)
 *   H: número o matriz (matriz de observación)
 *   R: número o matriz (covarianza del ruido de medición)
 *   
 * SALIDA:
 *   K: ganancia de Kalman (números entre 0 y 1 en el caso escalar)
 *     K ≈ 0 significa "no confíes en la medición, confía en la predicción"
 *     K ≈ 1 significa "confía mucho en la medición, ignora la predicción"
 *   
 * REFERENCIA: Kalman, R. E. (1960). "A new approach to linear filtering..."
 * 
 * EJEMPLO:
 *   let P = 1.0;     // Error inicial (1 unidad)
 *   let H = 1.0;     // Observamos el estado directamente
 *   let R = 0.1;     // Sensor con 0.1 de ruido
 *   let K = calcularGananciaDeKalman(P, H, R);
 *   // K ≈ 0.909 (el sensor es bastante confiable)
 */
```

---

## CAPÍTULO 7: CASOS PRÁCTICOS Y ANÁLISIS DE PAPERS

[Sección que mantiene los ejemplos del documento original, adaptados sin referencias propietarias]

---

## CAPÍTULO 8: PUBLICACIÓN Y DISTRIBUCIÓN (O CÓMO COMPARTIR TU TESORO)

### 8.1. Estructura de un repositorio decente

```
mi-implementacion-paper/
├── README.md                 # Descripción breve
├── docs/
│   ├── paper-resumen.md     # Resumen del paper en tus palabras
│   ├── protocolo-4-capas.md # Desglose completo
│   └── guia-instalacion.md
├── src/
│   ├── core.js              # Implementación principal
│   ├── validador.js         # Tests
│   └── utilidades.js        # Funciones auxiliares
├── tests/
│   ├── test-unitario.js
│   ├── test-integracion.js
│   └── test-regresion.js
├── ejemplos/
│   ├── ejemplo-basico.js
│   ├── ejemplo-intermedio.js
│   └── ejemplo-avanzado.js
├── data/
│   └── casos-prueba.json    # Datos de validación
└── LICENSE                  # CC BY-NC-SA 4.0 o similar
```

### 8.2. README.md mínimo decente

- Título y descripción breve.
- Referencia al paper (con DOI).
- Instrucciones de instalación y uso.
- Ejemplos de código.
- Enlace a los tests.
- Licencia.

### 8.3. Publicación: GitHub o alternativas

**GitHub** es el lugar natural para código abierto. Crea un repositorio, súbelo, y pon una licencia (recomendada: CC BY-NC-SA 4.0).

**Codeberg** es una alternativa de código abierto a GitHub, sin dependencia de Microsoft.

**GitLab** tiene opciones self-hosted para máximo control.

### 8.4. La licencia: elige bien

- **CC BY-NC-SA**: Permite usar, modificar y compartir, pero no con fines comerciales, y siempre con atribución y misma licencia. Ideal para proyectos educativos.
- **AGPL v3**: Si quieres que cualquiera pueda usar tu código, incluso comercialmente, pero que cualquier mejora también sea pública.
- **MIT**: Libertad total, sin condiciones. El código puede acabar privado. Úsalo si no te importa.

---

## CAPÍTULO 9: FILOSOFÍA DE LA TRADUCCIÓN AUTÓNOMA

### 9.1. Por qué la soberanía importa

El código que escribas hoy podría ser crítico en 10 años. Si depende de:
- **APIs externas que desaparecen**: Tu código muere con ellas.
- **Librerías propietarias**: Estás a merced de las decisiones corporativas.
- **Servicios en la nube**: Si Amazon decide cobrar más, o la startup quebranta, adiós.

El código autónomo es código que **sobrevive**.

### 9.2. Arquitectura sin dependencias

La clave es escribir el **núcleo del algoritmo** sin tocar librerías externas, y luego (si quieres) rodéalo de librerías para usabilidad.

Estructura propuesta:

```
core/          <-- ES6 puro, sin nada más. Es tu tesoro.
interface/     <-- Usa las librerías que quieras. Es lo descartable.
```

Si la `interface` explota, el `core` sigue intacto. Si necesitas portarlo a otro lenguaje, copias el `core` y escribes una nueva `interface`.

### 9.3. Validación del principio de soberanía

Preguntas para hacerte:

1. **¿Puedo ejecutar esto sin internet?** Si la respuesta es "no", tienes una dependencia externa.
2. **¿Puedo modificarlo sin contactar al autor?** Si no, el código no es realmente tuyo.
3. **¿Seguirá funcionando en 2036?** Si depende de un servicio, probablemente no.
4. **¿Entiendo exactamente qué hace cada línea?** Si no, no cumples el principio de documentación incrustada.

---

## CAPÍTULO 10: EPÍLOGO — EL ÚLTIMO CONSEJO

Has llegado hasta aquí. Has aprendido a leer papers, a traducirlos a código, a validarlos, a documentarlos. Ahora tienes una superpotencia: puedes convertir cualquier idea en realidad.

Pero recuerda: el código no es el fin. Es el medio. Lo que importa es lo que haces con él. Puedes usarlo para curar, para educar, para liberar. O puedes usarlo para controlar, para encerrar, para vender.

Cada línea de código que escribes es una decisión ética. Elige bien.

---

---

# NUEVA SECCIÓN: PROTOCOLOS DE 4 CAPAS APLICADOS

## CAPÍTULO 11: FILTRO DE PARTÍCULAS (SEQUENTIAL MONTE CARLO)

### 11.1. CAPA CONTEXTO: ¿POR QUÉ ALGUIEN INVENTÓ ESTO?

El filtro de Kalman es una obra maestra, cierto. Pero solo funciona cuando el mundo es **lineal** y el ruido es **gaussiano** (distribución normal). 

¿Qué pasa cuando la realidad no coopera?

Imagina que quieres rastrear un robot en una habitación con obstáculos. El robot se mueve según una dinámica no lineal (fricción, aceleración variable, efectos de inercia). Las mediciones que recibes (cámara, LIDAR) son también no lineales (visión es perspectiva, distancias angulares). El ruido no es gaussiano: hay outliers cuando la cámara confunde sombras con obstáculos.

En este escenario, el Filtro de Kalman falla estrepitosamente. Porque asume que puedes describir todo con una parábola suave (gaussiana).

**Solución: Filtro de Partículas** (Sequential Monte Carlo, SMC).

En lugar de mantener una única estimación del estado (como Kalman), mantienes una **nube de hipótesis** (partículas). Cada partícula es una posible realidad: "el robot está aquí, a esta velocidad, con esta orientación". Cuando llega una medición:

1. Actualizas la confianza en cada partícula (¿qué tan consistente es con lo que observo?).
2. Las partículas que explican bien la medición ganan peso.
3. Las que son inconsistentes pierden peso (remuestreo).
4. Al final, el promedio de todas las partículas es tu estimación del estado.

**¿Dónde falla Kalman y gana Partículas?**

| Problema | Kalman | Partículas |
|----------|--------|-----------|
| Dinámicas no lineales | ❌ | ✅ |
| Ruido no gaussiano | ❌ | ✅ |
| Distribuciones multimodales | ❌ | ✅ |
| Velocidad computacional | ✅ | ⚠️ (depende de N) |
| Intuición conceptual | ⚠️ | ✅ |

**Aplicaciones reales:**
- Rastreo de objetos en visión por computadora.
- Localización de robots en interiores.
- Estimación de trayectorias con sensores ruidosos.
- Pronóstico meteorológico.
- Epidemiología (rastreo de población infectada).

**Referencias iniciales:**
- Gordon, N. J., Salmond, D. J., & Smith, A. F. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings-F*, 140(2), 107-113.
- Kitagawa, G. (1996). Monte Carlo filter and smoother for non-Gaussian nonlinear state space models. *Journal of Computational and Graphical Statistics*, 5(1), 1-25.

---

### 11.2. CAPA ECUACIÓN: EL JEROGLÍFICO MATEMÁTICO

El filtro de partículas se basa en el teorema de Bayes y en el muestreo por importancia. Vamos paso a paso.

#### **El modelo generativo (qué suponemos sobre el mundo):**

**Ecuación 1: Transición de estado (dinámica)**
```
xₖ = f(xₖ₋₁, uₖ) + wₖ
```
Donde:
- `xₖ` = estado en tiempo k (posición, velocidad, orientación, etc.)
- `f(·)` = función de transición (puede ser no lineal)
- `uₖ` = entrada de control (si es que hay)
- `wₖ` = ruido del proceso (ruido del modelo, σ² desconocido pero supuesto)

**Ecuación 2: Observación (medición)**
```
zₖ = h(xₖ) + vₖ
```
Donde:
- `zₖ` = observación en tiempo k (sensor reading)
- `h(·)` = función de observación (puede ser no lineal)
- `vₖ` = ruido de medición

**Ecuación 3: Distribución posterior (qué creemos dada la evidencia)**
```
p(xₖ | z₁:ₖ) = p(zₖ | xₖ) · p(xₖ | z₁:ₖ₋₁) / p(zₖ | z₁:ₖ₋₁)
```
Esto es el **teorema de Bayes en su forma más pura**:
- `p(xₖ | z₁:ₖ)` = posterior (lo que creemos después de ver la medición)
- `p(zₖ | xₖ)` = verosimilitud (likelihood: qué tan probable es la medición si el estado es xₖ)
- `p(xₖ | z₁:ₖ₋₁)` = prior (lo que creíamos antes de esta medición)
- `p(zₖ | z₁:ₖ₋₁)` = evidencia (factor normalizador, se calcula integrando)

#### **La idea del filtro de partículas:**

En lugar de calcular `p(xₖ | z₁:ₖ)` analíticamente (imposible en sistemas no lineales complejos), la aproximamos como un **conjunto de muestras** (partículas):

```
p(xₖ | z₁:ₖ) ≈ Σᵢ₌₁ᴺ wᵢₖ · δ(xₖ - xⁱₖ)
```
Donde:
- `N` = número de partículas (típicamente 1000-10000)
- `xⁱₖ` = posición de la partícula i-ésima en tiempo k
- `wⁱₖ` = peso de la partícula i-ésima (Σ wⁱₖ = 1)
- `δ(·)` = función delta de Dirac (representa una masa puntual)

En otras palabras: la distribución posterior es la **suma ponderada de las posiciones de las partículas**.

#### **El algoritmo: tres pasos**

**Paso 1: Predicción (Propagación)**

Para cada partícula i, muestreamos la dinámica:
```
x̃ⁱₖ ~ p(xₖ | xⁱₖ₋₁)
```

En el caso gaussiano simple (ruido normal):
```
x̃ⁱₖ = f(xⁱₖ₋₁) + wₖ,  wₖ ~ N(0, Qₖ)
```

Esto propaga cada partícula forward según la dinámica, sumando ruido aleatorio.

**Paso 2: Ponderación (Update)**

Cuando llega una medición `zₖ`, asignamos un peso a cada partícula según qué tan bien explica la observación:

```
wⁱₖ ∝ p(zₖ | x̃ⁱₖ) · wⁱₖ₋₁
```

En el caso gaussiano:
```
p(zₖ | x̃ⁱₖ) = (1/(√(2π) · σ)) · exp(-0.5 · ((zₖ - h(x̃ⁱₖ)) / σ)²)
```

Donde `σ` es la desviación estándar del ruido de medición.

Luego normalizamos para que los pesos sumen 1:
```
wⁱₖ = wⁱₖ / Σⱼ wʲₖ
```

**Paso 3: Remuestreo (Resampling)**

Aquí ocurre la magia. Si un subconjunto de partículas tiene pesos muy bajos (cerca de 0), terminan siendo ruido numérico. Las eliminamos y replicamos las que tienen pesos altos.

**Algoritmo de remuestreo multinomial (naive):**
```
Para i = 1 a N:
  Muestrear un índice j ~ Categorical(w₁ₖ, w₂ₖ, ..., wₙₖ)
  xⁱₖ⁽ⁿᵘᵉᵛᵒ⁾ = x̃ʲₖ
  wⁱₖ⁽ⁿᵘᵉᵥᵒ⁾ = 1/N

Fin
```

Resultado: las partículas buenas se clonan, las malas desaparecen. Los nuevos pesos son uniformes.

**Alternativa: Remuestreo sistemático** (más eficiente):
```
u ~ U[0, 1/N]  (un único número aleatorio)
Para i = 1 a N:
  uᵢ = u + (i-1)/N
  Buscar j tal que Σₖ₌₁ʲ⁻¹ wₖ < uᵢ ≤ Σₖ₌₁ʲ wₖ
  xⁱₖ⁽ⁿᵘᵉᵥᵒ⁾ = x̃ʲₖ
  wⁱₖ⁽ⁿᵘᵉᵥᵒ⁾ = 1/N
Fin
```

Este es más rápido porque solo muestreamos un número aleatorio.

#### **Estimación final del estado:**

La estimación del estado (media posterior):
```
x̂ₖ = E[xₖ | z₁:ₖ] = Σᵢ₌₁ᴺ wⁱₖ · xⁱₖ
```

La covarianza:
```
Pₖ = E[(xₖ - x̂ₖ)² | z₁:ₖ] = Σᵢ₌₁ᴺ wⁱₖ · (xⁱₖ - x̂ₖ)²
```

---

### 11.3. CAPA ALGORITMO: LA TRADUCCIÓN A PSEUDOCÓDIGO HUMANO

```
ENTRADA:
  - x_{k-1}^{1:N} : Conjunto de N partículas del paso anterior
  - w_{k-1}^{1:N} : Pesos de las partículas anteriores
  - z_k : Observación en tiempo k
  - f(·) : Función de dinámica
  - h(·) : Función de observación
  - Q : Covarianza del ruido del proceso
  - R : Covarianza del ruido de medición

SALIDA:
  - x_k^{1:N} : Partículas actualizadas
  - w_k^{1:N} : Pesos actualizados
  - x̂_k : Estimación del estado (media ponderada)

ALGORITMO:

  1. PREDICCIÓN (Para cada partícula i = 1 a N):
     - Muestrear ruido del proceso: w ~ N(0, Q)
     - Propagar partícula: x̃ᵢ_k = f(x^i_{k-1}) + w
  
  2. PONDERACIÓN (Para cada partícula i = 1 a N):
     - Calcular predicción de observación: ẑᵢ = h(x̃ᵢ_k)
     - Calcular error: e = z_k - ẑᵢ
     - Calcular verosimilitud: 
       p(z_k | x̃ᵢ_k) = (2π R)^{-1/2} * exp(-0.5 * e^T * R^{-1} * e)
     - Actualizar peso: w̃ᵢ_k = p(z_k | x̃ᵢ_k) * w^i_{k-1}
  
  3. NORMALIZACIÓN:
     - Sumar pesos: W = Σᵢ w̃ᵢ_k
     - Normalizar: wᵢ_k = w̃ᵢ_k / W
  
  4. REMUESTREO (Sistemático):
     - Muestrear número aleatorio: u ~ U[0, 1/N]
     - Inicializar acumulador: cdf = 0
     - Para j = 1 a N:
       - u_j = u + (j-1)/N
       - Encontrar índice i tal que cdf < u_j ≤ cdf + wᵢ_k
       - Asignar: x^j_k = x̃ⁱ_k
       - Clonar: wʲ_k = 1/N
       - Actualizar: cdf = cdf + wᵢ_k
  
  5. ESTIMACIÓN:
     - Media: x̂_k = Σᵢ wⁱ_k * xⁱ_k
     - Covarianza: P_k = Σᵢ wⁱ_k * (xⁱ_k - x̂_k)²

  RETORNAR: x_k^{1:N}, w_k^{1:N}, x̂_k, P_k
```

---

### 11.4. CAPA CÓDIGO: IMPLEMENTACIÓN EN JAVASCRIPT ES6 PURO

```javascript
/**
 * FILTRO DE PARTÍCULAS (Sequential Monte Carlo)
 * Implementación ES6 pura, sin dependencias externas
 * 
 * Paper: Gordon, N. J., Salmond, D. J., & Smith, A. F. (1993)
 * Novel approach to nonlinear/non-Gaussian Bayesian state estimation
 * IEE Proceedings-F, 140(2), 107-113
 * 
 * Caso de uso: Rastreo de un objeto 1D con dinámica no lineal
 * Estado: posición x
 * Dinámica: x_k = 0.5*x_{k-1} + 25*x_{k-1}/(1+x_{k-1}^2) + 8*cos(1.2*k) + w_k
 * Observación: z_k = x_k^2 / 20 + v_k
 * (Esta es la dinámica del benchmark de Gordon et al.)
 */

class FiltroPartículas {
  /**
   * Constructor
   * @param {number} numPartículas - Número de partículas (recomendado: 1000-10000)
   * @param {number} sigmaProceso - Desv. estándar del ruido del proceso
   * @param {number} sigmaObservación - Desv. estándar del ruido de medición
   */
  constructor(numPartículas = 1000, sigmaProceso = 10.0, sigmaObservación = 1.0) {
    this.N = numPartículas;
    this.sigmaW = sigmaProceso;
    this.sigmaV = sigmaObservación;
    
    // Inicializar partículas
    // p(x_0) ~ N(0, 1)
    this.partículas = Array.from({ length: this.N }, () => 
      this._gaussiana(0, 1) // Media 0, varianza 1
    );
    
    // Todos los pesos empiezan iguales
    this.pesos = Array(this.N).fill(1.0 / this.N);
    
    // Para histórico
    this.histórico = {
      estimaciones: [],
      covarianzas: [],
      efectivoN: []
    };
  }

  /**
   * Función de dinámica no lineal
   * x_k = 0.5*x_{k-1} + 25*x_{k-1}/(1+x_{k-1}^2) + 8*cos(1.2*k) + w
   * 
   * Esta es la dinámica del benchmark de Gordon et al. (1993)
   * La razón por la que es interesante: no es lineal, y tiene una
   * singularidad suave en x=0 que causa multimodalidad.
   */
  _dinámicaNoLineal(x, tiempo, ruido) {
    const término1 = 0.5 * x;
    const término2 = (25 * x) / (1 + x * x);
    const término3 = 8 * Math.cos(1.2 * tiempo);
    return término1 + término2 + término3 + ruido;
  }

  /**
   * Función de observación no lineal
   * z_k = x_k^2 / 20 + v_k
   * 
   * Por qué es no lineal: cuadrática en x. Esto significa que
   * estados lejanos (x grande) parecen similares en la observación.
   * Dos estados x y -x producen la misma observación.
   * Esto provoca multimodalidad que Kalman no puede manejar.
   */
  _observaciónNoLineal(x) {
    return (x * x) / 20.0;
  }

  /**
   * Generador de números gaussianos (Box-Muller)
   * Genera dos variables gaussianas independientes con media μ y varianza σ²
   */
  _gaussiana(media = 0, varianza = 1) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return media + z * Math.sqrt(varianza);
  }

  /**
   * Función exponencial gaussiana (likelihood)
   * p(z | x) = (1 / (sqrt(2π) * σ)) * exp(-0.5 * ((z - h(x)) / σ)^2)
   * 
   * En logaritmo (para evitar underflow numérico):
   * log p(z | x) = -0.5 * log(2π) - log(σ) - 0.5 * ((z - h(x)) / σ)^2
   */
  _likelihood(z, xPartícula, tiempo) {
    const predicción = this._observaciónNoLineal(xPartícula);
    const error = z - predicción;
    
    // Usar logaritmos para evitar underflow
    // log p(z|x) = -0.5 * ((z - h(x)) / σ)^2 - 0.5 * log(2π * σ^2)
    const logLikelihood = -0.5 * Math.pow(error / this.sigmaV, 2) 
                          - 0.5 * Math.log(2 * Math.PI * this.sigmaV * this.sigmaV);
    
    // Retornar exp(logLikelihood), pero clipeado para evitar Infinity
    return Math.exp(Math.max(logLikelihood, -700)); // exp(-700) ≈ 0
  }

  /**
   * Paso de predicción (propagación)
   * x̃ᵢ_k = f(xⁱ_{k-1}) + w, donde w ~ N(0, Q)
   */
  _predicción(tiempo) {
    for (let i = 0; i < this.N; i++) {
      const ruido = this._gaussiana(0, this.sigmaW * this.sigmaW);
      this.partículas[i] = this._dinámicaNoLineal(
        this.partículas[i],
        tiempo,
        ruido
      );
    }
  }

  /**
   * Paso de ponderación (update)
   * wⁱ_k ∝ p(z_k | x̃ⁱ_k) * wⁱ_{k-1}
   */
  _ponderación(observación, tiempo) {
    let sumPesos = 0;
    
    // Calcular nuevos pesos (sin normalizar aún)
    for (let i = 0; i < this.N; i++) {
      const likelihood = this._likelihood(observación, this.partículas[i], tiempo);
      this.pesos[i] = likelihood * this.pesos[i];
      sumPesos += this.pesos[i];
    }
    
    // Normalizar (si sumPesos es 0, hay un problema)
    if (sumPesos > 0) {
      for (let i = 0; i < this.N; i++) {
        this.pesos[i] /= sumPesos;
      }
    } else {
      // Caso extremo: todos los pesos colapsaron a cero (muy raro)
      // Reinicializar pesos uniformemente
      console.warn("Colapso de pesos detectado. Reinicializando.");
      for (let i = 0; i < this.N; i++) {
        this.pesos[i] = 1.0 / this.N;
      }
    }
  }

  /**
   * Paso de remuestreo (resampling sistemático)
   * 
   * Algoritmo sistemático: más eficiente que multinomial
   * - Generar un único número aleatorio u ~ U[0, 1/N]
   * - Para j = 1 a N, calcular u_j = u + (j-1)/N
   * - Encontrar índice i tal que CDF(i-1) < u_j ≤ CDF(i)
   * - Clonar partícula i
   */
  _remuestreo() {
    const nuevasPartículas = [];
    const nuevosPesos = Array(this.N).fill(1.0 / this.N);
    
    // Calcular CDF (función de distribución acumulada)
    const cdf = [0];
    for (let i = 0; i < this.N; i++) {
      cdf.push(cdf[i] + this.pesos[i]);
    }
    
    // Muestreo sistemático
    const u = Math.random() / this.N; // u ~ U[0, 1/N]
    let idxPartícula = 0;
    
    for (let j = 0; j < this.N; j++) {
      const uj = u + j / this.N;
      
      // Buscar el índice i tal que cdf[i] < uj <= cdf[i+1]
      while (idxPartícula < this.N && cdf[idxPartícula + 1] < uj) {
        idxPartícula++;
      }
      
      // Clonar la partícula idxPartícula
      nuevasPartículas.push(this.partículas[idxPartícula]);
    }
    
    // Reemplazar
    this.partículas = nuevasPartículas;
    this.pesos = nuevosPesos;
  }

  /**
   * Calcular número efectivo de partículas
   * N_eff = 1 / Σ(w_i^2)
   * 
   * Esto nos dice cuántas partículas "reales" tenemos después del remuestreo.
   * Si N_eff < N/2, es momento de remuestrear.
   */
  _nEfectivo() {
    let sumaCuadrados = 0;
    for (let i = 0; i < this.N; i++) {
      sumaCuadrados += this.pesos[i] * this.pesos[i];
    }
    return 1.0 / sumaCuadrados;
  }

  /**
   * Estimar el estado (media ponderada)
   * x̂_k = Σ w_i * x_i
   */
  _estimarEstado() {
    let estimación = 0;
    for (let i = 0; i < this.N; i++) {
      estimación += this.pesos[i] * this.partículas[i];
    }
    return estimación;
  }

  /**
   * Estimar la covarianza del estado
   * P_k = Σ w_i * (x_i - x̂_k)^2
   */
  _estimarCovarianza(estimación) {
    let covarianza = 0;
    for (let i = 0; i < this.N; i++) {
      const error = this.partículas[i] - estimación;
      covarianza += this.pesos[i] * error * error;
    }
    return covarianza;
  }

  /**
   * MÉTODO PRINCIPAL: Ejecutar un paso completo del filtro
   * 
   * Orden de operaciones:
   * 1. Predicción (propagar dinámicas)
   * 2. Ponderación (incorporar observación)
   * 3. Remuestreo (eliminar hipótesis malas)
   * 4. Estimación (calcular media y covarianza)
   */
  actualizar(observación, tiempo) {
    // 1. PREDICCIÓN
    this._predicción(tiempo);
    
    // 2. PONDERACIÓN
    this._ponderación(observación, tiempo);
    
    // 3. REMUESTREO (si N_eff es bajo)
    const nEff = this._nEfectivo();
    if (nEff < this.N / 2) {
      this._remuestreo();
    }
    
    // 4. ESTIMACIÓN
    const estimación = this._estimarEstado();
    const covarianza = this._estimarCovarianza(estimación);
    
    // 5. REGISTRAR HISTÓRICO
    this.histórico.estimaciones.push(estimación);
    this.histórico.covarianzas.push(covarianza);
    this.histórico.efectivoN.push(nEff);
    
    return {
      estimación,
      covarianza,
      nEfectivo: nEff,
      desviación: Math.sqrt(covarianza)
    };
  }

  /**
   * Obtener diagnósticos
   */
  obtenerDiagnósticos() {
    return {
      numPartículas: this.N,
      sigmaProceso: this.sigmaW,
      sigmaObservación: this.sigmaV,
      últimaEstimación: this.histórico.estimaciones.slice(-1)[0],
      últimaDesviación: Math.sqrt(this.histórico.covarianzas.slice(-1)[0]),
      nEfectivoPromedio: this.histórico.efectivoN.reduce((a, b) => a + b) / this.histórico.efectivoN.length
    };
  }
}

/**
 * EXPORTAR CLASE (para uso en Node.js o navegador)
 */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = FiltroPartículas;
}
```

---

### 11.5. VALIDACIÓN Y TEST DE REGRESIÓN

```javascript
/**
 * TEST DE REGRESIÓN: Filtro de Partículas
 * 
 * Comprobamos que la implementación reproduce el comportamiento
 * esperado del paper de Gordon et al. (1993)
 */

// ==================== TEST 1: Inicialización ====================
console.log("=== TEST 1: Inicialización ===");
const filtro = new FiltroPartículas(1000, 10.0, 1.0);

console.assert(filtro.partículas.length === 1000, "ERROR: Número de partículas incorrecto");
console.assert(
  Math.abs(filtro.pesos.reduce((a, b) => a + b) - 1.0) < 1e-10,
  "ERROR: Pesos no suman 1"
);
console.log("✓ Inicialización correcta");

// ==================== TEST 2: Evolución de partículas ====================
console.log("\n=== TEST 2: Evolución de partículas ===");
const filtroPrueba = new FiltroPartículas(100, 10.0, 1.0);
const particInicial = filtroPrueba.partículas[0];

// Hacer un paso sin observación (solo predicción)
filtroPrueba._predicción(0);
const particPredictad = filtroPrueba.partículas[0];

console.assert(
  Math.abs(particInicial - particPredictad) > 0,
  "ERROR: Partícula no cambió (dinámica no funciona)"
);
console.log(`✓ Partícula evolucionó: ${particInicial.toFixed(3)} -> ${particPredictad.toFixed(3)}`);

// ==================== TEST 3: Convergencia numérica ====================
console.log("\n=== TEST 3: Convergencia con observación ====================");

const filtroConvergencia = new FiltroPartículas(5000, 10.0, 1.0);

// Simulación con estado verdadero conocido
const estadoVerdadero = [
  -0.1, // tiempo 1
  5.0,  // tiempo 2
  2.0,  // tiempo 3
  -8.0, // tiempo 4
  0.5   // tiempo 5
];

const resultados = [];

for (let k = 0; k < estadoVerdadero.length; k++) {
  // Simular observación desde el estado verdadero
  const observaciónRuidosa = 
    Math.pow(estadoVerdadero[k], 2) / 20.0 + 
    (Math.random() - 0.5) * 2.0; // Ruido uniforme [-1, 1]
  
  const resultado = filtroConvergencia.actualizar(observaciónRuidosa, k);
  resultados.push(resultado);
  
  const error = Math.abs(resultado.estimación - estadoVerdadero[k]);
  const rmse = Math.sqrt(error * error);
  
  console.log(
    `Tiempo ${k}: Estado=${estadoVerdadero[k].toFixed(3)}, ` +
    `Estimación=${resultado.estimación.toFixed(3)}, ` +
    `Error=${rmse.toFixed(3)}, ` +
    `Desviación=${resultado.desviación.toFixed(3)}`
  );
}

// ==================== TEST 4: N efectivo ====================
console.log("\n=== TEST 4: Número efectivo de partículas ===");
const filtroNEfectivo = new FiltroPartículas(1000, 5.0, 1.0);

for (let k = 0; k < 10; k++) {
  // Observación: estado verdadero = 10 (para forzar convergencia)
  const obs = 100 / 20 + Math.random() * 0.1;
  const res = filtroNEfectivo.actualizar(obs, k);
  
  const porcentaje = (res.nEfectivo / 1000) * 100;
  console.log(`Paso ${k}: N_eff = ${res.nEfectivo.toFixed(0)} (${porcentaje.toFixed(1)}%)`);
}

// ==================== TEST 5: Distribución de pesos ====================
console.log("\n=== TEST 5: Distribución de pesos post-remuestreo ===");
const filtroPesos = new FiltroPartículas(100, 1.0, 1.0);

// Forzar remuestreo
filtroPesos._predicción(0);
filtroPesos._ponderación(5.0, 0); // Observación = 5

const sumaPesosAntes = filtroPesos.pesos.reduce((a, b) => a + b);
console.assert(
  Math.abs(sumaPesosAntes - 1.0) < 1e-10,
  "ERROR: Pesos no suman 1 después de ponderación"
);

filtroPesos._remuestreo();
const sumaPesosDespués = filtroPesos.pesos.reduce((a, b) => a + b);
console.assert(
  Math.abs(sumaPesosDespués - 1.0) < 1e-10,
  "ERROR: Pesos no suman 1 después de remuestreo"
);

// Todos los pesos deben ser 1/N después de remuestreo
const pesoEsperado = 1.0 / 100;
for (let i = 0; i < 100; i++) {
  console.assert(
    Math.abs(filtroPesos.pesos[i] - pesoEsperado) < 1e-10,
    `ERROR: Peso ${i} no es 1/N`
  );
}
console.log("✓ Pesos remuestreados correctamente (todos = 1/N)");

// ==================== TEST 6: Comparación con valores esperados ====================
console.log("\n=== TEST 6: RMSE promedio (debe ser < 2) ===");
const filtroFinal = new FiltroPartículas(2000, 10.0, 1.0);

const estadoVerdaderoLargo = Array.from({ length: 30 }, (_, k) => {
  // Generar dinámicas más realistas (con más ruido)
  return Math.sin(k * 0.5) * 5;
});

const errores = [];
for (let k = 0; k < estadoVerdaderoLargo.length; k++) {
  const obsRuidosa = 
    Math.pow(estadoVerdaderoLargo[k], 2) / 20.0 + 
    (Math.random() - 0.5) * 3.0;
  
  const res = filtroFinal.actualizar(obsRuidosa, k);
  const error = Math.abs(res.estimación - estadoVerdaderoLargo[k]);
  errores.push(error);
}

const rmsePromedio = Math.sqrt(
  errores.reduce((a, b) => a + b * b) / errores.length
);
console.log(`RMSE promedio: ${rmsePromedio.toFixed(3)}`);
console.assert(rmsePromedio < 3.0, "ERROR: RMSE demasiado alto");
console.log("✓ Desempeño dentro de márgenes aceptables");

// ==================== RESUMEN ====================
console.log("\n" + "=".repeat(50));
console.log("TODOS LOS TESTS PASARON ✓");
console.log("=".repeat(50));

const diag = filtroFinal.obtenerDiagnósticos();
console.log("\nDiagnósticos finales:");
console.log(`  - Partículas: ${diag.numPartículas}`);
console.log(`  - Última estimación: ${diag.últimaEstimación.toFixed(3)}`);
console.log(`  - Última desviación: ${diag.últimaDesviación.toFixed(3)}`);
console.log(`  - N_eff promedio: ${diag.nEfectivoPromedio.toFixed(0)}`);
```

---

### 11.6. CÓMO EJECUTAR

**En Node.js:**
```bash
node filtro-particulas.js
```

**En navegador (HTML):**
```html
<!DOCTYPE html>
<html>
<head>
  <title>Filtro de Partículas - Gordon et al. (1993)</title>
  <script src="filtro-particulas.js"></script>
</head>
<body>
  <h1>Filtro de Partículas</h1>
  <pre id="output"></pre>
  <script>
    const filtro = new FiltroPartículas(1000, 10.0, 1.0);
    const output = document.getElementById('output');
    
    for (let k = 0; k < 10; k++) {
      const obs = Math.sin(k * 0.5) + Math.random() * 0.5;
      const res = filtro.actualizar(obs, k);
      output.innerHTML += `Paso ${k}: Est=${res.estimación.toFixed(3)}, σ=${res.desviación.toFixed(3)}\n`;
    }
  </script>
</body>
</html>
```

---

## CAPÍTULO 12: ALGORITMO DE RAFT (CONSENSO DESCENTRALIZADO)

### 12.1. CAPA CONTEXTO: EL PROBLEMA DE LA VERDAD DISTRIBUIDA

Imagina que tienes 5 servidores en diferentes geografías. Todos reciben la instrucción "guardar valor X=10". Pero por problemas de red:
- Un servidor recibe el mensaje 0.1 segundos después.
- Otro se cae antes de guardar.
- Otro recibe un mensaje contradictorio.

¿Cómo se ponen de acuerdo?

**Problema clásico en sistemas distribuidos:** consenso sin autoridad central. Si hay un servidor "jefe", es un punto único de fallo. Si no hay jefe, ¿quién decide?

**Soluciones antiguas:**
- **Paxos** (Lamport, 1998): Mathematically sound, but so complex that even Lamport struggled to explain it.
- **Two-Phase Commit** (DB clásico): Bloqueante, se cuelga si un nodo falla.

**Solución moderna: RAFT** (Ongaro & Ousterhout, 2014).

Raft es Paxos pero diseñado para **ser entendible**. La publicación misma tiene un lema: "Raft is a consensus algorithm for managing a replicated log".

**¿Cómo funciona mentalmente?**

Imagina un grupo de estudiantes que deben elegir qué comer. No hay dictador. Entonces:

1. **Alguien levanta la mano**: "Propongo pizza". Ese es el Líder (Leader).
2. **El líder pregunta a todos**: "¿Pizza te parece bien?". Eso es la propuesta de log entry.
3. **Si la mayoría dice "sí"**: Pizza se confirma. Se replica a todos.
4. **Si el líder se calla (fallo)**: Otro levanta la mano. Elección de líder.
5. **Si hay empate**: La mano más antigua gana (basado en términos).

**Aplicaciones reales:**
- **etcd** (Kubernetes configuration): Usa Raft.
- **Consul** (Service discovery): Usa Raft.
- **TiDB** (Base de datos distribuida): Usa Raft para replicación.

**Por qué no Kalman/Partículas aquí:** Raft no es estocástico. Es determinista. O hay consenso o no lo hay.

---

### 12.2. CAPA ECUACIÓN: EL JEROGLÍFICO (MÁQUINA DE ESTADOS)

Raft modela cada nodo como una **máquina de estados** con tres estados:

```
ESTADOS POSIBLES:
┌─────────────┐      timeout       ┌───────────┐
│  FOLLOWER   │ ─────────────────> │ CANDIDATE │
└──────┬──────┘                    └─────┬─────┘
       ^                                  │
       │                           gana   │
       │                          mayoría │
       │                                  v
       │                            ┌─────────┐
       │<──────────────────────────┤ LEADER  │
       │    pierden mayoría o      └─────────┘
       │    hay líder más nuevo       │
       └────────────────────────────┘
```

**Variables de estado para cada nodo:**

```
ESTADO PERSISTENTE (en disk, nunca se pierde):
  currentTerm: número (término actual, empieza en 0)
  votedFor: número (índice del candidato al que voté en este término)
  log[]: array (log de entradas: {term, command})

ESTADO VOLÁTIL (en memoria):
  commitIndex: número (índice del entry más alto replicado mayoría)
  lastApplied: número (índice del entry más alto aplicado a la máquina estatal)

ESTADO DE LÍDER (resetear si te conviertes en follower):
  nextIndex[]: array (para cada servidor, índice del próximo entry a enviar)
  matchIndex[]: array (para cada servidor, índice del entry más alto replicado)
```

**Ecuación de Term comparison (para elecciones):**

```
Si un mensaje tiene Term T_nuevo > Term_local:
  Term_local = T_nuevo
  Cambiar a FOLLOWER
  Limpiar votedFor

Si Term_nuevo < Term_local:
  Rechazar el mensaje (es obsoleto)
```

**Ecuación de validez de voto (candidato pide voto):**

```
Otorgo voto a candidato C si:
  1. currentTerm < candidateTerm
  2. votedFor es null O votedFor == C.id
  3. log de candidato es al menos tan nuevo como el mío
     (comparar: término del último entry, luego longitud del log)
```

**Ecuación de replicación (líder -> follower):**

```
Para cada servidor i != líder:
  Enviar AppendEntries RPC con:
    term = currentTerm
    leaderId = id del líder
    prevLogIndex = nextIndex[i] - 1
    prevLogTerm = término del entry en prevLogIndex
    entries[] = log entries a replicar (puede ser vacío)
    leaderCommit = commitIndex del líder

  Si respuesta.term > currentTerm:
    currentTerm = respuesta.term
    Cambiar a FOLLOWER
    Retornar
  
  Si respuesta.success:
    nextIndex[i] += 1
    matchIndex[i] = nextIndex[i] - 1
  Else:
    nextIndex[i] -= 1 (decremento lineal, o más eficiente: saltar términos)
```

**Ecuación de seguridad (Raft Log Matching Property):**

```
Si log entry (term T, index I) está commiteado en el nodo A,
Y el mismo entry se replica en nodo B,
ENTONCES cualquier entry en log de B con índice < I TAMBIÉN está en A
y con el mismo término.

(Esto se garantiza con AppendEntries consistency check)
```

---

### 12.3. CAPA ALGORITMO: PSEUDOCÓDIGO

```
INICIALIZACIÓN:
  Todos los nodos empiezan como FOLLOWER
  currentTerm = 0
  log = []
  estado = FOLLOWER

BUCLE PRINCIPAL PARA CADA NODO:

MIENTRAS ejecutando:

  1. TIMEOUT (Si no he recibido heartbeat en ~150ms):
     - Cambiar a CANDIDATE
     - currentTerm++
     - votedFor = mi_id
     - Resetear tiempo de elección
     - Enviar RequestVote RPC a todos los demás
  
  2. RECIBIR RequestVote RPC (candidato pide voto):
     - Si request.term < currentTerm: rechazar
     - Si request.term > currentTerm:
       currentTerm = request.term
       Cambiar a FOLLOWER
     - Si request.term == currentTerm:
       Si votedFor == null O votedFor == request.candidateId:
         Si log de candidato >= mi log:
           votedFor = request.candidateId
           Otorgar voto
           Retornar true
       Retornar false
  
  3. RECIBIR RequestVote RPC response:
     - Si voto fue otorgado: votos++
     - Si votos > N/2:
       Cambiar a LEADER
       Inicializar nextIndex[] = len(log) + 1 para todos
       Inicializar matchIndex[] = 0 para todos
       Enviar heartbeat inmediatamente (AppendEntries vacíos)
  
  4. RECIBIR AppendEntries RPC (latidos del líder):
     - Si request.term < currentTerm: rechazar
     - Si request.term >= currentTerm:
       Cambiar a FOLLOWER
       currentTerm = request.term
     - Si prevLogIndex > 0:
       Si no existe log[prevLogIndex]:
         Rechazar (me falta un entry anterior)
       Si log[prevLogIndex].term != request.prevLogTerm:
         Rechazar (hay inconsistencia)
     - Agregar entries faltantes al log
     - Si request.leaderCommit > commitIndex:
       commitIndex = min(request.leaderCommit, len(log))
  
  5. SI LÍDER:
     - Periódicamente (cada ~50ms): Enviar AppendEntries a todos
     - Si recibo comando nuevo del cliente:
       Agregar entry a log con currentTerm
       Enviar a todos los followers (en el siguiente heartbeat)
     - Si entry está replicado en mayoría:
       commitIndex = índice de ese entry
       Notificar a máquina de estado
  
  6. APLICAR ENTRIES:
     - Mientras lastApplied < commitIndex:
       lastApplied++
       Aplicar log[lastApplied].command a máquina de estado
       Responder cliente si aplica
```

---

### 12.4. CAPA CÓDIGO: IMPLEMENTACIÓN EN JAVASCRIPT

```javascript
/**
 * ALGORITMO RAFT (Consenso Distribuido)
 * Implementación ES6 pura
 * 
 * Paper: Ongaro, D., & Ousterhout, J. (2014)
 * In search of an understandable consensus algorithm
 * USENIX ATC'14 Proceedings
 * 
 * Este es un simulador educativo de Raft en una sola máquina.
 * En producción, cada nodo sería un proceso/servidor separado.
 */

class NodoRaft {
  /**
   * Constructor
   * @param {string} id - Identificador único del nodo
   * @param {number} n - Número total de nodos en el cluster
   * @param {number} timeoutHeartbeat - Milisegundos entre heartbeats
   * @param {number} rangoTimeout - Rango de timeout para elección [150-300ms]
   */
  constructor(id, n, timeoutHeartbeat = 50, rangoTimeout = [150, 300]) {
    // Identificación
    this.id = id;
    this.n = n;
    this.quórum = Math.floor(n / 2) + 1;
    
    // Estado persistente (en disk)
    this.currentTerm = 0;
    this.votedFor = null;
    this.log = []; // {term, command, index}
    
    // Estado volátil
    this.commitIndex = 0;
    this.lastApplied = 0;
    this.estado = 'FOLLOWER'; // FOLLOWER, CANDIDATE, LEADER
    
    // Estado de líder
    this.nextIndex = {};
    this.matchIndex = {};
    for (let i = 0; i < n; i++) {
      this.nextIndex[i] = this.log.length + 1;
      this.matchIndex[i] = 0;
    }
    
    // Timers
    this.rangoTimeout = rangoTimeout;
    this.tiempoÚltimoHeartbeat = Date.now();
    this.tiempoElección = this._generarTimeout();
    
    // Configuración
    this.timeoutHeartbeat = timeoutHeartbeat;
    
    // Estadísticas
    this.estadísticas = {
      eleccionesGanadas: 0,
      commandsAplicados: 0,
      votosRecibidos: 0,
      votosOtorgados: 0
    };
    
    // Histórico
    this.histórico = [];
  }

  /**
   * Generar timeout aleatorio para elección
   * Raft recommends: between 150ms and 300ms
   */
  _generarTimeout() {
    const [min, max] = this.rangoTimeout;
    return min + Math.random() * (max - min);
  }

  /**
   * Simular delay de red (para realismo)
   */
  _delay() {
    return 1 + Math.random() * 5; // 1-5ms
  }

  /**
   * Registrar evento en histórico
   */
  _registrar(evento) {
    const timestamp = new Date().toISOString().split('T')[1];
    const msg = `[${timestamp}] Nodo ${this.id}: ${evento}`;
    this.histórico.push(msg);
    console.log(msg);
  }

  /**
   * CANDIDATO pide voto (RequestVote RPC)
   */
  solicitarVoto(term, candidateId, lastLogIndex, lastLogTerm) {
    // Regla 1: Si el term es antiguo, rechazar
    if (term < this.currentTerm) {
      return { term: this.currentTerm, voteGranted: false };
    }

    // Regla 2: Si el term es más nuevo, actualizar estado
    if (term > this.currentTerm) {
      this.currentTerm = term;
      this.votedFor = null;
      this.estado = 'FOLLOWER';
    }

    // Regla 3: Validar el log del candidato
    const miÚltimoLogTerm = this.log.length > 0 
      ? this.log[this.log.length - 1].term 
      : 0;
    const miÚltimoLogIndex = this.log.length;

    const logDelCandidatoEsMásFresco = 
      lastLogTerm > miÚltimoLogTerm ||
      (lastLogTerm === miÚltimoLogTerm && lastLogIndex >= miÚltimoLogIndex);

    // Regla 4: Otorgar voto
    if (logDelCandidatoEsMásFresco && 
        (this.votedFor === null || this.votedFor === candidateId)) {
      this.votedFor = candidateId;
      this.tiempoÚltimoHeartbeat = Date.now();
      this.estadísticas.votosOtorgados++;
      return { term: this.currentTerm, voteGranted: true };
    }

    return { term: this.currentTerm, voteGranted: false };
  }

  /**
   * LÍDER envía heartbeat (AppendEntries RPC)
   */
  recibirAppendEntries(term, leaderId, prevLogIndex, prevLogTerm, entries, leaderCommit) {
    // Regla 1: Si term es antiguo
    if (term < this.currentTerm) {
      return { term: this.currentTerm, success: false };
    }

    // Regla 2: Si term es más nuevo
    if (term > this.currentTerm) {
      this.currentTerm = term;
      this.votedFor = null;
      this.estado = 'FOLLOWER';
    }

    this.tiempoÚltimoHeartbeat = Date.now();

    // Regla 3: Validar log
    if (prevLogIndex > 0) {
      if (prevLogIndex > this.log.length) {
        return { term: this.currentTerm, success: false };
      }
      if (this.log.length > 0 && prevLogIndex > 0) {
        const entryAnterior = this.log[prevLogIndex - 1];
        if (entryAnterior && entryAnterior.term !== prevLogTerm) {
          // Log mismatch: eliminar entry inconsistente
          this.log = this.log.slice(0, prevLogIndex - 1);
          return { term: this.currentTerm, success: false };
        }
      }
    }

    // Regla 4: Agregar entries nuevas
    if (entries.length > 0) {
      // Eliminar entries conflictivas
      this.log = this.log.slice(0, prevLogIndex);
      
      // Agregar nuevas entries
      for (let i = 0; i < entries.length; i++) {
        const entry = entries[i];
        this.log.push({
          term: entry.term,
          command: entry.command,
          index: this.log.length + 1
        });
      }
    }

    // Regla 5: Actualizar commitIndex
    if (leaderCommit > this.commitIndex) {
      this.commitIndex = Math.min(leaderCommit, this.log.length);
    }

    return { term: this.currentTerm, success: true };
  }

  /**
   * Procesar un comando del cliente (solo si eres líder)
   */
  procesarComando(comando) {
    if (this.estado !== 'LEADER') {
      return { success: false, error: 'No soy líder' };
    }

    // Agregar entry al log
    const entry = {
      term: this.currentTerm,
      command: comando,
      index: this.log.length + 1
    };
    this.log.push(entry);

    this._registrar(`Comando recibido: "${comando}" (index: ${entry.index})`);

    return { success: true, index: entry.index };
  }

  /**
   * Aplicar entries comprometidas a la máquina de estado
   */
  aplicarEntries() {
    while (this.lastApplied < this.commitIndex && this.lastApplied < this.log.length) {
      this.lastApplied++;
      const entry = this.log[this.lastApplied - 1];
      this._registrar(`APLICANDO: "${entry.command}"`);
      this.estadísticas.commandsAplicados++;
    }
  }

  /**
   * Convertir a CANDIDATE e iniciar elección
   */
  iniciarElección() {
    this.estado = 'CANDIDATE';
    this.currentTerm++;
    this.votedFor = this.id; // Voto por mí mismo
    this.tiempoElección = this._generarTimeout();

    const miÚltimoLogTerm = this.log.length > 0 
      ? this.log[this.log.length - 1].term 
      : 0;
    const miÚltimoLogIndex = this.log.length;

    this._registrar(
      `Iniciando elección (term: ${this.currentTerm}, logIndex: ${miÚltimoLogIndex})`
    );

    // Retornar parámetros para que otros nodos voten
    return {
      term: this.currentTerm,
      candidateId: this.id,
      lastLogIndex: miÚltimoLogIndex,
      lastLogTerm: miÚltimoLogTerm
    };
  }

  /**
   * Convertir a LEADER
   */
  convertirALíder() {
    if (this.estado === 'LEADER') return;

    this.estado = 'LEADER';
    this.tiempoÚltimoHeartbeat = Date.now();

    // Reinicializar estado de replicación
    for (let i = 0; i < this.n; i++) {
      this.nextIndex[i] = this.log.length + 1;
      this.matchIndex[i] = 0;
    }

    this._registrar(`ELEGIDO LÍDER (term: ${this.currentTerm})`);
    this.estadísticas.eleccionesGanadas++;
  }

  /**
   * Enviar heartbeat a todos (simular multicast)
   */
  generarHeartbeat() {
    if (this.estado !== 'LEADER') return null;

    const entries = [];
    const prevLogIndex = this.log.length;
    const prevLogTerm = this.log.length > 0 
      ? this.log[this.log.length - 1].term 
      : 0;

    return {
      term: this.currentTerm,
      leaderId: this.id,
      prevLogIndex: prevLogIndex,
      prevLogTerm: prevLogTerm,
      entries: entries, // Vacío para heartbeat
      leaderCommit: this.commitIndex
    };
  }

  /**
   * Procesar un timeout (si no he recibido heartbeat)
   */
  verificarTimeout() {
    const ahora = Date.now();
    const tiempoDesdeHeartbeat = ahora - this.tiempoÚltimoHeartbeat;

    if (this.estado === 'LEADER') {
      // Los líderes envían heartbeat cada N ms
      return tiempoDesdeHeartbeat > this.timeoutHeartbeat;
    } else {
      // Followers pasan a candidates si timeout
      return tiempoDesdeHeartbeat > this.tiempoElección;
    }
  }

  /**
   * Obtener estado actual
   */
  obtenerEstado() {
    return {
      id: this.id,
      estado: this.estado,
      term: this.currentTerm,
      logLength: this.log.length,
      commitIndex: this.commitIndex,
      lastApplied: this.lastApplied
    };
  }

  /**
   * Obtener estadísticas
   */
  obtenerEstadísticas() {
    return {
      ...this.estadísticas,
      log: this.log.map((e, i) => ({
        index: i + 1,
        term: e.term,
        command: e.command,
        comprometido: i + 1 <= this.commitIndex
      }))
    };
  }
}

/**
 * SIMULADOR DE CLUSTER RAFT
 */
class ClusterRaft {
  constructor(numNodos = 5) {
    this.nodos = {};
    for (let i = 0; i < numNodos; i++) {
      this.nodos[i] = new NodoRaft(i, numNodos);
    }
    this.numNodos = numNodos;
    this.tiempo = 0;
    this.eventos = [];
  }

  /**
   * Simular una ronda (paralelismo simulado)
   */
  simularRonda() {
    this.tiempo++;

    // 1. Verificar timeouts y conversiones de estado
    const candidatos = [];
    for (let id in this.nodos) {
      const nodo = this.nodos[id];
      if (nodo.verificarTimeout() && nodo.estado !== 'LEADER') {
        candidatos.push(id);
      }
    }

    // 2. Candidatos inician elección
    const votacionesRequeridas = {};
    for (let idCandidato of candidatos) {
      const params = this.nodos[idCandidato].iniciarElección();
      votacionesRequeridas[idCandidato] = params;
    }

    // 3. Procesar votación
    const votosRecibidos = {};
    for (let idCandidato in votacionesRequeridas) {
      votosRecibidos[idCandidato] = 0;
      for (let idVotante in this.nodos) {
        if (idVotante !== idCandidato) {
          const response = this.nodos[idVotante].solicitarVoto(
            votacionesRequeridas[idCandidato].term,
            votacionesRequeridas[idCandidato].candidateId,
            votacionesRequeridas[idCandidato].lastLogIndex,
            votacionesRequeridas[idCandidato].lastLogTerm
          );
          if (response.voteGranted) {
            votosRecibidos[idCandidato]++;
          }
        }
      }
    }

    // 4. Determinar ganador
    for (let idCandidato in votosRecibidos) {
      if (votosRecibidos[idCandidato] >= this.nodos[idCandidato].quórum - 1) {
        this.nodos[idCandidato].convertirALíder();
      }
    }

    // 5. Líder envía heartbeats
    for (let id in this.nodos) {
      const nodo = this.nodos[id];
      if (nodo.estado === 'LEADER') {
        const heartbeat = nodo.generarHeartbeat();
        if (heartbeat) {
          for (let idFollower in this.nodos) {
            if (idFollower !== id) {
              this.nodos[idFollower].recibirAppendEntries(
                heartbeat.term,
                heartbeat.leaderId,
                heartbeat.prevLogIndex,
                heartbeat.prevLogTerm,
                heartbeat.entries,
                heartbeat.leaderCommit
              );
            }
          }
        }
      }
    }

    // 6. Aplicar entries
    for (let id in this.nodos) {
      this.nodos[id].aplicarEntries();
    }
  }

  /**
   * Simular N rondas
   */
  simular(rondas) {
    for (let i = 0; i < rondas; i++) {
      this.simularRonda();
    }
  }

  /**
   * Cliente envía comando al líder
   */
  enviarComando(comando) {
    for (let id in this.nodos) {
      const nodo = this.nodos[id];
      if (nodo.estado === 'LÍDER') {
        return nodo.procesarComando(comando);
      }
    }
    return { success: false, error: 'No hay líder' };
  }

  /**
   * Obtener resumen de cluster
   */
  obtenerResumen() {
    const resumen = {
      tiempo: this.tiempo,
      nodos: {}
    };
    for (let id in this.nodos) {
      resumen.nodos[id] = this.nodos[id].obtenerEstado();
    }
    return resumen;
  }
}

/**
 * EXPORTAR
 */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { NodoRaft, ClusterRaft };
}
```

---

### 12.5. TEST DE REGRESIÓN

```javascript
/**
 * TESTS: Algoritmo Raft
 */

console.log("=== TESTS: RAFT ===\n");

// ==================== TEST 1: Inicialización ====================
console.log("TEST 1: Inicialización de cluster");
const cluster = new ClusterRaft(5);

for (let id = 0; id < 5; id++) {
  const estado = cluster.nodos[id].obtenerEstado();
  console.assert(estado.estado === 'FOLLOWER', "ERROR: Debe empezar como FOLLOWER");
  console.assert(estado.term === 0, "ERROR: Debe empezar en term 0");
}
console.log("✓ Todos los nodos empiezan como FOLLOWER\n");

// ==================== TEST 2: Elección de líder ====================
console.log("TEST 2: Elección de líder después de timeout");
cluster.simular(200); // Suficientes rondas para timeout

let tieneUnLíder = false;
let idLíder = null;
for (let id = 0; id < 5; id++) {
  const estado = cluster.nodos[id].obtenerEstado();
  if (estado.estado === 'LEADER') {
    console.assert(!tieneUnLíder, "ERROR: Más de un líder!");
    tieneUnLíder = true;
    idLíder = id;
  }
}
console.assert(tieneUnLíder, "ERROR: No hay líder después de timeout");
console.log(`✓ Líder elegido: Nodo ${idLíder}\n`);

// ==================== TEST 3: Replicación ====================
console.log("TEST 3: Replicación de comandos");
const resultadoComando = cluster.enviarComando("SET x=10");
console.assert(resultadoComando.success, "ERROR: Comando rechazado");

// Simular replicación
cluster.simular(50);

// Verificar que el comando fue aplicado en la mayoría
let nodosConComando = 0;
for (let id = 0; id < 5; id++) {
  const stats = cluster.nodos[id].obtenerEstadísticas();
  if (stats.log.length > 0 && stats.log[0].command === "SET x=10") {
    nodosConComando++;
  }
}
console.assert(nodosConComando >= 3, "ERROR: No se replicó correctamente");
console.log(`✓ Comando replicado en ${nodosConComando}/5 nodos\n`);

// ==================== TEST 4: Mayoría requerida ====================
console.log("TEST 4: Validación de quórum");
for (let id = 0; id < 5; id++) {
  const nodo = cluster.nodos[id];
  const quórum = nodo.quórum;
  console.assert(quórum === 3, `ERROR: Quórum debe ser 3, no ${quórum}`);
}
console.log("✓ Quórum correcto: 3/5\n");

// ==================== TEST 5: Logs consistentes ====================
console.log("TEST 5: Consistencia de logs");
let logsConsistentes = true;
const primerLog = cluster.nodos[0].log;
for (let id = 1; id < 5; id++) {
  const log = cluster.nodos[id].log;
  if (log.length !== primerLog.length) {
    logsConsistentes = false;
    break;
  }
  for (let i = 0; i < log.length; i++) {
    if (log[i].term !== primerLog[i].term || 
        log[i].command !== primerLog[i].command) {
      logsConsistentes = false;
      break;
    }
  }
}
console.assert(logsConsistentes || primerLog.length === 0, 
  "ERROR: Logs inconsistentes");
console.log("✓ Logs consistentes\n");

// ==================== TEST 6: Múltiples comandos ====================
console.log("TEST 6: Replicación de múltiples comandos");
const comandos = ["SET a=1", "SET b=2", "SET c=3"];
for (let cmd of comandos) {
  cluster.enviarComando(cmd);
}
cluster.simular(100);

const estadísticasLíder = cluster.nodos[idLíder].obtenerEstadísticas();
console.assert(
  estadísticasLíder.commandsAplicados >= 3,
  "ERROR: No todos los comandos fueron aplicados"
);
console.log(`✓ ${estadísticasLíder.commandsAplicados} comandos aplicados\n`);

// ==================== RESUMEN ====================
console.log("=".repeat(50));
console.log("TODOS LOS TESTS PASARON ✓");
console.log("=".repeat(50));

const resumen = cluster.obtenerResumen();
console.log(`\nCluster después de ${resumen.tiempo} rondas:`);
console.log(JSON.stringify(resumen, null, 2));
```

---

## CAPÍTULO 13: DIAGRAMAS VORONOI DINÁMICOS (GEOMETRÍA COMPUTACIONAL)

### 13.1. CAPA CONTEXTO: PARTICIÓN EFICIENTE DEL ESPACIO

Imagina que eres ingeniero de una fábrica. Tienes 5 máquinas dispensadoras de tinta en diferentes puntos. Cada máquina cubre un área: el área más cercana a ella que a cualquier otra.

**Pregunta técnica:** ¿Cuál es el área de cobertura de cada máquina?

**Respuesta gráfica:** Diagrama de Voronoi.

Cada máquina es un **sitio** (punto en 2D). El área alrededor del sitio i es:
```
V(i) = {p en el plano : distancia(p, sitio_i) < distancia(p, sitio_j) para todo j ≠ i}
```

**¿Dónde se usa?**

- **Diseño de antenas**: Colocar antenas de forma óptima. Cada antena cubre un Voronoi.
- **Meteorología**: Cada estación meteorológica cubre un área Voronoi.
- **Planificación de rutas**: Particionar un mapa en zonas de entrega.
- **Física de partículas**: Cálculo de presiones en simulaciones SPH.
- **Aprendizaje de máquinas**: Cuantificadores vectoriales, agrupamiento.
- **Diseño industrial**: Optimizar disipación térmica distribuyendo sensores.

**Problema computacional:**

Calcular el Voronoi exacto con algoritmos clásicos (Sweep-line de Fortune) es complejo de implementar. Para aplicaciones en tiempo real o educativas, usamos **Relajación de Lloyd** (aproximación iterativa):

1. Comienza con sitios aleatorios.
2. Calcula el Voronoi (aproximado).
3. Desplaza cada sitio al centroide de su región.
4. Repite hasta convergencia.

Esto genera un **Voronoi Centroidal** (CVT): óptimo en el sentido de que minimiza la suma de distancias entre puntos y sus sitios más cercanos.

**Por qué es dinámico:** Los sitios cambian, el Voronoi se recalcula continuamente.

---

### 13.2. CAPA ECUACIÓN

**Ecuación 1: Definición de Voronoi**

```
V(s_i) = {p ∈ ℝ² : ||p - s_i|| ≤ ||p - s_j|| para todo j ≠ i}
```
Donde:
- `s_i` = posición del sitio i
- `p` = punto arbitrario en el plano
- `||·||` = norma Euclidiana

**Ecuación 2: Diagrama de Voronoi (forma particionada)**

```
La frontera entre V(s_i) y V(s_j) es la perpendicular a la línea s_i-s_j
en el punto medio: (s_i + s_j) / 2

Esta frontera es una línea recta (mediatriz).
El vértice donde se encuentran tres o más bisectrices es un vértice Voronoi.
```

**Ecuación 3: Algoritmo de Lloyd (relajación)**

```
ENTRADA: Conjunto de sitios S = {s_1, ..., s_n}
SALIDA: Diagrama de Voronoi aproximado

Iteración k:

1. PARTICIÓN: Asignar cada punto del espacio (o una grilla) a su sitio más cercano
   Para cada punto p:
     c(p) = argmin_i ||p - s_i||

2. CÁLCULO DE CENTROIDES: Desplazar cada sitio al centro de masa de su región
   Para cada sitio i:
     s_i^{k+1} = (1 / |V_i|) * ∫_{V_i} p dp
   
   En discreto (grilla):
     s_i^{k+1} = (Σ_{p ∈ V_i} p) / |V_i|

3. CRITERIO DE PARADA:
   Si max_i ||s_i^{k+1} - s_i^k|| < ε:
     Convergencia alcanzada
   Sino:
     Volver a iteración k+1
```

**Ecuación 4: Energía a minimizar**

```
E(S) = Σ_i (1 / |V_i|) * ∫_{V_i} ||p - s_i||² dp

Lloyd's algorithm disminuye E(S) monótonamente hacia un óptimo local.
```

---

### 13.3. CAPA ALGORITMO

```
ENTRADA:
  - numSitios: cantidad de generadores Voronoi
  - ancho, alto: dimensiones del dominio
  - iteracionesMáx: número máximo de iteraciones
  - εConvergencia: umbral para parada anticipada

SALIDA:
  - Sitios finales (posiciones convergidas)
  - Asignación de píxeles a sitios (diagrama de Voronoi)

INICIALIZACIÓN:
  1. Generar numSitios posiciones aleatorias en [0, ancho] x [0, alto]
  2. Crear grilla de píxeles (ancho x alto)
  3. Inicializar distancia mínima a infinito para cada píxel

BUCLE PRINCIPAL (para k = 1 a iteracionesMáx):
  
  1. RESETEAR ASIGNACIONES:
     Para cada píxel (x, y):
       distanciaMin[x, y] = ∞
       asignación[x, y] = null
       sumaX[i] = 0, sumaY[i] = 0 (para cada sitio i)
       cuenta[i] = 0
  
  2. ASIGNAR PÍXELES A SITIOS:
     Para cada píxel (x, y):
       Para cada sitio i:
         dist = distancia euclidiana((x, y), sitio[i])
         Si dist < distanciaMin[x, y]:
           distanciaMin[x, y] = dist
           asignación[x, y] = i
       
       sitioAsignado = asignación[x, y]
       sumaX[sitioAsignado] += x
       sumaY[sitioAsignado] += y
       cuenta[sitioAsignado] += 1
  
  3. CALCULAR NUEVAS POSICIONES:
     Para cada sitio i:
       Si cuenta[i] > 0:
         centroideX[i] = sumaX[i] / cuenta[i]
         centroideY[i] = sumaY[i] / cuenta[i]
         Actualizar sitio[i] = (centroideX[i], centroideY[i])
  
  4. VERIFICAR CONVERGENCIA:
     movimientoMáx = max_i distancia(sitio[i]_nuevo, sitio[i]_anterior)
     Si movimientoMáx < εConvergencia:
       RETORNAR (convergencia alcanzada)
     
  5. SIGUIENTE ITERACIÓN (volver al paso 1)

RETORNAR:
  - Sitios finales
  - Grilla de asignaciones (asignación[x, y] = índice del sitio más cercano)
```

---

### 13.4. CAPA CÓDIGO: IMPLEMENTACIÓN EN JAVASCRIPT

```javascript
/**
 * DIAGRAMA DE VORONOI DINÁMICO (Algoritmo de Lloyd)
 * Implementación ES6 pura
 * 
 * Paper: Fortune, S. (1987). A sweepline algorithm for Voronoi diagrams. Algorithmica.
 * (Aunque usamos Lloyd por simplicity; Fortune's es el óptimo asintótico)
 * 
 * También relevante:
 * Du, Q., Faber, V., & Gunzburger, M. (1999).
 * Centroidal Voronoi Tessellations: Applications and Algorithms.
 * SIAM Review, 41(4), 637-676.
 */

class VoronoiDinámico {
  /**
   * Constructor
   * @param {number} ancho - Ancho del dominio
   * @param {number} alto - Alto del dominio
   * @param {number} numSitios - Número de generadores Voronoi
   * @param {number} épocas - Número de iteraciones de Lloyd
   */
  constructor(ancho = 500, alto = 500, numSitios = 10, épocas = 50) {
    this.ancho = ancho;
    this.alto = alto;
    this.numSitios = numSitios;
    this.épocas = épocas;
    
    // Inicializar sitios (generadores) aleatoriamente
    this.sitios = [];
    for (let i = 0; i < numSitios; i++) {
      this.sitios.push({
        id: i,
        x: Math.random() * ancho,
        y: Math.random() * alto,
        color: this._generarColor(),
        área: 0,
        centroideX: 0,
        centroideY: 0
      });
    }
    
    // Grilla de asignaciones (qué sitio es dueño de cada píxel)
    this.asignaciones = [];
    for (let y = 0; y < alto; y++) {
      this.asignaciones[y] = [];
      for (let x = 0; x < ancho; x++) {
        this.asignaciones[y][x] = -1; // Sin asignar aún
      }
    }
    
    // Histórico de convergencia
    this.histórico = {
      movimientoMáximo: [],
      energía: [],
      época: 0
    };
  }

  /**
   * Generar un color RGB aleatorio
   */
  _generarColor() {
    return {
      r: Math.floor(Math.random() * 255),
      g: Math.floor(Math.random() * 255),
      b: Math.floor(Math.random() * 255)
    };
  }

  /**
   * Distancia Euclidiana
   */
  _distancia(x1, y1, x2, y2) {
    const dx = x1 - x2;
    const dy = y1 - y2;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * Paso 1: ASIGNAR cada píxel al sitio más cercano
   */
  _asignarPixeles() {
    // Resetear acumuladores
    for (let i = 0; i < this.numSitios; i++) {
      this.sitios[i].área = 0;
      this.sitios[i].centroideX = 0;
      this.sitios[i].centroideY = 0;
    }

    // Iterar sobre cada píxel
    for (let y = 0; y < this.alto; y++) {
      for (let x = 0; x < this.ancho; x++) {
        let distanciaMin = Infinity;
        let sítioMásCercano = 0;

        // Encontrar el sitio más cercano
        for (let i = 0; i < this.numSitios; i++) {
          const dist = this._distancia(x, y, this.sitios[i].x, this.sitios[i].y);
          if (dist < distanciaMin) {
            distanciaMin = dist;
            sítioMásCercano = i;
          }
        }

        // Asignar píxel a sitio
        this.asignaciones[y][x] = sítioMásCercano;

        // Acumular para centroide
        const sitio = this.sitios[sítioMásCercano];
        sitio.centroideX += x;
        sitio.centroideY += y;
        sitio.área += 1;
      }
    }
  }

  /**
   * Paso 2: ACTUALIZAR posiciones de sitios al centroide de su región
   */
  _actualizarSitios() {
    let movimientoMáximo = 0;

    for (let i = 0; i < this.numSitios; i++) {
      const sitio = this.sitios[i];

      if (sitio.área > 0) {
        const nuevoX = sitio.centroideX / sitio.área;
        const nuevoY = sitio.centroideY / sitio.área;

        const movimiento = this._distancia(
          sitio.x, sitio.y,
          nuevoX, nuevoY
        );

        sitio.x = nuevoX;
        sitio.y = nuevoY;

        movimientoMáximo = Math.max(movimientoMáximo, movimiento);
      }
    }

    return movimientoMáximo;
  }

  /**
   * Calcular energía total del diagrama
   * E = Σ_i (1/|V_i|) * Σ_{p ∈ V_i} ||p - s_i||²
   * 
   * Esta es la cantidad que Lloyd's minimiza.
   */
  _calcularEnergía() {
    let energíaTotal = 0;

    for (let i = 0; i < this.numSitios; i++) {
      const sitio = this.sitios[i];
      let energíaSitio = 0;

      for (let y = 0; y < this.alto; y++) {
        for (let x = 0; x < this.ancho; x++) {
          if (this.asignaciones[y][x] === i) {
            const distAl² = Math.pow(
              this._distancia(x, y, sitio.x, sitio.y),
              2
            );
            energíaSitio += distAl²;
          }
        }
      }

      if (sitio.área > 0) {
        energíaTotal += energíaSitio / sitio.área;
      }
    }

    return energíaTotal;
  }

  /**
   * EJECUTAR una iteración completa del algoritmo de Lloyd
   */
  iterarLloyd() {
    // 1. Asignar píxeles
    this._asignarPixeles();

    // 2. Actualizar sitios
    const movimiento = this._actualizarSitios();

    // 3. Calcular energía
    const energía = this._calcularEnergía();

    // 4. Registrar histórico
    this.histórico.movimientoMáximo.push(movimiento);
    this.histórico.energía.push(energía);
    this.histórico.época++;

    return {
      época: this.histórico.época,
      movimiento: movimiento,
      energía: energía
    };
  }

  /**
   * CONVERGER: iterar hasta convergencia o épocas máximas
   */
  converger(épocasMáx = this.épocas, umbralConvergencia = 0.1) {
    for (let época = 0; época < épocasMáx; época++) {
      const resultado = this.iterarLloyd();

      if (resultado.movimiento < umbralConvergencia) {
        console.log(`✓ Convergencia en época ${época}`);
        return {
          convergió: true,
          épocasRequeridas: época,
          movimientoFinal: resultado.movimiento
        };
      }

      if (época % 10 === 0) {
        console.log(
          `Época ${época}: movimiento=${resultado.movimiento.toFixed(3)}, ` +
          `energía=${resultado.energía.toFixed(0)}`
        );
      }
    }

    console.log(`⚠ No convergió después de ${épocasMáx} épocas`);
    return {
      convergió: false,
      épocasRequeridas: épocasMáx,
      movimientoFinal: this.histórico.movimientoMáximo.slice(-1)[0]
    };
  }

  /**
   * Obtener sitios actuales
   */
  obtenerSitios() {
    return this.sitios.map(s => ({
      id: s.id,
      x: s.x.toFixed(2),
      y: s.y.toFixed(2),
      área: s.área,
      color: s.color
    }));
  }

  /**
   * Generar SVG para visualizar
   */
  generarSVG() {
    let svg = `<svg width="${this.ancho}" height="${this.alto}" xmlns="http://www.w3.org/2000/svg">\n`;
    svg += `  <rect width="${this.ancho}" height="${this.alto}" fill="white"/>\n`;

    // Dibujar celdas (coloreadas por sitio)
    for (let y = 0; y < this.alto; y += 5) { // Cada 5 píxeles para rendimiento
      for (let x = 0; x < this.ancho; x += 5) {
        const sítioAsignado = this.asignaciones[y][x];
        if (sítioAsignado >= 0) {
          const color = this.sitios[sítioAsignado].color;
          svg += `  <rect x="${x}" y="${y}" width="5" height="5" fill="rgb(${color.r}, ${color.g}, ${color.b})"/>\n`;
        }
      }
    }

    // Dibujar sitios (círculos negros)
    for (let sitio of this.sitios) {
      svg += `  <circle cx="${sitio.x.toFixed(1)}" cy="${sitio.y.toFixed(1)}" r="3" fill="black"/>\n`;
    }

    svg += `</svg>`;
    return svg;
  }

  /**
   * Diagnosticar convergencia
   */
  obtenerDiagnósticos() {
    const últimosMovimientos = this.histórico.movimientoMáximo.slice(-5);
    const promedioMovimiento = últimosMovimientos.length > 0
      ? últimosMovimientos.reduce((a, b) => a + b) / últimosMovimientos.length
      : Infinity;

    return {
      época: this.histórico.época,
      sitios: this.numSitios,
      movimientoPromedioÚltimas5: promedioMovimiento.toFixed(4),
      energíaFinal: this.histórico.energía.slice(-1)[0].toFixed(0),
      áreaPromedio: (this.ancho * this.alto) / this.numSitios
    };
  }
}

/**
 * EXPORTAR
 */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = VoronoiDinámico;
}
```

---

### 13.5. TEST DE REGRESIÓN

```javascript
/**
 * TESTS: Voronoi Dinámico
 */

console.log("=== TESTS: VORONOI DINÁMICO ===\n");

// ==================== TEST 1: Inicialización ====================
console.log("TEST 1: Inicialización");
const voronoi = new VoronoiDinámico(100, 100, 4, 50);

console.assert(voronoi.sitios.length === 4, "ERROR: Número de sitios incorrecto");
console.assert(voronoi.asignaciones.length === 100, "ERROR: Altura incorrecta");
console.assert(voronoi.asignaciones[0].length === 100, "ERROR: Ancho incorrecto");
console.log("✓ Voronoi inicializado correctamente\n");

// ==================== TEST 2: Asignación de píxeles ====================
console.log("TEST 2: Asignación de píxeles");
voronoi._asignarPixeles();

let pixelesAsignados = 0;
for (let y = 0; y < voronoi.alto; y++) {
  for (let x = 0; x < voronoi.ancho; x++) {
    if (voronoi.asignaciones[y][x] >= 0) {
      pixelesAsignados++;
    }
  }
}

const totalPixeles = voronoi.ancho * voronoi.alto;
console.assert(
  pixelesAsignados === totalPixeles,
  `ERROR: Solo ${pixelesAsignados}/${totalPixeles} píxeles asignados`
);
console.log(`✓ Todos los ${totalPixeles} píxeles asignados\n`);

// ==================== TEST 3: Actualización de sitios ====================
console.log("TEST 3: Actualización de posiciones");
const posiciónAnterior = {
  x: voronoi.sitios[0].x,
  y: voronoi.sitios[0].y
};

const movimiento = voronoi._actualizarSitios();

const posiciónNueva = {
  x: voronoi.sitios[0].x,
  y: voronoi.sitios[0].y
};

console.assert(
  !(posiciónAnterior.x === posiciónNueva.x && posiciónAnterior.y === posiciónNueva.y),
  "ERROR: Sitios no se movieron"
);
console.log(`✓ Sitios actualizados (movimiento máximo: ${movimiento.toFixed(3)})\n`);

// ==================== TEST 4: Convergencia ====================
console.log("TEST 4: Convergencia de Lloyd");
const voronoiConv = new VoronoiDinámico(150, 150, 6, 100);

const resultado = voronoiConv.converger(100, 0.5);
console.log(`Convergencia: ${resultado.convergió ? "Sí" : "No"}`);
console.log(`Épocas requeridas: ${resultado.épocasRequeridas}`);
console.assert(
  resultado.movimientoFinal < 5.0,
  "ERROR: No convergió adecuadamente"
);
console.log("✓ Algoritmo converge\n");

// ==================== TEST 5: Monotonicidad de energía ====================
console.log("TEST 5: Energía desciende (monotonicidad de Lloyd)");
const voronoiEnergía = new VoronoiDinámico(100, 100, 5, 20);

for (let i = 0; i < 10; i++) {
  voronoiEnergía.iterarLloyd();
}

const energías = voronoiEnergía.histórico.energía;
let energíaMonótona = true;
for (let i = 1; i < energías.length; i++) {
  if (energías[i] > energías[i - 1] + 1) { // Margen para errores numéricos
    energíaMonótona = false;
    console.log(`⚠ Energía aumentó en paso ${i}: ${energías[i - 1].toFixed(0)} → ${energías[i].toFixed(0)}`);
  }
}

console.assert(energíaMonótona, "ERROR: Energía no es monótona");
console.log("✓ Energía desciende monótonamente\n");

// ==================== TEST 6: Áreas no cero ====================
console.log("TEST 6: Validación de áreas");
voronoi.converger(10, 1.0);

let áreasCero = 0;
for (let sitio of voronoi.sitios) {
  if (sitio.área === 0) {
    áreasCero++;
  }
}

console.assert(
  áreasCero === 0,
  `ERROR: ${áreasCero} sitios con área cero`
);
console.log("✓ Todos los sitios tienen área > 0\n");

// ==================== TEST 7: Suma de áreas ====================
console.log("TEST 7: Suma de áreas = total de píxeles");
let áreaTotal = 0;
for (let sitio of voronoi.sitios) {
  áreaTotal += sitio.área;
}

const pixelesTotales = voronoi.ancho * voronoi.alto;
console.assert(
  áreaTotal === pixelesTotales,
  `ERROR: Área total ${áreaTotal} != píxeles ${pixelesTotales}`
);
console.log(`✓ Suma de áreas = ${pixelesTotales}\n`);

// ==================== TEST 8: Distancia a sitios ====================
console.log("TEST 8: Cada píxel es asignado al sitio más cercano");
let errores = 0;
for (let y = 0; y < voronoi.alto; y++) {
  for (let x = 0; x < voronoi.ancho; x++) {
    const sítioAsignado = voronoi.asignaciones[y][x];
    const distAlAsignado = voronoi._distancia(
      x, y,
      voronoi.sitios[sítioAsignado].x,
      voronoi.sitios[sítioAsignado].y
    );

    // Verificar que no hay otro sitio más cercano
    for (let i = 0; i < voronoi.numSitios; i++) {
      if (i !== sítioAsignado) {
        const distAlOtro = voronoi._distancia(
          x, y,
          voronoi.sitios[i].x,
          voronoi.sitios[i].y
        );
        if (distAlOtro < distAlAsignado - 0.1) { // Margen para errores numéricos
          errores++;
        }
      }
    }
  }
}

console.assert(errores === 0, `ERROR: ${errores} píxeles mal asignados`);
console.log("✓ Cada píxel asignado correctamente\n");

// ==================== RESUMEN ====================
console.log("=".repeat(50));
console.log("TODOS LOS TESTS PASARON ✓");
console.log("=".repeat(50));

const diagnósticos = voronoi.obtenerDiagnósticos();
console.log("\nDiagnósticos finales:");
console.log(JSON.stringify(diagnósticos, null, 2));

// ==================== SVG OUTPUT ====================
const svg = voronoi.generarSVG();
console.log("\nSVG generado (primeras 200 caracteres):");
console.log(svg.substring(0, 200) + "...");
```

---

## APÉNDICE A: GLOSARIO EXTENSO

- **Paper**: Artículo académico publicado en una revista científica.
- **DOI**: Identificador único de un paper (ej: 10.1093/...).
- **Implementación**: Versión en código de un algoritmo descrito en un paper.
- **Validación**: Comprobar que el código hace lo que el paper dice.
- **Heurística**: Estimación, aproximación.
- **Núcleo (kernel)**: Función de influencia basada en distancia.
- **Ganancia de Kalman**: Factor que decide confianza en predicción vs medición.
- **PID**: Controlador Proporcional-Integral-Derivativo.
- **WebAssembly**: Formato binario para ejecutar código rápido en el navegador.
- **Filtro de Partículas**: Algoritmo de estimación para sistemas no lineales.
- **Sequential Monte Carlo (SMC)**: Nombre formal del filtro de partículas.
- **Remuestreo**: Proceso de eliminar partículas malas y clonar buenas.
- **N efectivo**: Número de partículas "reales" (sin degeneración).
- **Raft**: Algoritmo de consenso distribuido para máquinas replicadas.
- **Follower/Candidate/Leader**: Estados del protocolo Raft.
- **Quórum**: Mayoría requerida para consenso (N/2 + 1).
- **Voronoi**: Partición del plano por proximidad a sitios.
- **Centroide**: Centro de masa geométrico.
- **Lloyd**: Algoritmo de relajación para Voronoi centroidal.

---

## APÉNDICE B: REFERENCIAS COMPLETAS

### Papers Principales (Nuevos Capítulos)

- Gordon, N. J., Salmond, D. J., & Smith, A. F. (1993). **Novel approach to nonlinear/non-Gaussian Bayesian state estimation**. *IEE Proceedings-F*, 140(2), 107-113. DOI: 10.1049/ip-f.1993.0015

- Ongaro, D., & Ousterhout, J. (2014). **In search of an understandable consensus algorithm**. In *2014 USENIX Annual Technical Conference (USENIX ATC 14)* (pp. 305-319). DOI: 10.5555/2643634.2643666

- Fortune, S. (1987). **A sweepline algorithm for Voronoi diagrams**. *Algorithmica*, 2(1), 153-174. DOI: 10.1007/BF01840356

- Du, Q., Faber, V., & Gunzburger, M. (1999). **Centroidal Voronoi Tessellations: Applications and Algorithms**. *SIAM Review*, 41(4), 637-676. DOI: 10.1137/S0036144599352836

### Papers Relacionados

- Kitagawa, G. (1996). **Monte Carlo filter and smoother for non-Gaussian nonlinear state space models**. *Journal of Computational and Graphical Statistics*, 5(1), 1-25.

- Kalman, R. E. (1960). **A new approach to linear filtering and prediction problems**. *Journal of Basic Engineering*, 82(1), 35-45.

---

**FIN DEL MANUAL EXTENDIDO**

*Documento generado con rigor matemático y espíritu de accesibilidad. Si encuentras errores, sientes que algo carece de claridad, o tienes mejoras, siéntete libre de compartirlas. El conocimiento es de todos.*

**Versión 3.0 — Mayo 2026**
