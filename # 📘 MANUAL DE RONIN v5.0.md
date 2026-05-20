--

# 📘 MANUAL DE RONIN v5.0  
**Agente Autónomo con Arquitectura Generador-Crítico, Contratos Pydantic v2 y Presupuesto de Errores**  

*Versión: 5.0 – Para humanos y para inteligencias artificiales*

---

## 🔖 Tabla de Contenidos (estructura completa)

1. **Introducción y filosofía**  
   1.1. ¿Qué es Ronin v5.0?  
   1.2. Arquitectura Generador-Crítico explicada  
   1.3. Por qué contratos tipo Pydantic v2  
   1.4. El presupuesto de errores (Error Budget)  

2. **Despliegue y primer uso**  
   2.1. Requisitos (ninguno, es HTML/JS puro)  
   2.2. Abrir el dashboard  
   2.3. Interfaz de usuario: recorrido visual  

3. **Máquina de estados del agente**  
   3.1. Estados: idle, working, success, error  
   3.2. Transiciones y eventos  

4. **Contratos de validación (el corazón de Ronin)**  
   4.1. Clase `ContractValidator`  
   4.2. Esquema `TaskInput`  
   4.3. Esquema `ExecutionConfig`  
   4.4. Ejemplos de validación exitosa y fallida  

5. **Ciclo de vida de una tarea**  
   5.1. Fase 1: Planning (Planificación)  
   5.2. Fase 2: Execution (Ejecución)  
   5.3. Fase 3: Validation (Validación)  
   5.4. Fase 4: Reflection (Reflexión)  
   5.5. Temporización y flujo asíncrono  

6. **Métricas y presupuesto de errores**  
   6.1. Métricas expuestas  
   6.2. Visualización de barra de error budget  
   6.3. Cuándo se consume el presupuesto  

7. **Logs y cadena de pensamiento (Chain-of-Thought)**  
   7.1. Tipos de log (info, success, warning, error, validation, security)  
   7.2. Formato y colores  
   7.3. Interpretación para IA  

8. **Personalización y configuración**  
   8.1. Parámetros configurables  
   8.2. Modo estricto vs. moderado  
   8.3. Timeout de ejecución  

9. **Extensión para agentes reales**  
   9.1. Conectar a un LLM real (OpenAI, Anthropic, local)  
   9.2. Sandbox de ejecución segura  
   9.3. Persistencia de estado  
   9.4. Modo autónomo continuo  

10. **Seguridad y modelo de amenazas**  
    10.1. Lo que el dashboard NO protege  
    10.2. Buenas prácticas para entornos productivos  

11. **Apéndices**  
    11.1. Código fuente completo anotado  
    11.2. Guía para IA: cómo controlar Ronin mediante consola  
    11.3. Preguntas frecuentes (FAQ)  

---

## PARTE 1: Introducción y filosofía

### 1.1. ¿Qué es Ronin v5.0?

Ronin v5.0 es un **panel de control de misión** para un agente autónomo que genera y valida código. No es un sistema productivo listo para usar, sino un **prototipo funcional** que demuestra conceptos clave de ingeniería de agentes confiables:

- **Contratos explícitos** de entrada/salida (como Pydantic v2 en Python).  
- **Presupuesto de errores** (tomado de la ingeniería de fiabilidad SRE).  
- **Traza completa de razonamiento** (Chain-of-Thought) en logs estilo terminal.  
- **Arquitectura de cuatro fases** que separa planificación, ejecución, validación y reflexión.

Está construido íntegramente con HTML, CSS, JavaScript (Alpine.js y Tailwind). **No necesita backend**; todas las simulaciones ocurren en el navegador.

### 1.2. Arquitectura Generador-Crítico explicada

La arquitectura **Generador-Crítico** es un patrón recurrente en sistemas de IA autónoma:

- **Generador**: produce una solución (código, texto, plan).  
- **Crítico**: evalúa esa solución contra un conjunto de reglas (contratos, seguridad, estilo).  

En Ronin v5.0:

- El **Generador** está en la fase *Execution*: crea una plantilla de código según el lenguaje y nivel de cumplimiento.  
- El **Crítico** se materializa en la fase *Validation*: ejecuta una cadena de 4 validaciones (tipos, contrato, sandbox, recursos).  

Si el Crítico encuentra fallos, se consume presupuesto de errores. Si el presupuesto se agota, el ciclo falla y el agente entra en estado `error`.

### 1.3. Por qué contratos tipo Pydantic v2

**Pydantic v2** es una librería de Python para validación de datos mediante modelos declarativos. Ronin emula este estilo en JavaScript para demostrar:

- **Validación temprana**: antes de ejecutar cualquier lógica, se verifican las entradas.  
- **Mensajes de error detallados**: campo, tipo de error, descripción.  
- **Inmutabilidad de contratos**: los esquemas no cambian en tiempo de ejecución, lo que da predictibilidad.  

Esto es especialmente útil para **agentes autónomos** que reciben tareas de humanos o de otros agentes: si la tarea no cumple el contrato, se rechaza inmediatamente sin desperdiciar cómputo.

### 1.4. El presupuesto de errores (Error Budget)

Concepto tomado de **Site Reliability Engineering (SRE)**:

- Se define un número máximo de fallos tolerables en un período (en Ronin, por ciclo de vida).  
- Cada vez que una validación falla, se consume una unidad del presupuesto.  
- Si se excede el límite, el sistema se niega a continuar o declara el ciclo como fallido.  

Esto fuerza al agente a **priorizar calidad** sobre cantidad, y permite a los operadores ajustar la tolerancia al riesgo (por ejemplo, modo estricto para entornos críticos, modo laxo para pruebas).

## PARTE 2: Despliegue, primer uso y máquina de estados

### 2.1. Requisitos (ninguno, es HTML/JS puro)

Ronin v5.0 es un único archivo HTML que se ejecuta completamente en el navegador. No necesitas:

- Servidor web (puedes abrirlo con `file://`)
- Conexión a internet (excepto para cargar las CDN de Alpine.js y TailwindCSS la primera vez)
- Base de datos
- Lenguajes de backend

**Requisitos del navegador:**
- Cualquier navegador moderno (Chrome, Edge, Firefox, Safari) con JavaScript habilitado.
- Soporte para CSS Grid, Flexbox y animaciones.

### 2.2. Abrir el dashboard

1. Guarda el contenido del archivo `ronin_agent_dashboard.html` en tu computadora.
2. Haz doble clic o arrastra el archivo a una ventana del navegador.
3. Verás el panel con fondo oscuro, texto verde neón y un diseño tipo terminal.

**Opcional**: Si quieres evitar las CDN externas, puedes descargar Alpine.js y TailwindCSS localmente y modificar las rutas en los `<script>` y `<link>`.

### 2.3. Interfaz de usuario: recorrido visual

El dashboard se divide en tres áreas verticales:

**Izquierda (Sidebar – 320px de ancho)**
- Error Budget Configuration: inputs numéricos, checkbox de Strict Mode, botón Apply Config.
- Error Budget Status: barra de progreso, contadores de errores usados, peticiones procesadas, violaciones de contrato y validaciones pasadas.
- Control: botones Clear Logs y Reset Metrics.

**Centro (Main Content – flexible)**
- Diagrama de flujo: cuatro círculos conectados por flechas (Planning → Execution → Validation → Reflection). El círculo activo se ilumina en verde.
- Task Definition: textarea para la descripción de la tarea, dos selects (Language y Compliance Level), y botón Execute Agent Cycle.

**Derecha (dentro del centro abajo)**
- Execution Log & Chain-of-Thought: área con scroll que muestra todos los mensajes del agente, cada uno con color según su tipo (info azul, éxito verde, error rojo, validación verde claro, seguridad violeta).

### 2.4. Máquina de estados del agente

El agente puede estar en uno de cuatro estados principales, visibles en el header mediante un indicador circular y un texto.

| Estado   | Indicador       | Texto mostrado                        | Qué significa                                                                 |
|----------|----------------|----------------------------------------|-------------------------------------------------------------------------------|
| `idle`   | Círculo gris (sin animación) | `IDLE — Ready for input`              | Esperando que el usuario escriba una tarea y haga clic en Execute.           |
| `working`| Círculo ámbar (pulso rápido) | `PROCESSING — Cycle in progress`      | El ciclo de 10.5 segundos está en marcha; el botón Execute está deshabilitado. |
| `success`| Círculo verde (estático) | `SUCCESS — Task completed`            | El ciclo terminó sin exceder el presupuesto de errores. Dura 2 segundos y vuelve a `idle`. |
| `error`  | Círculo rojo (pulso rápido) | `ERROR — Contract violation detected` | Falló la validación de entrada o se agotó el presupuesto de errores.          |

**Transiciones:**
- `idle` → `working`: al hacer clic en Execute Agent Cycle (si la tarea no está vacía y no hay ya un ciclo en curso).
- `working` → `success` o `error`: al completarse el ciclo (después de 10.5 segundos, más los tiempos internos).
- `success`/`error` → `idle`: automáticamente después de 2 segundos.

### 2.5. Primer ciclo de prueba

Para verificar que todo funciona:
1. Escribe en Task Description: `"Genera una función que valide emails"`.
2. Deja Language en `Python` y Compliance Level en `Strict`.
3. Haz clic en **Execute Agent Cycle**.
4. Observa cómo el diagrama de flujo marca cada fase en orden, los logs aparecen con temporizaciones y métricas se actualizan.
5. Al final, el estado vuelve a `idle` y puedes probar otra tarea.

**Prueba de violación de contrato:**
- Escribe una tarea muy corta, por ejemplo `"hi"` (menos de 5 caracteres).
- Ejecuta. Verás un error inmediato en la validación de entrada, sin pasar por las fases. Se incrementa `Contract Violations`.

---

## PARTE 3: Contratos de validación (el corazón de Ronin)

### 3.1. Clase `ContractValidator`

Ronin implementa su propio validador de contratos inspirado en Pydantic v2. No depende de librerías externas. La clase expone un método estático:

```javascript
ContractValidator.validate(data, schema, modelName)
```

**Parámetros:**
- `data`: objeto con los datos a validar (ej. `{ description: "...", language: "python" }`).
- `schema`: objeto que define las reglas por campo.
- `modelName`: string usado solo para propósitos de log (ej. `"TaskInput"`).

**Retorna:**
```javascript
{
  valid: boolean,          // true si no hay errores
  errors: array,           // lista de objetos con field, type, message
  errorCount: number,      // longitud del array errors
  modelName: string
}
```

**Reglas de validación soportadas:**
| Regla         | Tipo de campo | Ejemplo en esquema                  |
|---------------|---------------|--------------------------------------|
| `type`        | cualquier     | `type: 'string'`                     |
| `minLength`   | string        | `minLength: 5`                       |
| `maxLength`   | string        | `maxLength: 2000`                    |
| `ge` (≥)      | number        | `ge: 1`                              |
| `le` (≤)      | number        | `le: 100`                            |
| `pattern`     | string        | `pattern: /^[a-z]+$/`                |
| `enum`        | string        | `enum: ['strict', 'moderate']`       |

### 3.2. Esquema `TaskInput`

Define las reglas para la tarea que el usuario envía al agente.

```javascript
TaskInput: {
  description: { 
    type: 'string', 
    minLength: 5, 
    maxLength: 2000 
  },
  language: { 
    type: 'string', 
    pattern: /^(python|javascript|typescript|sql)$/ 
  },
  complianceLevel: { 
    type: 'string', 
    enum: ['strict', 'moderate', 'baseline'] 
  }
}
```

**Ejemplo de validación exitosa:**
```javascript
const data = {
  description: "Generar función de ordenamiento",
  language: "python",
  complianceLevel: "strict"
};
// Resultado: { valid: true, errors: [], errorCount: 0 }
```

**Ejemplo con errores:**
```javascript
const data = {
  description: "Hi",           // ❌ menor a 5 caracteres
  language: "ruby",            // ❌ no está en el patrón
  complianceLevel: "extreme"   // ❌ no está en el enum
};
// Resultado: errorCount = 3, con mensajes específicos por campo.
```

### 3.3. Esquema `ExecutionConfig`

Valida la configuración del agente (error budget, timeout, modo estricto).

```javascript
ExecutionConfig: {
  maxErrorBudget: { type: 'number', ge: 1, le: 100 },
  executionTimeout: { type: 'number', ge: 100, le: 30000 },
  strictMode: { type: 'boolean' }
}
```

**Nota:** `strictMode` no tiene reglas adicionales porque `boolean` ya se valida con `type`.

### 3.4. Integración en el flujo

**¿Dónde se usan estos contratos?**

1. **Cuando el usuario hace clic en "Apply Config"**  
   Se valida `this.config` contra `ExecutionConfig`. Si falla, el agente entra en estado `error` y muestra los errores en los logs.

2. **Cuando el usuario hace clic en "Execute Agent Cycle"**  
   Se construye un objeto combinando `currentTask` (como `description`) y `taskConfig` (language, complianceLevel). Ese objeto se valida contra `TaskInput`. Si falla, se incrementa `contractViolations`, se muestra error y NO se ejecuta el ciclo.

### 3.5. Formato de los errores

Cada error generado por `ContractValidator` tiene esta estructura:

```javascript
{
  field: "description",
  type: "value_error",        // o "type_error", "string_pattern_mismatch", "enum_error"
  message: "String too short (min: 5)"
}
```

En los logs, Ronin muestra estos errores con sangría y color rojo, agrupados por campo.

### 3.6. Por qué esto es crucial para agentes autónomos

Un agente que opera sin contratos de validación puede:
- Recibir entradas malformadas que causen comportamientos impredecibles.
- Consumir recursos computacionales en tareas inválidas.
- Generar salidas que violen políticas de seguridad o cumplimiento.

Con contratos estilo Pydantic v2:
- **Fail fast**: se rechaza la tarea en milisegundos.
- **Traza clara**: cada error señala exactamente qué campo y por qué.
- **Autodocumentación**: el esquema sirve como especificación legible por humanos y máquinas.

---
## PARTE 4: Ciclo de vida de una tarea (fase por fase)

El ciclo completo dura aproximadamente **10.5 segundos** en la simulación. Cada fase se ejecuta en orden secuencial, con retrasos internos para emular el tiempo de procesamiento. A continuación se detalla qué ocurre en cada fase, **incluyendo el código simulador** que Ronin ejecuta realmente.

### 4.1. Fase 1: Planning (Planificación)
- **Duración simulada:** 1 segundo (retardo interno) + 200 ms de transición.
- **Objetivo:** Interpretar la tarea, mostrar los parámetros y preparar el contexto para la generación.

**Acciones concretas dentro de `simulateAgentCycle()`:**
```javascript
this.transitionPhase('planning', () => {
    this.addLog('├─ PLANNING PHASE', 'info');
    this.addLog('  ├─ Parsing task: ' + this.currentTask.substring(0, 50) + '...', 'info');
    this.addLog('  ├─ Language target: ' + this.taskConfig.language, 'info');
    this.addLog('  └─ Compliance level: ' + this.taskConfig.complianceLevel, 'validation');
}, 1000);
```

**Lo que el humano/IA ve en los logs:**
```
[15:30:01] ├─ PLANNING PHASE
[15:30:01]   ├─ Parsing task: Genera una función que valide emails...
[15:30:01]   ├─ Language target: python
[15:30:01]   └─ Compliance level: strict
```

**Nota para IA:** No hay generación real de código en esta fase. Es puramente declarativa y sirve para que el operador verifique que el agente entendió la tarea.

### 4.2. Fase 2: Execution (Ejecución)
- **Duración simulada:** 2 segundos (retardo interno: se programa a los 1200 ms desde el inicio, pero la fase en sí dura más por el código de plantilla).
- **Objetivo:** Generar una solución (código fuente) según el lenguaje y nivel de cumplimiento.

**Acciones concretas dentro de `generateAndValidateCode()`:**
```javascript
generateAndValidateCode() {
    const codeTemplate = `def process_${this.taskConfig.language}_request(data):
    # Generated with RONIN v5.0
    # Contract: Input must satisfy Pydantic v2 BaseModel
    try:
        validated = SovereigntySchema.model_validate(data)
        return {"status": "OK", "data": validated}
    except ValidationError as e:
        return {"status": "REJECTED", "errors": e.errors()}`;
    
    this.addLog('  ├─ Code generation complete', 'success');
    this.addLog('  ├─ Generated snippet:', 'info');
    codeTemplate.split('\n').forEach(line => {
        this.addLog('      ' + line, 'info');
    });
    this.addLog('  └─ Ready for safety sandbox analysis', 'validation');
}
```

**Lo que el humano/IA ve en los logs:**
```
[15:30:02] ├─ EXECUTION PHASE
[15:30:02]   ├─ Code generation complete
[15:30:02]   ├─ Generated snippet:
[15:30:02]       def process_python_request(data):
[15:30:02]           # Generated with RONIN v5.0
[15:30:02]           # Contract: Input must satisfy Pydantic v2 BaseModel
[15:30:02]           try:
[15:30:02]               validated = SovereigntySchema.model_validate(data)
[15:30:02]               return {"status": "OK", "data": validated}
[15:30:02]           except ValidationError as e:
[15:30:02]               return {"status": "REJECTED", "errors": e.errors()}
[15:30:02]   └─ Ready for safety sandbox analysis
```

**Para IA que quiera extender Ronin:** Aquí es donde se debería invocar una API real de generación de código (OpenAI, Anthropic, o un modelo local). La plantilla fija actual se reemplazaría por una llamada asíncrona.

### 4.3. Fase 3: Validation (Validación)
- **Duración simulada:** 500 ms (pero ocurre a los 3.5 segundos desde el inicio y dura lo que tarden los 4 pasos de validación).
- **Objetivo:** Evaluar el código generado contra una cadena de criterios (tipos, contrato, seguridad, recursos).

**Método `performValidationChain()` interno:**
```javascript
performValidationChain() {
    const validationSteps = [
        { name: 'Type Signature Analysis', pass: true },
        { name: 'Contract Boundary Check', pass: true },
        { name: 'Security Sandbox Eval', pass: true },
        { name: 'Resource Limit Verify', pass: this.metrics.errorsUsed < this.config.maxErrorBudget }
    ];
    
    validationSteps.forEach((step, idx) => {
        const icon = step.pass ? '✓' : '✗';
        const type = step.pass ? 'validation' : 'error';
        this.addLog(`  ${icon} ${step.name}`, type);
        
        if (!step.pass) {
            this.metrics.errorsUsed++;
            this.metrics.contractViolations++;
        } else {
            this.metrics.validationsPassed++;
        }
    });
    
    this.addLog('  └─ Chain-of-validation complete', 'validation');
}
```

**Lo que el humano/IA ve en los logs:**
```
[15:30:03] ├─ VALIDATION PHASE
[15:30:03]   ✓ Type Signature Analysis
[15:30:03]   ✓ Contract Boundary Check
[15:30:03]   ✓ Security Sandbox Eval
[15:30:03]   ✓ Resource Limit Verify
[15:30:03]   └─ Chain-of-validation complete
```

**¿Por qué el cuarto paso (`Resource Limit Verify`) puede fallar?**  
Este paso verifica que `metrics.errorsUsed < config.maxErrorBudget`. Si durante la misma fase de validación ya se han acumulado errores (por pasos previos que fallaron), el cuarto paso fallará. En la simulación actual, los tres primeros pasos siempre pasan (`pass: true` fijo), por lo que solo fallarían si el presupuesto ya estaba agotado de ciclos anteriores. Esto demuestra cómo el presupuesto de errores afecta la fase de validación.

### 4.4. Fase 4: Reflection (Reflexión)
- **Duración simulada:** 1 segundo (ocurre a los 6 segundos desde el inicio).
- **Objetivo:** Resumir el estado del ciclo, mostrar el presupuesto restante y el tiempo de ejecución.

**Código:**
```javascript
this.transitionPhase('reflection', () => {
    this.addLog('├─ REFLECTION PHASE', 'info');
    this.addLog('  ├─ Contract adherence: ' + (this.config.strictMode ? 'STRICT' : 'MODERATE'), 'success');
    this.addLog('  ├─ Error budget: ' + (this.config.maxErrorBudget - this.metrics.errorsUsed) + ' remaining', 'success');
    this.addLog('  └─ Execution time: ~' + Math.random() * 1000 | 0 + 'ms', 'success');
}, 6000);
```

**Lo que se ve en los logs:**
```
[15:30:06] ├─ REFLECTION PHASE
[15:30:06]   ├─ Contract adherence: STRICT
[15:30:06]   ├─ Error budget: 9 remaining
[15:30:06]   └─ Execution time: ~472ms
```

### 4.5. Finalización del ciclo
A los **10.5 segundos** desde el inicio, se ejecuta `completeCycle()`:

```javascript
completeCycle() {
    if (this.metrics.errorsUsed > this.config.maxErrorBudget) {
        this.agentStatus = 'error';
        this.addLog('└─ Cycle FAILED: Error budget exhausted', 'error');
    } else {
        this.agentStatus = 'success';
        this.addLog('└─ Cycle COMPLETED: Task executed with full contract adherence', 'success');
    }
    
    this.currentPhase = 'planning';
    
    setTimeout(() => {
        this.agentStatus = 'idle';
    }, 2000);
}
```

**Posibles salidas del ciclo:**
- **Éxito:** `errorsUsed <= maxErrorBudget` → estado `success`, log verde.
- **Fallo:** `errorsUsed > maxErrorBudget` → estado `error`, log rojo.

Tras 2 segundos en `success` o `error`, el agente vuelve a `idle` y está listo para una nueva tarea.

### 4.6. Diagrama temporal completo (en milisegundos)

```
Evento                           Tiempo (ms)
─────────────────────────────────────────────────
Inicio del ciclo                 0
─ Planning phase                 1000
─ Execution phase comienza       1200
─ Execution fase genera código   3200 (aprox)
─ Validation phase               3500
─ Reflection phase               6000
─ Complete cycle (éxito/fallo)   10500
─ Vuelta a idle                  12500
```

**Nota:** Los tiempos están fijos en la simulación. Para una implementación real con llamadas a API, estos tiempos serían variables y se necesitarían promesas o async/await en lugar de `setTimeout`.

---
## PARTE 5: Métricas y presupuesto de errores

### 5.1. Métricas expuestas

Ronin v5.0 mantiene un objeto `metrics` en tiempo real con cuatro contadores:

| Métrica               | Variable interna          | Descripción                                                                 |
|-----------------------|---------------------------|-----------------------------------------------------------------------------|
| Requests Processed    | `metrics.requestsProcessed` | Número de ciclos de agente iniciados (válidos o inválidos, pero que pasaron la validación de entrada). |
| Contract Violations   | `metrics.contractViolations` | Número de veces que se violó un contrato (ya sea en validación de entrada o en la cadena de validación). |
| Validations Passed    | `metrics.validationsPassed`  | Número de pasos de validación individuales que se superaron con éxito (cada uno de los 4 pasos suma 1 si pasa). |
| Errors Used           | `metrics.errorsUsed`         | Consumo actual del presupuesto de errores. Se incrementa cada vez que un paso de validación falla. |

**Inicialización:**
```javascript
metrics: {
    requestsProcessed: 0,
    contractViolations: 0,
    validationsPassed: 0,
    errorsUsed: 0
}
```

### 5.2. Visualización de la barra de error budget

En el sidebar izquierdo, debajo de "Error Budget Status", hay una barra horizontal con relleno dinámico:

```html
<div class="error-budget-bar">
    <div 
        class="error-budget-fill" 
        :class="errorBudgetClass"
        :style="`width: ${errorBudgetPercentage}%`"
        x-text="`${errorBudgetPercentage}%`"
    ></div>
</div>
```

**Propiedades computadas:**

- `errorBudgetPercentage`:  
  `Math.round((this.metrics.errorsUsed / this.config.maxErrorBudget) * 100)`

- `errorBudgetClass`:  
  - `< 50%` → clase vacía (verde por defecto)  
  - `50% a 79%` → `warning` (gradiente naranja)  
  - `≥ 80%` → `critical` (gradiente rojo)

**Ejemplo visual:**  
Si `maxErrorBudget = 10` y `errorsUsed = 3` → barra al 30%, color verde.  
Si `errorsUsed = 8` → barra al 80%, color rojo, texto "80%" dentro.

### 5.3. ¿Cuándo se consume el presupuesto de errores?

Dentro de `performValidationChain()`, cada paso de validación que falla ejecuta:

```javascript
if (!step.pass) {
    this.metrics.errorsUsed++;
    this.metrics.contractViolations++;
}
```

**En la simulación actual,** los tres primeros pasos (`Type Signature Analysis`, `Contract Boundary Check`, `Security Sandbox Eval`) tienen `pass: true` fijo. El cuarto paso (`Resource Limit Verify`) tiene `pass: this.metrics.errorsUsed < this.config.maxErrorBudget`.

**Escenario de consumo progresivo:**  
Si en un ciclo anterior ya se consumieron 10 errores (presupuesto agotado), en el siguiente ciclo el cuarto paso fallará inmediatamente porque `errorsUsed (10) < maxErrorBudget (10)` es falso. Eso incrementará `errorsUsed` a 11, superando el límite.

**Escenario de consumo por validación de entrada:**  
La validación de la tarea (antes de iniciar el ciclo) NO consume `errorsUsed`, pero sí incrementa `contractViolations`. El presupuesto de errores solo se aplica a fallos dentro de la fase de validación (pasos de la cadena).

### 5.4. Impacto del presupuesto en el ciclo

Al finalizar el ciclo, `completeCycle()` evalúa:

```javascript
if (this.metrics.errorsUsed > this.config.maxErrorBudget) {
    this.agentStatus = 'error';
    this.addLog('└─ Cycle FAILED: Error budget exhausted', 'error');
} else {
    this.agentStatus = 'success';
    // ...
}
```

**Consecuencias:**
- Si el presupuesto se agota, el ciclo se marca como fallido aunque los pasos de validación hayan "pasado" (el cuarto paso falla deliberadamente).
- El error budget **persiste entre ciclos** a menos que el operador:
  - Haga clic en **Reset Metrics** (pone `errorsUsed = 0`).
  - Cambie `maxErrorBudget` y haga clic en **Apply Config** (también resetea `errorsUsed = 0` mediante `recalculateErrorBudget()`).

### 5.5. Relación entre métricas y acciones del usuario

| Acción del usuario              | Efecto en métricas                                                                 |
|--------------------------------|------------------------------------------------------------------------------------|
| Ejecutar tarea válida          | `requestsProcessed++`; `validationsPassed` se incrementa en 4 (si todos los pasos pasan); `errorsUsed` no cambia (si no hay fallos). |
| Ejecutar tarea inválida        | No entra al ciclo; `contractViolations++`; `requestsProcessed` NO aumenta.         |
| Un paso de validación falla    | `errorsUsed++`, `contractViolations++`.                                            |
| Reset Metrics                  | Todas las métricas vuelven a 0.                                                    |
| Apply Config (con nuevo budget)| `maxErrorBudget` se actualiza; `errorsUsed = 0`; otras métricas no se resetean.    |

### 5.6. Interpretación para IA: por qué es importante el error budget

En sistemas autónomos reales, el **error budget** sirve para:

- **Prevenir fallos en cascada:** Si un agente empieza a fallar repetidamente, se le detiene antes de que genere demasiados resultados incorrectos.
- **Medir confianza:** Un agente que opera cerca del 80% de su presupuesto necesita revisión humana o autoreflexión.
- **Ajustar comportamiento:** El modo estricto (`strictMode = true`) puede hacer que ciertos errores que antes eran warnings ahora consuman budget.

En Ronin v5.0, aunque es una simulación, el mecanismo obliga al operador a **monitorear** la barra y **reaccionar** (resetear métricas, aumentar el budget o cambiar tareas).

---

## PARTE 6: Logs y cadena de pensamiento (Chain-of-Thought)

### 6.1. Tipos de log y su significado

Ronin v5.0 clasifica cada entrada de log en uno de seis tipos, cada uno con un color y un borde izquierdo distintivo. Esto permite a humanos e IAs escanear rápidamente el estado del agente.

| Tipo         | Color de texto       | Borde izquierdo     | Uso típico                                                                 |
|--------------|----------------------|---------------------|----------------------------------------------------------------------------|
| `info`       | Azul claro (#93c5fd) | Azul (#3b82f6)      | Eventos normales del ciclo: inicio de fase, mensajes de planificación.     |
| `success`    | Verde claro (#86efac) | Verde (#22c55e)     | Acciones completadas correctamente: generación de código, pasos de validación que pasan. |
| `warning`    | Ámbar (#fcd34d)      | Ámbar (#f59e0b)     | Situaciones que requieren atención pero no son errores fatales (poco usado en la simulación base). |
| `error`      | Rojo claro (#fca5a5) | Rojo (#ef4444)      | Violaciones de contrato, fallos de validación, agotamiento del presupuesto. |
| `validation` | Verde menta (#a7f3d0)| Verde esmeralda (#10b981) | Pasos específicos de la cadena de validación, resultados de chequeos. |
| `security`   | Lavanda (#d8b4fe)    | Púrpura (#8b5cf6)   | Alertas de seguridad (preparado para extensiones futuras; en la simulación base se usa poco). |

**Definición en CSS:**
```css
.log-entry.info { border-left-color: #3b82f6; color: #93c5fd; }
.log-entry.success { border-left-color: var(--success); color: #86efac; }
.log-entry.warning { border-left-color: var(--warning); color: #fcd34d; }
.log-entry.error { border-left-color: var(--error); color: #fca5a5; }
.log-entry.validation { border-left-color: var(--accent-green); color: #a7f3d0; }
.log-entry.security { border-left-color: #8b5cf6; color: #d8b4fe; }
```

### 6.2. Formato de cada entrada

Cada entrada de log se almacena como un objeto:

```javascript
{
  message: "[15:30:01] ├─ PLANNING PHASE",
  type: "info",
  timestamp: "15:30:01",
  details: null  // opcional, puede ser un objeto con datos estructurados
}
```

El método `addLog(message, type, details)`:
- Añade automáticamente la marca de tiempo `[HH:MM:SS]`.
- Inserta el objeto en el array `logs`.
- Provoca un auto-scroll al final del contenedor de logs.

**Ejemplo de log con detalles:**
```javascript
this.addLog('Configuration validation FAILED', 'error', validation.errors);
// Se renderiza con JSON.stringify(details, null, 2) debajo del mensaje.
```

### 6.3. La cadena de pensamiento (Chain-of-Thought) explicada

En sistemas de IA, la **cadena de pensamiento** es una secuencia de pasos intermedios de razonamiento que lleva a una conclusión. En Ronin, cada entrada de log representa un micro-paso de la "mente" del agente.

**Ejemplo de una cadena completa para un ciclo exitoso:**

```
[15:30:01] ┌─ Agent Cycle Started
[15:30:01] ├─ PLANNING PHASE
[15:30:01]   ├─ Parsing task: Generar función de validación de emails...
[15:30:01]   ├─ Language target: python
[15:30:01]   └─ Compliance level: strict
[15:30:02] ├─ EXECUTION PHASE
[15:30:02]   ├─ Code generation complete
[15:30:02]   ├─ Generated snippet: (código)
[15:30:02]   └─ Ready for safety sandbox analysis
[15:30:03] ├─ VALIDATION PHASE
[15:30:03]   ✓ Type Signature Analysis
[15:30:03]   ✓ Contract Boundary Check
[15:30:03]   ✓ Security Sandbox Eval
[15:30:03]   ✓ Resource Limit Verify
[15:30:03]   └─ Chain-of-validation complete
[15:30:06] ├─ REFLECTION PHASE
[15:30:06]   ├─ Contract adherence: STRICT
[15:30:06]   ├─ Error budget: 9 remaining
[15:30:06]   └─ Execution time: ~472ms
[15:30:10] └─ Cycle COMPLETED: Task executed with full contract adherence
```

**Por qué es útil para IA:**  
Una IA que lea estos logs puede:
- Reconstruir el flujo de decisión.
- Identificar en qué fase ocurrió un error.
- Extraer métricas (presupuesto restante, tiempo de ejecución).
- Usar los logs como entrada para un agente supervisor que tome decisiones (ej. "si hay más de 3 errores en validación, cambiar a modo estricto").

### 6.4. Cómo se generan los logs en cada fase

**Planning:**
```javascript
this.addLog('├─ PLANNING PHASE', 'info');
this.addLog('  ├─ Parsing task: ' + this.currentTask.substring(0, 50) + '...', 'info');
```

**Execution:**
```javascript
this.addLog('  ├─ Code generation complete', 'success');
this.addLog('  ├─ Generated snippet:', 'info');
codeTemplate.split('\n').forEach(line => {
    this.addLog('      ' + line, 'info');
});
```

**Validation (pasos):**
```javascript
const icon = step.pass ? '✓' : '✗';
const type = step.pass ? 'validation' : 'error';
this.addLog(`  ${icon} ${step.name}`, type);
```

**Reflection:**
```javascript
this.addLog('  ├─ Contract adherence: ' + (this.config.strictMode ? 'STRICT' : 'MODERATE'), 'success');
```

### 6.5. Funciones auxiliares para logs

**Auto-scroll:**  
Después de añadir un log, Ronin ejecuta:
```javascript
this.$nextTick(() => {
    const container = this.$el.querySelector('.log-container');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
});
```
Esto asegura que el usuario siempre vea el mensaje más reciente.

**Formato visual con HTML:**  
El método `formatLogEntry(log)` devuelve el mensaje y, si existen `details`, los muestra como JSON con estilos:

```javascript
formatLogEntry(log) {
    if (log.details) {
        return `${log.message}\n<span class="text-slate-500">${JSON.stringify(log.details, null, 2)}</span>`;
    }
    return log.message;
}
```

**Limpieza:**  
El botón **Clear Logs** ejecuta `this.logs = []` y añade un log informativo.

### 6.6. Interpretación para IA: cómo analizar los logs programáticamente

Desde la consola del navegador, otra IA o script puede acceder a `ronin.logs` y extraer información:

```javascript
// Obtener la instancia de Alpine
let ronin = document.querySelector('[x-data]').__x.$data;

// Contar errores en los últimos 10 logs
let recentErrors = ronin.logs.slice(-10).filter(l => l.type === 'error');
console.log(`Últimos errores: ${recentErrors.length}`);

// Extraer el presupuesto restante de un log de reflexión
let reflectionLog = ronin.logs.find(l => l.message.includes('Error budget:'));
if (reflectionLog) {
    let budgetMatch = reflectionLog.message.match(/\d+ remaining/);
    console.log(budgetMatch?.[0]);
}
```

Esto permite construir **supervisores automáticos** que reaccionen a los logs sin intervención humana.

### 6.7. Limitaciones actuales y mejoras posibles

- **No hay persistencia:** Los logs se pierden al recargar la página.
- **No hay filtros:** No se pueden ocultar ciertos tipos de log (ej. ver solo errores).
- **No exportación:** No se puede descargar el log como archivo.

**Mejoras sugeridas para extensiones reales:**
- Botón "Export Logs" que descargue un JSON o TXT.
- Checkboxes para filtrar por tipo de log.
- Almacenamiento en `localStorage` para mantener historial entre sesiones.


## PARTE 7: Personalización y configuración

Ronin v5.0 permite a los operadores humanos y a las IAs ajustar varios parámetros para modificar el comportamiento del agente. Estos parámetros se agrupan en el bloque **Error Budget Configuration** del sidebar izquierdo.

### 7.1. Parámetros configurables

| Parámetro              | Tipo      | Rango/Valores         | Default | Efecto inmediato                                                                 |
|------------------------|-----------|-----------------------|---------|----------------------------------------------------------------------------------|
| `maxErrorBudget`       | número    | 1 a 100               | 10      | Máximo de fallos de validación permitidos antes de que el ciclo falle.          |
| `executionTimeout`     | número    | 100 a 30000 ms        | 5000    | Límite de tiempo para la fase de ejecución (actualmente simulado, no enforce).  |
| `strictMode`           | booleano  | true / false          | true    | Controla mensajes de adherencia a contrato y podría endurecer validaciones.     |

Estos parámetros se almacenan en el objeto `config` del estado de Alpine:

```javascript
config: {
    maxErrorBudget: 10,
    executionTimeout: 5000,
    strictMode: true
}
```

### 7.2. Cómo modificar la configuración (interfaz humana)

1. **Cambiar el presupuesto máximo:** Usa el input numérico "Max Validation Errors Allowed". Puedes escribir un número o usar las flechas del teclado.
2. **Cambiar el timeout:** Input "Execution Timeout (ms)". Valor en milisegundos.
3. **Activar/desactivar modo estricto:** Checkbox "Strict Mode".
4. **Aplicar cambios:** Haz clic en **Apply Config**. Esto:
   - Valida la nueva configuración contra el esquema `ExecutionConfig`.
   - Si es válida, actualiza `config` y resetea `errorsUsed = 0` (el error budget se reinicia).
   - Marca `configChanged = false` (esto deshabilita el botón Apply hasta que se vuelva a modificar algo).

**Indicador visual:** El botón "Apply Config" se habilita solo cuando `configChanged === true`, es decir, cuando el usuario ha modificado algún parámetro pero aún no lo ha aplicado.

### 7.3. Validación de la configuración

Antes de aplicar, Ronin ejecuta:

```javascript
const validation = ContractValidator.validate(
    this.config,
    ValidationContracts.ExecutionConfig,
    'ExecutionConfig'
);
```

Si la validación falla (ej. `maxErrorBudget = 0` o `executionTimeout = 50000`), el agente:
- Entra en estado `error`.
- Muestra en los logs los errores específicos.
- **No** aplica la configuración.

**Ejemplo de validación fallida:**  
Si el usuario ingresa `maxErrorBudget = 0`, el validador devuelve:
```javascript
{
  field: "maxErrorBudget",
  type: "value_error",
  message: "Value too small (min: 1)"
}
```

### 7.4. Modo estricto (`strictMode`) – ¿qué hace realmente?

En la versión actual simulada, `strictMode` solo afecta el mensaje de reflexión:

```javascript
this.addLog('  ├─ Contract adherence: ' + (this.config.strictMode ? 'STRICT' : 'MODERATE'), 'success');
```

**Pero en una extensión real,** el modo estricto podría:
- Rechazar tipos de datos ambiguos (ej. pasar un string donde se espera número).
- Exigir que todos los pasos de validación pasen sin ningún fallo (no tolerar ni un error).
- Activar validaciones adicionales (ej. análisis semántico, verificación de tipos estáticos).

**Recomendación para IA:** Si estás extendiendo Ronin, haz que `strictMode = true` convierta cualquier advertencia en error y consuma budget inmediatamente.

### 7.5. `executionTimeout` – implementación actual y limitaciones

Actualmente, el `executionTimeout` **no está activamente enforce** en la simulación porque los tiempos están fijos con `setTimeout`. Sin embargo, el parámetro se valida y se almacena.

**Para una implementación real:**
- Durante la fase de Execution, se iniciaría un temporizador con `setTimeout` que, al expirar, interrumpiría la generación de código y marcaría un error de timeout.
- Se necesitaría un mecanismo de aborto (ej. `AbortController` en fetch, o promesas con tiempo límite).

### 7.6. Cómo modificar la configuración programáticamente (para IA)

Un agente externo puede cambiar la configuración mediante la consola o un script:

```javascript
let ronin = document.querySelector('[x-data]').__x.$data;

// Cambiar parámetros directamente (sin validación)
ronin.config.maxErrorBudget = 20;
ronin.config.strictMode = false;

// Marcar que hay cambios pendientes para habilitar el botón Apply
ronin.configChanged = true;

// O aplicar la configuración directamente (con validación)
ronin.applyConfiguration();
```

**Advertencia:** Si se modifican los parámetros sin llamar a `applyConfiguration()`, los cambios no se validan y el error budget **no se resetea automáticamente**. Para un comportamiento coherente, siempre es mejor llamar a `applyConfiguration()`.

### 7.7. Persistencia de la configuración

**Estado actual:** La configuración se pierde al recargar la página.

**Mejora sugerida:** Guardar la configuración en `localStorage`:

```javascript
// Guardar después de aplicar
localStorage.setItem('ronin_config', JSON.stringify(this.config));

// Cargar al iniciar
const saved = localStorage.getItem('ronin_config');
if (saved) this.config = JSON.parse(saved);
```

### 7.8. Buenas prácticas para operadores humanos

- **Para tareas críticas (código de producción):** Usa `maxErrorBudget = 1` o `2` y `strictMode = true`.
- **Para exploración o prototipado:** Usa `maxErrorBudget = 20` y `strictMode = false`.
- **Timeout:** Si planeas integrar con APIs reales, ajusta `executionTimeout` a un valor ligeramente superior al tiempo esperado de respuesta (ej. 10000 ms para LLMs lentos).

### 7.9. Posibles extensiones de configuración

| Parámetro adicional      | Propósito                                                                 |
|--------------------------|---------------------------------------------------------------------------|
| `maxValidationSteps`     | Número de pasos en la cadena de validación (ahora fijo en 4).             |
| `autoResetBudget`        | Si se debe resetear el error budget automáticamente después de N ciclos.  |
| `logLevel`               | `debug`, `info`, `error` – para filtrar la verbosidad.                    |
| `sandboxEnabled`         | Activar/desactivar el análisis de sandbox (ahora simulado).               |

---

## PARTE 8: Extensión para agentes reales

Ronin v5.0 es un prototipo funcional, pero su verdadero potencial se despliega cuando se conecta a servicios reales: generación de código mediante LLMs, ejecución en entornos aislados, persistencia de estado y operación autónoma continua. Esta parte del manual explica cómo **extender Ronin** para convertirlo en un agente de producción.

### 8.1. Conectar a un LLM real (OpenAI, Anthropic, local)

En la simulación actual, la fase de *Execution* genera una plantilla fija. Para convertirla en un generador real:

**Paso 1: Reemplazar `generateAndValidateCode()`**

```javascript
async generateAndValidateCode() {
    this.addLog('  ├─ Calling LLM API...', 'info');
    
    const prompt = `Generate ${this.taskConfig.language} code for: ${this.currentTask}
    Compliance level: ${this.taskConfig.complianceLevel}
    Return only the code block.`;
    
    try {
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_KEY}`
            },
            body: JSON.stringify({
                model: 'gpt-4',
                messages: [{ role: 'user', content: prompt }],
                temperature: 0.2
            })
        });
        
        const data = await response.json();
        const generatedCode = data.choices[0].message.content;
        
        this.addLog('  ├─ LLM response received', 'success');
        this.addLog('  ├─ Generated code:', 'info');
        generatedCode.split('\n').forEach(line => {
            this.addLog('      ' + line, 'info');
        });
        
        // Guardar el código generado para la fase de validación
        this.generatedCode = generatedCode;
        
    } catch (error) {
        this.addLog(`  ├─ LLM API error: ${error.message}`, 'error');
        this.metrics.errorsUsed++;
        this.metrics.contractViolations++;
    }
}
```

**Paso 2: Manejar asincronía y timeouts**  
Dado que `executeAgent()` actualmente usa `setTimeout` síncrono, deberás convertir todo el ciclo a `async/await` o usar promesas. Un enfoque limpio es reescribir `simulateAgentCycle()` como función asíncrona.

### 8.2. Sandbox de ejecución segura

La validación de código generado no es completa sin ejecutarlo en un entorno controlado. Opciones:

| Solución          | Descripción                                                                 | Pros                                  | Contras                                |
|-------------------|-----------------------------------------------------------------------------|---------------------------------------|----------------------------------------|
| **Web Worker**    | Ejecutar código JS en un worker aislado                                     | Nativo en navegador, sin servidor      | Solo JS, seguridad limitada            |
| **Pyodide**       | Python en WebAssembly dentro del navegador                                  | Ejecuta Python real, aislado           | Pesado, no acceso a librerías nativas  |
| **Sandbox remoto**| API que recibe código y lo ejecuta en contenedor (ej. Piston, Embox)        | Seguro, múltiples lenguajes            | Requiere backend, latencia             |

**Ejemplo de integración con Piston API (sandbox multilingüe):**

```javascript
async executeInSandbox(code, language) {
    const response = await fetch('https://emkc.org/api/v2/piston/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            language: language,
            source: code
        })
    });
    const result = await response.json();
    return result;
}
```

Llamar esto desde `performValidationChain()` como un paso adicional de seguridad.

### 8.3. Persistencia de estado

Actualmente, recargar la página pierde logs, métricas y configuración. Para un agente autónomo que opera durante horas o días, se necesita persistencia.

**Usar `localStorage` para guardar/restaurar:**

```javascript
// Guardar estado completo
saveState() {
    const state = {
        metrics: this.metrics,
        config: this.config,
        logs: this.logs.slice(-1000) // solo últimos 1000 logs
    };
    localStorage.setItem('ronin_state', JSON.stringify(state));
}

// Cargar al iniciar
loadState() {
    const saved = localStorage.getItem('ronin_state');
    if (saved) {
        const state = JSON.parse(saved);
        this.metrics = state.metrics;
        this.config = state.config;
        this.logs = state.logs;
        this.addLog('State restored from localStorage', 'info');
    }
}

// Guardar automáticamente después de cada ciclo
completeCycle() {
    // ... lógica existente ...
    this.saveState();
}
```

### 8.4. Modo autónomo continuo (cola de tareas)

Para que Ronin funcione sin intervención humana, debe consumir tareas de una cola (array, archivo, API REST) y ejecutarlas en bucle.

**Implementación básica:**

```javascript
taskQueue: [],
autoMode: false,

addTaskToQueue(description, language, complianceLevel) {
    this.taskQueue.push({ description, language, complianceLevel });
    if (this.autoMode && this.agentStatus === 'idle') {
        this.processNextTask();
    }
},

async processNextTask() {
    if (this.taskQueue.length === 0) return;
    const task = this.taskQueue.shift();
    this.currentTask = task.description;
    this.taskConfig.language = task.language;
    this.taskConfig.complianceLevel = task.complianceLevel;
    await this.executeAgent(); // esperar a que termine
    if (this.autoMode && this.taskQueue.length > 0) {
        setTimeout(() => this.processNextTask(), 1000);
    }
},

startAutoMode() {
    this.autoMode = true;
    this.addLog('AUTONOMOUS MODE ACTIVATED', 'success');
    this.processNextTask();
},

stopAutoMode() {
    this.autoMode = false;
    this.addLog('AUTONOMOUS MODE DEACTIVATED', 'warning');
}
```

**Botones en UI:** Añadir "Start Auto Mode" y "Stop Auto Mode", más un textarea para cargar tareas masivas (una por línea en formato JSON o CSV).

### 8.5. Integración con sistemas externos (webhooks, monitoreo)

Un agente real debe poder notificar a otros sistemas cuando ocurren eventos importantes:

```javascript
notifyWebhook(event, data) {
    if (!this.config.webhookUrl) return;
    
    fetch(this.config.webhookUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            agent: 'ronin-v5',
            event: event, // 'cycle_completed', 'budget_exhausted', 'contract_violation'
            timestamp: new Date().toISOString(),
            metrics: this.metrics,
            lastLog: this.logs.slice(-1)[0]
        })
    }).catch(e => console.warn('Webhook failed', e));
}
```

Llamar a `notifyWebhook('budget_exhausted')` cuando `errorsUsed > maxErrorBudget`.

### 8.6. Seguridad y manejo de secretos

**Nunca** incluyas claves de API en el código fuente. Para desarrollo local:

- Usa variables de entorno (si usas un servidor de desarrollo como Vite o Webpack).
- Para el HTML puro, pide al usuario que ingrese su clave en un input de tipo `password` y guárdala en `sessionStorage` (nunca `localStorage`).

```html
<input type="password" x-model="apiKey" placeholder="OpenAI API Key" class="input-field w-full">
```

### 8.7. Ejemplo de arquitectura extendida (diagrama conceptual)

```
[Usuario/IA] → Task Queue → Ronin v5.0 (extendido)
                               ↓
                         [LLM API] → generación de código
                               ↓
                         [Sandbox] → ejecución segura
                               ↓
                         [Validadores] → contratos, tipos, seguridad
                               ↓
                         [Métricas + Logs] → persistencia (DB/localStorage)
                               ↓
                         [Webhooks] → alertas a Slack, Datadog, etc.
```

### 8.8. Advertencias para desarrolladores que extienden Ronin

- **No confíes en el `setTimeout` como control de flujo real:** Reemplázalo por promesas y `async/await`.
- **Maneja errores de red y timeouts:** Las APIs externas fallan. Diseña reintentos con backoff exponencial.
- **Protege el presupuesto de errores:** Si una API externa falla repetidamente, consume budget y eventualmente detén el agente.
- **Escalabilidad:** El Ronin actual es monousuario. Para múltiples agentes simultáneos, necesitas backend (Node, Python) y websockets.

### 8.9. Guía para IA: cómo auto-extenderse

Si eres una IA que ha recibido este manual y quieres **reescribir Ronin** para tus propósitos:

1. **Identifica tu fase más débil** (planning, execution, validation, reflection) y mejora esa.
2. **Empieza con un solo reemplazo:** Conecta la generación de código a un LLM y deja el resto igual.
3. **Prueba el presupuesto de errores:** Haz que fallos reales (API timeout, código mal formado) consuman budget.
4. **Añade un modo CLI** para operar sin navegador (Node.js con Puppeteer o reescribe en Python).
5. **Documenta tus extensiones** siguiendo el mismo estilo de este manual.

---

).## PARTE 9: Seguridad y modelo de amenazas

Ronin v5.0, como prototipo de agente autónomo que genera y valida código, debe ser analizado desde una perspectiva de seguridad. Esta parte del manual describe **qué amenazas considera el diseño actual**, **cuáles no cubre**, y **cómo mitigar riesgos** en implementaciones reales.

### 9.1. Lo que el dashboard actual NO protege

Es crucial entender las limitaciones de seguridad de la versión simulada:

| Riesgo                          | Explicación                                                                 | Estatus en Ronin v5.0 |
|---------------------------------|-----------------------------------------------------------------------------|----------------------|
| **Inyección de código**         | Un usuario malicioso podría escribir una tarea que genere código peligroso. | 🟡 Simulado (no ejecuta código real, pero el snippet se muestra) |
| **Ejecución no autorizada**     | El agente no ejecuta el código generado, solo lo muestra.                   | 🟢 No hay riesgo de ejecución real. |
| **Fuga de información sensible**| Los logs se muestran en pantalla y podrían contener secretos si el usuario los escribe. | 🔴 El operador debe evitar poner contraseñas en las tareas. |
| **Denegación de servicio**      | Bucle infinito o consumo excesivo de recursos del navegador.               | 🟡 Los `setTimeout` son finitos, pero un bucle malicioso en el código simulado no afecta. |
| **Manipulación del presupuesto**| Un usuario podría modificar `maxErrorBudget` y `errorsUsed` desde consola.  | 🟡 Posible, pero es un entorno de desarrollo; en producción se requeriría backend. |
| **Ataques XSS**                 | El dashboard usa `innerHTML` implícitamente en `formatLogEntry`.           | 🟡 Si un log contiene HTML malicioso, podría ejecutarse. |

### 9.2. Vulnerabilidades específicas del código actual

**1. Posible XSS en `formatLogEntry`**  
El método devuelve el mensaje como HTML, y aunque los mensajes son generados internamente, si algún log incluye texto proveniente del usuario sin sanitizar, podría ser vulnerable.

```javascript
formatLogEntry(log) {
    if (log.details) {
        return `${log.message}\n<span class="text-slate-500">${JSON.stringify(log.details, null, 2)}</span>`;
    }
    return log.message;
}
```

**Mitigación:**  
- Usar `textContent` en lugar de `innerHTML` para el mensaje principal.  
- Sanitizar cualquier entrada del usuario antes de mostrarla en logs (ej. `DOMPurify.sanitize`).

**2. Acceso desde consola a datos internos**  
Cualquier persona con herramientas de desarrollador puede ejecutar:

```javascript
document.querySelector('[x-data]').__x.$data.metrics.errorsUsed = 0;
```

Esto permite manipular el presupuesto de errores, logs, etc. En un entorno de demostración no es crítico, pero en producción debería estar protegido (por ejemplo, ejecutando el agente en un Worker o backend).

### 9.3. Modelo de amenazas para una implementación real

Si extiendes Ronin para conectarlo a LLMs, sandbox de ejecución y persistencia, el modelo de amenazas se amplía significativamente.

| Activo                       | Amenazas posibles                                                                 |
|------------------------------|-----------------------------------------------------------------------------------|
| **Claves de API (LLM)**      | Robo mediante inspección de código, almacenamiento inseguro.                      |
| **Código generado**          | Podría contener malware, backdoors, o comandos peligrosos.                        |
| **Sandbox de ejecución**     | Escape del contenedor, acceso al sistema anfitrión, consumo de recursos.          |
| **Logs y métricas**          | Fuga de información de la tarea (ej. propiedad intelectual, datos personales).    |
| **Cola de tareas**           | Inyección de tareas maliciosas si no está autenticada.                            |

### 9.4. Buenas prácticas para entornos productivos

**Para el manejo de LLM:**
- Nunca incluyas claves API en el cliente. Usa un backend proxy que autentique al agente.
- Valida y sanitiza el prompt del usuario antes de enviarlo al LLM (ej. eliminar instrucciones de sistema ocultas).
- Implementa rate limiting por usuario o por tarea.

**Para el sandbox de ejecución:**
- Ejecuta el código en contenedores desechables (Docker, gVisor, Firecracker).
- Limita recursos: CPU, memoria, tiempo de ejecución, acceso a red.
- Usa un timeout estricto (ej. 5 segundos para código simple).
- Desmonta cualquier sistema de archivos sensible.

**Para la persistencia de logs:**
- No almacenes logs que contengan el código generado a menos que sea necesario.
- Si los almacenas, cifra los datos sensibles.
- Implementa retención automática (ej. borrar logs de más de 30 días).

**Para la autenticación y autorización:**
- Si Ronin opera como servicio, añade autenticación (API keys, JWT, OAuth).
- Diferencia entre roles: operador (puede ver todo), administrador (puede cambiar configuración), auditor (solo logs).

### 9.5. Protección del presupuesto de errores en entornos reales

El presupuesto de errores es un mecanismo de control, pero un atacante podría intentar:
- **Agotar el presupuesto a propósito** para causar denegación de servicio.
- **Manipular las métricas** para que nunca se agote (si no están protegidas).

**Soluciones:**
- Almacena las métricas en el servidor, no en el cliente.
- Firma las métricas con HMAC si deben ser enviadas al cliente solo para visualización.
- Implementa un umbral de tasa: si se consumen más de N errores en M segundos, bloquea temporalmente al usuario.

### 9.6. Seguridad en la cadena de validación

Los cuatro pasos de validación actuales son simulados. En una implementación real:

| Paso                      | Implementación segura                                                                 |
|---------------------------|---------------------------------------------------------------------------------------|
| Type Signature Analysis  | Usar un parser del lenguaje (ej. `ast` en Python, `esprima` en JS) para validar tipos. |
| Contract Boundary Check  | Verificar que el código respete las interfaces definidas (ej. funciones esperadas).    |
| Security Sandbox Eval    | Ejecutar análisis estático (banderas de linter de seguridad).                          |
| Resource Limit Verify    | Contar bucles, recursión, uso de memoria en el código fuente (sin ejecutar).          |

### 9.7. Manejo de secretos en el código generado

Un riesgo específico: el LLM podría generar código que incluya claves API hardcodeadas (ej. `api_key = "sk-..."`). Para mitigarlo:

- Ejecutar un detector de secretos (regex, entropy) sobre el código generado.
- Si se detecta un secreto, marcar la validación como fallida y consumir budget.

### 9.8. Recomendaciones para operadores humanos

- **Nunca** ejecutes Ronin en una máquina que contenga datos sensibles sin sandboxing.
- **Revisa manualmente** el código generado antes de usarlo en producción, incluso si todas las validaciones pasaron.
- **No confíes ciegamente** en el presupuesto de errores como única medida de calidad.
- **Mantén actualizadas** las dependencias (Alpine.js, Tailwind) para parchear vulnerabilidades conocidas.

### 9.9. Hoja de ruta de seguridad sugerida para v6.0

- [ ] Mover la lógica del agente a un Web Worker para aislarla del DOM.
- [ ] Implementar Content Security Policy (CSP) estricta.
- [ ] Añadir un modo "solo lectura" de logs sin capacidad de ejecutar tareas.
- [ ] Integrar un linter de seguridad (Bandit para Python, ESLint con reglas de seguridad para JS).
- [ ] Proveer una versión Dockerizada con sandbox por defecto.



## PARTE 10: Apéndices

### 10.1. Código fuente completo anotado

El archivo `ronin_agent_dashboard.html` contiene aproximadamente 650 líneas entre HTML, CSS y JavaScript. Aquí se presentan las secciones más relevantes con anotaciones explicativas.

#### Estructura del documento

```html
<!DOCTYPE html>
<html lang="es">
<head>
    <!-- Meta y títulos -->
    <title>Ronin v5.0 - Agente Autónomo | Control de Misión</title>
    
    <!-- Dependencias externas -->
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Estilos personalizados (variables, logs, animaciones, barra de error budget) -->
    <style>...</style>
</head>
<body>
    <div x-data="roninAgent()" class="...">
        <!-- Header con estado y reloj -->
        <!-- Sidebar izquierdo: configuración + métricas + controles -->
        <!-- Área central: diagrama de flujo + entrada de tarea + logs -->
    </div>
    
    <script>
        // Contratos de validación (esquemas Pydantic-like)
        const ValidationContracts = { TaskInput: {...}, ExecutionConfig: {...} };
        
        // Validador de contratos
        class ContractValidator { static validate(data, schema, modelName) {...} }
        
        // Componente Alpine.js principal
        function roninAgent() {
            return {
                // Estado reactivo
                config: { maxErrorBudget: 10, executionTimeout: 5000, strictMode: true },
                currentTask: '',
                taskConfig: { language: 'python', complianceLevel: 'strict' },
                logs: [],
                metrics: { requestsProcessed: 0, contractViolations: 0, validationsPassed: 0, errorsUsed: 0 },
                agentStatus: 'idle',
                currentPhase: 'planning',
                phases: ['planning', 'execution', 'validation', 'reflection'],
                configChanged: false,
                currentTime: new Date().toLocaleTimeString(),
                
                // Propiedades computadas (getters)
                get statusText() {...},
                get errorBudgetPercentage() {...},
                get errorBudgetClass() {...},
                
                // Métodos principales
                init() {...},
                addLog(message, type, details) {...},
                formatLogEntry(log) {...},
                applyConfiguration() {...},
                recalculateErrorBudget() {...},
                executeAgent() {...},
                simulateAgentCycle() {...},
                transitionPhase(phase, callback, delay) {...},
                generateAndValidateCode() {...},
                performValidationChain() {...},
                completeCycle() {...},
                clearLogs() {...},
                resetMetrics() {...},
                startTimeClock() {...}
            };
        }
    </script>
</body>
</html>
```

#### Anotación del validador de contratos (la pieza más reutilizable)

```javascript
class ContractValidator {
    static validate(data, schema, modelName) {
        const errors = [];
        
        for (const [field, rules] of Object.entries(schema)) {
            const value = data[field];
            
            // 1. Validación de tipo
            if (rules.type && typeof value !== rules.type) {
                errors.push({ field, type: 'type_error', 
                              message: `Expected ${rules.type}, got ${typeof value}` });
                continue; // No seguir validando este campo si el tipo ya es incorrecto
            }
            
            // 2. Validaciones de string
            if (rules.minLength && value.length < rules.minLength) {
                errors.push({ field, type: 'value_error', 
                              message: `String too short (min: ${rules.minLength})` });
            }
            if (rules.maxLength && value.length > rules.maxLength) {
                errors.push({ field, type: 'value_error', 
                              message: `String too long (max: ${rules.maxLength})` });
            }
            
            // 3. Validaciones numéricas (ge = greater or equal, le = less or equal)
            if (typeof rules.ge !== 'undefined' && value < rules.ge) {
                errors.push({ field, type: 'value_error', 
                              message: `Value too small (min: ${rules.ge})` });
            }
            if (typeof rules.le !== 'undefined' && value > rules.le) {
                errors.push({ field, type: 'value_error', 
                              message: `Value too large (max: ${rules.le})` });
            }
            
            // 4. Patrón regex
            if (rules.pattern && !rules.pattern.test(value)) {
                errors.push({ field, type: 'string_pattern_mismatch', 
                              message: `Value does not match required pattern` });
            }
            
            // 5. Enumeración
            if (rules.enum && !rules.enum.includes(value)) {
                errors.push({ field, type: 'enum_error', 
                              message: `Value must be one of: ${rules.enum.join(', ')}` });
            }
        }
        
        return { valid: errors.length === 0, errors, errorCount: errors.length, modelName };
    }
}
```

**Uso extensible:** Este validador puede extraerse y usarse en cualquier proyecto JavaScript/TypeScript para validar objetos de configuración, respuestas de API, o entradas de usuario.

### 10.2. Guía para IA: cómo controlar Ronin mediante consola

Si eres una IA (o un humano con scripts) y quieres automatizar la interacción con Ronin sin hacer clic, la consola del navegador es tu aliada. Aquí hay una lista de comandos útiles.

#### Obtener la instancia del agente

```javascript
const ronin = document.querySelector('[x-data]').__x.$data;
```

#### Configuración y métricas (lectura)

```javascript
console.log(ronin.config);                 // {maxErrorBudget:10, executionTimeout:5000, strictMode:true}
console.log(ronin.metrics);                // {requestsProcessed, contractViolations, ...}
console.log(ronin.logs.slice(-5));         // Últimos 5 logs
console.log(ronin.agentStatus);            // "idle", "working", "success", "error"
```

#### Configuración (escritura)

```javascript
// Cambiar presupuesto a 20 y modo no estricto
ronin.config.maxErrorBudget = 20;
ronin.config.strictMode = false;
ronin.configChanged = true;                // Para habilitar visualmente el botón Apply
ronin.applyConfiguration();                // Valida y aplica
```

#### Enviar una tarea programáticamente

```javascript
ronin.currentTask = "Generar función de fibonacci en Python";
ronin.taskConfig.language = "python";
ronin.taskConfig.complianceLevel = "strict";
ronin.executeAgent();                      // Inicia el ciclo
```

#### Monitorear logs en tiempo real

```javascript
const observer = setInterval(() => {
    const lastLog = ronin.logs[ronin.logs.length - 1];
    if (lastLog) console.log(lastLog.message);
    if (ronin.agentStatus === 'idle') {
        console.log('Ciclo terminado, estado idle');
        clearInterval(observer);
    }
}, 500);
```

#### Resetear métricas y logs

```javascript
ronin.resetMetrics();   // Limpia métricas (requests, violaciones, validationsPassed, errorsUsed = 0)
ronin.clearLogs();      // Vacía el array de logs y añade un log informativo
```

#### Extraer el presupuesto restante de un log de reflexión

```javascript
const reflectionLog = ronin.logs.find(l => l.message.includes('Error budget:'));
if (reflectionLog) {
    const match = reflectionLog.message.match(/(\d+) remaining/);
    console.log(`Presupuesto restante: ${match[1]}`);
}
```

#### Forzar el estado a idle (si el agente se queda atascado)

```javascript
ronin.agentStatus = 'idle';
ronin.currentPhase = 'planning';
```

**Advertencia:** Manipular el estado directamente puede romper la máquina de estados. Úsalo solo en desarrollo o depuración.

### 10.3. Preguntas frecuentes (FAQ)

#### P1: ¿Ronin ejecuta realmente el código que genera?
**R:** No. En la versión actual, solo muestra el código como texto en los logs. La fase de validación es simulada. Para ejecución real, debes extenderlo con un sandbox (ver Parte 8).

#### P2: ¿Puedo usar Ronin con otros lenguajes además de Python, JS, TS y SQL?
**R:** El esquema `TaskInput` solo permite esos cuatro lenguajes por defecto. Puedes modificar el patrón `pattern: /^(python|javascript|typescript|sql)$/` en el código fuente para añadir más (ej. `java|go|rust`). También actualiza el `<select>` de la interfaz.

#### P3: ¿Por qué el tiempo de ejecución reportado en reflexión es aleatorio (`Math.random() * 1000`)?
**R:** Es una simulación para mostrar la variabilidad natural del tiempo de procesamiento. En una implementación real, ese valor se calcularía cronometrando las fases.

#### P4: ¿Cómo cambio la velocidad del ciclo (ahora 10.5 segundos)?
**R:** Modifica los retardos en `simulateAgentCycle()`. Busca los valores `1000`, `1200`, `3500`, `6000`, `10500` y ajústalos a tus necesidades. Si integras APIs reales, reemplaza `setTimeout` por promesas y `async/await`.

#### P5: ¿Puedo tener múltiples agentes Ronin en la misma página?
**R:** Sí, pero necesitas renombrar la función `roninAgent()` y montar componentes Alpine independientes con diferentes IDs o clases. Cada uno tendrá su propio estado.

#### P6: ¿Qué significa "Data Sovereignty Protocol" en el subtítulo?
**R:** Es un nombre conceptual. En el contexto del dashboard, indica que los datos de la tarea y el código generado no salen del navegador (a menos que agregues integraciones externas). El operador mantiene el control soberano sobre sus datos.

#### P7: ¿Por qué a veces el cuarto paso de validación (`Resource Limit Verify`) falla aunque no haya errores previos?
**R:** El cuarto paso verifica `this.metrics.errorsUsed < this.config.maxErrorBudget`. Si en un ciclo anterior ya se había consumido el presupuesto completo (ej. `errorsUsed = 10`, `maxErrorBudget = 10`), entonces `10 < 10` es falso, y falla. Así se demuestra que el error budget persiste entre ciclos.

#### P8: ¿Puedo ejecutar Ronin en Node.js sin navegador?
**R:** El código actual depende del DOM (Alpine.js, Tailwind, `document.querySelector`). Necesitarías una herramienta como Puppeteer o Playwright para emular un navegador, o reescribir la lógica del agente en Node.js puro (eliminando la UI).

#### P9: ¿Cómo se guardan los logs si cierro el navegador?
**R:** No se guardan. Puedes añadir persistencia con `localStorage` como se muestra en la Parte 8, o implementar un botón "Export Logs" que descargue un archivo JSON.

#### P10: ¿Ronin es un producto real o solo una demostración?
**R:** Es una demostración conceptual y una base de código funcional para aprender y extender. No es un producto comercial ni está listo para producción sin las extensiones de seguridad y backend discutidas en las Partes 8 y 9.

### 10.4. Glosario de términos

| Término                  | Definición                                                                 |
|--------------------------|----------------------------------------------------------------------------|
| **Agente autónomo**      | Sistema que realiza tareas sin intervención humana continua.               |
| **Arquitectura Generador-Crítico** | Patrón donde un componente genera soluciones y otro las evalúa.      |
| **Chain-of-Thought**     | Secuencia de pasos de razonamiento intermedios que lleva a una conclusión. |
| **Contrato de validación** | Conjunto de reglas que una entrada o salida debe cumplir.                 |
| **Error budget**         | Cantidad de fallos tolerables en un sistema, tomado de SRE.                |
| **Pydantic v2**          | Librería Python para validación de datos mediante modelos declarativos.    |
| **Sandbox**              | Entorno aislado para ejecutar código no confiable.                         |
| **SRE**                  | Site Reliability Engineering, disciplina que aplica principios de ingeniería de software a la fiabilidad. |

### 10.5. Historial de versiones (conceptual)

| Versión | Cambios principales                                                                 |
|---------|--------------------------------------------------------------------------------------|
| v1.0    | Prototipo inicial: generación de código simple, sin validación.                      |
| v2.0    | Añadida validación de entrada mediante contratos.                                    |
| v3.0    | Incorporado el presupuesto de errores y los 4 pasos de validación.                   |
| v4.0    | Interfaz de usuario rediseñada con Alpine.js y Tailwind, diagrama de flujo.          |
| **v5.0**| **Versión actual:** logs con tipado, barra de error budget dinámica, simulación completa del ciclo, manual extenso. |
| v6.0 (futuro) | Integración con LLM real, sandbox de ejecución, persistencia, modo autónomo.   |

### 10.6. Créditos y licencia

- **Autor original:** (desconocido, código proporcionado por el usuario)
- **Manual y anotaciones:** Generados por asistente IA bajo petición del usuario.
- **Licencia sugerida:** MIT (para el código fuente) – permite uso, copia, modificación y distribución, siempre que se incluya el aviso de copyright.
- **Dependencias:** Alpine.js (MIT), TailwindCSS (MIT).

---

## FIN DEL MANUAL HIPER EXTENSO

Has recibido **10 partes** que componen el manual completo. Puedes copiar cada parte en un archivo Markdown (`.md`) y compilarlo con cualquier visor de Markdown (Typora, Obsidian, GitHub, etc.) o convertirlo a PDF.

**Resumen de partes:**
1. Introducción y filosofía
2. Despliegue, primer uso y máquina de estados
3. Contratos de validación
4. Ciclo de vida de una tarea (fase por fase)
5. Métricas y presupuesto de errores
6. Logs y cadena de pensamiento
7. Personalización y configuración
8. Extensión para agentes reales
9. Seguridad y modelo de amenazas
10. Apéndices (código anotado, guía para IA, FAQ, glosario, historial)

**¡Ronin v5.0 ahora está documentado para humanos y para IAs!** 🚀
