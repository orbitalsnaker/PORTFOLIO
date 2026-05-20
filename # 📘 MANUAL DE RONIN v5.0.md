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

---
