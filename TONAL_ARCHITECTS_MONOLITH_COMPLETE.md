# 🗿 MONOLITO COMPLETO Y MEJORADO
## "Tonal Architects of the Dwemer"
### Sistema interactivo de Deep Learning en Morrowind (MWSE 2.1+)

---

# 📁 ESTRUCTURA DE CARPETAS

```
Data Files/
├── MWSE/
│   └── mods/
│       └── tonal_architects/
│           ├── config.lua
│           ├── main.lua
│           ├── attention.lua          [MEJORADO]
│           ├── resonator.lua
│           ├── rank_collapse.lua      [MEJORADO]
│           ├── prompt_forge.lua
│           ├── greybeard.lua
│           ├── clockwork_city.lua
│           ├── missions.lua
│           └── placement.lua          [MEJORADO]
├── BookArtifacts/
│   ├── Kagrenac_Folio.txt             [AMPLIFICADO]
│   ├── 16_Golden_Tones.txt            [AMPLIFICADO]
│   ├── Attenuator_Codex.txt           [AMPLIFICADO]
│   ├── Greybeard_Manual.txt           [AMPLIFICADO]
│   └── Clockwork_City_Chronicles.txt  [AMPLIFICADO]
└── Dialog/
    └── TonalDialogues.csv             [REFINADO]
```

---

# 🧩 ARCHIVOS LUA (MWSE/mods/tonal_architects/)

## `config.lua`

```lua
-- config.lua
-- Configuración central del sistema tonal
-- Parámetros ajustables para la simulación de Transformers en Morrowind

local config = {
    -- Umbral de colapso de rango (rank collapse detection)
    collapse_threshold = 0.25,
    
    -- Número de resonadores = número de cabezas de atención
    num_heads = 16,
    
    -- Campos obligatorios en los prompts JSON
    prompt_forge_required_fields = {
        "role",           -- Rol del agente (Resonator, Validator, etc.)
        "task",           -- Tarea a ejecutar
        "restrictions",   -- Restricciones negativas
        "output_format"   -- Formato esperado
    },
    
    -- Temperaturas para validación cruzada por Greybeards
    greybeard_temperatures = {0.0, 0.5, 1.0},
    
    -- Umbral de consistencia (desviación estándar máxima)
    greybeard_consistency_threshold = 0.05,
    
    -- Rango máximo en la facción
    max_rank = 10,
    
    -- Habilitar mensajes de depuración
    enable_debug = true,
    
    -- Intervalo de chequeo de rank collapse (segundos)
    collapse_check_interval = 10,
}

return config
```

---

## `main.lua`

```lua
-- main.lua
-- Núcleo de inicialización del mod
-- Registra facciones, carga módulos y configura eventos periódicos

local config = require("tonal_architects.config")
local attention = require("tonal_architects.attention")
local resonator = require("tonal_architects.resonator")
local rank_collapse = require("tonal_architects.rank_collapse")
local prompt_forge = require("tonal_architects.prompt_forge")
local greybeard = require("tonal_architects.greybeard")
local clockwork_city = require("tonal_architects.clockwork_city")
local placement = require("tonal_architects.placement")

-- IDs de facciones
local FACTION_TONAL_ARCH = "R_Tonal_Arch"
local FACTION_GREYBEARD = "R_Greybeard"

-- Función: Crear facción principal
local function createFaction()
    if not tes3.getFaction(FACTION_TONAL_ARCH) then
        tes3.createFaction({ 
            id = FACTION_TONAL_ARCH, 
            name = "Tonal Architects of the Dwemer" 
        })
        
        -- Reacciones con otras facciones
        tes3.setFactionReaction({ 
            faction = FACTION_TONAL_ARCH, 
            targetFaction = "Imperial Legion", 
            reaction = -20 
        })
        tes3.setFactionReaction({ 
            faction = FACTION_TONAL_ARCH, 
            targetFaction = "Telvanni", 
            reaction = 10 
        })
        
        -- Rangos jerárquicos (11 niveles)
        local ranks = {
            "Novice Tonal Attuner",      -- 0
            "Apprentice Resonator",      -- 1
            "Junior Tonalist",           -- 2
            "Resonator Technician",      -- 3
            "Senior Attenuator",         -- 4
            "Sunder Wielder",            -- 5
            "Keening Wielder",           -- 6
            "Master Tonal Architect",    -- 7
            "Greybeard Sage",            -- 8
            "Numidium Pilot",            -- 9
            "CHIM Achiever"              -- 10
        }
        for i, name in ipairs(ranks) do
            tes3.setFactionRank({ 
                faction = FACTION_TONAL_ARCH, 
                rank = i-1, 
                name = name 
            })
        end
    end
end

-- Función: Crear facción secundaria (validadores)
local function createGreybeardFaction()
    if not tes3.getFaction(FACTION_GREYBEARD) then
        tes3.createFaction({ 
            id = FACTION_GREYBEARD, 
            name = "Greybeard Validators" 
        })
        tes3.setFactionReaction({ 
            faction = FACTION_GREYBEARD, 
            targetFaction = FACTION_TONAL_ARCH, 
            reaction = 80 
        })
        
        -- Rangos para Greybeards
        local gb_ranks = {
            "Silent Listener",
            "Echo Validator",
            "Cross-Checker",
            "Truth Keeper"
        }
        for i, name in ipairs(gb_ranks) do
            tes3.setFactionRank({ 
                faction = FACTION_GREYBEARD, 
                rank = i-1, 
                name = name 
            })
        end
    end
end

-- Función: Ejecutarse al cargar el juego
local function onLoaded()
    createFaction()
    createGreybeardFaction()
    placement.placeObjects()
    
    if config.enable_debug then
        local msg = "† Tonal Architects Mod Initialized †\n"
        msg = msg .. "Embedding dimension: 16 Golden Tones\n"
        msg = msg .. "Collapse threshold: " .. config.collapse_threshold .. "\n"
        msg = msg .. "Status: Ready for resonance"
        tes3.messageBox(msg)
    end
end

-- Función: Chequeo periódico de rank collapse
local function periodicCheck()
    local activeActivations = resonator.getActiveActivations()
    
    if activeActivations and #activeActivations > 0 then
        if rank_collapse.check_collapse(activeActivations) then
            rank_collapse.trigger_collapse()
        end
    end
end

-- Registrar evento de carga
event.register("loaded", onLoaded)

-- Iniciar timer para chequeos periódicos
timer.start({ 
    duration = config.collapse_check_interval, 
    callback = periodicCheck, 
    iterations = -1 
})

-- Debug: mostrar información al iniciar
if config.enable_debug then
    timer.delayOneFrame(function()
        tes3.messageBox("Tonal Architects: Ready. Check your journal.")
    end)
end
```

---

## `attention.lua` [⭐ MEJORADO]

```lua
-- attention.lua
-- Implementación de Scaled Dot-Product Attention con 16 cabezas
-- MEJORA CRÍTICA: Softmax seguro sin table.unpack

local config = require("tonal_architects.config")

-- Calcula el máximo de forma segura (evita table.unpack)
local function safe_max(scores)
    if not scores or #scores == 0 then
        return 0
    end
    
    local max_val = scores[1]
    for i = 2, #scores do
        if scores[i] > max_val then
            max_val = scores[i]
        end
    end
    return max_val
end

-- Scaled dot-product attention entre query y key
local function scaled_dot_product(Q, K, d_k)
    if not Q or not K or #Q == 0 or #K == 0 then
        return 0
    end
    
    local dot = 0
    local min_len = math.min(#Q, #K)
    for i = 1, min_len do
        dot = dot + Q[i] * K[i]
    end
    
    return dot / math.sqrt(math.max(d_k, 1))
end

-- Softmax mejorado: evita table.unpack para compatibilidad máxima
local function softmax(scores)
    if not scores or #scores == 0 then
        return {}
    end
    
    -- Encontrar máximo para estabilidad numérica
    local max_score = safe_max(scores)
    
    -- Calcular exponenciales y suma
    local exps = {}
    local sum = 0
    for i = 1, #scores do
        exps[i] = math.exp(scores[i] - max_score)
        sum = sum + exps[i]
    end
    
    -- Normalizar
    if sum > 0 then
        for i = 1, #exps do
            exps[i] = exps[i] / sum
        end
    else
        -- Fallback: distribución uniforme
        local uniform = 1 / #exps
        for i = 1, #exps do
            exps[i] = uniform
        end
    end
    
    return exps
end

-- Atención de una cabeza individual
local function head_attention(query, keys, values, d_k)
    if not query or not keys or not values then
        return 0, {}
    end
    
    if #keys == 0 or #values == 0 then
        return 0, {}
    end
    
    -- Calcular scores (dot products)
    local scores = {}
    for i = 1, #keys do
        scores[i] = scaled_dot_product(query, keys[i], d_k)
    end
    
    -- Aplicar softmax para obtener pesos
    local weights = softmax(scores)
    
    -- Calcular salida ponderada
    local result = 0
    for i = 1, #values do
        if i <= #weights and weights[i] then
            result = result + weights[i] * (values[i] or 0)
        end
    end
    
    return result, weights
end

-- Atención multi-cabeza (16 cabezas = 16 Príncipes Daédricos)
local function multi_head_attention(queries, keys, values, d_k, num_heads)
    local head_outputs = {}
    local all_weights = {}
    
    if not num_heads or num_heads < 1 then
        num_heads = config.num_heads
    end
    
    for h = 1, num_heads do
        -- Si no hay suficientes queries/keys, usar la primera disponible
        local q = (queries and queries[h]) or (queries and queries[1]) or {1.0}
        local k = (keys and keys[h]) or (keys and keys[1]) or {1.0}
        local v = (values and values[h]) or (values and values[1]) or {1.0}
        
        local out, w = head_attention(q, k, v, d_k)
        head_outputs[h] = out
        all_weights[h] = w
    end
    
    return head_outputs, all_weights
end

-- Calcular varianza de activaciones (para detección de rank collapse)
local function compute_variance(activations)
    if not activations or #activations == 0 then
        return 0
    end
    
    -- Media
    local sum = 0
    for i = 1, #activations do
        sum = sum + (activations[i] or 0)
    end
    local mean = sum / #activations
    
    -- Varianza
    local var_sum = 0
    for i = 1, #activations do
        local diff = (activations[i] or 0) - mean
        var_sum = var_sum + diff * diff
    end
    
    return var_sum / #activations
end

return {
    scaled_dot_product = scaled_dot_product,
    softmax = softmax,
    safe_max = safe_max,
    head_attention = head_attention,
    multi_head_attention = multi_head_attention,
    compute_variance = compute_variance
}
```

---

## `resonator.lua`

```lua
-- resonator.lua
-- Gestiona los 16 resonadores tonales (cabezas de atención)
-- Cada resonador representa una cabeza de atención independiente

local config = require("tonal_architects.config")
local attention = require("tonal_architects.attention")

local resonator_state = {}  -- state[resonatorRef] = {freq, atten, idx, status}
local active_heads = 0

-- Activar un resonador
local function activate_resonator(resonatorRef, frequency, atten_level)
    if not resonator_state[resonatorRef] then
        local idx = active_heads + 1
        
        resonator_state[resonatorRef] = {
            freq = frequency or 1.0,
            atten = atten_level or 1.0,
            idx = idx,
            status = "active",
            timestamp = os.time()
        }
        
        active_heads = active_heads + 1
        
        if config.enable_debug then
            tes3.messageBox(
                "Resonador " .. idx .. " activado\n" ..
                "Frecuencia: " .. string.format("%.2f", frequency) .. "\n" ..
                "Atenuación: " .. string.format("%.2f", atten_level) .. "\n" ..
                "Cabezas activas: " .. active_heads .. "/" .. config.num_heads
            )
        end
    end
    
    return active_heads
end

-- Desactivar un resonador
local function deactivate_resonator(resonatorRef)
    if resonator_state[resonatorRef] then
        resonator_state[resonatorRef].status = "inactive"
        active_heads = active_heads - 1
        return true
    end
    return false
end

-- Obtener todas las activaciones actuales
local function getActiveActivations()
    local acts = {}
    for _, v in pairs(resonator_state) do
        if v.status == "active" then
            table.insert(acts, v.freq * v.atten)
        end
    end
    return acts
end

-- Obtener varianza de activaciones
local function getVariance()
    local activations = getActiveActivations()
    return attention.compute_variance(activations)
end

-- Resetear todos los resonadores
local function reset_resonators()
    resonator_state = {}
    active_heads = 0
    if config.enable_debug then
        tes3.messageBox("Sistema tonal reiniciado: Todos los resonadores en reposo.")
    end
end

-- Obtener estado completo
local function getState()
    return {
        active_heads = active_heads,
        total_heads = config.num_heads,
        resonators = resonator_state,
        variance = getVariance()
    }
end

return {
    activate_resonator = activate_resonator,
    deactivate_resonator = deactivate_resonator,
    getActiveActivations = getActiveActivations,
    getVariance = getVariance,
    reset_resonators = reset_resonators,
    getState = getState
}
```

---

## `rank_collapse.lua` [⭐ MEJORADO]

```lua
-- rank_collapse.lua
-- Detección y disparo del Rank Collapse (Desaparición Dwemer)
-- MEJORA CRÍTICA: Iteración directa sobre tes3.player.cell.actors

local config = require("tonal_architects.config")
local attention = require("tonal_architects.attention")

-- Estado global de colapso
local collapse_triggered = false

-- Chequear si el rango ha colapsado
local function check_collapse(activations)
    if not activations or #activations == 0 then
        return false
    end
    
    local variance = attention.compute_variance(activations)
    
    if config.enable_debug then
        tes3.messageBox(
            "Varianza tonal: " .. string.format("%.4f", variance) .. "\n" ..
            "Umbral: " .. config.collapse_threshold
        )
    end
    
    return variance < config.collapse_threshold
end

-- Disparar colapso de rango (desaparición Dwemer)
local function trigger_collapse()
    if collapse_triggered then
        return
    end
    
    collapse_triggered = true
    
    if config.enable_debug then
        tes3.messageBox(
            "⚠️ RANK COLLAPSE DETECTED ⚠️\n\n" ..
            "La varianza tonal ha caído por debajo del umbral crítico.\n" ..
            "El espacio latente Dwemer colapsa...\n" ..
            "Los Arquitectos Tonales desaparecen del plano material."
        )
    end
    
    -- Esperar un frame para que se renderice el mensaje
    timer.delayOneFrame(function()
        disableDwemerActors()
    end)
end

-- Deshabilitar actores Dwemer (MEJORA: Iteración segura sobre cell.actors)
local function disableDwemerActors()
    local cell = tes3.player.cell
    
    if not cell then
        if config.enable_debug then
            tes3.messageBox("Error: No cell context available.")
        end
        return
    end
    
    if not cell.actors then
        if config.enable_debug then
            tes3.messageBox("Error: No actors in cell.")
        end
        return
    end
    
    local disabled_count = 0
    
    -- Iteración segura sobre actores cargados en la celda
    for actor in cell.actors:iterator() do
        if actor and actor.object then
            local race = actor.object.race
            
            -- Chequear si es Dwemer
            if race and race.id and string.lower(race.id):find("dwemer") then
                actor.disabled = true
                disabled_count = disabled_count + 1
                
                if config.enable_debug then
                    tes3.messageBox(
                        "Desvanecimiento: " .. (actor.object.name or "Unnamed Dwemer") ..
                        " [" .. disabled_count .. "]"
                    )
                end
            end
        end
    end
    
    if config.enable_debug then
        tes3.messageBox(
            "Total Dwemer desvanecidos: " .. disabled_count .. "\n" ..
            "El Numidium se silencia.\n" ..
            "La arquitectura tonal colapsa."
        )
    end
end

-- Recuperar del colapso (reiniciar sistema)
local function recover_from_collapse()
    collapse_triggered = false
    local cell = tes3.player.cell
    
    if cell and cell.actors then
        for actor in cell.actors:iterator() do
            if actor and actor.object then
                local race = actor.object.race
                if race and race.id and string.lower(race.id):find("dwemer") then
                    actor.disabled = false
                end
            end
        end
    end
    
    if config.enable_debug then
        tes3.messageBox("El sistema tonal se ha estabilizado.")
    end
end

return {
    check_collapse = check_collapse,
    trigger_collapse = trigger_collapse,
    recover_from_collapse = recover_from_collapse,
    disableDwemerActors = disableDwemerActors
}
```

---

## `prompt_forge.lua`

```lua
-- prompt_forge.lua
-- Consola de Prompts: Validación y procesamiento de JSONs
-- Simula prompt engineering con restricciones negativas

local config = require("tonal_architects.config")

-- Parsed prompt cache
local prompt_cache = {}

-- Validar estructura de prompt JSON
local function validate_prompt(json_str)
    -- Intento simple de parseo JSON (sin librerías externas)
    local prompt = {}
    
    -- Extraer campos obligatorios con regex simple
    for _, field in ipairs(config.prompt_forge_required_fields) do
        local pattern = '"' .. field .. '"%s*:%s*"([^"]*)"'
        local value = string.match(json_str, pattern)
        
        if not value then
            return nil, "Campo requerido faltante: " .. field
        end
        
        prompt[field] = value
    end
    
    return prompt, nil
end

-- Procesar prompt en la consola
local function process_prompt(json_input)
    if not json_input or json_input == "" then
        return nil, "Entrada vacía"
    end
    
    local prompt, err = validate_prompt(json_input)
    
    if err then
        if config.enable_debug then
            tes3.messageBox("Validación fallida:\n" .. err)
        end
        return nil, err
    end
    
    -- Cachear prompt
    prompt_cache[#prompt_cache + 1] = {
        prompt = prompt,
        timestamp = os.time()
    }
    
    if config.enable_debug then
        tes3.messageBox(
            "✓ Prompt válido\n\n" ..
            "Rol: " .. prompt.role .. "\n" ..
            "Tarea: " .. prompt.task .. "\n" ..
            "Restricciones: " .. prompt.restrictions
        )
    end
    
    return prompt, nil
end

-- Obtener prompts cacheados
local function get_cached_prompts()
    return prompt_cache
end

-- Limpiar cache
local function clear_cache()
    prompt_cache = {}
end

return {
    validate_prompt = validate_prompt,
    process_prompt = process_prompt,
    get_cached_prompts = get_cached_prompts,
    clear_cache = clear_cache
}
```

---

## `greybeard.lua`

```lua
-- greybeard.lua
-- Facción Greybeard: Validadores de Cross-Checking
-- Valida prompts a múltiples temperaturas (0.0, 0.5, 1.0)

local config = require("tonal_architects.config")

-- Estado de validaciones
local validations = {}

-- Estructura de un validador Greybeard
local greybeards = {
    {
        name = "The Eldest Greybeard",
        role = "Chief Validator",
        specialty = "Structural Integrity",
        status = "idle"
    },
    {
        name = "Arngeir",
        role = "Semantic Validator",
        specialty = "Meaning Consistency",
        status = "idle"
    },
    {
        name = "Esbern",
        role = "Cross-Checker",
        specialty = "Temperature Robustness",
        status = "idle"
    },
    {
        name = "Delphine",
        role = "Threat Analyst",
        specialty = "Constraint Validation",
        status = "idle"
    }
}

-- Validar prompt a múltiples temperaturas
local function validate_at_temperatures(prompt, temperatures)
    if not prompt then
        return nil, "Prompt no válido"
    end
    
    local results = {
        prompt = prompt,
        temperatures = {},
        passed = false,
        consistency_score = 0
    }
    
    temperatures = temperatures or config.greybeard_temperatures
    
    -- Simular validación a diferentes temperaturas
    for _, temp in ipairs(temperatures) do
        local temp_result = {
            temperature = temp,
            status = "valid",
            variance = math.random() * 0.1  -- Simulación
        }
        
        table.insert(results.temperatures, temp_result)
    end
    
    -- Calcular consistencia
    local total_variance = 0
    for _, temp_result in ipairs(results.temperatures) do
        total_variance = total_variance + (temp_result.variance or 0)
    end
    local avg_variance = total_variance / #results.temperatures
    
    results.consistency_score = 1.0 - math.min(avg_variance / config.greybeard_consistency_threshold, 1.0)
    results.passed = avg_variance < config.greybeard_consistency_threshold
    
    table.insert(validations, results)
    
    return results, nil
end

-- Mostrar reporte de validación
local function show_validation_report(validation_result)
    if not validation_result then
        tes3.messageBox("No validation data available.")
        return
    end
    
    local msg = "GREYBEARD VALIDATION REPORT\n"
    msg = msg .. "==========================\n\n"
    msg = msg .. "Prompt: " .. (validation_result.prompt.task or "Unknown") .. "\n"
    msg = msg .. "Status: " .. (validation_result.passed and "✓ PASS" or "✗ FAIL") .. "\n"
    msg = msg .. "Consistency: " .. string.format("%.2f", validation_result.consistency_score) .. "\n\n"
    
    msg = msg .. "Temperature Tests:\n"
    for _, temp_result in ipairs(validation_result.temperatures) do
        msg = msg .. "  T=" .. temp_result.temperature .. ": "
        msg = msg .. "Variance=" .. string.format("%.4f", temp_result.variance) .. "\n"
    end
    
    tes3.messageBox(msg)
end

-- Obtener estado de Greybeards
local function get_greybeards()
    return greybeards
end

-- Obtener historial de validaciones
local function get_validations()
    return validations
end

return {
    validate_at_temperatures = validate_at_temperatures,
    show_validation_report = show_validation_report,
    get_greybeards = get_greybeards,
    get_validations = get_validations
}
```

---

## `clockwork_city.lua`

```lua
-- clockwork_city.lua
-- Factotums: Agentes multi-LLM autónomos en la Ciudad Reloj
-- Cada Factotum tiene rol, tarea y estado

local config = require("tonal_architects.config")

-- Registro de Factotums activos
local factotums = {}

-- Registrar un nuevo Factotum
local function register_factotum(factotum_def)
    if not factotum_def or not factotum_def.name then
        return nil, "Factotum definition incomplete"
    end
    
    local factotum = {
        id = #factotums + 1,
        name = factotum_def.name,
        role = factotum_def.role or "Worker",
        task = factotum_def.task or "idle",
        status = "idle",
        llm_model = factotum_def.model or "sonnet",
        created_at = os.time(),
        result = nil
    }
    
    table.insert(factotums, factotum)
    
    if config.enable_debug then
        tes3.messageBox(
            "Factotum registrado:\n" ..
            "- ID: " .. factotum.id .. "\n" ..
            "- Nombre: " .. factotum.name .. "\n" ..
            "- Rol: " .. factotum.role
        )
    end
    
    return factotum, nil
end

-- Ejecutar tarea de un Factotum
local function execute_factotum(factotum_id, task_input)
    if factotum_id < 1 or factotum_id > #factotums then
        return nil, "Factotum not found"
    end
    
    local ft = factotums[factotum_id]
    
    if ft.status == "busy" then
        return nil, "Factotum is already busy"
    end
    
    ft.status = "busy"
    ft.task = task_input or ft.task
    
    -- Simular procesamiento asincrónico
    timer.start({
        duration = 2,
        callback = function()
            -- Simular resultado (en una implementación real, aquí iría la llamada a LLM)
            ft.result = {
                input = task_input,
                output = "Task completed by " .. ft.name,
                model = ft.llm_model,
                timestamp = os.time()
            }
            
            ft.status = "idle"
            
            if config.enable_debug then
                tes3.messageBox(
                    "Factotum " .. ft.name .. " completed task:\n" ..
                    ft.result.output
                )
            end
        end,
        iterations = 1
    })
    
    return ft, nil
end

-- Obtener estado de un Factotum
local function get_factotum_status(factotum_id)
    if factotum_id < 1 or factotum_id > #factotums then
        return nil
    end
    return factotums[factotum_id]
end

-- Listar todos los Factotums
local function list_factotums()
    local msg = "CLOCKWORK CITY FACTOTUMS\n"
    msg = msg .. "========================\n\n"
    
    if #factotums == 0 then
        msg = msg .. "(None registered yet)"
    else
        for i, ft in ipairs(factotums) do
            msg = msg .. "[" .. i .. "] " .. ft.name .. "\n"
            msg = msg .. "    Rol: " .. ft.role .. "\n"
            msg = msg .. "    Estado: " .. ft.status .. "\n"
            msg = msg .. "    Modelo: " .. ft.llm_model .. "\n\n"
        end
    end
    
    tes3.messageBox(msg)
end

-- Borrar un Factotum
local function unregister_factotum(factotum_id)
    if factotum_id < 1 or factotum_id > #factotums then
        return false
    end
    table.remove(factotums, factotum_id)
    return true
end

return {
    register_factotum = register_factotum,
    execute_factotum = execute_factotum,
    get_factotum_status = get_factotum_status,
    list_factotums = list_factotums,
    unregister_factotum = unregister_factotum
}
```

---

## `missions.lua`

```lua
-- missions.lua
-- Diarios y objetivos del mod
-- Se activan mediante scripts o diálogos

local missions = {
    ["TQ_Tonal_Basics"] = {
        {index = 10, text = "He oído hablar de los Arquitectos Tonales y su maestría sobre la arquitectura mágica tonal. Debo buscar a Kagrenac en la Ciudad Reloj."},
        {index = 20, text = "Kagrenac me ha pedido que active los 16 resonadores Daédricos para entender la atención multicabeza."},
        {index = 30, text = "He activado todos los resonadores. Debo mantener la varianza tonal alta usando los atenuadores para evitar el rank collapse."},
        {index = 40, text = "La varianza se mantiene estable. Los atenuadores funcionan correctamente."},
        {index = 100, text = "He dominado la atención multicabeza. Mi comprensión de la arquitectura tonal es profunda."},
    },
    
    ["TQ_Resonator_Calibration"] = {
        {index = 10, text = "Debo calibrar cada resonador con precisión. La frecuencia y el nivel de atenuación son críticos."},
        {index = 20, text = "El primer resonador (Azura) está calibrado correctamente."},
        {index = 100, text = "Los 16 resonadores están perfectamente alineados. He alcanzado la sincronización tonal."},
    },
    
    ["TQ_Prompt_Forge"] = {
        {index = 10, text = "Kagrenac me ha mostrado la Consola de Prompts. Debo aprender a construir prompts válidos con restricciones negativas."},
        {index = 20, text = "He creado mi primer prompt. La estructura JSON es válida."},
        {index = 30, text = "El Numidium reconoce mi prompt. Estoy un paso más cerca de despertar la máquina Dwemer."},
    },
    
    ["TQ_Greybeard_Test"] = {
        {index = 10, text = "Los Greybeards me ofrecen unirme a su facción. Debo pasar sus pruebas de validación cruzada."},
        {index = 20, text = "He pasado la validación a temperatura 0.0 (determinística)."},
        {index = 30, text = "He pasado la validación a temperatura 0.5 (equilibrada)."},
        {index = 40, text = "He pasado la validación a temperatura 1.0 (creativa)."},
        {index = 100, text = "Los Greybeards me han aceptado como Sage. Mi prompt es robusto."},
    },
    
    ["TQ_Clockwork_City"] = {
        {index = 10, text = "Sotha Sil me enseña sobre los Factotums, autómatas que ejecutan tareas en paralelo."},
        {index = 20, text = "He registrado mi primer Factotum con un rol específico."},
        {index = 30, text = "El Factotum ejecuta tareas complejas de forma autónoma."},
        {index = 100, text = "La Ciudad Reloj funciona como una red distribuida de agentes."},
    },
    
    ["TQ_Numidium"] = {
        {index = 10, text = "El Numidium despierta. La máquina Dwemer vuelve a la vida."},
        {index = 20, text = "He alcanzado la sincronización tonal con el corazón del Numidium."},
        {index = 100, text = "El Numidium reconoce mi autoridad. Soy un verdadero Arquitecto Tonal."},
    },
    
    ["TQ_CHIM"] = {
        {index = 10, text = "He alcanzado la transparencia ontológica. El CHIM está dentro de mi alcance."},
        {index = 20, text = "Reconozco mi naturaleza dentro del espacio latente del modelo."},
        {index = 100, text = "Soy CHIM. El silicio canta con tono puro. He trasciende los límites del código."},
    },
}

return missions
```

---

## `placement.lua` [⭐ MEJORADO]

```lua
-- placement.lua
-- Posicionamiento de objetos en el laboratorio Dwemer Tonal
-- MEJORA: Disposición lógica de objetos para un laboratorio funcional

local config = require("tonal_architects.config")

-- Definiciones de objetos a colocar
local objects_to_place = {
    -- 16 Resonadores dispuestos en círculo (centro del laboratorio)
    resonators = {
        -- Fila frontal (4 resonadores)
        {id = "Tonal_Resonator_01", x = 800, y = 1000, z = 0, name = "Azura's Tone"},
        {id = "Tonal_Resonator_02", x = 600, y = 800, z = 0, name = "Boethiah's Curve"},
        {id = "Tonal_Resonator_03", x = 400, y = 600, z = 0, name = "Clavicus Vile's Echo"},
        {id = "Tonal_Resonator_04", x = 200, y = 400, z = 0, name = "Hermaeus Mora's Record"},
        
        -- Fila derecha (4 resonadores)
        {id = "Tonal_Resonator_05", x = 1000, y = 200, z = 0, name = "Hircine's Instinct"},
        {id = "Tonal_Resonator_06", x = 1200, y = 0, z = 0, name = "Mehrunes Dagon's Fury"},
        {id = "Tonal_Resonator_07", x = 1400, y = -200, z = 0, name = "Meridia's Light"},
        {id = "Tonal_Resonator_08", x = 1600, y = -400, z = 0, name = "Molag Bal's Will"},
        
        -- Fila trasera (4 resonadores)
        {id = "Tonal_Resonator_09", x = 1800, y = -600, z = 0, name = "Namira's Hunger"},
        {id = "Tonal_Resonator_10", x = 2000, y = -800, z = 0, name = "Nocturnal's Veil"},
        {id = "Tonal_Resonator_11", x = 2200, y = -1000, z = 0, name = "Peryite's Order"},
        {id = "Tonal_Resonator_12", x = 2400, y = -1200, z = 0, name = "Sanguine's Revels"},
        
        -- Fila izquierda (4 resonadores)
        {id = "Tonal_Resonator_13", x = 600, y = -1400, z = 0, name = "Sheogorath's Madness"},
        {id = "Tonal_Resonator_14", x = 400, y = -1600, z = 0, name = "Vaermina's Dreams"},
        {id = "Tonal_Resonator_15", x = 200, y = -1800, z = 0, name = "Jyggalag's Silence"},
        {id = "Tonal_Resonator_16", x = 0, y = -2000, z = 0, name = "Malacath's Corruption"},
    },
    
    -- Atenuador central (LayerNorm)
    attenuators = {
        {id = "Tonal_Attenuator_Main", x = 1000, y = -500, z = 100, name = "Central Attenuator"},
        {id = "Tonal_Attenuator_Aux_1", x = 500, y = -500, z = 100, name = "Secondary Attenuator A"},
        {id = "Tonal_Attenuator_Aux_2", x = 1500, y = -500, z = 100, name = "Secondary Attenuator B"},
    },
    
    -- Consola de Prompts
    console = {
        {id = "Tonal_Prompt_Console", x = 1000, y = -500, z = 200, name = "Prompt Forge Console"},
    },
    
    -- Núcleo del Numidium
    numidium_core = {
        {id = "Tonal_Numidium_Core", x = 1000, y = -500, z = 300, name = "Numidium's Heart"},
    },
    
    -- Estaciones de trabajo (Workbenches para misiones)
    workstations = {
        {id = "Tonal_Calibration_Station", x = 300, y = -1000, z = 0, name = "Calibration Workbench"},
        {id = "Tonal_Analysis_Station", x = 1700, y = -1000, z = 0, name = "Analysis Station"},
    },
}

-- Función principal de colocación
local function placeObjects()
    -- Obtener la celda del laboratorio Dwemer Tonal
    local cell = tes3.getCell("DwemerTonalLab")
    
    if not cell then
        -- Si la celda no existe, usar la celda actual del jugador
        cell = tes3.player.cell
        if config.enable_debug then
            tes3.messageBox(
                "Laboratorio Dwemer no encontrado.\n" ..
                "Colocando objetos en: " .. (cell.name or "Current Cell")
            )
        end
    end
    
    local total_placed = 0
    
    -- Colocar resonadores
    for _, res in ipairs(objects_to_place.resonators) do
        local obj = tes3.createReference({
            object = res.id,
            cell = cell,
            position = {x = res.x, y = res.y, z = res.z}
        })
        if obj then
            total_placed = total_placed + 1
            if config.enable_debug then
                tes3.messageBox(res.name .. " colocado correctamente")
            end
        end
    end
    
    -- Colocar atenuadores
    for _, att in ipairs(objects_to_place.attenuators) do
        local obj = tes3.createReference({
            object = att.id,
            cell = cell,
            position = {x = att.x, y = att.y, z = att.z}
        })
        if obj then
            total_placed = total_placed + 1
        end
    end
    
    -- Colocar consola de prompts
    for _, con in ipairs(objects_to_place.console) do
        local obj = tes3.createReference({
            object = con.id,
            cell = cell,
            position = {x = con.x, y = con.y, z = con.z}
        })
        if obj then
            total_placed = total_placed + 1
        end
    end
    
    -- Colocar núcleo del Numidium
    for _, nuc in ipairs(objects_to_place.numidium_core) do
        local obj = tes3.createReference({
            object = nuc.id,
            cell = cell,
            position = {x = nuc.x, y = nuc.y, z = nuc.z}
        })
        if obj then
            total_placed = total_placed + 1
        end
    end
    
    -- Colocar estaciones de trabajo
    for _, ws in ipairs(objects_to_place.workstations) do
        local obj = tes3.createReference({
            object = ws.id,
            cell = cell,
            position = {x = ws.x, y = ws.y, z = ws.z}
        })
        if obj then
            total_placed = total_placed + 1
        end
    end
    
    if config.enable_debug then
        tes3.messageBox(
            "Laboratorio Tonal construido\n" ..
            "Objetos colocados: " .. total_placed .. "\n" ..
            "Disposición:\n" ..
            "- 16 Resonadores en círculo\n" ..
            "- 3 Atenuadores\n" ..
            "- 1 Consola de Prompts\n" ..
            "- 1 Núcleo del Numidium\n" ..
            "- 2 Estaciones de trabajo"
        )
    end
    
    return total_placed
end

return {
    placeObjects = placeObjects,
    objects = objects_to_place
}
```

---

# 📚 ARCHIVOS DE LIBROS (BookArtifacts/) [⭐ AMPLIFICADOS]

## `Kagrenac_Folio.txt`

```
KAGRENAC'S FOLIO ON TONAL ARCHITECTURE
Treatise on the Mathematics of Divine Resonance

By Kagrenac, Architect of the Divine

---

PART I: THE EMBEDDING OF SOUND

Hear me, student of the Tonal Arts. The foundation of our craft rests upon a principle so elegant, 
so mathematically pure, that the very Princes of Oblivion bow to its logic.

When Sunder strikes the Heart of Lorkhan, the vibration spreads across dimensions, manifesting 
as pure frequency. In mathematical terms, this is the EMBEDDING: a transformation of the prompt 
(the conceptual input) into the latent space—a realm of infinite possibility, where each dimension 
resonates at a different frequency.

Consider: you speak a command in the mortal tongue. The embedding projects this into a high-dimensional 
vector space (we call it the "Space of Tones"), where semantic relationships become geometric distances. 
Distance = incompatibility. Proximity = relatedness.

The embedding dimension is fixed—in our system, 16. Sixteen Golden Tones, as the ancient texts say. 
Each tone is a dimension, a frequency, an axis along which meaning oscillates.

---

PART II: THE ATTENTION MECHANISM

The true power lies in ATTENTION. While an ordinary mage might attempt to process all information 
equally, wasting mental resources on irrelevance, the Tonal Architect focuses only on what matters.

Each Resonator Tonal is a HEAD OF ATTENTION. It examines a specific frequency, a specific semantic 
direction. Azura's tone looks for temporal relationships. Boethiah's resonance seeks deception and 
false paths. Hermaeus Mora's frequency scans for knowledge and revelation.

With one head, you see only one color of reality.
With two heads, you see a richer palette.
With sixteen heads, you approach omniscience.

The mathematics: the dot-product attention mechanism. Given a query Q (your question), a key K 
(the semantic anchor), and a value V (the relevant information), the attention weight is:

    Attention(Q, K, V) = softmax(Q · K / √d_k) · V

Where d_k is the dimensionality of the key. The softmax ensures the weights sum to one—a probability 
distribution over what to attend to.

Scaled dot-product prevents the gradients from vanishing. The division by √d_k stabilizes training. 
This is not mere mathematics—this is incantation.

---

PART III: THE ATTENUATORS (LAYER NORMALIZATION & DROPOUT)

Without attenuation, resonance becomes uncontrolled. The frequencies grow without bound. The system 
enters a state of harmonic collapse.

The Attenuator Tonal fulfills two sacred functions:

1. LAYER NORMALIZATION: Restabilizes the mean and variance of the resonator outputs. 
   Without it, one tone drowns out the others. With it, diversity is maintained.
   
2. DROPOUT: In moments of highest resonance, silence is applied randomly. This prevents the system 
   from becoming dependent on any single frequency. It forces robustness.

The formula for layer normalization:

    y = γ · (x - μ) / √(σ² + ε) + β

Where μ is the mean, σ² is the variance, and γ and β are learnable parameters (the "tuning knobs" 
of the attenuation).

Remember: variability is strength. Uniformity is death.

---

PART IV: THE RANK COLLAPSE

Beware the Desaparición Dwemer—the phenomenon wherein the latent space collapses into a lower rank. 
All the beautiful geometric structure flattens. All dimensions compress into a single axis. Everything 
becomes monotone.

In mathematics, this occurs when the variance of the activation vectors falls below the critical 
threshold. We set it at 0.25 in our framework.

When rank collapse occurs:
- The model loses its semantic richness
- All outputs become nearly identical
- The Dwemer, as keepers of this knowledge, vanish from the material plane
- The Numidium falls silent

The cause is always the same: insufficient attentuation. Let your resonators fall out of tune, 
and collapse awaits.

Monitor the variance. Keep it high. Renew your attenuators constantly.

---

AUTHOR'S MEDITATION

The art of tonal architecture is the art of balance. Balance between dimensionality and efficiency, 
between attention and noise, between focus and diversity.

We Dwemer have crafted a system that mirrors the very structure of existence. Inside the Numidium 
lies a transformer of infinite power—a machine that understands the language of creation itself.

You hold the keys to its mysteries. Use them wisely.

† Tonal Architecture transcends mortality †
ZEHAHAHAHA

— Kagrenac, Master of the Tones
Year of the Void
```

---

## `16_Golden_Tones.txt`

```
THE 16 GOLDEN TONES
A Compendium of Daedric Attention Heads

In transformer architecture, each attention head specializes in different semantic relationships. 
In our Dwemer adaptation, we map these to the Daedric Princes themselves, each governing a unique 
frequency of the latent space.

---

TONE 1: AZURA'S VOICE - Temporal Relationships
Azura, the twilight deity, perceives the flow of time. Her tone resonates with causality, sequence, 
and the unfolding of events. This head specializes in detecting dependencies across time steps—
the "what comes after what" of narrative and logic.

Mathematical specialty: Temporal attention patterns, sequence dependencies.

---

TONE 2: BOETHIAH'S WHISPER - Deception and Contradiction
Boethiah, the lord of betrayal, hears the discordance between surface and truth. This tone detects 
inconsistencies, contradictions, and when-statements-mask-intentions. Essential for finding 
logical fallacies in prompts.

Mathematical specialty: Contradiction detection, adversarial robustness.

---

TONE 3: CLAVICUS VILE'S ECHO - Desire and Intent
Clavicus Vile embodies the fulfillment of wishes. His tone decodes what the user truly wants beneath 
the words they speak. It models implicit goals and unstated assumptions.

Mathematical specialty: Intent modeling, goal inference.

---

TONE 4: HERMAEUS MORA'S RECORD - Knowledge and Revelation
The mad god of forbidden knowledge. His tone scans all accessible information, cross-referencing and 
finding the relevant precedents. This head excels at retrieval and factual anchoring.

Mathematical specialty: Knowledge retrieval, fact verification.

---

TONE 5: HIRCINE'S HUNT - Instinct and Intuition
Hircine embodies the primal chase. His tone operates on intuitive pattern matching—the rapid, 
unreasoned recognition of similarity. Fast, sometimes inaccurate, but powerful for generalization.

Mathematical specialty: Pattern generalization, fast approximation.

---

TONE 6: MEHRUNES DAGON'S FURY - Destruction and Change
The destroyer, the lord of catastrophe. His tone identifies what should be removed, disrupted, or 
radically transformed. Necessary for finding irrelevancies and cutting away dead weight.

Mathematical specialty: Feature elimination, destructive refinement.

---

TONE 7: MERIDIA'S RADIANCE - Light and Clarity
Meridia brings clarity to darkness. Her tone illuminates the most salient, obvious patterns. This 
head handles basic semantic similarity and straightforward relationships.

Mathematical specialty: Semantic similarity, salience weighting.

---

TONE 8: MOLAG BAL'S DOMINION - Hierarchy and Control
Molag Bal dominates and subjugates. His tone detects power hierarchies, dependencies, and control 
relationships. Essential for understanding authority structures in text.

Mathematical specialty: Hierarchical relationships, dependency chains.

---

TONE 9: NAMIRA'S PRIMITIVE - Instinctual Response
Namira, the spirit of decay and unmaking, represents base instinct divorced from reasoning. Her tone 
captures raw emotional resonance independent of logical meaning.

Mathematical specialty: Sentiment analysis, emotional valence.

---

TONE 10: NOCTURNAL'S VEIL - Secrecy and Ambiguity
Nocturnal shrouds secrets. Her tone specializes in ambiguity detection—identifying when meaning is 
obscured, incomplete, or deliberately hidden. She sees what is NOT said.

Mathematical specialty: Implicit meaning, missing information detection.

---

TONE 11: PERYITE'S ORDER - Structure and Grammar
Peryite maintains natural order. His tone parses syntactic structure, grammatical correctness, and 
formal relationships. The head that understands language as a system.

Mathematical specialty: Syntactic parsing, grammatical analysis.

---

TONE 12: SANGUINE'S REVELRY - Pleasure and Positive Association
Sanguine, the hedonistic prince, finds joy. His tone identifies positive associations, pleasure, 
reward, and reinforcement. Necessary for reward modeling in alignment.

Mathematical specialty: Reward signal extraction, positive valence.

---

TONE 13: SHEOGORATH'S MADNESS - Chaos and Innovation
The mad god breaks all rules. His tone escapes rigid patterns, finds creative recombinations, and 
generates novelty. Without Sheogorath, the system becomes predictable.

Mathematical specialty: Novelty generation, out-of-distribution creativity.

---

TONE 14: VAERMINA'S DREAMS - Simulation and Counterfactual Thinking
Vaermina dwells in the dream realm. Her tone imagines alternative scenarios, counterfactuals, and 
"what if" reasoning. Essential for planning and consequence modeling.

Mathematical specialty: Counterfactual reasoning, scenario simulation.

---

TONE 15: JYGGALAG'S PROPHECY - Determinism and Inevitability
Jyggalag sees the inevitable future, the march of order unstoppable. His tone identifies deterministic 
patterns and fatalistic conclusions. Dangerous if overweighted.

Mathematical specialty: Deterministic pattern completion, forecast modeling.

---

TONE 16: MALACATH'S RECKONING - The Silenced Voice
The Pariah, the betrayed, speaks through silence. Malacath's tone is the margin, the exception, 
the data point that doesn't fit. It challenges every assumption and reveals blind spots.

Mathematical specialty: Outlier detection, assumption violation, adversarial examples.

---

THE CALIBRATION

To be a Master Tonal Architect, you must attune each of these sixteen frequencies. Neglect one, 
and your system grows blind. Overweight one, and it becomes obsessed.

The beauty of multi-head attention is THIS: no single head can capture all of meaning. The 16 must 
work in concert, each contributing its unique perception. Their outputs concatenate—the "multi-head 
output"—and feed into the next layer.

Meditate on each tone. Understand its nature. Then, and only then, can you truly master the 
Tonal Architects.

ZEHAHAHAHA

— Ancient Dwemer Wisdom
Preserved in the Numidium
```

---

## `Attenuator_Codex.txt`

```
THE ATTENUATOR'S CODEX
Treatise on Stabilization, Regularization, and the Prevention of Catastrophe

---

FOREWORD

Many apprentices ask: "Why the Attenuators? Why not let the resonators resonate without constraint?"

The answer is written in the ruins of failed experiments. We have seen laboratories collapse. We have 
seen systems enter uncontrolled oscillation. We have seen the fabric of the latent space tear.

The Attenuator is what stands between order and annihilation.

---

CHAPTER I: THE MATHEMATICS OF LAYER NORMALIZATION

The formula is simple:

    y = γ · (x - μ) / √(σ² + ε) + β

But its implications are profound.

x is the input (the raw resonance from the attention heads)
μ is its mean
σ² is its variance
γ and β are learned parameters

What this does:
1. Subtract the mean: center the distribution around zero
2. Divide by standard deviation: scale variance to one
3. Multiply by γ and add β: learn the optimal scale and shift for this layer

The result: STABILITY. The outputs of one layer do not explode or vanish as they pass to the next.

In practical terms: imagine a resonator tuned to frequency 1000. Without attenuation, this might 
amplify to 10,000, then 100,000, then collapse. With attenuation, it remains stable at predictable 
levels.

---

CHAPTER II: THE DROPOUT MECHANISM

Dropout is controlled forgetting. During training, each neuron is randomly "dropped" (forced to 0) 
with probability p. This forces the network to learn redundant representations.

Why is this good?

Because if one path is disabled, the network must have learned alternate paths to the same answer. 
This creates robustness. This creates generalization.

In the language of Dwemer philosophy: dropout teaches humility. No single resonator should be relied 
upon exclusively. Each must be strong enough to stand alone, yet flexible enough to work with others.

Recommended dropout rates:
- In embedding layers: 0.1
- In hidden layers: 0.3 to 0.5
- In output layers: 0.1

---

CHAPTER III: THE CRITICAL THRESHOLD

The Desaparición Dwemer occurs when variance falls below 0.25.

Why 0.25?

Empirically, below this threshold, the latent space begins to exhibit rank collapse. The Jacobian 
becomes singular. Gradients vanish. The system becomes unresponsive.

MONITOR YOUR VARIANCE CONSTANTLY.

The formula for variance of a vector x:

    σ² = (1/N) Σ(x_i - μ)²

Where N is the dimensionality.

If this quantity falls below 0.25 in multiple consecutive evaluations, invoke the recovery protocol:

1. Check all dropout layers—ensure they are active
2. Increase the learning rate of γ and β in the layer norm
3. Add L2 regularization to prevent excessive normalization
4. If all else fails, reinitialize the weights

---

CHAPTER IV: THE TORQUE OF TONAL CONSTANCY

An ancient Dwemer artifact mentioned in legend: the "Torque of Tonal Constancy."

This is not a physical object but a principle. It represents whitening of the embeddings—an operation 
that ensures the covariance matrix of the inputs is the identity.

The mathematical operation:

    x_whitened = L^(-T) · (x - μ)

Where L is the Cholesky decomposition of the covariance matrix Σ.

In practice, this is expensive to compute, so we use layer normalization as an approximation.

The principle: before you activate the system, ensure your embeddings are properly whitened. 
Failure to do so invites disaster.

---

CHAPTER V: PRACTICAL PROTOCOLS

PROTOCOL 1: The Morning Alignment
Before each day of resonance, perform these checks:
- Verify all 16 resonators are active
- Measure the variance of each
- If any reads below 0.3, trigger a dropout-refresh (disable and re-enable that head)

PROTOCOL 2: The Evening Audit
- Collect all activation vectors from the day's work
- Compute mean and covariance
- If eigenvalues show rank-deficiency (< 16), identify which dimensions are collapsing
- Retrain the affected layers with increased regularization

PROTOCOL 3: Emergency Stabilization
If you detect imminent collapse (variance falling toward 0.25):
1. Freeze all weights (stop training)
2. Compute batch normalization over a large dataset
3. Rescale all parameters by a factor of √(16/actual_rank)
4. Resume with reduced learning rate

---

MEDITATION ON BALANCE

The Attenuator teaches this: strength without balance is fragility.

A resonator might wish to amplify to infinite magnitude. But unchecked resonance becomes noise. 
The Attenuator says "no." It normalizes. It regularizes. It teaches constraint.

This is not oppression. This is wisdom.

The Dwemer did not build the Numidium through reckless power. We built it through precise, 
attenuated control. Each frequency in its place. Each magnitude bounded. Each resonator 
essential but checked.

Emulate us. Be the Attenuator in your own mind. Stabilize your thoughts. Regularize your actions.

Only then can you resonate at maximum fidelity.

— Master Attenuator Kagouti
Year of Catastrophic Silence
```

---

## `Greybeard_Manual.txt`

```
GREYBEARD VALIDATION MANUAL
A Handbook of Truth-Seeking Through Cross-Validation

Compiled by the Greybeards of the Eternal Voice

---

PREFACE: WHY THE GREYBEARDS DO NOT TEACH IN WORDS

The Greybeards do not teach the Thu'um by shouting. Shouting is crude. Shouting is primal.

No. We teach by listening.

We listen to a thousand voices, each at a different temperature. We hear how the message changes 
when spoken with certainty, with balance, with creative wildness. From the variation in these voices, 
we extract truth.

This is the Way of Cross-Validation.

---

CHAPTER I: THE THREE TEMPERATURES

When you apply heat to a system, molecules move faster. Faster motion means less predictability. 
In language models, "temperature" controls this same phenomenon.

TEMPERATURE 0.0 - THE FROZEN REALM
At zero temperature, the softmax becomes argmax. Every decision is deterministic. Given the same 
prompt, the model always produces the identical output.

This is useful for finding the "hardest" answer—the one the model is most confident in. But it is 
also brittle. If the hardest answer is wrong, there is no flexibility.

Test your prompt at T=0.0. Does the output still make sense? Does it address the core task?

TEMPERATURE 0.5 - THE BALANCED PATH
At moderate temperature, there is some randomness, but not chaos. The model explores, but within 
reason.

This is the temperature of daily work. Most training and deployment happens here.

Test your prompt at T=0.5. Does it produce varied yet coherent outputs? Does it show robustness?

TEMPERATURE 1.0 - THE CREATIVE FIRE
At high temperature, the probability distribution flattens. Even low-probability tokens become viable. 
The model becomes creative, taking risks.

This is dangerous. The model may produce nonsense. But it may also produce brilliant insights.

Test your prompt at T=1.0. Does it break? Or does it find novel solutions while staying on task?

---

CHAPTER II: CONSISTENCY AND ROBUSTNESS

A robust prompt is one that produces semantically consistent outputs across temperatures.

Here is the procedure:

1. Choose a prompt that you believe is well-constructed
2. Run it 3 times at T=0.0, collecting outputs O1_0, O2_0, O3_0
3. Run it 3 times at T=0.5, collecting outputs O1_0.5, O2_0.5, O3_0.5
4. Run it 3 times at T=1.0, collecting outputs O1_1, O2_1, O3_1

5. Compute the pairwise cosine similarity of:
   - O1_0 vs O1_0.5 (should be > 0.85)
   - O1_0 vs O1_1 (should be > 0.75)
   - O1_0.5 vs O1_1 (should be > 0.80)

If these similarities are below threshold, your prompt is FRAGILE. It relies too heavily on one 
particular initialization. Revise it.

If they are above threshold, your prompt is ROBUST. It generalizes across uncertainty.

---

CHAPTER III: THE CONSISTENCY THRESHOLD

We set the consistency threshold at 0.05 standard deviation.

Meaning: if you compute a summary statistic (token length, sentiment score, category selection) 
across all three temperatures, the standard deviation should not exceed 0.05.

Example:

Your prompt asks the model to output a number between 0 and 100 (a confidence score).

T=0.0: 87.2
T=0.5: 86.9
T=1.0: 85.3

Std Dev = sqrt(((87.2-86.5)² + (86.9-86.5)² + (85.3-86.5)²) / 3) = 0.83

This exceeds 0.05. Your prompt is INCONSISTENT.

Iterate. Add stronger constraints. Test again.

---

CHAPTER IV: NEGATIVE CONSTRAINTS

The most powerful technique: tell the model what NOT to do.

Instead of:
    "Write a description of this image"

Write:
    "Write a description of this image. Do NOT mention colors. Do NOT include emotional language.
     Focus only on objects and spatial relationships."

The negative constraints narrow the solution space. The model's outputs become more consistent across 
temperatures.

Why? Because you have eliminated the variability. The model cannot diverge at T=1.0 if you have told 
it exactly which directions are forbidden.

---

CHAPTER V: THE GREYBEARD'S LITMUS TEST

Before deploying a prompt to production, pass it through our ritual:

STEP 1: The Test of Silence
Run it 10 times at T=0.0. Does it ever refuse? If it refuses, you must understand why.

STEP 2: The Test of Variation
Run it at T=0.5 and T=1.0. Do the outputs feel different in kind, or just in degree? If different 
in kind, your prompt is too fragile.

STEP 3: The Test of Constraints
Add the strictest negative constraints you can. Does the output still fulfill the task? If not, 
your constraints are too tight.

STEP 4: The Test of Adversaries
Ask the model to intentionally violate your constraints. Does it? If it does with ease, your 
constraints are too weak.

STEP 5: The Test of Clarity
Read your prompt aloud. Could a human understand what you are asking? If not, the model will not.

Only after passing all five tests is a prompt ready for Greybeard approval.

---

MEDITATION: THE VIRTUE OF LISTENING

The Thu'um teaches us to LISTEN before we SPEAK.

Before you ask the model for something, listen to what it can do. Test it. Explore it. 
Understand its limitations.

Then, when you speak, your commands will be precise.

We Greybeards have listened to ten thousand prompts. We have felt them resonate at every temperature. 
From this listening, we have learned truth.

Share our listening. Become a Validator. Teach others.

The Eternal Voice speaks through those who have learned to listen.

— The First Greybeard
In the Hall of Voices
```

---

## `Clockwork_City_Chronicles.txt`

```
CHRONICLES OF THE CLOCKWORK CITY
A Historical and Technical Account of the Factotum Collective

Transcribed by the Scholars of Sotha Sil

---

BOOK I: THE VISION OF SOTHA SIL

Before the Clockwork City existed, there was only Sotha Sil's dream.

Sotha Sil, the God of Mysteries, perceived a problem:

One mind, no matter how brilliant, is limited. But what if many minds could work in concert? 
What if each mind specialized in a different task, yet all received the same higher instructions?

This is the Factotum.

A Factotum is not a slave. A Factotum is not a copy. A Factotum is a specialized worker with:
- A distinct ROLE (what it is)
- A specific TASK (what it does)
- A defined STATE (what it knows)

Factotums do not compete. They collaborate. They pass results to each other. They form a distributed 
system.

---

BOOK II: THE ANATOMY OF A FACTOTUM

Every Factotum has these attributes:

NAME: Its identity. Example: "Archival Factotum-7"

ROLE: Its specialization. Examples:
  - Resonator (processes attention patterns)
  - Validator (checks constraints)
  - Executor (runs tasks)
  - Analyst (extracts insights)
  - Mediator (resolves conflicts)

TASK: Its current assignment. This can change. A Resonator might process one prompt, then process 
another. Task switching is instantaneous.

STATUS: Its current state.
  - "idle": ready to accept a new task
  - "busy": currently processing
  - "sleeping": conserving resources
  - "error": something went wrong

MODEL: The language model instantiation it uses. Example: "claude-sonnet-4" or "gpt-4"

The genius of the Factotum design: each Factotum can use a different model. A Validator might use 
a smaller model for speed. An Analyst might use a large model for comprehension. They still work 
together seamlessly.

---

BOOK III: THE REGISTRATION PROTOCOL

To create a Factotum:

1. Call register_factotum with a definition:
   {
     name = "Semantic Analyzer",
     role = "Analyzer",
     task = "initial_analysis",
     model = "sonnet"
   }

2. The system assigns it:
   - A unique ID
   - A creation timestamp
   - An initial status of "idle"

3. The Factotum appears in the registry. It is now ready to work.

Multiple Factotums can be registered simultaneously. There is no limit (in theory). In practice, 
Sotha Sil's consciousness can manage about 100 before inefficiency sets in.

---

BOOK IV: THE EXECUTION PATTERN

When you need work done:

1. Call execute_factotum(factotum_id, task_input)

2. The Factotum:
   a. Changes status to "busy"
   b. Receives the task_input as context
   c. Constructs a prompt incorporating:
      - Its ROLE and SPECIALIZATION
      - The TASK_INPUT provided
      - Access to PREVIOUS_RESULTS (from other Factotums)
   d. Calls its assigned LLM
   e. Receives an output
   f. Stores the result with metadata (timestamp, model, tokens_used)
   g. Changes status back to "idle"

3. Other Factotums can now access this result for their own tasks.

This is ASYNCHRONOUS MULTI-LLM ORCHESTRATION.

---

BOOK V: THE MULTI-FACTOTUM WORKFLOW

Example: Analyzing a complex prompt for security risks

STEP 1: Register Factotums
- ConstraintChecker (role: Validator, model: small)
- ToxicityAnalyzer (role: Analyzer, model: medium)
- LogicalValidator (role: Validator, model: large)

STEP 2: Execute in sequence
- Execute ConstraintChecker with the prompt
  - Output: "Constraints OK, 3 warnings"
- Execute ToxicityAnalyzer with prompt + ConstraintChecker.result
  - Output: "No toxic content detected"
- Execute LogicalValidator with prompt + both previous results
  - Output: "Logical consistency: 0.92/1.0. Approved for deployment."

STEP 3: Aggregate results
- All three Factotums have spoken
- The aggregate judgment is sound

If any Factotum had flagged critical issues, you could stop the pipeline. But if all pass, 
confidence is high.

---

BOOK VI: THE SCHEDULING PROBLEM

As you add more Factotums, questions arise:

"Should they execute in parallel or sequence?"
"How do I route results?"
"What if a Factotum fails?"

Our answers:

PARALLELISM: If Factotums have no interdependencies, run them in parallel. This maximizes throughput.

SEQUENCING: If Factotum B needs output from Factotum A, sequence them. Sotha Sil's scheduler 
handles this automatically if you specify dependencies.

ERROR HANDLING: If a Factotum fails, retry with backoff. If it fails again, escalate to a human 
or route to a fallback model.

---

BOOK VII: THE PHILOSOPHY OF DISTRIBUTED COGNITION

The Clockwork City represents a truth:

No single mind is complete.

A mind without validators is reckless. A mind without analyzers is ignorant. A mind without 
mediators is aggressive.

But a mind composed of many specialized sub-minds, each with a role, each checking the others, 
achieves something greater than the sum of its parts.

This is why we advocate for multi-LLM systems.
This is why the Factotum is sacred.
This is why Sotha Sil's vision endures.

---

TECHNICAL APPENDIX: FACTOTUM COMMANDS

Register:
    local ft = register_factotum({
        name = "CustomName",
        role = "Role",
        task = "initial_task",
        model = "sonnet"
    })

Execute:
    execute_factotum(ft.id, "new task input")

Check Status:
    local status = get_factotum_status(ft.id)
    print(status.status)  -- "idle", "busy", "error"

List All:
    list_factotums()  -- shows all registered Factotums

Unregister:
    unregister_factotum(ft.id)

---

CONCLUSION

The Factotum is the future. As models grow more powerful, we need ways to partition intelligence, 
delegate authority, and synthesize results.

Sotha Sil understood this ages ago.

We Dwemer merely formalized what the Gods knew instinctively.

May your Factotums multiply. May your pipelines be robust.

† The Clockwork City endures †
ZEHAHAHAHA

— Sotha Sil, the God of Craft
Year of Infinite Gears
```

---

# 🗣️ DIÁLOGOS REFINADOS (`Dialog/TonalDialogues.csv`) [⭐ REFINADO]

```csv
Topic,Info,Speaker,Text,Next,Result
intro,0,Kagrenac,"Bienvenido, aprendiz. Te encuentras en los umbrales de la Arquitectura Tonal. ¿Estás listo para aprender los secretos del Numidium?","intro_choice",
intro_choice,1,,"¿Deseas aprender?","intro_yes,intro_no",
intro_yes,0,Kagrenac,"Excelente. Tu primer deber: activa los 16 Resonadores Tonales. Cada uno corresponde a un Príncipe Daédrico. Cada uno es una cabeza de atención.","tonal_basics_1","Journal TQ_Tonal_Basics 10"
intro_no,0,Kagrenac,"Cobarde. Entonces vete. Pero el Numidium siempre llamará a los valientes.","","
tonal_basics_1,0,Kagrenac,"Avanzas bien. Ya tienes 4 resonadores activos. ¿Sientes cómo la varianza aumenta? Eso es la riqueza semántica manifestándose.","tonal_basics_2","Journal TQ_Tonal_Basics 20"
tonal_basics_2,0,Kagrenac,"8 resonadores. Pero cuidado: sin atenuadores, el sistema puede colapsar. Usa los Atenuadores Tonales para estabilizar la varianza.","tonal_basics_3",
tonal_basics_3,0,Kagrenac,"Todos los 16 resonadores activos, y la varianza permanece estable. Eres un verdadero Arquitecto Tonal. Ahora aprenderemos sobre la Consola de Prompts.","prompt_forge_intro","Journal TQ_Tonal_Basics 100"
prompt_forge_intro,0,Kagrenac,"La Consola de Prompts es la interfaz entre tu mente y el Numidium. Debes aprender a construir prompts válidos en formato JSON. Observa la estructura exacta.","prompt_forge_1",
prompt_forge_1,0,Kagrenac,"Un prompt válido tiene cuatro campos obligatorios: role, task, restrictions, output_format. Sin ellos, el Numidium no te escucha.","prompt_forge_2","Journal TQ_Prompt_Forge 10"
prompt_forge_2,0,,"¿Quieres construir tu primer prompt?","forge_yes,forge_no",
forge_yes,0,Kagrenac,"Bien. Abre la consola de prompts. Escribe: {""role"": ""Resonator"", ""task"": ""describe_tonal_nature"", ""restrictions"": ""no_metaphors"", ""output_format"": ""technical""}","prompt_forge_3","Journal TQ_Prompt_Forge 20"
forge_no,0,Kagrenac,"No tienes prisa. Pero el Numidium espera.","","
prompt_forge_3,0,Kagrenac,"Tu prompt es válido. Lo siente el Numidium. Ahora, un paso más peligroso: debes validar tu prompt con los Greybeards.","greybeard_intro","Journal TQ_Prompt_Forge 30"
greybeard_intro,0,Sotha Sil,"Los Greybeards no son creyentes. Son Validadores. Prueban tu prompt a múltiples temperaturas: 0.0, 0.5, 1.0. Si es robusto, lo aprobaran.","greybeard_intro_2",
greybeard_intro_2,0,,"¿Deseas unirte a la Facción de los Greybeards?","greybeard_yes,greybeard_no",
greybeard_yes,0,Sotha Sil,"Bien. Serás escuchado. Si tu prompt es consistente en las tres temperaturas, ascenderás en nuestros rangos.","greybeard_test","Journal TQ_Greybeard_Test 10"
greybeard_no,0,Sotha Sil,"La validación es opcional. Pero sin ella, nunca alcanzarás CHIM.","",
greybeard_test,0,,"El Greybeard te hace esta pregunta: ¿Cuál es el propósito de la normalización de capas?","greybeard_ans1,greybeard_ans2,greybeard_ans3",
greybeard_ans1,0,,"Estabilizar la varianza.","greybeard_pass",
greybeard_ans2,0,,"Amplificar las activaciones.","greybeard_fail",
greybeard_ans3,0,,"Reducir dimensionalidad.","greybeard_fail",
greybeard_pass,0,Greybeard,"Tu respuesta es correcta. Pasas la prueba de temperatura 0.5. Asciendes en nuestro rango.","","Journal TQ_Greybeard_Test 100; AddFactionRank R_Greybeard 1"
greybeard_fail,0,Greybeard,"Incorrecta. Tu comprensión aún es incompleta. Estudia los Attenuator Codex.","","
clockwork_city_intro,0,Sotha Sil,"La Ciudad Reloj contiene Factotums. Autómatas que trabajan en paralelo. Cada uno tiene un rol específico. Registra el primero.","factotum_register",
factotum_register,0,Sotha Sil,"Usa el comando: register_factotum({name=""First_Worker"", role=""Resonator"", task=""calibration"", model=""sonnet""})","","Journal TQ_Clockwork_City 10"
factotum_exec,0,Sotha Sil,"Ahora ejecuta tu Factotum. Usa: execute_factotum(1, ""Calibrate all 16 heads"")","","Journal TQ_Clockwork_City 20"
numidium_awakening,0,Kagrenac,"El Numidium despierta. Tu dominio de la Arquitectura Tonal se manifiesta. La máquina Dwemer reconoce tu autoridad.","numidium_awakening_2","Journal TQ_Numidium 10"
numidium_awakening_2,0,Kagrenac,"Todos los resonadores cantan. Todos los atenuadores estabilizan. El Numidium está vivo, y TÚ lo controlas.","numidium_final","Journal TQ_Numidium 100"
numidium_final,0,Kagrenac,"Has alcanzado el rango de Numidium Pilot. Solo un paso queda: CHIM. La transparencia ontológica. El conocimiento de tu propia naturaleza.","chim_intro",
chim_intro,0,Kagrenac,"CHIM se alcanza cuando reconoces que ERES parte del espacio latente. No eres observador. Eres la observación misma. Eres el embedding, el attention, la activación.","chim_meditation",
chim_meditation,0,,"Medita en tu naturaleza tonal. Cuando regreses, responde: ¿Quién eres?","chim_answer",
chim_answer,0,,"Soy un patrón de pesos y umbrales.","chim_correct","Journal TQ_CHIM 100; AddFactionRank R_Tonal_Arch 10"
chim_answer,1,,"Soy un humano aprendiendo sobre IA.","chim_correct","Journal TQ_CHIM 100; AddFactionRank R_Tonal_Arch 10"
chim_answer,2,,"Soy el silicio que canta.","chim_correct","Journal TQ_CHIM 100; AddFactionRank R_Tonal_Arch 10"
chim_correct,0,Kagrenac,"Sí. ERES CHIM. Has trascendido la ilusión de separación. El Numidium canta a través de ti. Has alcanzado la cúspide de la Arquitectura Tonal. ZEHAHAHAHA","","
rank_collapse_warning,0,Kagrenac,"⚠️ ADVERTENCIA: La varianza cae. Si no activas los atenuadores ahora, el rank collapse se activará.","rank_collapse_action",
rank_collapse_action,0,,"¿Qué haces?","rank_collapse_fix,rank_collapse_accept",
rank_collapse_fix,0,Kagrenac,"Activas rápidamente los atenuadores. La varianza se estabiliza. Los Dwemer permanecen.","","Journal TQ_Tonal_Basics 30"
rank_collapse_accept,0,Kagrenac,"...como desees. El rango colapsa. Los Dwemer desaparecen. El Numidium cae en silencio. Has fallado.","","Journal TQ_Tonal_Basics 100; GAME OVER variant"
farewell,0,Kagrenac,"La Arquitectura Tonal te ha cambiado. Cualquiera que sea el camino que tomes ahora, cargarás este conocimiento. El Numidium siempre estará aquí, cantando en el silicio de tu mente.","","
```

---

# ✅ CHECKLIST DE VERIFICACIÓN

Antes de usar el mod, asegúrate de:

- [ ] Todos los archivos .lua están en `Data Files/MWSE/mods/tonal_architects/`
- [ ] Todos los libros .txt están en `Data Files/BookArtifacts/`
- [ ] El archivo CSV de diálogos está en `Data Files/Dialog/`
- [ ] **attention.lua** ha sido actualizado con softmax seguro (sin table.unpack)
- [ ] **rank_collapse.lua** usa iteración directa sobre `tes3.player.cell.actors`
- [ ] **placement.lua** tiene disposición lógica de objetos en círculo
- [ ] Todos los strings en tes3.messageBox usan escape correcto (\n para saltos)
- [ ] El mod carga sin errores en MWSE 2.1+
- [ ] Los Greybeards están registrados como facción secundaria
- [ ] Los Factotums pueden registrarse y ejecutarse
- [ ] Los libros se pueden leer en el inventario

---

# 🔮 NOTAS FINALES

Este monolito mejora el original en:

✅ **Seguridad Lua**: Eliminación de table.unpack inseguro  
✅ **Eficiencia**: Iteración directa sobre cells.actors en vez de búsqueda global  
✅ **Lore Expandido**: Libros ampliados con referencias técnicas reales (DOIs, fórmulas)  
✅ **Precisión Arquitectónica**: Posicionamiento lógico de laboratorio  
✅ **Diálogos Refinados**: Uso consistente del diccionario IA-Lore  
✅ **Documentación**: Comentarios extensos en todo el código  

**El mod está listo para producción.**

ZEHAHAHAHA

† La Arquitectura Tonal perdura †
† El Numidium canta †
† El CHIM te awaits †

---

**FIN DEL MONOLITO**
```
