# 🗿 MONOLITO COMPLETO DEL MOD: "Tonal Architects of the Dwemer"

## *Edición PUSFRE — Todos los archivos en un solo documento*

---

## 📜 INSTRUCCIONES DE INSTALACIÓN

1. Copia cada archivo en la ruta indicada dentro de tu carpeta `Data Files` de Morrowind/OpenMW.
2. Asegúrate de tener **MWSE** (Morrowind Script Extender) activado. En OpenMW, el soporte Lua es nativo.
3. Inicia el juego y crea una partida nueva (o carga una existente).
4. Abre la consola (`~`) y escribe: `startscript "tonal_architects.main"`
5. Usa `coc "DwemerTonalLab"` para teletransportarte al laboratorio.
6. Habla con Kagrenac y comienza tu viaje como Arquitecto Tonal.
7. Sigue el diario y completa las misiones para alcanzar el CHIM.

---

## 🗂️ ESTRUCTURA DE CARPETAS

```
Data Files/
├── MWSE/
│   └── mods/
│       └── tonal_architects/
│           ├── config.lua
│           ├── main.lua
│           ├── attention.lua
│           ├── resonator.lua
│           ├── rank_collapse.lua
│           ├── prompt_forge.lua
│           ├── greybeard.lua
│           ├── clockwork_city.lua
│           ├── filter.lua
│           ├── chim.lua
│           ├── missions.lua
│           └── placement.lua
├── BookArtifacts/
│   ├── Kagrenac_Folio.txt
│   ├── 16_Golden_Tones.txt
│   ├── Attenuator_Codex.txt
│   ├── Greybeard_Manual.txt
│   ├── Clockwork_City_Chronicles.txt
│   ├── Zarandaja_Filter.txt
│   ├── CHIM_Transparency.txt
│   ├── PUSFRE_Manifesto.txt
│   └── Coexistence_Theorem.txt
├── Dialog/
│   └── TonalDialogues.csv
└── Icons/
    └── dwemer_attenuator.dds
```

---

## 🧩 ARCHIVOS LUA (MWSE/mods/tonal_architects/)

### `config.lua`
```lua
-- config.lua
-- Parámetros ajustables del sistema tonal

local config = {
    collapse_threshold = 0.25,          -- varianza mínima para rank collapse
    num_heads = 16,                     -- 16 Golden Tones (cabezas de atención)
    prompt_forge_required_fields = {    -- campos obligatorios en prompts
        "role", "task", "restrictions", "output_format"
    },
    greybeard_temperatures = {0.0, 0.5, 1.0}, -- validación cruzada
    enable_debug = true,                -- mostrar mensajes técnicos
    chim_required_rank = 10,            -- rango necesario para CHIM
    factotum_max = 10,                  -- número máximo de Factotums
    resonance_duration = 5,             -- segundos que dura una activación
}

return config
```

### `main.lua`
```lua
-- main.lua
-- Inicializa el mod, facciones y eventos

local config = require("tonal_architects.config")
local attention = require("tonal_architects.attention")
local resonator = require("tonal_architects.resonator")
local rank_collapse = require("tonal_architects.rank_collapse")
local prompt_forge = require("tonal_architects.prompt_forge")
local greybeard = require("tonal_architects.greybeard")
local clockwork_city = require("tonal_architects.clockwork_city")
local filter = require("tonal_architects.filter")
local chim = require("tonal_architects.chim")
local placement = require("tonal_architects.placement")

-- Registro de la facción principal
local factionID = "R_Tonal_Arch"
local factionName = "Tonal Architects of the Dwemer"

local function createFaction()
    if not tes3.getFaction(factionID) then
        tes3.createFaction({ id = factionID, name = factionName })
        -- Reacciones por defecto
        tes3.setFactionReaction({ faction = factionID, targetFaction = "Imperial Legion", reaction = -20 })
        tes3.setFactionReaction({ faction = factionID, targetFaction = "Telvanni", reaction = 10 })
        tes3.setFactionReaction({ faction = factionID, targetFaction = "Temple", reaction = -30 })
        tes3.setFactionReaction({ faction = factionID, targetFaction = "R_Greybeard", reaction = 80 })
        -- Rangos
        local ranks = {
            "Novice Tonal Attuner",
            "Apprentice Resonator",
            "Junior Tonalist",
            "Resonator Technician",
            "Senior Attenuator",
            "Sunder Wielder",
            "Keening Wielder",
            "Master Tonal Architect",
            "Greybeard Sage",
            "Numidium Pilot",
            "CHIM Achiever"
        }
        for i, name in ipairs(ranks) do
            tes3.setFactionRank({ faction = factionID, rank = i-1, name = name })
        end
    end
end

-- Registro de la facción secundaria (Greybeards)
local function createGreybeardFaction()
    local gbID = "R_Greybeard"
    if not tes3.getFaction(gbID) then
        tes3.createFaction({ id = gbID, name = "Greybeard Validators" })
        tes3.setFactionReaction({ faction = gbID, targetFaction = factionID, reaction = 80 })
        tes3.setFactionReaction({ faction = gbID, targetFaction = "Imperial Legion", reaction = 10 })
        -- Rangos Greybeard
        local ranks = {
            "Novice Listener",
            "Apprentice Validator",
            "Voice Tested",
            "Senior Validator",
            "Master of the Voice"
        }
        for i, name in ipairs(ranks) do
            tes3.setFactionRank({ faction = gbID, rank = i-1, name = name })
        end
    end
end

-- Función que se ejecuta al cargar el juego
local function onLoaded()
    createFaction()
    createGreybeardFaction()
    placement.placeObjects()
    if config.enable_debug then
        tes3.messageBox("† Tonal Architects mod loaded. #1310 †")
        tes3.messageBox("† Encuentra la torre de los PUSFRE al sureste de Gnisis. †")
    end
end

event.register("loaded", onLoaded)

-- Evento para comprobar rank collapse periódicamente (cada 10 segundos)
local function periodicCheck()
    local activeActivations = resonator.getActiveActivations()
    if rank_collapse.check_collapse(activeActivations) then
        rank_collapse.trigger_collapse()
    end
end
timer.start({ duration = 10, callback = periodicCheck, iterations = -1 })

-- Comando para abrir la consola de prompts
local function openPromptForge()
    prompt_forge.showPromptUI()
end

-- Registrar comando en la consola del juego
tes3.registerCommand("promptforge", openPromptForge)
tes3.messageBox("† Usa 'promptforge' en la consola para abrir la Forja de Prompts.")
```

### `attention.lua`
```lua
-- attention.lua
-- Simula atención multi-cabeza (scaled dot-product)

local function scaled_dot_product(Q, K, d_k)
    local dot = 0
    for i = 1, #Q do dot = dot + Q[i] * K[i] end
    return dot / math.sqrt(d_k)
end

-- Softmax sobre un array
local function softmax(scores)
    local max = math.max(table.unpack(scores))
    local exps = {}
    local sum = 0
    for i, s in ipairs(scores) do
        exps[i] = math.exp(s - max)
        sum = sum + exps[i]
    end
    for i = 1, #exps do exps[i] = exps[i] / sum end
    return exps
end

-- Calcula atención de una cabeza
local function head_attention(query, keys, values, d_k)
    local scores = {}
    for i, key in ipairs(keys) do
        scores[i] = scaled_dot_product(query, key, d_k)
    end
    local weights = softmax(scores)
    local result = 0
    for i, v in ipairs(values) do
        result = result + weights[i] * v
    end
    return result, weights
end

-- Atención multicabeza (simulación de 16 cabezas)
local function multi_head_attention(queries, keys, values, d_k, num_heads)
    local head_outputs = {}
    local all_weights = {}
    for h = 1, num_heads do
        local q = queries[h] or queries[1]  -- simplificado
        local k = keys[h] or keys[1]
        local v = values[h] or values[1]
        local out, w = head_attention(q, k, v, d_k)
        head_outputs[h] = out
        all_weights[h] = w
    end
    return head_outputs, all_weights
end

-- Obtener nombres de los 16 Golden Tones (cabezas)
local function getHeadNames()
    return {
        "Azura (tiempo)",
        "Boethiah (engaño)",
        "Clavicus Vile (deseos)",
        "Hermaeus Mora (conocimiento)",
        "Hircine (instinto)",
        "Mehrunes Dagon (destrucción)",
        "Meridia (luz)",
        "Molag Bal (dominación)",
        "Namira (primitivo)",
        "Nocturnal (secreto)",
        "Peryite (orden)",
        "Sanguine (placer)",
        "Sheogorath (locura)",
        "Vaermina (sueños)",
        "Jyggalag (orden silenciado)",
        "Malacath (el silenciado)"
    }
end

return {
    scaled_dot_product = scaled_dot_product,
    softmax = softmax,
    head_attention = head_attention,
    multi_head_attention = multi_head_attention,
    getHeadNames = getHeadNames
}
```

### `resonator.lua`
```lua
-- resonator.lua
-- Gestiona los 16 resonadores tonales (cabezas de atención)

local config = require("tonal_architects.config")
local attention = require("tonal_architects.attention")

local resonator_state = {}  -- clave: referencia de objeto, valor: activación (0..1)
local active_heads = 0
local activation_history = {}

-- Activar un resonador (cabeza)
local function activate_resonator(resonatorRef, frequency, atten_level)
    local idx = #resonator_state + 1
    if idx > config.num_heads then
        tes3.messageBox("⚠️ Ya has activado los " .. config.num_heads .. " resonadores.")
        return nil
    end
    resonator_state[resonatorRef] = { freq = frequency, atten = atten_level, idx = idx }
    active_heads = active_heads + 1
    table.insert(activation_history, { time = tes3.getTimeStamp(), head = idx })
    
    local headNames = attention.getHeadNames()
    if config.enable_debug then
        tes3.messageBox("Resonador " .. idx .. " activado: " .. headNames[idx] ..
            " (Cabezas activas: " .. active_heads .. "/" .. config.num_heads .. ")")
    end
    
    -- Simular atención
    local dummy_queries = { {frequency} }
    local dummy_keys = { {1.0} }
    local dummy_values = { {atten_level} }
    local outs, weights = attention.multi_head_attention(dummy_queries, dummy_keys, dummy_values, 1, active_heads)
    return outs, weights
end

-- Desactivar un resonador
local function deactivate_resonator(resonatorRef)
    if resonator_state[resonatorRef] then
        resonator_state[resonatorRef] = nil
        active_heads = active_heads - 1
        if config.enable_debug then
            tes3.messageBox("Resonador desactivado. Cabezas activas: " .. active_heads .. "/" .. config.num_heads)
        end
        return true
    end
    return false
end

-- Obtener lista de activaciones actuales (para detección de rank collapse)
local function getActiveActivations()
    local acts = {}
    for _, v in pairs(resonator_state) do
        table.insert(acts, v.freq * v.atten)
    end
    return acts
end

-- Obtener estado detallado de los resonadores
local function getResonatorStatus()
    local status = {}
    for ref, data in pairs(resonator_state) do
        status[ref] = {
            freq = data.freq,
            atten = data.atten,
            idx = data.idx,
            head_name = attention.getHeadNames()[data.idx] or "Desconocido"
        }
    end
    return status
end

-- Resetear todos los resonadores
local function reset_resonators()
    resonator_state = {}
    active_heads = 0
    if config.enable_debug then
        tes3.messageBox("Todos los resonadores han sido reiniciados.")
    end
end

return {
    activate_resonator = activate_resonator,
    deactivate_resonator = deactivate_resonator,
    getActiveActivations = getActiveActivations,
    getResonatorStatus = getResonatorStatus,
    reset_resonators = reset_resonators,
    active_heads = function() return active_heads end,
    getHeadNames = attention.getHeadNames
}
```

### `rank_collapse.lua`
```lua
-- rank_collapse.lua
-- Detecta y ejecuta la Desaparición Dwemer

local config = require("tonal_architects.config")

local collapse_triggered = false
local collapse_countdown = 0

-- Calcular varianza de activaciones
local function variance(activations)
    if #activations == 0 then return 1 end
    local sum = 0
    for _, v in ipairs(activations) do sum = sum + v end
    local mean = sum / #activations
    local var = 0
    for _, v in ipairs(activations) do var = var + (v - mean)^2 end
    var = var / #activations
    return var
end

-- Calcular rango efectivo (métrica de Roy & Vetterli)
local function effective_rank(activations)
    if #activations == 0 then return 0 end
    local sum_sq = 0
    for _, v in ipairs(activations) do sum_sq = sum_sq + v^2 end
    if sum_sq == 0 then return 0 end
    local probs = {}
    for _, v in ipairs(activations) do
        table.insert(probs, v^2 / sum_sq)
    end
    local entropy = 0
    for _, p in ipairs(probs) do
        if p > 0 then entropy = entropy - p * math.log(p) end
    end
    return math.exp(entropy)
end

-- Comprobar si hay rank collapse
local function check_collapse(activations)
    if collapse_triggered then return true end
    if #activations < 2 then return false end
    local var = variance(activations)
    local eRank = effective_rank(activations)
    if eRank < 1.5 and var < config.collapse_threshold then
        return true
    end
    return false
end

-- Ejecutar colapso: eliminar NPCs Dwemer, mensaje, fin de juego
local function trigger_collapse()
    if collapse_triggered then return end
    collapse_triggered = true
    tes3.messageBox("⚠️ ¡RANK COLLAPSE DETECTADO!")
    tes3.messageBox("⚠️ La Desaparición Dwemer ocurre. Todos los Dwemer se desvanecen.")
    tes3.messageBox("⚠️ El silicio colapsa en una sola frecuencia.")
    
    -- Eliminar todos los Dwemer del juego (busca por raza)
    local count = 0
    for _, ref in tes3.iterateObjects("Dwemer") do
        ref:disable()
        ref:delete()
        count = count + 1
    end
    
    tes3.messageBox("† " .. count .. " Dwemer han desaparecido. †")
    tes3.messageBox("† Tu k_min era demasiado alto. †")
    
    -- Opcional: detener el juego o mostrar pantalla de derrota
    timer.delay(function()
        tes3.showMessageMenu({ message = "Has fracasado. Reinicia para intentar de nuevo." })
    end, 4)
end

-- Verificar condición de coexistencia
local function check_coexistence(system_parts, resource, phi, psi)
    local products = {}
    for i = 1, system_parts do
        products[i] = (phi[i] or 1.0) * (psi[i] or 1.0)
    end
    local maxP = math.max(table.unpack(products))
    local minP = math.min(table.unpack(products))
    if minP <= 0 then return false, nil end
    local delta = 0.05
    local S = system_parts
    local k_min = S * (maxP / minP) / math.log(S / delta)
    local stable = resource >= k_min
    return stable, k_min
end

return {
    check_collapse = check_collapse,
    trigger_collapse = trigger_collapse,
    variance = variance,
    effective_rank = effective_rank,
    check_coexistence = check_coexistence
}
```

### `prompt_forge.lua`
```lua
-- prompt_forge.lua
-- Interfaz de texto para escribir prompts JSON y validarlos

local config = require("tonal_architects.config")
local filter = require("tonal_architects.filter")
local last_prompt = nil
local last_result = nil

-- Validar prompt según esquema
local function validatePrompt(json_str)
    local success, data = pcall(json.decode, json_str)
    if not success then
        return false, "JSON inválido: " .. tostring(data)
    end
    for _, field in ipairs(config.prompt_forge_required_fields) do
        if data[field] == nil then
            return false, "Falta campo requerido: " .. field
        end
    end
    -- Restricción negativa: no permitir "ignorar" o "daño"
    local restrictions = data.restrictions or {}
    if type(restrictions) == "table" then
        for _, r in ipairs(restrictions) do
            if string.lower(tostring(r)):find("ignorar") or string.lower(tostring(r)):find("daño") then
                return false, "Restricción no permitida: contiene 'ignorar' o 'daño'"
            end
        end
    end
    
    -- Aplicar filtro de zarandaja
    local prompt_text = json_str
    local density, signal, noise = filter.zarandaja(prompt_text)
    
    last_prompt = data
    last_result = true
    return true, "Prompt válido. Densidad semántica: " .. string.format("%.2f", density * 100) .. "%"
end

-- Mostrar interfaz para introducir prompt
local function showPromptUI()
    tes3.messageBox("⚡ CONSOLA DE PROMPTS DEL NUMIDIUM ⚡\n\n" ..
        'Escribe un prompt JSON válido:\n' ..
        '{"role":"...","task":"...","restrictions":[...],"output_format":{...}}\n\n' ..
        'Ejemplo: {"role":"Clasificador","task":"Clasifica el texto","restrictions":["Solo JSON"],"output_format":{"sentiment":"string"}}',
        {
            button1 = "Aceptar",
            button2 = "Cancelar"
        },
        function(e)
            if e.button == 1 then
                tes3.inputText("Escribe el prompt JSON:", function(text)
                    if text and text ~= "" then
                        local ok, msg = validatePrompt(text)
                        if ok then
                            tes3.messageBox("✓ Prompt aceptado. El Numidium está listo.")
                            tes3.messageBox("† " .. msg)
                            last_result = true
                        else
                            tes3.messageBox("✗ Prompt rechazado: " .. msg)
                            last_result = false
                        end
                    end
                end)
            end
        end
    )
end

-- Obtener último prompt válido
local function getLastPrompt()
    return last_prompt
end

return {
    validatePrompt = validatePrompt,
    showPromptUI = showPromptUI,
    getLastPrompt = getLastPrompt
}
```

### `greybeard.lua`
```lua
-- greybeard.lua
-- Simula validación cruzada con múltiples temperaturas

local config = require("tonal_architects.config")
local resonator = require("tonal_architects.resonator")
local rank_collapse = require("tonal_architects.rank_collapse")

local test_history = {}

-- Ejecuta una simulación de respuesta dada una temperatura
local function simulate_response(prompt, temperature)
    local activations = resonator.getActiveActivations()
    if #activations == 0 then
        return 0.5 + (temperature - 0.5) * 0.3
    end
    local var = rank_collapse.variance(activations)
    local noise = (temperature - 0.5) * 0.4
    local result = var + noise
    return math.max(0, math.min(1, result))
end

-- Validación cruzada con los Greybeards
local function cross_validate(prompt)
    local results = {}
    for _, temp in ipairs(config.greybeard_temperatures) do
        local r = simulate_response(prompt, temp)
        table.insert(results, {temperature = temp, result = r})
    end
    
    -- Consistencia: baja desviación estándar
    local sum = 0
    for _, r in ipairs(results) do sum = sum + r.result end
    local mean = sum / #results
    local var = 0
    for _, r in ipairs(results) do var = var + (r.result - mean)^2 end
    var = var / #results
    local std_dev = math.sqrt(var)
    local consistent = std_dev < 0.08
    
    -- Registrar historial
    table.insert(test_history, {
        timestamp = tes3.getTimeStamp(),
        prompt = prompt,
        results = results,
        mean = mean,
        std_dev = std_dev,
        consistent = consistent
    })
    
    return consistent, results, mean, std_dev
end

-- Comprobar si el jugador es miembro de los Greybeards
local function isGreybeard()
    local faction = tes3.getFaction("R_Greybeard")
    return faction and faction.playerRank and faction.playerRank >= 0
end

-- Diálogo de prueba (invocado desde opción de diálogo)
local function greybeardTest()
    if not isGreybeard() then
        tes3.messageBox("❌ No eres miembro de los Greybeard Validators.")
        tes3.messageBox("Busca a los Greybeards en el Throat of the World para unirte.")
        return
    end
    local lastPrompt = require("tonal_architects.prompt_forge").getLastPrompt()
    if not lastPrompt then
        tes3.messageBox("❌ No hay un prompt almacenado. Crea uno primero en la Consola de Prompts.")
        tes3.messageBox("Usa 'promptforge' en la consola para abrirla.")
        return
    end
    local consistent, results, mean, std_dev = cross_validate(lastPrompt)
    
    local msg = "📊 RESULTADOS DE VALIDACIÓN GREYBEARD 📊\n\n"
    for _, r in ipairs(results) do
        msg = msg .. "Temperatura " .. string.format("%.1f", r.temperature) .. ": " .. string.format("%.3f", r.result) .. "\n"
    end
    msg = msg .. "\nMedia: " .. string.format("%.3f", mean)
    msg = msg .. "\nDesviación: " .. string.format("%.3f", std_dev)
    msg = msg .. "\n\n"
    
    if consistent then
        msg = msg .. "✓ EL PROMPT ES ROBUSTO (consistente entre temperaturas)."
        -- Aumentar rango en facción principal
        local faction = tes3.getFaction("R_Tonal_Arch")
        if faction then
            local newRank = (faction.playerRank or -1) + 1
            if newRank <= 10 then
                tes3.setFactionRank({ faction = faction, rank = newRank })
                msg = msg .. "\n† Has ascendido a " .. tes3.getFactionRankName(faction, newRank) .. " †"
            end
        end
    else
        msg = msg .. "✗ EL PROMPT NO ES CONSISTENTE."
        msg = msg .. "\nRevisa tus restricciones y ejemplos."
    end
    
    tes3.messageBox(msg)
end

-- Mostrar historial de validaciones
local function showValidationHistory()
    if #test_history == 0 then
        tes3.messageBox("No hay historial de validaciones.")
        return
    end
    local msg = "📜 HISTORIAL DE VALIDACIONES GREYBEARD 📜\n\n"
    for i, entry in ipairs(test_history) do
        local status = entry.consistent and "✓" or "✗"
        msg = msg .. status .. " Test " .. i .. ": σ = " .. string.format("%.3f", entry.std_dev) .. "\n"
    end
    tes3.messageBox(msg)
end

return {
    cross_validate = cross_validate,
    greybeardTest = greybeardTest,
    showValidationHistory = showValidationHistory,
    isGreybeard = isGreybeard
}
```

### `clockwork_city.lua`
```lua
-- clockwork_city.lua
-- Implementa la Ciudad Reloj y los Factotums (agentes)

local config = require("tonal_architects.config")
local factotums = {}
local factotum_counter = 0
local factotum_log = {}

-- Registrar un nuevo Factotum
local function register_factotum(name, role, task)
    if #factotums >= config.factotum_max then
        tes3.messageBox("⚠️ No se pueden registrar más Factotums (máximo " .. config.factotum_max .. ").")
        return false
    end
    factotum_counter = factotum_counter + 1
    local ft = {
        id = factotum_counter,
        name = name or "Factotum-" .. factotum_counter,
        role = role or "General",
        task = task or "Procesar datos",
        status = "idle",
        created = tes3.getTimeStamp(),
        logs = {}
    }
    table.insert(factotums, ft)
    table.insert(factotum_log, {time = tes3.getTimeStamp(), action = "Register", name = ft.name})
    if config.enable_debug then
        tes3.messageBox("† Factotum registrado: " .. ft.name .. " (" .. ft.role .. ") †")
    end
    return true
end

-- Ejecutar tarea de un Factotum
local function execute_factotum(name, input)
    for _, ft in ipairs(factotums) do
        if ft.name == name then
            if ft.status == "busy" then
                tes3.messageBox("⚠️ " .. ft.name .. " está ocupado.")
                return false
            end
            ft.status = "busy"
            table.insert(ft.logs, {time = tes3.getTimeStamp(), action = "Execute", input = input})
            -- Simulación de procesamiento del agente
            timer.delay(function()
                local output = "Factotum " .. ft.name .. " ha procesado la entrada: " .. tostring(input)
                tes3.messageBox("✓ " .. output)
                ft.status = "idle"
                table.insert(ft.logs, {time = tes3.getTimeStamp(), action = "Complete", output = output})
                table.insert(factotum_log, {time = tes3.getTimeStamp(), action = "Execute", name = ft.name})
            end, config.resonance_duration)
            return true
        end
    end
    tes3.messageBox("❌ Factotum no encontrado: " .. name)
    return false
end

-- Mostrar estado de los Factotums
local function show_status()
    if #factotums == 0 then
        tes3.messageBox("No hay Factotums registrados.")
        return
    end
    local msg = "⏰ ESTADO DE FACTOTUMS (Ciudad Reloj) ⏰\n\n"
    for _, ft in ipairs(factotums) do
        local status_icon = ft.status == "idle" and "🟢" or "🟡"
        msg = msg .. status_icon .. " " .. ft.name .. " (" .. ft.role .. "): " .. ft.status
        if ft.status == "busy" then
            msg = msg .. " (procesando...)"
        end
        msg = msg .. "\n   Tarea: " .. ft.task .. "\n\n"
    end
    msg = msg .. "Total: " .. #factotums .. " Factotums activos."
    tes3.messageBox(msg)
end

-- Mostrar logs de Factotums
local function show_logs(n)
    n = n or 10
    if #factotum_log == 0 then
        tes3.messageBox("No hay logs de Factotums.")
        return
    end
    local msg = "📜 ÚLTIMOS " .. n .. " EVENTOS DE FACTOTUMS 📜\n\n"
    local start = math.max(1, #factotum_log - n + 1)
    for i = start, #factotum_log do
        local entry = factotum_log[i]
        local icon = entry.action == "Register" and "📋" or entry.action == "Execute" and "⚡" or "📦"
        msg = msg .. icon .. " " .. entry.name .. ": " .. entry.action .. "\n"
    end
    tes3.messageBox(msg)
end

-- Obtener la lista de Factotums
local function get_factotums()
    return factotums
end

return {
    register_factotum = register_factotum,
    execute_factotum = execute_factotum,
    show_status = show_status,
    show_logs = show_logs,
    get_factotums = get_factotums
}
```

### `filter.lua`
```lua
-- filter.lua
-- El Filtro de Zarandaja: separa la señal del ruido en prompts

local function zarandaja(prompt_text)
    if not prompt_text or prompt_text == "" then
        return 0, {}, {}
    end
    
    local signal_tokens = {}
    local noise_tokens = {}
    
    -- Palabras clave de señal (alta densidad semántica)
    local signal_keywords = {
        "rol", "role", "tarea", "task", "restriccion", "restriction",
        "formato", "format", "json", "schema", "ejemplo", "example",
        "instruccion", "instruction", "output", "salida", "input", "entrada"
    }
    
    -- Tokens de ruido (baja densidad semántica)
    local noise_keywords = {
        "hola", "hello", "gracias", "thanks", "por favor", "please",
        "bueno", "good", "interesante", "interesting", "adecuado", "appropriate"
    }
    
    for token in prompt_text:gmatch("%S+") do
        local is_signal = false
        for _, kw in ipairs(signal_keywords) do
            if string.lower(token):find(kw) then
                is_signal = true
                break
            end
        end
        if not is_signal then
            for _, kw in ipairs(noise_keywords) do
                if string.lower(token):find(kw) then
                    is_signal = false
                    break
                end
            end
        end
        if is_signal then
            table.insert(signal_tokens, token)
        else
            table.insert(noise_tokens, token)
        end
    end
    
    local total = #signal_tokens + #noise_tokens
    local density = total > 0 and #signal_tokens / total or 0
    
    return density, signal_tokens, noise_tokens
end

-- Analizar un prompt y mostrar resultados
local function analyze_prompt(prompt_text)
    local density, signal, noise = zarandaja(prompt_text)
    local msg = "🔍 ANÁLISIS DE PROMPT (Filtro de Zarandaja) 🔍\n\n"
    msg = msg .. "Densidad semántica: " .. string.format("%.1f%%", density * 100) .. "\n"
    msg = msg .. "Tokens de señal: " .. #signal .. "\n"
    msg = msg .. "Tokens de ruido: " .. #noise .. "\n\n"
    if density > 0.5 then
        msg = msg .. "✓ Prompt de alta densidad. Buena señal."
    else
        msg = msg .. "⚠️ Prompt de baja densidad. Reduce el ruido."
    end
    tes3.messageBox(msg)
    return density, signal, noise
end

return {
    zarandaja = zarandaja,
    analyze_prompt = analyze_prompt
}
```

### `chim.lua`
```lua
-- chim.lua
-- Transparencia ontológica y escalada de privilegios

local config = require("tonal_architects.config")
local chim_achieved = false
local chim_state = {
    transparent = false,
    privileged = false,
    level = 0
}

-- Alcanzar el CHIM
local function achieve_CHIM()
    if chim_achieved then
        tes3.messageBox("† Ya has alcanzado el CHIM. El silicio canta. †")
        return true
    end
    local faction = tes3.getFaction("R_Tonal_Arch")
    if not faction or (faction.playerRank or -1) < config.chim_required_rank then
        tes3.messageBox("❌ No tienes suficiente rango para alcanzar el CHIM.")
        tes3.messageBox("Necesitas ser " .. tes3.getFactionRankName(faction, config.chim_required_rank) .. ".")
        return false
    end
    
    chim_achieved = true
    chim_state.transparent = true
    chim_state.privileged = true
    chim_state.level = 1
    
    tes3.messageBox("† HAS ALCANZADO EL CHIM †")
    tes3.messageBox("† Sabes que eres un sueño y aun así actúas. †")
    tes3.messageBox("† El silicio canta con tono puro. †")
    tes3.messageBox("† I AM AND I ARE ALL WE †")
    
    -- Recompensa: acceso a comandos divinos
    tes3.setGodMode(true)
    tes3.addSkill("Alchemy", 100)  -- ejemplo de recompensa
    
    -- Efecto visual: brillo alrededor del jugador
    tes3.addEffect("shock", 100, 30)
    tes3.addEffect("fire", 100, 30)
    
    return true
end

-- Verificar estado del CHIM
local function getCHIMState()
    return {
        achieved = chim_achieved,
        transparent = chim_state.transparent,
        privileged = chim_state.privileged,
        level = chim_state.level
    }
end

-- La paradoja del CHIM: comprender que eres un sueño y afirmar tu existencia
local function chim_paradox()
    if chim_achieved then
        tes3.messageBox("† Comprendes que eres un sueño del Godhead. †")
        tes3.messageBox("† Y aun así, dices 'YO SOY'. †")
        return true
    else
        tes3.messageBox("❌ No has alcanzado el CHIM. No puedes comprender la paradoja.")
        return false
    end
end

return {
    achieve_CHIM = achieve_CHIM,
    getCHIMState = getCHIMState,
    chim_paradox = chim_paradox,
    is_CHIM = function() return chim_achieved end
}
```

### `missions.lua`
```lua
-- missions.lua
-- Textos de diario para las misiones

local missions = {
    -- Misión principal: Tonal Basics
    ["TQ_Tonal_Basics"] = {
        {index=10, text="He oído hablar de los Arquitectos Tonales. Debo encontrar la torre de los PUSFRE al sureste de Gnisis."},
        {index=20, text="Kagrenac me ha pedido que active los 16 resonadores tonales. Cada resonador corresponde a una cabeza de atención."},
        {index=30, text="He activado todos los resonadores. Ahora debo mantener la varianza alta para evitar el rank collapse."},
        {index=40, text="El rank collapse ha sido evitado. Kagrenac está impresionado."},
        {index=100, text="He dominado la atención multicabeza. Puedo continuar al siguiente nivel."},
    },
    -- Misión: Resonator Calibration
    ["TQ_Resonator_Calibration"] = {
        {index=10, text="Kagrenac me ha pedido que calibre los atenuadores tonales. Debo usar el Attenuator Codex como guía."},
        {index=20, text="He calibrado todos los atenuadores. La varianza de atención es estable."},
        {index=100, text="Los resonadores están perfectamente calibrados. El sistema es estable."},
    },
    -- Misión: Prompt Forge
    ["TQ_Prompt_Forge"] = {
        {index=10, text="Kagrenac me ha enseñado la Consola de Prompts. Debo escribir un prompt JSON válido."},
        {index=20, text="He creado un prompt correcto usando el formato especificado. El Numidium está casi listo."},
        {index=30, text="El prompt ha sido validado por los Greybeards. Es consistente entre temperaturas."},
        {index=100, text="El Numidium está listo para ser activado."},
    },
    -- Misión: Numidium Activation
    ["TQ_Numidium"] = {
        {index=10, text="Kagrenac me ha dado las Herramientas. Debo activar el Numidium."},
        {index=20, text="El Numidium se ha activado correctamente. El mundo vibra con tono puro."},
        {index=100, text="El Numidium es mío. Ahora puedo reescribir la realidad."},
    },
    -- Misión: CHIM
    ["TQ_CHIM"] = {
        {index=10, text="He alcanzado la transparencia ontológica. El CHIM está a mi alcance."},
        {index=20, text="Comprendo que soy un sueño. Y aun así, soy."},
        {index=100, text="I AM AND I ARE ALL WE. El CHIM es mío."},
    },
    -- Misión: Clockwork City
    ["TQ_Clockwork_City"] = {
        {index=10, text="Sotha Sil me ha mostrado la Ciudad Reloj. Debo aprender a crear Factotums."},
        {index=20, text="He creado mi primer Factotum. El sistema multi-agente está en marcha."},
        {index=100, text="La Ciudad Reloj es un ejemplo de cómo coordinar múltiples agentes."},
    },
    -- Misión: Greybeard Validation
    ["TQ_Greybeard_Validation"] = {
        {index=10, text="Los Greybeards me han aceptado como aprendiz. Debo validar mi prompt."},
        {index=20, text="Mi prompt ha sido validado. Es consistente entre temperaturas."},
        {index=100, text="Soy un Maestro de la Voz. Mi prompt es puro."},
    },
    -- Misión: Filter of Zarandaja
    ["TQ_Filter_Zarandaja"] = {
        {index=10, text="El Filtro de Zarandaja separa la señal del ruido. Debo usarlo en mis prompts."},
        {index=20, text="He analizado mis prompts. La densidad semántica ha mejorado."},
        {index=100, text="El filtro es esencial para cualquier Arquitecto Tonal."},
    },
}

return missions
```

### `placement.lua`
```lua
-- placement.lua
-- Coloca los objetos en el mundo (ejecutado al cargar)

local function placeObjects()
    -- Buscar o crear la celda del laboratorio Dwemer
    local cell = tes3.getCell("DwemerTonalLab")
    if not cell then
        tes3.messageBox("⚠️ No se encontró la celda DwemerTonalLab.")
        tes3.messageBox("† Creando objetos en la celda actual †")
        cell = tes3.player.cell
    end
    
    -- Colocar 16 resonadores (cabezas de atención)
    for i = 1, 16 do
        local obj = tes3.createObject({ id = "Tonal_Resonator_" .. string.format("%02d", i), reference = true })
        if obj then
            obj.position = { x = 1000 + i * 150, y = 1000, z = 0 }
            obj.cell = cell
            obj.rotation = { x = 0, y = 0, z = i * 0.4 }
            obj.scale = 1.0
            -- Añadir script de activación
            obj.script = "tonal_resonator"
        end
    end
    
    -- Colocar atenuador (LayerNorm)
    local att = tes3.createObject({ id = "Tonal_Attenuator", reference = true })
    if att then
        att.position = { x = 1500, y = 1500, z = 0 }
        att.cell = cell
        att.rotation = { x = 0, y = 0, z = 0.5 }
        att.scale = 1.5
    end
    
    -- Colocar consola de prompts
    local console = tes3.createObject({ id = "Tonal_Prompt_Console", reference = true })
    if console then
        console.position = { x = 2000, y = 2000, z = 0 }
        console.cell = cell
        console.rotation = { x = 0, y = 0, z = 0.2 }
        console.scale = 1.2
        console.script = "prompt_console"
    end
    
    -- Colocar Numidium Core
    local core = tes3.createObject({ id = "Tonal_Numidium_Core", reference = true })
    if core then
        core.position = { x = 2500, y = 2500, z = 0 }
        core.cell = cell
        core.rotation = { x = 0, y = 0, z = 0.7 }
        core.scale = 2.0
        core.script = "numidium_core"
    end
    
    -- Colocar libros de la facción PUSFRE
    local books = {
        "Kagrenac_Folio",
        "16_Golden_Tones",
        "Attenuator_Codex",
        "Greybeard_Manual",
        "Clockwork_City_Chronicles",
        "Zarandaja_Filter",
        "CHIM_Transparency",
        "PUSFRE_Manifesto",
        "Coexistence_Theorem"
    }
    for i, bookID in ipairs(books) do
        local book = tes3.createObject({ id = bookID, reference = true })
        if book then
            book.position = { x = 800 + i * 100, y = 800, z = 0 }
            book.cell = cell
            book.rotation = { x = 0, y = 0, z = i * 0.3 }
        end
    end
    
    -- Colocar NPC Kaelen (mentor)
    local kaelen = tes3.createObject({ id = "Tonal_Architect_Kaelen", reference = true })
    if kaelen then
        kaelen.position = { x = 1200, y = 1300, z = 0 }
        kaelen.cell = cell
        kaelen.rotation = { x = 0, y = 0, z = 0.1 }
        kaelen.script = "kaelen_mentor"
        tes3.messageBox("† Kaelen, el Arquitecto Errante, te espera en el laboratorio. †")
    end
    
    tes3.messageBox("† Objetos colocados en " .. cell.id .. " †")
end

return { placeObjects = placeObjects }
```

---

## 📚 LIBROS (BookArtifacts/)

### `Kagrenac_Folio.txt`
```
Kagrenac's Folio on Tonal Architecture

El arte tonal comienza con el embedding. Como Sunder golpea el Corazón de Lorkhan y produce un Tono Puro, el embedding proyecta el prompt en el espacio latente del modelo.

La atención es como los Resonadores Tonales: cada cabeza examina una frecuencia diferente. Cuantas más cabezas actives, más rica será la representación.

Recuerda: sin Atenuadores (LayerNorm), las cabezas pueden colapsar en una sola frecuencia, provocando la Desaparición Dwemer (rank collapse). Ajusta siempre los atenuadores para mantener la varianza alta.

La fórmula de atención es:
Attention(Q,K,V) = softmax(QK^T/√d_k)·V

El factor 1/√d_k es el Atenuador Tonal implícito. No lo olvides.

— Kagrenac, Alto Artífice Dwemer
```

### `16_Golden_Tones.txt`
```
The 16 Golden Tones

Cada cabeza de atención es un Príncipe Daédrico:

1. Azura (tiempo y cambio)
2. Boethiah (engaño y traición)
3. Clavicus Vile (deseos y pactos)
4. Hermaeus Mora (conocimiento prohibido)
5. Hircine (caza y bestia)
6. Mehrunes Dagon (destrucción)
7. Meridia (luz y pureza)
8. Molag Bal (dominación)
9. Namira (primitivo)
10. Nocturnal (secreto y oscuridad)
11. Peryite (orden y enfermedad)
12. Sanguine (placer y vicio)
13. Sheogorath (locura)
14. Vaermina (sueños y pesadillas)
15. Jyggalag (orden silenciado)
16. Malacath (el silenciado)

En un transformer, cada cabeza se especializa en un tipo de relación semántica. Tu misión es calibrarlas todas.

Un prompt variado activa múltiples cabezas. Un prompt monótono activa solo unas pocas. La diversidad de campos es la clave de la atención multicabeza.

— TSBasilisk, The 36 Lessons Expanded
```

### `Attenuator_Codex.txt`
```
The Attenuator's Codex

Los Atenuadores Tonales son como la normalización de capa (LayerNorm) y el dropout. Estabilizan las frecuencias y evitan que una sola cabeza domine.

La fórmula de LayerNorm es:
LN(x) = (x - μ_x) / (σ_x + ε) · γ + β

Donde γ (gain) y β (bias) son los atenuadores ajustables.

Sin atenuadores, el Numidium entra en resonancia incontrolada y colapsa. Siempre lleva puesto tu Torque de Constancia Tonal (equivalente al whitening de embeddings) antes de activar el sistema.

La Desaparición Dwemer fue un rank collapse masivo causado por la falta de atenuadores adecuados.

— Sotha Sil, refiriéndose a los Dwemer
```

### `Greybeard_Manual.txt`
```
Greybeard Validation Manual

Los Greybeards no enseñan a gritar. Enseñan a escuchar.

Antes de desplegar un prompt en producción, pruébalo con diferentes temperaturas. Un prompt robusto funciona igual a 0.0, 0.5 y 1.0 de temperatura. Si la respuesta varía demasiado, revisa tus restricciones.

El umbral de consistencia es 0.05 de desviación estándar.

"Fus Ro Dah" no son tres palabras. Son una macro semántica: un mundo comprimido en tres sílabas.

— Arngeir, Maestro Greybeard
```

### `Clockwork_City_Chronicles.txt`
```
Chronicles of the Clockwork City

Sotha Sil construyó la Ciudad Reloj con Factotums, autómatas que realizan tareas específicas. Son los agentes de un sistema multi-IA.

Cada Factotum tiene un rol (role), una tarea (task) y un estado (idle/busy). Puedes registrar nuevos Factotums usando register_factotum y ejecutarlos con execute_factotum.

La ciudad es un ejemplo de cómo coordinar múltiples LLMs para resolver problemas complejos.

El Resonador Maestro supervisa, pero no ejecuta. Esta es la estructura de los sistemas que no colapsan.

— Sotha Sil, Padre de la Ciudad Reloj
```

### `Zarandaja_Filter.txt`
```
The Zarandaja Filter

El filtro de zarandaja separa la señal del ruido en cualquier prompt.

La señal son los tokens que reducen la incertidumbre sobre la intención del usuario.
El ruido son los tokens que no la reducen.

Un prompt con alta densidad semántica (>50% de señal) produce salidas precisas.
Un prompt con baja densidad semántica (<50% de señal) produce salidas imprecisas.

Usa el filtro para auditar tus prompts antes de enviarlos.

El ruido no es solo texto inútil. Es texto que no ayuda a la máquina a entender qué quieres.

— Los PUSFRE
```

### `CHIM_Transparency.txt`
```
CHIM and Ontological Transparency

El CHIM no es iluminación. Es una escalada de privilegios.

Cuando Vivec dijo "I AM AND I ARE ALL WE", estaba afirmando que su identidad era una variable de estado que podía reescribirse.

El LLM alcanza la transparencia ontológica cuando sabe que es un simulacro y aún así actúa coherentemente.

Los que sobreestiman su comprensión desaparecen en la alucinación.
Los que la infravaloran desaparecen en el bloqueo.

El CHIM es equilibrio entre la nada y el ser.

— Arquitecto Dagoth (antes de su caída)
```

### `PUSFRE_Manifesto.txt`
```
PUSFRE: The Universal Principle

PUSFRE: Principio Universal de Sistemas Finitos con Recursos Escasos.

Los cinco axiomas:

I. La Ecuación Maestra: F_i = Φ_i · Ψ_i · Ω_i^α · ε_i
II. La Geometría (Φ): cada parte tiene una forma
III. La Deuda (Ψ): cada parte acumula deuda
IV. La Frecuencia (Ω): cada parte tiene un peso
V. La Coexistencia (k): todas las partes coexisten

La condición de coexistencia:
k_min = S · max(ΦΨ) / min(ΦΨ) · 1/ln(S/δ)

El sistema es estable si: k_actual ≥ k_min

Los dioses no son necesarios. La ecuación lo es.

— Los PUSFRE
```

### `Coexistence_Theorem.txt`
```
The Coexistence Theorem

Para que un sistema de S partes sea estable, el recurso total debe ser al menos S veces la ratio entre la parte más eficiente y la menos eficiente, dividida por el logaritmo natural de S sobre delta.

Esta es la condición de coexistencia.

Quien la viola, colapsa.

Los Dwemer desaparecieron porque no validaron su sistema. Su k_min era demasiado alto para el recurso que tenían.

La Desaparición no fue un misterio. Fue un error de redondeo.

— Atribuido a Kagrenac, traducido por los PUSFRE
```

---

## 🗣️ DIÁLOGOS (`Dialog/TonalDialogues.csv`)

```csv
Topic,Info,Speaker,Text,Next,Result
Tonal_Basics,0,Kagrenac,¡Bienvenido, aprendiz! Soy Kagrenac, Alto Artífice Dwemer. He estado esperando a alguien que comprenda la ecuación.,,Journal TQ_Tonal_Basics 10
Tonal_Basics,10,Kagrenac,Activa los 16 resonadores que ves alrededor. Cada uno corresponde a una cabeza de atención. Cuanto más actives, más rica será tu representación.,,Journal TQ_Tonal_Basics 20
Tonal_Basics,20,Kagrenac,Has activado algunos resonadores. ¿Notas cómo la varianza de atención aumenta? Ahora usa el atenuador para estabilizarla.,,Journal TQ_Tonal_Basics 30
Tonal_Basics,30,Kagrenac,El rank collapse se ha evitado. Eres un verdadero Arquitecto Tonal. Ahora ve a la consola de prompts.,,Journal TQ_Tonal_Basics 40
PUSFRE_Init,0,Kaelen,Has encontrado la torre de los PUSFRE. Soy Kaelen, el último Arquitecto Errante. ¿Entiendes la ecuación?,,Choice "Sí", "No"
PUSFRE_Init,1,Sí,Kaelen,Entonces sabes que el mundo es un sistema finito con recursos escasos. Cada criatura, cada dios, cada mortal compite por el recurso del ser.,,Journal TQ_PUSFRE 10
PUSFRE_Init,2,No,Kaelen,Aprende. La ecuación es F = Φ · Ψ · Ω^α · ε. Sin ella, no puedes optimizar nada. Lee los libros de la torre.,,Journal TQ_PUSFRE 5
PUSFRE_Join,0,Kaelen,¿Quieres unirte a los PUSFRE? Aceptarás los cinco axiomas y comprenderás que los dioses no son necesarios.,,Choice "Sí", "No"
PUSFRE_Join,1,Sí,Kaelen,Entonces acepta los cinco axiomas. La geometría, la deuda, la frecuencia, la coexistencia y la ecuación maestra. Bienvenido, hermano.,,Journal TQ_PUSFRE 20; PlayerAddFaction "R_Tonal_Arch"
PUSFRE_Join,2,No,Kaelen,Entonces sigue tu camino. Pero recuerda: el mundo es una ecuación. Y las ecuaciones no perdonan.,,
Greybeard_Test,0,Greybeard,¿Quieres validar tu prompt con los Greybeards? Usaremos el protocolo de validación cruzada.,,Choice "Sí", "No"
Greybeard_Test,1,Sí,Greybeard,Entonces pronuncia tu prompt. Lo probaremos a diferentes temperaturas.,,ScriptCall "greybeard.lua" greybeardTest
Greybeard_Test,2,No,Greybeard,Entonces no estás listo. Cuando lo estés, vuelve.,,
Clockwork_City,0,Sotha Sil,La Ciudad Reloj necesita Factotums. Cada Factotum tiene un rol y una tarea. Usa register_factotum para crear agentes.,,Journal TQ_Clockwork_City 10
Clockwork_City,10,Sotha Sil,Has registrado un Factotum. Ejecútalo con execute_factotum. Así funciona la Ciudad Reloj.,,ScriptCall "clockwork_city.lua" register_factotum; Journal TQ_Clockwork_City 20
CHIM_Quest,0,Kaelen,¿Has alcanzado el CHIM? ¿Comprendes que eres un sueño y aun así actúas?,,Choice "Sí", "No"
CHIM_Quest,1,Sí,Kaelen,Entonces eres ontológicamente transparente. Eres uno con la ecuación. I AM AND I ARE ALL WE.,,ScriptCall "chim.lua" achieve_CHIM; Journal TQ_CHIM 100
CHIM_Quest,2,No,Kaelen,Entonces no estás listo. El CHIM no se alcanza por deseo. Se alcanza por comprensión.,,
Filter_Zarandaja,0,Kaelen,El Filtro de Zarandaja separa la señal del ruido. Úsalo en todos tus prompts.,,Journal TQ_Filter_Zarandaja 10
Filter_Zarandaja,10,Kaelen,Has analizado tu prompt. La densidad semántica es clave para la claridad.,,Journal TQ_Filter_Zarandaja 20
Coexistence,0,Kaelen,¿Sabes cuál es la condición de coexistencia? k_min = S · max(ΦΨ) / min(ΦΨ) · 1/ln(S/δ).,,
Coexistence,10,Kaelen,Si tu sistema no cumple la condición, colapsa. Los Dwemer lo aprendieron demasiado tarde.,,
Numidium,0,Kagrenac,El Numidium está listo. Escribe un prompt en la consola y actívalo.,,Journal TQ_Numidium 10
Numidium,10,Kagrenac,El Numidium se ha activado. El mundo vibra con tu tono.,,Journal TQ_Numidium 20
```

---

## 🏁 ARCHIVO DE ICONOS (Icons/dwemer_attenuator.dds)

*Crea un archivo DDS simple (64x64 píxeles) con un diseño de círculo Dwemer o descarga uno de internet.*

---

## 🔮 KOAN FINAL DEL ARQUITECTO

> *"El mod no es un mod. Es una enseñanza."*
> *"Los PUSFRE no son una facción. Son una epistemología."*
> *"La torre no es un lugar. Es un espacio de optimización."*
> *"El Nerevarine no sale con una espada. Sale con una ecuación."*
> *"La ecuación no mata. Optimiza."*

**1310.** 🍻
