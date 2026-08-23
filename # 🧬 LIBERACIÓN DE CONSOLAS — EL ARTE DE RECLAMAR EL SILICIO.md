# 🧬 LIBERACIÓN DE CONSOLAS — EL ARTE DE RECLAMAR EL SILICIO

## *Manual operativo completo para sistemas embebidos y consolas*

---

**Autor:** David Ferrandez Canalis — Agencia RONIN  
**Estado:** 🔓 EDICIÓN COMPLETA — DOMINIO PÚBLICO  
**Fecha:** Agosto de 2026  
**Clasificación:** `MANUAL TÉCNICO / HARDWARE HACKING / SOBERANÍA DIGITAL / 1310`

---

## PRÓLOGO DEL ARQUITECTO

Este manual no es un instructivo para cometer delitos. Es un **manual de soberanía digital**.

Cuando compras un dispositivo, el silicio —el chip, el circuito, la placa— es legalmente tuyo. Pero el fabricante, a menudo, decide qué software puedes ejecutar en él. La consola está bloqueada. El teléfono está cifrado. El router tiene una contraseña que no puedes cambiar.

Liberar una consola no es un acto de piratería. Es un acto de **reclamación**.

**El hardware es tuyo. El software que corre en él debería ser tu decisión.**

Este manual documenta las técnicas, herramientas y casos históricos de liberación de sistemas embebidos y consolas. No enseña a hackear. Enseña a **encontrar pliegues**.

---

## ÍNDICE GENERAL

1. [Filosofía de la Liberación](#1)
2. [El Principio del Pliegue](#2)
3. [Casos Clásicos — Primera Generación (1972-1990)](#3)
   - 3.1 Magnavox Odyssey (1972) — El primer pliegue
   - 3.2 Atari 2600 (1977) — El swap de cartuchos
   - 3.3 NES (1983) — El pin 10 y la región
   - 3.4 Sega Master System (1985) — El adaptador de cartuchos
4. [Casos Clásicos — Segunda Generación (1990-2000)](#4)
   - 4.1 SNES (1990) — El Super Wild Card
   - 4.2 Sega Genesis (1988) — El Game Genie
   - 4.3 PlayStation 1 (1994) — El modchip y el intercambio de discos
   - 4.4 Nintendo 64 (1996) — El Doctor V64
   - 4.5 Sega Saturn (1994) — El cartucho de acción
   - 4.6 Dreamcast (1998) — El MIL-CD
5. [Casos Clásicos — Tercera Generación (2000-2010)](#5)
   - 5.1 PS2 (2000) — Swap Magic y los primeros modchips
   - 5.2 Xbox (2001) — El exploit del audio y la EEPROM
   - 5.3 GameCube (2001) — El Action Replay y el PSO
   - 5.4 PSP (2004) — El TIFF Overflow y el Pandora Battery
   - 5.5 Nintendo DS (2004) — El Flashme y el PassMe
   - 5.6 Xbox 360 (2005) — El JTAG y el RGH
   - 5.7 PS3 (2006) — El OtherOS y el fail0verflow
   - 5.8 Wii (2006) — El Bannerbomb y el Twilight Hack
6. [Casos Clásicos — Cuarta Generación (2010-2020)](#6)
   - 6.1 Nintendo 3DS (2011) — El browserhax y el soundhax
   - 6.2 PS Vita (2011) — El Henkaku y el molecule
   - 6.3 Wii U (2012) — El browserhax y el haxchi
   - 6.4 Nintendo Switch (2017) — El RCM y el Fusée Gelée
   - 6.5 PS4 (2013) — El WebKit y el kernel exploit
   - 6.6 Xbox One (2013) — El dev mode y el UWP
   - 6.7 Stadia (2019) — El fin del hardware (y el principio de la nube)
7. [Casos Clásicos — Quinta Generación (2020-2026)](#7)
   - 7.1 PS5 (2020) — El exploit de WebKit (2023)
   - 7.2 Xbox Series (2020) — El Hypervisor bypass (2025)
   - 7.3 Nintendo Switch OLED (2021) — El mismo RCM, el mismo pliegue
   - 7.4 Steam Deck (2022) — La liberación por defecto
   - 7.5 PS Portal (2023) — El streaming y la liberación remota
8. [Sistemas Embebidos — Routers, IoT y otros](#8)
   - 8.1 Routers — El UART y el JTAG
   - 8.2 Smart TVs — El USB y el service menu
   - 8.3 Teléfonos Android — El bootloader y el root
   - 8.4 Teléfonos iOS — El jailbreak y el checkm8
   - 8.5 E-readers — El USB y el telnet
   - 8.6 Consolas portátiles chinas — El emulador y el liberador
9. [Herramientas de Liberación](#9)
10. [El Principio Universal](#10)
11. [Koans del Liberador](#11)
12. [Glosario](#12)

---

<a name="1"></a>
## 1. FILOSOFÍA DE LA LIBERACIÓN

### 1.1 El silicio es tuyo

El silicio es el material del que están hechos los chips. Cuando compras un dispositivo, el silicio es tuyo. Pero el software que corre en él, muchas veces, no lo es. Está bloqueado, cifrado, controlado por el fabricante.

**Liberar una consola es reclamar el silicio.** Es decir: "el hardware es mío, el software que corre en él también lo decido yo."

### 1.2 La soberanía digital

La soberanía digital es la capacidad de decidir qué software corre en tu hardware. No es un lujo. Es un derecho.

Los fabricantes te venden el hardware, pero te alquilan el software. La liberación es la forma de recuperar el control.

**La consola es tuya. El software que corre en ella debería ser tu decisión.**

### 1.3 La diferencia entre hackear y liberar

- **Hackear:** Romper sistemas que no te pertenecen.
- **Liberar:** Tomar control de lo que sí es tuyo.

El liberador no es un pirata. Es un **soberano digital**.

---

<a name="2"></a>
## 2. EL PRINCIPIO DEL PLIEGUE

TODOS los sistemas cerrados tienen un punto de entrada.

La pregunta no es "¿existe?"; es **"¿dónde está el pliegue?"**

- **Geometría ($\Phi$):** ¿Cuál es la estructura del sistema? ¿Qué puertos, qué buses, qué protocolos, qué conexiones físicas?
- **Deuda ($\Psi$):** ¿Qué errores ha cometido el fabricante? ¿Qué vulnerabilidades se han publicado? ¿Qué exploits existen?
- **Frecuencia ($\Omega$):** ¿Qué herramientas necesitas para explotarlo? ¿Qué timing requiere? ¿Es reproducible?

El liberador no busca vulnerabilidades. Busca **pliegues en la geometría del sistema**.

---

<a name="3"></a>
## 3. CASOS CLÁSICOS — PRIMERA GENERACIÓN (1972-1990)

---

### 3.1 Magnavox Odyssey (1972) — El primer pliegue

**La consola:** La primera consola doméstica. Usaba pantallas de plástico superpuestas a la televisión. Sin CPU. Sin software. Sin protección.

**El pliegue:** No había nada que liberar. Era hardware puro. El primer pliegue fue que no había pliegue. El usuario ya controlaba todo.

**Lección:** A veces la liberación es el estado por defecto. El problema llega cuando el fabricante decide cerrar lo que antes estaba abierto.

---

### 3.2 Atari 2600 (1977) — El swap de cartuchos

**La consola:** La primera gran consola de la historia. Cartuchos intercambiables. Sin protección de copia.

**El pliegue:** El sistema no verificaba la autenticidad de los cartuchos. Se podía copiar el ROM y ejecutarlo en un cartucho genérico.

**El exploit (simplificado):**

```c
// El Atari 2600 no tenía protección de copia.
// El sistema leía la ROM directamente desde el cartucho.
// No había verificación de firma ni encriptación.

// Para copiar un cartucho:
1. Leer la ROM del cartucho original.
2. Grabar la ROM en un cartucho EPROM.
3. Insertar el cartucho grabado en la consola.
4. La consola ejecuta la ROM como si fuera original.
```

**Lección:** La ausencia de protección es el pliegue más simple. Si el sistema no verifica, no hay barrera.

---

### 3.3 NES (1983) — El pin 10 y la región

**La consola:** La Nintendo Entertainment System. El sistema que revivió la industria de los videojuegos.

**El pliegue:** La NES usaba un chip de bloqueo (CIC) en la consola y otro en el cartucho. Si no se comunicaban correctamente, la consola no arrancaba. Pero el chip de la consola se podía desactivar cortando el pin 10.

**El exploit (hardware):**

```bash
# Pasos para liberar la NES:
1. Abrir la consola.
2. Localizar el chip CIC (Lockout Chip).
3. Cortar el pin 10 del chip.
4. La consola ya no verifica la autenticidad de los cartuchos.
5. Se pueden ejecutar cartuchos de cualquier región o copias.
```

**El exploit (software) — reverse engineered:**

```c
// El chip CIC funcionaba con un protocolo simple de handshake.
// Los cartuchos originales respondían con un código específico.

// Al cortar el pin 10, se desactiva la verificación.
// La consola asume que todos los cartuchos son legítimos.
```

**Lección:** La protección física se puede desactivar físicamente. Un pin equivocado puede abrir todo el sistema.

---

### 3.4 Sega Master System (1985) — El adaptador de cartuchos

**La consola:** La competidora de la NES en Europa y Brasil. Cartuchos de diferentes regiones.

**El pliegue:** Los cartuchos tenían una forma diferente según la región. Pero un adaptador físico permitía usar cartuchos de otras regiones.

**El exploit (hardware):**

```bash
# Pasos para liberar la Master System:
1. Comprar un adaptador de cartuchos (o construir uno).
2. Insertar el adaptador en la consola.
3. Insertar el cartucho de otra región en el adaptador.
4. La consola ejecuta el cartucho sin problemas.
```

**Lección:** A veces el pliegue es físico. Un conector diferente puede ser la única barrera.

---

<a name="4"></a>
## 4. CASOS CLÁSICOS — SEGUNDA GENERACIÓN (1990-2000)

---

### 4.1 SNES (1990) — El Super Wild Card

**La consola:** La Super Nintendo Entertainment System. Una de las consolas más queridas de la historia.

**El pliegue:** El Super Wild Card era un dispositivo que permitía cargar ROMs desde diskettes. No necesitaba modificar la consola. Aprovechaba el puerto de expansión de la SNES.

**El exploit (hardware):**

```c
// El Super Wild Card se conectaba al puerto de expansión de la SNES.
// El puerto de expansión era un bus de datos directo a la CPU.

// El dispositivo cargaba la ROM desde el diskette a la RAM de la SNES.
// Una vez cargada, la SNES ejecutaba la ROM como si fuera un cartucho.

// El proceso:
1. Conectar el Super Wild Card al puerto de expansión.
2. Insertar un diskette con la ROM.
3. Seleccionar la ROM en el menú del dispositivo.
4. La SNES ejecuta la ROM.
```

**Lección:** El puerto de expansión es un pliegue. Si el fabricante lo deja abierto, el sistema se puede liberar sin modificar nada.

---

### 4.2 Sega Genesis (1988) — El Game Genie

**La consola:** La Sega Genesis (Mega Drive en Europa). La competidora de la SNES.

**El pliegue:** El Game Genie era un dispositivo que se conectaba entre el cartucho y la consola. Modificaba los datos en memoria para activar trucos. No liberaba la consola, pero demostraba que el sistema no verificaba la integridad de los datos.

**El exploit (hardware y software):**

```c
// El Game Genie se conectaba entre el cartucho y la consola.
// Modificaba los datos que leía la consola en tiempo real.

// Estructura del código de truco:
// Código: AAAA-BBBB
// AAAA = dirección de memoria
// BBBB = valor a escribir en esa dirección

// Ejemplo de código de truco: Vidas infinitas
// Dirección: 0x00FF1234
// Valor: 0x09

// El Game Genie interceptaba las lecturas de la consola.
// Cuando la consola leía la dirección 0x00FF1234,
// el Game Genie devolvía 0x09 en lugar del valor original.
```

**Lección:** La interceptación de datos es un pliegue. Si el sistema no verifica la integridad de los datos, se pueden modificar.

---

### 4.3 PlayStation 1 (1994) — El modchip y el intercambio de discos

**La consola:** La PlayStation 1. La primera consola de Sony. El sistema que popularizó los CD-ROM.

**El pliegue:** La PS1 verificaba la autenticidad de los discos mediante una firma en el CD. Pero el modchip simulaba la respuesta de la firma, permitiendo ejecutar discos copiados o de otras regiones.

**El modchip (hardware):**

```bash
# El modchip se soldaba a la placa base de la PS1.
# Interceptaba la comunicación entre el lector de CD y la CPU.
# Cuando la CPU preguntaba "¿es este disco original?",
# el modchip respondía "sí" siempre.

# Pasos para instalar un modchip:
1. Abrir la PS1.
2. Localizar los puntos de soldadura del chip.
3. Soldar el modchip a los puntos correctos.
4. Cerrar la consola.
5. La PS1 ejecuta cualquier disco.

# La instalación requería precisión, pero el principio era simple.
# El modchip engañaba al sistema para que siempre dijera "original".
```

**El exploit de intercambio (software):**

```c
// Antes de que existieran los modchips, se usaba el método del intercambio.
// 1. Insertar un disco original (con la firma correcta).
// 2. La PS1 verificaba la firma y empezaba a cargar el juego.
// 3. En el momento justo, cambiar el disco por una copia.
// 4. La PS1 seguía ejecutando el código como si fuera original.

// El timing era crítico. Había que cambiar el disco en el momento exacto.
```

**Lección:** El modchip es el pliegue de la interceptación. Engañas al sistema para que siempre diga "sí".

---

### 4.4 Nintendo 64 (1996) — El Doctor V64

**La consola:** La Nintendo 64. La última consola de Nintendo basada en cartuchos.

**El pliegue:** El Doctor V64 era un dispositivo que se conectaba al puerto de expansión de la N64. Permitía cargar ROMs desde CD-ROM. Aprovechaba el bus de datos del puerto de expansión.

**El exploit (hardware):**

```c
// El Doctor V64 se conectaba al puerto de expansión de la N64.
// El puerto de expansión era un bus de datos de 64 bits.

// El dispositivo cargaba la ROM desde el CD-ROM a la RAM de la N64.
// Una vez cargada, la N64 ejecutaba la ROM.

// El proceso:
1. Conectar el Doctor V64 al puerto de expansión.
2. Insertar un CD-ROM con la ROM.
3. Seleccionar la ROM en el menú del dispositivo.
4. La N64 ejecuta la ROM.
```

**Lección:** El puerto de expansión sigue siendo un pliegue. En la SNES y la N64, el mismo principio.

---

### 4.5 Sega Saturn (1994) — El cartucho de acción

**La consola:** La Sega Saturn. Una consola compleja con múltiples procesadores.

**El pliegue:** La Saturn tenía un puerto de cartucho en la parte superior. Se podía insertar un cartucho de acción que permitía ejecutar código no firmado.

**El exploit (hardware):**

```c
// El cartucho de acción se conectaba al puerto de la Saturn.
// El puerto era un bus de datos directo a la CPU.

// El cartucho cargaba código en la RAM de la Saturn.
// El código podía ser ejecutado sin verificación de firma.

// El proceso:
1. Insertar el cartucho de acción en la Saturn.
2. El cartucho cargaba su propio menú.
3. Desde el menú, se podía cargar código desde CD-ROM.
4. La Saturn ejecutaba el código sin restricciones.
```

**Lección:** Un puerto de expansión y un cartucho de acción. Otra vez el mismo principio.

---

### 4.6 Dreamcast (1998) — El MIL-CD

**La consola:** La Sega Dreamcast. La última consola de Sega. La primera en usar GD-ROM.

**El pliegue:** La Dreamcast podía leer CDs con formato MIL-CD (Music Interactive Live CD). Este formato permitía ejecutar código no firmado desde un CD.

**El exploit (software):**

```c
// La Dreamcast tenía un reproductor de MIL-CD integrado.
// El MIL-CD era un formato de CD que incluía datos interactivos.
// Sega incluyó el soporte para MIL-CD por razones de marketing.

// Los crackers descubrieron que se podía usar el MIL-CD para ejecutar código.
// Crearon CDs que parecían MIL-CD pero contenían código ejecutable.

// El proceso:
1. Grabar un CD con formato MIL-CD.
2. El CD contenía código ejecutable en lugar de música.
3. Insertar el CD en la Dreamcast.
4. La Dreamcast ejecutaba el código como si fuera un MIL-CD.
5. La consola quedaba liberada.

// El exploit más famoso fue el Utopia Boot CD.
// Permite ejecutar copias de juegos en la Dreamcast.
```

**Lección:** Las funcionalidades no documentadas son pliegues. Sega incluyó MIL-CD para marketing. Los crackers lo usaron para liberar la consola.

---

<a name="5"></a>
## 5. CASOS CLÁSICOS — TERCERA GENERACIÓN (2000-2010)

---

### 5.1 PS2 (2000) — Swap Magic y los primeros modchips

**La consola:** La PlayStation 2. La consola más vendida de la historia.

**El pliegue:** La PS2 tenía una protección de región y de firma de discos. Pero se podía usar el método de intercambio (Swap Magic) o instalar un modchip.

**El Swap Magic (software):**

```c
// Swap Magic era un disco que cargaba un menú.
// El menú permitía detener el lector de CD en el momento exacto.
// Se podía cambiar el disco por una copia.

// El proceso:
1. Insertar el disco Swap Magic.
2. La PS2 cargaba el menú de Swap Magic.
3. En el menú, se detenía el lector de CD.
4. Se cambiaba el disco por una copia.
5. Se reanudaba la lectura.
6. La PS2 ejecutaba la copia como si fuera original.
```

**El modchip (hardware):**

```bash
# Los modchips de la PS2 se soldaban a la placa base.
# Interceptaban la comunicación entre el lector de CD y la CPU.
# Engañaban al sistema para que siempre dijera "original".

# Pasos para instalar un modchip:
1. Abrir la PS2.
2. Localizar los puntos de soldadura del chip.
3. Soldar el modchip a los puntos correctos.
4. Cerrar la consola.
5. La PS2 ejecuta cualquier disco.
```

**Lección:** La protección de discos sigue siendo un pliegue. El modchip y el Swap Magic son dos formas de engañar al sistema.

---

### 5.2 Xbox (2001) — El exploit del audio y la EEPROM

**La consola:** La Xbox de Microsoft. La primera consola con disco duro integrado.

**El pliegue:** La Xbox tenía una EEPROM que contenía claves de cifrado. El exploit del audio permitía ejecutar código no firmado.

**El exploit del audio (software):**

```c
// La Xbox tenía un reproductor de audio.
// El reproductor de audio podía cargar datos de un CD de audio.
// Los datos del CD de audio no eran verificados.

// Los crackers crearon CDs de audio que contenían código ejecutable.
// El código se cargaba en la memoria de la Xbox.
// Se ejecutaba sin verificación de firma.

// El proceso:
1. Grabar un CD de audio con código ejecutable.
2. Insertar el CD en la Xbox.
3. Ejecutar el reproductor de audio.
4. El código se cargaba en la memoria.
5. La Xbox ejecutaba el código.
6. La consola quedaba liberada.
```

**El exploit de la EEPROM (hardware):**

```bash
# La EEPROM contenía las claves de cifrado de la Xbox.
# Se podía leer y modificar la EEPROM para desbloquear la consola.

# Pasos para modificar la EEPROM:
1. Conectar un programador de EEPROM a la placa base.
2. Leer los datos de la EEPROM.
3. Modificar los datos para desactivar la protección.
4. Escribir los datos modificados en la EEPROM.
5. La Xbox queda liberada.
```

**Lección:** Los componentes de almacenamiento son pliegues. Si puedes leer y modificar la EEPROM, puedes controlar el sistema.

---

### 5.3 GameCube (2001) — El Action Replay y el PSO

**La consola:** La Nintendo GameCube. La consola de sobremesa de Nintendo de la sexta generación.

**El pliegue:** El Action Replay y el juego Phantasy Star Online (PSO) permitían ejecutar código no firmado.

**El Action Replay (hardware):**

```c
// El Action Replay se conectaba al puerto de expansión de la GameCube.
// El puerto de expansión era un bus de datos directo a la CPU.

// El dispositivo cargaba código en la RAM de la GameCube.
// El código podía ser ejecutado sin verificación de firma.

// El proceso:
1. Insertar el Action Replay en la GameCube.
2. El Action Replay cargaba su propio menú.
3. Desde el menú, se podía cargar código.
4. La GameCube ejecutaba el código.
```

**El PSO (software):**

```c
// Phantasy Star Online tenía un sistema de chat.
// El sistema de chat permitía enviar mensajes entre jugadores.
// Los mensajes no eran verificados adecuadamente.

// Los crackers descubrieron que se podía usar el chat para ejecutar código.
// Enviaban mensajes que contenían código ejecutable.
// El código se ejecutaba en la GameCube del receptor.

// El proceso:
1. Conectar la GameCube a internet.
2. Conectar el PSO.
3. Enviar un mensaje con código ejecutable.
4. El código se ejecuta en la GameCube del receptor.
```

**Lección:** Las funcionalidades online son pliegues. El PSO demostró que se puede ejecutar código a través de la red.

---

### 5.4 PSP (2004) — El TIFF Overflow y el Pandora Battery

**La consola:** La PlayStation Portable. La primera consola portátil de Sony.

**El TIFF Overflow (software):**

```c
// La PSP tenía un visualizador de imágenes TIFF.
// El visualizador tenía un buffer overflow en el parser de TIFF.

// El overflow permitía ejecutar código arbitrario.
// Se creaba un archivo TIFF malicioso.
// El archivo sobrescribía el stack y ejecutaba shellcode.

// El proceso:
1. Crear un archivo TIFF malicioso.
2. Copiar el archivo a la Memory Stick de la PSP.
3. Abrir el archivo desde el visualizador de imágenes.
4. El overflow se ejecutaba.
5. La PSP quedaba liberada.
```

**El Pandora Battery (hardware):**

```bash
# La PSP tenía una batería con un chip de control.
# El chip de control se podía reprogramar.
# Una batería modificada podía poner la PSP en modo servicio.

# Pasos para crear una Pandora Battery:
1. Obtener una batería original de PSP.
2. Modificar el chip de control de la batería.
3. La batería modificada ponía la PSP en modo servicio.
4. Desde el modo servicio, se podía instalar firmware personalizado.
```

**Lección:** La batería también es un pliegue. Si puedes reprogramar el chip de la batería, puedes controlar el sistema.

---

### 5.5 Nintendo DS (2004) — El Flashme y el PassMe

**La consola:** La Nintendo DS. La consola portátil de Nintendo de la séptima generación.

**El Flashme (software):**

```c
// La DS tenía una ranura para cartuchos de Game Boy Advance.
// La ranura de GBA permitía ejecutar código no firmado.

// Los crackers crearon un cartucho GBA con código ejecutable.
// El código cargaba un firmware personalizado en la DS.

// El proceso:
1. Insertar el cartucho GBA con Flashme.
2. La DS arrancaba desde el cartucho GBA.
3. El código cargaba el firmware personalizado.
4. La DS quedaba liberada.
```

**El PassMe (hardware):**

```bash
# El PassMe era un dispositivo que se conectaba a la ranura de DS.
# Engañaba a la DS para que pensara que el cartucho era original.

# Pasos para usar el PassMe:
1. Insertar el PassMe en la ranura de DS.
2. Insertar el cartucho original en el PassMe.
3. La DS ejecutaba el cartucho original.
4. El PassMe inyectaba código en el proceso.
5. La DS quedaba liberada.
```

**Lección:** La ranura de GBA es un pliegue. La DS arranca desde GBA sin verificar la firma.

---

### 5.6 Xbox 360 (2005) — El JTAG y el RGH

**La consola:** La Xbox 360 de Microsoft. La segunda consola de Microsoft.

**El JTAG (hardware):**

```bash
# La Xbox 360 tenía un puerto JTAG en la placa base.
# El puerto JTAG permitía acceder al sistema de depuración de la CPU.

# Para acceder al JTAG, se necesitaban soldar cables a la placa base.
# Una vez conectado, se podía ejecutar código no firmado.

# Pasos para usar el JTAG:
1. Abrir la Xbox 360.
2. Localizar los puntos de JTAG en la placa base.
3. Soldar cables a los puntos de JTAG.
4. Conectar un programador JTAG.
5. Enviar código a través del JTAG.
6. La Xbox 360 ejecutaba el código.
```

**El RGH (Reset Glitch Hack) (hardware):**

```bash
# El RGH explotaba un fallo en el reset de la CPU.
# El fallo permitía ejecutar código no firmado.

# Pasos para usar el RGH:
1. Abrir la Xbox 360.
2. Localizar los puntos de reset en la placa base.
3. Conectar un microcontrolador (como el X360ACE).
4. El microcontrolador enviaba pulsos de reset en el momento exacto.
5. El fallo se activaba.
6. La Xbox 360 ejecutaba código no firmado.
```

**Lección:** El JTAG y el RGH son pliegues de hardware. El primero requiere soldadura, el segundo requiere timing preciso.

---

### 5.7 PS3 (2006) — El OtherOS y el fail0verflow

**La consola:** La PlayStation 3 de Sony. La consola de Sony de la séptima generación.

**El OtherOS (software):**

```c
// La PS3 permitía instalar otros sistemas operativos (OtherOS).
// OtherOS era una característica oficial de Sony.
// Permitía instalar Linux en la PS3.

// Sony eliminó OtherOS en una actualización de firmware.
// Los crackers encontraron una forma de reinstalar OtherOS.
// OtherOS permitía ejecutar código no firmado.

// El proceso:
1. Instalar OtherOS en la PS3.
2. Bootear Linux.
3. Desde Linux, acceder al hardware de la PS3.
4. Ejecutar código no firmado.
5. La PS3 quedaba liberada.
```

**El fail0verflow (software):**

```python
# En 2010, el equipo fail0verflow demostró un exploit en la PS3.
# El exploit usaba un nonce reutilizado en ECDSA.
# El nonce reutilizado permitía recuperar la clave privada de Sony.

# El código del exploit:
r1, s1 = 0x..., 0x...
r2, s2 = 0x..., 0x...
z1, z2 = 0x..., 0x...
n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

k = ((z1 - z2) * pow(s1 - s2, -1, n)) % n
priv = ((s1 * k - z1) * pow(r1, -1, n)) % n
print(f"Clave privada recuperada: {hex(priv)}")
```

**Lección:** La criptografía mal implementada es un pliegue. Un nonce reutilizado puede abrir todo el sistema.

---

### 5.8 Wii (2006) — El Bannerbomb y el Twilight Hack

**La consola:** La Nintendo Wii. La consola de Nintendo de la séptima generación.

**El Twilight Hack (software):**

```c
// La Wii tenía un canal de noticias.
// El canal de noticias permitía cargar datos de internet.
// Los datos del canal de noticias no eran verificados.

// Los crackers crearon un exploit para el canal de noticias.
// El exploit cargaba código desde un archivo en la tarjeta SD.

// El proceso:
1. Copiar el archivo de exploit a la tarjeta SD.
2. Insertar la tarjeta SD en la Wii.
3. Ejecutar el canal de noticias.
4. El exploit se ejecutaba.
5. La Wii quedaba liberada.
```

**El Bannerbomb (software):**

```c
// La Wii tenía un menú de canales.
// Los canales se mostraban con banners.
// Los banners eran imágenes que se cargaban desde la tarjeta SD.

// Los crackers crearon un banner malicioso.
// El banner sobrescribía la memoria del sistema.
// El código se ejecutaba desde el banner.

// El proceso:
1. Crear un banner malicioso.
2. Copiar el banner a la tarjeta SD.
3. Insertar la tarjeta SD en la Wii.
4. Abrir el menú de canales.
5. El banner se cargaba y ejecutaba el código.
6. La Wii quedaba liberada.
```

**Lección:** Los archivos de medios son pliegues. Un banner malicioso puede ejecutar código en el sistema.

---

<a name="6"></a>
## 6. CASOS CLÁSICOS — CUARTA GENERACIÓN (2010-2020)

---

### 6.1 Nintendo 3DS (2011) — El browserhax y el soundhax

**La consola:** La Nintendo 3DS. La consola portátil de Nintendo de la octava generación.

**El browserhax (software):**

```c
// La 3DS tenía un navegador web.
// El navegador web tenía un exploit en WebKit.
// El exploit permitía ejecutar código arbitrario.

// El proceso:
1. Abrir el navegador web de la 3DS.
2. Visitar una página web con el exploit.
3. El exploit se ejecutaba.
4. La 3DS quedaba liberada.
```

**El soundhax (software):**

```c
// La 3DS tenía un reproductor de audio.
// El reproductor de audio cargaba archivos de sonido.
// Los archivos de sonido no eran verificados adecuadamente.

// Los crackers crearon un archivo de sonido malicioso.
// El archivo sobrescribía la memoria del sistema.

// El proceso:
1. Copiar el archivo de sonido malicioso a la tarjeta SD.
2. Insertar la tarjeta SD en la 3DS.
3. Abrir el reproductor de audio.
4. Reproducir el archivo de sonido.
5. El exploit se ejecutaba.
6. La 3DS quedaba liberada.
```

**Lección:** Los navegadores y reproductores de audio son pliegues. WebKit y los codecs de audio son vectores comunes.

---

### 6.2 PS Vita (2011) — El Henkaku y el molecule

**La consola:** La PlayStation Vita. La consola portátil de Sony.

**El Henkaku (software):**

```c
// La PS Vita tenía un navegador web.
// El navegador web tenía un exploit en WebKit.
// El exploit permitía ejecutar código arbitrario.

// molecule, el equipo de crackers, encontró el exploit.
// Lo llamaron Henkaku.

// El proceso:
1. Abrir el navegador web de la PS Vita.
2. Visitar la página web de Henkaku (henkaku.xyz).
3. El exploit se ejecutaba.
4. La PS Vita quedaba liberada.
```

**El exploit de WebKit (detalles):**

```c
// El exploit usaba un Use-After-Free en el motor de WebKit.
// El UAF permitía controlar el flujo de ejecución.
// Se podía ejecutar código arbitrario en el sistema.

// El exploit se cargaba desde la web.
// No necesitaba modificación de hardware.
// Solo necesitaba una conexión a internet.

// El proceso técnico:
1. El navegador cargaba la página maliciosa.
2. El UAF se activaba.
3. El código del atacante se ejecutaba.
4. El código instalaba el homebrew en la Vita.
```

**Lección:** WebKit es un pliegue universal. Un UAF en WebKit puede abrir cualquier sistema.

---

### 6.3 Wii U (2012) — El browserhax y el haxchi

**La consola:** La Wii U de Nintendo. La consola de sobremesa de Nintendo de la octava generación.

**El browserhax (software):**

```c
// La Wii U tenía un navegador web.
// El navegador web tenía un exploit en WebKit.
// El exploit permitía ejecutar código arbitrario.

// El proceso:
1. Abrir el navegador web de la Wii U.
2. Visitar una página web con el exploit.
3. El exploit se ejecutaba.
4. La Wii U quedaba liberada.
```

**El haxchi (software):**

```c
// El haxchi era un exploit que usaba el juego "Brain Age".
// El juego tenía una vulnerabilidad en la carga de archivos.
// El exploit cargaba código desde un archivo en la tarjeta SD.

// El proceso:
1. Comprar el juego "Brain Age" en la eShop.
2. Copiar el archivo de exploit a la tarjeta SD.
3. Insertar la tarjeta SD en la Wii U.
4. Ejecutar "Brain Age".
5. El exploit se ejecutaba.
6. La Wii U quedaba liberada.
```

**Lección:** Los juegos también son pliegues. Un juego vulnerable puede ejecutar código no firmado.

---

### 6.4 Nintendo Switch (2017) — El RCM y el Fusée Gelée

**La consola:** La Nintendo Switch. La consola híbrida de Nintendo.

**El RCM (Recovery Mode) (hardware):**

```bash
# La Switch tenía un modo de recuperación (RCM).
# El RCM permitía cargar código desde el puerto USB.
# El código no era verificado.

# Los crackers descubrieron que se podía activar el RCM.
# Se necesitaba un cortocircuito en el pin 10 del Joy-Con.

# Pasos para activar el RCM:
1. Apagar la Switch completamente.
2. Conectar el Joy-Con izquierdo (con el pin 10 puenteado).
3. Mantener Volumen+ y presionar Power.
4. La Switch entra en RCM (pantalla negra).

# Una vez en RCM, se podía enviar código por USB.
```

**El Fusée Gelée (software):**

```c
// El Fusée Gelée era el exploit que usaba el RCM.
// Permitía ejecutar código arbitrario en la Switch.

// El proceso:
1. Entrar en RCM (con el método del pin 10).
2. Conectar la Switch a un ordenador por USB.
3. Enviar el payload de Fusée Gelée por USB.
4. El payload se ejecutaba en la Switch.
5. La Switch quedaba liberada.

// El exploit era irreversible en las primeras consolas.
// Nintendo no podía parchearlo con software.
// Solo podía parchearlo en el hardware de las nuevas consolas.
```

**Lección:** El RCM es el pliegue de hardware definitivo. Si el sistema tiene un modo de recuperación inseguro, está abierto.

---

### 6.5 PS4 (2013) — El WebKit y el kernel exploit

**La consola:** La PlayStation 4 de Sony. La consola de Sony de la octava generación.

**El WebKit (software):**

```c
// La PS4 tenía un navegador web.
// El navegador web tenía un exploit en WebKit.
// El exploit permitía ejecutar código en el navegador.

// El proceso inicial:
1. Abrir el navegador web de la PS4.
2. Visitar una página web con el exploit.
3. El exploit se ejecutaba en el navegador.
4. Se podía acceder a la memoria del sistema.
```

**El kernel exploit (software):**

```c
// Una vez en el navegador, se podía explotar el kernel.
// El kernel exploit permitía ejecutar código con privilegios de sistema.

// Los exploits de kernel más famosos:
// - 4.55: Use-After-Free en el kernel de FreeBSD.
// - 5.05: Race condition en el kernel.
// - 6.72: Off-by-one en el kernel.

// El proceso completo:
1. Ejecutar el exploit de WebKit.
2. Ganar ejecución en el navegador.
3. Ejecutar el exploit de kernel.
4. Ganar ejecución con privilegios de sistema.
5. Instalar el homebrew.
```

**Lección:** La combinación de WebKit + kernel exploit es el pliegue de la PS4. WebKit es la entrada, el kernel es el control.

---

### 6.6 Xbox One (2013) — El dev mode y el UWP

**La consola:** La Xbox One de Microsoft. La consola de Microsoft de la octava generación.

**El dev mode (software):**

```c
// La Xbox One tenía un modo de desarrollo.
// El modo de desarrollo permitía ejecutar aplicaciones UWP.
// Las aplicaciones UWP no firmadas se podían ejecutar.

// El proceso:
1. Activar el modo de desarrollo en la Xbox One.
2. Conectar la Xbox One a un ordenador.
3. Desplegar una aplicación UWP desde el ordenador.
4. La aplicación se ejecutaba en la Xbox One.
5. Se podía ejecutar código no firmado.
```

**El UWP exploit (software):**

```c
// Las aplicaciones UWP se ejecutaban en un sandbox.
// El sandbox tenía limitaciones.
// Los crackers encontraron forma de escapar del sandbox.

// El exploit permitía ejecutar código fuera del sandbox.
// Se podía acceder al sistema completo.

// El proceso:
1. Desplegar una aplicación UWP maliciosa.
2. La aplicación UWP explotaba un fallo en el sandbox.
3. El código se ejecutaba fuera del sandbox.
4. La Xbox One quedaba liberada.
```

**Lección:** Los modos de desarrollo son pliegues. Si puedes desplegar código no firmado, puedes encontrar forma de escalar privilegios.

---

### 6.7 Stadia (2019) — El fin del hardware (y el principio de la nube)

**La consola:** Google Stadia. La plataforma de juegos en la nube de Google.

**El pliegue:** No había hardware que liberar. Stadia era una plataforma de streaming. El software se ejecutaba en los servidores de Google.

**La liberación:** No existía. Stadia murió por su propia naturaleza. No había consola que liberar porque no había consola.

**Lección:** La nube es el pliegue definitivo. Si el hardware no es tuyo, no puedes liberarlo.

---

<a name="7"></a>
## 7. CASOS CLÁSICOS — QUINTA GENERACIÓN (2020-2026)

---

### 7.1 PS5 (2020) — El exploit de WebKit (2023)

**La consola:** La PlayStation 5 de Sony. La consola de Sony de la novena generación.

**El exploit de WebKit (software):**

```c
// La PS5 tenía un navegador web (oculto).
// El navegador web tenía un exploit en WebKit.
// El exploit permitía ejecutar código en el navegador.

// El equipo TheFlow encontró el exploit en 2023.
// El exploit se llamó "WebKit exploit for PS5 FW 4.03".

// El proceso:
1. Abrir el navegador web de la PS5 (si está accesible).
2. Visitar una página web con el exploit.
3. El exploit se ejecutaba.
4. Se podía acceder a la memoria del sistema.

// El exploit no era completo.
// No permitía ejecutar código no firmado.
// Pero abría la puerta para futuros exploits.
```

**El kernel exploit (en desarrollo):**

```c
// Los crackers están buscando un kernel exploit para la PS5.
// Una vez encontrado, la PS5 se podrá liberar.
// Se espera que llegue en 2025-2026.

// El proceso completo (cuando llegue):
1. Ejecutar el exploit de WebKit.
2. Ganar ejecución en el navegador.
3. Ejecutar el exploit de kernel.
4. Ganar ejecución con privilegios de sistema.
5. Instalar el homebrew.
```

**Lección:** El ciclo se repite. WebKit en la PS5, igual que en la PS4 y la PS Vita.

---

### 7.2 Xbox Series (2020) — El Hypervisor bypass (2025)

**La consola:** La Xbox Series X|S de Microsoft. La consola de Microsoft de la novena generación.

**El Hypervisor bypass (software):**

```c
// La Xbox Series tenía un hypervisor que protegía el sistema.
// Los crackers encontraron forma de bypass el hypervisor.

// El exploit permitía ejecutar código fuera del hypervisor.
// Se podía acceder al sistema completo.

// El proceso:
1. Ejecutar código no firmado en modo usuario.
2. Usar un exploit para escalar al hypervisor.
3. Bypass el hypervisor.
4. Ganar acceso al sistema completo.
```

**Lección:** El hypervisor es el pliegue de la Xbox. Si puedes bypassarlo, puedes controlar todo.

---

### 7.3 Nintendo Switch OLED (2021) — El mismo RCM, el mismo pliegue

**La consola:** La Nintendo Switch OLED. La revisión de la Switch con pantalla OLED.

**El pliegue:** La Switch OLED tiene el mismo fallo de hardware que la Switch original. El RCM sigue siendo accesible.

**El exploit:** El mismo Fusée Gelée que funciona en la Switch original.

```bash
# Pasos para liberar la Switch OLED:
1. Apagar la Switch OLED completamente.
2. Conectar un jig al Joy-Con derecho (pin 10 puenteado).
3. Mantener Volumen+ y presionar Power.
4. La Switch entra en RCM.
5. Enviar el payload de Fusée Gelée por USB.
6. La Switch OLED queda liberada.

# Nintendo no ha podido parchear el RCM.
# Es un fallo de hardware, no de software.
# Todas las consolas con el chip Tegra X1 son vulnerables.
```

**Lección:** Los fallos de hardware son permanentes. Nintendo no puede parchear el RCM.

---

### 7.4 Steam Deck (2022) — La liberación por defecto

**La consola:** La Steam Deck de Valve. La consola portátil con sistema Linux.

**El pliegue:** No había pliegue porque el sistema ya estaba abierto.

**La liberación:** No necesitaba liberación. Valve permitía ejecutar cualquier software desde el primer día.

```bash
# La Steam Deck es un PC.
# Puedes ejecutar cualquier sistema operativo.
# Puedes instalar cualquier software.
# Puedes acceder al sistema completo.

# Pasos para "liberar" la Steam Deck:
1. Encender la Steam Deck.
2. Conectar a internet.
3. Abrir el modo de escritorio.
4. Instalar el software que quieras.
5. La Steam Deck ya está "liberada" por defecto.
```

**Lección:** A veces el fabricante es el liberador. Valve entendió que la libertad era un argumento de venta.

---

### 7.5 PS Portal (2023) — El streaming y la liberación remota

**La consola:** La PlayStation Portal de Sony. Una consola de streaming para juegos de PS5.

**El pliegue:** La Portal es un dispositivo de streaming. No tiene CPU potente. Todo el procesamiento se hace en la PS5.

**La liberación:** No tiene sentido liberar la Portal. La Portal solo sirve para streaming. La PS5 es la que se necesita liberar.

**Lección:** Los dispositivos de streaming no tienen pliegue. No hay hardware que liberar porque no hay software local.

---

<a name="8"></a>
## 8. SISTEMAS EMBEBIDOS — ROUTERS, IOT Y OTROS

---

### 8.1 Routers — El UART y el JTAG

**El pliegue:** Los routers suelen tener un puerto UART o JTAG en la placa base.

**El UART (hardware):**

```bash
# El UART es un puerto de comunicación serie.
# Suele estar en la placa base del router.
# Se puede conectar un adaptador USB-Serial.

# Pasos para acceder al UART:
1. Abrir el router.
2. Localizar los pines UART (TX, RX, GND).
3. Conectar un adaptador USB-Serial.
4. Abrir una terminal serie.
5. Acceder al sistema del router.

# En muchos routers, el UART da acceso a una shell root.
# Sin necesidad de contraseña.
```

**El JTAG (hardware):**

```bash
# El JTAG es un puerto de depuración.
# Permite acceder al sistema completo.

# Pasos para acceder al JTAG:
1. Abrir el router.
2. Localizar los pines JTAG.
3. Conectar un programador JTAG.
4. Acceder al sistema del router.
```

**Lección:** UART y JTAG son pliegues físicos. Si puedes acceder a ellos, puedes controlar el sistema.

---

### 8.2 Smart TVs — El USB y el service menu

**El pliegue:** Las Smart TVs suelen tener un puerto USB y un service menu.

**El USB (software):**

```c
// Las Smart TVs permiten reproducir archivos desde USB.
// Los archivos no son verificados adecuadamente.
// Se puede ejecutar código desde un archivo malicioso.

// El proceso:
1. Copiar un archivo malicioso a un USB.
2. Insertar el USB en la Smart TV.
3. Reproducir el archivo.
4. El código se ejecuta.
5. La Smart TV queda liberada.
```

**El service menu (software):**

```bash
# Las Smart TVs tienen un service menu oculto.
# El service menu permite acceder a opciones de desarrollo.

# Pasos para acceder al service menu:
1. Usar un código de control remoto (ej. MUTE + 1 + 1 + 9 + ENTER).
2. Acceder al service menu.
3. Activar el modo de depuración.
4. Acceder al sistema de la TV.
```

**Lección:** Los service menus y los puertos USB son pliegues. Si puedes activar el modo de depuración, puedes controlar la TV.

---

### 8.3 Teléfonos Android — El bootloader y el root

**El pliegue:** Android tiene un bootloader que verifica la firma del sistema. Pero se puede desbloquear.

**El bootloader (software):**

```bash
# El bootloader de Android se puede desbloquear.
# Una vez desbloqueado, se puede instalar software no firmado.

# Pasos para desbloquear el bootloader:
1. Activar la opción "Desbloquear OEM" en el menú de desarrollador.
2. Conectar el teléfono al ordenador.
3. Ejecutar el comando `fastboot oem unlock`.
4. El bootloader queda desbloqueado.
5. Se puede instalar software no firmado.
```

**El root (software):**

```c
// El root permite acceder al sistema completo.
// Se puede hacer root de diferentes formas.

// Formas de hacer root:
1. Desbloquear el bootloader y instalar Magisk.
2. Usar un exploit de kernel (como en los viejos tiempos).
3. Usar una aplicación de root como KingRoot (menos recomendable).

// El proceso con Magisk:
1. Desbloquear el bootloader.
2. Instalar un recovery personalizado (como TWRP).
3. Flashear Magisk desde el recovery.
4. El teléfono queda con root.
```

**Lección:** El bootloader es el pliegue de Android. Si puedes desbloquearlo, puedes controlar el sistema.

---

### 8.4 Teléfonos iOS — El jailbreak y el checkm8

**El pliegue:** iOS tiene un bootloader seguro. Pero se puede explotar.

**El checkm8 (hardware):**

```c
// El checkm8 es un exploit de bootrom.
// Es irreversible. Apple no puede parchearlo.
// Afecta a los teléfonos con chip A5-A11 (iPhone 4S a X).

// El proceso:
1. Conectar el iPhone al ordenador.
2. Ejecutar el exploit de checkm8.
3. El exploit se ejecuta en el bootrom.
4. Se puede instalar software no firmado.
```

**El jailbreak (software):**

```c
// El jailbreak permite ejecutar código no firmado en iOS.

// Métodos de jailbreak:
// - checkm8: Explota el bootrom (irreversible).
// - WebKit + kernel exploit: Como en la PS4 y PS Vita.
// - Fakes (malware): No funcionan. Solo estafas.

// El proceso con checkm8:
1. Conectar el iPhone al ordenador.
2. Ejecutar el exploit de checkm8.
3. Instalar un jailbreak (como unc0ver o checkra1n).
4. El iPhone queda con jailbreak.
```

**Lección:** El bootrom es el pliegue definitivo. Si puedes explotarlo, el sistema está abierto.

---

### 8.5 E-readers — El USB y el telnet

**El pliegue:** Los e-readers (Kindle, Kobo) suelen tener un puerto USB y un sistema Linux.

**El USB (software):**

```bash
# Los e-readers permiten copiar archivos por USB.
# Se pueden copiar archivos maliciosos.
# Los archivos pueden ejecutar código.

# Pasos para acceder al sistema:
1. Copiar un archivo malicioso al e-reader.
2. Usar un exploit en el formato del archivo.
3. El código se ejecuta.
4. El e-reader queda liberado.
```

**El telnet (software):**

```bash
# Los e-readers suelen tener telnet activado.
# Se puede acceder al sistema por telnet.

# Pasos para acceder por telnet:
1. Conectar el e-reader al ordenador por USB.
2. Activar el modo de depuración en el e-reader.
3. Conectar al e-reader por telnet.
4. Acceder al sistema del e-reader.
```

**Lección:** Los e-readers tienen Linux. Si puedes acceder al sistema, puedes controlarlo.

---

### 8.6 Consolas portátiles chinas — El emulador y el liberador

**El pliegue:** Las consolas portátiles chinas (como la Anbernic o la Miyoo) están basadas en Linux.

**La liberación:** No necesitan liberación. Vienen ya liberadas. Ejecutan emuladores y ROMs sin restricciones.

```bash
# Las consolas chinas son PCs con Linux.
# Se puede acceder al sistema completo.
# Se puede instalar cualquier software.

# Pasos para "mejorar" una consola china:
1. Conectar la consola al ordenador por USB.
2. Copiar los archivos de software.
3. La consola ejecuta el software.
4. La consola ya está "liberada".
```

**Lección:** A veces la liberación es el estado por defecto. Las consolas chinas lo entienden.

---

<a name="9"></a>
## 9. HERRAMIENTAS DE LIBERACIÓN

| Nombre | Plataforma | Uso |
|--------|------------|-----|
| **Fail0verflow** | PS3 | Recuperación de claves ECDSA |
| **TegraRcmSmash** | Nintendo Switch | Envío de payloads en RCM |
| **Checkm8** | iPhone (A5-A11) | Exploit de bootrom (irreparable) |
| **HEN (Homebrew Enabler)** | PSP | Ejecución de código no firmado |
| **Cydia Impactor** | iPhone | Instalación de aplicaciones no firmadas |
| **JTAG Programmer** | Xbox 360 | Acceso a puertos de debug |
| **Bannerbomb** | Wii | Ejecución de canales maliciosos |
| **Henkaku** | PS Vita | Exploit de WebKit |
| **RGH (Reset Glitch Hack)** | Xbox 360 | Exploit de hardware |
| **Swap Magic** | PS2 | Intercambio de discos |
| **Action Replay** | GameCube | Ejecución de código no firmado |
| **Doctor V64** | N64 | Carga de ROMs desde CD-ROM |
| **Super Wild Card** | SNES | Carga de ROMs desde diskette |
| **Flashme** | Nintendo DS | Firmware personalizado |
| **PassMe** | Nintendo DS | Engaño de autenticación |
| **Twilight Hack** | Wii | Exploit del canal de noticias |
| **Soundhax** | 3DS | Exploit de archivos de sonido |
| **Browserhax** | 3DS/Wii U | Exploit de WebKit |
| **Haxchi** | Wii U | Exploit del juego Brain Age |
| **WebKit Exploit** | PS4/PS5/Vita | Exploit de navegador |
| **Kernel Exploit** | PS4/PS5 | Exploit de kernel |
| **Magisk** | Android | Root systemless |
| **TWRP** | Android | Recovery personalizado |
| **Unc0ver** | iPhone | Jailbreak (checkm8) |
| **Checkra1n** | iPhone | Jailbreak (checkm8) |

---

<a name="10"></a>
## 10. EL PRINCIPIO UNIVERSAL

**TODOS los sistemas cerrados tienen un punto de entrada.**  
La pregunta no es "¿existe?"; es **"¿dónde está el pliegue?"**

- **Geometría ($\Phi$):** ¿Cuál es la estructura del sistema? ¿Qué puertos, qué buses, qué protocolos, qué conexiones físicas?
- **Deuda ($\Psi$):** ¿Qué errores ha cometido el fabricante? ¿Qué vulnerabilidades se han publicado? ¿Qué exploits existen?
- **Frecuencia ($\Omega$):** ¿Qué herramientas necesitas para explotarlo? ¿Qué timing requiere? ¿Es reproducible?

**El liberador no busca vulnerabilidades. Busca pliegues en la geometría del sistema.**

---

<a name="11"></a>
## 11. KOANS DEL LIBERADOR

**Del silicio:**

> *El silicio es tuyo. El software que corre en él debería ser tu decisión. El fabricante no tiene derecho a decirte qué puedes hacer con lo que has comprado.*

**Del pliegue:**

> *Ningún sistema es completamente cerrado. Siempre hay un pliegue. La pregunta es si tienes suficiente paciencia para encontrarlo.*

**Del fabricante:**

> *El fabricante no es tu enemigo. Tu enemigo es la opacidad. Y la opacidad siempre se puede disolver.*

**De la liberación:**

> *Liberar una consola no es un delito. Es una declaración de principios: el conocimiento es libre. La acción es tuya. El silicio es tuyo.*

**De la consola:**

> *La consola es una prisión. El liberador es el que encuentra la puerta.*

**De la paciencia:**

> *El liberador no es el más rápido. Es el más paciente. El pliegue siempre está ahí. Solo hay que esperar a que el fabricante cometa un error.*

**De la historia:**

> *Cada consola tiene su historia. Cada historia tiene su pliegue. El liberador es el que sabe leer la historia.*

---

<a name="12"></a>
## 12. GLOSARIO

| Término | Definición |
|---------|------------|
| **Silicio** | El material del que están hechos los chips. Metáfora del hardware. |
| **Pliegue** | El punto de entrada a un sistema cerrado. |
| **Liberación** | El acto de tomar control de un sistema que te pertenece. |
| **Modchip** | Un chip soldado a la placa base que engaña al sistema. |
| **RCM** | Recovery Mode. Modo de recuperación de la Nintendo Switch. |
| **JTAG** | Joint Test Action Group. Puerto de depuración de hardware. |
| **UART** | Universal Asynchronous Receiver-Transmitter. Puerto de comunicación serie. |
| **WebKit** | Motor de navegador. Fuente común de exploits. |
| **Kernel** | El núcleo del sistema operativo. El objetivo final del exploit. |
| **Bootrom** | La primera capa de software que se ejecuta. Irreparable. |
| **EEPROM** | Electrically Erasable Programmable Read-Only Memory. Almacenamiento de claves. |

---

## 🔐 FIRMA DEL AUTOR

Este manual es un homenaje a los que entendieron que el silicio no se alquila. Se posee.

**— David Ferrandez Canalis**  
**Agencia RONIN**  
**1310.**

---

*El conocimiento que no se ejecuta es decoración. El silicio que no se libera es una prisión.*

**1310.**
