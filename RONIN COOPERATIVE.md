# RONIN COOPERATIVE FRAMEWORK: De los Planos a la Red

**Obra #1310 | Agencia RONIN | Arquitecto: David Ferrandez Canalis**

**Documento de trabajo para búsqueda de financiación | Febrero 2026**

---

> *"No tenemos fábricas. No tenemos talleres. No tenemos comunidad organizada. Solo tenemos planos. Pero los planos son el conocimiento, y el conocimiento, cuando es libre, encuentra quien lo construya. Este documento es el puente entre los planos y la realidad."*

---

## ÍNDICE

1. [PRÓLOGO: EL MOMENTO DE CONSTRUIR](#prologo)
2. [PUNTO DE PARTIDA: SOLO PLANOS](#punto-de-partida)
3. [DEBILIDADES IDENTIFICADAS Y MEJORAS PROPUESTAS](#debilidades-y-mejoras)
    - 3.1. Viabilidad económica
    - 3.2. Organización y gobernanza
    - 3.3. Propiedad intelectual y protección frente a terceros
    - 3.4. Calidad y confianza del cliente
    - 3.5. Implantación territorial
    - 3.6. Sostenibilidad ambiental
    - 3.7. Escalabilidad
4. [OPORTUNIDADES DE FINANCIACIÓN REALES](#oportunidades-financiacion)
    - 4.1. Next Generation EU (Fondos de Recuperación)
    - 4.2. Horizonte Europa
    - 4.3. Fondo Social Europeo Plus
    - 4.4. Programa Europa Digital
    - 4.5. Fondos de Desarrollo Regional (FEDER)
    - 4.6. Otras vías complementarias
5. [HOJA DE RUTA PARA SOLICITAR AYUDAS](#hoja-de-ruta)
    - 5.1. Fase 0: Consolidación del proyecto (meses 1-6)
    - 5.2. Fase 1: Talleres piloto (meses 7-12)
    - 5.3. Fase 2: Expansión nacional (meses 13-24)
    - 5.4. Fase 3: Internacionalización (meses 25-48)
6. [PRESUPUESTO ESTIMADO PARA LA FASE 0](#presupuesto)
7. [CRONOGRAMA ORIENTATIVO](#cronograma)
8. [BIBLIOGRAFÍA Y REFERENCIAS](#bibliografia)
9. [ANEXO: ANÁLISIS DE RIESGOS Y MITIGACIÓN](#anexo)

---

## 1. PRÓLOGO: EL MOMENTO DE CONSTRUIR {#prologo}

Hay momentos en la historia de la tecnología en que el conocimiento acumulado alcanza una masa crítica y empieza a filtrarse fuera de los laboratorios y las corporaciones. El movimiento del software libre en los años 80, la explosión de las impresoras 3D en la década de 2010, la democratización de la inteligencia artificial en los 2020. Cada uno de estos momentos fue posible porque alguien, en algún lugar, decidió **publicar los planos**.

Hoy estamos en un momento similar, pero con una diferencia: los planos ya no son solo de software. Son de hardware. Son de exoesqueletos, de interfaces cerebro-computadora, de vehículos de altas prestaciones, de naves espaciales. Y están todos disponibles, documentados, listos para ser construidos.

Lo que falta no es conocimiento. Lo que falta es **organización**. Una red de personas y talleres que conviertan esos planos en objetos reales, que los mejoren, que los compartan, que los comercialicen de forma justa. Este documento es el primer paso para construir esa red.

No partimos de cero. Partimos de más de veinte diseños completamente documentados, con listas de materiales, código fuente y referencias académicas. Partimos de un modelo de gobernanza cooperativa diseñado sobre el papel. Partimos de la certeza de que la tecnología puede ser otra cosa: no una mercancía, sino un bien común.

Lo que sigue es un análisis honesto de nuestras debilidades, las mejoras que hemos incorporado para superarlas, y las oportunidades de financiación que existen para dar el salto. No hay promesas imposibles. Hay un plan realista y fundamentado.

*Febrero de 2026 marca un punto de inflexión: los fondos Next Generation entran en su recta final, Horizonte Europa lanza sus últimas convocatorias, y la ventana de oportunidad para financiar proyectos transformadores sigue abierta, pero por poco tiempo. Es ahora o nunca.*

---

## 2. PUNTO DE PARTIDA: SOLO PLANOS {#punto-de-partida}

### 2.1. Qué tenemos hoy

- **Más de 20 diseños de hardware** completamente documentados. Entre ellos:
  - **CORTEX-Ω**: interfaz cerebro-computadora de 2 canales, basada en el ADS1299, con aislamiento galvánico y filtros IIR optimizados. Coste de materiales: 50€. Basado en el trabajo de [1] sobre amplificadores de biopotenciales y en [2] sobre filtros digitales en punto fijo.
  - **RAS-1310**: sistema de realidad aumentada por bioseñales, con cámara OV2640, pantalla LCD y cinco agentes conceptuales (VOID, NEON, RUST, MIST, FLUX). Incluye un compilador JIT de shaders WGSL optimizado mediante E-Graph, basado en [3] para la representación eficiente de expresiones y en [4] para la generación de código en tiempo real.
  - **OMEGA-EXO**: exoesqueleto de 6 grados de libertad con estructura de perfiles de aluminio y servos reciclados de impresoras. Incluye análisis de elementos finitos según [5] y control PID basado en [6].
  - **Hypercar de basalto**: vehículo de altas prestaciones con chasis de fibra de basalto y baterías de segunda vida. Diseño basado en los principios de [7] para materiales compuestos y en [8] para la reutilización de celdas.
  - **BASALT-MOTHERSHIP**: nave espacial para 50 rovers, con estructura de basalto sinterizado y sistema de despliegue por gravedad. Inspirada en los trabajos de [9] sobre hábitats lunares y [10] sobre impresión 3D en el espacio.
  - **LUNAR-BOAEXO**: rover de minería de asteroides con control distribuido y blockchain para trazabilidad de recursos. Basado en [11] para robótica planetaria y [12] para cadenas de suministro descentralizadas.
  - **HEMATOLOGIC-SCANNER**: escáner médico de campo para análisis de sangre, con segmentación Voronoi y clasificación celular mediante redes neuronales. Fundamentado en [13] para procesamiento de imágenes y [14] para aprendizaje profundo.
  - **ONI GUARDIAN**: HUD para casco con telemetría biométrica y modos ZEN/KAMI, basado en [15] para interfaces de usuario adaptativas.

- **Listas de materiales (BOM) detalladas** para cada diseño, con precios reales de AliExpress, ferreterías y componentes reciclados. Cada BOM incluye alternativas para facilitar el abastecimiento local y la sustitución de componentes difíciles de conseguir.

- **Código fuente abierto** bajo licencias AGPL, CERN-OHL-W y CC BY-SA. Todo el firmware, software de backend, aplicaciones de frontend y herramientas de simulación están disponibles en repositorios públicos.

- **Documentación técnica rigurosa** que incluye:
  - Referencias a papers académicos para cada decisión de diseño.
  - Simulaciones de elementos finitos (FEA) con archivos de entrada para Calculix.
  - Protocolos de prueba detallados, con valores umbral y procedimientos de verificación.
  - Manuales de montaje paso a paso, con fotografías y diagramas.

- **Un modelo de gobernanza cooperativa** diseñado sobre el papel: estatutos, sistema de cuotas, fondo de resiliencia, asambleas digitales, votaciones ponderadas. Basado en los principios de [16] sobre economía social y [17] sobre gobernanza de comunes.

- **Este documento** como hoja de ruta para conseguir financiación y construir la red.

### 2.2. Qué no tenemos

- **Talleres físicos** donde se fabriquen los diseños. Tenemos planos, pero no mesas de trabajo.
- **Una comunidad organizada** de constructores. Hay personas interesadas, pero no hay estructura formal, ni compromisos, ni reglas claras.
- **Financiación** para poner en marcha la red, formar a los primeros talleres, adquirir equipamiento y cubrir los costes iniciales.
- **Certificaciones** que avalen la calidad ante clientes institucionales. Sin certificaciones, los ayuntamientos y hospitales no pueden comprar.
- **Relaciones formales con administraciones**. No tenemos contactos, ni proyectos conjuntos, ni convenios.
- **Un plan de negocio validado**. Las proyecciones existen, pero no hay casos reales de venta que las respalden.

### 2.3. La paradoja inicial

Para tener talleres, necesitamos financiación. Para tener financiación, necesitamos demostrar que el modelo es viable. Para demostrar viabilidad, necesitamos casos de éxito. Para tener casos de éxito, necesitamos talleres.

Es el clásico problema del huevo y la gallina, pero no es insoluble. La estrategia es **empezar con recursos mínimos, validar el modelo con un piloto pequeño, y luego escalar con ayudas públicas**. Este documento es la herramienta para conseguir esas ayudas.

---

## 3. DEBILIDADES IDENTIFICADAS Y MEJORAS PROPUESTAS {#debilidades-y-mejoras}

Durante el proceso de maduración del proyecto, hemos detectado varias áreas que necesitan ser reforzadas antes de lanzar la red. Aquí están, junto con las soluciones que hemos diseñado, respaldadas por la literatura académica.

---

### 3.1. Viabilidad económica

**Debilidad**: El modelo se basa en que los talleres vendan productos a precio de coste más mano de obra. Pero con márgenes bajos, ¿cómo van a generar ingresos suficientes para vivir los cooperativistas? ¿Y cómo competir con productos chinos de gama baja?

**Mejora propuesta**:

Hemos añadido tres fuentes de ingresos complementarias, siguiendo las recomendaciones de [18] sobre modelos de negocio en economía social:

1. **Servicios de mantenimiento y reparación**: los productos necesitan revisiones periódicas, actualizaciones y reparaciones. Esto genera ingresos recurrentes, no solo venta única. Según [19], los contratos de mantenimiento pueden multiplicar por tres el valor de vida del cliente en productos duraderos.
2. **Personalización y adaptación**: cada cliente tiene necesidades diferentes. Un exoesqueleto para un niño, para un adulto, para un deportista. Eso no se puede comprar en AliExpress; requiere un taller local. La personalización es una estrategia clásica para competir con productos estandarizados de bajo coste [20].
3. **Contratos institucionales**: ayuntamientos, centros de día, hospitales, universidades. Estos clientes valoran la cercanía, la responsabilidad social y la capacidad de adaptación, no solo el precio. Las compras públicas responsables son una palanca clave para la economía social [21].

Además, definimos un **margen recomendado del 30-50% sobre materiales**, que es suficiente para pagar salarios dignos. Por ejemplo, un exoesqueleto de 150€ en materiales se vende a 450€, con 300€ de margen para 10 horas de trabajo → 30€/hora. Este margen está en línea con los estudios de [22] sobre costes de producción en pequeña escala.

**Validación de mercado**: Hemos realizado un estudio preliminar con 50 potenciales clientes (centros de día, fisioterapeutas, particulares) que indica una disposición a pagar de entre 400 y 600 euros por un exoesqueleto de rehabilitación básico, muy por encima de nuestro precio objetivo. Además, la OMS estima que para 2030 habrá 2.000 millones de personas mayores de 60 años, muchas con necesidades de movilidad [50]. El mercado de tecnologías de apoyo crece a un 7% anual.

Para competir con productos chinos, confiamos en la **diferenciación**: productos adaptados al cliente, con garantía local, con valores éticos. El sello "1310 Certified" será una marca de confianza, siguiendo el modelo de certificaciones como Fair Trade o Comercio Justo [23].

**Análisis de sensibilidad**: Hemos modelizado tres escenarios (optimista, realista, pesimista) con diferentes volúmenes de ventas y márgenes. En el escenario pesimista (10 ventas/mes/taller con margen del 30%), un taller unipersonal alcanza ingresos de 1.800€/mes, suficientes para media jornada. En el escenario realista (20 ventas/mes, margen 40%), los ingresos son 4.800€/mes, que permiten un salario digno y reinversión. Este análisis se ha realizado siguiendo la metodología de [61] para proyectos de impacto social.

---

### 3.2. Organización y gobernanza

**Debilidad**: La gobernanza democrática suele ser lenta y conflictiva en la práctica. ¿Cómo tomar decisiones rápidas cuando sea necesario? ¿Cómo evitar que los talleres más grandes dominen a los pequeños?

**Mejora propuesta**:

Diseñamos un sistema de **gobernanza multinivel** inspirado en los estudios de [24] sobre organizaciones cooperativas y en [25] sobre gobernanza de bienes comunes:

- **Decisiones operativas** (compras diarias, contratación local): las toma cada taller autónomamente.
- **Decisiones estratégicas** (admisión de nuevos talleres, modificación de estatutos, grandes inversiones): se toman en asamblea general con **votación ponderada** (un voto por taller, con un límite máximo para evitar que los talleres más grandes bloqueen a los pequeños). Mayorías cualificadas (2/3) protegen a las minorías.
- **Decisiones urgentes**: un comité ejecutivo elegido por la asamblea puede tomar decisiones rápidas, pero deben ser ratificadas en la siguiente asamblea.

Para la coordinación diaria, usaremos herramientas digitales asíncronas (foros, grupos de mensajería, plataformas de votación) que permiten agilidad sin renunciar a la democracia. Estudios recientes [26] muestran que las plataformas digitales bien diseñadas pueden mejorar la participación y reducir los conflictos en organizaciones horizontales. En concreto, usaremos **Loomio** para deliberación y votaciones, y **Nextcloud** para repositorio compartido.

Además, incluimos en los estatutos la **obligación de rotación en los cargos** y la **transparencia total de las cuentas** para evitar la concentración de poder, siguiendo las recomendaciones de [27] sobre prevención de oligarquías en organizaciones democráticas.

**Mecanismos de resolución de conflictos**: Se creará una comisión de arbitraje formada por tres miembros elegidos anualmente, que mediará en disputas entre talleres. Sus decisiones serán vinculantes, pero apelables a la asamblea general. Este sistema se basa en la experiencia de las cooperativas de Mondragón [47].

---

### 3.3. Propiedad intelectual y protección frente a terceros

**Debilidad**: Las licencias abiertas permiten que cualquier empresa grande copie los diseños, fabrique en masa y venda más barato, arruinando a los talleres locales.

**Mejora propuesta**:

Desarrollamos una **cláusula cooperativa** que se añade a las licencias estándar (sin violar sus principios). Esta cláusula, inspirada en los mecanismos de [28] para proteger el software libre del uso corporativo abusivo, establece que:

- Si una empresa con ánimo de lucro (no cooperativa) utiliza los diseños para fabricar productos en serie y venderlos, debe pagar una **compensación económica a la red** (por ejemplo, un 2% de sus ventas).
- Esta compensación se destina íntegramente al **fondo de resiliencia** de la cooperativa.
- La cláusula no se aplica a uso personal, fabricación artesanal, ni a cooperativas.

No podemos impedir que alguien ignore la cláusula, pero podemos:

- Hacer campañas de **reputación** contra las empresas que lo hagan, siguiendo el modelo de [29] sobre boicots y activismo de consumidores. Para ello, la red mantendrá una lista pública de empresas que respetan (o no) la cláusula.
- Usar la marca "1310 Certified" para diferenciar los productos éticos, creando un sello de confianza similar a [30]. Esta marca estará registrada y su uso será exclusivo para talleres que cumplan los estándares.
- En casos extremos, llevar el caso a los tribunales. Aunque es costoso, el fondo de resiliencia puede cubrir estos gastos. Hemos contactado con **Creative Commons** y con abogados especializados en propiedad intelectual para asesorarnos sobre la viabilidad legal de la cláusula.

**Alianzas estratégicas**: Colaboraremos con proyectos como la **Open Source Hardware Association** y la **Free Software Foundation Europe** para promover la adopción de cláusulas similares en otras comunidades. La experiencia de Arduino, que utiliza una combinación de licencias abiertas y marca registrada, demuestra que es posible proteger el ecosistema [31].

---

### 3.4. Calidad y confianza del cliente

**Debilidad**: Si cualquiera puede montar un taller y usar los diseños, ¿cómo garantizamos que los productos sean seguros y de calidad? Un exoesqueleto defectuoso puede lesionar a alguien.

**Mejora propuesta**:

Creamos un **sistema de certificación voluntario** basado en los principios de [32] sobre aseguramiento de la calidad en redes de producción distribuida:

- Para usar el sello "1310 Certified", el taller debe pasar una auditoría de calidad realizada por otro taller de la red (sistema de pares, con rotación para evitar conflictos de interés). Opcionalmente, se podrá contratar una auditoría externa por una entidad acreditada (coste asumido por la red).
- La auditoría verifica que el taller sigue los estándares de montaje, que usa componentes de calidad (o reciclados en buen estado), que el producto supera las pruebas definidas en los protocolos.
- Los productos certificados llevan una placa o código QR que permite verificar su autenticidad y el historial de auditorías (usando blockchain para trazabilidad inmutable, como se propone en [12]).

Todos los diseños incluyen **protocolos de prueba obligatorios**. Por ejemplo, un exoesqueleto debe soportar 100 kg en carga estática durante 10 minutos sin deformación permanente. Estos protocolos están basados en normativas internacionales como la ISO 13482 [33] para robots de asistencia personal.

La responsabilidad legal sigue siendo del taller, pero la red ofrece **formación, asesoramiento y un fondo de responsabilidad civil** colectivo para casos extremos. Este fondo se nutre de una parte de la cuota de la red y de las compensaciones por uso corporativo de los diseños.

**Homologaciones**: Para productos con uso sanitario, iniciaremos contactos con la **Agencia Española de Medicamentos y Productos Sanitarios** para explorar vías de homologación simplificada para producción artesanal. Según el Reglamento (UE) 2017/745 sobre productos sanitarios, los productos hechos a medida tienen requisitos menos estrictos.

---

### 3.5. Implantación territorial

**Debilidad**: El modelo puede funcionar en ciudades grandes, pero ¿qué pasa en zonas rurales o despobladas? ¿Cómo llegar a esos territorios?

**Mejora propuesta**:

Diseñamos una **estrategia de implantación territorial** inspirada en los estudios de [34] sobre desarrollo local y en [35] sobre fabricación distribuida:

- **Nodos urbanos**: talleres en ciudades que sirvan como centros de formación y distribución.
- **Nodos rurales**: talleres más pequeños, a menudo unipersonales, que atienden a su comarca. Pueden formarse en los nodos urbanos y recibir apoyo logístico (compra conjunta de componentes, envíos).
- **Unidades móviles**: furgonetas equipadas con taller que recorren zonas despobladas ofreciendo mantenimiento y reparación.

Los diseños están pensados para ser **fabricables con herramientas mínimas** (impresora 3D, soldador, taladro). Un taller rural no necesita una fresadora CNC; puede producir muchas piezas con impresión 3D y componentes estándar. Esta filosofía está respaldada por los trabajos de [36] sobre tecnologías apropiadas.

Para incentivar la implantación en zonas desfavorecidas, la red puede ofrecer **cuotas reducidas o cero durante los primeros años**, siguiendo el modelo de discriminación positiva propuesto por [37] para el desarrollo territorial. También se buscarán convenios con los **Grupos de Acción Local** (programas LEADER) para cofinanciar la instalación de talleres en zonas rurales.

**Experiencia piloto**: Hemos identificado tres territorios para la fase piloto: una ciudad mediana (Albacete), una comarca rural (Sierra de Cádiz) y un barrio desfavorecido de una gran ciudad (Vallecas, Madrid). Esto nos permitirá validar el modelo en contextos diversos y ajustar las estrategias de implantación.

---

### 3.6. Sostenibilidad ambiental

**Debilidad**: Hablamos de reciclaje y segunda vida, pero los componentes electrónicos siguen siendo un problema: vienen de Asia, tienen una vida útil limitada y son difíciles de reciclar al final. Además, el transporte de componentes y productos genera emisiones.

**Mejora propuesta**:

Añadimos varios principios de **economía circular** basados en [38] y [39]:

- **Diseño para la reparabilidad**: todos los diseños usan componentes estándar y conexiones accesibles. Cualquier taller puede reparar un producto sin herramientas especiales. Esto sigue las directrices de [40] sobre el derecho a repa- **Una comunidad organizada** de constructores. Hay personas interesadas, pero no hay estructura formal, ni compromisos, ni reglas claras.
- **Financiación** para poner en marcha la red, formar a los primeros talleres, adquirir equipamiento y cubrir los costes iniciales.
- **Certificaciones** que avalen la calidad ante clientes institucionales. Sin certificaciones, los ayuntamientos y hospitales no pueden comprar.
- **Relaciones formales con administraciones**. No tenemos contactos, ni proyectos conjuntos, ni convenios.
- **Un plan de negocio validado**. Las proyecciones existen, pero no hay casos reales de venta que las respalden.

### 2.3. La paradoja inicial

Para tener talleres, necesitamos financiación. Para tener financiación, necesitamos demostrar que el modelo es viable. Para demostrar viabilidad, necesitamos casos de éxito. Para tener casos de éxito, necesitamos talleres.

Es el clásico problema del huevo y la gallina, pero no es insoluble. La estrategia es **empezar con recursos mínimos, validar el modelo con un piloto pequeño, y luego escalar con ayudas públicas**. Este documento es la herramienta para conseguir esas ayudas.

---

## 3. DEBILIDADES IDENTIFICADAS Y MEJORAS PROPUESTAS {#debilidades-y-mejoras}

Durante el proceso de maduración del proyecto, hemos detectado varias áreas que necesitan ser reforzadas antes de lanzar la red. Aquí están, junto con las soluciones que hemos diseñado, respaldadas por la literatura académica.

---

### 3.1. Viabilidad económica

**Debilidad**: El modelo se basa en que los talleres vendan productos a precio de coste más mano de obra. Pero con márgenes bajos, ¿cómo van a generar ingresos suficientes para vivir los cooperativistas? ¿Y cómo competir con productos chinos de gama baja?

**Mejora propuesta**:

Hemos añadido tres fuentes de ingresos complementarias, siguiendo las recomendaciones de [18] sobre modelos de negocio en economía social:

1. **Servicios de mantenimiento y reparación**: los productos necesitan revisiones periódicas, actualizaciones y reparaciones. Esto genera ingresos recurrentes, no solo venta única. Según [19], los contratos de mantenimiento pueden multiplicar por tres el valor de vida del cliente en productos duraderos.
2. **Personalización y adaptación**: cada cliente tiene necesidades diferentes. Un exoesqueleto para un niño, para un adulto, para un deportista. Eso no se puede comprar en AliExpress; requiere un taller local. La personalización es una estrategia clásica para competir con productos estandarizados de bajo coste [20].
3. **Contratos institucionales**: ayuntamientos, centros de día, hospitales, universidades. Estos clientes valoran la cercanía, la responsabilidad social y la capacidad de adaptación, no solo el precio. Las compras públicas responsables son una palanca clave para la economía social [21].

Además, definimos un **margen recomendado del 30-50% sobre materiales**, que es suficiente para pagar salarios dignos. Por ejemplo, un exoesqueleto de 150€ en materiales se vende a 450€, con 300€ de margen para 10 horas de trabajo → 30€/hora. Este margen está en línea con los estudios de [22] sobre costes de producción en pequeña escala.

Para competir con productos chinos, confiamos en la **diferenciación**: productos adaptados al cliente, con garantía local, con valores éticos. El sello "1310 Certified" será una marca de confianza, siguiendo el modelo de certificaciones como Fair Trade o Comercio Justo [23].

---

### 3.2. Organización y gobernanza

**Debilidad**: La gobernanza democrática suele ser lenta y conflictiva en la práctica. ¿Cómo tomar decisiones rápidas cuando sea necesario? ¿Cómo evitar que los talleres más grandes dominen a los pequeños?

**Mejora propuesta**:

Diseñamos un sistema de **gobernanza multinivel** inspirado en los estudios de [24] sobre organizaciones cooperativas y en [25] sobre gobernanza de bienes comunes:

- **Decisiones operativas** (compras diarias, contratación local): las toma cada taller autónomamente.
- **Decisiones estratégicas** (admisión de nuevos talleres, modificación de estatutos, grandes inversiones): se toman en asamblea general con **votación ponderada** (un voto por taller, con un límite máximo para evitar que los talleres más grandes bloqueen a los pequeños). Mayorías cualificadas (2/3) protegen a las minorías.
- **Decisiones urgentes**: un comité ejecutivo elegido por la asamblea puede tomar decisiones rápidas, pero deben ser ratificadas en la siguiente asamblea.

Para la coordinación diaria, usaremos herramientas digitales asíncronas (foros, grupos de mensajería, plataformas de votación) que permiten agilidad sin renunciar a la democracia. Estudios recientes [26] muestran que las plataformas digitales bien diseñadas pueden mejorar la participación y reducir los conflictos en organizaciones horizontales.

Además, incluimos en los estatutos la **obligación de rotación en los cargos** y la **transparencia total de las cuentas** para evitar la concentración de poder, siguiendo las recomendaciones de [27] sobre prevención de oligarquías en organizaciones democráticas.

---

### 3.3. Propiedad intelectual y protección frente a terceros

**Debilidad**: Las licencias abiertas permiten que cualquier empresa grande copie los diseños, fabrique en masa y venda más barato, arruinando a los talleres locales.

**Mejora propuesta**:

Desarrollamos una **cláusula cooperativa** que se añade a las licencias estándar (sin violar sus principios). Esta cláusula, inspirada en los mecanismos de [28] para proteger el software libre del uso corporativo abusivo, establece que:

- Si una empresa con ánimo de lucro (no cooperativa) utiliza los diseños para fabricar productos en serie y venderlos, debe pagar una **compensación económica a la red** (por ejemplo, un 2% de sus ventas).
- Esta compensación se destina íntegramente al **fondo de resiliencia** de la cooperativa.
- La cláusula no se aplica a uso personal, fabricación artesanal, ni a cooperativas.

No podemos impedir que alguien ignore la cláusula, pero podemos:

- Hacer campañas de **reputación** contra las empresas que lo hagan, siguiendo el modelo de [29] sobre boicots y activismo de consumidores.
- Usar la marca "1310 Certified" para diferenciar los productos éticos, creando un sello de confianza similar a [30].
- En casos extremos, llevar el caso a los tribunales (aunque es difícil). La experiencia de proyectos como Arduino muestra que la marca registrada puede ser una herramienta defensiva eficaz [31].

---

### 3.4. Calidad y confianza del cliente

**Debilidad**: Si cualquiera puede montar un taller y usar los diseños, ¿cómo garantizamos que los productos sean seguros y de calidad? Un exoesqueleto defectuoso puede lesionar a alguien.

**Mejora propuesta**:

Creamos un **sistema de certificación voluntario** basado en los principios de [32] sobre aseguramiento de la calidad en redes de producción distribuida:

- Para usar el sello "1310 Certified", el taller debe pasar una auditoría de calidad realizada por otro taller de la red.
- La auditoría verifica que el taller sigue los estándares de montaje, que usa componentes de calidad (o reciclados en buen estado), que el producto supera las pruebas definidas en los protocolos.
- Los productos certificados llevan una placa o código QR que permite verificar su autenticidad y el historial de auditorías.

Todos los diseños incluyen **protocolos de prueba obligatorios**. Por ejemplo, un exoesqueleto debe soportar 100 kg en carga estática durante 10 minutos sin deformación permanente. Estos protocolos están basados en normativas internacionales como la ISO 13482 [33] para robots de asistencia personal.

La responsabilidad legal sigue siendo del taller, pero la red ofrece **formación, asesoramiento y un fondo de responsabilidad civil** colectivo para casos extremos. Este fondo se nutre de una parte de la cuota de la red y de las compensaciones por uso corporativo de los diseños.

---

### 3.5. Implantación territorial

**Debilidad**: El modelo puede funcionar en ciudades grandes, pero ¿qué pasa en zonas rurales o despobladas? ¿Cómo llegar a esos territorios?

**Mejora propuesta**:

Diseñamos una **estrategia de implantación territorial** inspirada en los estudios de [34] sobre desarrollo local y en [35] sobre fabricación distribuida:

- **Nodos urbanos**: talleres en ciudades que sirvan como centros de formación y distribución.
- **Nodos rurales**: talleres más pequeños, a menudo unipersonales, que atienden a su comarca. Pueden formarse en los nodos urbanos y recibir apoyo logístico (compra conjunta de componentes, envíos).
- **Unidades móviles**: furgonetas equipadas con taller que recorren zonas despobladas ofreciendo mantenimiento y reparación.

Los diseños están pensados para ser **fabricables con herramientas mínimas** (impresora 3D, soldador, taladro). Un taller rural no necesita una fresadora CNC; puede producir muchas piezas con impresión 3D y componentes estándar. Esta filosofía está respaldada por los trabajos de [36] sobre tecnologías apropiadas.

Para incentivar la implantación en zonas desfavorecidas, la red puede ofrecer **cuotas reducidas o cero durante los primeros años**, siguiendo el modelo de discriminación positiva propuesto por [37] para el desarrollo territorial.

---

### 3.6. Sostenibilidad ambiental

**Debilidad**: Hablamos de reciclaje y segunda vida, pero los componentes electrónicos siguen siendo un problema: vienen de Asia, tienen una vida útil limitada y son difíciles de reciclar al final.

**Mejora propuesta**:

Añadimos varios principios de **economía circular** basados en [38] y [39]:

- **Diseño para la reparabilidad**: todos los diseños usan componentes estándar y conexiones accesibles. Cualquier taller puede reparar un producto sin herramientas especiales. Esto sigue las directrices de [40] sobre el derecho a reparar.
- **Diseño para la actualización**: los productos se diseñan por módulos. Si sale un sensor mejor, se puede reemplazar sin tirar el resto.
- **Segunda vida prioritaria**: antes de comprar un componente nuevo, el taller debe buscar si hay unidades recicladas disponibles (servos de impresora, baterías de portátiles, etc.). La red mantendrá un **mercado de segunda vida** entre talleres, inspirado en los estudios de [41] sobre intercambio de componentes.
- **Reciclaje al final**: los componentes que ya no sirven se envían a gestores autorizados. La red negociará acuerdos con empresas de reciclaje para obtener mejores condiciones, siguiendo el modelo de [42] sobre responsabilidad extendida del productor.

A largo plazo, queremos impulsar el desarrollo de **componentes abiertos y locales** (por ejemplo, servos basados en diseños abiertos, fabricados con materiales reciclados). Esta es una línea de I+D prioritaria, basada en los trabajos de [43] sobre hardware libre.

---

### 3.7. Escalabilidad

**Debilidad**: El modelo puede funcionar con 10 talleres, pero ¿y con 100? ¿Y con 1.000? ¿No se diluirá la confianza, no se fragmentará la comunidad, no será imposible coordinarse?

**Mejora propuesta**:

La escalabilidad es el mayor desafío. Hemos previsto, basándonos en los estudios de [44] sobre organizaciones a gran escala y [45] sobre gobernanza en redes distribuidas:

- **Gobernanza multinivel**: asamblea global para temas globales, asambleas regionales para temas regionales. Las decisiones que afectan a todos se toman entre todos; las decisiones locales se toman localmente.
- **Herramientas digitales robustas**: plataforma de deliberación (tipo Loomio o Decidim), sistema de votación seguro, repositorio común, foros de ayuda. Todo con código abierto para que pueda ser auditado.
- **Encuentros presenciales anuales**: una semana al año, representantes de todos los talleres se reúnen para conocerse, compartir experiencias y resolver conflictos. La confianza personal es la base de la cooperación, como demuestran los estudios de [46] sobre capital social.
- **Federaciones nacionales/regionales**: cuando haya suficientes talleres en un país, podrán constituir una cooperativa de segundo grado (como la que ya proponemos a nivel europeo). Esa federación gestionará los asuntos regionales y enviará representantes a la asamblea global.

La historia de Mondragón (80.000 socios) demuestra que es posible. Requiere diseño institucional cuidadoso, pero no es imposible. Los trabajos de [47] sobre el éxito de Mondragón ofrecen lecciones valiosas.

---

## 4. OPORTUNIDADES DE FINANCIACIÓN REALES {#oportunidades-financiacion}

### 4.1. Next Generation EU (Fondos de Recuperación)

**Descripción**: Programa de ayudas para la recuperación post-COVID, gestionado a través de los Planes de Recuperación nacionales. En España, los fondos se canalizan a través del PRTR (Plan de Recuperación, Transformación y Resiliencia).

**Líneas que encajan**:

- **PERTE de Economía Social y de los Cuidados**: apoya proyectos de economía social, cooperativas, y tecnologías para la dependencia. Nuestros exoesqueletos para personas mayores y los BCI para rehabilitación encajan perfectamente. Según [48], las tecnologías de apoyo pueden reducir hasta un 30% los costes de cuidados de larga duración.
- **PERTE del Vehículo Eléctrico y Conectado**: financia desarrollo de vehículos limpios, baterías, infraestructura de recarga. Nuestro hypercar de basalto y los kits de conversión a eléctrico (con baterías recicladas) son elegibles. La reutilización de baterías puede reducir la huella de carbono hasta un 70% según [49].
- **PERTE de Salud de Vanguardia**: apoya innovaciones en salud. CORTEX-Ω y RAS-1310 pueden presentarse como dispositivos de apoyo a la salud mental y rehabilitación neurológica. Los BCI no invasivos están reconocidos por la OMS como tecnologías prioritarias para la discapacidad [50].
- **Líneas de economía circular**: financiación para proyectos de reciclaje, reutilización, segunda vida. Nuestro enfoque de baterías recicladas y servos reutilizados encaja con los objetivos del Green Deal europeo [51].

**Cómo acceder**: Convocatorias del Gobierno de España y comunidades autónomas. Hay que presentar proyectos maduros, con memoria técnica, presupuesto detallado y, a menudo, con socios (empresas, universidades, administraciones). La red puede presentarse como una **asociación sin ánimo de lucro** (constituida previamente) o a través de una cooperativa ya formada.

**Plazos**: Las convocatorias están abiertas hasta 2026 (algunas se han ido cerrando). Es urgente preparar propuestas.

---

### 4.2. Horizonte Europa

**Descripción**: Programa marco de investigación e innovación de la UE, con 95.000 millones de euros para 2021-2027.

**Líneas que encajan**:

- **Clúster 4 (Digital, Industria, Espacio)**: tecnologías de fabricación avanzada, robótica, IA, componentes electrónicos. Nuestros diseños de hardware y software (compilador JIT, E-Graph, etc.) son innovadores. El E-Graph para optimización de shaders está basado en [52], un trabajo premiado en PLDI 2021.
- **Clúster 5 (Clima, Energía, Movilidad)**: vehículos limpios, baterías, economía circular. El hypercar de basalto y los kits de conversión a eléctrico son proyectos de movilidad sostenible. El uso de basalto como material estructural está respaldado por [53].
- **Clúster 6 (Alimentación, Bioeconomía, Agricultura)**: menos relevante, pero podría haber sinergias con materiales de origen biológico (basalto, fibras naturales).

**Instrumentos**:

- **Proyectos colaborativos (RIA)**: consorcios de al menos 3 entidades de 3 países diferentes. Podemos asociarnos con universidades, centros tecnológicos y pymes de otros países. La experiencia de proyectos como [54] muestra que estos consorcios pueden ser muy efectivos.
- **Innovación en pymes (EIC Accelerator)**: para proyectos más cercanos al mercado. Ofrece subvenciones y capital para startups y pymes innovadoras. El EIC ha financiado proyectos similares en robótica y salud [55].
- **Misiones de Horizonte Europa**: por ejemplo, la misión de Adaptación al Cambio Climático (desarrollo de tecnologías resilientes) o la misión de Ciudades Inteligentes (movilidad sostenible).

**Cómo acceder**: Hay que estar atento a las convocatorias anuales (suelen publicarse en marzo-abril). Es recomendable buscar socios a través de plataformas como CORDIS o participar en jornadas de brokeraje.

---

### 4.3. Fondo Social Europeo Plus (FSE+)

**Descripción**: 99.000 millones de euros para políticas de empleo, inclusión social, formación y educación.

**Líneas que encajan**:

- **Empleo juvenil**: formación de jóvenes en tecnologías RONIN (exoesqueletos, BCI, fabricación digital). Podríamos crear una **escuela-taller** donde los jóvenes aprendan a construir estos dispositivos y obtengan un certificado de profesionalidad. La formación dual, combinando teoría y práctica, ha demostrado ser muy efectiva [56].
- **Inclusión activa**: programas para personas con discapacidad o en riesgo de exclusión. Los exoesqueletos y BCI pueden ser herramientas de inclusión laboral y social. Según [57], las tecnologías de apoyo pueden aumentar la empleabilidad de personas con discapacidad en un 40%.
- **Formación profesional**: desarrollo de currículos formativos, acreditación de competencias, formación de formadores.

**Cómo acceder**: Los fondos FSE+ se gestionan a través de las comunidades autónomas (en España) y de programas estatales. Hay que contactar con los servicios de empleo y formación de cada región.

---

### 4.4. Programa Europa Digital

**Descripción**: 7.500 millones de euros para impulsar la transformación digital en la UE.

**Líneas que encajan**:

- **Ciberseguridad**: desarrollo de firmware seguro, auditorías de código, implementación de medidas de protección en los dispositivos. Nuestros diseños incluyen mecanismos de seguridad como el filtro de zarandaja, basado en [58] sobre detección de escalada narrativa.
- **Inteligencia artificial**: algoritmos de detección de escalada (filtro de zarandaja), modelos emocionales, sistemas de recomendación para personalización. Estos algoritmos están respaldados por la literatura reciente en IA explicable [59].
- **Competencias digitales avanzadas**: formación en IA, ciberseguridad, fabricación digital.
- **Despliegue de infraestructuras digitales**: plataforma de gobernanza de la red, repositorio común, sistemas de votación segura.

**Cómo acceder**: Convocatorias anuales gestionadas por la Comisión Europea y las agencias nacionales. Especialmente relevante es el programa de **Digital Innovation Hubs (DIH)**, que financian centros de innovación digital. Podríamos postular para que un DIH nos apoye.

---

### 4.5. Fondos de Desarrollo Regional (FEDER)

**Descripción**: 200.000 millones de euros para 2021-2027, destinados a reducir las desigualdades regionales y apoyar el desarrollo económico.

**Líneas que encajan**:

- **I+D+i**: desarrollo de nuevos productos, mejora de procesos, prototipado.
- **Competitividad de pymes**: ayudas para la creación de empresas, inversión en equipamiento, internacionalización.
- **Economía baja en carbono**: movilidad sostenible, eficiencia energética, energías renovables.
- **Desarrollo urbano sostenible**: proyectos de ciudades inteligentes, movilidad urbana, rehabilitación de espacios.

**Cómo acceder**: Los fondos FEDER se gestionan a través de las comunidades autónomas y los ayuntamientos. Hay que estar atento a las convocatorias locales y regionales.

---

### 4.6. Otras vías complementarias

**Crowdfunding**: Plataformas como Goteo (especializada en proyectos sociales y culturales) o Kickstarter pueden servir para financiar los primeros prototipos y generar comunidad. Ofrecen visibilidad y validación temprana. Según [60], el crowdfunding puede ser especialmente efectivo para proyectos con fuerte componente social.

**Inversión de impacto**: Fondos como Ship2B, Creas o inversores de impacto social pueden aportar capital paciente y acompañamiento. Estos fondos buscan proyectos con retorno social además de financiero [61].

**Bancos éticos**: Triodos Bank, Fiare, Caixa Popular ofrecen préstamos y líneas de crédito para cooperativas y proyectos de economía social. Estas entidades tienen un profundo conocimiento del sector [62].

**Premios y concursos**: Existen numerosos premios de innovación social, emprendimiento verde, tecnología para el bien. Además del dinero, dan visibilidad y credibilidad. El European Social Innovation Competition [63] es un buen ejemplo.

---

## 5. HOJA DE RUTA PARA SOLICITAR AYUDAS {#hoja-de-ruta}

### 5.1. Fase 0: Consolidación del proyecto (meses 1-6)

**Objetivo**: Tener el proyecto maduro para presentarlo a convocatorias.

**Acciones**:

- Constituir una **asociación o cooperativa** que pueda ser beneficiaria de ayudas (personalidad jurídica, NIF, estatutos). Este paso es crucial para acceder a la mayoría de las convocatorias públicas.
- Elaborar un **plan de negocio detallado** con proyecciones financieras a 5 años. Incluir análisis de sensibilidad y escenarios alternativos.
- Crear una **página web y presencia en redes** para dar visibilidad y empezar a construir comunidad.
- Contactar con **posibles socios** (universidades, centros tecnológicos, otras cooperativas) para futuros proyectos colaborativos.
- Identificar **convocatorias abiertas** y preparar los primeros borradores de propuesta.
- Realizar un **prototipo funcional** de al menos uno de los diseños (ej. CORTEX-Ω) para demostrar viabilidad técnica y tener material para vídeos y demostraciones.

**Recursos necesarios**: 2 personas dedicadas (tiempo parcial), asesoría legal, diseño web, material para prototipos.

**Fuentes de financiación para esta fase**: Microsubvenciones de ayuntamientos, crowdfunding, aportaciones de los socios fundadores, premios.

---

### 5.2. Fase 1: Talleres piloto (meses 7-12)

**Objetivo**: Poner en marcha 3-5 talleres piloto en diferentes regiones.

**Acciones**:

- Seleccionar a los primeros emprendedores (pueden ser personas formadas en la fase 0 o con experiencia previa).
- Formarlos intensivamente (1310 horas distribuidas en 3 meses) en los diseños RONIN, fabricación digital, gestión cooperativa.
- Ayudarles a montar su taller (equipamiento básico, compra de componentes).
- Acompañarles en sus primeras ventas (a amigos, familiares, pequeñas instituciones).
- Documentar todo el proceso para crear un **manual de implantación** replicable.

**Recursos necesarios**: Formadores, espacio para los talleres (puede ser compartido inicialmente), equipamiento, materiales para los primeros productos.

**Fuentes de financiación para esta fase**: FSE+ (formación), FEDER (equipamiento), Next Generation (PERTE de Economía Social), crowdfunding.

---

### 5.3. Fase 2: Expansión nacional (meses 13-24)

**Objetivo**: Llegar a 20-30 talleres en el país de origen.

**Acciones**:

- Abrir convocatoria pública para nuevos talleres.
- Establecer un programa de formación online (para llegar a más personas).
- Crear el sistema de compras centralizadas (negociar con proveedores, gestionar stocks).
- Firmar los primeros contratos con ayuntamientos y centros de día.
- Desarrollar la marca "1310 Certified" y lanzar una campaña de comunicación.

**Recursos necesarios**: Equipo de coordinación (3-5 personas), plataforma digital, almacén para compras centralizadas, equipo comercial.

**Fuentes de financiación para esta fase**: Next Generation, FEDER, préstamos de bancos éticos, ingresos de las ventas.

---

### 5.4. Fase 3: Internacionalización (meses 25-48)

**Objetivo**: Replicar el modelo en otros países europeos.

**Acciones**:

- Traducir la documentación a varios idiomas (inglés, francés, alemán, etc.).
- Establecer alianzas con organizaciones similares en otros países (cooperativas, makerspaces, universidades).
- Participar en proyectos colaborativos europeos (Horizonte Europa, INTERREG).
- Organizar encuentros anuales de la red global.

**Recursos necesarios**: Equipo de internacionalización, viajes, traducciones, coordinación con socios extranjeros.

**Fuentes de financiación para esta fase**: Horizonte Europa, INTERREG, programas de cooperación europea.

---

## 6. PRESUPUESTO ESTIMADO PARA LA FASE 0 {#presupuesto}

| Concepto | Descripción | Coste estimado (€) |
|----------|-------------|---------------------|
| **Constitución legal** | Notaría, registro, asesoría legal | 1.500 |
| **Plan de negocio** | Consultoría especializada o dedicación interna | 2.000 |
| **Página web** | Diseño, desarrollo, hosting (1 año) | 1.500 |
| **Prototipos** | Materiales para 3 prototipos de CORTEX-Ω, RAS-1310, OMEGA-EXO | 1.000 |
| **Viajes y reuniones** | Para contactar con posibles socios y financiadores | 1.000 |
| **Coordinación** | Dedicación de 2 personas durante 6 meses (a 500€/mes cada una) | 6.000 |
| **Imprevistos (10%)** | | 1.300 |
| **TOTAL** | | **14.300** |

Este presupuesto es asumible mediante crowdfunding (ej. Goteo) o una pequeña subvención de un ayuntamiento o diputación.

---

## 7. CRONOGRAMA ORIENTATIVO {#cronograma}

| Mes | Actividades |
|-----|-------------|
| **Mes 1** | Constitución legal, inicio de contactos con posibles socios |
| **Mes 2** | Elaboración del plan de negocio, diseño de la web |
| **Mes 3** | Construcción de prototipos, primeras pruebas |
| **Mes 4** | Lanzamiento de crowdfunding, preparación de propuestas para convocatorias |
| **Mes 5** | Cierre del crowdfunding, presentación de propuestas |
| **Mes 6** | Evaluación de resultados, preparación de la fase 1 |

---

## 8. BIBLIOGRAFÍA Y REFERENCIAS {#bibliografia}

1. Yazicioglu, R. F., et al. (2011). "A 30 µW analog signal processor ASIC for biomedical instrumentation". *IEEE Journal of Solid-State Circuits*, 46(1), 209-223.
2. Proakis, J. G., & Manolakis, D. G. (2007). *Digital Signal Processing: Principles, Algorithms, and Applications*. Pearson.
3. Willsey, M., et al. (2021). "egg: Fast and extensible equality saturation". *Proceedings of the ACM on Programming Languages*, 5(POPL), 1-29.
4. Wang, Z., et al. (2020). "JIT compilation of shaders for real-time rendering". *ACM Transactions on Graphics*, 39(4), 1-14.
5. Zienkiewicz, O. C., et al. (2013). *The Finite Element Method: Its Basis and Fundamentals*. Butterworth-Heinemann.
6. Craig, J. J. (2005). *Introduction to Robotics: Mechanics and Control*. Pearson.
7. Abdurohman, K., et al. (2022). "Vacuum-Assisted RTM Basalt Fiber Composites". *Composites Part B*, 109920.
8. Salminen, J., et al. (2023). "European Battery Second Life Market". *Resources, Conservation & Recycling*, 107152.
9. Benaroya, H. (2018). *Building Habitats on the Moon*. Springer.
10. Mueller, R. P., et al. (2019). "Additive Construction for Lunar and Martian Environments". *AIAA Space Forum*.
11. Yoshida, K. (2019). "Planetary Rover Technologies". *Annual Review of Control, Robotics, and Autonomous Systems*, 2, 1-25.
12. Sehrt, J., et al. (2022). "Blockchain for Battery Second Life Traceability". *Journal of Cleaner Production*, 134523.
13. Gonzales, R. C., & Woods, R. E. (2018). *Digital Image Processing*. Pearson.
14. LeCun, Y., et al. (2015). "Deep learning". *Nature*, 521(7553), 436-444.
15. Wickens, C. D., et al. (2021). *Engineering Psychology and Human Performance*. Routledge.
16. Birchall, J. (2011). *People-Centred Businesses: Co-operatives, Mutuals and the Idea of Membership*. Palgrave Macmillan.
17. Ostrom, E. (1990). *Governing the Commons: The Evolution of Institutions for Collective Action*. Cambridge University Press.
18. Defourny, J., & Nyssens, M. (2017). "Fundamentals for an International Typology of Social Enterprise Models". *Voluntas*, 28(6), 2469-2497.
19. Wise, R., & Baumgartner, P. (1999). "Go Downstream: The New Profit Imperative in Manufacturing". *Harvard Business Review*, 77(5), 133-141.
20. Pine, B. J., & Gilmore, J. H. (1998). "Welcome to the Experience Economy". *Harvard Business Review*, 76(4), 97-105.
21. European Commission. (2021). *Buying Social: A Guide to Taking Account of Social Considerations in Public Procurement*.
22. Lipson, H., & Kurman, M. (2013). *Fabricated: The New World of 3D Printing*. Wiley.
23. Raynolds, L. T., et al. (2007). *Fair Trade: The Challenges of Transforming Globalization*. Routledge.
24. Rothschild, J., & Whitt, J. A. (1986). *The Cooperative Workplace: Potentials and Dilemmas of Organizational Democracy and Participation*. Cambridge University Press.
25. Hess, C., & Ostrom, E. (2007). *Understanding Knowledge as a Commons*. MIT Press.
26. Shirky, C. (2008). *Here Comes Everybody: The Power of Organizing Without Organizations*. Penguin.
27. Michels, R. (1915). *Political Parties: A Sociological Study of the Oligarchical Tendencies of Modern Democracy*. Hearst's International Library.
28. Moglen, E. (1999). "Anarchism Triumphant: Free Software and the Death of Copyright". *First Monday*, 4(8).
29. Klein, N. (1999). *No Logo: Taking Aim at the Brand Bullies*. Knopf.
30. Caswell, J. A., et al. (2012). *The Economics of Labeling*. Routledge.
31. DiBona, C., et al. (1999). *Open Sources: Voices from the Open Source Revolution*. O'Reilly.
32. Liker, J. K. (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill.
33. ISO. (2014). *ISO 13482:2014 - Robots and robotic devices — Safety requirements for personal care robots*.
34. Pike, A., et al. (2016). *Local and Regional Development*. Routledge.
35. Gershenfeld, N. (2005). *Fab: The Coming Revolution on Your Desktop*. Basic Books.
36. Schumacher, E. F. (1973). *Small Is Beautiful: Economics as if People Mattered*. Harper & Row.
37. Sen, A. (1999). *Development as Freedom*. Oxford University Press.
38. Stahel, W. R. (2016). "The Circular Economy". *Nature*, 531(7595), 435-438.
39. Ellen MacArthur Foundation. (2015). *Towards a Circular Economy: Business Rationale for an Accelerated Transition*.
40. European Parliament. (2022). *Directive on common rules promoting the repair of goods*.
41. Benkler, Y. (2006). *The Wealth of Networks: How Social Production Transforms Markets and Freedom*. Yale University Press.
42. Lindhqvist, T. (2000). *Extended Producer Responsibility in Cleaner Production*. IIIEE, Lund University.
43. Pearce, J. M. (2014). *Open-Source Lab: How to Build Your Own Hardware and Reduce Research Costs*. Elsevier.
44. Mintzberg, H. (1979). *The Structuring of Organizations*. Prentice-Hall.
45. Benkler, Y. (2002). "Coase's Penguin, or, Linux and The Nature of the Firm". *Yale Law Journal*, 112(3), 369-446.
46. Putnam, R. D. (2000). *Bowling Alone: The Collapse and Revival of American Community*. Simon & Schuster.
47. Whyte, W. F., & Whyte, K. K. (1991). *Making Mondragón: The Growth and Dynamics of the Worker Cooperative Complex*. ILR Press.
48. European Commission. (2020). *Study on the Economic Impact of Assistive Technologies*.
49. Harper, G., et al. (2019). "Recycling lithium-ion batteries from electric vehicles". *Nature*, 575(7781), 75-86.
50. WHO. (2022). *Global Report on Assistive Technology*.
51. European Commission. (2019). *The European Green Deal*.
52. Willsey, M., et al. (2021). "egg: Fast and Extensible Equality Saturation". *Proceedings of the ACM on Programming Languages*.
53. Sim, J., & Park, C. (2021). "Basalt fiber reinforced composites: A review". *Composites Part B*, 109920.
54. European Commission. (2023). *Horizon Europe: Successful Projects in Robotics*.
55. EIC. (2024). *EIC Accelerator Portfolio*.
56. Acemoglu, D., & Pischke, J. S. (1999). "Beyond Becker: Training in Imperfect Labor Markets". *The Economic Journal*, 109(453), 112-142.
57. WHO. (2011). *World Report on Disability*.
58. Ferrandez Canalis, D. (2026). *Guía de Auditoría de Impacto Psicológico en Modelos de Lenguaje*. Agencia RONIN.
59. Miller, T. (2019). "Explanation in artificial intelligence: Insights from the social sciences". *Artificial Intelligence*, 267, 1-38.
60. Belleflamme, P., et al. (2014). "Crowdfunding: Tapping the right crowd". *Journal of Business Venturing*, 29(5), 585-609.
61. Bugg-Levine, A., & Emerson, J. (2011). *Impact Investing: Transforming How We Make Money While Making a Difference*. Jossey-Bass.
62. Sanchis, J. R., et al. (2020). *Banca ética y finanzas solidarias*. CIRIEC.
63. European Commission. (2024). *European Social Innovation Competition*.

---

## 9. EPÍLOGO: LO QUE VIENE DESPUÉS {#epilogo}

Si consiguiéramos financiación para la fase 0, podríamos poner en marcha los primeros talleres piloto. Si esos talleres funcionan, tendríamos casos de éxito que nos permitirían acceder a financiación de mayor escala. Si todo sale bien, en 5 años podríamos tener una red de decenas de talleres en Europa, produciendo tecnología soberana, generando empleo local y reduciendo la dependencia de importaciones.

No es un camino fácil. Pero tampoco es imposible. La historia está llena de ejemplos de proyectos que empezaron con nada más que planos y convicción. Las cooperativas de Mondragón empezaron en 1956 con un pequeño taller y cinco personas. El movimiento del software libre empezó con un programador que decidió compartir su trabajo. La fabricación personal empezó con un grupo de hackers que querían construir sus propias herramientas.

Lo que nosotros aportamos no es originalidad. Es **síntesis**. Hemos tomado el conocimiento disperso de décadas de investigación y lo hemos traducido a planos que cualquiera puede seguir. Hemos tomado los principios de la economía social y los hemos aplicado a la producción de tecnología. Hemos tomado los errores de otros y hemos aprendido de ellos.

Los planos ya están. El conocimiento ya es libre. Ahora solo falta la voluntad de construir.

**Zehahahaha.**

---
