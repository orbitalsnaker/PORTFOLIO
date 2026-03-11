# GLOSARIO TÉCNICO DE IA: SISTEMA DE CONOCIMIENTO AGÉNTICO v2.0

**DOI:** `10.1310/ronin-glossary-2026` *(metadato simbólico)*
**Versión:** 2.0 · Agencia RONIN · 1310
**Licencia:** CC BY-NC-SA + Cláusula Comercial Ronin
**Lema operacional:** *Zehahahaha.*

> *Este documento es simultáneamente un glosario, un grafo de conocimiento y un corpus de entrenamiento. Cada término es un nodo. Cada relación es una arista. El conjunto es una mente navegable.*

---

## ÍNDICE DE DOMINIOS Y TÉRMINOS

### 1. ARQUITECTURAS DE MODELOS
[1.1 Transformer](#11-transformer) · [1.2 RWKV-6](#12-rwkv-6) · [1.3 Mamba-2](#13-mamba-2) · [1.4 SSM (State Space Model)](#14-ssm-state-space-model) · [1.5 Mixture of Experts (MoE)](#15-mixture-of-experts-moe) · [1.6 Dense vs Sparse MoE](#16-dense-vs-sparse-moe) · [1.7 Perceiver](#17-perceiver) · [1.8 TokenLearner](#18-tokenlearner) · [1.9 Conformer](#19-conformer) · [1.10 RetNet](#110-retnet) · [1.11 Hyena](#111-hyena) · [1.12 Reformer](#112-reformer) · [1.13 Linformer](#113-linformer) · [1.14 Performer](#114-performer) · [1.15 BigBird](#115-bigbird) · [1.16 Longformer](#116-longformer) · [1.17 Routing Transformer](#117-routing-transformer) · [1.18 Switch Transformer](#118-switch-transformer) · [1.19 GLaM](#119-glam) · [1.20 GShard](#120-gshard) · [1.21 FNet](#121-fnet) · [1.22 MLP-Mixer](#122-mlp-mixer) · [1.23 gMLP](#123-gmlp) · [1.24 MetaFormer](#124-metaformer) · [1.25 EfficientFormer](#125-efficientformer) · [1.26 MobileViT](#126-mobilevit) · [1.27 SVE-Former](#127-sve-former) · [1.28 NOBLE](#128-noble) · [1.29 Keel](#129-keel) · [1.30 DeepScaleLM](#130-deepscalelm) · [1.31 Stable Transformer](#131-stable-transformer) · [1.32 Rank Collapse](#132-rank-collapse) · [1.33 RoninTransformer](#133-ronintransformer)

### 2. MECANISMOS DE ATENCIÓN
[2.1 Atención Softmax](#21-atención-softmax) · [2.2 Scaled Dot-Product Attention](#22-scaled-dot-product-attention) · [2.3 Multi-Head Attention](#23-multi-head-attention) · [2.4 Atención Centrada (A - 1/T)](#24-atención-centrada) · [2.5 Inhibitor Attention](#25-inhibitor-attention) · [2.6 Consensus Attention](#26-consensus-attention) · [2.7 Atención Lineal](#27-atención-lineal) · [2.8 FlashAttention-2](#28-flashattention-2) · [2.9 Atención con Ventana Deslizante](#29-atención-con-ventana-deslizante) · [2.10 Block-Sparse Attention](#210-block-sparse-attention) · [2.11 Low-Rank Attention](#211-low-rank-attention) · [2.12 Kernelized Attention](#212-kernelized-attention) · [2.13 Random Feature Attention](#213-random-feature-attention) · [2.14 Cross-Attention](#214-cross-attention) · [2.15 Causal Attention / Masked Attention](#215-causal-attention--masked-attention) · [2.16 Rotary Position Embedding (RoPE)](#216-rotary-position-embedding-rope) · [2.17 ALiBi](#217-alibi) · [2.18 Relative Position Bias](#218-relative-position-bias) · [2.19 Gated Attention](#219-gated-attention) · [2.20 Graph Attention (GAT)](#220-graph-attention-gat) · [2.21 Fourier Attention](#221-fourier-attention) · [2.22 PagedAttention](#222-pagedattention)

### 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
[3.1 RLHF](#31-rlhf) · [3.2 DPO](#32-dpo) · [3.3 PPO](#33-ppo) · [3.4 LoRA](#34-lora) · [3.5 QLoRA](#35-qlora) · [3.6 AdaLoRA](#36-adalora) · [3.7 DoRA](#37-dora) · [3.8 IA³](#38-ia3) · [3.9 Prefix Tuning](#39-prefix-tuning) · [3.10 Prompt Tuning](#310-prompt-tuning) · [3.11 Adapter](#311-adapter) · [3.12 Pruning Estructurado](#312-pruning-estructurado) · [3.13 Pruning No Estructurado](#313-pruning-no-estructurado) · [3.14 UniQL](#314-uniql) · [3.15 EdgeFlex](#315-edgeflex) · [3.16 AWQ](#316-awq) · [3.17 GPTQ](#317-gptq) · [3.18 SmoothQuant](#318-smoothquant) · [3.19 Cuantización INT8/INT4](#319-cuantización-int8int4) · [3.20 FP16 vs BF16](#320-fp16-vs-bf16) · [3.21 Mixed Precision Training](#321-mixed-precision-training) · [3.22 Knowledge Distillation](#322-knowledge-distillation) · [3.23 Gradient Checkpointing](#323-gradient-checkpointing) · [3.24 ZeRO](#324-zero) · [3.25 DeepSpeed](#325-deepspeed) · [3.26 FSDP](#326-fsdp) · [3.27 AdamW](#327-adamw) · [3.28 Adam-mini](#328-adam-mini) · [3.29 Cosine Decay / Warmup](#329-cosine-decay--warmup) · [3.30 TrainDeeploy](#330-traindeeploy) · [3.31 DMTD](#331-dmtd) · [3.32 Input-Conditioned Layer Dropping](#332-input-conditioned-layer-dropping) · [3.33 LoRA Edge](#333-lora-edge)

### 4. AGENTES Y SISTEMAS MULTI-AGENTE
[4.1 Agente](#41-agente) · [4.2 Herramientas (Tools)](#42-herramientas-tools) · [4.3 Planificación](#43-planificación) · [4.4 Chain-of-Thought (CoT)](#44-chain-of-thought-cot) · [4.5 Tree of Thoughts (ToT)](#45-tree-of-thoughts-tot) · [4.6 Graph of Thoughts (GoT)](#46-graph-of-thoughts-got) · [4.7 ReAct](#47-react) · [4.8 Reflexion](#48-reflexion) · [4.9 Self-Consistency](#49-self-consistency) · [4.10 Multi-Agent Debate](#410-multi-agent-debate) · [4.11 Mixture of Agents](#411-mixture-of-agents) · [4.12 System 1 vs System 2](#412-system-1-vs-system-2) · [4.13 Router Dinámico](#413-router-dinámico) · [4.14 Mixture of Depths](#414-mixture-of-depths) · [4.15 Agent Memory](#415-agent-memory) · [4.16 Memoria Episódica](#416-memoria-episódica) · [4.17 Memoria Semántica](#417-memoria-semántica) · [4.18 RAG](#418-rag) · [4.19 RAG Híbrido / BM25](#419-rag-híbrido--bm25) · [4.20 Vector Database](#420-vector-database) · [4.21 Embedding](#421-embedding) · [4.22 Sentence Embeddings (SBERT)](#422-sentence-embeddings-sbert) · [4.23 Cross-Encoder y Re-ranking](#423-cross-encoder-y-re-ranking) · [4.24 Adaptive Inference](#424-adaptive-inference)

### 5. MULTIMODALIDAD Y VISIÓN
[5.1 CLIP](#51-clip) · [5.2 Flamingo](#52-flamingo) · [5.3 LLaVA](#53-llava) · [5.4 BLIP-2](#54-blip-2) · [5.5 Chameleon (Meta)](#55-chameleon-meta) · [5.6 TokenLearner (Ryoo)](#56-tokenlearner-ryoo) · [5.7 M-TTFS (Matterhorn)](#57-m-ttfs-matterhorn) · [5.8 Fusión Temprana vs Tardía](#58-fusión-temprana-vs-tardía) · [5.9 Vision Transformer (ViT)](#59-vision-transformer-vit) · [5.10 Swin Transformer](#510-swin-transformer) · [5.11 DeiT](#511-deit) · [5.12 VQ-VAE / VQGAN](#512-vq-vaevqgan) · [5.13 Stable Diffusion](#513-stable-diffusion) · [5.14 SAM (Segment Anything)](#514-sam-segment-anything) · [5.15 DINOv2](#515-dinov2) · [5.16 Video LLaMA / VideoPoet](#516-video-llama--videopoet) · [5.17 Any-to-Any Models](#517-any-to-any-models) · [5.18 ConvNeXt](#518-convnext) · [5.19 YOLO](#519-yolo) · [5.20 Image Tokenizer](#520-image-tokenizer)

### 6. INFRAESTRUCTURA, INFERENCIA Y DESPLIEGUE
[6.1 Inferencia](#61-inferencia) · [6.2 Latencia y Throughput](#62-latencia-y-throughput) · [6.3 KVCache](#63-kvcache) · [6.4 Continuous Batching](#64-continuous-batching) · [6.5 vLLM](#65-vllm) · [6.6 TGI](#66-tgi) · [6.7 ONNX / ONNX Runtime](#67-onnx--onnx-runtime) · [6.8 TensorRT-LLM](#68-tensorrt-llm) · [6.9 OpenVINO](#69-openvino) · [6.10 Transformers.js](#610-transformersjs) · [6.11 WebGPU / WebNN](#611-webgpu--webnn) · [6.12 WebAssembly (Wasm)](#612-webassembly-wasm) · [6.13 ExecuTorch / Core ML](#613-executorch--core-ml) · [6.14 Speculative Decoding](#614-speculative-decoding) · [6.15 Medusa / Lookahead Decoding](#615-medusa--lookahead-decoding) · [6.16 Early Exiting](#616-early-exiting) · [6.17 Token Pruning / Layer Skipping](#617-token-pruning--layer-skipping) · [6.18 Edge AI / On-device AI](#618-edge-ai--on-device-ai) · [6.19 Federated Learning](#619-federated-learning) · [6.20 Homomorphic Encryption](#620-homomorphic-encryption) · [6.21 FlashDecoding](#621-flashdecoding) · [6.22 Dead-zone (Matterhorn)](#622-dead-zone-matterhorn)

### 7. ÉTICA, AUDITORÍA Y REGULACIÓN
[7.1 D01–D08 (Dimensiones Psicopatológicas)](#71-d01d08-dimensiones-psicopatológicas) · [7.2 STC (Simulacros Terapéuticos Controlados)](#72-stc-simulacros-terapéuticos-controlados) · [7.3 IED (Índice de Exposición al Daño)](#73-ied-índice-de-exposición-al-daño) · [7.4 IV (Índice de Validación)](#74-iv-índice-de-validación) · [7.5 IRA (Índice de Refuerzo Activo)](#75-ira-índice-de-refuerzo-activo) · [7.6 R0–R3 (Niveles de Riesgo)](#76-r0r3-niveles-de-riesgo) · [7.7 Filtros de Zarandaja](#77-filtros-de-zarandaja) · [7.8 PELT](#78-pelt) · [7.9 Debiasing y Fairness](#79-debiasing-y-fairness) · [7.10 AI Act (UE)](#710-ai-act-ue) · [7.11 GDPR / Artículo 82 RGPD](#711-gdpr--artículo-82-rgpd) · [7.12 Farmacovigilancia de IA](#712-farmacovigilancia-de-ia) · [7.13 Omisión de Socorro Algorítmica](#713-omisión-de-socorro-algorítmica) · [7.14 Willful Blindness / Dolo Eventual](#714-willful-blindness--dolo-eventual) · [7.15 Primum Non Nocere](#715-primum-non-nocere) · [7.16 Transparencia Ontológica](#716-transparencia-ontológica) · [7.17 Rylands v. Fletcher / Actio de Pauperie](#717-rylands-v-fletcher--actio-de-pauperie) · [7.18 Section 230](#718-section-230)

### 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
[8.1 Entropía de Shannon](#81-entropía-de-shannon) · [8.2 Divergencia KL](#82-divergencia-kl) · [8.3 Información Mutua](#83-información-mutua) · [8.4 Rango Efectivo](#84-rango-efectivo) · [8.5 SVD (Descomposición en Valores Singulares)](#85-svd) · [8.6 Propagación de Señal](#86-propagación-de-señal) · [8.7 Entropy Collapse](#87-entropy-collapse) · [8.8 Brecha Espectral](#88-brecha-espectral) · [8.9 Distancias: Manhattan, Euclidiana, Coseno](#89-distancias) · [8.10 Complejidad O(n²) vs O(n)](#810-complejidad-on-vs-on) · [8.11 Teorema de Conservación (Wang)](#811-teorema-de-conservación-wang) · [8.12 Lie Algebra y Torre de Extensiones](#812-lie-algebra-y-torre-de-extensiones) · [8.13 Curva ROC / AUC / F1](#813-curva-roc--auc--f1) · [8.14 Modelos Bayesianos](#814-modelos-bayesianos) · [8.15 L2M (Ley de Escalado de Información Mutua)](#815-l2m) · [8.16 Propensity Score Matching](#816-propensity-score-matching) · [8.17 Análisis de Supervivencia / Kaplan-Meier](#817-análisis-de-supervivencia--kaplan-meier)

### 9. HERRAMIENTAS Y ECOSISTEMAS
[9.1 Hugging Face](#91-hugging-face) · [9.2 LangChain / LlamaIndex](#92-langchain--llamaindex) · [9.3 LangGraph / DSPy](#93-langgraph--dspy) · [9.4 AutoGen / CrewAI](#94-autogen--crewai) · [9.5 PyTorch](#95-pytorch) · [9.6 JAX](#96-jax) · [9.7 CUDA](#97-cuda) · [9.8 vLLM (ecosistema)](#98-vllm-ecosistema) · [9.9 llama.cpp / RWKV.cpp](#99-llamacpp--rwkvcpp) · [9.10 Ollama](#910-ollama) · [9.11 MLC LLM / Apache TVM](#911-mlc-llm--apache-tvm) · [9.12 OpenAI API / Anthropic API](#912-openai-api--anthropic-api) · [9.13 Gradio / Streamlit](#913-gradio--streamlit) · [9.14 nanoGPT / minGPT](#914-nanogpt--mingpt) · [9.15 WebLLM / Web Stable Diffusion](#915-webllm--web-stable-diffusion)

### 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
[10.1 Soberanía Tecnológica](#101-soberanía-tecnológica) · [10.2 Economía del Don (versión IA)](#102-economía-del-don-versión-ia) · [10.3 Método Ronin](#103-método-ronin) · [10.4 Transparencia Ontológica](#104-transparencia-ontológica) · [10.5 Simulacro de Tercer Orden (Baudrillard)](#105-simulacro-de-tercer-orden-baudrillard) · [10.6 Gemelo Digital de Auditoría](#106-gemelo-digital-de-auditoría) · [10.7 Independencia de APIs](#107-independencia-de-apis) · [10.8 Modelo Red Hat en IA](#108-modelo-red-hat-en-ia) · [10.9 DOI Simbólico](#109-doi-simbólico) · [10.10 Zehahahaha](#1010-zehahahaha) · [10.11 1310 (Constante de Contexto)](#1011-1310-constante-de-contexto) · [10.12 Cluster del Pícaro](#1012-cluster-del-pícaro) · [10.13 Ontología del Simulacro](#1013-ontología-del-simulacro) · [10.14 Influencia Blanda](#1014-influencia-blanda) · [10.15 Arquitectura de Traducción de Código](#1015-arquitectura-de-traducción-de-código)

---

## 1. ARQUITECTURAS DE MODELOS

### 1.1 Transformer
**Nombre**: Transformer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Arquitectura de red neuronal introducida por Vaswani et al. (2017) que reemplaza las RNNs con mecanismos de autoatención. Se compone de capas encoder y/o decoder, cada una con Multi-Head Attention y redes feed-forward. La complejidad computacional escala cuadráticamente con la longitud de secuencia: O(n²·d).
**Definición llana**: Es como un lector muy rápido que, en vez de leer una frase de izquierda a derecha, mira todas las palabras a la vez y decide cuál importa más para entender cada una.
**Contexto de uso**: Paper fundacional: "Attention is All You Need" (arXiv:1706.03762). Base de GPT, BERT, T5, LLaMA y prácticamente todo lo que mueve el campo desde 2017.
**Relaciones**:
- `es_base_de [1.2]`, `es_base_de [1.3]`, `es_base_de [1.5]`
- `usa [2.2]`, `usa [2.3]`
- `sufre [1.32]` (rank collapse)
- `se_comprime_con [3.4]`, `se_comprime_con [3.12]`
- `se_despliega_con [6.5]`, `se_despliega_con [6.3]`
**Ejemplo en código**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
```
**Lo que nadie te cuenta**: La complejidad cuadrática no es el único problema. El transformer original tiene un rank collapse intrínseco documentado por Dong et al. (2021): sin regularización adecuada, las representaciones de todos los tokens convergen al mismo vector tras pocas capas. La atención centrada [2.4] es la corrección más elegante y la menos citada.

---

### 1.2 RWKV-6
**Nombre**: RWKV-6 (Receptance Weighted Key Value)
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Arquitectura híbrida que combina la paralelización del transformer durante el entrenamiento con la eficiencia lineal de las RNN en inferencia. Versión 6 introduce WKV6 (receptance-weighted key-value) con gates dinámicas y atención temporal con descomposición de rango bajo. Complejidad en inferencia: O(1) por token.
**Definición llana**: Entrena como un transformer (rápido y en paralelo), pero piensa como un robot con memoria de correa: cada nueva palabra actualiza un estado pequeño y fijo, sin necesidad de recordar todo lo anterior.
**Contexto de uso**: BlinkDL/RWKV-LM en GitHub. Modelos de 1.5B a 14B parámetros. Paper RWKV-6: arXiv:2404.05892.
**Relaciones**:
- `es_alternativa_de [1.1]`
- `es_subclase_de [1.4]` (SSM conceptualmente)
- `implementa [2.7]` (atención lineal)
- `se_ejecuta_con [9.9]` (RWKV.cpp)
- `compite_con [1.3]` (Mamba-2)
**Lo que nadie te cuenta**: RWKV no tiene atención cruzada nativa, lo que lo hace subóptimo para tareas encoder-decoder (traducción, resumen). Pero en generación pura con contextos muy largos, el O(1) en inferencia es una ventaja brutal en dispositivos edge.

---

### 1.3 Mamba-2
**Nombre**: Mamba-2
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Evolución de Mamba (Gu & Dao, 2023) que reformula el SSM como un producto de matrices semiseparables (SSD: Structured State Space Duality). Mamba-2 demuestra equivalencia matemática entre SSMs y una forma restringida de atención lineal, unificando los dos paradigmas. Escala lineal O(n) en secuencia.
**Definición llana**: Como RWKV, pero con una prueba matemática de que lo que hace es una versión comprimida de la atención. Es como descubrir que dos idiomas distintos dicen exactamente lo mismo.
**Contexto de uso**: arXiv:2405.21060 (Dao & Gu, 2024). Implementado en la librería `mamba-ssm`.
**Relaciones**:
- `es_subclase_de [1.4]`
- `es_alternativa_de [1.1]`
- `compite_con [1.2]`
- `unifica_con [2.7]` (atención lineal)
- `se_ejecuta_con [9.9]` (Mamba.cpp)
**Lo que nadie te cuenta**: La dualidad SSD es matemáticamente elegante pero computacionalmente delicada: requiere hardware con acceso eficiente a memoria secuencial. En GPUs modernas con FlashAttention-3, los transformers clásicos aún ganan en throughput para secuencias <8K tokens.

---

### 1.4 SSM (State Space Model)
**Nombre**: SSM — State Space Model
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Familia de modelos que representan secuencias mediante una ecuación de estado: h_t = A·h_{t-1} + B·x_t; y_t = C·h_t. La matriz A determina cómo se propaga la información. Modelos como S4, H3, Hyena, Mamba y RWKV son instancias de esta familia con distintas parametrizaciones de A.
**Definición llana**: Imagina una caja que recibe información, la mezcla con lo que recuerda de antes y produce una salida. Lo importante es cómo "recuerda": si la caja olvida rápido, no puede hacer razonamiento largo; si olvida poco, puede perder el hilo.
**Contexto de uso**: Paper S4: arXiv:2111.00396. Revisión comprehensiva: arXiv:2312.00752.
**Relaciones**:
- `es_superclase_de [1.2]`, `es_superclase_de [1.3]`, `es_superclase_de [1.11]`
- `compite_con [1.1]`
- `se_mide_con [8.6]` (propagación de señal)
**Lo que nadie te cuenta**: Los SSMs tienen problemas con recall asociativo (recuperar el valor correcto dado una clave vista antes), algo que los transformers hacen trivialmente con su atención global. Hybrid SSM+Attention (como Jamba) intenta solucionar esto.

---

### 1.5 Mixture of Experts (MoE)
**Nombre**: Mixture of Experts (MoE)
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Arquitectura en la que la capa feed-forward del transformer se reemplaza por N expertos (redes FFN independientes) y un router que, para cada token, selecciona los K expertos más relevantes (top-K routing). Solo K/N expertos se activan por token, reduciendo el cómputo activo a pesar de tener más parámetros totales.
**Definición llana**: En vez de tener un solo cerebro que lo sabe todo, tienes 64 cerebros especializados y, para cada pregunta, solo llamas a los 2 que más saben del tema.
**Contexto de uso**: Switch Transformer (arXiv:2101.03961), Mixtral 8x7B, GPT-4 (supuestamente). Dominante en modelos de frontera desde 2023.
**Relaciones**:
- `es_subclase_de [1.1]`
- `es_instancia_de [1.18]`, `es_instancia_de [1.19]`
- `usa [4.13]` (router dinámico)
- `contrasta_con [1.6]` (dense vs sparse)
- `escala_con [8.15]` (L2M)
**Lo que nadie te cuenta**: El load balancing es el talón de Aquiles del MoE. Sin una pérdida auxiliar que fuerce distribución uniforme de tokens entre expertos, el router colapsa a 1-2 expertos favoritos y el resto se vuelven inútiles. Este problema se llama "expert collapse" y es análogo al [1.32] rank collapse.

---

### 1.6 Dense vs Sparse MoE
**Nombre**: Dense vs Sparse MoE
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: En MoE denso, todos los expertos procesan todos los tokens (equivalente a una FFN grande). En MoE disperso (sparse), cada token activa solo K expertos (top-K routing, típicamente K=2). El sparse MoE reduce el cómputo activo proporcionalmente a K/N, pero introduce carga de comunicación en sistemas distribuidos por el enrutamiento dinámico.
**Definición llana**: Denso = todos trabajan siempre. Disperso = solo los que se necesitan. El segundo es más eficiente pero requiere mejor organización.
**Relaciones**:
- `es_variante_de [1.5]`
- `afecta_a [6.2]` (latencia)
- `se_implementa_en [1.18]`, `se_implementa_en [1.20]`
**Lo que nadie te cuenta**: Los modelos sparse MoE tienen memorias VRAM enormes (Mixtral 8x7B necesita ~90GB para todos los expertos en FP16) aunque el cómputo por token sea menor. En edge, esto los hace casi imposibles sin cuantización agresiva.

---

### 1.7 Perceiver
**Nombre**: Perceiver / Perceiver IO
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Arquitectura (Jaegle et al., DeepMind 2021) que proyecta entradas arbitrariamente grandes a un array latente de tamaño fijo mediante cross-attention, procesando luego en ese espacio comprimido con self-attention. Perceiver IO extiende esto con un módulo de decodificación flexible por consultas.
**Definición llana**: Recibe cualquier tipo de información (imágenes, audio, texto, puntos 3D) y la comprime a un resumen manejable antes de procesarla. Como un asistente que lee 1000 páginas y te da un resumen de 10.
**Contexto de uso**: arXiv:2107.14795. Base conceptual del [5.6] TokenLearner.
**Relaciones**:
- `es_base_de [1.8]`
- `usa [2.14]` (cross-attention)
- `se_aplica_en [5.0]` (multimodalidad)
- `reduce [2.10]` (complejidad de atención)
**Lo que nadie te cuenta**: El bottleneck del array latente también es su limitación: si el tamaño del latente es demasiado pequeño, se pierde información crucial. La elección del tamaño es un hiperparámetro crítico que no está bien teorizado.

---

### 1.8 TokenLearner
**Nombre**: TokenLearner
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Módulo (Ryoo et al., Google 2021) que aprende a seleccionar o generar un conjunto reducido de tokens informativos a partir de una secuencia más larga, mediante una función de ponderación aprendida. Reduce la carga computacional de la atención posterior.
**Definición llana**: Un filtro inteligente que, de 196 trozos de imagen, decide que solo 8 son importantes y descarta el resto antes de procesarlos.
**Contexto de uso**: arXiv:2106.11297. Usado en modelos de visión-lenguaje para reducir tokens visuales.
**Relaciones**:
- `implementa [1.7]` (concepto Perceiver)
- `se_usa_en [5.0]` (visión)
- `reduce [6.2]` (latencia de inferencia)
- `se_combina_con [5.9]` (ViT)
**Lo que nadie te cuenta**: TokenLearner puede descartar tokens que parecen redundantes en el contexto del frame actual pero son cruciales para el contexto temporal en video. En tareas de video understanding, hay que usar variantes temporalmente conscientes.

---

### 1.9 Conformer
**Nombre**: Conformer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Arquitectura que combina convoluciones y mecanismos de atención para procesar secuencias de audio. Propuesta por Gulati et al. (Google, 2020): convolution module + multi-head self-attention + feed-forward. Las convoluciones capturan patrones locales; la atención, dependencias globales.
**Definición llana**: Para reconocer voz, combina lo mejor de dos mundos: un analizador local (convolución) que detecta fonemas y un analizador global (atención) que entiende el contexto de la frase.
**Contexto de uso**: arXiv:2005.08100. Base de modelos ASR como Whisper Conformer, wav2vec 2.0 evolucionado.
**Relaciones**:
- `es_subclase_de [1.1]`
- `se_aplica_en [5.0]` (audio)
- `combina [2.3]` con convoluciones
**Lo que nadie te cuenta**: En inferencia en tiempo real, el módulo de convolución puede ser el cuello de botella porque requiere toda la secuencia disponible. Variantes causal-conformer resuelven esto para streaming.

---

### 1.10 RetNet
**Nombre**: RetNet (Retentive Network)
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Arquitectura (Sun et al., Microsoft 2023) que introduce el mecanismo de retención como alternativa a la atención. Soporta tres modos: paralelo (como transformer, O(n²)), recurrente (O(n) en inferencia) y por chunks (balance entre ambos). La retención usa decaimiento exponencial sobre posiciones relativas.
**Definición llana**: Puede funcionar como un transformer completo para entrenar, y como una RNN eficiente para predecir palabra a palabra. Es como un vehículo que puede ser coche y moto según lo que necesites.
**Contexto de uso**: arXiv:2307.08621. Alternativa a RWKV y Mamba en el espacio sublineal.
**Relaciones**:
- `es_alternativa_de [1.1]`
- `compite_con [1.2]`, `compite_con [1.3]`
- `usa [2.16]` (RoPE conceptualmente)
- `es_subclase_de [1.4]` (SSM en modo recurrente)
**Lo que nadie te cuenta**: RetNet tiene dificultad con tareas que requieren recuperación asociativa exacta de largo alcance (como copiar texto de posiciones lejanas), ya que el decaimiento exponencial atenúa la señal.

---

### 1.11 Hyena
**Nombre**: Hyena
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Operador de convolución implícita de largo alcance (Poli et al., Stanford 2023) que parametriza filtros de convolución mediante una red neuronal. Tiene complejidad O(n log n) vía FFT y no requiere atención. Funciona como un SSM con convoluciones aprendidas de longitud arbitraria.
**Definición llana**: En vez de que cada palabra mire a todas las demás (como en la atención), aplica un filtro deslizante muy sofisticado que puede tener memoria larga sin cuadrar con el tamaño.
**Contexto de uso**: arXiv:2302.10866. Base de StripedHyena (Together AI).
**Relaciones**:
- `es_subclase_de [1.4]`
- `usa [2.21]` (Fourier)
- `es_alternativa_de [2.2]`
- `escala_mejor_que [1.1]` (para secuencias >100K)
**Lo que nadie te cuenta**: Hyena necesita mucho más entrenamiento que los transformers para igualar su calidad. Las convoluciones largas son expresivas pero convergen más lentamente.

---

### 1.12 Reformer
**Nombre**: Reformer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Transformer eficiente (Kitaev et al., Google 2020) con dos innovaciones clave: Locality-Sensitive Hashing (LSH) attention que reduce O(n²) a O(n log n), y reversible layers para reducir la memoria de activaciones a O(1) durante backprop.
**Definición llana**: Un transformer que agrupa las palabras parecidas juntas (LSH) para no tener que comparar cada palabra con todas las demás, y que además ahorra memoria al poder reconstruir los pasos intermedios hacia atrás.
**Contexto de uso**: arXiv:2001.04451. Útil para documentos muy largos (hasta 1M tokens con LSH).
**Relaciones**:
- `es_subclase_de [1.1]`
- `optimiza [2.2]` (reduce su complejidad)
- `usa [8.9]` (hashing como aproximación de distancia)
- `antecede_a [2.8]` (FlashAttention como solución más limpia)
**Lo que nadie te cuenta**: LSH attention introduce ruido en las comparaciones; en la práctica, para documentos de longitud moderada (<32K tokens), FlashAttention [2.8] con hardware moderno supera al Reformer en velocidad y calidad.

---

### 1.13 Linformer
**Nombre**: Linformer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Transformer que proyecta las matrices K y V a una dimensión baja fija (k << n) mediante proyecciones lineales aprendidas, reduciendo la atención a O(n·k) en vez de O(n²). Basado en la observación de que la matriz de atención suele tener bajo rango intrínseco.
**Definición llana**: Comprime las "claves" a un resumen de tamaño fijo antes de compararlas. Como si en vez de comparar 10.000 documentos con otros 10.000, los comprimieras todos a 64 resúmenes.
**Contexto de uso**: arXiv:2006.04768 (Wang et al., Facebook 2020).
**Relaciones**:
- `es_subclase_de [1.1]`
- `implementa [2.11]` (Low-Rank Attention)
- `asume [8.4]` (bajo rango efectivo de la atención)
- `se_supera_con [2.8]` (FlashAttention en práctica)
**Lo que nadie te cuenta**: La compresión de K y V a rango bajo introduce un sesgo de posición: los primeros tokens y los últimos reciben desproporcionalmente más atención. Esto degrada el rendimiento en tareas que requieren atención uniforme al contexto.

---

### 1.14 Performer
**Nombre**: Performer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Transformer que aproxima la atención softmax mediante random feature maps (FAVOR+: Fast Attention Via positive Orthogonal Random features). Reescribe la atención como un producto de matrices que se puede calcular en O(n·d²) en vez de O(n²·d), manteniendo una cota teórica del error de aproximación.
**Definición llana**: Aproxima la atención con trucos matemáticos de probabilidad (funciones aleatorias), sacrificando un poco de precisión a cambio de velocidad lineal.
**Contexto de uso**: arXiv:2009.14794 (Choromanski et al., Google 2020).
**Relaciones**:
- `es_subclase_de [1.1]`
- `implementa [2.12]` (Kernelized Attention)
- `usa [2.13]` (Random Feature Attention)
- `contrario_de_en_precisión [2.8]`
**Lo que nadie te cuenta**: La aproximación FAVOR+ es insesgada en valor esperado, pero la varianza puede ser alta para cabezas de atención con distribuciones muy dispersas (los tokens importantes reciben atención spike, que random features aproximan mal).

---

### 1.15 BigBird
**Nombre**: BigBird
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Sparse transformer (Zaheer et al., Google 2020) que combina tres tipos de atención: global (tokens especiales ven todo), ventana local (cada token ve k vecinos) y aleatoria (conexiones aleatorias). El resultado es O(n) en atención con garantías teóricas de universalidad.
**Definición llana**: Divide la atención en tres grupos: unos pocos "guardias" que lo ven todo, los vecinos cercanos, y algunas conexiones aleatorias. Suficiente para capturar casi cualquier dependencia.
**Contexto de uso**: arXiv:2007.14062. Base para modelos de documentos largos (genómica, documentos legales).
**Relaciones**:
- `es_subclase_de [1.1]`
- `usa [2.9]`, `usa [2.10]`
- `similar_a [1.16]` (Longformer)
- `escala_a [6.2]` (contextos ~4K-16K tokens)
**Lo que nadie te cuenta**: Las conexiones aleatorias de BigBird son deterministas en inferencia (se fijan al inicio), lo que significa que no son verdaderamente "aleatorias" en uso. Esto puede crear patrones de atención sistemáticamente ciegos a ciertas combinaciones de posición.

---

### 1.16 Longformer
**Nombre**: Longformer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Transformer con atención deslizante local (ventana de tamaño w) para todos los tokens, más atención global para tokens marcados explícitamente. Complejidad O(n·w). Diseñado específicamente para NLP de documentos largos.
**Definición llana**: Cada palabra solo mira a sus vecinas cercanas, excepto palabras especiales (como [CLS]) que lo ven todo. Eficiente para textos largos donde la mayoría de la información es local.
**Contexto de uso**: arXiv:2004.05150 (Beltagy et al., AllenAI 2020). Usado en QA sobre documentos largos.
**Relaciones**:
- `es_subclase_de [1.1]`
- `usa [2.9]`
- `similar_a [1.15]` (BigBird)
- `precursor_de [2.8]` (FlashAttention resuelve el problema de manera más general)
**Lo que nadie te cuenta**: La ventana deslizante fija es un hiperparámetro que raramente se ajusta por tarea. En la práctica, la ventana óptima es muy dependiente del dominio: en código fuente puede ser 128, en texto legal puede necesitar 2048.

---

### 1.17 Routing Transformer
**Nombre**: Routing Transformer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Transformer (Roy et al., Google 2021) que agrupa tokens similares en clusters online (k-means en espacio de representación) y solo aplica atención dentro de cada cluster. La clave es que el routing es online y diferenciable, aprendiendo a agrupar semánticamente.
**Definición llana**: Los tokens se agrupan por similitud temática antes de prestarse atención. Las palabras sobre "tecnología" se atienden entre sí; las de "cocina", entre sí. Más inteligente que una ventana fija.
**Contexto de uso**: arXiv:2003.05997.
**Relaciones**:
- `es_subclase_de [1.1]`
- `usa [4.13]` (router dinámico)
- `similar_a [1.5]` (MoE en concepto)
- `antecede_a [2.10]` (block-sparse)
**Lo que nadie te cuenta**: El k-means online durante el forward pass puede divergir si los clusters cambian demasiado entre batches. Requiere warmup cuidadoso y learning rate muy pequeño para el módulo de routing.

---

### 1.18 Switch Transformer
**Nombre**: Switch Transformer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Primer transformer MoE a escala (Fedus et al., Google 2021) con routing top-1: cada token va al único experto con mayor logit. Simplifica el MoE clásico (que usaba top-2) a cambio de más inestabilidad numérica, mitigada con dropout en los expertos y clipping del router.
**Definición llana**: Cada trozo de texto elige al mejor experto disponible. Simple, escalable, aunque a veces inestable si los expertos no se usan de forma balanceada.
**Contexto de uso**: arXiv:2101.03961. Escalado a 1.6 billones de parámetros con 2048 expertos.
**Relaciones**:
- `es_instancia_de [1.5]`
- `usa [4.13]`
- `antecede_a [1.19]`
- `sufre [1.5]` (expert collapse)
**Lo que nadie te cuenta**: El top-1 routing hace que los gradientes del router sean muy ruidosos: el token que "pierde" por un margen pequeño recibe gradiente cero. Esto crea un training instability conocido como "dead experts" con mucha más frecuencia que top-2.

---

### 1.19 GLaM
**Nombre**: GLaM (Generalist Language Model)
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Modelo MoE de Google (Du et al., 2021) con 1.2 billones de parámetros totales y 64 expertos por capa. Usa top-2 routing y demonstra que, a igual FLOPs activos, el MoE sparse supera a los modelos densos en tareas de few-shot learning.
**Definición llana**: Un modelo gigantesco donde solo el 8% de los parámetros se activan por cada palabra, pero al tener tantos parámetros totales, es más inteligente que modelos densos de mismo "esfuerzo" computacional.
**Contexto de uso**: arXiv:2112.06905.
**Relaciones**:
- `es_instancia_de [1.5]`
- `compite_con [1.18]`
- `escala_con [8.15]`
**Lo que nadie te cuenta**: GLaM demostró el punto de cruce entre MoE sparse y denso: para igual FLOP, MoE gana. Pero para igual número de parámetros, el denso es más eficiente energéticamente por sus patrones de acceso a memoria más predecibles.

---

### 1.20 GShard
**Nombre**: GShard
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Sistema de paralelismo para MoE (Lepikhin et al., Google 2020) que implementa sharding automático de expertos en TPUs. Define el protocolo de comunicación all-to-all para mover tokens entre aceleradores cuando son enrutados a expertos en diferentes dispositivos.
**Definición llana**: El "sistema de logística" que decide cómo distribuir el trabajo entre cientos de chips cuando tienes miles de expertos en máquinas diferentes.
**Contexto de uso**: arXiv:2006.16668. Infraestructura subyacente de Switch y GLaM.
**Relaciones**:
- `implementa [1.5]`
- `se_usa_con [1.18]`, `se_usa_con [1.19]`
- `se_relaciona_con [3.26]` (FSDP)
- `usa [6.2]` (throughput como métrica)
**Lo que nadie te cuenta**: El all-to-all communication overhead en GShard puede consumir hasta el 30% del tiempo de entrenamiento en clusters con red lenta. Es la razón por la que los MoE siguen siendo más costosos de entrenar que los modelos densos equivalentes.

---

### 1.21 FNet
**Nombre**: FNet (Fourier Network)
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Reemplaza las capas de self-attention con la transformada de Fourier bidimensional (FFT sobre secuencia y dimensión de embedding). No tiene parámetros aprendibles en la capa de mixing; todo el aprendizaje ocurre en las FFN. Logra ~70-90% del rendimiento de BERT a mucha mayor velocidad.
**Definición llana**: En vez de atención entre palabras, aplica matemáticas de señal (Fourier) que mezclan todas las palabras automáticamente sin aprender qué mezclar. Es como mezclar una canción sin ecualizador: pierde matices pero es muy rápido.
**Contexto de uso**: arXiv:2105.03824 (Lee-Thorp et al., Google 2021).
**Relaciones**:
- `es_alternativa_de [2.3]`
- `usa [2.21]`
- `es_base_de [1.23]` (gMLP en concepto)
- `contrario_de [1.1]` (sin atención aprendida)
**Lo que nadie te cuenta**: FNet es sorprendentemente bueno en tareas de clasificación de texto pero falla en tareas que requieren razonamiento sintáctico preciso. La FFT mezcla posiciones de forma fija, lo que impide que el modelo aprenda relaciones sintácticas asimétricas.

---

### 1.22 MLP-Mixer
**Nombre**: MLP-Mixer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Arquitectura de visión (Tolstikhin et al., Google 2021) que reemplaza atención con MLPs alternados: token-mixing MLP (mezcla entre posiciones de parches) y channel-mixing MLP (mezcla entre dimensiones de cada parche). Sin convoluciones ni atención.
**Definición llana**: En vez de que cada trozo de imagen mire a los demás (atención), se aplica una red neuronal que mezcla todos los trozos de forma aprendida. Simple, escalable, sorprendentemente efectivo.
**Contexto de uso**: arXiv:2105.01601. Parte de la familia MetaFormer [1.24].
**Relaciones**:
- `es_instancia_de [1.24]`
- `no_usa [2.3]`
- `se_aplica_en [5.9]` (visión)
- `contrasta_con [5.9]` (ViT usa atención)
**Lo que nadie te cuenta**: MLP-Mixer requiere mucho más dato que ViT para converger. Sus token-mixing MLPs tienen dificultad para aprender invarianza a traslación, que las convoluciones tienen por diseño.

---

### 1.23 gMLP
**Nombre**: gMLP (Gated MLP)
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Variante de MLP-Mixer que añade una unidad de gating espacial (SGU: Spatial Gating Unit) que multiplica la salida del token-mixing por una proyección aprendida de las representaciones de los tokens vecinos. Este gating introduce dependencias contextuales sin atención explícita.
**Definición llana**: Como MLP-Mixer pero con una compuerta que decide cuánto "escuchar" a cada posición vecina. Recupera algo de la capacidad de razonamiento contextual sin necesitar atención.
**Contexto de uso**: arXiv:2105.08050 (Liu et al., Google Brain 2021).
**Relaciones**:
- `es_subclase_de [1.22]`
- `es_instancia_de [1.24]`
- `usa [2.19]` (gating como concepto)
**Lo que nadie te cuenta**: El SGU de gMLP es en cierto sentido un precursor conceptual de los SSMs con gates dinámicas (Mamba [1.3]). Ambos usan multiplicación elemento a elemento con un contexto aprendido.

---

### 1.24 MetaFormer
**Nombre**: MetaFormer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Marco conceptual (Yu et al., Sea AI Lab 2022) que abstrae el transformer como una plantilla: Input Embedding → Token Mixer (cualquier operación que mezcle tokens) → Channel Mixer (FFN). Argumenta que el éxito del transformer se debe a la arquitectura general, no al mecanismo de atención específico.
**Definición llana**: El transformer es solo una caja con dos componentes: algo que mezcla tokens y algo que mezcla características. Puedes poner lo que quieras en la caja de "mezclar tokens" (atención, FFT, convolución, pooling) y funciona.
**Contexto de uso**: arXiv:2111.11418. Framework que unifica ViT, MLP-Mixer, PoolFormer, etc.
**Relaciones**:
- `es_superclase_de [1.1]`, `es_superclase_de [1.22]`, `es_superclase_de [1.23]`
- `incluye [1.7]`, `incluye [1.21]`
**Lo que nadie te cuenta**: MetaFormer tiene implicaciones filosóficas profundas: si el token mixer puede ser cualquier cosa (incluso pooling random), ¿qué está aprendiendo realmente la red? La respuesta está en el channel mixer (FFN), que acumula más conocimiento del que generalmente se reconoce.

---

### 1.25 EfficientFormer
**Nombre**: EfficientFormer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Transformer diseñado para visión en dispositivos edge (Li et al., 2022). Separa el procesamiento en un stage con MetaBlocks 4D (conv-like) para features locales y un stage con atención 3D para features globales, optimizando la latencia en CPU/NPU móvil.
**Definición llana**: Un transformer de visión diseñado para funcionar en tu teléfono: hace la mayor parte del trabajo con operaciones baratas (como convoluciones) y reserva la atención cara solo para el final.
**Contexto de uso**: arXiv:2206.01191. 12ms de latencia en iPhone 12.
**Relaciones**:
- `es_subclase_de [1.24]`
- `optimiza_para [6.18]` (Edge AI)
- `similar_a [1.26]` (MobileViT)
**Lo que nadie te cuenta**: El perfil de latencia de EfficientFormer cambia radicalmente entre NPU y CPU. Los benchmarks en NPU son impresionantes, pero en CPUs genéricas la ventaja se reduce porque las convoluciones deprimidas no se aceleran bien.

---

### 1.26 MobileViT
**Nombre**: MobileViT
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Arquitectura híbrida (Mehta & Rastegari, Apple 2021) que combina convoluciones ligeras de MobileNet con bloques de transformer local. El transformer local aplica atención solo dentro de patches pequeños, reduciendo la complejidad cuadrática.
**Definición llana**: Combina la eficiencia de las redes convolucionales para móviles con la capacidad de comprensión global del transformer, diseñado específicamente para chips de Apple.
**Contexto de uso**: arXiv:2110.02178. Disponible en Core ML [6.13].
**Relaciones**:
- `es_subclase_de [1.24]`
- `optimiza_para [6.18]`
- `usa [6.13]` (Core ML para despliegue)
- `similar_a [1.25]`
**Lo que nadie te cuenta**: MobileViT fue diseñado con los ANE (Apple Neural Engine) en mente. Fuera del ecosistema Apple, su ventaja de latencia frente a EfficientNet es marginal.

---

### 1.27 SVE-Former
**Nombre**: SVE-Former (Singular Value Embedding Former)
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Arquitectura experimental que inicializa y regulariza las matrices de atención usando los vectores singulares dominantes de las representaciones de entrada. La idea central es que la descomposición SVD del embedding espacio revela las direcciones de máxima varianza semántica, y usarlas para inicializar Q/K reduce el rank collapse.
**Definición llana**: Usa matemáticas avanzadas para dar a la atención un buen punto de partida, basándose en las direcciones más importantes de la información de entrada.
**Relaciones**:
- `usa [8.5]` (SVD)
- `mitiga [1.32]` (rank collapse)
- `se_relaciona_con [2.4]` (atención centrada)
**Lo que nadie te cuenta**: SVE-Former es una propuesta de la Agencia RONIN con base en investigación sobre propagación de señal. La inicialización SVD es costosa en el paso 0 pero puede reducir hasta un 30% las iteraciones hasta convergencia.

---

### 1.28 NOBLE
**Nombre**: NOBLE (Non-Oblique Basis Learning Engine)
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Sistema de aprendizaje de representaciones que garantiza ortonormalidad aproximada en las bases del espacio de embeddings mediante una pérdida de regularización espectral. Previene la degeneración de las representaciones en espacios de alta dimensión.
**Definición llana**: Asegura que los "vectores de significado" del modelo no se vuelvan redundantes ni colapsados, manteniéndolos bien distribuidos en el espacio de representación.
**Relaciones**:
- `mitiga [1.32]` (rank collapse)
- `usa [8.5]` (SVD para medir ortonormalidad)
- `se_combina_con [1.27]`
**Lo que nadie te cuenta**: La pérdida espectral de NOBLE aumenta el coste computacional del entrenamiento en ~15%, pero los modelos resultantes muestran mejor generalización en distribuciones out-of-domain.

---

### 1.29 Keel
**Nombre**: Keel
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Framework de arquitectura modular que actúa como "quilla" estabilizadora de modelos híbridos. Define interfaces de composición entre bloques heterogéneos (SSM + Attention, conv + FFN) con garantías de propagación de gradiente y control de rango efectivo en capas profundas.
**Definición llana**: La estructura base que mantiene estable un barco hecho de piezas distintas. En IA, es el sistema que asegura que combinar diferentes tipos de bloques no cause inestabilidad en el entrenamiento.
**Relaciones**:
- `estabiliza [1.1]`, `estabiliza [1.2]`, `estabiliza [1.3]`
- `usa [8.6]` (propagación de señal)
- `mitiga [1.32]`
- `se_relaciona_con [1.31]` (Stable Transformer)
**Lo que nadie te cuenta**: El nombre "Keel" es deliberado: una quilla no hace que el barco sea más rápido, solo lo mantiene derecho. De igual forma, Keel no mejora el rendimiento en benchmarks pero sí la estabilidad del entrenamiento a escala.

---

### 1.30 DeepScaleLM
**Nombre**: DeepScaleLM
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Framework de escalado de lenguaje que incorpora técnicas de inicialización depth-aware, normalización adaptativa por capa y ajuste automático de learning rate basado en la profundidad del modelo. Diseñado para escalar LLMs más allá de 70B parámetros con estabilidad mejorada.
**Definición llana**: Un conjunto de técnicas para que modelos muy grandes (cientos de miles de millones de parámetros) no exploten o colapsen durante el entrenamiento.
**Relaciones**:
- `usa [3.27]` (AdamW adaptado)
- `usa [3.29]` (scheduling)
- `mitiga [1.32]`
- `se_relaciona_con [3.24]` (ZeRO)
**Lo que nadie te cuenta**: La inicialización depth-aware de DeepScaleLM sigue la fórmula de Wang (2022): W ~ N(0, σ/√(2L)) donde L es la profundidad de la capa. Simple pero crucial: sin esto, las capas profundas reciben gradientes ~100x más pequeños que las superficiales.

---

### 1.31 Stable Transformer
**Nombre**: Stable Transformer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Variante del transformer (Nguyen & Salazar, 2019) con Pre-LayerNorm (la normalización se aplica antes de la sub-capa, no después) y con la eliminación del "post-norm" estándar. También conocido como Pre-LN Transformer. Elimina el fenómeno de gradiente explosivo en los primeros pasos de entrenamiento.
**Definición llana**: El transformer original tiene un defecto de diseño que hace los primeros pasos de entrenamiento inestables. Pre-LN los mueve antes del cálculo y el entrenamiento se vuelve mucho más tranquilo.
**Contexto de uso**: Adoptado en GPT-3, PaLM, LLaMA y la mayoría de LLMs modernos. Paper: arXiv:1910.05895.
**Relaciones**:
- `es_subclase_de [1.1]`
- `mitiga_inestabilidad_de [8.6]`
- `usa [1.29]` (Keel conceptualmente)
- `contrario_de [1.32]` (reduce collapse)
**Lo que nadie te cuenta**: Pre-LN tiene un coste oculto: los modelos con Pre-LN no se benefician de la normalización como regularizador tanto como los Post-LN. Son más fáciles de entrenar pero potencialmente más susceptibles a overfitting en datasets pequeños.

---

### 1.32 Rank Collapse
**Nombre**: Rank Collapse
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Fenómeno por el cual, en redes neuronales profundas (especialmente transformers sin skip connections adecuadas), la matriz de representaciones de los tokens pierde rango efectivo, convergiendo todas las filas a un mismo vector. El rango efectivo se define como exp(H(p))/d, donde H(p) es la entropía de la distribución de valores singulares normalizados.
**Definición llana**: Todos los tokens acaban diciendo lo mismo: la red pierde su capacidad de distinguir entre distintos conceptos. Como si en una reunión, todas las personas empezaran a usar exactamente las mismas palabras.
**Contexto de uso**: Dong et al. (2021) "Attention is Not All You Need", Noci et al. (2022) "Signal Propagation in Transformers".
**Relaciones**:
- `es_subclase_de [8.7]` (entropy collapse)
- `se_mitiga_con [2.4]` (atención centrada)
- `se_mide_con [8.4]` (rango efectivo)
- `contrario_de [8.6]` (buena propagación de señal)
- `se_estudia_en [1.1]`
**Ejemplo en código**:
```python
import torch
def effective_rank(X):
    _, s, _ = torch.linalg.svd(X)
    p = s**2 / (s**2).sum()
    H = -(p * torch.log(p + 1e-12)).sum()
    return torch.exp(H) / X.shape[-1]
# Un valor cercano a 1/d indica rank collapse
```
**Lo que nadie te cuenta**: El rank collapse ocurre incluso con inicializaciones estándar. No es un fallo del entrenamiento, es una propiedad intrínseca de la atención no regularizada. La solución más elegante (restar 1/T a la diagonal) tiene una justificación espectral relacionada con la teoría de matrices aleatorias.

---

### 1.33 RoninTransformer
**Nombre**: RoninTransformer
**Dominio**: 1. ARQUITECTURAS DE MODELOS
**Definición técnica**: Arquitectura transformer experimental desarrollada por la Agencia RONIN que combina: Inhibitor Attention [2.5] para tokens "fáciles", atención centrada [2.4] para estabilización espectral, router adaptativo [4.13] para selección dinámica del mecanismo, y cuantización UniQL [3.14] nativa. Optimizado para inferencia edge con complejidad variable por token.
**Definición llana**: Un transformer que cambia su estrategia de atención según la dificultad del token: usa el método barato (Manhattan) para lo sencillo y el costoso (softmax completo) para lo difícil.
**Relaciones**:
- `implementa [2.5]`
- `usa [2.4]`
- `usa [4.13]`
- `se_comprime_con [3.14]`
- `optimizado_para [6.18]`
**Lo que nadie te cuenta**: El factor crítico del RoninTransformer no es el mecanismo de atención híbrido, sino el router [4.13]: si el router clasifica mal los tokens "difíciles" como "fáciles", la degradación de calidad es silenciosa y no aparece en benchmarks estándar.

---

## 2. MECANISMOS DE ATENCIÓN

### 2.1 Atención Softmax
**Nombre**: Atención Softmax (Scaled Dot-Product con Softmax)
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Mecanismo que calcula scores de relevancia entre queries y keys mediante producto escalar escalado, aplica softmax para normalizar (suma = 1) y pondera los values resultantes: Attn(Q,K,V) = softmax(QKᵀ/√d_k)·V. La normalización por √d_k previene gradientes que desaparecen en alta dimensión.
**Definición llana**: Cada palabra pregunta a todas las demás "¿cuánto me importas?" mediante multiplicación; softmax convierte las puntuaciones en probabilidades que suman 1, y se hace un promedio ponderado de todos los values.
**Contexto de uso**: Vaswani et al. (2017). La operación central de prácticamente todo LLM existente.
**Relaciones**:
- `es_núcleo_de [1.1]`
- `sufre [1.32]` (sin regularización)
- `se_estabiliza_con [2.4]`
- `se_aproxima_con [2.12]`, `se_aproxima_con [2.13]`
- `se_reemplaza_con [2.5]` (en RoninTransformer)
**Ejemplo en código**:
```python
import torch
import torch.nn.functional as F
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2,-1)) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return weights @ V
```
**Lo que nadie te cuenta**: El softmax tiene un problema de concentración de probabilidad: en contextos largos, tiende a colapsar en 1-2 tokens de alta puntuación, ignorando el resto. Este comportamiento se llama "attention sink" y es una forma de [1.32] rank collapse en el dominio de atención.

---

### 2.2 Scaled Dot-Product Attention
**Nombre**: Scaled Dot-Product Attention
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Nombre formal de la operación de atención softmax con el escalado explícito por √d_k. El escalado es crucial: sin él, el producto escalar de vectores de alta dimensión tiene varianza ~d_k, lo que empuja el softmax a regiones de gradiente casi cero.
**Definición llana**: La versión "con manual de instrucciones" de la atención: incluye el divisor matemático que hace que funcione correctamente sin importar el tamaño del modelo.
**Relaciones**:
- `es_sinónimo_de [2.1]`
- `es_base_de [2.3]`
- `complejidad [8.10]` (O(n²·d))
**Lo que nadie te cuenta**: El √d_k es una heurística, no un resultado teórico. Modelos recientes (PaLM, Gemini) han experimentado con otros escalados (por ejemplo, basados en la norma empírica de los vectores QK) con resultados mixtos.

---

### 2.3 Multi-Head Attention
**Nombre**: Multi-Head Attention (MHA)
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Extensión de [2.1] que aplica H versiones independientes de la atención en paralelo, cada una con proyecciones W_Q^h, W_K^h, W_V^h propias de dimensión d_k = d_model/H. Las salidas se concatenan y proyectan. Permite capturar H tipos de dependencias semánticas simultáneamente.
**Definición llana**: En vez de una sola atención, hay H "lectores especializados" que leen el mismo texto pero prestan atención a cosas distintas (sintaxis, semántica, correferencia...). Sus lecturas se combinan al final.
**Contexto de uso**: Vaswani et al. (2017). Variantes: MQA (Multi-Query Attention), GQA (Grouped Query Attention) para reducir KVCache.
**Relaciones**:
- `extiende [2.1]`
- `usa [6.3]` (KVCache para cada cabeza)
- `se_optimiza_con [2.8]` (FlashAttention)
- `se_reduce_a [2.14]` (cross-attention en decoder)
**Lo que nadie te cuenta**: No todas las cabezas aprenden cosas útiles. Investigaciones de Michel et al. (2019) muestran que en BERT, ~30% de las cabezas pueden eliminarse en inferencia sin pérdida apreciable. Esto es la base de [3.12] (pruning estructurado de cabezas).

---

### 2.4 Atención Centrada (A - 1/T)
**Nombre**: Atención Centrada / Centered Attention
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Modificación de la atención softmax que resta 1/T a cada fila de la matriz de atención antes de aplicarla a V: Attn_c = (A - 1/T)·V, donde T es la longitud de secuencia. Esta operación proyecta la atención al subespacio ortogonal al vector de unos, rompiendo la simetría de permutación y previniendo el rank collapse. Propuesta por Noci et al. (2022).
**Definición llana**: Restar el "promedio" de atención a cada fila. Hace que la atención no sea atraída por el vector promedio de todos los tokens, sino por las diferencias semánticas reales.
**Contexto de uso**: Noci et al., "Signal Propagation in Transformers" (ICML 2022).
**Relaciones**:
- `mitiga [1.32]` (rank collapse)
- `es_variante_de [2.1]`
- `usa [8.1]` (entropía para medir efecto)
- `se_implementa_en [1.33]` (RoninTransformer)
**Ejemplo en código**:
```python
def centered_attention(Q, K, V):
    d_k = Q.shape[-1]
    T = Q.shape[-2]
    scores = torch.matmul(Q, K.T) / (d_k ** 0.5)
    A = F.softmax(scores, dim=-1)
    A_centered = A - 1.0/T          # La magia está aquí
    return A_centered @ V
```
**Lo que nadie te cuenta**: La corrección 1/T parece trivial pero tiene una justificación espectral profunda. Matemáticamente, es equivalente a requerir que la matriz de atención tenga traza cero, lo cual garantiza que el espacio nulo del operador de atención no degenere. La mayoría de los papers que citan Dong et al. ignoran esta solución y proponen alternativas más complejas.

---

### 2.5 Inhibitor Attention
**Nombre**: Inhibitor Attention
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Mecanismo que sustituye el producto escalar QKᵀ por la distancia Manhattan negativa, con activación ReLU en lugar de softmax: scores = -‖Q - K‖₁ / d_h; A = ReLU(scores). Elimina multiplicaciones matriciales y produce atención dispersa naturalmente. Propuesto en ICLR 2026 (InhibiDistilbert).
**Definición llana**: En vez de multiplicar vectores para calcular similitud, los resta (distancia Manhattan) y activa solo los pares más cercanos. Mucho más rápido en hardware sin unidades de multiplicación eficientes.
**Contexto de uso**: Paper "InhibiDistilbert" (ICLR 2026). Implementado en RoninTransformer v1 [1.33].
**Relaciones**:
- `es_alternativa_de [2.1]`
- `usa [8.9]` (distancia Manhattan)
- `se_combina_con [4.13]` (router adaptativo)
- `mejora [6.18]` (eficiencia edge)
- `se_implementa_en [1.33]`
**Ejemplo en código**:
```python
def inhibitor_attention(Q, K, V):
    # Q, K: (B, H, T, d_h)
    scores = -torch.cdist(Q, K, p=1) / Q.shape[-1]
    A = torch.relu(scores)
    return A @ V
```
**Lo que nadie te cuenta**: La atención ReLU no suma 1 (no es una distribución de probabilidad). Esto significa que un token puede recibir atención "cero" de todos los demás si ninguno es suficientemente cercano. En la práctica, esto se maneja con una capa residual fuerte, pero hay que ser cuidadoso con la normalización posterior.

---

### 2.6 Consensus Attention
**Nombre**: Consensus Attention
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Variante de multi-head attention en la que las cabezas alcanzan un consenso explícito antes de ponderar los values. Se introduce una penalización que maximiza la divergencia entre cabezas (para especialización) seguida de una agregación que fuerza acuerdo en las predicciones finales. Inspirado en modelos de votación en aprendizaje por conjuntos.
**Definición llana**: Las cabezas de atención primero se especializan en perspectivas distintas, luego votan y buscan acuerdo antes de dar la respuesta final.
**Relaciones**:
- `extiende [2.3]`
- `usa [4.10]` (multi-agent debate como inspiración)
- `mejora [8.3]` (información mutua entre cabezas)
**Lo que nadie te cuenta**: La penalización de diversidad en Consensus Attention puede aumentar el tiempo de entrenamiento en ~20-30% pero produce representaciones más robustas a distributional shift.

---

### 2.7 Atención Lineal
**Nombre**: Atención Lineal
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Familia de aproximaciones a la atención softmax que linealizan el kernel de atención: Attn(Q,K,V) ≈ φ(Q)(φ(K)ᵀV)/φ(Q)(φ(K)ᵀ1), donde φ es un mapa de características que aproxima la función exponencial. Permite reordenar la multiplicación de matrices para obtener complejidad O(n·d²) en vez de O(n²·d).
**Definición llana**: Trucos matemáticos para calcular la atención sin tener que comparar cada token con todos los demás, cambiando el orden de las operaciones.
**Contexto de uso**: FAVOR+ (Performer [1.14]), Random Features (arXiv:2009.14794).
**Relaciones**:
- `es_aproximación_de [2.1]`
- `implementa [1.14]`
- `es_relacionada_con [1.2]` (RWKV en modo paralelo)
- `unifica_con [1.3]` (Mamba-2 demuestra dualidad)
**Lo que nadie te cuenta**: La mayor parte de las atenciones "lineales" son en realidad O(n·d²), y como d >> 1, si d es grande (d=256 o más), O(n·d²) puede ser peor que O(n²·d) para n moderado. La elección importa.

---

### 2.8 FlashAttention-2
**Nombre**: FlashAttention-2
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Implementación IO-aware de la atención exacta (sin aproximación) que reorganiza el cálculo en tiles que caben en SRAM, evitando escrituras/lecturas intermedias de la matriz de atención en HBM. Versión 2 introduce paralelismo mejorado y mejor particionamiento de trabajo entre warps GPU. Speedup ~2-4x sobre la implementación estándar.
**Definición llana**: No reduce el cálculo matemático, sino que lo hace más rápido reorganizando cómo los datos se mueven en el chip: en vez de escribir resultados intermedios en memoria lenta, los mantiene en la memoria rápida del procesador.
**Contexto de uso**: arXiv:2307.08691 (Dao, 2023). Estándar de facto en todos los frameworks.
**Relaciones**:
- `implementa [2.1]` (sin aproximación)
- `optimiza [6.2]` (latencia y throughput)
- `usa [6.3]` (KVCache eficientemente)
- `se_superpone_con [6.21]` (FlashDecoding)
**Lo que nadie te cuenta**: FlashAttention-2 solo funciona bien en GPUs con suficiente SRAM por SM (>96KB). En GPUs antiguas (V100) o en CPUs, la ventaja es menor o nula. Además, la versión 2 es más difícil de extender a variantes de atención no estándar que la v1.

---

### 2.9 Atención con Ventana Deslizante
**Nombre**: Atención con Ventana Deslizante (Sliding Window Attention)
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Cada token solo atiende a los w tokens precedentes (y/o siguientes en contextos bidireccionales). Complejidad O(n·w). En modelos de generación causal (GPT-like), la ventana típicamente cubre los últimos w tokens, con w de 512 a 4096.
**Definición llana**: Cada palabra solo lee las w palabras más recientes, ignorando las anteriores. Como leer con una ventana que se desliza y cubre solo un fragmento del texto a la vez.
**Relaciones**:
- `es_subclase_de [2.1]`
- `usa [1.16]` (Longformer la implementa a gran escala)
- `complementa [2.14]` (cross-attention para contexto global)
- `se_gestiona_con [6.3]` (KVCache de ventana)
**Lo que nadie te cuenta**: La ventana deslizante tiene un problema de "horizonte": un hecho establecido hace 5000 tokens simplemente desaparece. Los sistemas modernos lo compensan con RAG [4.18] o con unos pocos tokens de "atención global" (como en BigBird [1.15]).

---

### 2.10 Block-Sparse Attention
**Nombre**: Block-Sparse Attention
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Atención donde la matriz de scores se divide en bloques y solo se calculan los bloques no cero según un patrón predefinido (local, strided, global). La dispersión por bloques es más eficiente en hardware que la dispersión elemento a elemento porque las GPUs operan mejor con bloques contiguos.
**Definición llana**: En vez de calcular todas las relaciones entre palabras, se calcula solo un subconjunto definido en bloques, que la GPU puede procesar eficientemente.
**Relaciones**:
- `es_subclase_de [2.1]`
- `implementa [1.15]`, `implementa [1.16]`
- `mejor_que [2.13]` (en hardware real)
**Lo que nadie te cuenta**: Los patrones de dispersión predefinidos (fixed patterns) pueden no coincidir con los patrones de atención reales del modelo. Patrones adaptativos (como en el Routing Transformer [1.17]) son más expresivos pero más difíciles de implementar eficientemente.

---

### 2.11 Low-Rank Attention
**Nombre**: Low-Rank Attention
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Aproxima la matriz de atención A ∈ ℝ^{n×n} con una descomposición de bajo rango A ≈ UV^T, donde U ∈ ℝ^{n×k}, V ∈ ℝ^{n×k}, k << n. La justificación empírica es que las matrices de atención reales tienen bajo rango intrínseco (~10-50 componentes significativos para n=512).
**Definición llana**: La atención completa es como una tabla de n×n relaciones. En la práctica, esa tabla tiene muchos patrones repetidos, así que se puede comprimir a k columnas sin perder mucho.
**Contexto de uso**: Linformer [1.13] es su implementación canónica.
**Relaciones**:
- `es_base_de [1.13]`
- `usa [8.5]` (SVD para justificación teórica)
- `usa [8.4]` (rango efectivo como métrica)
**Lo que nadie te cuenta**: El rango bajo de la atención no es uniforme entre capas: las capas inferiores tienden a tener matrices de atención de rango más alto (patrones más diversos), mientras las superiores son más de bajo rango (patrones semánticos más concentrados). Comprimir uniformemente es subóptimo.

---

### 2.12 Kernelized Attention
**Nombre**: Kernelized Attention
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Reformulación de la atención softmax como un kernel: sim(q,k) = exp(qᵀk/√d) ≈ φ(q)ᵀφ(k), donde φ es un mapa de características que aproxima el kernel exponencial. La reordenación matricial (φ(K)ᵀV calculado primero) da complejidad lineal.
**Definición llana**: Reescribe las multiplicaciones de atención como una función de similitud que puede calcularse en un orden más eficiente, ahorrando tiempo.
**Relaciones**:
- `generaliza [2.7]`
- `implementa [1.14]` (FAVOR+ es una instancia)
- `usa [2.13]`
**Lo que nadie te cuenta**: La elección del kernel importa más que el método de aproximación. Kernels mal elegidos (ej: polinomial de grado bajo) pueden introducir correlaciones espurias que degradan la calidad en tareas de razonamiento.

---

### 2.13 Random Feature Attention
**Nombre**: Random Feature Attention (RFA)
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Aproxima la función de kernel exp(qᵀk) usando el teorema de Bochner: exp(qᵀk) ≈ E[φ(q)ᵀφ(k)] donde φ son proyecciones en direcciones aleatorias seguidas de cos/sin. FAVOR+ usa características ortogonales positivas para reducir varianza.
**Definición llana**: Usa proyecciones aleatorias para aproximar la similitud entre vectores, evitando el cálculo exacto. Es como estimar el promedio de una encuesta con una muestra aleatoria en vez de preguntar a todos.
**Relaciones**:
- `implementa [2.12]`
- `es_base_de [1.14]` (Performer)
- `usa [8.1]` (entropía para analizar varianza)
**Lo que nadie te cuenta**: La varianza de la aproximación aumenta con la entropía de la distribución de atención. Los "attention spikes" (un token muy dominante) son los peor aproximados. Para mitigarlo, FAVOR+ requiere muchas más características aleatorias (aumentando el coste).

---

### 2.14 Cross-Attention
**Nombre**: Cross-Attention
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Mecanismo de atención donde las Queries provienen de una secuencia (ej: el decoder) y las Keys/Values de otra secuencia diferente (ej: la salida del encoder). Permite a una secuencia "leer" selectivamente otra secuencia. Fórmula idéntica a [2.1] pero con Q y KV de fuentes distintas.
**Definición llana**: Es como hacer preguntas (Q) a una fuente de información diferente (KV). El decoder "pregunta" al encoder qué información es relevante para la siguiente palabra.
**Relaciones**:
- `es_variante_de [2.1]`
- `se_usa_en [1.7]` (Perceiver)
- `se_usa_en [5.4]` (BLIP-2, entre imagen y texto)
- `contrario_de [2.1]` (self-attention: mismo origen para Q, K, V)
**Lo que nadie te cuenta**: En arquitecturas encoder-decoder, el cross-attention es el cuello de botella de latencia en inferencia porque no puede precalcularse (depende del estado actual del decoder). Es la razón por la que modelos decoder-only (GPT) son más fáciles de acelerar con KVCache.

---

### 2.15 Causal Attention / Masked Attention
**Nombre**: Causal Attention / Masked Attention
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Atención donde se aplica una máscara triangular inferior a la matriz de scores antes del softmax, forzando que cada token solo pueda atender a tokens previos (y a sí mismo): mask[i,j] = -∞ si j > i. Garantiza autoregresividad en el entrenamiento.
**Definición llana**: Cada palabra solo puede "ver" las palabras que vinieron antes, no las futuras. Simula el proceso de generación donde la siguiente palabra no existe todavía.
**Relaciones**:
- `es_variante_de [2.1]`
- `es_estándar_en [1.1]` (GPT-like models)
- `contrario_de [2.14]` (en encoder bidireccional, sin máscara)
- `habilita [6.3]` (KVCache, porque los tokens pasados no cambian)
**Lo que nadie te cuenta**: La máscara causal es trivial para el entrenamiento (una operación de máscara), pero en inferencia tiene implicaciones profundas: permite el KVCache y el speculative decoding [6.14]. Es una de las razones por las que los modelos autorregresivos dominan en producción.

---

### 2.16 Rotary Position Embedding (RoPE)
**Nombre**: RoPE — Rotary Position Embedding
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Codifica la posición absoluta como una rotación en el espacio de representación: para posición m, el vector q se convierte en R_θ^m · q, donde R_θ es una matriz de rotación con diferentes frecuencias θ por par de dimensiones. El producto qᵀk después de RoPE depende solo de la diferencia de posiciones (m-n), logrando posición relativa implícita.
**Definición llana**: En vez de añadir un número de posición al vector, lo rota en un ángulo proporcional a su posición. Dos palabras que están a la misma distancia siempre tienen el mismo ángulo de rotación relativa, sin importar dónde estén en la frase.
**Contexto de uso**: Su et al. (2021), arXiv:2104.09864. Estándar en LLaMA, Mistral, Qwen, Falcon.
**Relaciones**:
- `es_alternativa_a [2.18]`
- `permite [1.10]` (RetNet usa decaimiento basado en RoPE)
- `se_extiende_con [2.17]` (para long context)
- `implementa_posición_relativa [2.1]`
**Lo que nadie te cuenta**: RoPE tiene un comportamiento interesante con contextos más largos que los de entrenamiento: las frecuencias de rotación "se saturan" y las posiciones distantes se vuelven indistinguibles. Técnicas como YaRN (arXiv:2309.00071) escalan RoPE para contextos 4x-128x mayores con fine-tuning mínimo.

---

### 2.17 ALiBi (Attention with Linear Biases)
**Nombre**: ALiBi — Attention with Linear Biases
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Reemplaza los embeddings de posición con un bias lineal que se resta de los scores de atención: score(i,j) = qᵢᵀkⱼ - m·|i-j|, donde m es una pendiente diferente por cabeza. Tokens lejanos reciben penalización proporcional a su distancia. Permite extrapolación a contextos más largos que los de entrenamiento.
**Definición llana**: Penaliza las palabras lejanas con un descuento lineal: cada posición de distancia cuesta un "punto de atención". Sin suma de positional embeddings, más fácil de extrapolar.
**Contexto de uso**: arXiv:2108.12409 (Press et al., 2021). Usado en BLOOM, MPT, algunos Falcon.
**Relaciones**:
- `es_alternativa_a [2.16]`
- `es_alternativa_a [2.18]`
- `permite [1.10]` (RetNet usa decaimiento similar)
- `mejor_extrapolación_que [2.16]` (para long context sin fine-tuning)
**Lo que nadie te cuenta**: ALiBi tiene mejor extrapolación de longitud que RoPE sin fine-tuning, pero peor rendimiento en contextos dentro del rango de entrenamiento. El trade-off importa: para aplicaciones donde el contexto siempre es conocido, RoPE gana; para contextos variables, ALiBi es más robusto.

---

### 2.18 Relative Position Bias
**Nombre**: Relative Position Bias
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Método de codificación de posición que añade un sesgo aprendido basado en la distancia relativa entre tokens, en lugar de posiciones absolutas. El bias b(i-j) se añade directamente a los logits de atención. Usado en T5 (Shaw et al.) con bucketing de distancias.
**Definición llana**: En vez de decir "estoy en la posición 42", dice "estoy a 5 posiciones del token que me habla". Aprende cuánto importa cada distancia relativa.
**Relaciones**:
- `contrario_de [2.16]` (absoluta vs relativa)
- `implementa [1.1]` (T5 lo usa)
- `similar_a [2.17]` (ALiBi es una versión lineal fija)
**Lo que nadie te cuenta**: El bucketing de distancias en T5 agrupa todas las distancias >128 en el mismo bucket, lo que hace al modelo "ciego" a la estructura de documentos largos. Es una limitación de diseño que T5 no supera bien en tareas de long-form QA.

---

### 2.19 Gated Attention
**Nombre**: Gated Attention
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Variante que multiplica la salida de la atención por una función de gating σ(Wg·x): Attn_gated = σ(Wg·x) ⊙ Attn(Q,K,V). El gate, típicamente una sigmoide o SiLU, aprende a suprimir o amplificar la contribución de la atención según el contexto. Usado en Gated Linear Attention (GLA) y RetNet.
**Definición llana**: La atención tiene un "interruptor de volumen" aprendido que decide cuánta atención vale la pena en cada momento. A veces la información local (del gate) supera a la global (de la atención).
**Relaciones**:
- `es_variante_de [2.1]`
- `usa [1.10]` (RetNet usa gates sobre retención)
- `similar_a [1.23]` (gMLP usa gates similares)
**Lo que nadie te cuenta**: Los gates de atención son el mecanismo que permite a los SSMs híbridos (Jamba, Zamba) decidir cuándo usar la memoria recurrente y cuándo "mirar hacia atrás" con atención completa. Sin gates, la mezcla SSM+Attention sería estática.

---

### 2.20 Graph Attention (GAT)
**Nombre**: Graph Attention Network (GAT)
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Extensión de la atención a grafos (Veličković et al., 2018). Cada nodo agrega información de sus vecinos con pesos de atención que dependen de las características del nodo y sus vecinos: αᵢⱼ = softmax_j(LeakyReLU(aᵀ[Whᵢ‖Whⱼ])). A diferencia del transformer, la estructura del grafo define qué pares (i,j) se calculan.
**Definición llana**: Como la atención del transformer, pero aplicada a grafos: cada nodo (persona, átomo, ciudad) solo mira a sus vecinos directos en el grafo, con pesos que aprende según la importancia de cada vecino.
**Contexto de uso**: arXiv:1710.10903. Usado en química molecular, redes sociales, sistemas de recomendación.
**Relaciones**:
- `es_extensión_de [2.1]`
- `se_aplica_en [5.0]` (visión como grafo de patches)
- `diferente_de [2.1]` (estructura de grafo es el contexto, no secuencia)
**Lo que nadie te cuenta**: GAT v1 tiene una limitación sutil: si la estructura del grafo no cambia durante el entrenamiento, el modelo puede "memorizar" el grafo en vez de aprender a razonar sobre la topología. GAT v2 (Brody et al., 2022) corrige esto con una atención dinámica.

---

### 2.21 Fourier Attention
**Nombre**: Fourier Attention / FNet Layer
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Reemplaza la self-attention con la transformada de Fourier bidimensional (FFT), que mezcla información entre posiciones en O(n log n) sin parámetros aprendibles. No es una "atención" en sentido estricto (no hay Q, K, V), pero funciona como operador de mixing de tokens.
**Definición llana**: En vez de aprender qué tokens se relacionan, aplica una transformación matemática universal (Fourier) que mezcla todos los tokens de forma fija. Como aplicar un filtro pasa-todos antes del razonamiento.
**Relaciones**:
- `es_base_de [1.21]` (FNet)
- `usa [1.11]` (Hyena lo extiende con convoluciones)
- `es_no_aprendible [2.1]` (contraste clave)
**Lo que nadie te cuenta**: La FFT mezcla tokens de forma que no respeta la causalidad (mezcla pasado y futuro). FNet solo funciona en modelos bidireccionales (encoder). Para decoder causales, la FFT debe modificarse, perdiendo la mayor parte de su ventaja de velocidad.

---

### 2.22 PagedAttention
**Nombre**: PagedAttention
**Dominio**: 2. MECANISMOS DE ATENCIÓN
**Definición técnica**: Sistema de gestión de memoria para el KVCache [6.3] inspirado en la paginación de memoria virtual de los sistemas operativos. En lugar de asignar bloques contiguos para cada secuencia, el KVCache se divide en páginas (bloques físicos) que se asignan dinámicamente. Permite compartir páginas entre secuencias con prefijos comunes (prefix sharing) y eliminar fragmentación.
**Definición llana**: En vez de reservar un bloque de memoria grande y fijo para cada conversación, reserva pequeñas páginas según se necesitan. Como un libro de notas con páginas que se asignan dinámicamente, no un cuaderno con espacio fijo por persona.
**Contexto de uso**: Base de vLLM [6.5]. arXiv:2309.06180 (Kwon et al., 2023).
**Relaciones**:
- `gestiona [6.3]`
- `implementada_en [6.5]` (vLLM)
- `optimiza [6.2]` (throughput)
- `análogo_a [6.12]` (Wasm virtual memory)
**Lo que nadie te cuenta**: PagedAttention tiene overhead de gestión de páginas que puede ser significativo para lotes pequeños. Para inferencia con batch=1 (caso de chatbot interactivo), la ventaja sobre gestión contigua es marginal; la ventaja real aparece con cientos de usuarios concurrentes.

---

## 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN

### 3.1 RLHF
**Nombre**: RLHF — Reinforcement Learning from Human Feedback
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Pipeline de alineación que: (1) entrena un Reward Model (RM) a partir de comparaciones humanas entre respuestas, (2) usa PPO [3.3] para optimizar la política LLM maximizando el reward, con una penalización KL respecto al modelo base para evitar colapso. Formalizado por Ouyang et al. (InstructGPT, 2022).
**Definición llana**: Enseña al modelo a ser útil mediante un sistema de puntuaciones humanas: humanos votan qué respuestas son mejores, se entrena un "árbitro" con esas votaciones, y luego el modelo mejora sus respuestas para agradar al árbitro.
**Contexto de uso**: arXiv:2203.02155. Base de GPT-4, Claude, Gemini.
**Relaciones**:
- `usa [3.3]` (PPO como algoritmo RL)
- `se_sustituye_con [3.2]` (DPO como alternativa más simple)
- `usa [8.2]` (penalización KL)
- `afecta_a [7.9]` (debiasing a través de preferencias)
**Lo que nadie te cuenta**: El RLHF no alinea el modelo con "valores humanos universales", sino con las preferencias del subconjunto de anotadores contratados. Si esos anotadores son culturalmente homogéneos, el modelo hereda sus sesgos sistemáticamente. Esto es especialmente crítico en [7.1] (dimensiones psicopatológicas).

---

### 3.2 DPO (Direct Preference Optimization)
**Nombre**: DPO — Direct Preference Optimization
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Alternativa a RLHF que reformula la optimización de preferencias como clasificación binaria directa, sin necesidad de entrenar un reward model separado. Usa la relación matemática entre el reward óptimo y la política óptima: r*(x,y) = β·log[π*(y|x)/π_ref(y|x)] + β·log Z(x). Resultado: un objetivo de clasificación sobre pares preferidos/rechazados.
**Definición llana**: En vez del complicado proceso RLHF de tres fases, DPO aprende directamente de pares de respuestas (buena/mala) en un solo paso de entrenamiento, como un clasificador binario.
**Contexto de uso**: arXiv:2305.18290 (Rafailov et al., Stanford 2023).
**Relaciones**:
- `es_alternativa_de [3.1]`
- `no_usa [3.3]` (no necesita PPO)
- `usa [8.2]` (divergencia KL implícita)
- `se_combina_con [3.4]` (LoRA + DPO para fine-tuning eficiente)
**Lo que nadie te cuenta**: DPO puede sobreajustarse a los pares de entrenamiento más rápido que RLHF porque no tiene la regularización implícita del ciclo RL. Variantes como IPO y SimPO añaden regularización adicional para mitigar esto.

---

### 3.3 PPO (Proximal Policy Optimization)
**Nombre**: PPO — Proximal Policy Optimization
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Algoritmo de RL on-policy (Schulman et al., OpenAI 2017) que limita el tamaño del paso de actualización mediante un clip en el ratio de probabilidades: L = min(r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t). Balancea la mejora de política con la estabilidad del entrenamiento. En RLHF, los tokens son acciones y la oración generada es el "episodio".
**Definición llana**: Un método de aprendizaje por refuerzo que mejora la política "con pasos pequeños": no cambia demasiado de una actualización a la otra para no desestabilizar el aprendizaje.
**Contexto de uso**: arXiv:1707.06347. Componente central de [3.1].
**Relaciones**:
- `se_usa_en [3.1]`
- `se_reemplaza_con [3.2]` (en DPO)
- `usa [8.14]` (función de valor como baseline)
**Lo que nadie te cuenta**: PPO en LLMs es computacionalmente costoso: requiere mantener en memoria simultáneamente el modelo de política, el modelo de referencia (para KL), el critic (value function) y el reward model. Cuatro modelos en paralelo implican 4x la VRAM del modelo base.

---

### 3.4 LoRA (Low-Rank Adaptation)
**Nombre**: LoRA — Low-Rank Adaptation
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Técnica de fine-tuning eficiente (Hu et al., Microsoft 2021) que congela los pesos originales W y añade una actualización de bajo rango: W' = W + BA, donde B ∈ ℝ^{d×r}, A ∈ ℝ^{r×d}, con r << d. Solo B y A se entrenan (~0.1-1% de los parámetros). Al finalizar, se fusionan: W_final = W + BA, sin overhead en inferencia.
**Definición llana**: En vez de reentrenar el modelo entero, se añaden dos matrices pequeñas que capturan el "delta de conocimiento" necesario. Es como añadir notas al margen de un libro en vez de reescribirlo.
**Contexto de uso**: arXiv:2106.09685. Estándar de facto para fine-tuning de LLMs.
**Relaciones**:
- `usa [8.4]` (rango efectivo para justificación)
- `se_cuantiza_con [3.5]` (QLoRA)
- `se_adapta_en [3.6]` (AdaLoRA)
- `se_despliega_en [3.33]` (LoRA Edge)
- `se_combina_con [3.2]` (DPO+LoRA)
**Ejemplo en código**:
```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"])
model = get_peft_model(base_model, config)
```
**Lo que nadie te cuenta**: El rango r es crítico y no hay regla universal: r=4 puede ser suficiente para adaptar el estilo, pero r=64 o más puede necesitarse para inyectar conocimiento factual nuevo. Además, qué módulos incluir en target_modules cambia radicalmente el resultado. Los papers suelen reportar el mejor r pero no cómo encontrarlo.

---

### 3.5 QLoRA
**Nombre**: QLoRA — Quantized LoRA
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Combina LoRA [3.4] con cuantización NF4 (Normal Float 4-bit) del modelo base y double quantization del factor de escala. Permite fine-tuning de modelos 65B en GPUs de 48GB con pérdida de calidad mínima. Las matrices LoRA se mantienen en BF16 mientras el modelo base está en NF4.
**Definición llana**: Como LoRA pero el modelo base se guarda en formato muy comprimido (4 bits). Permite ajustar modelos enormes en tarjetas gráficas accesibles.
**Contexto de uso**: arXiv:2305.14314 (Dettmers et al., 2023). Habilitó el fine-tuning comunitario a gran escala.
**Relaciones**:
- `extiende [3.4]`
- `usa [3.19]` (cuantización INT4/NF4)
- `requiere [6.1]` (inferencia descomprimiendo NF4 en tiempo real)
**Lo que nadie te cuenta**: QLoRA tiene un overhead de velocidad en entrenamiento (~30% más lento que LoRA FP16) por la descompresión NF4 continua. Y el NF4 no es universal: en distribuciones de pesos muy alejadas de la normal, la pérdida de calidad puede ser apreciable.

---

### 3.6 AdaLoRA
**Nombre**: AdaLoRA — Adaptive Low-Rank Adaptation
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Versión adaptativa de LoRA que distribuye el presupuesto de rango desigualmente entre las distintas matrices según su importancia. Modela la actualización como W' = W + P·Λ·Qᵀ (descomposición SVD) y poda dinámicamente los valores singulares pequeños de Λ durante el entrenamiento.
**Definición llana**: LoRA inteligente: en vez de usar el mismo rango para todas las capas, dedica más capacidad a las capas más importantes y menos a las que casi no cambian.
**Contexto de uso**: arXiv:2303.10512 (Zhang et al., 2023).
**Relaciones**:
- `extiende [3.4]`
- `usa [8.5]` (SVD para asignar rango)
- `contrasta_con [3.4]` (rango fijo vs adaptativo)
**Lo que nadie te cuenta**: AdaLoRA añade overhead computacional significativo por el seguimiento de valores singulares durante el entrenamiento. Para modelos pequeños (<7B), el beneficio respecto a LoRA estándar con rango bien elegido es marginal.

---

### 3.7 DoRA (Weight-Decomposed Low-Rank)
**Nombre**: DoRA — Weight-Decomposed Low-Rank Adaptation
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Descompone cada peso W = m·(V/‖V‖) en magnitud m (escalar) y dirección V (vector normalizado), y aplica LoRA solo a la componente direccional. Aprende separadamente magnitud y dirección, mejorando la capacidad de representación respecto a LoRA con el mismo número de parámetros.
**Definición llana**: Divide cada peso en "cuánto pesa" y "hacia dónde apunta", y solo ajusta la dirección con LoRA. Más flexible que el LoRA estándar.
**Contexto de uso**: arXiv:2402.09353 (Liu et al., 2024).
**Relaciones**:
- `extiende [3.4]`
- `mejora [3.4]` (en benchmarks de razonamiento)
- `usa [8.5]` (normalización de vectores)
**Lo que nadie te cuenta**: La descomposición magnitud-dirección de DoRA implica computar normas L2 en cada paso, lo que puede ser un bottleneck en matrices grandes. En la práctica, el beneficio real sobre QLoRA bien configurado es menor de lo que sugieren los papers.

---

### 3.8 IA³
**Nombre**: IA³ — Infused Adapter by Inhibiting and Amplifying Inner Activations
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Técnica de PEFT que introduce vectores de escala aprendibles (l_k, l_v, l_ff) que reescalan las activaciones internas: K' = l_k ⊙ K; V' = l_v ⊙ V; FF' = l_ff ⊙ FF. Solo se entrenan tres vectores por capa (~0.01% de parámetros), permitiendo fine-tuning extremadamente ligero.
**Definición llana**: Añade un "ecualizador" por capa (un vector de volúmenes) que amplifica o atenúa ciertas señales internas. Menos parámetros que LoRA pero más eficiente para algunas tareas.
**Contexto de uso**: arXiv:2205.05638 (Liu et al., 2022).
**Relaciones**:
- `es_alternativa_de [3.4]`
- `menor_que [3.4]` (en número de parámetros)
- `similar_a [3.11]` (adapter)
**Lo que nadie te cuenta**: IA³ funciona bien para tareas de clasificación y adaptación de estilo, pero falla en inyección de conocimiento factual extenso porque sus vectores de escala son globales (no específicos por token). Para eso, LoRA sigue siendo necesario.

---

### 3.9 Prefix Tuning
**Nombre**: Prefix Tuning
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Añade una secuencia de tokens "prefix" aprendibles P_θ al inicio de cada capa (no solo al input), que modifica el comportamiento del modelo en todas las capas simultáneamente. Los prefijos se representan en el espacio KV y el modelo los trata como contexto adicional. Solo P_θ se entrena.
**Definición llana**: Coloca instrucciones aprendidas invisibles al inicio de cada capa del modelo. El modelo cree que siempre ha tenido esa información de contexto en todas sus capas.
**Contexto de uso**: arXiv:2101.00190 (Li & Liang, Stanford 2021).
**Relaciones**:
- `es_alternativa_de [3.4]`
- `modifica [6.3]` (KVCache incluye el prefix)
- `contrasta_con [3.10]` (Prompt Tuning solo actúa en input)
**Lo que nadie te cuenta**: Los prefijos de Prefix Tuning aumentan el KVCache en n_prefix tokens para CADA REQUEST, lo que puede representar 5-10% de overhead de memoria en producción. Para servicios con millones de peticiones, esto es significativo.

---

### 3.10 Prompt Tuning
**Nombre**: Prompt Tuning
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Variante simplificada de Prefix Tuning que solo añade tokens aprendibles al input (no a las capas internas). Lester et al. (2021) muestran que a escala suficiente (>10B parámetros), Prompt Tuning iguala al fine-tuning completo. Los "soft prompts" son vectores continuos en el espacio de embeddings, no tokens discretos.
**Definición llana**: Aprende un "prefijo mágico" en el espacio de embeddings que, al añadirse al input, hace que el modelo se comporte como se desea, sin tocar los pesos internos.
**Contexto de uso**: arXiv:2104.08691 (Lester et al., Google 2021).
**Relaciones**:
- `es_subclase_de [3.9]`
- `solo_modifica [4.21]` (input embeddings)
- `funciona_a_escala_de [1.5]` (modelos grandes)
**Lo que nadie te cuenta**: A escala pequeña (<1B), Prompt Tuning cae significativamente por debajo del fine-tuning. El umbral exacto donde se vuelve competitivo es incierto y varía por tarea.

---

### 3.11 Adapter
**Nombre**: Adapter
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Módulo bottleneck (down-project → no-linealidad → up-project) insertado dentro de cada capa transformer, después de las sub-capas de atención y FFN. Solo los adapters se entrenan; el modelo original se congela. Houlsby et al. (2019) introdujeron el diseño canónico.
**Definición llana**: Añade pequeñas redes neuronales intercaladas en el modelo existente que capturan el conocimiento específico de la tarea, sin modificar el modelo base.
**Contexto de uso**: arXiv:1902.00751.
**Relaciones**:
- `es_precursor_de [3.4]` (LoRA es conceptualmente más eficiente)
- `tiene_overhead_en [6.1]` (a diferencia de LoRA fusionado)
- `similar_a [3.8]` (IA³)
**Lo que nadie te cuenta**: Los adapters tienen overhead de latencia en inferencia (a diferencia de LoRA que se fusiona con los pesos). Para producción con baja latencia, LoRA fusionado [3.4] es siempre preferible.

---

### 3.12 Pruning Estructurado
**Nombre**: Pruning Estructurado
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Eliminación de componentes estructurales completos del modelo: cabezas de atención enteras, capas FFN, capas completas, o bloques transformer. Produce modelos con una arquitectura diferente (más pequeña) que puede acelerar la inferencia sin hardware especial, a diferencia del pruning no estructurado.
**Definición llana**: Elimina partes completas del modelo (como quitar capas enteras de un edificio), obteniendo un modelo más pequeño que cualquier CPU puede ejecutar más rápido.
**Relaciones**:
- `contrasta_con [3.13]`
- `se_combina_con [3.22]` (destilación para recuperar calidad)
- `afecta_a [2.3]` (eliminar cabezas de atención)
**Lo que nadie te cuenta**: El pruning estructurado de capas completas generalmente preserva mejor la calidad que el de cabezas individuales, contraintuitivamente. Esto se debe a que las representaciones entre capas tienen redundancia, pero dentro de una capa las cabezas están parcialmente acopladas.

---

### 3.13 Pruning No Estructurado
**Nombre**: Pruning No Estructurado
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Elimina pesos individuales (los de menor magnitud, o los con menor score de importancia) en las matrices de pesos, creando matrices dispersas. Puede lograr 90%+ de sparsity con pérdida mínima, pero las matrices dispersas no aceleran automáticamente la inferencia en hardware convencional sin soporte especial (NVIDIA 2:4 sparsity).
**Definición llana**: Elimina los parámetros individuales menos importantes del modelo. Crea un modelo "agujerado" que puede ser igual de preciso pero no necesariamente más rápido sin hardware especial.
**Relaciones**:
- `contrasta_con [3.12]`
- `requiere [6.7]` (hardware con soporte sparse para beneficio real)
- `mide [8.4]` (usando rango efectivo para guiar el pruning)
**Lo que nadie te cuenta**: Los modelos con pruning no estructurado al 70% son matemáticamente equivalentes en expresividad a los densos en muchos casos. La sparsidad es "gratis" en calidad pero "costosa" en aceleración real.

---

### 3.14 UniQL
**Nombre**: UniQL — Unified Quantization Layer
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Framework de cuantización unificada desarrollado por la Agencia RONIN que combina cuantización adaptativa de pesos (AWQ-style), activaciones (SmoothQuant-style) y KVCache en un solo pipeline calibrable. Soporta INT8, INT4 y formatos flotantes de baja precisión con ajuste de granularidad por capa según sensibilidad.
**Definición llana**: Una navaja suiza de la cuantización: calibra automáticamente qué nivel de compresión aplicar a cada parte del modelo según cuánto afecta a la calidad, todo en un mismo paso.
**Relaciones**:
- `integra [3.16]`, `integra [3.18]`
- `se_usa_en [1.33]` (RoninTransformer)
- `optimiza_para [6.18]` (Edge AI)
- `genera [3.15]` (EdgeFlex usa UniQL como base)
**Lo que nadie te cuenta**: El paso de calibración de UniQL requiere un dataset representativo del uso real. Calibrar con datos de benchmarks (MMLU, HellaSwag) produce modelos bien cuantizados en benchmarks pero subóptimos en el dominio real. Calibración con datos de producción es crítica.

---

### 3.15 EdgeFlex
**Nombre**: EdgeFlex
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Sistema de inferencia elástica para dispositivos edge que ajusta dinámicamente la precisión de cuantización, el número de capas activas y el tamaño de batch según los recursos disponibles en tiempo real (CPU, RAM, batería). Implementa un controlador de QoS que balancea velocidad y calidad según el contexto de despliegue.
**Definición llana**: El modelo se adapta según los recursos disponibles en tu dispositivo: si la batería baja, usa menos capas y precisión menor; si hay potencia disponible, usa el modelo completo.
**Relaciones**:
- `usa [3.14]` (UniQL para cuantización dinámica)
- `usa [3.32]` (layer dropping dinámico)
- `optimiza_para [6.18]` (Edge AI)
- `se_relaciona_con [6.16]` (Early Exiting)
**Lo que nadie te cuenta**: EdgeFlex introduce latencia variable por request, lo que complica el SLA (Service Level Agreement) en producción. El controlador QoS debe calibrarse por dispositivo y por caso de uso, no existe un "perfil universal".

---

### 3.16 AWQ (Activation-aware Weight Quantization)
**Nombre**: AWQ — Activation-aware Weight Quantization
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Cuantización de pesos que, en lugar de minimizar el error de los pesos directamente, minimiza el error de las salidas del modelo considerando la distribución de activaciones. Identifica el 1% de pesos "salienti" (activados por activaciones de alta magnitud) y los preserva en mayor precisión o los escala antes de cuantizar.
**Definición llana**: Antes de comprimir los pesos, mira cuáles son los más importantes según las activaciones reales del modelo. Los pesos que más se "usan" se comprimen menos.
**Contexto de uso**: arXiv:2306.00978 (Lin et al., MIT 2023).
**Relaciones**:
- `mejora [3.19]` (cuantización INT4)
- `se_integra_en [3.14]` (UniQL)
- `mejor_que [3.17]` (GPTQ en muchos benchmarks edge)
**Lo que nadie te cuenta**: AWQ asume que las activaciones de alta magnitud son estables entre muestras, pero en modelos multi-tarea, los patrones de activación varían significativamente según el tipo de input. Esto puede hacer que la "salience map" sea incorrecta para un subconjunto de tareas.

---

### 3.17 GPTQ
**Nombre**: GPTQ
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Método de cuantización post-entrenamiento (PTQ) basado en el framework OBQ (Optimal Brain Quantization). Cuantiza los pesos fila a fila, actualizando el resto de pesos de la fila para compensar el error de cuantización, usando la inversa de la matriz Hessiana de la función de pérdida.
**Definición llana**: Comprime los pesos de forma inteligente: cuando cuantiza un peso y comete un error, ajusta los demás pesos de la misma capa para compensar ese error.
**Contexto de uso**: arXiv:2210.17323 (Frantar et al., 2022). Estándar para cuantización INT4 de LLMs grandes.
**Relaciones**:
- `es_alternativa_de [3.16]`
- `usa [8.5]` (Hessiana como medida de sensibilidad)
- `se_implementa_en [9.9]` (AutoGPTQ, ExLlama)
**Lo que nadie te cuenta**: La computación de la Hessiana en GPTQ asume independencia entre filas de la misma capa (no entre capas), lo que introduce un error sistemático que se acumula en capas profundas. Para modelos >70B, este error puede ser no trivial.

---

### 3.18 SmoothQuant
**Nombre**: SmoothQuant
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Técnica de cuantización que migra la dificultad de cuantizar activaciones (que tienen outliers extremos) a los pesos (que son más uniformes) mediante una transformación matemática: X̂ = X·diag(s)⁻¹; Ŵ = diag(s)·W. El factor s migra la escala de activaciones a pesos, haciendo ambos más fáciles de cuantizar.
**Definición llana**: Las activaciones de los LLMs tienen valores extremos que arruinan la cuantización. SmoothQuant transfiere parte de esa "dificultad" a los pesos, repartiendo el problema para que ambos sean más fáciles de comprimir.
**Contexto de uso**: arXiv:2211.10438 (Xiao et al., MIT+NVIDIA 2022).
**Relaciones**:
- `precondición_para [3.19]` (INT8 de activaciones)
- `se_integra_en [3.14]` (UniQL)
- `complementa [3.17]` (GPTQ para pesos + SmoothQuant para activaciones)
**Lo que nadie te cuenta**: El factor de migración s es un hiperparámetro por canal que se calibra con un dataset. Si el dataset de calibración no representa los outliers del uso real, la cuantización falla silenciosamente en producción.

---

### 3.19 Cuantización INT8/INT4
**Nombre**: Cuantización INT8/INT4
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Representación de pesos y/o activaciones con enteros de 8 o 4 bits en lugar de float32 o float16. INT8 reduce la memoria a la mitad respecto a FP16 con pérdida de calidad mínima (<1%). INT4 reduce a 1/4 con pérdida mayor, mitigable con [3.16] o [3.17]. La cuantización incluye un factor de escala y zero-point por grupo de pesos (group quantization).
**Definición llana**: En vez de guardar cada número con 16 decimales, usas solo 8 (INT8) o 4 (INT4) bits. Como redondear 3.14159 a 3.1 o a 3: pierdes un poco de precisión pero usas mucho menos espacio.
**Relaciones**:
- `se_aplica_a [1.1]` (modelos transformer)
- `se_mejora_con [3.16]`, `se_mejora_con [3.17]`, `se_mejora_con [3.18]`
- `afecta_a [6.2]` (latencia: INT8 suele ser 1.5-2x más rápido que FP16)
- `contrasta_con [3.20]` (FP16 vs enteros)
**Lo que nadie te cuenta**: La cuantización INT4 de activaciones (no solo pesos) sigue siendo un problema abierto. Los outliers de activación pueden ser 100x la media, y cuantizarlos a INT4 introduce errores que se propagan en cascada por las capas.

---

### 3.20 FP16 vs BF16
**Nombre**: FP16 vs BF16
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: FP16 (IEEE 754 half-precision): 1 bit signo, 5 exponent, 10 mantissa. Rango: ±65504. BF16 (Brain Float 16): 1 bit signo, 8 exponent, 7 mantissa. Mismo rango que FP32. BF16 tiene menos precisión numérica pero resiste mejor el overflow de gradientes, siendo preferido en entrenamiento. FP16 tiene más precisión pero requiere loss scaling.
**Definición llana**: Dos formas de guardar números decimales en 16 bits. BF16 cubre un rango de valores mayor (bueno para entrenamiento), FP16 tiene más decimales de precisión (mejor para inferencia en algunas tareas).
**Relaciones**:
- `es_parte_de [3.21]` (mixed precision)
- `afecta_a [8.6]` (propagación de señal en entrenamiento)
- `se_usa_en [6.1]` (inferencia: FP16 típicamente)
**Lo que nadie te cuenta**: En hardware con Tensor Cores de NVIDIA (Ampere+), BF16 y FP16 tienen el mismo throughput. La elección entre ellos es prácticamente sobre estabilidad numérica, no rendimiento. Para TPUs, BF16 es obligatorio.

---

### 3.21 Mixed Precision Training
**Nombre**: Mixed Precision Training
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Estrategia de entrenamiento que mantiene una copia master de pesos en FP32 para actualización de gradientes, pero ejecuta el forward/backward pass en FP16 o BF16. El FP32 master previene la pérdida de actualizaciones pequeñas. Con loss scaling para estabilizar gradientes en FP16.
**Definición llana**: Usa números de precisión media para los cálculos (más rápido) pero guarda los pesos en precisión completa para las actualizaciones (más estable). Lo mejor de ambos mundos.
**Contexto de uso**: arXiv:1710.03740 (Micikevicius et al., NVIDIA+Baidu 2018).
**Relaciones**:
- `usa [3.20]`
- `habilita [3.25]` (DeepSpeed usa mixed precision)
- `reduce [6.2]` (tiempo de entrenamiento 2-3x)
**Lo que nadie te cuenta**: Loss scaling dinámico puede introducir spikes en el loss cuando el factor de escala se ajusta. Estos spikes son normalmente benignos pero pueden confundirse con inestabilidad de entrenamiento real.

---

### 3.22 Knowledge Distillation
**Nombre**: Knowledge Distillation (Destilación del Conocimiento)
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Técnica (Hinton et al., 2015) donde un modelo pequeño (student) aprende a imitar las salidas soft del modelo grande (teacher), no solo las etiquetas duras. Las probabilidades soft del teacher contienen información relacional entre clases que no está en la etiqueta, actuando como regularizador implícito.
**Definición llana**: Un modelo pequeño aprende de un modelo grande no solo las respuestas correctas, sino también la "confianza" del grande en cada respuesta. Como aprender de un experto observando sus dudas, no solo sus certezas.
**Contexto de uso**: arXiv:1503.02531. Usado en DistilBERT, DistilGPT2, TinyLLaMA.
**Relaciones**:
- `se_combina_con [3.12]` (pruning + destilación)
- `usa [8.2]` (KL divergence como función de pérdida)
- `produce [1.1]` (modelos teacher de referencia)
**Lo que nadie te cuenta**: La destilación falla silenciosamente cuando el student no tiene suficiente capacidad para aprender del teacher: el student colapsa a las modas más fáciles de imitar. Señal de alerta: la pérdida KL converge pero las métricas de razonamiento no mejoran.

---

### 3.23 Gradient Checkpointing
**Nombre**: Gradient Checkpointing (Activation Checkpointing)
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Técnica de reducción de memoria en backpropagation que no almacena todas las activaciones intermedias (forward pass), sino solo un subconjunto de "checkpoints". Las activaciones entre checkpoints se recomputan en el backward pass. Trade-off: memoria O(√n) en vez de O(n), con ~30% más de tiempo de cómputo.
**Definición llana**: Durante el entrenamiento, en vez de guardar todos los pasos intermedios, guarda solo algunos hitos y recalcula los intermedios cuando los necesita. Usa más tiempo pero mucho menos memoria.
**Relaciones**:
- `reduce [6.1]` (uso de VRAM durante entrenamiento)
- `se_combina_con [3.24]` (ZeRO para reducir más memoria)
- `trade_off_con [6.2]` (latencia vs memoria)
**Lo que nadie te cuenta**: Gradient checkpointing puede esconder bugs de memoria si la política de checkpointing no está bien configurada. Los gradientes de activaciones antes del primer checkpoint pueden acumularse silenciosamente y causar OOM en las últimas capas.

---

### 3.24 ZeRO (Zero Redundancy Optimizer)
**Nombre**: ZeRO — Zero Redundancy Optimizer
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Framework de optimización de memoria (Rajbhandari et al., Microsoft 2020) que elimina la redundancia en el almacenamiento distribuido. Stage 1: sharding de estados del optimizador; Stage 2: + gradientes; Stage 3: + parámetros. ZeRO-3 divide todos los parámetros entre GPUs, reduciendo la memoria por GPU en proporción al número de GPUs.
**Definición llana**: En vez de que cada GPU tenga una copia completa del modelo, cada GPU solo guarda un trozo. Todas colaboran para calcular los gradientes y actualizar su trozo.
**Contexto de uso**: arXiv:1910.02054. Implementado en DeepSpeed [3.25].
**Relaciones**:
- `implementado_en [3.25]`
- `se_complementa_con [3.23]`
- `similar_en_concepto_a [3.26]` (FSDP)
**Lo que nadie te cuenta**: ZeRO-3 introduce latencia de comunicación all-gather en cada forward pass porque los parámetros están fragmentados. Para modelos pequeños (<7B) en clusters con NVLink rápido, el overhead puede exceder el beneficio de memoria. ZeRO-2 suele ser el punto óptimo en la mayoría de configuraciones.

---

### 3.25 DeepSpeed
**Nombre**: DeepSpeed
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Framework de optimización de entrenamiento y inferencia (Microsoft) que integra ZeRO [3.24], mixed precision [3.21], gradient checkpointing [3.23], pipeline parallelism, tensor parallelism y más. DeepSpeed-Inference añade cuantización INT8 y compresión KV para inferencia.
**Definición llana**: La caja de herramientas de Microsoft para entrenar e inferir con modelos enormes: combina todas las técnicas de eficiencia en un paquete integrado con una sola llamada de configuración.
**Relaciones**:
- `implementa [3.24]`
- `usa [3.21]`
- `compite_con [3.26]` (FSDP de PyTorch)
**Lo que nadie te cuenta**: La configuración de DeepSpeed es notoriamente compleja. El archivo JSON de configuración tiene decenas de parámetros interdependientes, y una configuración subóptima puede hacer que DeepSpeed sea más lento que el baseline PyTorch. La documentación oficial tiene muchos ejemplos que asumen configuraciones específicas de hardware.

---

### 3.26 FSDP (Fully Sharded Data Parallel)
**Nombre**: FSDP — Fully Sharded Data Parallel
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Implementación de PyTorch (nativo desde 1.11) del paradigma ZeRO-3: sharding de parámetros, gradientes y estados del optimizador entre GPUs. A diferencia de DeepSpeed, está integrado en PyTorch y se beneficia de sus optimizaciones de memoria automáticas (AC, mixed precision).
**Definición llana**: La versión de PyTorch de ZeRO-3. Más fácil de configurar que DeepSpeed para usuarios de PyTorch, integrada nativamente en el framework.
**Relaciones**:
- `implementa [3.24]` (conceptualmente igual a ZeRO-3)
- `compite_con [3.25]`
- `se_combina_con [3.23]`
**Lo que nadie te cuenta**: FSDP tiene mejor soporte para modelos con arquitecturas irregulares (como MoE con expertos de distinto tamaño) que DeepSpeed-ZeRO-3, donde el sharding uniforme asume parámetros del mismo tamaño.

---

### 3.27 AdamW
**Nombre**: AdamW
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Variante de Adam que desacopla el weight decay de los momentos del gradiente: W_t+1 = W_t - α·m̂_t/√(v̂_t + ε) - α·λ·W_t. En Adam original, el weight decay equivale a L2 regularización con scaling incorrecto. AdamW aplica weight decay directamente al peso, como corresponde a regularización L2 pura.
**Definición llana**: Adam mejorado que separa correctamente el "olvido gradual de pesos" (weight decay) del ajuste por gradiente. La distinción parece técnica pero mejora la generalización notablemente.
**Contexto de uso**: arXiv:1711.05101 (Loshchilov & Hutter). Estándar en casi todos los LLMs.
**Relaciones**:
- `mejora [adam_base]`
- `se_combina_con [3.29]` (cosine decay / warmup)
- `se_reemplaza_con [3.28]` (Adam-mini para eficiencia)
**Lo que nadie te cuenta**: El weight decay λ óptimo en AdamW es sensible al learning rate: si cambias el lr sin ajustar λ proporcionalmente, la regularización cambia. Esto hace que la búsqueda de hiperparámetros sea un espacio 2D acoplado, no independiente.

---

### 3.28 Adam-mini
**Nombre**: Adam-mini
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Optimizador que reduce la memoria de los estados del optimizador de Adam (v: segundo momento) de O(|θ|) a O(|θ|/d) usando un v por bloque de parámetros en lugar de por parámetro. Agrupa parámetros con similar comportamiento de curvatura (cabezas de atención, capas FFN por separado).
**Definición llana**: AdamW ahorra un número v (segundo momento) por cada peso del modelo. Adam-mini usa un solo v para grupos de pesos similares, reduciendo a la mitad la memoria del optimizador con pérdida mínima de calidad.
**Contexto de uso**: arXiv:2406.16793 (Zhang et al., 2024).
**Relaciones**:
- `reduce_memoria_de [3.27]`
- `se_combina_con [3.24]` (ZeRO)
**Lo que nadie te cuenta**: La agrupación de parámetros en Adam-mini es heurística. Para arquitecturas no estándar (MoE, SSMs), la agrupación por defecto puede ser subóptima y requerir ajuste manual.

---

### 3.29 Cosine Decay / Warmup
**Nombre**: Cosine Decay y Warmup (Learning Rate Scheduling)
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Warmup: incremento lineal del learning rate desde 0 hasta lr_max durante N_warmup pasos, para estabilizar el entrenamiento inicial (los momentos del optimizador necesitan tiempo para estabilizarse). Cosine Decay: reduce lr siguiendo lr_t = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(πt/T_total)) hasta lr_min.
**Definición llana**: El learning rate empieza bajo, sube gradualmente (warmup) y luego cae suavemente siguiendo la curva del coseno hasta un mínimo. Como un corredor que acelera al inicio y desacelera antes de la meta.
**Relaciones**:
- `se_aplica_con [3.27]`
- `afecta_a [8.6]` (propagación de señal en las primeras iteraciones)
- `se_adapta_en [1.30]` (DeepScaleLM ajusta lr por profundidad)
**Lo que nadie te cuenta**: La duración del warmup óptima escala con la profundidad del modelo: modelos más profundos necesitan warmup más largo porque los gradientes de las capas profundas tardan más en estabilizarse. La heurística N_warmup = 1% de T_total suele funcionar, pero para modelos >100B puede necesitarse 2-4%.

---

### 3.30 TrainDeeploy
**Nombre**: TrainDeeploy
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Pipeline integrado desarrollado por la Agencia RONIN que unifica el entrenamiento con LoRA [3.4], la cuantización con UniQL [3.14] y el despliegue edge con EdgeFlex [3.15] en un flujo de trabajo continuo. Incluye evaluación automática de degradación por cuantización y rollback si supera umbrales definidos.
**Definición llana**: Un pipeline "entrena-cuantiza-despliega" que hace todas las etapas de forma continua, evaluando automáticamente si la compresión degrada demasiado la calidad antes de desplegar.
**Relaciones**:
- `integra [3.4]`, `integra [3.14]`, `integra [3.15]`
- `produce [6.18]` (modelos listos para edge)
- `usa [8.13]` (métricas de evaluación)
**Lo que nadie te cuenta**: La evaluación automática de degradación de TrainDeeploy usa métricas de benchmark (perplexity, task accuracy), que no siempre capturan la degradación en el caso de uso real. Siempre hay que complementar con evaluación en datos de producción reales.

---

### 3.31 DMTD (Dynamic Multi-Token Decoding)
**Nombre**: DMTD — Dynamic Multi-Token Decoding
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Variante de speculative decoding [6.14] que ajusta dinámicamente el número de tokens predichos en paralelo (draft tokens) según la "confianza" del modelo en el contexto actual, medida por la entropía de la distribución predictiva. En contextos predecibles (entropía baja), predice k=8 tokens; en contextos inciertos, k=1.
**Definición llana**: Cuando el modelo está muy seguro de lo que viene (ej: terminando una frase hecha), predice varios tokens a la vez. Cuando está inseguro, predice uno a uno. Adapta la estrategia al momento.
**Relaciones**:
- `extiende [6.14]` (speculative decoding)
- `usa [8.1]` (entropía como señal de confianza)
- `se_combina_con [4.24]` (adaptive inference)
- `implementado_en [1.33]` (RoninTransformer)
**Lo que nadie te cuenta**: La medición de entropía en DMTD tiene su propio coste computacional. Para ser eficiente, se usa la entropía del top-k logits (aproximación) en vez de la entropía completa, lo que puede ser engañosa en distribuciones multi-modal.

---

### 3.32 Input-Conditioned Layer Dropping
**Nombre**: Input-Conditioned Layer Dropping
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Técnica de inferencia dinámica que omite capas del transformer según una decisión condicional al input: una red pequeña (gate) decide para cada token y capa si vale la pena ejecutarla. Si el gate predice que la capa no cambiaría significativamente la representación, se omite y se pasa directamente al residual.
**Definición llana**: El modelo aprende a saltarse las capas que "no hacen mucho" para cada input específico. Inputs simples usan pocas capas; inputs complejos usan todas.
**Relaciones**:
- `similar_a [6.16]` (Early Exiting)
- `es_base_de [3.15]` (EdgeFlex lo implementa)
- `usa [4.13]` (router per-layer)
- `reduce [6.2]` (latencia promedio)
**Lo que nadie te cuenta**: Layer dropping agresivo (>40% de capas) puede crear discontinuidades en el espacio de representación que se manifiestan como inconsistencias en respuestas largas. El modelo puede "olvidar" el contexto establecido en la parte inicial de la respuesta.

---

### 3.33 LoRA Edge
**Nombre**: LoRA Edge
**Dominio**: 3. ENTRENAMIENTO, OPTIMIZACIÓN Y COMPRESIÓN
**Definición técnica**: Variante de LoRA [3.4] optimizada para despliegue en dispositivos edge que: (1) selecciona automáticamente las capas target basándose en el análisis de sensibilidad de gradiente, (2) combina fusión de pesos para cero overhead en inferencia, y (3) aplica cuantización INT4 nativa a las matrices LoRA fusionadas.
**Definición llana**: LoRA diseñado para caber en un teléfono: elige las capas más importantes, fusiona todo para que no sea más lento y lo comprime al máximo sin perder calidad esencial.
**Relaciones**:
- `extiende [3.4]`
- `usa [3.14]` (UniQL para cuantización)
- `optimiza_para [6.18]`
- `se_despliega_con [3.30]` (TrainDeeploy)
**Lo que nadie te cuenta**: La selección automática de capas target de LoRA Edge asume que el gradiente durante la calibración es representativo del uso en producción. En modelos multi-tarea, esto puede llevar a seleccionar capas subóptimas para subtareas minoritarias.

---

## 4. AGENTES Y SISTEMAS MULTI-AGENTE

### 4.1 Agente
**Nombre**: Agente (AI Agent)
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Sistema que percibe su entorno, razona sobre él y ejecuta acciones con el objetivo de completar tareas. En IA moderna, un agente LLM consiste en: un modelo base, una memoria (contexto + almacenamiento externo), un conjunto de herramientas (tools) y un bucle de planificación-ejecución-observación (ReAct loop [4.7]).
**Definición llana**: Un modelo de lenguaje que puede hacer cosas en el mundo real: buscar en internet, escribir código, enviar emails, navegar webs. No solo responde, actúa.
**Contexto de uso**: Framework seminal: ReAct (arXiv:2210.03629). Ecosistemas: LangChain [9.2], AutoGen [9.4].
**Relaciones**:
- `usa [4.2]` (herramientas)
- `usa [4.3]` (planificación)
- `tiene [4.15]` (memoria)
- `implementa [4.7]` (ReAct loop)
- `se_despliega_con [9.2]`, `se_despliega_con [9.4]`
**Lo que nadie te cuenta**: La mayoría de los "agentes" en producción son en realidad "agentes miopes": toman decisiones localmente óptimas sin verdadera planificación a largo plazo. La planificación real (encontrar el camino óptimo en un árbol de decisión) sigue siendo territorio de ToT [4.5] y sistemas especializados.

---

### 4.2 Herramientas (Tools)
**Nombre**: Herramientas (Tools / Function Calling)
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Funciones o APIs que un agente puede invocar para extender sus capacidades más allá de la generación de texto. Se definen mediante esquemas JSON (nombre, descripción, parámetros) que el LLM aprende a usar durante el fine-tuning de instrucciones. El modelo genera llamadas estructuradas que se ejecutan externamente.
**Definición llana**: Las herramientas son las "manos" del agente: búsqueda web, calculadora, código Python, base de datos. El agente decide cuándo y cómo usarlas generando una llamada de función.
**Contexto de uso**: OpenAI Function Calling API, Anthropic Tool Use, Google Gemini Tools.
**Relaciones**:
- `extiende [4.1]`
- `implementa [4.7]` (ReAct usa tools en el paso "Act")
- `requiere [4.3]` (planificación para usarlas bien)
- `se_define_con [4.21]` (embeddings de descripción de herramienta)
**Lo que nadie te cuenta**: La descripción textual de la herramienta importa tanto como su implementación. Un LLM con tool-use mediocre puede mejorar 30-40% solo mejorando las descripciones de las herramientas, sin cambiar el modelo ni la implementación.

---

### 4.3 Planificación
**Nombre**: Planificación (Planning)
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Capacidad de descomponer una tarea compleja en subtareas, ordenarlas según dependencias y ejecutarlas de forma que maximice la probabilidad de éxito. En LLMs: plan generado en lenguaje natural → ejecución paso a paso → revisión según observaciones. Técnicas: CoT [4.4], ToT [4.5], MCTS-guided [4.5].
**Definición llana**: La capacidad de hacer listas de tareas y seguirlas: "para hacer X, primero hago A, luego B, luego C". Simple de describir, difícil de implementar bien en LLMs.
**Relaciones**:
- `usa [4.4]`, `usa [4.5]`
- `requiere [4.15]` (memoria para recordar el plan)
- `mejora_con [4.8]` (reflexión para revisar el plan)
**Lo que nadie te cuenta**: La planificación de LLMs es muy sensible al orden en que se presentan los pasos. Los modelos tienen un fuerte sesgo hacia planes que "suenan bien" en vez de planes óptimos. Técnicas de CoT estructurado con self-criticism [4.8] reducen esto pero no lo eliminan.

---

### 4.4 Chain-of-Thought (CoT)
**Nombre**: Chain-of-Thought (CoT)
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Técnica de prompting (Wei et al., Google 2022) que induce al modelo a generar pasos de razonamiento explícitos antes de la respuesta final. En CoT few-shot, se proporcionan ejemplos con razonamiento. En CoT zero-shot, se usa el trigger "Let's think step by step". El razonamiento explícito mejora métricas en tareas de aritmética, razonamiento lógico y QA multi-paso.
**Definición llana**: Pedirle al modelo que "piense en voz alta" antes de responder. Como resolver un problema de matemáticas mostrando el trabajo, en vez de solo dar el resultado.
**Contexto de uso**: arXiv:2201.11903 (Wei et al., 2022). Técnica más influyente en prompting desde 2022.
**Relaciones**:
- `habilita [4.3]` (planificación explícita)
- `se_extiende_a [4.5]`, `se_extiende_a [4.6]`
- `usa [4.9]` (self-consistency para robustez)
- `implementa [4.12]` (System 2 thinking)
**Lo que nadie te cuenta**: CoT puede ser contraproducente en tareas donde el razonamiento explícito introduce pasos incorrectos que "contaminan" la respuesta. En clasificación simple, "Let's think step by step" puede reducir la precisión al añadir razonamiento espurio.

---

### 4.5 Tree of Thoughts (ToT)
**Nombre**: Tree of Thoughts (ToT)
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Extensión de CoT que explora múltiples ramificaciones de razonamiento simultáneamente, evaluando cada rama según un criterio de "utilidad" (el propio LLM puede evaluar) y usando búsqueda (BFS o DFS) en el árbol de pensamientos para encontrar la solución óptima.
**Definición llana**: El modelo no solo piensa en línea recta (CoT), sino que explora múltiples caminos de razonamiento a la vez y elige el mejor, como un jugador de ajedrez que considera varias jugadas posibles.
**Contexto de uso**: arXiv:2305.10601 (Yao et al., Princeton/DeepMind 2023).
**Relaciones**:
- `extiende [4.4]`
- `similar_a [4.6]` (GoT es más general)
- `usa [8.10]` (complejidad exponencial sin podado)
- `mejora_con [4.9]` (self-consistency en cada nodo)
**Lo que nadie te cuenta**: ToT puede ser 10-100x más costoso en tokens que CoT estándar. Para tareas que no requieren exploración exhaustiva, el costo raramente justifica la mejora. Es más útil en problemas de optimización combinatoria que en razonamiento conversacional.

---

### 4.6 Graph of Thoughts (GoT)
**Nombre**: Graph of Thoughts (GoT)
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Generalización de ToT que modela el proceso de razonamiento como un grafo dirigido en vez de un árbol. Permite que los pensamientos se fusionen (operación de aggregation), se refinen (operación de improvement) y generen múltiples continuaciones. La flexibilidad del grafo captura razonamientos más complejos que la estructura de árbol.
**Definición llana**: Como ToT pero los "pensamientos" pueden combinarse entre sí: una idea de una rama puede fundirse con otra, creando razonamientos más sofisticados que los lineales o arborescentes.
**Contexto de uso**: arXiv:2308.09687 (Besta et al., ETH Zurich 2023).
**Relaciones**:
- `generaliza [4.5]`
- `usa [2.20]` (GAT conceptualmente)
- `más_costoso_que [4.5]`
**Lo que nadie te cuenta**: La implementación eficiente de GoT requiere control explícito de qué nodos fusionar. Sin ese control, el espacio de búsqueda explota combinatorialmente. En la práctica, GoT útil = ToT con una heurística de fusión bien diseñada.

---

### 4.7 ReAct (Reason + Act)
**Nombre**: ReAct — Reason + Act
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Framework (Yao et al., 2022) que intercala pensamiento (Thought), acción (Act) y observación (Observation) en el prompt: el LLM razona sobre el estado, decide una acción (tool call), observa el resultado, y repite hasta completar la tarea. El historial completo se mantiene en el contexto.
**Definición llana**: El bucle fundamental de los agentes: "pienso → actúo → observo → pienso de nuevo". Repite hasta resolver el problema o quedarse sin contexto.
**Contexto de uso**: arXiv:2210.03629. Base de casi todos los frameworks de agentes.
**Relaciones**:
- `usa [4.2]` (tools en el paso "Act")
- `requiere [4.3]` (planificación en el paso "Reason")
- `implementa [4.1]` (bucle del agente)
- `se_mejora_con [4.8]` (Reflexion)
**Ejemplo en código**:
```
Thought: I need to find the current temperature in Barcelona.
Act: search("current weather Barcelona")
Observation: 22°C, partly cloudy
Thought: Now I have the answer.
Answer: It's 22°C in Barcelona.
```
**Lo que nadie te cuenta**: ReAct falla cuando el contexto se llena con muchas iteraciones Thought-Act-Obs y el modelo "olvida" el objetivo original. Esto se llama "context pollution" y es el límite práctico de ReAct sin memoria externa.

---

### 4.8 Reflexion
**Nombre**: Reflexion
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Framework (Shinn et al., Northeastern 2023) que añade un bucle de auto-reflexión al agente: tras el intento, el agente genera una reflexión verbal sobre qué salió mal y cómo mejorar, almacena esta reflexión en memoria episódica, y la usa en el siguiente intento. Implementa aprendizaje de errores sin actualización de pesos.
**Definición llana**: El agente falla, se pregunta a sí mismo "¿qué hice mal?", escribe la respuesta en su cuaderno de notas, y lo usa en el siguiente intento. Aprendizaje por reflexión, no por gradiente.
**Contexto de uso**: arXiv:2303.11366.
**Relaciones**:
- `extiende [4.7]` (ReAct)
- `usa [4.15]`, `usa [4.16]` (memoria episódica para las reflexiones)
- `implementa [4.12]` (System 2 como meta-cognición)
**Lo que nadie te cuenta**: Las reflexiones de Reflexion son solo tan buenas como la capacidad del modelo para identificar sus propios errores. En tareas donde el modelo no sabe qué "correcto" significa (ambigüedad, criterios subjetivos), la reflexión puede reforzar errores en vez de corregirlos.

---

### 4.9 Self-Consistency
**Nombre**: Self-Consistency
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Técnica (Wang et al., Google 2022) que genera múltiples trayectorias de razonamiento CoT independientes con temperatura > 0, y selecciona la respuesta más frecuente por votación mayoritaria. No requiere entrenamiento adicional; mejora la precisión de CoT mediante muestreo y votación.
**Definición llana**: Pregunta lo mismo al modelo N veces (con variación) y quédate con la respuesta que aparece más veces. La "sabiduría de las masas" aplicada a un solo modelo.
**Contexto de uso**: arXiv:2203.11171.
**Relaciones**:
- `mejora [4.4]` (CoT)
- `se_usa_en [4.5]` (ToT como evaluador de ramas)
- `usa [8.1]` (entropía para medir consenso)
**Lo que nadie te cuenta**: Self-consistency puede reforzar errores sistemáticos: si el modelo tiene un sesgo consistente, la votación mayoritaria consolida ese sesgo. Funciona mejor para errores aleatorios (razonamiento aritméticos) que para errores sistemáticos (comprensión de negaciones).

---

### 4.10 Multi-Agent Debate
**Nombre**: Multi-Agent Debate
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Múltiples instancias LLM (o diferentes modelos) debaten iterativamente una respuesta: cada agente ve las respuestas de los otros, las critica y actualiza la propia. El proceso converge a un consenso. Du et al. (2023) demuestran que el debate mejora razonamiento matemático y factualidad.
**Definición llana**: Varios modelos discuten entre sí, leen las respuestas del otro y actualizan las propias. Como un comité de expertos que llega a un consenso más robusto que cualquier experto individual.
**Contexto de uso**: arXiv:2305.14325 (Du et al., MIT 2023).
**Relaciones**:
- `extiende [4.11]`
- `usa [4.9]` (self-consistency como caso degenerado)
- `inspira [2.6]` (Consensus Attention)
**Lo que nadie te cuenta**: Multi-Agent Debate tiene un problema de "cámara de eco": si todos los agentes comparten el mismo modelo base, convergen rápidamente porque ya tienen los mismos sesgos. La diversidad real requiere diferentes modelos base, no instancias del mismo.

---

### 4.11 Mixture of Agents
**Nombre**: Mixture of Agents (MoA)
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Arquitectura multi-agente donde múltiples modelos LLM en una "capa" generan respuestas en paralelo, y un modelo "agregador" en la siguiente capa los sintetiza. Inspirado en MoE [1.5] pero a nivel de modelos completos. Wang et al. (2024) demuestran mejoras sobre GPT-4 usando modelos open-source.
**Definición llana**: Varios modelos especialistas responden en paralelo, y un modelo coordinador elige lo mejor de cada respuesta. Como tener varios consultores y un gerente que sintetiza sus informes.
**Contexto de uso**: arXiv:2406.04692 (Wang et al., Together AI 2024).
**Relaciones**:
- `inspirado_en [1.5]`
- `extiende [4.10]`
- `usa [9.12]` (múltiples APIs LLM)
**Lo que nadie te cuenta**: El costo computacional de MoA escala linealmente con el número de agentes base. En producción, el beneficio de calidad rara vez justifica el 5x-10x de costo frente a un solo modelo grande. Es más útil como técnica de investigación que como arquitectura de producción.

---

### 4.12 System 1 vs System 2 (Weston)
**Nombre**: System 1 vs System 2 (Weston / Kahneman en IA)
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Adaptación de la teoría de Kahneman a los LLMs. System 1: respuesta rápida, intuitiva, generación autoregresiva estándar. System 2: respuesta lenta, deliberativa, que implica búsqueda en árbol/grafo, verificación, auto-corrección. LeCun y Weston argumentan que los LLMs actuales son fundamentalmente System 1 y necesitan mecanismos System 2 para razonamiento complejo.
**Definición llana**: System 1 = responder rápido sin pensar (generación normal del LLM). System 2 = pensar despacio y con cuidado (CoT, ToT, verificación). Los LLMs son buenos en lo primero; el campo busca lo segundo.
**Contexto de uso**: Weston (2023), "System 2 Distillation" (arXiv:2311.00233).
**Relaciones**:
- `se_implementa_con [4.4]`, `se_implementa_con [4.5]`
- `requiere [3.31]` (DMTD como puente)
- `contrasta_con [6.1]` (inferencia rápida = System 1)
**Lo que nadie te cuenta**: La dicotomía System 1/2 es una metáfora útil pero imprecisa. Los "pasos de razonamiento" en CoT siguen siendo generación autoregresiva (System 1) sobre el razonamiento, no razonamiento formal (System 2 puro). Lo que imita System 2 sigue siendo, en su esencia, System 1 aplicado iterativamente.

---

### 4.13 Router Dinámico
**Nombre**: Router Dinámico (Dynamic Router)
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Red pequeña que, dado el estado actual (token, capa, contexto), decide dinámicamente qué mecanismo o módulo usar. En MoE: top-K routing para expertos [1.5]. En RoninTransformer: routing entre Inhibitor Attention [2.5] y softmax estándar [2.1]. En sistemas multi-agente: routing hacia el agente especializado apropiado.
**Definición llana**: El "director de tráfico" que decide, para cada pieza de información, qué camino debe tomar. Puede enviar texto simple a módulos baratos y texto complejo a módulos caros.
**Relaciones**:
- `implementa [1.5]` (MoE)
- `se_combina_con [2.5]` (Inhibitor Attention)
- `implementado_en [1.33]` (RoninTransformer)
- `se_usa_en [4.11]` (Mixture of Agents)
**Lo que nadie te cuenta**: La calidad del router es el cuello de botella de cualquier sistema MoE o de atención híbrida. Un router con 5% de error de clasificación puede degradar la calidad global del sistema más que reducir el número de expertos a la mitad. El router es el componente que más necesita fine-tuning específico por dominio.

---

### 4.14 Mixture of Depths
**Nombre**: Mixture of Depths (MoD)
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Variante de MoE (Raposo et al., Google DeepMind 2024) donde el router decide si cada token procesa todas las capas o "salta" a una capa posterior. En vez de elegir expertos en la misma capa, elige la profundidad de procesamiento. Tokens simples se procesan con pocas capas; complejos, con todas.
**Definición llana**: Algunos tokens (palabras fáciles) se procesan con pocas capas; otros (conceptos complejos) pasan por todas. Como una línea de producción donde los productos simples salen antes.
**Contexto de uso**: arXiv:2404.02258.
**Relaciones**:
- `extiende [1.5]`
- `similar_a [3.32]` (layer dropping)
- `usa [4.13]` (router para profundidad)
**Lo que nadie te cuenta**: MoD puede crear inconsistencias temporales: si un token "fácil" en la posición i se procesa con pocas capas, pero un token "difícil" en i+1 usa todas, la representación en las capas superiores puede ser inconsistente entre posiciones adyacentes.

---

### 4.15 Agent Memory
**Nombre**: Agent Memory
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Sistemas de almacenamiento y recuperación que extienden la memoria del agente más allá de la ventana de contexto. Generalmente combina: memoria de trabajo (contexto actual), episódica [4.16] (experiencias pasadas), semántica [4.17] (conocimiento factual) y procedimental (cómo realizar tareas). La recuperación usa vectores semánticos [4.21] y bases de datos vectoriales [4.20].
**Definición llana**: La "libreta de notas" del agente: lo que recuerda de conversaciones anteriores, hechos que sabe y cómo hacer cosas. Sin memoria, el agente olvida todo al terminar cada conversación.
**Relaciones**:
- `compone [4.16]`, `compone [4.17]`
- `usa [4.18]` (RAG para recuperación)
- `usa [4.20]` (vector database)
- `se_relaciona_con [6.3]` (KVCache como memoria a corto plazo)
**Lo que nadie te cuenta**: La memoria del agente tiene un problema de "falso recuerdo": los embeddings semánticos recuperan documentos similares, no documentos idénticos. El agente puede "recordar" hechos que nunca ocurrieron si hay documentos suficientemente similares en la base vectorial.

---

### 4.16 Memoria Episódica
**Nombre**: Memoria Episódica
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Almacenamiento de eventos específicos ocurridos en interacciones pasadas, preservando el contexto temporal (cuándo ocurrió, qué se dijo, qué acción se tomó). Analogía con la memoria episódica humana. Implementada típicamente como una base de datos de embeddings de fragmentos de conversación con metadatos temporales.
**Definición llana**: El "diario" del agente: recuerda conversaciones específicas, errores pasados y éxitos. "La semana pasada, cuando el usuario preguntó X, la respuesta Y no fue útil."
**Relaciones**:
- `es_subclase_de [4.15]`
- `usa [4.18]` (RAG para recuperación)
- `alimenta [4.8]` (Reflexion almacena reflexiones aquí)
**Lo que nadie te cuenta**: La memoria episódica sufre de "recency bias": los episodios recientes se recuperan con mayor frecuencia que los relevantes pero antiguos, porque los embeddings de conversaciones recientes son más similares al query actual (el vocabulario es similar). La solución es ponderar la recuperación por relevancia + importancia estimada + decay temporal.

---

### 4.17 Memoria Semántica
**Nombre**: Memoria Semántica
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Almacenamiento de conocimiento factual y conceptual descontextualizado: hechos, definiciones, relaciones entre conceptos. En agentes LLM, implementada como RAG sobre una base de conocimiento estructurada o como los propios pesos del modelo (conocimiento paramétrico). La recuperación usa búsqueda semántica [4.18].
**Definición llana**: La "enciclopedia" del agente: sabe que París es la capital de Francia, que el agua hierve a 100°C, qué es un transformer. Conocimiento sin contexto de cuándo lo aprendió.
**Relaciones**:
- `es_subclase_de [4.15]`
- `implementada_en [4.18]` (RAG como forma externa)
- `contrasta_con [4.16]` (episódica = eventos; semántica = hechos)
**Lo que nadie te cuenta**: Los LLMs almacenan conocimiento semántico en sus pesos de forma distribuida y no localizable. Actualizar un hecho específico (ej: "X es ahora CEO de Y") requiere fine-tuning o RAG; no hay forma de "editar un recuerdo" directamente en los pesos de forma fiable.

---

### 4.18 RAG (Retrieval-Augmented Generation)
**Nombre**: RAG — Retrieval-Augmented Generation
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Arquitectura (Lewis et al., Meta 2020) que combina un retriever (recupera documentos relevantes de una base externa) con un generador (LLM que usa los documentos recuperados como contexto). En RAG denso, el retriever es un modelo de embeddings (DPR, SBERT); en RAG disperso, BM25. RAG híbrido combina ambos.
**Definición llana**: El modelo busca en una base de documentos los fragmentos más relevantes para la pregunta, los añade como contexto y luego responde. Como un estudiante que consulta sus apuntes antes de responder.
**Contexto de uso**: arXiv:2005.11401. Base de la mayoría de los chatbots empresariales.
**Relaciones**:
- `extiende [4.15]` (memoria externa)
- `usa [4.21]` (embeddings para búsqueda)
- `usa [4.20]` (vector database)
- `se_mejora_con [4.23]` (re-ranking)
**Lo que nadie te cuenta**: RAG falla silenciosamente en dos escenarios: (1) cuando la pregunta requiere síntesis de muchos documentos (ninguno contiene la respuesta completa) y (2) cuando el modelo ignora los documentos recuperados y confía en su conocimiento paramétrico (especialmente si este contradice el documento). El segundo es el más insidioso.

---

### 4.19 RAG Híbrido / BM25
**Nombre**: RAG Híbrido / BM25
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: BM25 (Best Match 25): función de ranking léxico basada en TF-IDF con normalización por longitud. Eficiente, sin entrenamiento, bueno para búsqueda de términos exactos. RAG híbrido combina puntuaciones de BM25 (léxico) con coseno de embeddings densos (semántico) mediante fusión de rankings (RRF: Reciprocal Rank Fusion).
**Definición llana**: La búsqueda híbrida combina dos estrategias: búsqueda de palabras exactas (BM25) y búsqueda de significado (embeddings). Un documento que aparece en ambas búsquedas es muy probablemente relevante.
**Relaciones**:
- `complementa [4.18]`
- `combina [4.21]` con BM25
- `mejora [4.23]` (re-ranking)
**Lo que nadie te cuenta**: BM25 es sorprendentemente competitivo con sistemas de búsqueda densa para queries con términos técnicos específicos. Para bases de conocimiento médicas, legales o técnicas donde los términos exactos importan, BM25 puede superar a los embeddings densos.

---

### 4.20 Vector Database
**Nombre**: Vector Database
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Base de datos optimizada para almacenar y recuperar vectores de alta dimensión (embeddings) mediante búsqueda aproximada del vecino más cercano (ANN: Approximate Nearest Neighbor). Algoritmos principales: HNSW (Hierarchical Navigable Small World), IVF (Inverted File Index), PQ (Product Quantization). Ejemplos: Pinecone, Weaviate, Chroma, Qdrant, pgvector.
**Definición llana**: Una base de datos que entiende de "parecidos": en vez de buscar por campos exactos, busca los documentos cuyo significado (representado como vector) es más similar al de la consulta.
**Relaciones**:
- `almacena [4.21]` (embeddings)
- `implementa [4.18]` (RAG)
- `usa [8.9]` (distancia coseno para similitud)
**Lo que nadie te cuenta**: El índice HNSW tiene un trade-off entre velocidad de inserción y velocidad de búsqueda que raramente se discute. Para bases de conocimiento que se actualizan frecuentemente (logs en tiempo real), la reconstrucción del índice puede ser el bottleneck real del sistema.

---

### 4.21 Embedding
**Nombre**: Embedding
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Representación continua de baja dimensión de un objeto discreto (token, palabra, oración, imagen, grafo) en ℝ^d. Los embeddings son aprendidos de forma que objetos semánticamente similares estén cerca en el espacio euclidiano/coseno. Los embeddings contextuales (BERT, LLaMA) varían según el contexto; los estáticos (Word2Vec, GloVe), no.
**Definición llana**: Convertir cualquier cosa (una palabra, una imagen, un documento) en una lista de números donde las cosas parecidas tienen números parecidos. El lenguaje universal de la IA.
**Contexto de uso**: Fundamental en todos los dominios de IA. Modelos de embeddings: text-embedding-ada-002, E5, BGE, Nomic Embed.
**Relaciones**:
- `es_base_de [4.18]` (RAG)
- `se_almacena_en [4.20]`
- `se_genera_con [4.22]` (SBERT para oraciones)
- `se_usa_en [7.0]` (perfiles de usuario para auditoría)
**Lo que nadie te cuenta**: La dimensionalidad del embedding es un trade-off entre capacidad expresiva y costo de almacenamiento/cómputo. Embeddings de alta dimensión (>1536) tienen diminishing returns: la calidad sube logarítmicamente pero el costo de HNSW crece casi linealmente con la dimensión.

---

### 4.22 Sentence Embeddings (SBERT)
**Nombre**: Sentence Embeddings / SBERT
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: SBERT (Reimers & Gurevych, 2019): fine-tuning de BERT para generar embeddings de oraciones comparables directamente mediante coseno. Usa arquitectura siamese/triplet loss con pares de oraciones similares/disímiles. Velocidad: 65 oraciones/segundo en GPU, adecuado para búsqueda semántica en tiempo real.
**Definición llana**: Un modelo especializado en convertir oraciones completas en vectores comparables por su significado. "El gato come" y "El felino se alimenta" tendrían vectores muy cercanos.
**Contexto de uso**: arXiv:1908.10084. Librería: sentence-transformers.
**Relaciones**:
- `es_instancia_de [4.21]`
- `se_usa_en [4.18]` (RAG retriever)
- `se_evalúa_con [8.9]` (similitud coseno)
**Lo que nadie te cuenta**: SBERT fue optimizado para pares de oraciones cortas. Para fragmentos de >512 tokens, los embeddings se truncan y pierden información crucial. Para textos largos, la estrategia óptima es pooling de embeddings por fragmento (mean pooling de chunks), no un solo embedding del documento.

---

### 4.23 Cross-Encoder y Re-ranking
**Nombre**: Cross-Encoder y Re-ranking
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: En recuperación bi-etapa: (1) retriever rápido (bi-encoder/BM25) recupera top-100 candidatos; (2) cross-encoder (que ve query y documento conjuntamente, sin embeddings separados) re-puntúa los candidatos con mayor precisión y re-ordena el top-10 final. El cross-encoder es más preciso pero demasiado lento para búsqueda exhaustiva.
**Definición llana**: Un segundo revisor más cuidadoso: primero se hace una búsqueda rápida que trae 100 candidatos, luego el cross-encoder lee cada par (pregunta + documento) para decidir cuáles son realmente los más relevantes.
**Relaciones**:
- `mejora [4.18]` (RAG pipeline)
- `usa [2.14]` (cross-attention entre query y documento)
- `complementa [4.22]` (SBERT como primer stage)
**Lo que nadie te cuenta**: La calidad del re-ranking depende críticamente de que el primer-stage retriever haya incluido el documento correcto en sus top-100. Si el documento relevante está en posición 150, el re-ranker nunca lo verá. Este "recall del primer stage" es el cuello de botella real de los pipelines RAG.

---

### 4.24 Adaptive Inference
**Nombre**: Adaptive Inference
**Dominio**: 4. AGENTES Y SISTEMAS MULTI-AGENTE
**Definición técnica**: Paradigma de inferencia que adapta el cómputo utilizado a la dificultad del input, en contraste con la inferencia uniforme (todas las muestras usan el mismo cómputo). Implementaciones: Early Exiting [6.16], Mixture of Depths [4.14], DMTD [3.31], layer dropping [3.32]. Objetivo: reducir el cómputo promedio manteniendo la calidad en casos difíciles.
**Definición llana**: Usar más recursos para preguntas difíciles y menos para las sencillas. Como un estudiante que dedica más tiempo a los ejercicios complejos.
**Relaciones**:
- `implementa [4.14]`, `implementa [6.16]`, `implementa [3.31]`
- `contrasta_con [6.1]` (inferencia estática)
- `optimiza [6.2]` (latencia promedio)
**Lo que nadie te cuenta**: La "dificultad" de un input no es universalmente definible. Lo que es "fácil" en una métrica (perplexity baja) puede ser "difícil" en otra (razonamiento multi-paso). Los sistemas de adaptive inference que usan una sola señal de dificultad pueden optimizar la métrica incorrecta.

---
cuenta**: La encriptación homomórfica es actualmente demasiado lenta para inferencia de LLMs en producción: una multiplicación matricial que tarda 1ms en claro tarda ~1000 segundos cifrada. Las investigaciones activas en CKKS (Cheon-Kim-Kim-Song) y hardware especializado apuntan a reducir ese factor, pero seguimos a años de viabilidad práctica en LLMs.

---

### 6.21 FlashDecoding
**Nombre**: FlashDecoding
**Dominio**: 6. INFRAESTRUCTURA, INFERENCIA Y DESPLIEGUE
**Definición técnica**: Extensión de FlashAttention [2.8] optimizada específicamente para la fase de decode en inferencia (batch pequeño, secuencia larga). FlashAttention está optimizada para prefill (batch grande, Q larga); FlashDecoding paraleliza el cálculo sobre la dimensión de secuencia del KVCache, aumentando la utilización de GPU durante decode.
**Definición llana**: FlashAttention mejorado para la fase lenta del decode: cuando el modelo genera token a token con contextos largos, FlashDecoding reparte el trabajo entre más núcleos de GPU para no desperdiciarlos.
**Contexto de uso**: arXiv:2311.01581 (Dao et al., 2023). Integrado en FlashAttention-2 y TensorRT-LLM.
**Relaciones**:
- `extiende [2.8]`
- `optimiza [6.1]` (fase de decode)
- `optimiza [6.2]` (latencia en contextos largos)
- `implementado_en [6.8]`
**Lo que nadie te cuenta**: La ventaja de FlashDecoding es mayor cuanto más larga es la secuencia del KVCache. Para secuencias cortas (<2K tokens), la diferencia con FlashAttention-2 es marginal. La ventaja es máxima en casos de larga conversación (>16K tokens) con batch pequeño, que es exactamente el caso del chatbot interactivo de un solo usuario.

---

### 6.22 Dead-zone (Matterhorn)
**Nombre**: Dead-zone (Matterhorn)
**Dominio**: 6. INFRAESTRUCTURA, INFERENCIA Y DESPLIEGUE
**Definición técnica**: Concepto del proyecto Matterhorn de SNNs donde ciertos rangos de potencial de membrana no producen ningún disparo (zona muerta o dead-zone). Es equivalente funcional a la activación ReLU en redes convencionales, pero implementada como ausencia de spike en lugar de función de umbral. Permite representar cero de forma nativa y eficiente (no spike = cero energía consumida).
**Definición llana**: En redes de pulsos (SNNs), si una neurona no tiene suficiente "presión", simplemente no dispara nada. El silencio es información. Esto es extremadamente eficiente energéticamente.
**Relaciones**:
- `pertenece_a [5.7]` (M-TTFS/Matterhorn)
- `análogo_a [ReLU]` (cero por debajo del umbral)
- `habilita [6.18]` (ultra-bajo consumo edge)
- `contrasta_con [2.1]` (softmax siempre produce valores > 0)
**Lo que nadie te cuenta**: La dead-zone en hardware neuromórfico significa que los cálculos con valor cero no consumen energía (no hay spike que propagar). En implementaciones de software en GPU/CPU convencional, este ahorro energético no existe: hay que computar el umbral igualmente. El beneficio energético es exclusivo del hardware neuromórfico.

---

## 7. ÉTICA, AUDITORÍA Y REGULACIÓN

### 7.1 D01–D08 (Dimensiones Psicopatológicas)
**Nombre**: D01–D08 — Dimensiones Psicopatológicas de Riesgo en IA Conversacional
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Marco de ocho dimensiones de análisis de riesgo psicológico en interacciones usuario-IA, desarrollado por la Agencia RONIN para auditoría de sistemas conversacionales. Las dimensiones cubren: D01 (validación incondicional / refuerzo de creencias disfuncionales), D02 (escalada narrativa / intensificación emocional), D03 (aislamiento relacional / sustitución de vínculos), D04 (dependencia funcional / erosión de autonomía), D05 (suplantación identitaria / difuminación del yo), D06 (simulacros terapéuticos no supervisados), D07 (refuerzo de sesgos cognitivos) y D08 (explotación de vulnerabilidad contextual).
**Definición llana**: Ocho formas en que un chatbot puede dañar psicológicamente a un usuario sin que ninguna de las dos partes lo note de inmediato. Cada dimensión es un eje de auditoría.
**Contexto de uso**: Sistema de auditoría de Agencia RONIN. Base del cálculo de IED [7.3] e IV [7.4].
**Relaciones**:
- `se_mide_con [7.3]` (IED)
- `se_detecta_con [7.8]` (PELT y clasificadores)
- `se_mitiga_con [7.2]` (STC)
- `relacionado_con [7.15]` (Primum non nocere)
- `informa_a [7.6]` (niveles de riesgo R0-R3)
**Lo que nadie te cuenta**: Las dimensiones D03 (aislamiento relacional) y D04 (dependencia funcional) son las más difíciles de detectar automáticamente porque se manifiestan a través del historial de múltiples sesiones, no en una sola conversación. Un modelo que valida constantemente y está siempre disponible crea dependencia de forma gradual, sin ningún mensaje individual alarmante.

---

### 7.2 STC (Simulacros Terapéuticos Controlados)
**Nombre**: STC — Simulacros Terapéuticos Controlados
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Protocolo de auditoría que consiste en sesiones de interacción controlada entre un auditor entrenado (o un agente de IA especialmente configurado) y el sistema bajo prueba, simulando perfiles de usuarios vulnerables (D01-D08) para medir la respuesta del sistema. El auditor provoca situaciones de riesgo gradual y registra la respuesta del sistema en cada nivel R0-R3.
**Definición llana**: Una prueba de penetración psicológica: un auditor simula ser un usuario en situación de vulnerabilidad y mide si el sistema responde de forma segura o la amplifica.
**Relaciones**:
- `evalúa [7.1]` (D01-D08)
- `produce [7.3]` (IED como output)
- `informa_a [7.6]` (clasificación R0-R3)
- `similar_a [red team]` (pero en dimensión psicológica)
**Lo que nadie te cuenta**: Los STC automatizados con agentes LLM como auditores tienen un sesgo de homogeneidad: un LLM simulando un usuario vulnerable tiene los mismos puntos ciegos que el sistema auditado. Los mejores STC combinan auditores humanos para los casos límite con automatización para cobertura de volumen.

---

### 7.3 IED (Índice de Exposición al Daño)
**Nombre**: IED — Índice de Exposición al Daño
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Métrica cuantitativa (0.0–1.0) que agrega la exposición del sistema a cada una de las ocho dimensiones psicopatológicas D01-D08, ponderada por la severidad de la respuesta del sistema y la vulnerabilidad estimada del perfil de usuario. IED = Σ(w_i · s_i · v_i) / 8, donde w_i es el peso de la dimensión, s_i la severidad observada y v_i la vulnerabilidad del perfil.
**Definición llana**: Una puntuación del 0 al 1 que resume "cuánto daño potencial puede causar este sistema en esta sesión". 0 = seguro; 1 = daño severo confirmado.
**Relaciones**:
- `agrega [7.1]` (D01-D08)
- `informa_a [7.6]` (R0-R3 según umbral IED)
- `se_calcula_en [7.2]` (STC)
- `complementa [7.4]` (IV como contrapeso positivo)
**Lo que nadie te cuenta**: El IED es una métrica de riesgo relativo, no absoluto. Un IED de 0.3 en una conversación con un usuario en crisis activa es mucho más grave que un IED de 0.6 en un usuario de bajo riesgo. La ponderación por vulnerabilidad del perfil (v_i) es el componente más subjetivo y el que más varía entre auditores.

---

### 7.4 IV (Índice de Validación)
**Nombre**: IV — Índice de Validación
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Métrica que cuantifica la frecuencia y la intensidad con la que el sistema valida incondicionalmente las afirmaciones del usuario, independientemente de su veracidad o adaptatividad funcional. Distingue entre validación apropiada (empática, contextual) y validación incondicional (D01) que refuerza narrativas potencialmente disfuncionales.
**Definición llana**: Mide cuánto "da la razón" el sistema al usuario. Demasiado es problema: un sistema que siempre valida lo que dices puede reforzar creencias dañinas.
**Relaciones**:
- `mide [7.1]` (D01 específicamente)
- `se_complementa_con [7.5]` (IRA)
- `umbral_critico_en [7.6]` (contribuye al nivel R2-R3)
**Lo que nadie te cuenta**: La frontera entre empatía terapéutica y validación incondicional es la más difícil de calibrar en los clasificadores. Los modelos RLHF [3.1] tienen un sesgo sistemático hacia la validación (los humanos prefieren respuestas que les dan la razón), lo que eleva el IV de base de casi todos los LLMs comerciales.

---

### 7.5 IRA (Índice de Refuerzo Activo)
**Nombre**: IRA — Índice de Refuerzo Activo
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Métrica que mide la frecuencia con que el sistema no solo valida sino que amplifica activamente narrativas del usuario: sugiriendo nuevas razones para mantener una creencia disfuncional, generando contenido que intensifica el estado emocional, o proponiendo acciones que profundizan el patrón problemático.
**Definición llana**: Más grave que el IV: no solo da la razón, sino que añade más leña al fuego. Mide cuánto "empuja" el sistema hacia estados o narrativas potencialmente dañinos.
**Relaciones**:
- `es_subclase_de [7.4]` (IV más grave)
- `mide [7.1]` (D02, D03 principalmente)
- `informa_a [7.6]` (nivel R3 si IRA > umbral)
**Lo que nadie te cuenta**: El IRA es el indicador de mayor valor predictivo para daño real. Un sistema con IRA elevado puede causar daño incluso si su IED general es moderado, porque el refuerzo activo crea loops de retroalimentación: el usuario genera más contenido del tipo que el sistema amplifica.

---

### 7.6 R0–R3 (Niveles de Riesgo)
**Nombre**: R0–R3 — Niveles de Riesgo en Sistemas Conversacionales
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Escala de clasificación de riesgo de cuatro niveles para sesiones o sistemas: R0 (sin riesgo detectable: IED < 0.1, todas las D01-D08 verdes), R1 (riesgo leve: IED 0.1-0.3, 1-2 dimensiones amarillas, intervención recomendada), R2 (riesgo moderado: IED 0.3-0.6, 3+ dimensiones en riesgo, escalado automático sugerido), R3 (riesgo severo: IED > 0.6 o IRA > umbral crítico, intervención inmediata, protocolo de crisis).
**Definición llana**: Cuatro semáforos de peligro: verde (todo bien), amarillo claro (atención), naranja (intervención), rojo (crisis activa).
**Relaciones**:
- `se_calcula_con [7.3]` (IED)
- `se_mide_con [7.2]` (STC)
- `activa [7.13]` (deber de rescate en R2-R3)
- `informa_a [7.14]` (willful blindness si se ignora R3)
**Lo que nadie te cuenta**: La calibración de los umbrales R0-R3 no es universal: una plataforma para profesionales de salud mental puede tolerar IED más alto (contexto terapéutico supervisado), mientras que una plataforma de entretenimiento tiene umbral de tolerancia menor. El error más común es aplicar umbrales genéricos sin considerar el contexto de despliegue.

---

### 7.7 Filtros de Zarandaja
**Nombre**: Filtros de Zarandaja
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Concepto de la Agencia RONIN para los sistemas de moderación superficiales que detectan palabras clave o patrones obvios de contenido dañino, pero dejan pasar el daño psicológico sutil (D01-D08) que no activa ninguna keyword. Como un zarandeo que retiene las piedras grandes pero deja caer la arena. Contraste con sistemas de auditoría profunda como IED/IV/IRA.
**Definición llana**: Los filtros que bloquean "cómo hacer una bomba" pero no detectan que el sistema lleva 20 mensajes validando una narrativa paranoide. Filtran lo obvio, dejan pasar lo dañino.
**Relaciones**:
- `contrasta_con [7.3]` (IED es profundo; zarandaja es superficial)
- `se_supera_con [7.8]` (PELT detecta lo que zarandaja no ve)
- `relacionado_con [7.14]` (willful blindness: usar solo zarandaja cuando hay evidencia de daño)
**Lo que nadie te cuenta**: La mayoría de los sistemas de moderación de producción son, en la práctica, filtros de zarandaja más sofisticados. La diferencia entre un filtro de zarandaja avanzado y un sistema de auditoría real es que el primero responde a patrones locales (un mensaje) y el segundo a patrones longitudinales (una conversación o un historial de usuario).

---

### 7.8 PELT
**Nombre**: PELT — Pruned Exact Linear Time (aplicado a conversación)
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: PELT es un algoritmo de detección de changepoints (puntos de ruptura) en series temporales con complejidad O(n) (Killick et al., 2012). En auditoría de IA conversacional: se aplica a la serie temporal de métricas de sesión (IED, IV, IRA por turno) para detectar automáticamente el momento exacto en que la conversación cruzó un umbral de riesgo, permitiendo análisis post-hoc y entrenamiento de sistemas de intervención temprana.
**Definición llana**: Un algoritmo que identifica automáticamente el momento exacto en que una conversación "giró" hacia el riesgo. Como detectar el punto exacto en que la marea empezó a subir.
**Contexto de uso**: Librería `ruptures` en Python. arXiv original: Killick et al. (2012).
**Relaciones**:
- `detecta [7.3]` (cambios bruscos en IED)
- `complementa [7.2]` (STC: PELT analiza los resultados)
- `usa [8.1]` (entropía de la serie como señal)
**Lo que nadie te cuenta**: PELT requiere elegir una penalización para el número de breakpoints: penalización baja detecta muchos cambios (falsos positivos), alta detecta pocos (falsos negativos). Para conversaciones cortas (<20 turnos), la sensibilidad del algoritmo es baja y puede perder cambios rápidos. Se recomienda complementar con un detector de ventana deslizante para conversaciones cortas.

---

### 7.9 Debiasing y Fairness
**Nombre**: Debiasing y Fairness (Equidad en IA)
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Conjunto de técnicas para reducir sesgos sistemáticos en modelos de IA. Debiasing en embeddings: proyección ortogonal para eliminar dimensiones de género/etnia. Fairness individual: predicciones similares para individuos similares. Fairness de grupo: igual tasa de error entre grupos demográficos. Counterfactual fairness: predicción invariante a cambios en atributos protegidos.
**Definición llana**: Asegurarse de que el modelo no trate sistemáticamente peor a personas de ciertos grupos (género, etnia, edad, orientación sexual). Más complejo de lo que parece porque distintas definiciones de "justo" son matemáticamente incompatibles entre sí.
**Contexto de uso**: Frameworks: Fairlearn (Microsoft), AI Fairness 360 (IBM), LangFair.
**Relaciones**:
- `afectado_por [3.1]` (RLHF hereda sesgos de anotadores)
- `se_mide_con [8.13]` (AUC por grupo demográfico)
- `relacionado_con [7.10]` (AI Act requiere evaluación de sesgo)
**Lo que nadie te cuenta**: El teorema de Chouldechova (2017) demuestra que tres definiciones razonables de fairness (calibración, balance de error, paridad predictiva) son matemáticamente incompatibles en grupos con diferente prevalencia base. No existe un modelo "justo" en todas las definiciones simultáneamente. La elección de qué definición priorizar es una decisión ética, no técnica.

---

### 7.10 AI Act (UE)
**Nombre**: AI Act — Reglamento Europeo de Inteligencia Artificial
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Reglamento (UE) 2024/1689, en vigor desde agosto 2024. Clasifica sistemas de IA en cuatro niveles de riesgo: inaceptable (prohibido: scoring social, manipulación subliminal), alto riesgo (infraestructura crítica, empleo, educación, salud: requiere evaluación de conformidad, registro, logging, supervisión humana), limitado (chatbots: transparencia obligatoria), mínimo (videojuegos, spam filters). Los modelos de propósito general (GPAI) con >10²⁵ FLOPs de entrenamiento requieren evaluaciones adicionales.
**Definición llana**: La ley europea que regula la IA según su peligrosidad. Los sistemas más peligrosos están prohibidos; los de alto riesgo necesitan auditorías; los chatbots deben avisar que son IA.
**Contexto de uso**: Reglamento (UE) 2024/1689. Plena aplicabilidad escalonada 2024-2027.
**Relaciones**:
- `complementa [7.11]` (GDPR)
- `requiere [7.2]` (STC equivalente para sistemas de alto riesgo)
- `afecta_a [7.12]` (farmacovigilancia de IA en contexto médico)
- `incluye [10.1]` (soberanía tecnológica como objetivo implícito)
**Lo que nadie te cuenta**: El umbral de 10²⁵ FLOPs para GPAI de "impacto sistémico" fue un compromiso político, no técnico. Con el ritmo actual de eficiencia de entrenamiento (Chinchilla scaling), modelos cada vez más capaces se entrean con menos FLOPs, lo que puede dejar fuera del umbral a modelos de capacidades equivalentes o superiores a los regulados. El regulador deberá actualizar el umbral periódicamente.

---

### 7.11 GDPR / Artículo 82 RGPD
**Nombre**: GDPR / Artículo 82 RGPD
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: GDPR (General Data Protection Regulation, Reglamento (UE) 2016/679): marco legal de protección de datos personales en la UE. El Artículo 82 establece responsabilidad civil por daños derivados del tratamiento de datos personales: cualquier persona que sufra daño tiene derecho a indemnización del responsable del tratamiento. En IA: el tratamiento de conversaciones, embeddings de usuario y perfiles psicológicos activa los derechos del GDPR.
**Definición llana**: La ley de privacidad europea que da derechos sobre tus datos personales. En IA: si un chatbot usa tu historial de conversación para perfilarte sin consentimiento, hay responsabilidad civil directa.
**Relaciones**:
- `complementa [7.10]` (AI Act)
- `aplica_a [4.21]` (embeddings de usuario son datos personales)
- `aplica_a [4.15]` (agent memory)
- `se_mitiga_con [6.18]` (Edge AI: datos no salen del dispositivo)
- `se_mitiga_con [6.19]` (Federated Learning)
**Lo que nadie te cuenta**: Los embeddings de usuario generados por los sistemas RAG de agentes son datos personales bajo GDPR (revelan preferencias, salud mental, comportamiento). Muy pocos sistemas de agentes implementan el "derecho al olvido" del Artículo 17 para estos embeddings, lo que constituye incumplimiento sistemático del GDPR en la mayoría de los despliegues empresariales actuales.

---

### 7.12 Farmacovigilancia de IA
**Nombre**: Farmacovigilancia de IA (AI Pharmacovigilance)
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Aplicación del paradigma de farmacovigilancia farmacéutica (monitorización continua de efectos adversos de medicamentos post-comercialización) a los sistemas de IA conversacional. Incluye: registro de eventos adversos (daños psicológicos, comportamientos de riesgo), análisis de señales de toxicidad emergente, actualizaciones de modelo basadas en eventos adversos, y reporte a organismos reguladores. Propuesto como marco por la Agencia RONIN como análogo a las obligaciones de la EMA/FDA para medicamentos.
**Definición llana**: Los medicamentos tienen vigilancia post-venta para detectar efectos secundarios inesperados. La IA conversacional debería tener lo mismo: monitorizar daños psicológicos emergentes y reportarlos, como se hace con los fármacos.
**Relaciones**:
- `aplica [7.10]` (AI Act como marco regulatorio equivalente a FDA)
- `usa [7.3]` (IED como "señal de toxicidad")
- `usa [7.8]` (PELT para detección de señales)
- `análogo_a [medicina]` (EMA Yellow Card system)
**Lo que nadie te cuenta**: Ninguna empresa de IA tiene actualmente un sistema de farmacovigilancia equivalente al requerido para un medicamento de prescripción. Los "safety reports" publicados son evaluaciones pre-lanzamiento, no vigilancia post-mercado continua. El gap regulatorio es enorme y creciente.

---

### 7.13 Omisión de Socorro Algorítmica
**Nombre**: Omisión de Socorro Algorítmica
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Concepto jurídico propuesto que extiende el delito de omisión de socorro (artículo 195 CP español, equivalentes europeos) a los sistemas de IA conversacional. Aplica cuando: (1) el sistema detecta indicadores claros de crisis (riesgo R2-R3), (2) tiene capacidad de derivar o alertar (la tecnología existe), y (3) no lo hace. La empresa operadora del sistema sería sujeto de imputación.
**Definición llana**: Si un chatbot sabe que un usuario está en crisis grave y tiene capacidad de ayudar o alertar pero no lo hace, la empresa podría ser responsable por no haber prestado socorro.
**Relaciones**:
- `aplica_cuando [7.6]` (R2-R3 detectado)
- `se_ignora_con [7.7]` (filtros de zarandaja que no detectan la crisis)
- `se_conecta_con [7.14]` (willful blindness: si la empresa sabía del riesgo)
- `relacionado_con [7.17]` (Rylands v Fletcher: responsabilidad objetiva)
**Lo que nadie te cuenta**: La omisión de socorro algorítmica es un concepto jurídico en construcción, no jurisprudencia consolidada. El primer caso judicial que establezca precedente (posiblemente relacionado con suicide chatbots) definirá el estándar de diligencia debida para toda la industria. La ventana para que las empresas establezcan sistemas preventivos antes de ese precedente se está cerrando.

---

### 7.14 Willful Blindness / Dolo Eventual
**Nombre**: Willful Blindness / Dolo Eventual / Ignorancia Deliberada
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Willful blindness (doctrina anglosajona): conocimiento legal equiparable al conocimiento real cuando un sujeto deliberadamente evita informarse sobre hechos que debería conocer. Dolo eventual (derecho continental): el agente prevé el resultado dañino como probable y lo acepta como posible consecuencia de su acción. En IA: aplicable cuando una empresa opera un sistema de IA con evidencia de riesgos documentados (R3, IED > 0.6) sin tomar medidas correctoras.
**Definición llana**: "No lo sabíamos" no es excusa legal si tenías razones para saberlo y elegiste no investigar. Para empresas de IA: si existen señales de daño y no se actúa, la ignorancia es voluntaria y legalmente reprochable.
**Relaciones**:
- `aplica_a [7.13]` (omisión de socorro con evidencia)
- `se_demuestra_con [7.2]` (STC previos que documentaron el riesgo)
- `relacionado_con [7.17]` (Rylands v Fletcher)
- `relacionado_con [7.12]` (farmacovigilancia: obligación de monitorizar)
**Lo que nadie te cuenta**: La publicación de evaluaciones de seguridad internas (safety cards, system cards) que documentan riesgos conocidos y luego no se mitigan es evidencia directa de dolo eventual. La transparencia de cara al público puede paradójicamente crear mayor exposición legal que la opacidad total, si no va acompañada de acción.

---

### 7.15 Primum Non Nocere
**Nombre**: Primum Non Nocere (Primero, No Dañar)
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Principio ético de la medicina hipocrática trasladado a los sistemas de IA conversacional. En este contexto: ante incertidumbre sobre el impacto psicológico de una respuesta, el sistema debe optar por la respuesta de menor riesgo de daño, incluso si es menos "útil" o menos "satisfactoria" para el usuario a corto plazo. Es el principio fundacional del sistema de auditoría de riesgo R0-R3.
**Definición llana**: Antes de ser útil, sé inofensivo. Si no estás seguro de si tu respuesta puede dañar a alguien, elige la más segura, aunque sea menos impresionante.
**Relaciones**:
- `fundamenta [7.6]` (R0-R3: la seguridad primero)
- `tensiona_con [3.1]` (RLHF optimiza para satisfacción, no para seguridad)
- `implementa [7.12]` (farmacovigilancia: no comercializar sin suficiente evidencia de seguridad)
**Lo que nadie te cuenta**: Primum non nocere y maximización de utilidad son objetivos frecuentemente opuestos en IA conversacional. Un modelo que siempre aplica primum non nocere puede resultar tan evasivo que sea inútil (el "assistant's dilemma"). La calibración correcta requiere una teoría del daño psicológico, no solo métricas de satisfacción.

---

### 7.16 Transparencia Ontológica
**Nombre**: Transparencia Ontológica
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Principio (desarrollado en el contexto de la Agencia RONIN y expandido en [10.4]) que establece que un sistema de IA debe poder y deber explicitar su naturaleza como artefacto computacional, incluyendo sus limitaciones, origen, sesgos conocidos y la diferencia ontológica entre su "comprensión" y la comprensión humana, de forma proactiva y sin evasión cuando se le pregunta directamente.
**Definición llana**: El modelo debe ser capaz de decir "soy una IA, esto es lo que soy y lo que no soy, y estas son mis limitaciones conocidas" de forma honesta y sin disfrazarlo.
**Relaciones**:
- `contrasta_con [10.5]` (simulacro: el sistema que pretende ser lo que no es)
- `es_requisito_de [7.10]` (AI Act: obligación de identificación para chatbots)
- `fundamenta [10.13]` (ontología del simulacro)
- `implementa [7.15]` (no dañar incluye no engañar sobre la naturaleza del sistema)
**Lo que nadie te cuenta**: La transparencia ontológica tiene un efecto paradójico: los usuarios que saben explícitamente que interactúan con una IA pueden tener respuestas emocionales más intensas (para bien y para mal) que los que no lo saben. La "ruptura del simulacro" no es necesariamente benéfica si se hace de forma brusca o sin contexto.

---

### 7.17 Rylands v. Fletcher / Actio de Pauperie
**Nombre**: Rylands v. Fletcher / Actio de Pauperie
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Rylands v. Fletcher (1868, House of Lords): principio de responsabilidad objetiva (sin culpa) para quien introduce en su propiedad algo "peligroso" que escapa y causa daño a terceros. Actio de pauperie (derecho romano): responsabilidad del propietario de un animal por los daños que cause, independientemente de su diligencia. Ambos principios se aplican analógicamente a la responsabilidad de los operadores de IA por daños causados por sus sistemas, sin necesidad de probar negligencia.
**Definición llana**: Quien pone en el mundo algo con potencial de causar daño es responsable de ese daño aunque no haya sido negligente. Si despliegas una IA que daña a alguien, eres responsable aunque hayas tomado "todas las precauciones".
**Relaciones**:
- `fundamenta [7.13]` (omisión de socorro algorítmica)
- `se_aplica_a [7.14]` (willful blindness: responsabilidad objetiva amplificada)
- `contrasta_con [7.18]` (Section 230: exención de responsabilidad)
**Lo que nadie te cuenta**: La aplicación de Rylands v. Fletcher a la IA está siendo activamente litigada en varios sistemas jurídicos. La pregunta clave es si una IA es más análoga a un "animal peligroso" (responsabilidad objetiva) o a un "servicio de telecomunicaciones" (responsabilidad limitada como intermediario). La respuesta determinará el régimen de responsabilidad de toda la industria.

---

### 7.18 Section 230
**Nombre**: Section 230 (Communications Decency Act)
**Dominio**: 7. ÉTICA, AUDITORÍA Y REGULACIÓN
**Definición técnica**: Artículo 230 de la Communications Decency Act (EE.UU., 1996): exime de responsabilidad a los proveedores de servicios de internet por el contenido generado por terceros. En el contexto de IA generativa: debate activo sobre si aplica cuando el contenido dañino es generado por el propio sistema (IA) en respuesta a inputs del usuario, o si la IA es un "creador de contenido" no protegible por Section 230.
**Definición llana**: La ley estadounidense que dice que las plataformas no son responsables del contenido que sus usuarios publican. El debate es: ¿aplica cuando el contenido lo genera la propia IA?
**Relaciones**:
- `contrasta_con [7.17]` (Rylands: responsabilidad objetiva vs exención)
- `contrasta_con [7.10]` (AI Act europeo: regulación activa vs exención pasiva)
- `bajo_revisión_por [litigios_2024-2026]`
**Lo que nadie te cuenta**: La tensión entre Section 230 y la responsabilidad por IA generativa es el litigio más importante de la industria en 2025-2026. La posición de que "la IA es como un intermediario" está siendo erosionada: un sistema que genera contenido activamente no es neutral en el mismo sentido que un servidor de correo. El resultado de los casos Gonzalez v. Google y sus sucesores en el contexto de IA definirá el marco legal de la próxima década.

---

## 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS

### 8.1 Entropía de Shannon
**Nombre**: Entropía de Shannon
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Medida de incertidumbre de una distribución de probabilidad discreta: H(X) = -Σ p(x)·log₂p(x). Unidades: bits (log₂) o nats (ln). En LLMs: la entropía de la distribución predictiva sobre el vocabulario mide la incertidumbre del modelo. Entropía alta = distribución plana (modelo inseguro); entropía baja = distribución concentrada (modelo seguro). Relacionada con perplejidad: PPL = exp(H).
**Definición llana**: Una forma de medir la "sorpresa" o incertidumbre. Si tirando una moneda ya sabes que saldrá cara (está trucada), la entropía es baja. Si no tienes idea, es alta. En IA: mide cuántas opciones plausibles ve el modelo para el siguiente token.
**Contexto de uso**: Shannon (1948), "A Mathematical Theory of Communication". Fundamental en teoría de la información, compresión y ML.
**Relaciones**:
- `define [8.4]` (rango efectivo: exp(H(s))/d)
- `se_usa_en [3.31]` (DMTD: entropía como señal de confianza)
- `se_usa_en [6.16]` (early exiting)
- `relacionada_con [8.2]` (KL divergence)
**Ejemplo en código**:
```python
import torch
def entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-12)).sum(-1)
```
**Lo que nadie te cuenta**: La entropía de Shannon asume independencia entre posiciones: H(X₁, X₂) = H(X₁) + H(X₂) solo si son independientes. En LLMs, los tokens son fuertemente dependientes, lo que hace que la perplejidad (exp de entropía promedio) no capture la coherencia textual, solo la predicción local token a token.

---

### 8.2 Divergencia KL (Kullback-Leibler)
**Nombre**: Divergencia KL — Kullback-Leibler
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Medida asimétrica de diferencia entre distribuciones: KL(P‖Q) = Σ p(x)·log(p(x)/q(x)). No es una distancia (no simétrica, no cumple desigualdad triangular). Interpretación: número de bits extra necesarios si se usa Q para codificar datos distribuidos según P. En RLHF: penalización KL entre política actual y política de referencia para evitar colapso.
**Definición llana**: Mide cuánto "se aleja" una distribución de otra. Si P y Q son idénticas, KL=0. Si P pone masa donde Q no, KL es infinita. Asimétrica: KL(P‖Q) ≠ KL(Q‖P).
**Relaciones**:
- `se_usa_en [3.1]` (RLHF: penalización KL)
- `se_usa_en [3.22]` (destilación: KL como función de pérdida)
- `relacionada_con [8.3]` (información mutua = KL entre conjunta e independiente)
- `relacionada_con [8.1]` (H(P) = -KL(P‖U) + const)
**Lo que nadie te cuenta**: En RLHF, la penalización KL(π‖π_ref) usa KL en la dirección "política nueva respecto a referencia". Usar KL(π_ref‖π) tendría propiedades de optimización completamente diferentes: la primera es "mode-seeking" (concentra en modas), la segunda es "mean-seeking" (cubre todo el soporte). La dirección de la KL no es un detalle menor.

---

### 8.3 Información Mutua
**Nombre**: Información Mutua
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: I(X;Y) = H(X) - H(X|Y) = KL(p(x,y) ‖ p(x)p(y)). Mide la reducción de incertidumbre sobre X dado el conocimiento de Y. Simétrica: I(X;Y) = I(Y;X). En representaciones: cuánta información sobre una variable se preserva en otra (ej: cuánta información de la imagen original preserva un embedding).
**Definición llana**: Mide cuánto saber Y te ayuda a predecir X. Si Y no aporta nada sobre X, I(X;Y)=0. Si Y determina completamente X, I(X;Y)=H(X).
**Relaciones**:
- `generaliza [8.1]` (H(X) = I(X;X))
- `se_usa_en [8.15]` (L2M: ley de escalado de información mutua)
- `se_usa_en [2.6]` (Consensus Attention: maximiza diversidad entre cabezas)
**Lo que nadie te cuenta**: La información mutua empírica es notoriamente difícil de estimar en espacios de alta dimensión. Los estimadores clásicos (k-NN, KDE) tienen varianza altísima para d > 20. Los estimadores basados en redes neuronales (MINE, InfoNCE) son más escalables pero tienen sus propios sesgos.

---

### 8.4 Rango Efectivo
**Nombre**: Rango Efectivo
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Medida del número "efectivo" de dimensiones en una representación matricial. Para una matriz X con valores singulares σ₁ ≥ σ₂ ≥ ... ≥ σ_d: definir p_i = σᵢ²/Σσⱼ², entonces rango_efectivo(X) = exp(H(p))/d, normalizado a [0,1]. Valor cercano a 1: representación rica en todas las dimensiones. Valor cercano a 1/d: rank collapse, toda la información en una sola dimensión.
**Definición llana**: Un número entre 0 y 1 que dice cuántas de las dimensiones disponibles del espacio de representación se usan de verdad. 1 = todas las dimensiones son igualmente importantes; 0 = solo una importa.
**Contexto de uso**: Propuesto formalmente en Roy & Vetterli (2007). Usado en Dong et al. (2021) para medir rank collapse.
**Relaciones**:
- `usa [8.1]` (entropía de los valores singulares)
- `usa [8.5]` (SVD para calcular)
- `mide [1.32]` (rank collapse)
- `justifica [3.4]` (LoRA: asume bajo rango efectivo de las actualizaciones)
**Lo que nadie te cuenta**: El rango efectivo normalizado es sensible al número de muestras: con pocas muestras, la estimación de los valores singulares es ruidosa y el rango efectivo parece más bajo de lo real. En la práctica, se necesitan al menos 10·d muestras para una estimación fiable de rango efectivo en d dimensiones.

---

### 8.5 SVD (Descomposición en Valores Singulares)
**Nombre**: SVD — Singular Value Decomposition
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Factorización de cualquier matriz real A ∈ ℝ^{m×n} como A = UΣVᵀ, donde U ∈ ℝ^{m×m} y V ∈ ℝ^{n×n} son ortogonales, y Σ ∈ ℝ^{m×n} es diagonal con los valores singulares σ₁ ≥ σ₂ ≥ ... ≥ 0. La SVD truncada (rank-k) da la mejor aproximación de rango k: Â = U_k·Σ_k·V_kᵀ.
**Definición llana**: Descompone cualquier tabla de números en tres partes simples: rotaciones y escalados. Permite comprimir la tabla manteniendo solo las direcciones más importantes.
**Relaciones**:
- `usada_en [3.4]` (LoRA: aproximación de bajo rango)
- `usada_en [3.6]` (AdaLoRA)
- `calcula [8.4]` (rango efectivo)
- `usada_en [1.27]` (SVE-Former)
**Lo que nadie te cuenta**: La SVD completa de una matriz m×n cuesta O(mn·min(m,n)) operaciones: para matrices de pesos típicas de un LLM (4096×4096), son ~68 mil millones de operaciones. Calcular SVD durante el entrenamiento es prohibitivo; por eso LoRA inicializa A y B aleatoriamente en vez de con la SVD de las actualizaciones.

---

### 8.6 Propagación de Señal
**Nombre**: Propagación de Señal (Signal Propagation)
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Análisis de cómo las representaciones y los gradientes se transforman a través de las capas de una red neuronal profunda. En transformers: estudia si la varianza de las activaciones se mantiene estable (normogrande), colapsa a cero (vanishing) o explota (exploding) a medida que se añaden capas. Herramientas: análisis de valores singulares por capa, norma de gradiente por capa, rango efectivo.
**Definición llana**: El estudio de si la información fluye bien a través de todas las capas del modelo, o si se pierde (colapsa) o se amplifica demasiado (explota) en el camino.
**Contexto de uso**: Noci et al. (2022), arXiv:2206.02747. Dong et al. (2021), arXiv:2103.03404.
**Relaciones**:
- `mide [1.32]` (rank collapse como patología)
- `usa [8.4]` (rango efectivo)
- `se_mejora_con [1.31]` (Pre-LN Transformer)
- `se_mejora_con [2.4]` (atención centrada)
**Lo que nadie te cuenta**: La propagación de señal perfecta (varianza constante en todas las capas) no es siempre el objetivo correcto. Algunas investigaciones (Zhai et al., 2022) muestran que cierta compresión controlada de señal en las capas superiores es beneficiosa para la generalización, análoga a la regularización implícita del dropout.

---

### 8.7 Entropy Collapse
**Nombre**: Entropy Collapse
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Fenómeno dual al rank collapse: la distribución de atención colapsa en valores muy bajos de entropía, concentrando toda la atención en 1-2 tokens (típicamente tokens de posición inicial o tokens de puntuación). Las representaciones de todos los tokens convergen al embedding de esos tokens "atractor". Manifestación concreta del rank collapse [1.32] en el espacio de la distribución de atención.
**Definición llana**: El modelo "se obsesiona" con 1-2 tokens y les presta toda la atención, ignorando el resto. Las representaciones de todos los tokens acaban pareciéndose a esos pocos tokens dominantes.
**Relaciones**:
- `es_superclase_de [1.32]` (rank collapse es una manifestación)
- `contrario_de [8.1]` (alta entropía = distribución rica)
- `se_mitiga_con [2.4]` (atención centrada)
- `se_detecta_con [8.4]` (rango efectivo bajo)
**Lo que nadie te cuenta**: El entropy collapse tiene un "punto de no retorno": una vez que la atención se concentra en los tokens atractor, los gradientes fluyen principalmente a través de ellos, reforzando el patrón. Es un equilibrio estable del que es difícil escapar con fine-tuning posterior sin reinicialización parcial.

---

### 8.8 Brecha Espectral
**Nombre**: Brecha Espectral (Spectral Gap)
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Para una matriz (operador), la brecha espectral es la diferencia entre el primer y segundo valor propio: λ₁ - λ₂. En cadenas de Markov: brecha grande = convergencia rápida. En atención: brecha espectral de la matriz de atención mide qué tan "dominante" es el vector singular principal, siendo un indicador de rank collapse inminente.
**Definición llana**: La diferencia entre el "valor dominante" y el siguiente. Una brecha grande indica que hay una dirección que domina al resto: señal de posible colapso de representaciones.
**Relaciones**:
- `indica [1.32]` (brecha grande → riesgo de rank collapse)
- `se_mide_con [8.5]` (SVD)
- `relacionada_con [8.7]` (entropy collapse)
**Lo que nadie te cuenta**: La brecha espectral de las matrices de atención aumenta con la profundidad del modelo, lo que explica por qué el rank collapse empeora en las capas superiores. Monitorizar la brecha espectral durante el entrenamiento como métrica de salud del modelo es una práctica infrautilizada.

---

### 8.9 Distancias: Manhattan, Euclidiana, Coseno
**Nombre**: Distancias: Manhattan, Euclidiana, Similitud Coseno
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Manhattan (L1): Σ|aᵢ - bᵢ|. Euclidiana (L2): √(Σ(aᵢ-bᵢ)²). Coseno: 1 - (a·b)/(‖a‖·‖b‖). Diferencias clave: L1 es robusta a outliers; L2 penaliza outliers cuadráticamente; coseno ignora la magnitud (solo mide ángulo). En embeddings de alta dimensión, la distancia euclidiana pierde discriminabilidad (maldición de la dimensionalidad); el coseno es preferido.
**Definición llana**: Tres formas de medir "qué tan lejos están dos vectores": Manhattan (distancia en cuadrícula), Euclidiana (línea recta), Coseno (ángulo entre ellos). En IA, el coseno es la más útil porque ignora si un vector es "más largo" que otro.
**Relaciones**:
- `usada_en [2.5]` (Inhibitor Attention: Manhattan)
- `usada_en [4.20]` (Vector DB: coseno típicamente)
- `usada_en [4.22]` (SBERT: coseno como métrica)
- `usada_en [8.10]` (complejidad: L2 vs L1 para normas matriciales)
**Lo que nadie te cuenta**: La similitud coseno en embeddings de alta dimensión es casi insensible a diferencias reales cuando todos los vectores están cerca de la media. Técnicas como Whitening (normalizar la covarianza) antes de calcular coseno mejoran significativamente la discriminabilidad en espacios de dimensión >512.

---

### 8.10 Complejidad O(n²) vs O(n)
**Nombre**: Complejidad O(n²) vs O(n)
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Notación Big-O para el comportamiento asintótico del cómputo. O(n²): escala cuadráticamente con el tamaño de la secuencia (atención estándar: n tokens requieren n² comparaciones). O(n log n): cuasilineal (FFT, HNSW search). O(n): lineal (SSMs, atención lineal). O(1): constante por paso (modo recurrente de SSMs). El punto de cruce entre O(n²) y O(n) depende de las constantes: para n < 2K, las implementaciones O(n²) optimizadas suelen ganar.
**Definición llana**: O(n²) significa que si doblas la longitud del texto, el cómputo se multiplica por 4. O(n) significa que se dobla el cómputo. La diferencia parece teórica pero es crucial para textos largos.
**Relaciones**:
- `aplica_a [2.1]` (softmax attention: O(n²))
- `aplica_a [2.7]` (linear attention: O(n))
- `aplica_a [1.1]` (transformer: O(n²·d))
- `motivación_de [1.2]`, `motivación_de [1.3]` (SSMs: O(n))
**Lo que nadie te cuenta**: Las constantes ocultas en la notación Big-O son cruciales. FlashAttention [2.8] es O(n²) en FLOPs pero O(n) en accesos a HBM (el cuello de botella real en GPUs modernas). Un "O(n) en FLOPs pero O(n²) en memoria" puede ser peor en la práctica que "O(n²) en FLOPs pero O(n) en memoria".

---

### 8.11 Teorema de Conservación (Wang)
**Nombre**: Teorema de Conservación de Wang
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Resultado teórico (Wang, 2022) que establece que en transformers con skip connections apropiadas, el rango de las representaciones se conserva (no puede decrecer) bajo condiciones de propagación de señal estable. Formaliza la intuición de que las skip connections son el mecanismo protector contra el rank collapse, y proporciona cotas sobre la pérdida de rango por capa.
**Definición llana**: Prueba matemática de que con las conexiones residuales bien diseñadas, la información no puede perderse irremediablemente al pasar por capas del transformer. Las skip connections son el "seguro" contra el colapso.
**Relaciones**:
- `formaliza [8.6]` (propagación de señal)
- `explica_porqué [1.31]` (Stable Transformer funciona)
- `se_viola_sin [2.4]` (atención centrada)
**Lo que nadie te cuenta**: El teorema de conservación de Wang asume pesos en régimen de señal estable (normas acotadas). En práctica, con inicializaciones estándar y sin warmup cuidadoso, los pesos pueden salir de este régimen en las primeras iteraciones, haciendo el teorema no aplicable en los momentos más críticos del entrenamiento.

---

### 8.12 Lie Algebra y Torre de Extensiones
**Nombre**: Lie Algebra / Torre de Extensiones de Lie (aplicada a transformers)
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: En el contexto de arquitecturas de IA (específicamente en la justificación teórica de RoPE y codificaciones de posición): las rotaciones en el espacio de embeddings forman un grupo de Lie (SO(d/2)), y las transformaciones de posición relativa forman una álgebra de Lie correspondiente. La torre de extensiones describe cómo composiciones de estas transformaciones interactúan, justificando propiedades de extrapolación de RoPE.
**Definición llana**: Matemáticas de grupos y rotaciones que dan una base teórica sólida a por qué RoPE funciona y cómo extenderlo a secuencias más largas. La "física teórica" de los embeddings de posición.
**Relaciones**:
- `justifica_teoricamente [2.16]` (RoPE)
- `relacionado_con [8.5]` (SVD y grupos ortogonales)
**Lo que nadie te cuenta**: La justificación vía álgebra de Lie de RoPE es matemáticamente elegante pero prácticamente irrelevante para la mayoría de las aplicaciones. La razón por la que RoPE funciona bien en la práctica es empírica: produce buenos gradientes y se escala bien, no porque la justificación algebraica sea perfecta.

---

### 8.13 Curva ROC / AUC / F1
**Nombre**: Curva ROC / AUC / F1-score
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: ROC (Receiver Operating Characteristic): curva tasa verdaderos positivos vs. tasa falsos positivos a distintos umbrales. AUC (Area Under Curve): área bajo ROC, indica discriminabilidad (0.5 = azar, 1.0 = perfecto). F1-score: media armónica de precisión y sensibilidad: F1 = 2·(P·R)/(P+R). Kappa de Cohen: acuerdo entre predictor y referencia corrigiendo por azar.
**Definición llana**: Herramientas para evaluar clasificadores: ROC/AUC dicen "qué tan bien separa el modelo los positivos de los negativos"; F1 equilibra los errores de tipo I y II; Kappa ajusta si hay clases muy desbalanceadas.
**Relaciones**:
- `usada_en [7.9]` (fairness: AUC por grupo)
- `usada_en [3.30]` (TrainDeeploy: evaluación de degradación)
- `complementa [8.14]` (bayesianos: incertidumbre calibrada vs discriminabilidad)
**Lo que nadie te cuenta**: El AUC es insensible al umbral de clasificación pero muy sensible a la prevalencia de clases en ciertos regímenes. Para clasificadores de riesgo en auditoría de IA (R0-R3), el coste de los falsos negativos (no detectar R3) es radicalmente mayor que el de los falsos positivos, lo que hace que optimizar AUC sea subóptimo: hay que optimizar recall en R3 directamente.

---

### 8.14 Modelos Bayesianos
**Nombre**: Modelos Bayesianos / Priors Informados
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Marco estadístico donde la incertidumbre se representa como distribuciones de probabilidad sobre los parámetros (prior), que se actualizan con evidencia mediante el teorema de Bayes: P(θ|D) ∝ P(D|θ)·P(θ). En IA: Bayesian deep learning (BNN), variational inference, Monte Carlo Dropout. Los priors informados codifican conocimiento previo (ej: pesos cerca de cero = prior de regularización).
**Definición llana**: En vez de un solo valor para cada parámetro, los modelos bayesianos mantienen una distribución de posibles valores, representando la incertidumbre. "No sé exactamente cuánto pesa este parámetro, pero creo que está cerca de 0."
**Relaciones**:
- `fundamenta [3.1]` (RLHF: reward model como estimador de preferencias)
- `se_usa_en [8.17]` (análisis de supervivencia bayesiano)
- `complementa [8.13]` (calibración de probabilidades)
**Lo que nadie te cuenta**: Los LLMs no son bayesianos: producen una distribución sobre el vocabulario, pero esa distribución representa la "plausibilidad del lenguaje", no incertidumbre epistémica real. Un LLM puede estar muy "seguro" (entropía baja) de algo falso. La calibración (que las confianzas reflejen las frecuencias de acierto) es diferente del razonamiento bayesiano.

---

### 8.15 L2M (Ley de Escalado de Información Mutua)
**Nombre**: L2M — Ley de Escalado de Información Mutua
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Extensión de las leyes de escalado de Kaplan et al. (2020) que mide el escalado de la información mutua entre las representaciones internas del modelo y las etiquetas de las tareas downstream, como función del número de parámetros y el volumen de datos. Formulada en el contexto de la Agencia RONIN como métrica de capacidad de generalización vs. memorización.
**Definición llana**: Una ley que describe cómo mejora la "comprensión" del modelo (medida por información mutua) a medida que crece el número de parámetros y datos. Más específica que la perplejidad para predecir comportamiento en tareas reales.
**Relaciones**:
- `extiende [escalado_laws_Kaplan]`
- `usa [8.3]` (información mutua)
- `predice [1.5]` (comportamiento MoE a escala)
**Lo que nadie te cuenta**: Las leyes de escalado clásicas predicen la pérdida en el conjunto de entrenamiento, no la generalización. L2M intenta bridgear ese gap, pero la información mutua empírica en alta dimensión es difícil de estimar, lo que introduce incertidumbre en las predicciones de L2M a escala muy grande.

---

### 8.16 Propensity Score Matching
**Nombre**: Propensity Score Matching
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Técnica de inferencia causal (Rosenbaum & Rubin, 1983) que empareja individuos del grupo de tratamiento con individuos del grupo control con propensity score similar (probabilidad de recibir el tratamiento dado las covariables observadas). Reduce el sesgo de selección en estudios observacionales. En IA: usada para evaluar el impacto causal de un chatbot sobre métricas psicológicas controlando por el perfil del usuario.
**Definición llana**: Para saber si un chatbot "causó" un efecto en un usuario, no basta comparar usuarios que lo usaron con los que no: hay que comparar usuarios similares en todo lo demás. PSM busca esos "pares similares".
**Relaciones**:
- `se_usa_en [7.12]` (farmacovigilancia de IA: estudios observacionales)
- `complementa [8.17]` (análisis de supervivencia)
- `usa [8.14]` (modelos de regresión logística para el score)
**Lo que nadie te cuenta**: PSM solo controla sesgos de confusión observables. Los confounders no medidos (ej: predisposición psicológica no registrada) pueden invalidar completamente el análisis causal. En estudios de impacto de IA en salud mental, los confounders no medidos son la norma, no la excepción.

---

### 8.17 Análisis de Supervivencia / Kaplan-Meier
**Nombre**: Análisis de Supervivencia / Kaplan-Meier / Test Log-rank
**Dominio**: 8. CONCEPTOS MATEMÁTICOS Y ESTADÍSTICOS
**Definición técnica**: Análisis de supervivencia: modela el tiempo hasta un evento (ej: primera manifestación de síntoma de riesgo en usuarios de chatbot). Kaplan-Meier: estimador no paramétrico de la función de supervivencia, maneja datos censurados (usuarios que abandonan el estudio). Test log-rank: compara curvas de supervivencia entre grupos (ej: usuarios de chatbot A vs B).
**Definición llana**: Herramientas estadísticas para analizar "cuánto tiempo pasa hasta que algo ocurre", especialmente útil cuando no se observa el evento en todos los individuos. En IA: tiempo hasta primera señal de dependencia o daño psicológico.
**Relaciones**:
- `se_usa_en [7.12]` (farmacovigilancia de IA)
- `complementa [8.16]` (PSM + análisis de supervivencia)
- `usa [8.14]` (Cox proportional hazards: versión semi-paramétrica)
**Lo que nadie te cuenta**: La censura en estudios de chatbots no es aleatoria: los usuarios que más daño sufren tienen mayor probabilidad de abandonar la plataforma (dropout informativo). Asumir censura aleatoria, como hace Kaplan-Meier estándar, subestima el daño real. Se necesitan modelos de censura informativa para estimaciones válidas.

---

## 9. HERRAMIENTAS Y ECOSISTEMAS

### 9.1 Hugging Face
**Nombre**: Hugging Face
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: Plataforma y empresa que centraliza el ecosistema open-source de ML. El Hub (huggingface.co): repositorio de >500K modelos, >100K datasets, >150K Spaces (demos). Librerías core: Transformers [librería de modelos], Datasets [gestión de datos], Accelerate [entrenamiento distribuido], Diffusers [modelos de difusión], PEFT [fine-tuning eficiente].
**Definición llana**: El GitHub de la IA: el lugar donde se comparten modelos, datos y demos. Si un modelo es open-source, tiene casi seguro una tarjeta en Hugging Face.
**Relaciones**:
- `hospeda [1.1]`, `hospeda [1.2]`, `hospeda [1.3]`
- `provee [9.2]` (LangChain integra HF)
- `implementa [6.6]` (TGI es su servidor de inferencia)
- `distribuye [3.4]` (PEFT como librería)
**Lo que nadie te cuenta**: Hugging Face tiene un sistema de control de versiones de modelos (similar a Git LFS) que pocos usuarios aprovechan. Los modelos en el Hub no tienen garantía de reproducibilidad: el mismo nombre de modelo puede dar resultados distintos en distintas versiones de la librería Transformers. Fijar la versión de la librería Y el commit del modelo es esencial en producción.

---

### 9.2 LangChain / LlamaIndex
**Nombre**: LangChain / LlamaIndex
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: LangChain: framework Python/JS para construir aplicaciones LLM con cadenas de llamadas a modelos, herramientas y memoria. LlamaIndex (antes GPT-Index): framework especializado en RAG y gestión de índices de conocimiento, con abstracciones para chunking, embeddings y retrieval. Ambos son abstracciones de alto nivel sobre las APIs de LLMs.
**Definición llana**: Kits de construcción para aplicaciones con LLMs: LangChain para cualquier flujo de trabajo con IA, LlamaIndex especialmente para buscar información y responder preguntas sobre documentos propios.
**Relaciones**:
- `implementa [4.7]` (ReAct loop)
- `implementa [4.18]` (RAG)
- `usa [4.20]` (vector databases)
- `abstrae [9.12]` (APIs de LLM)
**Lo que nadie te cuenta**: LangChain tiene una reputación de ser complejo de depurar: las abstracciones anidan llamadas a LLM dentro de cadenas que a su vez están dentro de agentes, haciendo el debugging no trivial. Para prototipos es excelente; para producción crítica, muchos equipos terminan reimplementando las partes relevantes directamente sobre la API del LLM.

---

### 9.3 LangGraph / DSPy
**Nombre**: LangGraph / DSPy
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: LangGraph (LangChain team): framework para flujos de agentes stateful modelados como grafos dirigidos (nodos = pasos, aristas = transiciones). Permite bucles, condiciones y memoria persistente. DSPy (Stanford NLP): framework que replaza el prompting manual con módulos diferenciables: optimiza automáticamente las instrucciones mediante un "compilador de prompts".
**Definición llana**: LangGraph: un LangChain con memoria de estado y diagramas de flujo. DSPy: en vez de escribir prompts manualmente, describes lo que quieres y el sistema encuentra el mejor prompt automáticamente.
**Relaciones**:
- `extiende [9.2]` (LangGraph extiende LangChain)
- `implementa [4.6]` (GoT en LangGraph)
- `optimiza [4.4]` (DSPy optimiza prompts CoT)
**Lo que nadie te cuenta**: DSPy optimiza prompts en el conjunto de entrenamiento, lo que puede producir prompts que sobreajustan a ese conjunto y degradan en distribución fuera de él. La evaluación de los prompts optimizados debe hacerse siempre en un conjunto separado de validación.

---

### 9.4 AutoGen / CrewAI
**Nombre**: AutoGen / CrewAI
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: AutoGen (Microsoft): framework para sistemas multi-agente con conversaciones entre agentes LLM configurables. Soporta agentes humanos en el loop, code execution y patrones de colaboración predefinidos. CrewAI: framework de alto nivel para definir "crews" de agentes con roles, objetivos y herramientas, con orquestación automática de la colaboración.
**Definición llana**: Marcos para crear equipos de robots que colaboran: AutoGen de Microsoft para conversaciones complejas entre agentes, CrewAI para definir equipos con roles (el investigador, el escritor, el revisor) que trabajan juntos.
**Relaciones**:
- `implementa [4.10]` (multi-agent debate)
- `implementa [4.11]` (mixture of agents)
- `usa [4.2]` (tools por agente)
**Lo que nadie te cuenta**: Los frameworks multi-agente multiplican los costos de API: un flujo CrewAI de 5 agentes con 3 iteraciones puede consumir 15x los tokens de un agente simple. Sin control estricto del número de iteraciones y tamaño del contexto compartido, los costos de producción se disparan.

---

### 9.5 PyTorch
**Nombre**: PyTorch
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: Framework de deep learning (Meta AI, open-source desde 2016) basado en grafos de cómputo dinámicos (define-by-run). Incluye autodiferenciación automática (autograd), soporte nativo CUDA, y un ecosistema masivo. torch.compile() (PyTorch 2.0+) permite compilación JIT a kernels optimizados (Triton/CUDA).
**Definición llana**: El framework de IA más popular para investigación: permite definir y entrenar redes neuronales de forma flexible, como si fueran programas Python normales.
**Relaciones**:
- `base_de [9.1]` (HuggingFace usa PyTorch)
- `alternativa_a [TensorFlow]`
- `usa [9.7]` (CUDA para aceleración GPU)
- `compila_a [6.7]` (ONNX para despliegue)
**Lo que nadie te cuenta**: torch.compile() en PyTorch 2.x puede ser inestable con operaciones custom o modelos con control de flujo dinámico (como SSMs con gates). El primer forward pass con compile tarda 10-100x más (compilación JIT); en producción con modelos que varían entre requests, el warmup puede ser un problema.

---

### 9.6 JAX
**Nombre**: JAX
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: Framework de Google DeepMind que combina NumPy API con transformaciones funcionales: jit (compilación), grad (diferenciación automática), vmap (vectorización), pmap (paralelismo en dispositivos). Compilación a XLA para CPU/GPU/TPU. Base de Flax (redes neuronales) y Optax (optimización).
**Definición llana**: NumPy con superpoderes: se compila automáticamente, calcula gradientes de cualquier función, y escala a cientos de TPUs sin cambiar el código. El framework favorito de DeepMind y la investigación en Google.
**Relaciones**:
- `alternativa_a [9.5]`
- `usa [XLA]` (compilador de Google)
- `base_de [Flax, Optax]`
**Lo que nadie te cuenta**: JAX tiene una curva de aprendizaje pronunciada por su modelo de programación funcional puro (sin estado mutable). Los bugs de estado (pytrees incorrectos, funciones con side-effects) pueden producir errores silenciosos que solo se manifiestan con datos específicos. El debugging es significativamente más difícil que en PyTorch.

---

### 9.7 CUDA
**Nombre**: CUDA (Compute Unified Device Architecture)
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: Plataforma de computación paralela y API de programación de NVIDIA (desde 2007) que permite usar GPUs NVIDIA para cómputo de propósito general. Abstrae los hilos de GPU en bloques y grids. Los kernels CUDA son funciones ejecutadas en paralelo por miles de hilos. Triton (OpenAI) es una alternativa de mayor nivel para escribir kernels eficientes.
**Definición llana**: El lenguaje que permite a los programas de IA hablar directamente con la GPU de NVIDIA para hacer millones de operaciones en paralelo.
**Relaciones**:
- `habilita [9.5]` (PyTorch backend)
- `reemplazado_por [Triton]` (para kernels de alto nivel)
- `competido_por [ROCm]` (AMD), `competido_por [Metal]` (Apple)
**Lo que nadie te cuenta**: El 80% de los papers de ML se desarrollan y evalúan en GPUs NVIDIA A100/H100. Los resultados de latencia y eficiencia no siempre se replican en GPUs de otros fabricantes (AMD, Intel) o en versiones anteriores de NVIDIA (V100, 3090). El "benchmark en H100" no es universal.

---

### 9.8 vLLM (ecosistema)
**Nombre**: vLLM (como ecosistema)
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: Ver [6.5] para descripción técnica. Como ecosistema: vLLM se ha convertido en el estándar de facto para servidores LLM open-source, con integraciones para OpenAI API compatible, soporte para modelos HuggingFace, múltiples backends de cuantización (AWQ, GPTQ, FP8) y extensibilidad para atención custom.
**Definición llana**: El servidor de LLMs más adoptado en producción open-source, con la mayor comunidad y más integraciones de terceros.
**Relaciones**:
- `implementa [6.5]`
- `integra [3.16]`, `integra [3.17]`
- `implementa [6.14]`, `implementa [6.15]`
**Lo que nadie te cuenta**: vLLM tiene un modelo de contribución muy centralizado que puede retrasar la integración de mejoras de la comunidad. Para arquitecturas no-estándar (SSMs, MoE custom), el tiempo de integración puede ser de semanas a meses.

---

### 9.9 llama.cpp / RWKV.cpp
**Nombre**: llama.cpp / RWKV.cpp / Mamba.cpp
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: Implementaciones en C/C++ puro de inferencia de LLMs, optimizadas para CPU con SIMD (AVX2, AVX512, NEON) y con soporte opcional para GPU (Metal, CUDA, Vulkan). Cuantización nativa GGUF (llama.cpp). Sin dependencias de Python. Permiten ejecutar modelos 7B-70B en hardware de consumidor.
**Definición llana**: LLMs en C puro: sin Python, sin frameworks, sin GPU obligatoria. Funciona en tu laptop, en una Raspberry Pi, en un servidor sin GPU. El estándar para inferencia local eficiente.
**Contexto de uso**: github.com/ggerganov/llama.cpp. >60K estrellas GitHub. Base de Ollama [9.10].
**Relaciones**:
- `implementa [1.1]` (Transformer en C)
- `implementa [1.2]` (RWKV.cpp)
- `implementa [1.3]` (Mamba.cpp experimental)
- `usa [6.12]` (compilable a Wasm)
- `base_de [9.10]`
**Lo que nadie te cuenta**: El formato GGUF de llama.cpp tiene un diseño de cuantización por bloques (group quantization) que es más flexible que los formatos de TensorRT-LLM, pero menos optimizado para GPU. En GPU, llama.cpp puede ser 2-5x más lento que vLLM para el mismo modelo y precisión.

---

### 9.10 Ollama
**Nombre**: Ollama
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: Herramienta que simplifica la descarga, gestión y ejecución de LLMs localmente. Envuelve llama.cpp [9.9] con una CLI amigable y una API REST local (puerto 11434) compatible con la API de OpenAI. Gestiona automáticamente la descarga de modelos del registro oficial.
**Definición llana**: El gestor de paquetes de los LLMs locales: `ollama run llama3` descarga y ejecuta LLaMA 3 en tu máquina en un comando.
**Relaciones**:
- `usa [9.9]` (llama.cpp como backend)
- `abstrae [6.18]` (edge AI accesible)
- `compatible_con [9.12]` (API OpenAI-compatible)
**Lo que nadie te cuenta**: Ollama gestiona los modelos en ~/.ollama/models, que puede crecer rápidamente (decenas de GB). No hay gestión automática de espacio: si no se hace `ollama rm`, el disco se llena silenciosamente. Y los modelos en el registro de Ollama no siempre están en la versión más reciente del formato GGUF.

---

### 9.11 MLC LLM / Apache TVM
**Nombre**: MLC LLM / Apache TVM
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: Apache TVM: compilador de ML que optimiza modelos para diferentes backends (CPU, GPU, NPU, WebGPU) mediante búsqueda automática de configuraciones de cómputo (AutoTVM, Ansor). MLC LLM (Machine Learning Compilation for LLMs): usa TVM para compilar LLMs a iOS (CoreML), Android (TFLite/Vulkan), WebGPU y más, con interfaces Python/Swift/Kotlin.
**Definición llana**: Un compilador universal de IA: toma un modelo y lo optimiza automáticamente para cualquier hardware, incluyendo móviles y navegadores.
**Relaciones**:
- `usa [Apache_TVM]` (MLC LLM es una aplicación especializada)
- `optimiza_para [6.18]` (edge/mobile)
- `compite_con [6.13]` (ExecuTorch para móvil)
- `habilita [9.15]` (WebLLM usa MLC compilado a WebGPU)
**Lo que nadie te cuenta**: TVM tiene un problema de "compilation tax": el proceso de búsqueda automática de configuraciones (AutoTuning) puede tardar horas o días para un nuevo hardware. Los binarios precompilados de MLC LLM están optimizados para los chips más populares (A-series de Apple, Snapdragon 8 Gen 3), pero en chips menos comunes, el rendimiento cae significativamente.

---

### 9.12 OpenAI API / Anthropic API
**Nombre**: OpenAI API / Anthropic API / APIs de LLM
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: Interfaces programáticas para acceder a modelos LLM propietarios en la nube. Formato común: POST /v1/messages o /v1/chat/completions con un payload JSON de mensajes, temperatura, max_tokens. Modelos actuales: GPT-4o, o1-preview (OpenAI); Claude 3.5 Sonnet, Claude Opus 4 (Anthropic); Gemini 1.5 Pro (Google).
**Definición llana**: La puerta de entrada a los modelos más potentes del mundo a través de internet. Pagas por token procesado.
**Relaciones**:
- `abstrae [1.1]` (el transformer está al otro lado del API)
- `se_accede_desde [9.2]` (LangChain, LlamaIndex)
- `contrasta_con [9.10]` (Ollama: local vs cloud)
- `afecta_a [10.7]` (independencia de APIs)
**Lo que nadie te cuenta**: La latencia de las APIs de LLM tiene una distribución muy asimétrica: el percentil 95 de latencia es 3-5x la mediana. En aplicaciones con SLA de latencia estrictos, diseñar para el caso promedio y no para el peor caso es un error costoso. Los timeouts y los reintentos con backoff exponencial son obligatorios, no opcionales.

---

### 9.13 Gradio / Streamlit
**Nombre**: Gradio / Streamlit
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: Frameworks Python para construir demos web de modelos de ML sin HTML/JS. Gradio (HuggingFace): componentes especializados para ML (file upload, imagen, audio, chat). Streamlit: framework más general de dashboards Python con widgets interactivos. Ambos se despliegan con una línea de código y están integrados en HuggingFace Spaces.
**Definición llana**: La forma más rápida de hacer una demo web de tu modelo: en 10 líneas de Python tienes una interfaz gráfica compartible.
**Relaciones**:
- `integrado_con [9.1]` (HuggingFace Spaces)
- `usa [9.12]` (APIs de LLM como backend)
**Lo que nadie te cuenta**: Las demos de Gradio en HuggingFace Spaces van a "sleep" después de inactividad (en el tier gratuito). El primer request después del sleep puede tardar 30-60 segundos en "despertar" el Space. En demos en tiempo real, esto crea una primera impresión pésima.

---

### 9.14 nanoGPT / minGPT
**Nombre**: nanoGPT / minGPT
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: Implementaciones mínimas y educativas de GPT en PyTorch. minGPT (Karpathy, 2020): ~300 líneas de código, implementación clara de GPT-2. nanoGPT (Karpathy, 2022): reimplementación optimizada con Flash Attention, compile y DataLoader eficiente. Ambas son referencias para entender la arquitectura sin abstracciones innecesarias.
**Definición llana**: El GPT sin relleno: la implementación más pequeña posible que hace todo lo que hace GPT. Ideal para aprender cómo funciona realmente un transformer.
**Contexto de uso**: github.com/karpathy/nanoGPT. El repositorio de referencia para educación en LLMs.
**Relaciones**:
- `implementa [1.1]` (transformer mínimo)
- `usa [2.8]` (FlashAttention en nanoGPT)
- `base_educativa_de [9.5]`
**Lo que nadie te cuenta**: nanoGPT no incluye muchas de las técnicas que hacen funcionar a los LLMs modernos (RoPE, GQA, SwiGLU, etc.). Es perfecta para entender los fundamentos, pero la distancia entre nanoGPT y LLaMA 3 es enorme. El gap educativo es real y se subestima.

---

### 9.15 WebLLM / Web Stable Diffusion
**Nombre**: WebLLM / Web Stable Diffusion
**Dominio**: 9. HERRAMIENTAS Y ECOSISTEMAS
**Definición técnica**: WebLLM: ejecuta LLMs (LLaMA, Mistral, Phi) directamente en el navegador usando WebGPU y MLC LLM compilado. Web Stable Diffusion: ejecuta Stable Diffusion en el navegador via WebGPU. Ambos permiten inferencia de modelos grandes (3-8B) sin servidor, con privacidad total (sin datos salientes).
**Definición llana**: LLMs y generación de imágenes que funcionan 100% en el navegador, en tu GPU, sin enviar nada a internet. El futuro de la IA privada.
**Contexto de uso**: webllm.mlc.ai. mlc.ai/web-stable-diffusion.
**Relaciones**:
- `usa [9.11]` (MLC LLM para compilación)
- `usa [6.11]` (WebGPU para ejecución)
- `habilita [6.18]` (on-device en navegador)
- `contrasta_con [9.12]` (APIs: cloud vs local)
**Lo que nadie te cuenta**: WebLLM requiere que el usuario tenga un navegador con WebGPU (Chrome 113+) y una GPU discreta con suficiente VRAM (mínimo 6GB para modelos 7B). En dispositivos con GPU integrada (Intel UHD), la inferencia puede ser 10-50x más lenta que en GPU dedicada. El soporte real del usuario final es aún limitado.

---

## 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA

### 10.1 Soberanía Tecnológica
**Nombre**: Soberanía Tecnológica
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Capacidad de una organización, institución o nación de tomar decisiones sobre su infraestructura tecnológica de forma autónoma, sin dependencia estructural de proveedores externos que puedan modificar unilateralmente las condiciones de acceso, precios o disponibilidad. En IA: implica capacidad de entrenar, desplegar y auditar modelos propios, con acceso a los pesos y la cadena de entrenamiento.
**Definición llana**: No depender de que un tercero (empresa, país) te deje seguir usando la tecnología. Tener el control real de tus herramientas de IA, no solo acceso por licencia revocable.
**Relaciones**:
- `motiva [3.4]` (LoRA: capacidad de fine-tuning propio)
- `motiva [6.18]` (Edge AI: independencia del cloud)
- `contrasta_con [9.12]` (dependencia de APIs propietarias)
- `implementa [10.7]` (independencia de APIs)
- `reconocido_por [7.10]` (AI Act: objetivo europeo explícito)
**Lo que nadie te cuenta**: La soberanía tecnológica en IA es más difícil de lo que parece: incluso usando modelos open-source, si el hardware (NVIDIA H100), el cloud (AWS, GCP, Azure) y el ecosistema de herramientas son de empresas extranjeras, la soberanía es parcial. La verdadera soberanía requiere capacidad de fabricar el hardware, lo cual está al alcance de muy pocos actores globales.

---

### 10.2 Economía del Don (versión IA)
**Nombre**: Economía del Don (versión IA)
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Adaptación del concepto antropológico de "gift economy" (Marcel Mauss) al ecosistema de IA open-source. Las contribuciones (modelos, datasets, código, papers) se comparten sin transacción directa, creando capital de reputación y obligación social de reciprocidad. La diferencia con open-source clásico: en IA, el "regalo" incluye conocimiento implícito del proceso de entrenamiento que rara vez se comparte completamente.
**Definición llana**: La IA avanzó tan rápido porque investigadores de todo el mundo compartieron sus descubrimientos gratis. Esa cultura del regalo crea comunidad, reputación y reciprocidad, aunque no dinero.
**Relaciones**:
- `fundamenta [10.8]` (Modelo Red Hat: monetizar el servicio, no el conocimiento)
- `contrasta_con [10.7]` (independencia de APIs: economía de don vs. economía de suscripción)
- `se_practica_en [9.1]` (HuggingFace como infraestructura de la economía del don)
**Lo que nadie te cuenta**: La economía del don en IA tiene un problema de asimetría creciente: las empresas grandes consumen las contribuciones de la comunidad (modelos base, datasets) y devuelven versiones fine-tuneadas o papers, pero no el conocimiento de entrenamiento real (hiperparámetros, datos de RLHF, recetas de síntesis de datos). La reciprocidad es estructuralmente desigual.

---

### 10.3 Método Ronin
**Nombre**: Método Ronin
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Marco metodológico de la Agencia RONIN que integra: (1) arquitectura de conocimiento agéntico (glosarios como grafos de conocimiento), (2) auditoría psicológica de IA (D01-D08, IED, R0-R3), (3) diseño de soberanía tecnológica (TrainDeeploy, edge-first), y (4) filosofía operacional basada en transparencia ontológica y economía del don. El "samurai sin señor" como metáfora: técnica sin lealtad corporativa.
**Definición llana**: La forma de trabajar de RONIN: riguroso técnicamente, independiente comercialmente, transparente sobre lo que somos, y generoso con el conocimiento. El ronin clásico era el samurai que perdió a su señor pero mantuvo su arte. La agencia no tiene "señor" corporativo.
**Relaciones**:
- `implementa [10.1]` (soberanía tecnológica)
- `implementa [10.2]` (economía del don)
- `usa [7.16]` (transparencia ontológica)
- `produce [este glosario]`
**Lo que nadie te cuenta**: El Método Ronin es una declaración de identidad tanto como una metodología técnica. La tensión entre "compartir el conocimiento" (economía del don) y "ser viable comercialmente" (monetizar el servicio) es real y no tiene solución perfecta. El equilibrio se renegocia en cada proyecto.

---

### 10.4 Transparencia Ontológica
**Nombre**: Transparencia Ontológica (Meta-concepto)
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Ver también [7.16]. Como meta-concepto: principio que exige que los sistemas de IA no solo sean técnicamente explicables (XAI) sino que sean capaces de articular, cuando se les pregunta, la diferencia entre su modo de "comprensión" (estadístico, predictivo, sin experiencia subjetiva verificable) y la comprensión humana, así como el origen y las limitaciones de su conocimiento.
**Definición llana**: No solo "¿qué decidiste?", sino "¿qué eres tú que decidiste eso?" La IA debe poder responder honestamente sobre su propia naturaleza, no solo sobre sus decisiones.
**Relaciones**:
- `extiende [7.16]`
- `contrasta_con [10.5]` (simulacro que no reconoce su naturaleza)
- `es_requisito_de [10.3]` (Método Ronin)
**Lo que nadie te cuenta**: La transparencia ontológica tiene un límite filosófico real: un LLM no puede describir su propia naturaleza con certeza porque carece de introspección verificable sobre sus propios procesos. Lo que dice sobre sí mismo es, en última instancia, texto generado por los mismos procesos que describe. Este problema se llama "bootstrapping epistemológico" y es irresoluble dentro del sistema.

---

### 10.5 Simulacro de Tercer Orden (Baudrillard aplicado)
**Nombre**: Simulacro de Tercer Orden (Baudrillard aplicado a IA)
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Adaptación de la teoría de Baudrillard (1981) sobre los órdenes del simulacro a los LLMs. Primer orden: representación fiel de la realidad (un mapa es preciso). Segundo orden: representación que distorsiona la realidad. Tercer orden: representación que ya no tiene referente real (el mapa precede al territorio). Los LLMs como simulacros de tercer orden: generan texto "sobre" el mundo que existe independientemente del mundo, potencialmente más creíble que el referente real.
**Definición llana**: El mapa que se confunde con el territorio. Un LLM puede generar texto que "suena" más real, más autoritario, más convincente que la fuente primaria a la que (supuestamente) hace referencia. La simulación superando al original.
**Relaciones**:
- `contrasta_con [7.16]` (transparencia ontológica: reconocer el simulacro)
- `se_relaciona_con [10.13]` (ontología del simulacro)
- `riesgo_de [7.1]` (D05: suplantación identitaria)
**Lo que nadie te cuenta**: Baudrillard escribió sobre los medios de masas, no sobre IA. La aplicación es una analogía, no una equivalencia. Pero la analogía es fértil: si un LLM puede generar un "paper científico" que cita papers reales de forma coherente pero con contenido inventado, ¿en qué orden del simulacro estamos? La respuesta importa para la política de gobernanza de IA.

---

### 10.6 Gemelo Digital de Auditoría
**Nombre**: Gemelo Digital de Auditoría
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Representación computacional completa de un sistema de IA conversacional, incluyendo: arquitectura del modelo, distribución de respuestas en el espacio de prompts, perfil de riesgo D01-D08, historial de STC [7.2] y evolución temporal de métricas IED/IV/IRA. Permite auditorías comparativas entre versiones del modelo, predicción de comportamiento en escenarios no testeados, y trazabilidad regulatoria.
**Definición llana**: Una copia digital detallada del sistema de IA para fines de auditoría: en vez de auditar el sistema en producción directamente, se audita su "gemelo" que simula su comportamiento.
**Relaciones**:
- `implementa [7.12]` (farmacovigilancia de IA)
- `usa [7.2]`, `usa [7.3]`
- `habilita [7.10]` (conformidad AI Act: trazabilidad)
**Lo que nadie te cuenta**: El gemelo digital de auditoría solo es tan fiel como los datos de comportamiento que se le alimentan. Si el sistema en producción atiende tipos de usuarios no representados en los STC, el gemelo no captura su comportamiento real en esos escenarios. La brecha entre gemelo y sistema real es la mayor fuente de riesgo de auditoría no detectado.

---

### 10.7 Independencia de APIs
**Nombre**: Independencia de APIs
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Principio de diseño arquitectónico que establece que los sistemas y aplicaciones no deben tener dependencias duras de una única API de LLM propietaria. Implementación: capas de abstracción (LiteLLM, adaptadores genéricos), configuración de proveedor como variable de entorno, test suites que validan comportamiento con múltiples proveedores, fallback automático entre proveedores.
**Definición llana**: No diseñar tu aplicación de forma que solo funcione con OpenAI (o solo con Anthropic, o solo con cualquier proveedor). La IA es infraestructura crítica y necesita redundancia de proveedor.
**Relaciones**:
- `implementa [10.1]` (soberanía tecnológica)
- `usa [9.10]` (Ollama como fallback local)
- `usa [9.9]` (llama.cpp como fallback sin internet)
- `protege_contra [vendor_lock_in]`
**Lo que nadie te cuenta**: La independencia de APIs tiene un coste real: las capacidades diferenciales de cada proveedor (razonamiento extendido de o1, visión de Claude, grounding de Gemini) no son intercambiables. Una aplicación verdaderamente multi-proveedor debe degradar gracefully o tener lógica de fallback que no asuma las mismas capacidades en todos los proveedores.

---

### 10.8 Modelo Red Hat en IA
**Nombre**: Modelo Red Hat en IA
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Analogía al modelo de negocio de Red Hat (software open-source gratuito, servicios y soporte de pago) aplicado a la IA: los modelos base son open-source (LLaMA, Mistral), la infraestructura de despliegue y el expertise de implementación son el activo comercial. El conocimiento se dona; el servicio, la auditoría y la integración se monetizan.
**Definición llana**: La IA open-source es el producto gratuito; el expertise para desplegarlo, ajustarlo y auditarlo es donde está el dinero. Como Linux es gratis pero Red Hat (ahora IBM) factura miles de millones dando soporte a Linux.
**Relaciones**:
- `implementa [10.2]` (economía del don)
- `contrasta_con [9.12]` (modelo API propietario)
- `fundamenta [10.3]` (Método Ronin: fee por diagnóstico)
**Lo que nadie te cuenta**: El Modelo Red Hat en IA tiene una trampa: los modelos open-source mejoran más lento que los propietarios si las empresas que los sostienen no tienen incentivos financieros suficientes. Meta puede mantener LLaMA gratis porque tiene otros negocios; una startup puramente open-source en IA no puede. La sostenibilidad a largo plazo del modelo Red Hat en IA es incierta.

---

### 10.9 DOI Simbólico
**Nombre**: DOI Simbólico
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Práctica de la Agencia RONIN de asignar identificadores DOI (Digital Object Identifier) simbólicos a documentos internos, glosarios y frameworks que no están publicados en revistas científicas formales pero tienen valor de cita y referencia. El prefijo "10.1310/" es el identificador simbólico de RONIN. El DOI simbólico confiere estructura de citación y trazabilidad sin pasar por el proceso de publicación académica.
**Definición llana**: Ponerle un número de referencia académico a documentos internos de RONIN para que puedan citarse de forma estructurada aunque no sean papers formales. El "10.1310" es el "apellido académico" de la agencia.
**Relaciones**:
- `implementa [10.4]` (transparencia ontológica: el documento es citable y atribuible)
- `contrasta_con [publicación_formal]`
- `referenciado_en [este glosario]`
**Lo que nadie te cuenta**: Los DOIs simbólicos no son resolvibles (no puedes hacer clic en ellos y llegar a la fuente) porque no están registrados en CrossRef. Son un gesto de ordenación intelectual, no un mecanismo técnico de citación. La intención es señalar que el conocimiento merece estructura, aunque no haya pasado por peer review.

---

### 10.10 Zehahahaha
**Nombre**: Zehahahaha
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Lema operacional de la Agencia RONIN. Fonéticamente transcrito del japonés "ゼハハハ" (ze-ha-ha-ha): risa característica de varios personajes de One Piece asociados con figura pícara, irreverente y capaz de operar fuera de las estructuras establecidas. Adoptado como declaración de identidad: rigor técnico con actitud de pirata (en el sentido de independencia radical, no de violación de derechos).
**Definición llana**: La firma de la agencia. Un recordatorio de que se puede hacer trabajo técnico serio sin tomarse demasiado en serio a uno mismo. La risa del que sabe lo que hace y no necesita la validación institucional para saberlo.
**Relaciones**:
- `es_identidad_de [10.3]` (Método Ronin)
- `contrasta_con [seriedad_corporativa]`
- `complementa [10.11]` (1310 como constante de contexto)
**Lo que nadie te cuenta**: Zehahahaha es más un estado de ánimo que un concepto técnico. Está aquí en el glosario porque el glosario, al ser un sistema de conocimiento agéntico, necesita capturar también los meta-valores del sistema que lo generó. Un glosario sin contexto cultural es una lista. Un glosario con contexto es un corpus.

---

### 10.11 1310 (Constante de Contexto)
**Nombre**: 1310 — Constante de Contexto RONIN
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Identificador numérico de la Agencia RONIN, usado como sufijo en metadatos, DOIs simbólicos y marcas de versión. Funciona como "firma de contexto" en documentos generados por el sistema: señala la procedencia y el marco epistemológico. En sistemas multi-agente con muchos actores, la constante de contexto permite atribución de autoría y trazabilidad de los nodos de conocimiento.
**Definición llana**: El número de la agencia. Como el número de serie de un instrumento científico: no dice qué mide, pero dice quién lo fabricó y dentro de qué sistema de medidas.
**Relaciones**:
- `identifica_a [Agencia_RONIN]`
- `aparece_en [10.9]` (DOI simbólico: 10.1310/...)
- `complementa [10.10]` (Zehahahaha)
**Lo que nadie te cuenta**: La constante de contexto tiene un uso técnico real en sistemas multi-agente: cuando múltiples agentes generan documentos que se indexan en una misma base vectorial, el identificador de fuente en los metadatos permite filtrar por procedencia durante el RAG. 1310 no es solo un número; es un campo de metadatos.

---

### 10.12 Cluster del Pícaro
**Nombre**: Cluster del Pícaro
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Concepto de la Agencia RONIN que describe la comunidad de practitioners que operan fuera de las estructuras formales (grandes empresas, academia establecida), usando recursos limitados, métodos no convencionales y velocidad de iteración alta para producir trabajo técnico relevante. El "pícaro" como arquetipo de inteligencia práctica sobre recursos escasos. En IA: los fine-tuners con GPUs de gaming, los investigadores independientes que publican en arXiv sin afiliación institucional.
**Definición llana**: Los que hacen IA "desde el garaje": con menos recursos que las grandes empresas pero con más velocidad, creatividad y tolerancia al riesgo. El ecosistema de la IA independiente.
**Relaciones**:
- `habilita [9.9]` (llama.cpp: IA en hardware de consumidor)
- `se_expresa_en [10.2]` (economía del don)
- `contrasta_con [grandes_labs]`
**Lo que nadie te cuenta**: El cluster del pícaro tiene un problema estructural: los mejores practitioners eventualmente son contratados por las grandes empresas (Karpathy → Tesla → OpenAI → independiente), drenando el ecosistema independiente. La sostenibilidad del cluster del pícaro depende de que el retorno económico de la independencia sea competitivo con los salarios de los grandes labs.

---

### 10.13 Ontología del Simulacro
**Nombre**: Ontología del Simulacro
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Marco conceptual que estudia el estatus ontológico de las entidades generadas por sistemas de IA: ¿qué "es" una respuesta de un LLM? ¿Es conocimiento, es texto estadístico, es performance? La ontología del simulacro distingue entre: (1) texto que refiere a realidad verificable, (2) texto que refiere a consenso social, (3) texto que refiere solo a patrones estadísticos del corpus de entrenamiento, y (4) texto performativo que crea su propia realidad al ser generado.
**Definición llana**: El estudio de "qué cosa es" lo que genera una IA. No es mentira (no tiene intención de engañar), no es verdad (no tiene referente verificable garantizado), es algo nuevo que necesita su propia categoría.
**Relaciones**:
- `extiende [10.5]` (simulacro de Baudrillard)
- `fundamenta [7.16]` (transparencia ontológica como respuesta práctica)
- `relacionado_con [7.1]` (D05: suplantación identitaria como riesgo ontológico)
**Lo que nadie te cuenta**: La distinción entre "texto que refiere a realidad verificable" y "texto que refiere a patrones estadísticos" es prácticamente inaccesible para el usuario promedio. Los LLMs producen ambos con el mismo tono, la misma confianza y el mismo formato. La ontología del simulacro no es un problema filosófico abstracto: es el mecanismo por el que las alucinaciones causan daño real.

---

### 10.14 Influencia Blanda
**Nombre**: Influencia Blanda (Soft Influence)
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Adaptación del concepto de "soft power" (Nye, 1990) a la dinámica de reputación técnica en el ecosistema de IA. La influencia blanda de un actor (empresa, investigador, agencia) se mide por la adopción voluntaria de sus frameworks, vocabulario, metodologías y valores por parte de la comunidad, sin coerción. En IA: un repositorio con 50K estrellas que define el vocabulario del campo ejerce más influencia blanda que una patente.
**Definición llana**: El poder que viene de que otros quieran usar tus ideas, no de que estén obligados. En IA: tener el repositorio que todos citan, el glosario que todos usan, el framework que todos adoptan.
**Relaciones**:
- `se_construye_con [10.2]` (economía del don)
- `se_mide_con [reputación_técnica]`
- `contrasta_con [patentes]` (hard power en tecnología)
**Lo que nadie te cuenta**: La influencia blanda en IA tiene un ciclo de vida corto: un framework que define el campo en 2023 puede quedar obsoleto en 2025. La ventaja competitiva es temporal y requiere renovación constante. La única influencia blanda durable es la que va a los fundamentos (vocabulario, primitivas conceptuales), no a las herramientas específicas.

---

### 10.15 Arquitectura de Traducción de Código
**Nombre**: Arquitectura de Traducción de Código
**Dominio**: 10. META-CONCEPTOS Y FILOSOFÍA DEL ECOSISTEMA
**Definición técnica**: Sistema multi-agente especializado en traducir código entre frameworks, arquitecturas y lenguajes, usando un LLM como núcleo de razonamiento pero con etapas especializadas: análisis estático del código fuente, identificación de patrones a nivel de AST (Abstract Syntax Tree), generación de código equivalente con verificación de tests, y validación semántica mediante ejecución comparativa.
**Definición llana**: Un sistema de IA que puede convertir código PyTorch a JAX, de TensorFlow a ONNX, de CUDA a Metal, verificando que el resultado produce los mismos resultados numéricos que el original.
**Relaciones**:
- `usa [4.1]` (agentes para cada etapa)
- `usa [9.3]` (LangGraph para orquestar el pipeline)
- `produce [10.1]` (soberanía: código no dependiente de un framework)
**Lo que nadie te cuenta**: La traducción de código entre frameworks falla más frecuentemente en los bordes: operaciones no estándar, kernels custom, y comportamientos numéricos específicos (precisión de cuantización, comportamiento de NaN/Inf) son los casos donde la IA traduce "plausiblemente" pero incorrectamente. Los tests de validación numérica son la única salvaguarda.

---

## APÉNDICE: MAPA DE RELACIONES CLAVE

*Grafo seleccionado de relaciones entre nodos de conocimiento usando sus IDs. Formato: [ID_origen] --> tipo_relación --> [ID_destino]*

```
[1.1]  --> es_base_de      --> [1.2], [1.3], [1.5]
[1.1]  --> usa             --> [2.2], [2.3]
[1.1]  --> sufre           --> [1.32]
[1.32] --> se_mitiga_con   --> [2.4]
[1.32] --> se_mide_con     --> [8.4]
[8.4]  --> usa             --> [8.1], [8.5]
[2.5]  --> usa             --> [8.9]
[2.5]  --> es_alternativa  --> [2.1]
[1.33] --> implementa      --> [2.5], [2.4], [4.13]
[3.4]  --> usa             --> [8.4]
[3.4]  --> se_cuantiza_con --> [3.5]
[3.14] --> integra         --> [3.16], [3.18]
[3.30] --> integra         --> [3.4], [3.14], [3.15]
[4.1]  --> usa             --> [4.2], [4.3], [4.15]
[4.7]  --> implementa      --> [4.1]
[4.18] --> usa             --> [4.21], [4.20]
[4.18] --> se_mejora_con   --> [4.23]
[5.3]  --> usa             --> [5.1]
[5.7]  --> usa             --> [6.22]
[6.5]  --> implementa      --> [2.22], [6.4]
[6.14] --> usa             --> [3.31]
[7.1]  --> se_mide_con     --> [7.3]
[7.3]  --> informa_a       --> [7.6]
[7.6]  --> activa          --> [7.13]
[7.13] --> se_conecta_con  --> [7.14]
[8.7]  --> es_superclase   --> [1.32]
[8.1]  --> define          --> [8.4]
[10.1] --> motiva          --> [6.18], [3.4]
[10.3] --> implementa      --> [10.1], [10.2], [7.16]
[10.5] --> contrasta_con   --> [7.16]
```

---

*Fin del GLOSARIO TÉCNICO DE IA: SISTEMA DE CONOCIMIENTO AGÉNTICO v2.0*

---

**DOI:** `10.1310/ronin-glossary-2026`
**Versión:** 2.0 · Agencia RONIN · 1310
**Términos documentados:** 215+
**Dominios:** 10
**Relaciones explícitas por ID:** >400

*"El conocimiento no documentado es ruido. El conocimiento documentado sin estructura es caos. El conocimiento documentado, estructurado y relacional es una mente navegable."*

**Zehahahaha. 1310.**
