# Study Session — MLP vs CNN, fundamentos y función de activación

- **Date:** 2026-10-13
- **Skill:** /explain (explicaciones + HTML interactivos), sin /teach formal
- **Source material:** `mlp-vs-cnn` (map), `vector`, `matrix`, `tensor`, `inner-product`, `matrix-multiplication`, `linear-mapping`, `activation-function`, `relu`, `softmax`, `multilayer-perceptron`, `backpropagation`, `gradient-descent`; ejemplo XOR de Goodfellow, Bengio & Courville, *Deep Learning* §6.1 (no está en `raw/`)
- **Goal:** Entender la diferencia entre un MLP y una CNN, y después sus fundamentos: qué es un tensor, por qué importa la función de activación, dónde se aplica y de dónde salen los pesos.

## What we covered
1. **MLP vs CNN.** El MLP conecta todo con todo (cada conexión tiene su peso). La CNN desliza un filtro pequeño con los mismos pesos por toda la imagen. Una CNN es un MLP con reglas: su matriz de pesos casi toda en ceros y con valores repetidos (matriz de Toeplitz).
2. **Los fundamentos para entender MLP vs CNN,** en 6 pasos: datos como números → producto punto → neurona → apilar capas → aprender → sobreajuste.
3. **Paso 1 — tensores.** Vector (lista), matriz (cuadrícula), tensor (pila de cuadrículas). La analogía del cine (asiento → fila → sala → edificio → calle). Forma (*shape*), un índice por eje, total = producto de los tamaños, reshape y flatten.
4. **Paso 2 — producto punto.** Multiplicar posición por posición y sumar. Es grande cuando la entrada se parece a los pesos. Matriz × vector = un producto punto por fila. Una matriz mueve el espacio sin doblar las líneas.
5. **Paso 3 — la neurona.** z = w·x + b, luego h = σ(z). Una capa es h = σ(W·x + b).
6. **Por qué importa la activación.** Sin ella las capas se colapsan: W₂(W₁x) = (W₂W₁)x = W′x, y la red solo dibuja rectas. Cada neurona ReLU agrega un doblez; con muchos dobleces se forma casi cualquier curva.
7. **¿Por neurona o por capa?** Se aplica a cada neurona por separado (elemento por elemento), pero se elige una por capa. Softmax es la excepción, porque mira toda la capa. Va en las capas ocultas; en la salida depende del problema (ninguna, sigmoide o softmax).
8. **XOR.** Ninguna línea separa los puntos. La red h = ReLU(W·x + b), y = h₁ − 2h₂ los separa porque ReLU dobla el espacio. Qué es la `y` (la neurona de salida, con pesos +1 y −2) y la regla y > 0.5.
9. **De dónde salen los pesos.** Los del ejemplo eran a mano. En la vida real se aprenden: empezar al azar → predecir → medir el error L → gradiente (*backprop*) → w ← w − η·∂L/∂w → repetir. Hiperparámetros (los decides tú) vs parámetros (los aprende la red).

## Key takeaways (in the human's words)
- (Por llenar cuando lo expliques con tus palabras.) Lo que más se trabajó:
  - "Una CNN es un MLP con reglas: pocos pesos, compartidos."
  - "Sin activación, 100 capas son una sola capa."
  - "La activación se aplica a cada neurona, pero se escoge por capa."
  - "La `y` es solo otra neurona que junta lo que dijeron las ocultas."
  - "Los pesos no se dan: la red los aprende bajando el error."

## Mastery check
No hubo quiz formal con `/teach`. Esto es lo que se explicó y se aclaró en la conversación:
- [x] La diferencia MLP vs CNN (denso vs local + compartido)
- [x] Qué es un tensor y qué significa su forma (preguntó y se aclaró con la analogía del cine)
- [x] Para qué sirve la activación (el colapso de capas y el doblez)
- [x] Dónde se aplica: por neurona, elegida por capa; la excepción de softmax
- [ ] Leer la fórmula completa del XOR sin ayuda (preguntó qué era la `y`: repasar)
- [ ] El ciclo de entrenamiento y la regla w ← w − η·∂L/∂w (recién visto: practicar con "1 paso")
- [ ] Pasos 4–6 de los fundamentos (apilar capas, backprop a fondo, sobreajuste): pendientes

## Open questions / revisit next
- Pasos 4–6 de los fundamentos de MLP vs CNN.
- Backpropagation a fondo: cómo se calcula ∂L/∂w con la regla de la cadena.
- Escribir la página `convolution` / `convolutional-neural-network` (todavía no existe en la wiki).
- Hacer una sesión `/teach` en inglés (método CLIL) sobre un capítulo del libro de LLM.

## Spaced review (test yourself later — don't re-read, recall)
1. ¿Por qué una red de 50 capas sin activación equivale a una sola capa? Escribe la fórmula.
2. Un tensor tiene forma (2, 3, 4). ¿Cuántos números tiene, y cuántos índices necesitas para elegir uno?
3. Calcula a mano la red XOR para el punto (1,1): z₁, z₂, h₁, h₂, y. ¿Qué color dice?
4. ¿La activación se aplica por neurona o por capa? ¿Cuál es la excepción y por qué?
5. Explica en una frase qué hace w ← w − η·∂L/∂w.
6. ¿Por qué una CNN necesita muchos menos pesos que un MLP para una imagen?

## Artifacts
- `explainers/2026-10-13_mlp-vs-cnn.html` (inglés, con helper de vocabulario) y `explainers/2026-10-13_mlp-vs-cnn-es.html` (español)
- `explainers/2026-10-13_foundations-1-2-3.html` (inglés) y `explainers/2026-10-13_foundations-1-2-3-es.html` (español): vectores, tensores, producto punto, neurona, animación de dobleces
- `explainers/2026-10-13_funcion-de-activacion.html` (español): capa animada, colapso de matrices, XOR paso a paso, fórmula desarmada, modo manual de pesos, entrenamiento en vivo
- Wiki: `mlp-vs-cnn` (map, con sección "Foundations")
