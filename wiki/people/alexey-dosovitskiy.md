---
title: Alexey Dosovitskiy
type: person
tags: [computer-vision, transformers, vit, google]
sources: [llm-ch01-understanding-large-language-models]
created: 2026-10-13
updated: 2026-10-13
---

# Alexey Dosovitskiy (active 2010s–present)

> **Why they matter to ML:** first author of the **Vision Transformer (ViT)**, which showed that the same transformer used for text works on images too.

## Life (brief)
Alexey Dosovitskiy is a computer-vision researcher. He worked at the University of Freiburg (FlowNet, optical flow with neural networks) and Intel Labs (the CARLA driving simulator), and then at Google Research, Brain Team in Berlin, where the ViT paper was written. (Biographical details are general knowledge. The book names him only as "Dosovitskiy et al." in appendix B.)

## Key contributions
- **Vision Transformer — "An Image is Worth 16x16 Words"** (Dosovitskiy et al. 2020, arXiv 2010.11929). The book cites it as "the original vision transformer for classifying images, which illustrates that transformer architectures are not only restricted to text inputs." → [[transformer-architecture]], [[self-attention]]
  - **The core trick (external context):** cut the image into small square patches and treat each patch like a word [[token]]. The number of patches is $N = \dfrac{H \cdot W}{P^2}$. Symbol by symbol: $H$ = image height in pixels; $W$ = image width; $P$ = the side length of one patch (16 in the title); $P^2$ = pixels in one patch. Example: a $224 \times 224$ image with $P = 16$ gives $N = 50{,}176 / 256 = 196$ patches, so the transformer reads a "sentence" of 196 image tokens.
  - Each patch is flattened into a [[vector]] and multiplied by a learned [[matrix]] (a [[linear-layer]]) to get an embedding, exactly like a [[token-embedding]]. Then a [[positional-embedding]] is added so the model knows where each patch was. → [[linear-mapping]], [[embedding-dimension]]
- **Why the book mentions him.** Section 1.4's "Transformers vs. LLMs" box says that not all transformers are LLMs, because transformers are also used in **computer vision**. ViT is the reference for that claim. → [[large-language-model]]

## Connections
- Concepts: [[transformer-architecture]], [[self-attention]], [[token]], [[token-embedding]], [[positional-embedding]], [[linear-layer]], [[large-language-model]]
- People: [[ashish-vaswani]] (original transformer), [[jacob-devlin]] (BERT, whose [CLS]-token idea ViT reuses — external context)
- Appears in sources: [[llm-ch01-understanding-large-language-models]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book p. 10 ("Transformers vs. LLMs" sidebar: transformers are also used for computer vision).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 1 references, book p. 290 (Dosovitskiy et al. 2020 citation).
- Biography, the patch formula and the 196-patch example are general knowledge from the ViT paper, not from the book.
