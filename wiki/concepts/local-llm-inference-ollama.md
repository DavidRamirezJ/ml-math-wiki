---
title: Local LLM Inference with Ollama
type: concept
tags: [llm, inference, tooling, llama, evaluation]
difficulty: beginner
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Local LLM Inference with Ollama

> **One-liner:** Ollama lets you download a ready-made LLM and chat with it on your own computer, no cloud needed.

## 🟢 Intuition (Dummy)
Training a model is like building a car. *Inference* is just driving it. Ollama is a garage that hands you finished cars (pretrained, instruction-tuned LLMs) and lets you drive them on your laptop. It can't build or modify cars: it does **inference only**, no training or fine-tuning (book p. 238).

In Chapter 7 the book uses Ollama to run **Llama 3 8B** as a grader for our own fine-tuned model ([[llm-as-a-judge]]). Ollama is a friendly wrapper around **llama.cpp**, a library that implements LLMs in pure C/C++ to make them run as fast as possible (book p. 238). Why is inference so much lighter than training? Because with fixed weights you don't need gradients, optimizer state, or weight updates. (That last point is my explanation, not the book's.)

## 🟡 Mechanics (Practitioner)

### Install and start (book pp. 239–241)
1. Install from ollama.com. On macOS/Windows, open the app and say **Yes** to command-line usage; on Linux use the install command from the website.
2. Start the server: run `ollama serve` in a separate terminal, **or** keep the Ollama app running in the background (the macOS option in Figure 7.20, book p. 240).
3. In another terminal (not Python), run `ollama run llama3`. The first run downloads the model: **4.7 GB**. The log ends with "verifying sha256 digest … writing manifest … success".
4. Chat in the terminal (e.g., `>>> What do llamas eat?`). Type `/bye` to leave, but **keep the server running** for the rest of the chapter.

### Which model fits your machine (book p. 241)
| Command | Model | RAM needed |
|---|---|---|
| `ollama run llama3` | Llama 3 8B, instruction-tuned | ~16 GB |
| `ollama run phi3` | Phi-3, 3.8B | ~8 GB |
| `ollama run llama3:70b` | Llama 3 70B | much more |

(The book prints the phi3 command as "ollama run llama3", which is a typo; the intended command is `ollama run phi3`.)

### Check the server is running (book pp. 241–242)
```python
import psutil
def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

if not check_if_running("ollama"):
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
```
Expected output: `Ollama running: True`.

### Talking to it from Python: the REST API (Listing 7.10, book pp. 242–243)
Ollama listens on **`http://localhost:11434/api/chat`**.

```python
import json, urllib.request

def query_model(prompt, model="llama3",
                url="http://localhost:11434/api/chat"):
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {                 # settings for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048,
        },
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data
```

What the options mean (my explanations; the book only labels them "settings for deterministic responses"):
- `seed: 123` fixes the random number generator ([[random-seed-reproducibility]]).
- `temperature: 0` turns sampling into [[greedy-decoding]].
- `num_ctx: 2048` sets the context window to 2,048 tokens ([[context-length]]).

The answer arrives as several JSON lines (streamed chunks), and the loop glues their `"content"` pieces together. Example: `query_model("What do Llamas eat?", "llama3")` returns text that begins "Llamas are ruminant animals, which means they have a four-chambered stomach…" followed by a list ("1. Grasses: Llamas love to graze on grasses…") (book p. 243).

### Why temperature 0 means "deterministic"
[[temperature-scaling]] divides the [[logits]] by $T$ before the [[softmax]]:

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}
$$

- $p_i$ = probability of picking token $i$ next.
- $z_i$ = the logit (raw score) for token $i$.
- $T$ = temperature; smaller $T$ makes the distribution sharper.
- $V$ = vocabulary size; the sum in the denominator makes all $p_i$ add up to 1.

As $T \to 0$, all the probability piles onto the largest logit, so $p$ becomes one-hot on $\arg\max_i z_i$ ([[argmax]]).

**Caveat:** even with these settings the book warns that Ollama "is not deterministic" at the time of writing, and results can vary across operating systems (book pp. 241, 247).

## 🔴 Theory (Expert)
- **Limit of the tempered softmax.** Let $z_{(1)} > z_{(2)}$ be the top two logits. Then $p_{(2)}/p_{(1)} = \exp\!\big(-(z_{(1)}-z_{(2)})/T\big) \to 0$ as $T \to 0^+$, so the distribution converges to a point mass on the argmax. With ties, it converges to a uniform distribution over the tied tokens. That's one reason "temperature 0" isn't always perfectly deterministic.
- **Residual non-determinism.** Even greedy decoding can flip when two logits are nearly tied, because GPU/CPU floating-point sums aren't associative: different thread schedules, batch shapes, or kernels change the last bits ([[floating-point-precision]]). This is a plausible source of the cross-OS variation the book reports. (Background knowledge, not from the book.)
- **Quantization from the numbers.** The 8B download is 4.7 GB: $4.7\times 10^9 \cdot 8 / 8\times 10^9 \approx 4.7$ bits per parameter, far below 16-bit floats. So Ollama ships a **quantized** model by default. The "16 GB RAM" advice is roughly $2$ bytes × 8B parameters, a safe system recommendation rather than the weight size. (My arithmetic; see [[parameter-count-and-memory]].)
- **Integrity check.** "verifying sha256 digest" compares a cryptographic hash of the downloaded file to the expected one, so a corrupted or tampered file is rejected.
- **Chat format.** The `messages` list with `role`/`content` is the chat-template abstraction: the server turns it into the model's own special-token prompt format ([[special-tokens]], [[prompt-template]]), just as our `format_input` builds an Alpaca-style prompt.

## Connections
- Builds on: [[autoregressive-text-generation]], [[temperature-scaling]], [[softmax]], [[greedy-decoding]], [[pretrained-weights]], [[large-language-model]]
- Leads to: [[llm-as-a-judge]], [[llm-evaluation]]
- Related: [[decoding-strategies]], [[random-seed-reproducibility]], [[context-length]], [[parameter-count-and-memory]], [[floating-point-precision]], [[train-vs-eval-mode]], [[gpu-computing]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8 opening (Llama 3 8B judge via Ollama; Ollama NOTE: llama.cpp wrapper, inference only), book p. 238. (Section 7.7 ends on the same page.)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8 (GPT-4 web-API sidebar, install, Figure 7.20, 4.7 GB download, model/RAM sidebar, `check_if_running`, Listing 7.10 `query_model`, non-determinism), book pp. 238–247.
