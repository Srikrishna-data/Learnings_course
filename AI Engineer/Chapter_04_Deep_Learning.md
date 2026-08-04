# Chapter 4 — Deep Learning: Neural Networks, Vision, Sequences, and Transformers

> **Part 4 — Deep Learning**  
> **Target time:** 10–14 hours  
> **Outcome:** You can build, train, evaluate, and serve a small neural-network model, choose PyTorch or TensorFlow deliberately, and explain why transformers displaced many earlier sequence architectures.

## 1. What deep learning is

**Deep learning** is machine learning with neural networks containing multiple layers. A neural network learns transformations of input data rather than relying entirely on features designed by people. It is particularly effective for images, audio, language, and other complex unstructured inputs.

```text
raw input ─► learned representation layers ─► task head ─► prediction
 image     ─► edges → textures → objects     ─► class
 audio     ─► frequencies → phonemes          ─► transcript
 text      ─► token relationships              ─► next token / label
```

### Why it exists

Traditional ML often needs manually engineered inputs: word counts, image descriptors, or hand-built audio features. Neural networks can learn useful representations directly from data. This is a major advantage when the raw input is large and complicated.

### What it solves—and what it costs

| Strength | Industry use | Trade-off |
|---|---|---|
| Learns complex patterns | visual inspection, speech, translation | more data and compute |
| Reuses pretrained models | document extraction, image classification | transfer-learning evaluation needed |
| Handles multimodal inputs | image + text support systems | harder observability and privacy controls |
| Runs efficiently on accelerators | high-throughput inference | GPUs and serving operations cost money |

> **Practical rule:** For small structured/tabular datasets, start with logistic regression or gradient-boosted trees. Choose deep learning when data modality, scale, or pretrained representations provide a measured advantage.

---

## 2. Neural-network building blocks

### Neurons, layers, and parameters

One simplified neuron calculates a weighted sum, adds a bias, then applies an activation function:

```text
x1 ──(w1)─┐
x2 ──(w2)─┼──► z = w·x + b ─► activation(z) ─► output
x3 ──(w3)─┘
```

The weights and biases are **parameters**. Training adjusts them to reduce loss. A layer applies many such transformations in parallel. “Deep” means several layers are composed.

### Activation functions

Without activations, many stacked linear layers collapse into one linear transformation. Activation functions add nonlinearity.

| Activation | Why used | Caution |
|---|---|---|
| ReLU | fast default for hidden layers | can produce permanently inactive units |
| GELU | common in transformers | slightly more compute |
| Sigmoid | converts score to 0–1 | can saturate; usually output layer only |
| Softmax | turns class scores into a distribution | values can look like confidence but need calibration |

### Loss functions

Loss tells training how wrong a prediction is. The loss must match the task.

| Task | Typical loss | What it measures |
|---|---|---|
| Binary classification | binary cross-entropy | mismatch between probability and binary label |
| Multi-class classification | cross-entropy | mismatch between predicted class distribution and label |
| Regression | MSE or MAE | numeric prediction error |
| Token generation | next-token cross-entropy | probability assigned to actual next token |

### Backpropagation and optimizers

Backpropagation applies the chain rule to calculate how each parameter influenced loss. An optimizer such as SGD or Adam uses those gradients to update parameters.

```text
forward pass: inputs ─► prediction ─► loss
                                      │
backward pass: gradients ◄────────────┘
      │
optimizer update: weights = weights - learning_rate × gradient
```

You do not normally hand-code backpropagation: PyTorch and TensorFlow perform automatic differentiation. You must still select a loss, inspect training curves, and manage learning rate.

---

## 3. Training loop from first principles

### The essential steps

1. Load and validate training/validation/test data.
2. Transform inputs consistently (normalization, tokenization, augmentation).
3. Run a forward pass to produce predictions.
4. Calculate loss against known labels.
5. Backpropagate gradients and update parameters.
6. Evaluate on validation data without updates.
7. Save the best validated checkpoint, metadata, and metrics.

### PyTorch example: compact classifier

```python
import torch
from torch import nn

# Two numeric features; binary class output.
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

features = torch.tensor([[0.2, 1.0], [0.8, 0.1], [0.1, 1.2], [0.9, 0.2]])
labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

loss_fn = nn.BCEWithLogitsLoss()       # combines stable sigmoid + cross entropy
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    logits = model(features)
    loss = loss_fn(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

probabilities = torch.sigmoid(model(features))
print(probabilities.detach())
```

### TensorFlow/Keras equivalent

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0)],
)
model.fit(features.numpy(), labels.numpy(), validation_split=0.25, epochs=20, verbose=0)
```

### PyTorch vs TensorFlow: which should you use?

| Dimension | PyTorch | TensorFlow / Keras |
|---|---|---|
| Development style | Flexible, Python-first, eager execution | High-level Keras API plus graph/runtime tools |
| Common use | Research, custom models, open-model ecosystem | Established enterprise stacks, mobile/TF ecosystem |
| Debugging | Natural Python debugging | Simple with Keras; lower-level graph work can be less direct |
| Deployment | TorchServe alternatives, ONNX, custom services | TensorFlow Serving, TFLite, TF ecosystem |
| Decision | Default for new model-learning work | Choose when team/platform/mobile requirements favor it |

Both are capable. Team expertise, existing infrastructure, model availability, and deployment target matter more than ideology.

---

## 4. Generalization, regularization, and evaluation

### Overfitting in deep learning

Overfitting occurs when a network learns quirks of the training data rather than patterns that transfer to new inputs. Training loss continues falling while validation loss rises.

```text
loss
 ^       validation   /\
 |                    /  \  ← warning: overfitting
 |  training  _______/
 +----------------------------------> epochs
```

### Practical defenses

| Technique | What it does | Typical use |
|---|---|---|
| More representative data | reduces memorization pressure | always preferred |
| Data augmentation | creates realistic variations | images, audio, some text tasks |
| Weight decay | penalizes overly large weights | common default |
| Dropout | disables random units during training | some classifiers/heads |
| Early stopping | stop at best validation checkpoint | standard training control |
| Transfer learning | begin with pretrained knowledge | most practical vision/NLP projects |

### Production requirements

Record model architecture, code commit, dependency versions, training/validation dataset versions, random seed, hyperparameters, hardware, metrics, and checkpoint hash. Without this, a good result cannot be reproduced or audited.

> **Callout — data is part of the model.** A saved `.pt` or `.keras` file alone is not a model release. Preprocessing, label mapping, input schema, thresholds, and evaluation results are part of its behavior.

---

## 5. Convolutional neural networks (CNNs)

### What they are

CNNs process image-like grids using small learned filters (kernels) that slide across the input. A filter can respond to an edge, texture, or higher-level visual pattern. Shared filter weights make CNNs efficient and help them recognize a feature regardless of position.

```text
image pixels ─► convolution filters ─► feature maps ─► pooling/blocks ─► class head
               (edges/textures)        (patterns)       (object class)
```

### Industry use cases

- Defect detection in manufacturing.
- Medical-image assistance with clinical validation.
- Product catalog classification.
- OCR and document-layout components.
- Moderation and visual search.

### PyTorch CNN example

```python
from torch import nn

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 5)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
```

### Advantages, limitations, and costs

CNNs are efficient and inductively suited to images. Vision transformers and pretrained multimodal models increasingly compete with or replace them in broad applications. Training from scratch is expensive; transfer learning from a vetted pretrained model is normally the practical choice.

**Common mistakes:** ignoring class imbalance, evaluating only clean images, allowing train/test images from the same product batch, and treating a model’s label as a clinical or safety decision without human review.

---

## 6. RNNs and sequence models

### What they are

A recurrent neural network (RNN) processes a sequence one item at a time while carrying a hidden state forward. LSTM and GRU variants added gates to preserve or forget information more effectively.

```text
x1 ─► [RNN cell] ─► h1 ─► [RNN cell] ─► h2 ─► [RNN cell] ─► h3
                         x2                 x3
```

### Why they mattered

RNNs gave sequence models memory of previous steps and were widely used for language, time series, and speech. They introduced useful ideas such as hidden state and teacher forcing.

### Limitations

They process tokens sequentially, limiting parallelism. Long-range dependencies are difficult, and gradients can vanish or explode. Transformers solve many of these limitations for language and increasingly for vision/audio, though RNN-like models can still be appropriate for small streaming or constrained systems.

| Use case | Prefer RNN/LSTM when | Prefer transformer when |
|---|---|---|
| Small time series | low-resource, short sequences | long/multivariate patterns, pretrained models |
| Streaming signal | strict incremental processing | enough compute/latency budget exists |
| Language | legacy system constraints | almost all new general language work |

---

## 7. Attention and transformers

### Attention: what it is

Attention lets each token or element calculate how much to use information from other elements. A token creates three learned representations:

- **Query (Q):** what information am I seeking?
- **Key (K):** what kind of information do I contain?
- **Value (V):** what information should I contribute?

The model compares queries to keys, converts the scores to weights, and uses those weights to combine values.

```text
tokens ─► Q, K, V projections
              │
      Q × Kᵀ similarity scores
              │
          softmax weights
              │
            weights × V ─► context-aware token representations
```

In simplified notation:

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

You do not need to memorize this equation. Understand its consequence: a word can incorporate relevant information from other words, rather than receiving only a small previous hidden state.

### Transformer architecture

A transformer layer usually contains multi-head attention, a feed-forward network, residual connections, and normalization. Multiple heads can learn different relationship types. Encoder-only models are often used for embeddings/classification; decoder-only models generate text; encoder-decoder models are common for translation-style tasks.

```text
token IDs + position information
              │
      [attention → MLP] × N layers
              │
        task head / next-token distribution
```

### Positional encoding

Attention alone does not know token order. “Dog bites man” and “Man bites dog” would be a bag of tokens without positions. Positional embeddings or encodings add sequence order.

```text
token embedding("refund") + position embedding(4) = transformer input at position 4
```

### Advantages and limitations

Transformers parallelize training effectively and capture long-distance dependencies. Standard attention has memory/compute costs that grow quickly with sequence length, so long-context systems use optimizations and still need careful context selection. More context is not automatically better: irrelevant text can distract the model and increase cost.

### Production use cases

- Text embeddings for semantic search.
- Document classification and entity extraction.
- Speech/vision encoders.
- LLM generation, code assistance, and tool-calling systems.

### Interview questions

1. What problem does attention address relative to RNNs?
2. Why does a transformer need positional encoding?
3. Explain query, key, and value without equations.
4. Why can long context be expensive even when a model supports it?

---

## 8. Production deep-learning architecture

### Training path versus inference path

Keep training and online serving separate. Training can be long-running, expensive, and allowed to fail/retry; serving must satisfy user latency and reliability requirements.

```text
TRAINING
source data ─► validation ─► training jobs (GPU) ─► model registry ─► approval

INFERENCE
client ─► API/auth ─► input validation ─► model server ─► response
                  │                         │              │
                  └── telemetry ────────────┴── monitoring ─┘
```

### Serving decisions

| Need | Practical choice |
|---|---|
| Small traffic, standard model | managed endpoint or containerized FastAPI service |
| High GPU throughput | specialized inference server with batching |
| Mobile/edge | quantized model and on-device runtime |
| Long document/image jobs | object storage + queue + async worker |
| Regulated/high-risk result | human review and audit trail before action |

### Cost optimization

- Prefer pretrained checkpoints and fine-tune a small head or adapter where appropriate.
- Use batch inference for offline work; scale online services with queue depth or demand.
- Quantize only after task-specific quality evaluation.
- Cache immutable results such as image embeddings, but include model version in cache keys.
- Track GPU utilization, request latency, model-load time, and cost per accepted result.

### Reliability and security

Validate image/audio/text size and type before processing. Use signed object URLs, malware scanning where relevant, timeouts, retries for idempotent jobs, and redacted observability. Do not place raw regulated data into debugging logs or unapproved training sets.

---

## 9. Hands-on project — image-quality inspection service

Build a prototype that classifies product images as `acceptable`, `blurry`, or `damaged`.

### Required deliverables

1. Define labels with written examples; include an `uncertain` route.
2. Use transfer learning from a pretrained CNN or vision transformer.
3. Split data by product/batch to prevent nearly identical train/test images.
4. Train with augmentation that mirrors plausible camera variation.
5. Report class-level precision/recall, confusion matrix, p95 inference latency, and cost/image.
6. Expose a typed asynchronous upload/inference API.
7. Store model version and a redacted result trace.
8. Route low-confidence or safety-relevant results to a human reviewer.

### Suggested directory structure

```text
image-inspection/
├── data_contract.md
├── training/
│   ├── train.py
│   └── evaluate.py
├── service/
│   ├── app.py
│   └── schemas.py
├── tests/
├── evaluation/
│   └── held_out_metrics.json
└── README.md
```

### Common interview follow-ups

- Why use transfer learning instead of training from scratch?
- How do you detect whether the model relies on background artifacts?
- When would GPU inference be worth the operating cost?
- What is your rollback plan after a bad vision-model release?

---

## Summary

Deep learning learns representations through layered neural networks trained by backpropagation and optimization. PyTorch is a strong default for flexible, modern model work; TensorFlow/Keras is a solid choice where its ecosystem aligns with the team or platform. CNNs remain useful vision tools, RNNs teach sequence fundamentals but have largely ceded general language tasks to transformers, and attention allows transformers to connect distant information while positional encoding preserves order.

Production deep learning requires more than a checkpoint: versioned data and preprocessing, held-out evaluation, controlled deployment, observability, security, capacity planning, and an explicit human fallback.

## Further reading

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Learn](https://www.tensorflow.org/learn)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome)

---

**Next chapter:** Part 5 — Large Language Models: tokenization, embeddings, context windows, decoding, RLHF/DPO, fine-tuning, LoRA/QLoRA, MoE, quantization, and distillation.
