# Deep Dive into Quantization & Hardware Kernels

### 🚀 Overview

A structured exploration of Large Language Model (LLM) compression, moving from high-level API implementations to custom low-level hardware kernels. This repository documents my journey of understanding how to make AI more efficient, one bit at a time.
---

### ✨ The Journey So Far 

Quantization is often seen as a "black box." My goal with this project was to peel back the layers:
* **Third-Party Libs:** Started with high-level quantization APIs (GPTQ/PEFT) in Google Colab to understand the trade-offs between perplexity and memory.
* **Triton:** Custom wrote low-level kernels like LeakyReLU, Flash Attention to dive deeper understanding. 
* **Optimization (In Progress):** Currently implementing AWQ (Activation-aware Weight Quantization) to better preserve model weights based on activation distribution.

---

### 🛠️ Custom Triton Kernels

The highlight of this repo is the low-level kernel development. Instead of relying on pre-built libraries, I built these to master memory orchestration, tiling strategies, and GPU throughput.

* **1. Vectorized Softmax** 
**Summary:** Implemented a hardware-aware Softmax kernel to practice Triton's programming model and pointer arithmetic.
**Key Learning:** Focused on numerical stability and efficient reduction operations across thread blocks.

* **2. LeakyRELU (Autograd Compatible)**
**Summary:** Engineered custom Triton kernels for both forward and backward LeakyReLU passes, integrated into a torch.autograd.Function to support full backpropagation in PyTorch workflows.
**Key Learning:** Mastered the use of ctx.save_for_backward and tl.where logic to maintain gradient flow, ensuring the backward kernel correctly handles the non-zero derivative for negative inputs.

* **3. FlashAttention Forward Pipeline** 
**Summary:** Implemented a custom Triton kernel for the FlashAttention forward pass, utilizing tiling and online softmax to compute exact attention with O(N) memory complexity.
**Key Learnings:** Gained deep expertise in managing SRAM bank conflicts, optimizing tiling for L2 cache hits, and using the "online" normalization trick to avoid materializing the massive N×N attention matrix.
---

### 🏗️ Roadmap
| Feature | Status |
| :--- | :--- |
| **Leaky ReLU** (Forward & Backward) | ✅ Complete |
| **Flash Attention** (Forward Pass) | ✅ Complete |
| **Flash Attention** (Backward Pass) | 🚧 In Progress |
| **AWQ** (Activation-aware Quantization) | 🚧 In Progress |
| **RMSNorm** | 📅 Planned |
| **GeLU** | 📅 Planned |
| **Mamba** | 📅 Planned |

#### Resources & Credits
1. Reference Implementation on Flash Attention: Umar Jamil [project](https://www.youtube.com/watch?v=zy8ChVd_oTM)


