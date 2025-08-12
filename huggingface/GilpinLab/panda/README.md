---
license: cc-by-nc-4.0
---

# Model Card for **_Panda_**

*Panda*: Patched Attention for Nonlinear Dynamics.

Paper abstract:

>Chaotic systems are intrinsically sensitive to small errors, challenging efforts to construct predictive data-driven models of real-world dynamical systems such as fluid flows or neuronal activity. Prior efforts comprise either specialized models trained separately on individual time series, or foundation models trained on vast time series databases with little underlying dynamical structure. Motivated by dynamical systems theory, we present Panda, Patched Attention for Nonlinear DynAmics. We train Panda on a novel synthetic, extensible dataset of 2 \times 10^4 chaotic dynamical systems that we discover using an evolutionary algorithm. Trained purely on simulated data, Panda exhibits emergent properties: zero-shot forecasting of unseen real world chaotic systems, and nonlinear resonance patterns in cross-channel attention heads. Despite having been trained only on low-dimensional ordinary differential equations, Panda spontaneously develops the ability to predict partial differential equations without retraining. We demonstrate a neural scaling law for differential equations, underscoring the potential of pretrained models for probing abstract mathematical domains like nonlinear dynamics. 


- **Preprint:** [arXiv:2505.13755](https://arxiv.org/abs/2505.13755)
- **Repository:** https://github.com/abao1999/panda

<!-- This modelcard aims to be a base template for new models. It has been generated using [this raw template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md?plain=1). -->

NOTE: we are currently in the process of scaling up our model and training, so stay tuned!

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

If you find our work valuable for your research, please cite us:
```
@misc{lai2025panda,
      title={Panda: A pretrained forecast model for universal representation of chaotic dynamics}, 
      author={Jeffrey Lai and Anthony Bao and William Gilpin},
      year={2025},
      eprint={2505.13755},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.13755}, 
}
```
