# Literature Review: Where is “Washing Machine” Stored in LLMs?

## Review Scope

### Research Question
Where (and how) are composite concepts like “washing machine” represented in transformer residual streams—via sparse monosemantic features, superposed features, or compositional mixtures of more atomic concepts (e.g., “washing” + “machine”)?

### Inclusion Criteria
- Papers on feature superposition / polysemanticity or monosemantic feature discovery
- Mechanistic interpretability papers that localize knowledge or concept-like features
- Methods that probe or edit internal representations in transformer LMs

### Exclusion Criteria
- Purely behavioral probing without access to internal activations
- Non-transformer architectures unless directly comparable

### Time Frame
2019–present (emphasis on 2021–2024)

### Sources
- arXiv
- Semantic Scholar
- Transformer Circuits (Anthropic)
- ROME project site

## Search Log

| Date | Query | Source | Results | Notes |
|------|-------|--------|---------|-------|
| 2026-02-08 | "sparse autoencoders interpretable features language models" | arXiv / Semantic Scholar | 10+ | High relevance to feature discovery |
| 2026-02-08 | "toy models of superposition" | Semantic Scholar | 1 | Core theory of superposition |
| 2026-02-08 | "locating and editing factual associations in GPT" | arXiv / ROME site | 1 | Causal tracing + editing |
| 2026-02-08 | "interpretability in the wild IOI" | arXiv | 1 | Circuit-level causal analysis |
| 2026-02-08 | "knowledge neurons in pretrained transformers" | arXiv | 1 | Neuron-level attribution |
| 2026-02-08 | "towards monosemanticity dictionary learning" | Transformer Circuits | 1 | Dictionary learning for features |
| 2026-02-08 | "mathematical framework for transformer circuits" | Transformer Circuits | 1 | Foundational mechanistic interpretability |

## Screening Results

| Paper | Title Screen | Abstract Screen | Full-Text | Notes |
|------|-------------|----------------|-----------|-------|
| Toy Models of Superposition | Include | Include | Include | Foundational theory for superposition |
| Sparse Autoencoders Find Highly Interpretable Features | Include | Include | Include | Direct method for feature discovery |
| Interpretability in the Wild (IOI) | Include | Include | Include | Causal tracing of behavior |
| Locating and Editing Factual Associations (ROME) | Include | Include | Include | Localization of factual knowledge |
| Knowledge Neurons | Include | Include | Include | Neuron-level attribution |
| Towards Monosemanticity | Include | Include | Include | Dictionary learning for LMs |
| Mathematical Framework for Transformer Circuits | Include | Include | Include | Background for circuit analysis |

## Paper Summaries

### Toy Models of Superposition (2022)
- **Authors**: Nelson Elhage et al.
- **Source**: arXiv 2209.10652
- **Key Contribution**: Formalizes feature superposition and polysemanticity in simplified models.
- **Methodology**: Theoretical and toy-model analysis of linear representations under sparsity constraints.
- **Datasets**: Synthetic toy tasks
- **Results**: Shows why sparse features can be superposed in a single direction when representation capacity is limited.
- **Code Available**: Yes (Anthropic / Transformer Circuits)
- **Relevance**: Provides theoretical basis for expecting composite concepts to be mixtures rather than orthogonal directions.

### Sparse Autoencoders Find Highly Interpretable Features in Language Models (2023)
- **Authors**: Hoagy Cunningham et al.
- **Source**: arXiv 2309.08600
- **Key Contribution**: SAEs can recover interpretable sparse features from transformer activations.
- **Methodology**: Train sparse autoencoders on GPT-2 activations, analyze recovered features.
- **Datasets**: GPT-2 activations from text corpora
- **Results**: Many SAE features align with human-interpretable concepts; features are sparse and compositional.
- **Code Available**: Yes (openai/sparse_autoencoder)
- **Relevance**: Directly supports searching for “washing machine” as a sparse feature or combination of features.

### Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small (2022)
- **Authors**: Kevin Wang et al.
- **Source**: arXiv 2211.00593
- **Key Contribution**: Identifies a causal circuit for IOI behavior in GPT-2 small.
- **Methodology**: Activation patching / causal tracing across layers and heads.
- **Datasets**: Synthetic IOI prompts
- **Results**: A small set of heads and MLPs are causally responsible for IOI.
- **Code Available**: Yes (Transformer Circuits)
- **Relevance**: Provides methods to locate concept-related circuits in residual streams.

### Locating and Editing Factual Associations in GPT (ROME) (2022)
- **Authors**: Kevin Meng et al.
- **Source**: arXiv 2202.05262
- **Key Contribution**: Shows factual associations are localized in specific MLP layers and can be edited.
- **Methodology**: Causal tracing + rank-one model edits.
- **Datasets**: CounterFact, zsRE
- **Results**: Edits are localized and can be evaluated with minimal side effects.
- **Code Available**: Yes (kmeng01/rome)
- **Relevance**: Suggests “washing machine” knowledge might localize to mid-layer MLPs for factual associations.

### Knowledge Neurons in Pretrained Transformers (2021)
- **Authors**: Damai Dai et al.
- **Source**: arXiv 2104.08696
- **Key Contribution**: Identifies neurons strongly associated with factual knowledge.
- **Methodology**: Gradient-based attribution and neuron editing.
- **Datasets**: LAMA-style cloze facts
- **Results**: Specific neurons influence factual outputs across prompts.
- **Code Available**: Yes (Hunter-DDM/knowledge-neurons)
- **Relevance**: Provides neuron-level localization approach for concept probes.

### Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (2023)
- **Authors**: Anthropic (Transformer Circuits)
- **Source**: Transformer Circuits report
- **Key Contribution**: Dictionary learning yields more monosemantic features vs. raw neurons.
- **Methodology**: Sparse dictionary learning on residual stream activations.
- **Datasets**: LM activations
- **Results**: Monosemantic features appear in dictionary basis; supports sparse, decomposed representations.
- **Code Available**: Partial (SAE / dictionary learning tooling)
- **Relevance**: Directly tests whether “washing machine” is represented as a single feature or composition.

### A Mathematical Framework for Transformer Circuits (2021–2022)
- **Authors**: Anthropic (Transformer Circuits)
- **Source**: Transformer Circuits report
- **Key Contribution**: Defines the linear algebraic framework for analyzing attention and residual streams.
- **Methodology**: Theoretical decomposition of transformer computations.
- **Datasets**: N/A (theoretical)
- **Results**: Clarifies how information is routed and represented across layers.
- **Code Available**: N/A (report + tools in TransformerLens)
- **Relevance**: Framework needed to reason about where concept features can be stored.

## Themes and Synthesis

### Theme 1: Superposition and Polysemanticity
- Toy Models of Superposition argues that capacity constraints lead to feature superposition, motivating sparse decompositions.

### Theme 2: Sparse Feature Discovery
- SAEs and dictionary learning (Sparse Autoencoders, Towards Monosemanticity) find interpretable features in residual streams.

### Theme 3: Causal Localization
- IOI and ROME show that specific heads/MLPs carry causal responsibility for behaviors and knowledge.

### Theme 4: Neuron-Level Attribution
- Knowledge Neurons shows individual neuron sets can be linked to factual knowledge.

## Research Gaps
- Limited direct evidence for compositional concepts like “washing machine” in residual streams.
- Need explicit comparison between atomic (“washing”, “machine”) vs. composite (“washing machine”) feature activations.
- Sparse feature methods are promising but need targeted compositionality probes.

## Recommendations for Our Experiment

- **Recommended datasets**: WikiText-2 (fast activation sampling), LAMBADA (long-context prompts), CounterFact (knowledge probes).
- **Recommended baselines**: Neuron attribution (Knowledge Neurons), causal tracing (ROME), IOI-style activation patching.
- **Recommended metrics**: Feature activation sparsity, causal effect size from patching, concept separation score (e.g., cosine separation of “washing” vs “washing machine”).
- **Methodological considerations**:
  - Compare concept activation under isolated vs. composite prompts.
  - Use SAE features as candidate concept directions, then test causal impact via patching/ablation.
  - Analyze across multiple layers and resid locations (pre/post MLP).
