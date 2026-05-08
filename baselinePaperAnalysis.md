## Research Paper Analysis: Associative Syntax and Maximal Repetitions reveal context-dependent complexity in fruit bat communication

This report provides a detailed analysis of the preprint "Associative Syntax and Maximal Repetitions reveal context-dependent complexity in fruit bat communication" by Luigi Assom. The paper explores novel unsupervised methods to understand the communication complexity in graded vocal systems, using fruit bats as a case study.

### 1. Authors, Institution(s), and Notable Context

The sole author of this paper is **Luigi Assom**, affiliated with the **Department of Computer and Systems Sciences, Stockholm University**, Stockholm, Sweden. The author is identified as an alumnus of the department, indicating that this work likely originated from their graduate studies, specifically a Master's thesis, as explicitly acknowledged in the "Acknowledgments and Disclosure of Funding" section.

This work was presented at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop on AI for Non-Human Animal Communication. NeurIPS is a highly prestigious conference in the field of artificial intelligence and machine learning, and its workshops are recognized forums for presenting cutting-edge, often interdisciplinary, research. While this specific workshop submission is non-archival, its acceptance indicates recognition by the machine learning community for its novelty and relevance to AI applications in animal communication.

The author acknowledges financial aid from the Earth Species Project and the Department of Computer and Systems Sciences at Stockholm University for supporting the presentation of this work. The Earth Species Project is a non-profit organization dedicated to decoding animal communication using AI, which aligns perfectly with the subject matter of this paper. This connection suggests that Assom's work is part of a broader, collaborative effort at the intersection of AI, bioacoustics, and animal behavior, and benefits from the expertise and resources of leading organizations in the field. The public availability of the associated GitHub repository for the original Master's thesis further underscores a commitment to open science and reproducibility, although a cleaner, updated version is noted as being under development.

### 2. How This Work Fits into the Broader Research Landscape

Quantifying communication complexity in species with "graded vocal systems" – where vocalizations are continuous and highly modulated rather than composed of clearly discrete units – remains a significant challenge in animal communication research. Traditional methods often assume discrete units, which perform poorly for species like fruit bats, mice, or even human phonemes where vocalizations blend or overlap. This paper directly addresses this gap by developing and refining an unsupervised computational pipeline specifically tailored for such graded systems.

The research situates itself within the context of the "social complexity hypothesis for communicative complexity (SCHCC)," which posits a relationship between social and communicative complexity. However, the author critically points out a "circularity pitfall" in SCHCC, where definitions of communication and sociality can be non-independent. To mitigate this, the paper seeks to extend quantitative, information-theoretic metrics of communication complexity, specifically by introducing "Maximal Repeats (MRs)" as a novel metric for combinatorial complexity.

The work builds upon and critically evaluates existing methods:
*   **Sainburg et al. [13, 10, 11]:** These works utilize manifold learning (e.g., UMAP) to cluster vocal units from spectrograms. Assom's paper improves upon this baseline by systematically investigating how dimensionality reduction on mel-spectrograms affects clustering performance, particularly for graded vocalizations. The paper demonstrates an improvement from an original baseline that discriminates only two vocal unit types to identifying seven distinct types.
*   **Zhang et al. [15]:** This research analyzes syntax in horseshoe bats using behavioral classifiers, but it requires ground-truth syllable labels from experts. Assom's work extends this by adapting the method to use *automatically labeled* syllables and applying it to multiple behavioral contexts, thereby enhancing scalability and reducing reliance on labor-intensive expert annotation.

Furthermore, the paper draws inspiration from computational linguistics and genetics. The concept of Maximal Repeats, previously applied to written texts to study long-range dependencies and compressibility [4], is introduced to animal communication for the first time. This innovative cross-disciplinary application aims to capture combinatorial capacity that traditional information theory metrics like Shannon entropy might miss, as entropy primarily measures uncertainty in repertoire draws and doesn't fully capture long-range dependencies. The analogy to genetics, where limited repertoires (nucleotides) encode complex information (protein expressions), further motivates the use of MRs to understand how graded vocalizations might convey complex meaning. By exploring information decay patterns (exponential vs. power-law), the study aligns with comparative research in birdsong and human speech [12], suggesting potential common mechanisms across phylogenetically distant species.

### 3. Key Objectives and Motivation

The overarching motivation of this study is to overcome limitations in quantifying communication complexity in species with graded vocal systems. The paper aims to improve and extend unsupervised pipelines for inferring vocal repertoire and syntax, applying these advancements to fruit bat vocalizations as a representative case.

The research explicitly addresses two primary research questions (RQs):

*   **RQ1: How does dimensionality reduction affect unsupervised clustering on manifold learning for quantifying size and diversity of the repertoire?** This question focuses on refining the methodology for automatically identifying discrete vocal units (syllables) from continuous acoustic data, acknowledging the inherent challenges of graded vocal systems. The motivation here is to establish a more robust and accurate unsupervised labeling pipeline.
*   **RQ2: How do syntax and temporal structure encode contextual information?** This question delves into understanding how the identified vocal units are combined and organized over time to convey meaning, specifically in relation to different behavioral contexts. This involves investigating the type of syntax (associative vs. combinatorial) and the presence of complex temporal patterns.

The paper outlines four specific contributions that serve as key objectives:

1.  **Refinement of unsupervised pipeline:** Developing an improved unsupervised pipeline for repertoire quantification in graded vocal systems, surpassing previous baselines (e.g., [13]) and yielding results consistent with expert knowledge [1].
2.  **Context-dependent syntax analysis:** Adapting existing methods ([15]) to analyze context-dependent syntax using automatically labeled syllables across multiple behavioral contexts.
3.  **Novel application of Maximal Repeats:** Introducing Maximal Repeats (MRs) to animal communication as a metric for combinatorial complexity, providing evidence for heavy-tailed distributions in vocalizations.
4.  **Findings on communicative complexity:** Demonstrating that communicative complexity (indicated by longer MRs) is higher in conflict-related behaviors compared to cooperative ones, suggesting a link between social disagreement and signal complexity.

### 4. Methodology and Approach

The study employs a two-pronged experimental design: (1) unsupervised labeling to infer repertoire size and diversity, and (2) behavioral classification and statistical analysis of syllabic sequences to infer syntax type and temporal structures across behaviors. The dataset used is the annotated fruit bat vocalization dataset from Prat et al. [9], which includes emitter, addressee, and behavioral context labels for 41 specimens. Ambiguous contexts (Generic, Sleeping, Unknown) were excluded.

#### 4.1. Size and Diversity of Repertoire (Addressing RQ1)

This experiment aimed to optimize the unsupervised labeling of vocal units. The pipeline, inspired by Sainburg et al. [13], involved:
*   **Spectrogram Generation:** Mel-spectrograms (or their autoencoder representations) were created from audio segments. The study systematically varied spectrogram settings (time-frequency trade-offs) and dimensionality reduction techniques (PCA on Autoencoder latent representations with different AE architectures) to explore clustering performance.
*   **Segmentation:** Comparison between the original fixed noise floor segmentation [9] and Dynamic Threshold Segmentation [12], which dynamically estimates the noise floor and can isolate shorter sub-units (Table 2 details specific audio preprocessing parameters like bandpass, noise-removal, STFT, MFCCs, MEL-filterbank settings).
*   **Manifold Learning and Clustering:** UMAP (Uniform Manifold Approximation and Projection) was used to project spectrograms into a low-dimensional space, followed by HDBSCAN (Hierarchical Density-Based Spatial Clustering) for clustering.
*   **Evaluation:**
    *   **Internal Validation:** Silhouette Score measured HDBSCAN cluster consistency.
    *   **Agreement with Acoustic Similarity:** A proxy for ground truth was generated using Dynamic Time Warping (DTW) on Mel-Frequency Cepstral Coefficients (MFCCs) and Agglomerative Clustering. This yielded an estimated 27 ± 2 syllable types per emitter, consistent with prior research [1, 15]. The agreement between HDBSCAN labels and this proxy was measured using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI). This systematic approach aimed to identify the optimal configuration for unsupervised labeling in graded systems.

#### 4.2. Type of Syntax and Temporal Structures Conveying Contextual Information (Addressing RQ2)

This experiment used the unsupervised syllable labels to investigate syntax type, context-dependent syllable usage, and the distribution of Maximal Repeats. Three null hypotheses (HP) were tested:

*   **HP1₀: Syllable order does not affect context classification.**
    *   **Method:** A Random Forest (RF) classifier (replicated from [15]) was used to classify behavior based on 18 features engineered from syllabic sequences (e.g., syllable richness, sequence length, transition count, entropy, and various conditional probabilities, as detailed in Table 1). The classifier's performance was compared between original and permuted sequences.
    *   **Evaluation:** F1-scores were compared to determine if syllable order contributes to contextual meaning (combinatorial syntax) or if only syllable presence matters (associative syntax).

*   **HP2₀: Syllable usage is identical across behaviors.**
    *   **Evaluation:** Wilcoxon rank-sum tests were conducted on syllable frequency distributions between pairs of behavioral contexts to identify significant differences in syllable usage.

*   **HP3₀: The distribution of maximal repetitions follows an exponential distribution.**
    *   **Method:** Maximal Repeats (MRs), defined as the longest repeating subsequences, were extracted using a prefix-suffix tree algorithm. An exponential distribution would imply memory-less information decay, while a heavy-tailed distribution (e.g., power-law) would suggest long-range dependencies and combinatorial complexity.
    *   **Evaluation:** A likelihood ratio test compared exponential versus power-law distributions.
    *   **Further Analysis:** Mean MR lengths were compared across behaviors, and syllabic transition networks were constructed and qualitatively/quantitatively analyzed (e.g., graph metrics like small-world coefficients, clustering coefficient, density as shown in Table 3, Figs 5, 6).

### 5. Main Findings and Results

The study yielded several key findings regarding repertoire quantification and the nature of fruit bat communication complexity:

#### 5.1. Repertoire Size and Diversity (RQ1)

*   **Improved Clustering:** The refined unsupervised pipeline significantly improved clustering quality for continuous-type vocalizations. Specifically, coarse-graining the temporal dimension of spectrograms combined with dynamic segmentation yielded the best results (Silhouette Score > 0.5, 95% assignment accuracy). This configuration identified **seven distinct types of vocal units**, a notable improvement over the previous baseline which discriminated only two (mother-pup calls vs. adult vocalizations).
*   **Acoustic Similarity Agreement:** While the agglomerative clustering proxy (based on DTW/MFCCs) suggested an average of 27 ± 2 syllable types per emitter, consistent with existing literature, the agreement with the best HDBSCAN clustering (using mel-spectrograms retaining higher dimensionality) was moderate (Mean ARI = 0.12 ± 0.01, Mean NMI = 0.30 ± 0.01), suggesting a repertoire of approximately **14 syllables** from the unsupervised method.

#### 5.2. Syntax Type and Temporal Structures (RQ2)

*   **Syntax Type (HP1): Associative Syntax.** The permutation test revealed that the order of syllables did not significantly affect the performance of the behavioral classifier (F1-score > 0.9 for both original and permuted sequences). This result supports the rejection of HP1₀, indicating an **associative syntax** (where meaning is conveyed by the presence of syllables, not necessarily their order) rather than a combinatorial one, consistent with prior findings in fruit bats [1].
*   **Syllabic Distribution (HP2): Context-Dependent Syllable Usage.** Syllable distributions were found to be significantly different between the "Isolation" context (mother-pup interactions) and other contexts (p < 0.05, Wilcoxon rank-sum test). This aligns with known differences in mother-pup vocalizations [1]. However, for cooperative contexts such as Feeding, Grooming, and Kissing, there was generally no significant evidence to reject HP2₀, suggesting more uniform syllable usage across these specific behaviors. Heatmaps also indicated similar syllable usage among emitters from the same colony.
*   **Maximal Repeats Distribution (HP3): Heavy-Tailed Distribution.** The likelihood ratio test rejected HP3₀ (p < 0.05). The distribution of Maximal Repeat (MR) lengths was best described by a **truncated power-law** (exponent α = 1.79). This heavy-tailed distribution signifies long-range dependencies and complex temporal structures, rather than a simple memory-less process, suggesting an underlying combinatorial capacity in the syntactical patterns.
*   **Behavioral Complexity through MRs and Networks:**
    *   **MR Length:** The average length of MRs was significantly **greater in conflict-related contexts** (Mating Protest, Fighting, Threat-like) than in cooperative contexts (Feeding, Grooming, Kissing) and the Isolation context (Fig. 5).
    *   **Network Structure:** Syllabic transition networks further illuminated this complexity. Conflict-related contexts exhibited network metrics indicative of a **small-world architecture** (ω ≈ 0, Avg C > 0.4, as shown in Table 3 and Fig. 6a), characterized by high local clustering and efficient global connectivity. In contrast, cooperative contexts showed metrics suggestive of more **random, less structured networks** (ω > 0.5, lower Avg C) (Fig. 6b). The Isolation context notably displayed simple repetitions of a specific syllable, resulting in sparse, simple network graphs (Fig. 4b, Table 3: Avg C = 0.00).

### 6. Significance and Potential Impact

This study makes several significant contributions to the field of animal communication and computational ethology:

1.  **Advancement in Unsupervised Learning for Graded Vocal Systems:** The refined unsupervised pipeline offers a more effective method for quantifying repertoire and syntax in species with continuous, graded vocalizations. By demonstrating that temporal compression and dynamic segmentation aid cluster separation, the work provides crucial insights into how information might be encoded in such systems, moving beyond the limitations of methods designed for discrete vocalizations. This could generalize to a wider range of species, reducing the reliance on expert-labeled datasets and accelerating research.

2.  **Novel Metric for Combinatorial Complexity:** The introduction of Maximal Repeats (MRs) to animal communication is a highly impactful methodological innovation. By demonstrating heavy-tailed distributions of MRs in fruit bat vocalizations, the study provides a new, quantitative measure for identifying long-range dependencies and combinatorial capacity within vocal sequences. This metric offers a valuable complement to existing information-theoretic approaches, addressing the "circularity pitfall" in the SCHCC by providing a more independent measure of communicative complexity.

3.  **Context-Dependent Complexity and Social Interaction:** The finding that communicative complexity, as measured by MR length and network structure, is higher in conflict contexts than in cooperative or mother-pup interaction contexts is a major empirical result. This suggests that scenarios of social disagreement or negotiation may require more elaborate and less compressible signals. The interpretation that "higher-complexity observed in conflict-related communication may reflect lower compressibility of information conveying disagreement" offers a compelling theoretical framework for understanding the functional drivers of vocal complexity. This insight can inform hypotheses about the evolution of communication and the role of social dynamics in shaping signal structure across species.

4.  **Implications for Understanding Graded Systems:** The proposal that basic frequency-based utterances combine and are modulated in time to form more complex syllables, which are then assembled into sequences governed by combinatorial patterns, offers a speculative but compelling model for how graded vocal systems might convey rich meaning. The observed associative syntax, combined with combinatorial patterns revealed by MRs, suggests a hierarchical organization where basic units are context-dependent but their temporal arrangement, particularly in challenging social contexts, exhibits deeper structural complexity.

The potential impact of this work extends to developing more sophisticated AI tools for decoding non-human communication, advancing our understanding of cognitive abilities underlying animal vocalizations, and providing new avenues for comparative studies across diverse species. The proposed use of MRs in other species as a proxy for combinatorial capacity opens up a promising direction for future research.