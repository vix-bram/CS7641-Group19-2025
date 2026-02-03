# CS7641-group-19

## P.R.O.T.O.N: Predicting Regulatory Outcomes with Transformers and Omics-based Networks

### Introduction/Background
Understanding gene function is important to cancer research, yet accurately predicting gene roles remains a challenge due to the complexity of regulatory mechanisms, protein interactions, and contextual dependencies in biological literature [1]. Traditional approaches rely on experimental validation and single-modality computational methods, which fail to integrate the diverse biological signals required for gene function annotation [2].  The Enformer model [3] enhances gene expression prediction by capturing long-range regulatory interactions, while protein language models such as ProteinBERT [4] extract functional insights from amino acid sequences. Additionally, large language models (LLMs) like BioGPT [5] and SciBERT [6] extract knowledge from literature, offering contextual insights into gene-disease associations and molecular functions. 

### Dataset Description
We utilize multi-modal biological data for human genes, including RNA-Seq profiles from TCGA (gene expression) [7], DNA sequences from GRCh38 regulatory context [8], and protein sequences from Ensembl (structural insights) [8]. RNA-Seq data provides normalized expression levels, while genomic and proteomic data offer complementary regulatory and functional features. 

### Problem Definition
Classifying gene functions in cancer is challenging due to the reliance on single-source models, which fail to capture the complexity of gene functions. Existing models struggle with: 

- **One to Many Problem**: A single gene has multiple functions across different contexts, but most models fail to capture this variability.
- **Many To One Problem**: Different genes can share similar functions, yet models often miss these functional redundancies.
- **Degenerate DNA Codes**: Many models ignore ambiguous nucleotide symbols (e.g., Y, S), reducing accuracy in sequence-based predictions.

Our solution aims to build a multi-modal ML model that integrates diverse data sources to improve classification accuracy and uncover novel functional associations by capturing both contextual gene function and by recognizing functional redundancy. Our approach systematically combines regulatory and proteomic embeddings for a more comprehensive perspective.

### Methods
We propose the following preprocessing techniques:

- **Data Cleaning**: Remove low-expression genes, handle ambiguous bases in DNA, and ensure complete protein sequences.
- **Dimensionality Reduction**: Apply PCA or UMAP to reduce RNA-Seq variability and simplify protein/DNA embeddings.
- **Feature Engineering**: Use pretrained models (Enformer, ProteinBERT) for embeddings and engineer pathway-level features.

We will employ these machine learning algorithms:

- **Random Forest (Supervised)**: Robust and interpretable for classifying genes into functional categories, leveraging Scikit-learn’s `RandomForestClassifier`.
- **K-Means (Unsupervised)**: Effective for clustering genes into functional groups based on multi-modal embeddings using `KMeans` from Scikit-learn.
- **Gradient Boosting (Supervised)**: Handles non-linear relationships with strong performance, using XGBoost’s `XGBClassifier`.

### Results and Discussion
The quantitative metrics include:

- **Accuracy**: Measures overall correctness but may be misleading for imbalanced datasets.
- **Precision**: Avoids false positives, critical for accurate oncogene classification.
- **Recall**: Ensures no tumor suppressor genes are missed in the analysis.
- **F1 Score**: Balances precision and recall, ideal for imbalanced data.
- **Silhouette Score**: Evaluates cluster quality for unsupervised models.

From our supervised learning methods, we believe Enformer-based genomic embeddings with proteomic features from ProteinBERT will provide a more complete functional characterization, improving both classification accuracy and interpretability. For our unsupervised approaches, we anticipate that genes with similar functional roles will cluster together based on multi-modal embeddings. Biological pathway enrichment analysis will validate novel associations.

To ensure sustainability, we will use publicly available datasets to minimize the need for new data collection, reducing resource consumption. We will optimize computational efficiency through transfer learning and dimensionality reduction, lowering the environmental impact of training deep learning models. We will also make sure to safeguard any personally identifiable information. Additionally, we will use diverse datasets where possible and if not possible, we will acknowledge this limitation.

### References

[1] Vogelstein, B., Papadopoulos, N., Velculescu, V. E., Zhou, S., Diaz, L. A., Jr, & Kinzler, K. W. (2013). Cancer genome landscapes. *Science (New York, N.Y.)*, 339(6127), 1546–1558. [https://doi.org/10.1126/science.1235122](https://doi.org/10.1126/science.1235122)

[2] Greene, C. S., Tan, J., Ung, M., Moore, J. H., & Cheng, C. (2014). Big data bioinformatics. *Journal of cellular physiology*, 229(12), 1896–1900. [https://doi.org/10.1002/jcp.24662](https://doi.org/10.1002/jcp.24662)

[3] Avsec, Ž., Agarwal, V., Visentin, D. et al. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nat Methods*, 18, 1196–1203. [https://doi.org/10.1038/s41592-021-01252-x](https://doi.org/10.1038/s41592-021-01252-x)

[4] Brandes, N., Ofer, D., Peleg, Y., Rappoport, N., & Linial, M. (2022). ProteinBERT: A universal deep-learning model of protein sequence and function. *Bioinformatics*, 38(8), 2102–2110. [https://doi.org/10.1093/bioinformatics/btac020](https://doi.org/10.1093/bioinformatics/btac020)

[5] Luo, R., Sun, L., Xia, Y., et al. (2022). BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining. *arXiv preprint arXiv:2210.10341*. [https://arxiv.org/abs/2210.10341](https://arxiv.org/abs/2210.10341)

[6] Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A Pretrained Language Model for Scientific Text. *arXiv preprint arXiv:1903.10676*. [https://arxiv.org/abs/1903.10676](https://arxiv.org/abs/1903.10676)

[7] National Cancer Institute, “Genomic Data Commons Data Portal,” 2025. [Online]. Available: [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/). [Accessed: Feb. 21, 2025].

[8] Ensembl, “Homo sapiens - Genome assembly and annotation,” 2024. [Online]. Available: [https://useast.ensembl.org/Homo_sapiens/Info/Index?db=core](https://useast.ensembl.org/Homo_sapiens/Info/Index?db=core). [Accessed: Feb. 21, 2025].

### Project Proposal video:

<iframe width="560" height="315" src="https://www.youtube.com/embed/IZDXh4Ws4j4?si=kTwVJcB2Ht92_EN7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


### Gantt Chart
[View the Chart](https://gtvault-my.sharepoint.com/:x:/g/personal/ssuresh317_gatech_edu/EQj4-E6nTOVLtsVnny82GdoB655wx6uv04a6VxHP7BPTZw?e=K90yiM)


### Contributions Table

| Name            | Proposal Contribution             |
|---------------|---------------------------------|
| Khushi Vora   | Proposal Report Writeup        |
| Samyukta Singh | Literature Review + Presentation |
| Sanjana Suresh | Proposal Report Writeup + Gantt Chart |
| Vikram Kaushik | GitHub Page + Contribution Table |

