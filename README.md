# Quantum Machine Learning for Healthcare: A Comparative Study of Classical and Quantum Kernel Support Vector Machines for Breast Cancer Classification

## Abstract

Breast cancer classification is an important healthcare application in which distinguishing malignant from benign masses can support clinical assessment. This project investigates the application of quantum machine learning to this problem by comparing a classical Support Vector Machine (SVM) with a quantum kernel SVM. The Wisconsin Diagnostic Breast Cancer dataset, containing 569 samples and 30 numerical features, was used. Four features were selected using mutual information, and both models were evaluated using the same stratified training and test split. The classical model used a radial basis function (RBF) kernel, while the quantum model employed a four-qubit ZZ feature map to construct a fidelity-based quantum kernel. The classical SVM achieved 93.01% accuracy and a ROC-AUC of 0.9904, outperforming the quantum model, which achieved 91.61% accuracy and a ROC-AUC of 0.9254. Both models achieved a sensitivity of 83.02%. The quantum model required additional computation for state preparation and kernel construction and did not demonstrate a predictive or computational advantage in this experiment. These findings illustrate the feasibility of implementing a quantum kernel learning approach for healthcare classification while highlighting the importance of fair benchmarking, resource evaluation, and further investigation.

## 1. Introduction

Machine learning is increasingly used to investigate healthcare problems involving the classification of medical data. One relevant application is breast cancer classification, where computational models can distinguish between malignant and benign masses based on measurable characteristics. Although machine learning models can assist research into diagnostic methods, their reliability and limitations must be carefully evaluated, particularly when false-negative predictions may have serious consequences.

Classical machine learning techniques, including Support Vector Machines (SVMs), are widely used for classification tasks involving structured numerical data. Quantum machine learning introduces quantum circuits into machine learning workflows to represent data and calculate quantities used for prediction. One approach is quantum kernel learning, in which a quantum circuit maps input data into quantum states and calculates similarities between those states.

The potential of quantum machine learning motivates an investigation into whether quantum feature representations can improve classification performance. However, the use of a quantum circuit does not automatically guarantee better accuracy or computational efficiency. Its value must be established through a controlled comparison with an appropriate classical baseline.

This project compares a classical RBF SVM with a quantum kernel SVM for breast cancer classification. Both approaches use the same selected features and data partition, allowing their predictive performance and computational costs to be compared.

### 1.1 Research Objective

The primary objective is to evaluate whether a quantum kernel SVM can provide competitive classification performance compared with a classical RBF SVM when applied to the same healthcare dataset.

The study addresses the following research question:

*Can a quantum feature map improve the classification of malignant and benign breast masses compared with a classical RBF kernel, and how does its computational cost affect its practical usefulness?*

The investigation considers predictive performance, false-positive and false-negative predictions, and computational resource requirements.

## 2. Dataset and Preprocessing

### 2.1 Dataset Description

The Wisconsin Diagnostic Breast Cancer (WDBC) dataset was obtained through the scikit-learn dataset interface. It contains 569 samples, each described by 30 numerical features derived from digitized fine-needle aspiration images of breast masses. The features describe characteristics of cell nuclei, including radius, perimeter, area, texture, and concavity.

The dataset contains 357 benign cases and 212 malignant cases. The classification task was defined as binary prediction, with malignant cases designated as the positive class.

### 2.2 Training and Test Split

The dataset was divided into training and test sets using an 80:20-style split, implemented as 75% training data and 25% test data. Stratification was applied to preserve the class distribution, and a fixed random seed of 42 was used for reproducibility.

The resulting training set contained 426 samples, while the test set contained 143 samples.

### 2.3 Feature Selection

To reduce the computational demands of the quantum circuit, the original 30 features were reduced to four using the `SelectKBest` method with mutual information scoring. Feature selection was fitted exclusively on the training data to prevent information from the test set influencing the selected features.

The selected features were:

1. Mean concave points.
2. Worst radius.
3. Worst perimeter.
4. Worst area.

Both models used the same four features, ensuring that their comparison was based on equivalent input information.

The classical model used standardization through `StandardScaler`. The quantum model used `MinMaxScaler` to map the selected features to rotation angles between zero and π. This difference reflects the distinct input requirements of the two approaches.

## 3. Methodology

### 3.1 Classical Baseline: RBF Support Vector Machine

The classical baseline was an SVM with a radial basis function kernel. An SVM identifies a decision boundary that separates samples belonging to different classes. The RBF kernel enables nonlinear classification by measuring similarity between samples in a transformed feature space.

The model was configured with a regularization parameter of C = 1.0 and a gamma value of `scale`. Probability estimation was enabled to obtain class probabilities for ROC-AUC evaluation.

The classical model served as the baseline because SVMs are suitable for small, structured datasets and can model nonlinear decision boundaries without requiring a large neural network.

### 3.2 Quantum Kernel Support Vector Machine

The quantum model used a quantum fidelity kernel implemented with Qiskit. Rather than replacing the SVM decision rule, the quantum approach replaced the classical similarity function with a similarity measure calculated from quantum states.

The four selected input features were scaled and encoded using a four-qubit ZZ feature map with one repetition and linear entanglement. The circuit generated a quantum state for each sample.

The quantum kernel was calculated using the squared magnitude of the inner product between two encoded states:

K(x,z) = |⟨φ(x)|φ(z)⟩|²

Here, φ(x) and φ(z) represent the quantum states associated with input samples x and z. The resulting kernel values measure the fidelity between the encoded states and form the entries of the kernel matrix.

The training kernel matrix had dimensions of 426 × 426, while the test kernel matrix had dimensions of 143 × 426. These matrices were passed to an SVM using a precomputed kernel.

### 3.3 Quantum Simulation

The quantum feature map was evaluated using Qiskit's statevector simulator. This allowed the quantum circuit to be tested without access to a physical quantum processor.

The implementation therefore demonstrates a simulated quantum machine learning workflow. It does not demonstrate execution on quantum hardware or establish a quantum computational advantage.

### 3.4 Evaluation Metrics

Both models were evaluated using the same test set. The following metrics were calculated:

- **Accuracy:** proportion of all test samples classified correctly.
- **Precision:** proportion of predicted malignant cases that were actually malignant.
- **Sensitivity:** proportion of malignant cases correctly identified.
- **Specificity:** proportion of benign cases correctly identified.
- **F1-score:** harmonic mean of precision and sensitivity.
- **ROC-AUC:** ability to distinguish between the two classes across different classification thresholds.
- **Confusion matrix:** counts of correct and incorrect predictions for each class.

Computational measurements included model-fitting time, prediction time, quantum state preparation time, kernel construction time, qubit count, circuit depth, and kernel matrix size.

## 4. Results

### 4.1 Predictive Performance

The following results were obtained from the held-out test set.

| Metric | Classical RBF SVM | Quantum Kernel SVM |
|---|---:|---:|
| Accuracy | 93.01% | 91.61% |
| Precision | 97.78% | 93.62% |
| Sensitivity | 83.02% | 83.02% |
| Specificity | 98.89% | 96.67% |
| F1-score | 0.898 | 0.880 |
| ROC-AUC | 0.9904 | 0.9254 |
| False negatives | 9 | 9 |
| False positives | 1 | 3 |

The classical RBF SVM outperformed the quantum kernel SVM in accuracy, precision, specificity, F1-score, and ROC-AUC. Both models achieved the same sensitivity, identifying 44 of the 53 malignant cases and missing nine.

The classical model achieved an accuracy of 93.01%, compared with 91.61% for the quantum model. The absolute difference was 1.40 percentage points in favour of the classical model.

The ROC-AUC difference was −0.0650 when calculated as quantum ROC-AUC minus classical ROC-AUC. This indicates that the classical model demonstrated stronger discrimination between malignant and benign cases across classification thresholds on this test set.

### 4.2 Confusion Matrix Analysis

The classical model correctly classified 89 of the 90 benign cases and 44 of the 53 malignant cases. It produced one false positive and nine false negatives.

The quantum model correctly classified 87 benign cases and 44 malignant cases. It produced three false positives and nine false negatives.

Both models therefore missed the same number of malignant cases at their default classification thresholds. However, the quantum model incorrectly classified two additional benign cases as malignant.

In a healthcare context, false-negative predictions deserve particular attention because they represent malignant cases classified as benign. Although both models achieved identical sensitivity in this experiment, neither identified all malignant cases. Further work would be required to investigate alternative thresholds and approaches that might improve sensitivity while maintaining acceptable specificity.

### 4.3 Computational Resource Comparison

The recorded computational measurements are summarized below.

| Measurement | Classical SVM | Quantum Kernel SVM |
|---|---:|---:|
| Model-fitting time | 0.0118 s | 0.0200 s |
| Prediction time | 0.0039 s | 0.0019 s |
| State preparation time | Not applicable | 0.9308 s |
| Kernel construction time | Included in standard model computation | 0.0094 s |
| Qubits | Not applicable | 4 |
| Feature map depth | Not applicable | 11 |

The classical model required approximately 0.0118 seconds for fitting and 0.0039 seconds for test inference. The quantum model required approximately 0.0200 seconds for SVM fitting and 0.0019 seconds for prediction, in addition to state preparation and kernel construction.

The quantum pipeline's measured total time, calculated from state preparation, kernel construction, and SVM fitting, was approximately 0.9602 seconds. This was substantially longer than the measured classical fitting and inference time combined.

These measurements should be interpreted as observations from one execution rather than definitive hardware-independent benchmarks. In particular, the statevector simulator performs quantum-state calculations on a classical computer. The measured simulation time does not represent the execution time of a physical quantum processor.

The kernel matrices contained 181,476 training entries and 60,918 test entries. These are matrix-entry counts, not counts of physical quantum executions. A hardware implementation would require a separate accounting of circuit executions, measurement shots, noise, and device access overhead.

## 5. Discussion

### 5.1 Interpretation of the Findings

The results show that the quantum kernel SVM was functional and achieved competitive, although lower, predictive performance than the classical RBF SVM. The experiment demonstrates that a quantum feature map can be incorporated into a conventional SVM classification workflow for a healthcare-related problem.

However, the findings do not support the hypothesis that the selected quantum kernel improved classification performance. The classical model achieved a higher ROC-AUC, higher accuracy, and fewer false positives, while sensitivity remained unchanged.

One possible explanation is that the RBF kernel was already effective at modelling the relationships among the selected features. The quantum feature map did not provide a more useful similarity representation for this particular configuration and dataset. This is an interpretation of the observed results, not proof that quantum kernels are generally less effective.

### 5.2 Purpose and Potential of Quantum Integration

The purpose of the quantum integration was to investigate whether encoding the input data into quantum states and calculating fidelity-based similarities could offer an alternative representation for classification.

Quantum feature maps can represent relationships among input variables through quantum state amplitudes and entangling operations. In principle, these representations may be useful for some learning problems. Nevertheless, the benefit depends on the dataset, feature map, circuit configuration, and comparison method.

In this experiment, the quantum kernel did not produce a measurable predictive improvement over the classical baseline. Its contribution was therefore methodological: it enabled a controlled investigation of a quantum machine learning technique and its practical limitations.

### 5.3 Computational Tradeoffs

The experiment also illustrates the importance of accounting for the cost of quantum integration. Although the quantum model's SVM prediction step was faster in the recorded run, state preparation and kernel construction increased the total measured pipeline time.

Furthermore, the quantum computation was simulated classically. The experiment cannot establish whether a physical quantum implementation would be faster, slower, or more accurate. Such a claim would require appropriately controlled comparisons and an explicit analysis of hardware resources and measurement overhead.

### 5.4 Limitations

Several limitations affect the conclusions.

First, the experiment used a single stratified train/test split. The results may vary under different data partitions, and further cross-validation is required to assess robustness.

Second, only four features were selected from the original 30. Although this reduction made the quantum circuit more manageable, it may have excluded useful information.

Third, the model configurations were not exhaustively tuned. The observed performance reflects the selected configurations and should not be interpreted as the maximum achievable performance of either approach.

Fourth, the quantum model was evaluated with an ideal statevector simulator rather than physical quantum hardware. Hardware noise, measurement uncertainty, and execution costs were therefore not represented.

Finally, the dataset is relatively small and does not establish generalization to broader clinical populations. Neither model should be treated as a clinical diagnostic system.

## 6. Conclusion

This project implemented and evaluated classical and quantum kernel machine learning approaches for distinguishing malignant from benign breast masses using the Wisconsin Diagnostic Breast Cancer dataset.

The classical RBF SVM achieved 93.01% accuracy and a ROC-AUC of 0.9904, outperforming the quantum kernel SVM, which achieved 91.61% accuracy and a ROC-AUC of 0.9254. Both models achieved a sensitivity of 83.02%, with nine false-negative predictions each. The quantum model also produced more false positives and required additional time for simulated state preparation and kernel construction.

The findings demonstrate the feasibility of implementing a four-qubit quantum kernel classifier for a healthcare classification problem, but they do not demonstrate a predictive or computational advantage over the selected classical baseline.

Future work should investigate stratified cross-validation, alternative feature subsets, different quantum feature maps, hyperparameter tuning, and noise-aware quantum simulation. Evaluation on physical quantum hardware could further clarify the practical resource requirements of the approach.

Overall, the study highlights the importance of evaluating quantum machine learning using both predictive metrics and computational resource measurements. Quantum integration should be justified by measurable benefits rather than by the use of quantum circuits alone. Further validation would be required before either model could be considered for clinical applications.

## References

1. Wolberg, W. H., Street, W. N., & Mangasarian, O. L. *Breast Cancer Wisconsin (Diagnostic) Data Set*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

2. scikit-learn developers. *Breast cancer Wisconsin (diagnostic) dataset*. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

3. Qiskit contributors. *Qiskit documentation*. https://docs.quantum.ibm.com/

4. scikit-learn developers. *Support Vector Machines*. scikit-learn documentation. https://scikit-learn.org/stable/modules/svm.html
