## Project: FoodX-251 Food Image Classification

### Project Overview

This project tackles the challenge of large-scale food image classification using the **FoodX-251** dataset (251 classes). This dataset is characterized by a large number of images (approx. 118k in the provided training set) and the significant presence of unlabeled data (`train_unlabeled`) as well as irrelevant ("non-food") or ambiguous images within the training set.

The main objectives were:

1.  **Training Set Cleaning:** Identify and remove "non-food" or ambiguous images from the training set (labeled + unlabeled) using clustering techniques based on features extracted from a pre-trained network.
2.  **Exploration of Classification Methodologies:** Compare two main approaches:
    * Feature Extraction (from ResNet18/101) + SVM Classifier.
    * Fine-Tuning of pre-trained Convolutional Neural Networks (CNNs) (ResNet18/50/101) on the reduced labeled dataset (`train_small`), aided by Data Augmentation.
3.  **Dataset Expansion via Pseudo-Labeling:** Utilize the fine-tuned CNN model to assign labels (pseudo-labels) to the cleaned unlabeled set, selecting those with high confidence to iteratively expand the training set and retrain the model.
4.  **Analysis on Degraded Data:** Study the impact of common degradations (blur, Gaussian noise, JPEG compression) present in the validation set and evaluate the performance of models trained on artificially degraded versions of the training set.

### Methodology

1.  **Initial Cleaning via Clustering:**
    * Empirical identification of "non-food" images in the `train_unlabeled` set.
    * Feature extraction using a pre-trained **ResNet101**.
    * Application of **PCA** for dimensionality reduction (retaining 90% variance, 205 features).
    * **K-Means** clustering (k=251) on the reduced features.
    * Manual inspection of clusters to identify and remove those predominantly containing "non-food" or ambiguous images.
    * Obtained a "Cleaned Training Set" (approx. 111k images) and manually restored images removed from the `small_train_set` to maintain 20 images/class (total 5020).

2.  **Comparison of Classification Approaches (on `train_small_cleaned`):**
    * **Feature Extraction + SVM:** Features extracted from intermediate and final layers of ResNet18/101 were used to train a multi-class **SVM classifier (fitcecoc)**.
    * **CNN Fine-Tuning:** Modified the last layers of ResNet18/50/101 (added Dropout, changed output to 251 classes), froze initial layers (backbone), and trained on `train_small_cleaned`. **Data Augmentation** (rotation, reflection, translation, zoom) was used to artificially enlarge the dataset.

3.  **Iterative Pseudo-Labeling (with ResNet18):**
    * Chose CNN Fine-Tuning (ResNet18 for efficiency) as the primary method.
    * Used the trained model to predict labels on the `unlabeled_train_set_cleaned`.
    * Selected predictions with **high confidence** (tested thresholds: >=90%, >=75%, >=50%).
    * Added images with reliable pseudo-labels to the training set.
    * Managed the resulting **class imbalance** via **undersampling** (majority classes) and **oversampling** (using Data Augmentation for minority classes) before retraining.
    * Performed multiple "rounds" of pseudo-labeling and retraining.

4.  **Analysis on Degraded Data:**
    * Analyzed the validation set to quantify **blur** (Laplacian variance), **Gaussian noise** (Noise Level Estimation), and **JPEG compression** (BRISQUE, PIQE).
    * Created a "Reduced Degraded Validation Set" by removing images with excessive degradation.
    * **Artificial Degradation of Training Set:** Applied controlled Gaussian blur, Gaussian noise, and JPEG compression to portions of the training set (75% degraded, 25% clean per class).
    * Trained ResNet18 on these degraded sets and evaluated on the validation sets (clean, degraded, reduced degraded).

### Technologies Used

* **Language/Environment:** MATLAB
* **MATLAB Toolboxes:**
    * Deep Learning Toolbox (CNNs, ResNet, training, prediction, augmentation)
    * Image Processing Toolbox (image quality analysis, filters, noise, manipulation)
    * Statistics and Machine Learning Toolbox (PCA, K-Means, SVM - fitcecoc)
* **Models/Algorithms:**
    * Convolutional Neural Networks (CNN): ResNet18, ResNet50, ResNet101
    * Support Vector Machine (SVM) multi-class (ECOC)
    * K-Means Clustering
    * Principal Component Analysis (PCA)
    * Image Quality Metrics: BRISQUE, PIQE, Laplacian Variance (Blur), Noise Level Estimation
* **Main Techniques:**
    * Image Classification (Fine-grained)
    * Transfer Learning / Fine-Tuning
    * Feature Extraction
    * Unsupervised Clustering
    * Semi-Supervised Learning (Pseudo-Labeling)
    * Data Augmentation
    * Data Cleaning / Outlier Removal (clustering-based)
    * Degraded Data Analysis and Handling
    * Class Balancing (Oversampling/Undersampling)
* **Dataset:** FoodX-251 (with custom split: `train_small`, `train_unlabeled`, `validation`)

### Results Obtained

The results highlight the significant challenges posed by the dataset and the explored methodologies:

* **Feature Extraction + SVM:** Modest performance, with the best accuracy reaching **~26.0%** (ResNet101, final `pool5` layer) on the clean validation set.
* **CNN Fine-Tuning (Baseline on `train_small` clean + Augmentation):** Accuracy on the clean validation set reached **~32.3%** with ResNet18 (20 augmented images/class) and **~32.8%** with ResNet101 (80 augmented images/class), showing improvement over SVM but still limited.
* **Pseudo-Labeling (ResNet18):**
    * Adding pseudo-labels led to slight incremental improvements in accuracy on the clean validation set, peaking at **~33.5%** (3rd Round, conf >= 90%, max 180 images/class).
    * However, the increase was marginal, and issues with class imbalance and potential error propagation were noted (a final training run with all pseudo-labels reduced accuracy to **~29.9%**).
    * ResNet50 showed similar results (~33.5%) but with higher computational costs.
* **Degraded Data Analysis (ResNet18):**
    * Training on the small *degraded* set and testing on the *degraded* validation set yielded the best performance in this scenario: **~20.1%**.
    * Training on the full *degraded* set resulted in very low performance on both the clean (~1.1%) and degraded (~0.5%) validation sets, highlighting the domain mismatch.

**Key Accuracy Results Summary Table (Validation Set):**

| Train Set          | Validation Set    | Accuracy   | Note                                     |
| :----------------- | :---------------- | :--------- | :--------------------------------------- |
| Small Augmented    | Clean             | **32.32%** | Baseline Fine-Tuning ResNet18           |
| Full (Pseudo-Label)| Clean             | 29.87%     | Final attempt with all pseudo-labels     |
| Full (Pseudo-Label)| Degraded          | 18.92%     | Test on degraded data                  |
| Small Degraded     | **Degraded** | **20.12%** | Best result on degraded data           |
| Full Degraded      | Clean             | 1.10%      | Domain mismatch (degraded train)         |
| Full Degraded      | Degraded          | 0.54%      | Low performance on degraded data         |
| Full Degraded      | Reduced Degraded  | 0.86%      | Slight improvement removing worst cases  |
