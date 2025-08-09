# Customer Purchase Prediction with PyCaret and LightGBM

## Problem Statement and Goal of Project
This project aims to predict customer purchase behavior (choosing between two juice brands, CH or MM) based on historical purchase data. The goal is to build a robust classification model to understand factors influencing customer decisions, such as price, discounts, store specifics, and brand loyalty, to support targeted marketing strategies.

## Solution Approach
The project leverages **PyCaret**, an automated machine learning library, to streamline the model-building process, and **LightGBM**, a gradient boosting framework, for efficient and high-performance classification. The approach includes:

1. **Data Preprocessing**: Loading the `juice` dataset, encoding categorical variables (e.g., `Purchase` and `Store7`), and preparing features for modeling.
2. **PyCaret Setup**: Configuring a classification experiment with stratified k-fold cross-validation (10 folds) to ensure robust evaluation.
3. **LightGBM Model**: Training a custom LightGBM model with parameters optimized for binary classification (AUC metric).
4. **Model Evaluation**: Using PyCaret's tools to evaluate model performance, including confusion matrices and other metrics, and analyzing a Voting Classifier's predictions.
5. **Exploration with LDA**: Evaluating a Linear Discriminant Analysis (LDA) model interactively to explore alternative approaches.

This project demonstrates my ability to combine automated ML workflows with custom model tuning, showcasing both efficiency and hands-on model development.

## Technologies & Libraries
- **Python**: Core programming language (version 3.9.21).
- **PyCaret**: For automated machine learning setup, model comparison, and evaluation.
- **LightGBM**: For efficient gradient boosting classification.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For label encoding (`LabelEncoder`).
- **Jupyter Notebook**: For interactive development and visualization.

## Description about Dataset
The `juice` dataset, sourced from PyCaret's built-in datasets, contains 1070 records of customer purchase data with 19 features, including:
- **Target Variable**: `Purchase` (binary: CH or MM, indicating the chosen juice brand).
- **Key Features**:
  - `PriceCH`, `PriceMM`: Prices of CH and MM brands.
  - `DiscCH`, `DiscMM`: Discounts applied to CH and MM.
  - `SpecialCH`, `SpecialMM`: Indicators for special promotions.
  - `LoyalCH`: Customer loyalty score for the CH brand.
  - `SalePriceCH`, `SalePriceMM`: Final sale prices after discounts.
  - `PriceDiff`, `ListPriceDiff`: Price differences between brands.
  - `StoreID`, `Store7`, `STORE`: Store-related identifiers.
  - `WeekofPurchase`: Week of the purchase.
  - `PctDiscCH`, `PctDiscMM`: Percentage discounts for each brand.

The dataset is well-suited for binary classification, with numerical and categorical features influencing purchase decisions.

## Installation & Execution Guide
To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/imehranasgari/customer-purchase-prediction.git
   cd customer-purchase-prediction
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure Python 3.9+ is installed, then run:
   ```bash
   pip install pandas pycaret lightgbm scikit-learn jupyter
   ```

4. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook Pycart-Classification.ipynb
   ```

5. **Execute Cells**: Run all cells in the notebook to preprocess the data, train models, and visualize results.

*Note*: Ensure you have sufficient memory and computational resources, as LightGBM and PyCaret may require significant processing for large datasets.

## Key Results / Performance
The Voting Classifier, evaluated on the test set (321 samples), achieved the following performance metrics:
- **Accuracy**: 82.55%
- **AUC**: 82.09%
- **Recall**: 80.00%
- **Precision**: 76.34%
- **F1 Score**: 78.12%
- **Kappa**: 63.63%
- **MCC**: 63.68%

These results indicate a strong model performance in predicting customer purchases, with a balanced trade-off between precision and recall. The confusion matrix (visualized in the notebook) provides further insight into true positives, true negatives, false positives, and false negatives, highlighting the model's ability to distinguish between CH and MM purchases.

## Screenshots / Sample Outputs
Due to the interactive nature of the outputs (e.g., confusion matrix plot, evaluation widgets), they are best viewed within the Jupyter Notebook. Below is a brief description of key outputs:
- **Confusion Matrix**: Visualized for the Voting Classifier, showing the distribution of predicted vs. actual purchase labels.
- **Model Evaluation**: Interactive widget for the LDA model, allowing exploration of metrics like ROC curves and feature importance.
- **Data Snapshot**: The first five rows of the `juice` dataset are displayed, showcasing the feature structure.

> 💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. Please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*

## Additional Learnings / Reflections
This project was an opportunity to deepen my understanding of automated machine learning with PyCaret and gradient boosting with LightGBM. Key learnings include:
- **PyCaret's Efficiency**: PyCaret simplified the preprocessing, model selection, and evaluation pipeline, allowing rapid experimentation while maintaining robust cross-validation.
- **LightGBM Customization**: Tuning LightGBM parameters (e.g., `num_leaves`, `learning_rate`) highlighted the importance of balancing model complexity and generalization.
- **Exploration of LDA**: Evaluating an LDA model interactively demonstrated its utility for simpler datasets and provided a contrast to tree-based methods like LightGBM.
- **Feature Importance**: Features like `LoyalCH`, `PriceDiff`, and `SalePrice` likely played significant roles in predictions, aligning with intuitive business logic for customer behavior.

While the Voting Classifier achieved solid performance, experimenting with simpler models like LDA underscores my commitment to exploring diverse approaches to understand their strengths and limitations. This project reflects my ability to combine automated tools with custom model development for practical business applications.

## 👤 Author
**Mehran Asgari**  
**Email**: [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)  
**GitHub**: [https://github.com/imehranasgari](https://github.com/imehranasgari)

## 📄 License
This project is licensed under the MIT License – see the `LICENSE` file for details.
```