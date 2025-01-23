# Diamond Price Prediction Project

## Project Overview
💎 This project aims to predict the prices of diamonds based on their features using machine learning models. The project demonstrates a complete machine learning pipeline, including data ingestion, preprocessing, transformation, and model training.

## Dataset Details
The dataset is sourced from [Kaggle's Gem Price Prediction](https://www.kaggle.com/code/ramakrishnanthiyagu/gem-price-prediction). It contains the following features:
- **Categorical Features**: `cut`, `color`, `clarity`
- **Numerical Features**: `carat`, `depth`, `table`, `x`, `y`, `z`
- **Target Feature**: `price`

The dataset was split into training (70%) and testing (30%) datasets.

## Artifacts
📂 The following artifacts are generated and used in the project:
- `raw.csv`: Raw dataset.
- `train.csv` and `test.csv`: Training and testing datasets.
- `preprocessor.pkl`: Preprocessing pipeline for data transformation.
- `model.pkl`: Trained machine learning model.

## Technologies Used
- **Programming Language**: Python 🐍
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Preprocessing: `scikit-learn`
  - Logging and Exception Handling: `logging`, `sys`

## Project Workflow

### 1. Data Ingestion 📥
- Reads the raw dataset (`gemstone.csv`).
- Saves raw, training, and testing data into `artifacts/` directory.
- Splits the dataset into training and testing sets with a 70-30 split.

### 2. Data Transformation 🔄
- Handles missing values using `SimpleImputer`.
- Scales numerical features using `StandardScaler`.
- Encodes categorical features (`cut`, `color`, `clarity`) using `OrdinalEncoder` with predefined category rankings.
- Combines pipelines into a `ColumnTransformer` for seamless preprocessing.
- Saves the preprocessor object as `preprocessor.pkl`.

### 3. Model Training 🤖
- Models Used:
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
  - ElasticNet
  - Decision Tree Regressor
- Evaluates models using R² score.
- Selects the best model based on the highest R² score.
- Saves the best model as `model.pkl`.

### 4. Prediction Pipeline 🧪
The prediction pipeline (`PredictPipeline`) includes:
- Loading the saved preprocessor and model objects.
- Scaling input features and making predictions.
- Handles both manual input and data from a DataFrame.

## Results 📊
- The best-performing model was identified based on R² score.
- The R² scores for all models are logged and reported.

## Installation Instructions ⚙️
1. Clone the repository:
   ```bash
   git clone https://github.com/AryanDhanuka10/Diamond_Price_Prediction/tree/main
   ```
2. Navigate to the project directory:
   ```bash
   cd diamond-price-prediction
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage 🚀
1. Run the data ingestion module to generate train and test datasets:
   ```bash
   python src/data_ingestion.py
   ```
2. Run the data transformation module to preprocess data:
   ```bash
   python src/data_transformation.py
   ```
3. Train models and select the best model:
   ```bash
   python src/model_trainer.py
   ```
4. Use the saved model (`model.pkl`) for predictions:
   ```python
   from src.prediction_training import PredictPipeline, CustomData

   custom_data = CustomData(carat=0.5, depth=61.5, table=55, x=4.5, y=4.6, z=2.9, cut="Ideal", color="E", clarity="VVS1")
   df = custom_data.get_data_as_dataframe()

   predict_pipeline = PredictPipeline()
   prediction = predict_pipeline.predict(df)
   print("Predicted Price:", prediction)
   ```

## Challenges and Learnings 🧠
- **Challenge**: Handling categorical data encoding with ordinal features.
  - **Solution**: Used `OrdinalEncoder` with custom category rankings for categorical features.
- **Challenge**: Ensuring robust preprocessing for both training and testing datasets.
  - **Solution**: Implemented a unified `ColumnTransformer` pipeline.

## Future Scope 🌟
- Extend the project to include additional models such as Random Forest or Gradient Boosting.
- Develop a web application for real-time price prediction.

## Acknowledgments 🙏
- [Kaggle Dataset](https://www.kaggle.com/code/ramakrishnanthiyagu/gem-price-prediction) for providing the diamond dataset.
- Various Python libraries and tools used in the project.

## License 📜
This project is licensed under the MIT License.

