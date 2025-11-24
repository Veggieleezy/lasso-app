# 📊 Lasso Path Explorer

A professional, interactive Streamlit application for understanding and visualizing Lasso Regression with real-time exploration, cross-validation optimization, and model comparison.

## 🚀 Features

- **Data Management**: Upload your own CSV files or use built-in sample datasets (California Housing, Diabetes)
- **Interactive Lasso Path**: Visualize how coefficients change as regularization strength (α) increases
- **Cross-Validation Optimizer**: Automatically find the optimal alpha using LassoCV
- **Real-time Predictions**: Adjust feature values and see predictions update instantly
- **Model Comparison**: Compare Lasso, Ridge, and OLS regression side-by-side
- **Educational Content**: Built-in explanations and insights to help understand Lasso regression

## 🛠️ Installation

### Local Development

1. Clone this repository:
```bash
git clone <your-repo-url>
cd "Lasso app"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## 📦 Requirements

- Python 3.8+
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.17.0
- scikit-learn >= 1.3.0

## 🌐 Deployment

This app is ready for deployment on Streamlit Cloud:

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set the main file path to `app.py`
6. Click "Deploy"

## 📝 Usage

1. **Select Data**: Choose a sample dataset or upload your own CSV file
2. **Select Variables**: Pick your target variable and features
3. **Explore Path**: Use the interactive slider to see how coefficients change with alpha
4. **Optimize**: Let cross-validation find the best alpha value
5. **Predict**: Adjust feature values and see real-time predictions
6. **Compare**: View how Lasso compares to Ridge and OLS models

## 🎯 Key Features Explained

- **Regularization Path**: Shows how Lasso shrinks coefficients toward zero as α increases
- **Feature Selection**: Features with coefficients that reach zero are eliminated
- **Cross-Validation**: Finds the optimal balance between bias and variance
- **Fast Mode**: Optimized algorithms for faster computation on Streamlit Cloud

## 📄 License

This project is open source and available for educational and portfolio purposes.

## 👤 Author

Built with ❤️ using Streamlit, scikit-learn, and Plotly

