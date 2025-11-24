"""
Lasso Path Explorer - Interactive Lasso Regression Visualization Tool

A professional Streamlit application for understanding and visualizing
Lasso Regression with interactive features, cross-validation, and model comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, LinearRegression, lasso_path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Lasso Path Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading functions
@st.cache_data
def load_sample_dataset(dataset_name):
    """Load sample datasets with caching."""
    if dataset_name == "California Housing":
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        target = 'MedHouseVal'
    elif dataset_name == "Diabetes":
        data = load_diabetes(as_frame=True)
        df = data.frame
        target = 'target'
    else:
        return None, None
    return df, target

@st.cache_data
def compute_lasso_path(X, y, alphas, standardize=True):
    """Compute Lasso regularization path using optimized sklearn function."""
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
        scaler = None
    
    # Use sklearn's optimized lasso_path which is much faster
    # It computes the full path efficiently using coordinate descent
    alphas_path, coefs_path, _ = lasso_path(
        X_scaled, y, 
        alphas=alphas,
        max_iter=2000,
        return_n_iter=True
    )
    
    # Transpose to get shape (n_alphas, n_features)
    coefs_path = coefs_path.T
    
    return coefs_path, scaler

@st.cache_data
def compute_cv_results(X_train_scaled, y_train, alphas, cv_folds):
    """Compute cross-validation results with caching."""
    lasso_cv = LassoCV(
        alphas=alphas,
        cv=cv_folds,
        max_iter=2000,
        random_state=42,
        n_jobs=-1
    )
    lasso_cv.fit(X_train_scaled, y_train)
    return lasso_cv

@st.cache_data
def compute_comparison_models(X_train_scaled, y_train, alphas, cv_folds, optimal_alpha):
    """Compute Ridge and OLS models for comparison with caching."""
    # Ridge with optimal alpha from RidgeCV
    ridge_cv = RidgeCV(alphas=alphas, cv=cv_folds)
    ridge_cv.fit(X_train_scaled, y_train)
    optimal_ridge = Ridge(alpha=ridge_cv.alpha_)
    optimal_ridge.fit(X_train_scaled, y_train)
    
    # OLS
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    
    return optimal_ridge, ridge_cv.alpha_, ols

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Lasso Path Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive visualization and analysis of Lasso Regression</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset selection
        st.subheader("📁 Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Sample Dataset", "Upload CSV"],
            help="Select a built-in sample dataset or upload your own CSV file"
        )
        
        dataset_name = None
        uploaded_file = None
        
        if data_source == "Sample Dataset":
            dataset_name = st.selectbox(
                "Select sample dataset:",
                ["California Housing", "Diabetes"],
                help="California Housing: Predict median house values\nDiabetes: Predict disease progression"
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="Upload your own dataset in CSV format"
            )
        
        st.divider()
        
        # Alpha range configuration
        st.subheader("🔧 Model Parameters")
        alpha_min = st.slider(
            "Min Alpha (λ)",
            0.001, 10.0, 0.01,
            step=0.01,
            help="Minimum regularization strength"
        )
        alpha_max = st.slider(
            "Max Alpha (λ)",
            0.1, 100.0, 10.0,
            step=0.1,
            help="Maximum regularization strength"
        )
        n_alphas = st.slider(
            "Number of Alpha Values",
            30, 150, 50,
            help="More values = smoother path but slower computation. Default: 50 for faster performance."
        )
        
        # Performance mode toggle
        use_fast_mode = st.checkbox(
            "⚡ Fast Mode (Recommended)",
            value=True,
            help="Use optimized algorithms for faster computation. Uncheck for more precise results."
        )
        
        st.divider()
        
        # About section
        with st.expander("ℹ️ About Lasso Regression"):
            st.markdown("""
            **Lasso (Least Absolute Shrinkage and Selection Operator)** is a regularization 
            technique that:
            
            - Adds L1 penalty to the loss function
            - Shrinks coefficients toward zero
            - Can set coefficients to exactly zero (feature selection)
            - Helps prevent overfitting and improves interpretability
            
            The regularization path shows how coefficients change as α (lambda) increases.
            """)
    
    # Main content area
    if data_source == "Upload CSV" and uploaded_file is None:
        st.info("👆 Please upload a CSV file in the sidebar to get started.")
        st.markdown("""
        ### 📖 Getting Started
        
        1. **Upload Data**: Use the sidebar to upload your CSV file
        2. **Select Variables**: Choose your target variable and features
        3. **Explore Path**: Visualize how coefficients change with regularization
        4. **Optimize**: Find the best alpha using cross-validation
        5. **Predict**: Make predictions with the optimized model
        6. **Compare**: See how Lasso compares to Ridge and OLS
        
        ### 🎯 Key Features
        
        - **Interactive Lasso Path**: Watch coefficients shrink as regularization increases
        - **Cross-Validation**: Automatically find optimal alpha
        - **Real-time Predictions**: Adjust features and see predictions update
        - **Model Comparison**: Compare Lasso, Ridge, and OLS side-by-side
        """)
        return
    
    # Load data
    try:
        if data_source == "Sample Dataset":
            df, default_target = load_sample_dataset(dataset_name)
            if df is None:
                st.error("Failed to load sample dataset.")
                return
        else:
            df = pd.read_csv(uploaded_file)
            default_target = None
        
        # Data preview section
        st.header("📊 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        with st.expander("📋 View Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        with st.expander("📈 View Data Statistics", expanded=False):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Variable selection
        st.header("🎯 Variable Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            target_var = st.selectbox(
                "Select Target Variable:",
                df.columns.tolist(),
                index=df.columns.get_loc(default_target) if default_target and default_target in df.columns else 0,
                help="The variable you want to predict"
            )
        
        with col2:
            feature_vars = st.multiselect(
                "Select Features:",
                [col for col in df.columns if col != target_var],
                default=[col for col in df.columns if col != target_var][:min(10, len(df.columns)-1)],
                help="Select one or more features to use for prediction"
            )
        
        if len(feature_vars) == 0:
            st.warning("⚠️ Please select at least one feature variable.")
            return
        
        # Performance warning for too many features
        if len(feature_vars) > 20 and not use_fast_mode:
            st.warning(f"⚠️ You have selected {len(feature_vars)} features. Consider enabling Fast Mode or reducing the number of features for better performance.")
        
        # Prepare data
        X = df[feature_vars].copy()
        y = df[target_var].copy()
        
        # Handle missing values
        if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
            st.warning("⚠️ Missing values detected. Rows with missing values will be removed.")
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
        
        # Train-test split
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.success(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
        
        # Lasso Path Visualization
        st.header("📉 Lasso Regularization Path")
        
        st.markdown("""
        <div class="info-box">
        <strong>What is the Regularization Path?</strong><br>
        This plot shows how each feature's coefficient changes as the regularization strength (α) increases. 
        As α increases, coefficients shrink toward zero. Features that reach zero are effectively eliminated 
        from the model, demonstrating Lasso's feature selection capability.
        </div>
        """, unsafe_allow_html=True)
        
        # Generate alpha values (reduce for fast mode)
        effective_n_alphas = n_alphas if not use_fast_mode else min(n_alphas, 50)
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), effective_n_alphas)
        
        if use_fast_mode and n_alphas > 50:
            st.info(f"⚡ Fast Mode: Using {effective_n_alphas} alpha values for faster computation.")
        
        # Compute path (cached and optimized)
        with st.spinner("Computing Lasso path (this may take a moment on first run)..."):
            coefs_path, path_scaler = compute_lasso_path(X_train, y_train, alphas, standardize=True)
        
        # Create path plot
        fig_path = go.Figure()
        
        colors = px.colors.qualitative.Set3
        for i, feature in enumerate(feature_vars):
            fig_path.add_trace(go.Scatter(
                x=np.log10(alphas),
                y=coefs_path[:, i],
                mode='lines',
                name=feature,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{feature}</b><br>' +
                            'Log(α): %{x:.3f}<br>' +
                            'Coefficient: %{y:.4f}<extra></extra>'
            ))
        
        fig_path.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig_path.update_layout(
            title="Lasso Regularization Path",
            xaxis_title="Log₁₀(α)",
            yaxis_title="Coefficient Value",
            hovermode='closest',
            height=500,
            template="plotly_white",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        st.plotly_chart(fig_path, use_container_width=True)
        
        # Interactive alpha slider
        st.subheader("🎚️ Interactive Alpha Control")
        
        current_alpha_idx = st.slider(
            "Select Alpha (λ):",
            0, len(alphas)-1, len(alphas)//2,
            format="Alpha = %.4f",
            help="Adjust this slider to see how coefficients change at different alpha values"
        )
        
        current_alpha = alphas[current_alpha_idx]
        current_coefs = coefs_path[current_alpha_idx]
        
        # Show current coefficients
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Coefficient bar chart
            fig_coefs = go.Figure()
            colors_bar = ['#1f77b4' if c != 0 else '#d62728' for c in current_coefs]
            fig_coefs.add_trace(go.Bar(
                x=feature_vars,
                y=current_coefs,
                marker_color=colors_bar,
                text=[f'{c:.4f}' for c in current_coefs],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Coefficient: %{y:.4f}<extra></extra>'
            ))
            fig_coefs.update_layout(
                title=f"Coefficients at α = {current_alpha:.4f}",
                xaxis_title="Feature",
                yaxis_title="Coefficient Value",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig_coefs, use_container_width=True)
        
        with col2:
            st.metric("Alpha (λ)", f"{current_alpha:.4f}")
            n_nonzero = np.sum(current_coefs != 0)
            st.metric("Non-zero Coefficients", f"{n_nonzero}/{len(feature_vars)}")
            
            # List of eliminated features
            eliminated = [feature_vars[i] for i in range(len(feature_vars)) if current_coefs[i] == 0]
            if eliminated:
                st.write("**Eliminated Features:**")
                for feat in eliminated:
                    st.write(f"- {feat}")
            else:
                st.info("No features eliminated at this alpha.")
        
        # Cross-Validation Section
        st.header("🔍 Cross-Validation Optimization")
        
        st.markdown("""
        <div class="info-box">
        <strong>How to Read the CV Curve:</strong><br>
        Cross-validation helps find the optimal alpha that balances bias and variance. 
        The optimal alpha minimizes the cross-validation error. Points to the left (lower alpha) 
        may overfit, while points to the right (higher alpha) may underfit.
        </div>
        """, unsafe_allow_html=True)
        
        cv_folds = st.slider("CV Folds:", 3, 10, 5, help="Number of cross-validation folds")
        
        # Use fewer alphas for CV in fast mode to speed up computation
        cv_alphas = alphas if not use_fast_mode else np.logspace(
            np.log10(alpha_min), np.log10(alpha_max), min(30, len(alphas))
        )
        
        with st.spinner("Running cross-validation (cached after first run)..."):
            # Use cached CV computation
            lasso_cv = compute_cv_results(X_train_scaled, y_train, cv_alphas, cv_folds)
            optimal_alpha = lasso_cv.alpha_
            
            # Get CV scores (need to match the alphas used)
            cv_scores = -lasso_cv.mse_path_.mean(axis=1)
            cv_stds = lasso_cv.mse_path_.std(axis=1)
            cv_alphas_used = cv_alphas  # Use the alphas we passed to CV
        
        # Plot CV curve
        fig_cv = go.Figure()
        
        fig_cv.add_trace(go.Scatter(
            x=np.log10(cv_alphas_used),
            y=cv_scores,
            mode='lines',
            name='CV Score',
            line=dict(color='#1f77b4', width=2),
            error_y=dict(
                type='data',
                array=cv_stds,
                visible=True,
                color='rgba(31, 119, 180, 0.3)'
            ),
            hovertemplate='Log(α): %{x:.3f}<br>CV Score: %{y:.4f}<extra></extra>'
        ))
        
        # Mark optimal alpha
        optimal_idx = np.argmin(lasso_cv.mse_path_.mean(axis=1))
        fig_cv.add_trace(go.Scatter(
            x=[np.log10(optimal_alpha)],
            y=[cv_scores[optimal_idx]],
            mode='markers',
            name='Optimal α',
            marker=dict(
                symbol='star',
                size=15,
                color='red',
                line=dict(width=2, color='darkred')
            ),
            hovertemplate=f'Optimal α: {optimal_alpha:.4f}<br>CV Score: {cv_scores[optimal_idx]:.4f}<extra></extra>'
        ))
        
        fig_cv.update_layout(
            title="Cross-Validation Score vs Alpha",
            xaxis_title="Log₁₀(α)",
            yaxis_title="CV Score (MSE)",
            hovermode='closest',
            height=500,
            template="plotly_white",
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig_cv, use_container_width=True)
        
        # Train optimal model
        optimal_lasso = Lasso(alpha=optimal_alpha, max_iter=2000)
        optimal_lasso.fit(X_train_scaled, y_train)
        
        y_train_pred = optimal_lasso.predict(X_train_scaled)
        y_test_pred = optimal_lasso.predict(X_test_scaled)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        with col1:
            st.metric("Optimal α", f"{optimal_alpha:.4f}")
        with col2:
            st.metric("Train R²", f"{train_r2:.4f}")
        with col3:
            st.metric("Test R²", f"{test_r2:.4f}")
        with col4:
            st.metric("Test RMSE", f"{test_rmse:.4f}")
        
        # Prediction Interface
        st.header("🔮 Prediction Interface")
        
        st.markdown("""
        <div class="info-box">
        Adjust the feature values below to see real-time predictions from the optimized Lasso model.
        The model uses the optimal alpha found through cross-validation.
        </div>
        """, unsafe_allow_html=True)
        
        # Create input sliders for each feature
        input_values = {}
        n_cols = 3
        cols = st.columns(n_cols)
        
        for i, feature in enumerate(feature_vars):
            col_idx = i % n_cols
            with cols[col_idx]:
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                mean_val = float(X[feature].mean())
                
                input_values[feature] = st.number_input(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100,
                    key=f"input_{feature}"
                )
        
        # Make prediction
        input_array = np.array([[input_values[feat] for feat in feature_vars]])
        input_scaled = scaler.transform(input_array)
        prediction = optimal_lasso.predict(input_scaled)[0]
        
        # Display prediction
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Prediction Result")
            st.metric(
                f"Predicted {target_var}",
                f"{prediction:.4f}",
                help="Prediction based on current feature values"
            )
        
        with col2:
            # Show feature importance (absolute coefficients)
            coef_importance = pd.DataFrame({
                'Feature': feature_vars,
                'Coefficient': optimal_lasso.coef_,
                'Abs Coefficient': np.abs(optimal_lasso.coef_)
            }).sort_values('Abs Coefficient', ascending=False)
            
            fig_importance = go.Figure()
            fig_importance.add_trace(go.Bar(
                x=coef_importance['Feature'],
                y=coef_importance['Coefficient'],
                marker_color=coef_importance['Coefficient'].apply(
                    lambda x: '#1f77b4' if x > 0 else '#d62728'
                ),
                hovertemplate='<b>%{x}</b><br>Coefficient: %{y:.4f}<extra></extra>'
            ))
            fig_importance.update_layout(
                title="Feature Importance (Optimal Model)",
                xaxis_title="Feature",
                yaxis_title="Coefficient",
                height=300,
                template="plotly_white"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Model Comparison
        st.header("⚖️ Model Comparison")
        
        st.markdown("""
        <div class="info-box">
        Compare Lasso, Ridge, and Ordinary Least Squares (OLS) regression. 
        Notice how Lasso typically uses fewer features (sparsity) while maintaining 
        competitive performance.
        </div>
        """, unsafe_allow_html=True)
        
        # Train all models (cached)
        with st.spinner("Training comparison models (cached after first run)..."):
            optimal_ridge, optimal_ridge_alpha, ols = compute_comparison_models(
                X_train_scaled, y_train, cv_alphas, cv_folds, optimal_alpha
            )
        
        # Evaluate all models
        models = {
            'Lasso': optimal_lasso,
            'Ridge': optimal_ridge,
            'OLS': ols
        }
        
        comparison_data = []
        for name, model in models.items():
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            n_nonzero = np.sum(model.coef_ != 0) if hasattr(model, 'coef_') else len(model.coef_)
            
            comparison_data.append({
                'Model': name,
                'Train R²': r2_score(y_train, y_train_pred),
                'Test R²': r2_score(y_test, y_test_pred),
                'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'Test MAE': mean_absolute_error(y_test, y_test_pred),
                'Non-zero Coefs': n_nonzero,
                'Optimal α': optimal_alpha if name == 'Lasso' else (optimal_ridge_alpha if name == 'Ridge' else 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        format_dict = {
            'Train R²': '{:.4f}',
            'Test R²': '{:.4f}',
            'Train RMSE': '{:.4f}',
            'Test RMSE': '{:.4f}',
            'Test MAE': '{:.4f}',
        }
        # Format Optimal α column conditionally
        if comparison_df['Optimal α'].dtype != 'object':
            format_dict['Optimal α'] = '{:.4f}'
        else:
            format_dict['Optimal α'] = '{}'
        
        st.dataframe(
            comparison_df.style.format(format_dict)
            .highlight_max(subset=['Test R²'], color='lightgreen')
            .highlight_min(subset=['Test RMSE', 'Test MAE'], color='lightcoral')
            .highlight_min(subset=['Non-zero Coefs'], color='lightblue'),
            use_container_width=True
        )
        
        # Key Insights
        st.header("💡 Key Insights")
        
        insights = []
        
        # Sparsity insight
        n_eliminated = len(feature_vars) - np.sum(optimal_lasso.coef_ != 0)
        if n_eliminated > 0:
            eliminated_features = [feature_vars[i] for i in range(len(feature_vars)) 
                                 if optimal_lasso.coef_[i] == 0]
            insights.append(
                f"**Feature Selection**: Lasso eliminated {n_eliminated} feature(s): {', '.join(eliminated_features)}. "
                f"This reduces model complexity while maintaining performance."
            )
        else:
            insights.append(
                "**Feature Selection**: All features are retained in the optimal Lasso model. "
                "Consider increasing the alpha range to see feature elimination."
            )
        
        # Performance insight
        if test_r2 > 0.7:
            insights.append(
                f"**Strong Performance**: The model achieves a test R² of {test_r2:.3f}, "
                f"indicating good predictive power."
            )
        elif test_r2 > 0.5:
            insights.append(
                f"**Moderate Performance**: The model achieves a test R² of {test_r2:.3f}. "
                f"Consider feature engineering or trying different models."
            )
        else:
            insights.append(
                f"**Low Performance**: The model achieves a test R² of {test_r2:.3f}. "
                f"Consider feature engineering, checking for data quality issues, or trying non-linear models."
            )
        
        # Comparison insight
        lasso_r2 = comparison_df[comparison_df['Model'] == 'Lasso']['Test R²'].values[0]
        ols_r2 = comparison_df[comparison_df['Model'] == 'OLS']['Test R²'].values[0]
        lasso_coefs = comparison_df[comparison_df['Model'] == 'Lasso']['Non-zero Coefs'].values[0]
        ols_coefs = comparison_df[comparison_df['Model'] == 'OLS']['Non-zero Coefs'].values[0]
        
        if lasso_coefs < ols_coefs:
            if ols_r2 != 0:
                r2_retention = (lasso_r2 / ols_r2) * 100
                insights.append(
                    f"**Efficiency**: Lasso uses {lasso_coefs} features vs OLS's {ols_coefs}, "
                    f"achieving {r2_retention:.1f}% of OLS performance (R²: {lasso_r2:.3f} vs {ols_r2:.3f}). "
                    f"This demonstrates Lasso's ability to create simpler, more interpretable models."
                )
            else:
                insights.append(
                    f"**Efficiency**: Lasso uses {lasso_coefs} features vs OLS's {ols_coefs}, "
                    f"demonstrating feature selection capability while maintaining competitive performance."
                )
        
        for insight in insights:
            st.info(insight)
        
        # Footer
        st.divider()
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Lasso Path Explorer</strong> - Built with Streamlit, scikit-learn, and Plotly</p>
        <p>Explore the power of Lasso regularization for feature selection and model interpretability</p>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()

