import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# Page config
st.set_page_config(
    page_title="Crop Yield Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .stProgress .st-bo {
        background-color: #2E8B57;
    }
    .plot-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌾 Crop Yield Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["Data Overview", "Visualizations", "3D Analysis", "Model Training", "Predictions"]
)

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Load Dataset with error handling
@st.cache_data
def load_data():
    try:
        # Try multiple common paths
        possible_paths = [
            "crop_yield.csv",
            "data/crop_yield.csv",
            "C:/Users/oizys/Downloads/crop_yield.csv"
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Data loaded from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            # Create sample data if file not found
            st.sidebar.warning("📁 Using sample data - upload your crop_yield.csv file")
            df = create_sample_data()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane']
    seasons = ['Kharif', 'Rabi', 'Summer']
    states = ['Maharashtra', 'Punjab', 'Uttar Pradesh', 'Karnataka', 'Gujarat']
    
    data = {
        'State': np.random.choice(states, n_samples),
        'Crop': np.random.choice(crops, n_samples),
        'Crop_Year': np.random.randint(2010, 2024, n_samples),
        'Season': np.random.choice(seasons, n_samples),
        'Area': np.random.uniform(100, 10000, n_samples),
        'Annual_Rainfall': np.random.uniform(500, 2500, n_samples),
        'Fertilizer': np.random.uniform(50, 300, n_samples),
        'Pesticide': np.random.uniform(10, 100, n_samples),
        'Yield': np.random.uniform(1, 8, n_samples),
        'Production': None
    }
    
    df = pd.DataFrame(data)
    df['Production'] = df['Area'] * df['Yield']
    return df

# Load data
with st.spinner("Loading data..."):
    df = load_data()

# Data preprocessing
df = df.dropna()

if page == "Data Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Crops", df['Crop'].nunique() if 'Crop' in df.columns else 0)
    with col3:
        st.metric("States", df['State'].nunique() if 'State' in df.columns else 0)
    with col4:
        st.metric("Years", df['Crop_Year'].nunique() if 'Crop_Year' in df.columns else 0)
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Data Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Missing values check
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        st.subheader("Missing Values")
        st.bar_chart(missing_data[missing_data > 0])
    
    # Data distribution
    st.subheader("📈 Data Distribution")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_col = st.selectbox("Select column for distribution plot:", numeric_cols)
    
    if selected_col:
        fig_hist = px.histogram(df, x=selected_col, nbins=30, 
                               title=f'Distribution of {selected_col}',
                               template='plotly_white')
        st.plotly_chart(fig_hist, use_container_width=True)

elif page == "Visualizations":
    st.header("📈 Data Visualizations")
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3 = st.tabs(["Distribution Analysis", "Relationship Analysis", "Trend Analysis"])
    
    with tab1:
        st.subheader("🌾 Crop Yield Distribution")
        if 'Crop' in df.columns and 'Yield' in df.columns:
            fig1 = px.box(
                df, x='Crop', y='Yield', 
                color='Season' if 'Season' in df.columns else None,
                title='Crop-wise Yield Distribution by Season',
                template='plotly_white'
            )
            fig1.update_layout(xaxis_tickangle=45, height=500)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Save option
            if st.button("💾 Save Box Plot", key="save_box"):
                try:
                    fig1.write_image("outputs/fig1_box_yield_distribution.png")
                    st.success("✅ Box plot saved!")
                except:
                    st.warning("Could not save image. Install kaleido: pip install kaleido")
    
    with tab2:
        st.subheader("💧 Rainfall vs Yield Analysis")
        if all(col in df.columns for col in ['Annual_Rainfall', 'Yield']):
            fig2 = px.scatter(
                df, x='Annual_Rainfall', y='Yield', 
                color='Crop' if 'Crop' in df.columns else None,
                size='Area' if 'Area' in df.columns else None,
                hover_data=['State', 'Season'] if all(col in df.columns for col in ['State', 'Season']) else None,
                title='Rainfall vs Yield Relationship',
                template='plotly_white'
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Correlation analysis
            if st.checkbox("Show Correlation Matrix"):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                corr_matrix = df[numeric_cols].corr()
                
                fig_corr = px.imshow(corr_matrix, 
                                   text_auto=True, 
                                   aspect="auto",
                                   title="Correlation Matrix",
                                   template='plotly_white')
                st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Production Trends")
            if all(col in df.columns for col in ['Crop_Year', 'Production']):
                production_trend = df.groupby('Crop_Year')['Production'].sum().reset_index()
                fig3 = px.line(
                    production_trend, x='Crop_Year', y='Production', 
                    markers=True,
                    title='Total Agricultural Production Over Years',
                    template='plotly_white'
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.subheader("🌱 Crop Performance Analysis")
            if 'Crop' in df.columns and 'Yield' in df.columns:
                crop_avg = df.groupby('Crop')[['Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']].mean().reset_index()
                fig4 = px.bar(
                    crop_avg.sort_values('Yield', ascending=False), 
                    x='Crop', y='Yield', 
                    color='Annual_Rainfall' if 'Annual_Rainfall' in crop_avg.columns else None,
                    title='Crop-wise Average Yield & Rainfall',
                    labels={'Yield': 'Avg Yield (tons/hectare)', 'Annual_Rainfall': 'Rainfall (mm)'},
                    template='plotly_white'
                )
                fig4.update_layout(xaxis_tickangle=45, height=400)
                st.plotly_chart(fig4, use_container_width=True)

elif page == "3D Analysis":
    st.header("🧪 3D Scatter Plot Analysis")
    st.info("Interactive 3D visualization of Fertilizer, Pesticide, and Yield relationships")
    
    if all(col in df.columns for col in ['Fertilizer', 'Pesticide', 'Yield']):
        # Controls for 3D plot
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_size = st.slider("Sample Size (for performance)", 
                                   min_value=100, 
                                   max_value=min(2000, len(df)), 
                                   value=min(1000, len(df)))
        
        with col2:
            color_by = st.selectbox("Color by:", 
                                   ['Crop', 'Season', 'State'] if all(col in df.columns for col in ['Crop', 'Season', 'State']) else ['Crop'])
        
        with col3:
            symbol_by = st.selectbox("Symbol by:", 
                                    ['Season', 'Crop', 'None'] if 'Season' in df.columns else ['None'])
        
        # Sample data for performance
        plot_df = df.sample(sample_size) if len(df) > sample_size else df
        
        # Create 3D scatter plot
        fig5 = px.scatter_3d(
            plot_df, 
            x='Fertilizer', 
            y='Pesticide', 
            z='Yield',
            color=color_by if color_by in df.columns else None,
            symbol=symbol_by if symbol_by != 'None' and symbol_by in df.columns else None,
            title='🧪 3D View: Fertilizer, Pesticide vs Yield',
            labels={
                'Fertilizer': 'Fertilizer (kg/ha)', 
                'Pesticide': 'Pesticide (kg/ha)', 
                'Yield': 'Yield (tons/ha)'
            },
            template='plotly_white',
            height=600
        )
        
        # Customize 3D scene
        fig5.update_layout(
            scene=dict(
                xaxis_title='Fertilizer (kg/ha)',
                yaxis_title='Pesticide (kg/ha)',
                zaxis_title='Yield (tons/ha)',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        st.plotly_chart(fig5, use_container_width=True)
        
        # Save image option
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save 3D Plot", key="save_3d"):
                try:
                    fig5.write_image("outputs/fig5_3d_fertilizer_pesticide_yield.png")
                    st.success("✅ 3D plot saved to outputs/fig5_3d_fertilizer_pesticide_yield.png")
                except Exception as e:
                    st.warning(f"Could not save image: {str(e)}")
                    st.info("Install kaleido for image export: pip install kaleido")
        
        with col2:
            if st.button("🔄 Refresh Plot", key="refresh_3d"):
                st.rerun()
        
        # Statistical summary
        st.subheader("📊 3D Plot Statistics")
        stats_cols = ['Fertilizer', 'Pesticide', 'Yield']
        stats_df = plot_df[stats_cols].describe()
        st.dataframe(stats_df, use_container_width=True)
        
    else:
        missing_cols = [col for col in ['Fertilizer', 'Pesticide', 'Yield'] if col not in df.columns]
        st.error(f"Missing columns for 3D plot: {missing_cols}")
        st.info("Please ensure your dataset contains 'Fertilizer', 'Pesticide', and 'Yield' columns.")

elif page == "Model Training":
    st.header("🤖 Machine Learning Models")
    
    # Feature selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("⚠️ Not enough numeric columns for modeling")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            feature_cols = st.multiselect(
                "Select features for modeling:",
                numeric_cols,
                default=[col for col in numeric_cols if col != 'Yield'][:5]  # Limit default selection
            )
        
        with col2:
            target_col = st.selectbox(
                "Select target variable:",
                numeric_cols,
                index=numeric_cols.index('Yield') if 'Yield' in numeric_cols else 0
            )
        
        if len(feature_cols) > 0:
            X = df[feature_cols]
            y = df[target_col]
            
            # Model parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                test_size = st.slider("Test size ratio", 0.1, 0.4, 0.2)
            with col2:
                random_state = st.number_input("Random state", value=42)
            with col3:
                n_estimators = st.slider("Random Forest trees", 50, 200, 100)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state)
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Model training
            if st.button("🚀 Train Models", type="primary"):
                models = {}
                results = {}
                
                with st.spinner("Training models..."):
                    # Random Forest
                    try:
                        rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=int(random_state))
                        rf_model.fit(X_train_scaled, y_train)
                        y_pred_rf = rf_model.predict(X_test_scaled)
                        
                        models['Random Forest'] = rf_model
                        results['Random Forest'] = {
                            'MAE': mean_absolute_error(y_test, y_pred_rf),
                            'MSE': mean_squared_error(y_test, y_pred_rf),
                            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                            'R²': r2_score(y_test, y_pred_rf)
                        }
                        
                        # Store predictions for plotting
                        rf_predictions = y_pred_rf
                        
                    except Exception as e:
                        st.error(f"Random Forest error: {str(e)}")
                
                # Display results
                if results:
                    st.success("✅ Models trained successfully!")
                    
                    st.subheader("📊 Model Performance")
                    results_df = pd.DataFrame(results).T
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Performance comparison
                        fig_perf = px.bar(
                            results_df.reset_index(), 
                            x='index', 
                            y='R²',
                            title='Model R² Score Comparison',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
                    
                    with col2:
                        # Actual vs Predicted
                        if 'rf_predictions' in locals():
                            pred_df = pd.DataFrame({
                                'Actual': y_test,
                                'Predicted': rf_predictions
                            })
                            
                            fig_pred = px.scatter(
                                pred_df, 
                                x='Actual', 
                                y='Predicted',
                                title='Actual vs Predicted Values',
                                template='plotly_white'
                            )
                            # Add perfect prediction line
                            min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
                            max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
                            fig_pred.add_trace(
                                go.Scatter(
                                    x=[min_val, max_val], 
                                    y=[min_val, max_val],
                                    mode='lines',
                                    name='Perfect Prediction',
                                    line=dict(dash='dash', color='red')
                                )
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Feature importance
                    if 'Random Forest' in models:
                        st.subheader("🎯 Feature Importance")
                        importance_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': models['Random Forest'].feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig_imp = px.bar(
                            importance_df, 
                            x='Importance', 
                            y='Feature',
                            orientation='h',
                            title='Feature Importance (Random Forest)',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

elif page == "Predictions":
    st.header("🔮 Make Predictions")
    
    st.info("Enter feature values to predict crop yield using a simple model.")
    
    # Feature input form
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != 'Yield']
    
    if len(feature_cols) > 0:
        st.subheader("Input Features")
        
        input_data = {}
        
        # Create input fields dynamically
        cols = st.columns(min(3, len(feature_cols)))
        
        for i, col in enumerate(feature_cols):
            with cols[i % len(cols)]:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                
                input_data[col] = st.number_input(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100,
                    help=f"Range: {min_val:.2f} - {max_val:.2f}"
                )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Make Prediction", type="primary"):
                # Simple prediction using Random Forest
                try:
                    # Prepare data
                    X = df[feature_cols]
                    y = df['Yield'] if 'Yield' in df.columns else df[numeric_cols[-1]]
                    
                    # Train a quick model
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_scaled, y)
                    
                    # Make prediction
                    input_array = np.array([list(input_data.values())])
                    input_scaled = scaler.transform(input_array)
                    prediction = model.predict(input_scaled)[0]
                    
                    st.success(f"🎯 Predicted Yield: **{prediction:.2f} tons/hectare**")
                    
                    # Show confidence interval (approximate)
                    predictions_all = model.predict(X_scaled)
                    std_error = np.std(y - predictions_all)
                    
                    st.info(f"📊 Confidence Range: {prediction - std_error:.2f} - {prediction + std_error:.2f} tons/hectare")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        
        with col2:
            if st.button("🔄 Reset to Averages"):
                st.rerun()
        
        # Show input summary
        st.subheader("📝 Input Summary")
        input_df = pd.DataFrame([input_data])
        st.dataframe(input_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🌾 Crop Yield Prediction Dashboard | Built with Streamlit & Plotly</p>
        <p>💡 Features: Interactive 3D Plots, ML Models, Real-time Predictions</p>
    </div>
    """,
    unsafe_allow_html=True
)
