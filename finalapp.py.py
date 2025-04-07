import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# from xgboost import XGBRegressor # Type hint only
# from sklearn.preprocessing import StandardScaler # Type hint only
import joblib
import os

# --- Configuration ---
ARTIFACTS_DIR = 'model_artifacts'
FORECAST_YEARS = 5 # Forecast 5 years into the future
# Determine the start year dynamically or set statically
try:
    # A simple way is to assume the data used for saving artifacts is the same
    # Make sure the path is correct for your system
    _temp_data = pd.read_excel(r"C:\Users\ASUS\Downloads\V5_Capstone_Final_Dataset.xlsx")
    LATEST_HISTORICAL_YEAR = _temp_data['REF_DATE'].max()
    START_FORECAST_YEAR = LATEST_HISTORICAL_YEAR + 1
    del _temp_data # Clean up temp data
except Exception as e:
    LATEST_HISTORICAL_YEAR = 2022 # Fallback if loading fails
    START_FORECAST_YEAR = 2023
    st.sidebar.warning(f"Could not dynamically determine latest year (Error: {e}). Defaulting to 2022/2023.")


# --- Load Artifacts ---
@st.cache_resource # Cache loaded objects for efficiency
def load_artifacts():
    """Loads the saved model, scaler, and features."""
    try:
        model_d = joblib.load(os.path.join(ARTIFACTS_DIR, 'diabetes_model.joblib'))
        model_hbp = joblib.load(os.path.join(ARTIFACTS_DIR, 'hbp_model.joblib'))
        scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))
        selected_features = joblib.load(os.path.join(ARTIFACTS_DIR, 'selected_features.pkl'))
        # List of food categories for sliders (must be in selected_features)
        food_cols = [
            'Sugar and confectionery', 'Eggs', 'Bakery products', 'Butter',
            'Dairy products', 'Cheese', 'Fresh vegetables',
            'Preserved fruit and fruit preparations', 'Fish',
            'Non-alcoholic beverages', 'Preserved vegetables and vegetable preparations'
        ]
        # Verify food_cols are actually in selected_features
        food_cols = [col for col in food_cols if col in selected_features]
        if not food_cols:
             st.error("Error: No adjustable food columns found in 'selected_features.pkl'.")
             st.stop()

        return model_d, model_hbp, scaler, selected_features, food_cols
    except FileNotFoundError:
        st.error(f"Error loading artifacts from '{ARTIFACTS_DIR}'. Make sure the directory exists and contains "
                 f"'diabetes_model.joblib', 'hbp_model.joblib', 'scaler.joblib', and 'selected_features.pkl'. "
                 f"Run the 'save_artifacts.py' script first.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading artifacts: {e}")
        st.stop()

# --- Load Data ---
@st.cache_data # Cache loaded data
def load_data(filepath):
    """Loads and preprocesses the historical data."""
    try:
        # Make sure the path is correct for your system
        data = pd.read_excel(filepath)
        # --- Apply the *exact* same preprocessing as in save_artifacts.py ---
        data['Year'] = data['REF_DATE']
        # Replace 0 population with NaN BEFORE calculating per capita rates
        data['Actual Population'] = data['Actual Population'].replace(0, np.nan)
        data.dropna(subset=['Actual Population'], inplace=True) # Essential for per capita calculation

        # Calculate per capita rates
        data['Diabetes_per_capita'] = (data['Diabetes'] / data['Actual Population']) * 1000
        data['HBP_per_capita'] = (data['High Blood Pressure'] / data['Actual Population']) * 1000

        # Handle potential inf/-inf if Diabetes/HBP non-zero but pop was NaN before drop
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows if essential columns or target variables are NaN AFTER calculations
        data.dropna(subset=['GEO', 'Year', 'Actual Population', 'Diabetes_per_capita', 'HBP_per_capita'], inplace=True)

        data.sort_values(by=['GEO', 'Year'], inplace=True)

        # Convert year to int just in case
        data['Year'] = data['Year'].astype(int)
        return data
    except FileNotFoundError:
        st.error(f"Data file not found at {filepath}")
        st.stop()
    except KeyError as e:
        st.error(f"Missing expected column in data file: {e}. Please check '{os.path.basename(filepath)}'.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading or preprocessing data: {e}")
        st.stop()

# --- Forecasting Logic ---
def generate_future_features(historical_data, geo, start_year, num_years, features_to_forecast):
    """Generates extrapolated future feature values for a specific GEO."""
    # Filter data for the specific GEO and ensure it's sorted by year
    geo_data = historical_data[historical_data['GEO'] == geo].sort_values('Year')

    if geo_data.empty:
        # st.warning(f"No historical data found for {geo}. Cannot generate forecast.") # Warning is shown later if needed
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=features_to_forecast + ['Year'])

    # Find the actual latest year with data for this GEO
    last_known_year = geo_data['Year'].max()
    # Select the very last row for that year in case of duplicates
    # Check if last_known_year exists before trying to access it
    if last_known_year not in geo_data['Year'].values:
         return pd.DataFrame(columns=features_to_forecast + ['Year']) # Should not happen if geo_data not empty, but safe check
         
    last_known_values = geo_data[geo_data['Year'] == last_known_year].iloc[-1]

    future_df_list = []

    # --- Extrapolation Method (Simple Linear Trend per feature) ---
    growth_rates = {}
    for feature in features_to_forecast:
        if feature != 'Year':
            # Group by year and calculate mean for the feature (handles duplicate years)
            feature_series = geo_data.groupby('Year')[feature].mean()
            feature_series.dropna(inplace=True) # Crucial: Use only non-NaN values for trend calculation

            if len(feature_series) > 1:
                first_year = feature_series.index.min()
                last_year = feature_series.index.max()
                first_val = feature_series.loc[first_year]
                last_val = feature_series.loc[last_year]

                # Avoid division by zero if only one unique year had non-NaN data
                time_delta = last_year - first_year
                if time_delta > 0:
                     growth_rates[feature] = (last_val - first_val) / time_delta
                else:
                     growth_rates[feature] = 0 # No growth if only one year of data
            else:
                growth_rates[feature] = 0 # Assume no growth if <= 1 year of data points
    # --- End Extrapolation Method ---

    for i in range(num_years):
        current_future_year = start_year + i
        future_row = {'Year': current_future_year}

        # Calculate years forward from the last *actual* data point for this GEO
        years_forward = current_future_year - last_known_year

        for feature in features_to_forecast:
            if feature != 'Year':
                # Start with the last known value for this feature
                base_value = last_known_values.get(feature, np.nan) # Use NaN as default if missing

                # Handle potential NaN in last_known_values by falling back to the mean of the feature for that GEO
                if pd.isna(base_value):
                     geo_feature_mean = geo_data[feature].mean()
                     # Use the mean if it's valid, otherwise default to 0
                     base_value = 0 if pd.isna(geo_feature_mean) else geo_feature_mean

                # Get the calculated growth rate for the feature (defaults to 0 if not calculated)
                growth = growth_rates.get(feature, 0)

                # Calculate the future value: last known + (growth * time difference)
                future_row[feature] = base_value + growth * years_forward

                # Ensure non-negative values for specific features like population
                if feature == 'Actual Population' and future_row[feature] < 0:
                    future_row[feature] = 0 # Or a more reasonable floor like 1

        future_df_list.append(future_row)

    # Ensure the final DataFrame has all expected columns, even if generation failed partially
    final_df = pd.DataFrame(future_df_list)
    for col in features_to_forecast + ['Year']:
        if col not in final_df.columns:
            final_df[col] = np.nan # Add missing columns with NaN

    return final_df


# --- Feature Importance ---
# This function correctly retrieves static importance from the model
@st.cache_data # Cache the result as it doesn't change per model
def get_feature_importance(_model, feature_names):
    """Gets feature importance from the TRAINED model."""
    # The '_model' parameter avoids conflict with global model variables if any existed
    try:
        importance = _model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        return feature_importance.sort_values(by='Importance', ascending=False)
    except AttributeError:
        st.warning("Could not retrieve feature importance from the model.")
        return pd.DataFrame({'Feature': feature_names, 'Importance': [np.nan]*len(feature_names)})


# --- Main App Logic ---
st.set_page_config(layout="wide") # Use wider layout
st.title("Canada's Economic Shifts & Public Health: Diabetes and HBP Forecasting")
st.markdown("Explore potential future trends in Diabetes and High Blood Pressure prevalence based on projected food CPI and population changes.")

# Load artifacts and data
model_d, model_hbp, scaler, selected_features, food_cols = load_artifacts()
# Make sure the path is correct for your system
data_filepath = r"C:\Users\ASUS\Downloads\V5_Capstone_Final_Dataset.xlsx"
data = load_data(data_filepath)


# --- Sidebar Filters ---
st.sidebar.header("📍 Select Location")
# Ensure unique geos are sorted for consistent display
available_geos = sorted(data['GEO'].unique())
if not available_geos:
    st.error("No geographical locations found in the data after preprocessing.")
    st.stop()
# Use session state to potentially remember last selection if needed, otherwise default is fine
if 'geo_filter' not in st.session_state:
    st.session_state.geo_filter = available_geos[0] # Default to first option

geo_filter = st.sidebar.selectbox("Province/Territory:", available_geos, index=available_geos.index(st.session_state.geo_filter))
st.session_state.geo_filter = geo_filter # Update session state if changed

st.sidebar.header("📈 Adjust Future Food CPI (%)")
st.sidebar.caption("Modify the projected annual change for food categories (-10% to +10% relative to baseline trend).")
adjustments = {}
# Use unique keys to ensure sliders reset correctly when GEO changes
for food in food_cols:
    adjustments[food] = st.sidebar.slider(f"{food}", -10, 10, 0, key=f"slider_{food}_{geo_filter}") # Default 0%

# --- Generate Baseline Forecast ---
# Use st.spinner for user feedback during calculation
with st.spinner(f"Generating forecast for {geo_filter}..."):
    future_years_list = list(range(START_FORECAST_YEAR, START_FORECAST_YEAR + FORECAST_YEARS))
    # Make sure generate_future_features gets the correct list of features expected by the model
    baseline_future_features = generate_future_features(data, geo_filter, START_FORECAST_YEAR, FORECAST_YEARS, selected_features)

# Check if baseline generation was successful before proceeding
if baseline_future_features.empty or baseline_future_features[selected_features].isnull().all().all():
     st.markdown("---") # Add a separator
     st.warning(f"Could not generate a valid baseline forecast for {geo_filter}, possibly due to insufficient historical data. Cannot proceed with adjustments or predictions.")
     # Optionally display the (empty or all-NaN) baseline dataframe for debugging
     # st.dataframe(baseline_future_features)
     st.stop() # Stop execution if baseline is unusable


# --- Apply Adjustments ---
# CRITICAL: Make a deep copy to avoid modifying the baseline dataframe unintentionally
adjusted_future_features = baseline_future_features.copy(deep=True)

# Apply adjustments based on slider values
for food, percent_change in adjustments.items():
    if food in adjusted_future_features.columns:
        # Ensure the column is numeric before applying calculation
        if pd.api.types.is_numeric_dtype(adjusted_future_features[food]):
            adjustment_factor = 1.0 + (percent_change / 100.0)
            # Apply adjustment relative to the *baseline* projected value
            # Use .loc to ensure modification happens on the copy
            adjusted_future_features.loc[:, food] = baseline_future_features[food] * adjustment_factor
            # Optional: Ensure non-negative CPIs if needed
            # adjusted_future_features.loc[:, food] = adjusted_future_features[food].clip(lower=0)
        else:
             # This warning should ideally not appear if data preprocessing is correct
             st.warning(f"Column '{food}' is not numeric. Cannot apply CPI adjustment.")


# --- Optional Debugging Section ---
# Renamed checkbox for clarity
if st.sidebar.checkbox("Show Intermediate Feature Values (Debug)", False):
    st.subheader("Debug: Projected Feature Values Before Scaling")
    st.caption("Shows the projected input values used for the baseline and adjusted forecasts.")
    col_d1, col_d2 = st.columns(2)
    # Display the relevant features for clarity
    display_cols = ['Year'] + food_cols + ['Actual Population']
    # Ensure columns exist before trying to display them
    display_cols = [col for col in display_cols if col in baseline_future_features.columns]
    
    with col_d1:
        st.write(f"Baseline Features ({geo_filter}):")
        st.dataframe(baseline_future_features[display_cols].round(2))
    with col_d2:
        st.write(f"Adjusted Features ({geo_filter}):")
        st.dataframe(adjusted_future_features[display_cols].round(2))


# --- Ensure Correct Column Order and Scale ---
# Make sure both dataframes have the features in the exact order the scaler expects
try:
    # Reorder columns to match the list used for training/scaling
    baseline_features_ordered = baseline_future_features[selected_features]
    adjusted_features_ordered = adjusted_future_features[selected_features]

    # Check for NaNs introduced during generation/adjustment BEFORE scaling
    # Impute NaNs if any exist, as scaler cannot handle them
    if baseline_features_ordered.isnull().any().any():
        st.warning("NaN values detected in baseline features before scaling. Imputing with 0 for prediction.")
        baseline_features_ordered = baseline_features_ordered.fillna(0)
    if adjusted_features_ordered.isnull().any().any():
        st.warning("NaN values detected in adjusted features before scaling. Imputing with 0 for prediction.")
        adjusted_features_ordered = adjusted_features_ordered.fillna(0)

    # Apply the scaler
    baseline_features_scaled = scaler.transform(baseline_features_ordered)
    adjusted_features_scaled = scaler.transform(adjusted_features_ordered)

except KeyError as e:
    st.error(f"Feature mismatch error during scaling: {e}. Columns in generated data ('{baseline_future_features.columns.tolist()}') might not match expected features ('{selected_features}'). Check 'selected_features.pkl'.")
    st.stop()
except ValueError as e:
    st.error(f"Scaling error: {e}. This often happens if NaN values remain in the data passed to the scaler, even after attempting imputation. Check the 'Show Intermediate Feature Values' debug output.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during data scaling: {e}")
    st.stop()


# --- Make Predictions ---
# Use st.spinner for this calculation step as well
with st.spinner("Calculating health outcome predictions..."):
    try:
        baseline_pred_d = model_d.predict(baseline_features_scaled)
        baseline_pred_hbp = model_hbp.predict(baseline_features_scaled)
        adjusted_pred_d = model_d.predict(adjusted_features_scaled)
        adjusted_pred_hbp = model_hbp.predict(adjusted_features_scaled)

    except Exception as e:
         st.error(f"Error during model prediction: {e}")
         st.stop()


# --- Display Results ---
st.header(f"Forecast for {geo_filter} ({START_FORECAST_YEAR}-{START_FORECAST_YEAR + FORECAST_YEARS - 1})")

# Prepare results dataframe
# *** MODIFICATION: Comment out the rounding line to see raw differences ***
results_df = pd.DataFrame({
    'Year': future_years_list,
    'Baseline Diabetes': baseline_pred_d.clip(min=0), # Ensure non-negative predictions
    'Adjusted Diabetes': adjusted_pred_d.clip(min=0), # Uses the prediction from adjusted inputs
    'Baseline HBP': baseline_pred_hbp.clip(min=0),
    'Adjusted HBP': adjusted_pred_hbp.clip(min=0)     # Uses the prediction from adjusted inputs
})
# results_df = results_df.round(2) # <-- Temporarily commented out for debugging differences

# Display the forecast tables
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Diabetes Forecast")
    # Display unrounded values now
    st.dataframe(results_df[['Year', 'Baseline Diabetes', 'Adjusted Diabetes']].set_index('Year'))
with col2:
    st.subheader("📊 High Blood Pressure Forecast")
    # Display unrounded values now
    st.dataframe(results_df[['Year', 'Baseline HBP', 'Adjusted HBP']].set_index('Year'))

# --- Plotting ---
st.subheader("📈 Visual Forecast Trends")
# Filter historical data for plotting
historical_plot_data = data[(data['GEO'] == geo_filter) & (data['Year'] <= LATEST_HISTORICAL_YEAR)]

# Diabetes Plot
fig_d = go.Figure()
if not historical_plot_data.empty:
    fig_d.add_trace(go.Scatter(
        x=historical_plot_data['Year'], y=historical_plot_data['Diabetes_per_capita'],
        mode='lines+markers', name='Historical', line=dict(color='grey')
    ))
fig_d.add_trace(go.Scatter(
    x=results_df['Year'], y=results_df['Baseline Diabetes'], # Plotting potentially unrounded data
    mode='lines+markers', name='Baseline Forecast', line=dict(color='blue', dash='dash')
))
fig_d.add_trace(go.Scatter(
    x=results_df['Year'], y=results_df['Adjusted Diabetes'], # Plotting potentially unrounded data
    mode='lines+markers', name='Adjusted Forecast', line=dict(color='red')
))
fig_d.update_layout(
    title="Diabetes per 1000 People",
    xaxis_title="Year", yaxis_title="Cases per 1000 People", legend_title="Scenario"
)
# Set x-axis range to include historical and future years smoothly
all_years_d = pd.concat([historical_plot_data['Year'], results_df['Year']]).unique()
if len(all_years_d) > 0:
    fig_d.update_xaxes(range=[min(all_years_d) - 1, max(all_years_d) + 1])


# HBP Plot
fig_hbp = go.Figure()
if not historical_plot_data.empty:
    fig_hbp.add_trace(go.Scatter(
        x=historical_plot_data['Year'], y=historical_plot_data['HBP_per_capita'],
        mode='lines+markers', name='Historical', line=dict(color='grey')
    ))
fig_hbp.add_trace(go.Scatter(
    x=results_df['Year'], y=results_df['Baseline HBP'], # Plotting potentially unrounded data
    mode='lines+markers', name='Baseline Forecast', line=dict(color='green', dash='dash')
))
fig_hbp.add_trace(go.Scatter(
    x=results_df['Year'], y=results_df['Adjusted HBP'], # Plotting potentially unrounded data
    mode='lines+markers', name='Adjusted Forecast', line=dict(color='orange')
))
fig_hbp.update_layout(
    title="High Blood Pressure per 1000 People",
    xaxis_title="Year", yaxis_title="Cases per 1000 People", legend_title="Scenario"
)
all_years_hbp = pd.concat([historical_plot_data['Year'], results_df['Year']]).unique()
if len(all_years_hbp) > 0:
    fig_hbp.update_xaxes(range=[min(all_years_hbp) - 1, max(all_years_hbp) + 1])

# Show plots side-by-side
col_p1, col_p2 = st.columns(2)
with col_p1:
    st.plotly_chart(fig_d, use_container_width=True)
with col_p2:
    st.plotly_chart(fig_hbp, use_container_width=True)

# --- Feature Importance ---
st.subheader("🧬 Key Model Drivers (Feature Importance)")
# *** MODIFICATION: Added explanation why these are static ***
st.caption("""
These scores show how much each factor contributed *on average* to the model's predictions **during its training on the entire historical dataset**. 
Higher scores mean the model relied more on that feature historically. 
**These importance scores are static properties of the trained models and do not change when you select different provinces or adjust future CPI sliders.**
""")

col_fi1, col_fi2 = st.columns(2)
with col_fi1:
    # Call the function to get importance data
    diabetes_importance = get_feature_importance(model_d, selected_features)
    if not diabetes_importance.empty:
        fig_fi_d = go.Figure(go.Bar(
            y=diabetes_importance['Feature'],
            x=diabetes_importance['Importance'],
            orientation='h', # Horizontal bar chart
            marker_color='cornflowerblue'
        ))
        fig_fi_d.update_layout(
            title="Diabetes Model Importance", yaxis={'categoryorder':'total ascending'}, height=450, # Adjust height
             xaxis_title="Importance Score", margin=dict(l=180, r=20, t=50, b=50) # Adjust margins
        )
        st.plotly_chart(fig_fi_d, use_container_width=True)

with col_fi2:
    # Call the function to get importance data
    hbp_importance = get_feature_importance(model_hbp, selected_features)
    if not hbp_importance.empty:
        fig_fi_hbp = go.Figure(go.Bar(
             y=hbp_importance['Feature'],
             x=hbp_importance['Importance'],
             orientation='h',
             marker_color='lightseagreen'
        ))
        fig_fi_hbp.update_layout(
            title="HBP Model Importance", yaxis={'categoryorder':'total ascending'}, height=450, # Adjust height
            xaxis_title="Importance Score", margin=dict(l=180, r=20, t=50, b=50) # Adjust margins
        )
        st.plotly_chart(fig_fi_hbp, use_container_width=True)


st.sidebar.markdown("---")
st.sidebar.info("Forecasts are estimates based on historical trends and selected adjustments. Actual outcomes may vary.")