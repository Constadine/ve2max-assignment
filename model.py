import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load, process, and clean dataset
# ---------------------------------------------------------------------
def load_data(file_path, lines=None):
    if lines:
        data = pd.read_csv(file_path, nrows=lines)
    else:
        data = pd.read_csv(file_path)
    
    data = data.drop(columns=['temp_t+1', 'feels_like_t+1'])
    
    data['date'] = pd.to_datetime(data['date'])
    
    return data

# Exploration Vizualization
# --------------------------------------------------------------------
def exploration_plot(data, variables_to_plot, period=None):
    for variable in variables_to_plot:  
        if period:
            to_plot = data[['date', variable]].groupby(pd.Grouper(key='date',freq=period)).mean().reset_index()
        else:
            to_plot = data[['date', variable]]
        # Plot active_power over time
        plt.figure(figsize=(14, 7))
        plt.plot(to_plot['date'], to_plot[variable], label=variable.capitalize())
        plt.xlabel('Date')
        plt.ylabel(variable.capitalize())
        plt.title(f'{variable.capitalize()} Over Time')
        plt.legend()
        plt.show()
    
    

# Load, process, and clean dataset
# ---------------------------------------------------------------------
file_path = 'data.csv'
data = load_data(file_path, lines=None)

missing_values = data.isnull().sum()

data_description = data.describe()


# Exploration Vizualization
# --------------------------------------------------------------------
variables_to_plot = ['temp', 'active_power']
exploration_plot(data, variables_to_plot, period='M')
 
# Modeling
# --------------------------------------------------------------------

# Time-based Features
data['hour'] = data['date'].dt.hour
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month

# Normalize
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['date', 'main', 'description', 'active_power']))

columns_to_scale = data.columns.drop(['date', 'main', 'description', 'active_power'])

scaled_data = pd.DataFrame(scaled_features, columns=columns_to_scale)
scaled_data['active_power'] = data['active_power']
scaled_data['date'] = data['date']

# Build Model
# -------------------------------------------------------------------- 

# Split data
X = scaled_data.drop(columns=['date', 'active_power'])
y = scaled_data['active_power']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mae, mse, r2
    
# Save model
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')


