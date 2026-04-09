import pandas as pd
import matplotlib.pyplot as plt

# PREPROCESSING TEMPERATURE DATASET
temp_files = ['/Users\lilia\OneDrive\Desktop\Junior Project\Temp & Humidity\Temp Data Kitchen.csv',
              '/Users\lilia\OneDrive\Desktop\Junior Project\Temp & Humidity\Temp Data Mailroom.csv',
              '/Users\lilia\OneDrive\Desktop\Junior Project\Temp & Humidity\Temp Data Lab.csv']
labels = ['Kitchen', 'Mailroom', 'Lab']

plt.figure(figsize=(12, 6))

for file, label in zip(temp_files, labels):
    # 1. Load Data
    df = pd.read_csv(file)
    
    # 2. Standardize by Position (Safest way to avoid KeyErrors)
    # This renames the 1st column to 'timestamp' and 2nd to 'temp'
    df.columns = ['timestamp', 'temp'] + list(df.columns[2:])
    
    # 3. Formatting Data Types
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
    
    # 4. Handle Missing Values
    df = df.dropna(subset=['timestamp', 'temp'])
    df = df.sort_values('timestamp')
    
    # 5. Visualization
    plt.plot(df['timestamp'], df['temp'], label=f'{label} (°C)')

plt.title('Temperature Dataset (Standardized via Positional Indexing)')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# PREPROCESSING HUMIDITY DATASET
hum_files = ['/Users\lilia\OneDrive\Desktop\Junior Project\Temp & Humidity\Humidity Data Kitchen.csv',
             '/Users\lilia\OneDrive\Desktop\Junior Project\Temp & Humidity\Humidity Data Mailroom.csv',
             '/Users\lilia\OneDrive\Desktop\Junior Project\Temp & Humidity\Humidity  Data Lab.csv']
labels = ['Kitchen', 'Mailroom', 'Lab']

plt.figure(figsize=(12, 6))

for file, label in zip(hum_files, labels):
    df = pd.read_csv(file)
    
    # Standardize by Position (1st col: time, 2nd col: humidity)
    df.columns = ['timestamp', 'humidity'] + list(df.columns[2:])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
    
    df = df.dropna(subset=['timestamp', 'humidity'])
    df = df.sort_values('timestamp')
    
    plt.plot(df['timestamp'], df['humidity'], label=f'{label} (%)', linestyle='--')

plt.title('Humidity Dataset (Standardized via Positional Indexing)')
plt.xlabel('Time')
plt.ylabel('Humidity (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df.to_csv('sensor_data.csv', index=False)