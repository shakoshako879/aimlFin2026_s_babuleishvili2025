# DDoS Attack Detection Report

## Overview
In this report, I detail the steps taken to analyze the web server log file for detecting DDoS attack intervals using regression analysis. The analysis was performed using Python with libraries like pandas and statsmodels. The goal was to identify periods of unusually high request volumes, indicative of a DDoS attack, by modeling the request rates and detecting outliers via regression residuals.

The log file contains entries from a web server, including timestamps, which I parsed to aggregate request counts over time. I then applied linear regression to identify anomalous periods.

## Data Source
The event log file used for this analysis can be found at: [Event Log File](https://github.com/shakoshako879/aimlFin2026_s_babuleishvili2025/blob/main/task_3/s_babuleishvili25_67893_server.log)

## Methodology
### Step 1: Parsing the Log File
I read the log file and extracted timestamps from each entry. The timestamps are in the format `[YYYY-MM-DD HH:MM:SS+04:00]`. I used Python's `datetime` module to parse them.

Key code fragment:
```python
from datetime import datetime

with open('s_babuleishvili25_67893_server.log', 'r') as f:
    lines = f.readlines()

timestamps = []
for line in lines:
    start = line.find('[')
    end = line.find(']')
    if start != -1 and end != -1:
        ts_str = line[start+1:end]
        try:
            ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S%z')
            timestamps.append(ts)
        except ValueError:
            pass
```

### Step 2: Aggregating Requests per Minute
Using pandas, I created a DataFrame from the timestamps and resampled to get request counts per minute.

Key code fragment:
```python
import pandas as pd

df = pd.DataFrame({'timestamp': timestamps})
df['count'] = 1
df.set_index('timestamp', inplace=True)
counts_per_min = df.resample('1min').sum()  # Note: '1T' was used but deprecated; use '1min'
```

This gave me a time series of request counts, revealing the overall distribution. The mean count per minute was approximately 1090, with a standard deviation of about 1373. The maximum was 6561, and minimum 104.

### Step 3: Regression Analysis for Outlier Detection
To detect the DDoS intervals, I performed linear regression on the request counts over time (treating time as a numeric variable from 0 to 60). I fit an Ordinary Least Squares (OLS) model and used studentized residuals to identify outliers. Periods where the absolute studentized residual exceeded 3 were flagged as anomalous, indicating potential DDoS activity.

Key code fragment:
```python
import statsmodels.api as sm

df_reg = pd.DataFrame({'time': range(len(counts_per_min)), 'count': counts_per_min['count'].values})
X = sm.add_constant(df_reg['time'])
model = sm.OLS(df_reg['count'], X).fit()
influence = model.get_influence()
df_infl = influence.summary_frame()
df_reg['student_resid'] = df_infl['student_resid']
outliers = df_reg[abs(df_reg['student_resid']) > 3]
outlier_indices = outliers.index
outlier_times = counts_per_min.index[outlier_indices]
```

This approach models the expected linear trend in request rates (though minimal in this data) and highlights significant deviations.

### Step 4: Visualizations
To visualize the request rates and highlight the attack periods, I generated plots using matplotlib. The first plot shows request counts per minute, with outliers marked in red. The second plot shows the cumulative requests over time, where steep increases indicate the attack.

Key code fragment for plotting:
```python
import matplotlib.pyplot as plt

# Plot requests per minute
plt.figure(figsize=(12, 6))
plt.plot(counts_per_min.index, counts_per_min['count'], label='Requests per Minute')
plt.scatter(outlier_times, outliers['count'], color='red', label='Outliers (DDoS)')
plt.title('Web Server Requests per Minute')
plt.xlabel('Time')
plt.ylabel('Request Count')
plt.legend()
plt.grid(True)
plt.savefig('requests_per_minute.png')  # Save for inclusion in reports
plt.show()

# Plot cumulative requests
df_reg['cumulative'] = df_reg['count'].cumsum()
plt.figure(figsize=(12, 6))
plt.plot(counts_per_min.index, df_reg['cumulative'], label='Cumulative Requests')
plt.title('Cumulative Web Server Requests Over Time')
plt.xlabel('Time')
plt.ylabel('Cumulative Count')
plt.grid(True)
plt.savefig('cumulative_requests.png')
plt.show()
```

These plots clearly show spikes during the identified intervals.

## Results
The regression analysis identified the following time intervals as the DDoS attack periods based on outlier detection:

- **2024-03-22 18:13:00+04:00 to 2024-03-22 18:17:00+04:00**

(This covers the four consecutive minutes of extreme request volumes: 6290, 6561, 5580, and 5707 requests, respectively. The end time is exclusive, effectively up to 18:16:59+04:00.)

No other intervals showed such significant deviations.

## Reproduction Instructions
1. Download the log file and place it in your working directory as 's_babuleishvili25_67893_server.log'.
2. Install required libraries: `pip install pandas statsmodels matplotlib` (though in the environment, they are pre-installed).
3. Run the code fragments in sequence in a Python script or Jupyter notebook.
4. The outliers will be printed, and plots saved as PNG files.

5. Adjust the residual threshold if needed (I used >3 for strict outlier detection).
