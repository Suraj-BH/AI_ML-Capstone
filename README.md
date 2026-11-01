# AI_ML-Capstone

US Grocery Sales Demand Forecasting (Walmart M5)

Kaggle Dataset Link [https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview]

### Business Value (Why It Matters)

If this forecasting question remains unanswered, the retailer is forced to rely on guesswork for ordering inventory. This leads to two expensive problems: understocking (lost revenue and frustrated customers) and overstocking (massive amounts of spoiled perishable goods and high storage costs).

By answering this question utilizing highly granular point-of-sale data, this project provides a precise, data-driven purchasing schedule. It empowers store managers to order exactly what they need, drastically reducing food waste, cutting holding costs, and ensuring customers always find the products they want on the shelves.

### Expected Results

A robust demand forecasting model capable of predicting daily unit sales for distinct products (e.g., Foods, Hobbies, Household). It translates raw data—including local prices and government assistance (SNAP) distribution days—into a clear, day-by-day sales expectation.

### Methodology & Techniques

Time Series Feature Engineering: Using lag and rolling windows (7-day and 28-day) to track past sales and identify recurring shopping habits.
Regularized Regression (Ridge & Lasso): A reliable baseline method to measure exactly how much straightforward events (like price drops or SNAP days) impact baseline numbers.
Gradient Boosted Trees (XGBoost): The primary predictive engine, chosen to learn complex, tricky patterns—such as how SNAP distribution days might affect "Foods" heavily but have zero impact on "Hobbies".

### Link to Notebook

Click here to view the Initial EDA Jupyter Notebook
Link to [https://github.com/Suraj-BH/AI_ML-Capstone/blob/main/Capstone.ipynb]

Click here to view the Final Capstone Jupyter Notebook
Link to [https://github.com/Suraj-BH/AI_ML-Capstone/blob/main/Capstone.ipynb]

### Summary of Findings

Zero-Inflation is the Primary Challenge: The dataset consists of millions of sparse transactions where the daily unit sales for many individual items are exactly zero. Tree-based models (specifically LightGBM) drastically outperformed linear baselines (Ridge/Lasso) because they can handle non-linear logic and sparse data matrices without predicting negative inventory.
Memory Features Drive Accuracy: Standard date features (Month, Day of Week) were helpful, but supervised "memory" features engineered from the time-series were the ultimate drivers of accuracy. Specifically, the 28-day rolling average and 7-day lag features were the strongest predictors of future demand.
LightGBM Wins on Scale: While XGBoost and LightGBM produced very similar RMSE accuracy scores, LightGBM trained in a fraction of the time. In a real-world scenario with millions of items across thousands of stores, computational cost equals financial cost, making LightGBM the optimal production engine.
Actionable Business Value: The model successfully learned to anticipate weekly demand spikes (weekends) and economic signals (SNAP distribution days in California). Following this forecast allows store managers to order bulk inventory immediately before a spike, and taper orders down ahead of slow weekdays, effectively minimizing both stockouts and food spoilage.

### Next Steps (Phase 2 Development)

Scale to Full Dataset & State Expansion: Expand the pipeline from a single category (FOODS) in a single store (CA_1) to the entire Walmart M5 dataset (Hobbies, Household) across all 10 stores in California, Texas, and Wisconsin.
Implement Hierarchical Reconciliation: Apply Optimal Reconciliation techniques (e.g., MinT or Top-Down/Bottom-Up distributions). This will ensure that the individual item-level forecasts sum up perfectly to match the macro-level Store and State forecasts, providing a coherent unified plan for both store managers and regional CFOs.
Deepen Economic & Event Signals: Engineer more granular features out of the calendar.csv file, specifically isolating the impact of major religious holidays, sporting events (like the Superbowl), and long weekends, mapping their specific elasticities to different product categories (e.g., party supplies vs. fresh produce).
Memory Optimization: Implement heavy data-downcasting (converting float64 to float16 and int32 to int8) within the Pandas dataframes to allow the processing of the full 5-year dataset on standard RAM limits.


---

## Phase 2: Classical Time Series Forecasting 

### Expanding the Scope
Following the initial exploratory data analysis and baseline modeling in Phase 1, Phase 2 scales the business problem across all 10 stores in California, Texas, and Wisconsin. To align with the strict statistical rigor of the **UC Berkeley Professional Certificate in Machine Learning and AI**, we pivoted from black-box machine learning algorithms to highly interpretable **Classical Time Series** methodologies. We focused specifically on the `HOBBIES` category to build a robust, store-level inventory forecasting system.

### Modeling and Evaluation
Instead of treating forecasting as a simple regression problem, we treated it as a structural statistical problem. 
* **Time Series Decomposition:** We broke down the daily sales data into three distinct mathematical components: **Trend** (the underlying direction of sales), **Seasonality** (the repeating weekly spikes), and **Residuals** (the random noise).
* **Holt-Winters Exponential Smoothing:** We applied an Additive Holt-Winters model for each of the 10 stores. This algorithm explicitly learns the weekly seasonal patterns and the overall trend extracted during the decomposition phase to predict future sales.
* **Evaluation Metric:** We evaluated the models using **RMSE (Root Mean Squared Error)** over a 28-day holdout test period, as it heavily penalizes large, costly inventory forecasting errors.

### Key Findings
1. **Strong Weekly Seasonality:** The Time Series Decomposition revealed a strict 7-day recurring cycle across the stores. Sales consistently spike on weekends and dip mid-week. Capturing this mathematically allows for precise intra-week supply chain adjustments.
2. **Interpretability over Complexity:** The Holt-Winters model provided highly competitive RMSE scores while remaining completely transparent. Unlike black-box machine learning models, stakeholders can explicitly see *why* the model predicts a spike (e.g., "The model added 50 units due to the Saturday seasonal factor").
3. **Unbiased Error Distribution:** Analysis of the model's continuous residuals (errors) showed a normal distribution tightly centered around zero. This means the model is mathematically unbiased—it does not chronically over-forecast or under-forecast.

### Recommendations & Next Steps
1. **Dynamic Weekly Restocking:** Implement the Holt-Winters baseline forecasts into the store managers' weekly ordering systems. Because the model precisely captures the weekly seasonality, stores can shift deliveries from Mondays to Thursdays to anticipate the weekend surge without increasing total weekly holding costs.
2. **Threshold Alerts:** Utilize the historical residual distribution to set automated alerts. If actual sales deviate from the Holt-Winters forecast by more than 2 standard deviations, the system should flag the store manager to investigate local anomalies (e.g., a local event or a competitor stock-out).
3. **Next Steps for Data (Phase 3):** Incorporate an ARIMA/SARIMAX model variant to test whether adding exogenous variables (like SNAP benefit days or local weather data) reduces the residual noise further than pure Exponential Smoothing.

