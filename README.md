# AI_ML-Capstone

US Grocery Sales Demand Forecasting (Walmart M5)

Kaggle Dataset Link [https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview]

📌 *Business Value (Why It Matters)*

If this forecasting question remains unanswered, the retailer is forced to rely on guesswork for ordering inventory. This leads to two expensive problems: understocking (lost revenue and frustrated customers) and overstocking (massive amounts of spoiled perishable goods and high storage costs).

By answering this question utilizing highly granular point-of-sale data, this project provides a precise, data-driven purchasing schedule. It empowers store managers to order exactly what they need, drastically reducing food waste, cutting holding costs, and ensuring customers always find the products they want on the shelves.

🎯 Expected Results

A robust demand forecasting model capable of predicting daily unit sales for distinct products (e.g., Foods, Hobbies, Household). It translates raw data—including local prices and government assistance (SNAP) distribution days—into a clear, day-by-day sales expectation.

🔬 Methodology & Techniques

Time Series Feature Engineering: Using lag and rolling windows (7-day and 28-day) to track past sales and identify recurring shopping habits.
Regularized Regression (Ridge & Lasso): A reliable baseline method to measure exactly how much straightforward events (like price drops or SNAP days) impact baseline numbers.
Gradient Boosted Trees (XGBoost): The primary predictive engine, chosen to learn complex, tricky patterns—such as how SNAP distribution days might affect "Foods" heavily but have zero impact on "Hobbies".
📂 Link to Notebook

Click here to view the Jupyter Notebook
Link to [https://github.com/Suraj-BH/AI_ML-Capstone/blob/main/Capstone.ipynb]



Summary of Findings

Zero-Inflation is the Primary Challenge: The dataset consists of millions of sparse transactions where the daily unit sales for many individual items are exactly zero. Tree-based models (specifically LightGBM) drastically outperformed linear baselines (Ridge/Lasso) because they can handle non-linear logic and sparse data matrices without predicting negative inventory.
Memory Features Drive Accuracy: Standard date features (Month, Day of Week) were helpful, but supervised "memory" features engineered from the time-series were the ultimate drivers of accuracy. Specifically, the 28-day rolling average and 7-day lag features were the strongest predictors of future demand.
LightGBM Wins on Scale: While XGBoost and LightGBM produced very similar RMSE accuracy scores, LightGBM trained in a fraction of the time. In a real-world scenario with millions of items across thousands of stores, computational cost equals financial cost, making LightGBM the optimal production engine.
Actionable Business Value: The model successfully learned to anticipate weekly demand spikes (weekends) and economic signals (SNAP distribution days in California). Following this forecast allows store managers to order bulk inventory immediately before a spike, and taper orders down ahead of slow weekdays, effectively minimizing both stockouts and food spoilage.

Next Steps (Phase 2 Development)

Scale to Full Dataset & State Expansion: Expand the pipeline from a single category (FOODS) in a single store (CA_1) to the entire Walmart M5 dataset (Hobbies, Household) across all 10 stores in California, Texas, and Wisconsin.
Implement Hierarchical Reconciliation: Apply Optimal Reconciliation techniques (e.g., MinT or Top-Down/Bottom-Up distributions). This will ensure that the individual item-level forecasts sum up perfectly to match the macro-level Store and State forecasts, providing a coherent unified plan for both store managers and regional CFOs.
Deepen Economic & Event Signals: Engineer more granular features out of the calendar.csv file, specifically isolating the impact of major religious holidays, sporting events (like the Superbowl), and long weekends, mapping their specific elasticities to different product categories (e.g., party supplies vs. fresh produce).
Memory Optimization: Implement heavy data-downcasting (converting float64 to float16 and int32 to int8) within the Pandas dataframes to allow the processing of the full 5-year dataset on standard RAM limits.


