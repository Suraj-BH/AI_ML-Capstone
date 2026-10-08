# AI_ML-Capstone

US Grocery Sales Demand Forecasting (Walmart M5)

📌 Business Value (Why It Matters)

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
