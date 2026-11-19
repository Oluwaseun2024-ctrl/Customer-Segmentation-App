# CUSTOMER SEGMENTATION AND BEHAVIOURAL ANALYTICS

Table of Content
- Project Overview
- Data Understanding
- Data Preparation and Cleaning
- Exploratory Data Analysis (EDA)
- Customer-Level Feature Engineering
- Construction of the Final Customer-Level Dataset
- Feature Selection and Preprocessing
- Feature Transformation
- Feature Scaling
- Dimensionality Reduction (PCA)
- Customer Segmentation Using K-Means
- Customer Cluster Profiling
- Business Performance Analysis
- Business Insights and Recommendations
- Model Deployment and Application Development

## PROJECT OVERVIEW
**Project Description**

The Customer Segmentation and Behavioural Analytics project focuses on analysing customer demographics, purchasing patterns, spending behaviour, product preferences, and marketing engagement to identify distinct customer groups within an e-commerce retail business.

The project applies data analytics and unsupervised machine learning techniques, particularly Principal Component Analysis (PCA) and K-Means Clustering, to group customers based on similarities in their characteristics and purchasing behaviour.

The trained model is deployed through an interactive Streamlit application that supports individual and bulk customer segmentation, segment identification, business recommendations, visual summaries, and downloadable results.

**Project Background**

Understanding customer behaviour helps businesses improve marketing strategies, customer retention, and revenue generation. However, customers differ in their spending patterns, purchase frequency, product preferences, and marketing engagement.

This project uses historical customer and transaction data to uncover these behavioural differences, identify meaningful customer segments, and provide insights that support data-driven business decisions.

**Problem Statement**

Businesses often collect large volumes of customer data without an effective system for identifying distinct customer behaviours. Treating all customers similarly can lead to ineffective marketing campaigns, missed retention opportunities, and poor customer targeting.

This project addresses the challenge by developing a machine learning-based segmentation system that groups customers according to their demographic and purchasing characteristics and provides actionable business recommendations.

**Project Objectives**

The project aims to:
- Integrate and clean customer and transaction datasets.
- Perform exploratory data analysis to understand customer behaviour.
- Engineer customer-level features from transaction records.
- Select and preprocess relevant features for clustering.
- Apply PCA for dimensionality reduction.
- Implement K-Means clustering to identify customer segments.
- Evaluate and interpret the resulting clusters.
- Develop segment-specific business recommendations.
- Deploy the trained model using Streamlit for individual and bulk customer segmentation.

**Project Scope**

The project covers data integration, cleaning, exploratory data analysis, feature engineering, feature selection, dimensionality reduction, customer clustering, model evaluation, business interpretation, and deployment.

The analysis considers customer demographics, spending, purchase frequency, recency, product diversity, returns, discounts, marketing engagement, profitability, and sales channel behaviour.
The project focuses on historical behavioural segmentation and does not attempt to predict customer lifetime value or guarantee future purchasing behaviour.

**Tools and Technologies Used**
| Tool / Technology | Application |
|---|---|
| Python | Programming and analysis |
| Pandas and NumPy | Data manipulation and computation |
| Matplotlib and Seaborn | Data visualisation |
| Scikit-learn | Preprocessing, PCA, K-Means and evaluation |
| Joblib | Model serialization |
| Streamlit | Web application development |
| VS Code | Development environment |
| Git and GitHub | Version control and repository hosting |
| Streamlit Community Cloud | Application deployment |

## DATA UNDERSTANDING
**Dataset Overview**

The project uses two datasets representing customer demographic information and historical transaction records from an e-commerce retail business.

The first dataset (df1) contains customer information, while the second dataset (df2) contains transaction-level records. These datasets provide the foundation for analysing customer characteristics, purchasing behaviour, spending patterns, and engagement.

The initial dataset dimensions are presented below:
| Dataset | Description | Rows | Columns |
|---|---|---:|---:|
| df1 | Customer Demographics | 10,040 | 11 |
| df2 | Transaction Records | 65,889 | 15 |

**Dataset Structure and Variables**

The project consists of two datasets with different but complementary structures:
- Customer Dataset (df1): Contains customer-level demographic and profile information, with 11 variables.
- Transaction Dataset (df2): Contains individual transaction records and purchasing-related information, with 15 variables.

The customer dataset provides information about who the customers are, while the transaction dataset captures their purchasing activities and interactions with the business.

**Data Sources and File Structure**

The two datasets were loaded into separate Pandas DataFrames for independent inspection and subsequent processing:
- df1 – Customer demographic dataset.
- df2 – Transaction dataset.

Both datasets were retained separately at the initial stage to allow their structures, contents, and data quality to be examined before integration.

**Common Identifier and Data Integration**
The two datasets share a common identifier, customer_id, which serves as the linking key between customer records and their associated transactions.

This identifier provides the basis for combining customer demographic information with historical purchasing records to create a unified dataset for subsequent analysis.

**Data Understanding Summary**

The initial inspection established that the project contains 10,040 customer records and 65,889 transaction records across two datasets, with a combined total of 26 columns before integration.

Understanding the structure and relationship between these datasets provides the foundation for the next stage: Data Preparation and Cleaning.

**Data Dictionary**

A. Customers table – 11 columns
| Column | Data type | Description |
|---|---|---|
| customer_id | String | Unique customer identifier |
| age | Numeric | Customer's age |
| annual_income_ngn | Numeric | Estimated annual income in naira |
| region | Categorical | Customer's geographical region |
| city | Categorical | Customer's city |
| signup_date | Date | Customer registration date |
| acquisition_channel | Categorical | Original customer acquisition channel |
| membership_tier | Categorical | Standard, Silver, Gold or Platinum |
| preferred_contact_channel | Categorical | Preferred communication channel |
| marketing_consent | Categorical | Whether customer permits marketing |
| customer_satisfaction_score | Numeric | Customer satisfaction score (1–5) |

B. Transactions table – 15 columns
| Column | Data type | Description |
|---|---|---|
| transaction_id | String | Unique transaction identifier |
| customer_id | String | Customer foreign key |
| transaction_date | Date | Date of purchase |
| product_category | Categorical | Product category purchased |
| units | Integer | Quantity purchased |
| unit_price_ngn | Numeric | Price per unit |
| discount_pct | Numeric | Discount applied (0–1) |
| gross_amount_ngn | Numeric | Transaction amount before discount |
| net_amount_ngn | Numeric | Revenue after discount and returns |
| cost_amount_ngn | Numeric | Cost associated with the transaction |
| payment_method | Categorical | Payment method used |
| sales_channel | Categorical | Purchase channel |
| order_status | Categorical | Completed or returned |
| campaign_id | String | Associated marketing campaign |
| campaign_response | Categorical | Customer campaign response |

## DATA PREPARATION AND CLEANING
This stage involved preparing the customer and transaction datasets for analysis by removing duplicate records, integrating the datasets, inspecting data quality, correcting inconsistencies, and handling missing and abnormal values.

**Duplicate Record Identification and Removal**

Duplicate records were identified in both datasets using Pandas' duplicated() method. The detected duplicates were removed using drop_duplicates() to prevent repeated records from affecting subsequent analysis.

Python Implementation:
```Python
# Check and remove duplicates in customer dataset
df1.duplicated().sum()
df1.drop_duplicates(inplace=True)

# Check and remove duplicates in transaction dataset
df2.duplicated().sum()
df2.drop_duplicates(inplace=True)
```

Results:
| Dataset | Initial Records | Duplicates Removed | Remaining Records |
|---|---:|---:|---:|
| Customer (df1) | 10,040 | 40 | 10,000 |
| Transactions (df2) | 65,889 | 250 | 65,639 |
Conclusion: Duplicate records were successfully removed, leaving 10,000 customer records and 65,639 transaction records for further processing.

**Dataset Integration and Merging**

The cleaned customer and transaction datasets were merged using customer_id as the common identifier. A left join was applied, using the transaction dataset as the left table, to retain all transaction records while attaching their corresponding customer information.

Python Implementation:
```Python
df = pd.merge(
    df2,
    df1,
    on='customer_id',
    how='left'
)

print("Transactions shape:", df2.shape)
print("Customers shape:", df1.shape)
print("Merged dataset shape:", df.shape)
```

Results:
| Dataset | Rows | Columns |
|---|---:|---:|
| Transactions | 65,639 | 15 |
| Customers | 10,000 | 11 |
| Merged Dataset | 65,639 | 25 |
Conclusion: The merging operation produced a unified dataset containing 65,639 transaction records and 25 columns, combining transaction-level information with customer demographics.

**Data Inspection and Type Conversion**

The merged dataset was inspected to examine its columns, data types, and general structure. The transaction and signup date columns were converted to datetime format to support date-based operations and subsequent feature engineering.

Python Implementation:
```python
# Display dataset columns
for col in df.columns:
    print(col)

# Convert date columns
date_columns = [
    'transaction_date',
    'signup_date'
]

for col in date_columns:
    df[col] = pd.to_datetime(
        df[col],
        errors='coerce'
    )
```
Conclusion: The initial inspection established the structure of the merged dataset, while datetime conversion prepared the date variables for chronological analysis.

**Data Quality Assessment**

A column-by-column assessment was conducted to identify missing values, duplicates, inconsistent categories, and invalid numerical observations.
The inspection covered the following variables:
| Variable | Quality Checks Performed |
|---|---|
| transaction_id | Missing values and duplicates |
| customer_id | Missing values |
| transaction_date | Missing dates |
| product_category | Missing and inconsistent categories |
| units | Missing, negative and unusual values |
| unit_price_ngn | Missing and unreasonable prices |
| discount_pct | Missing, negative and above-range values |
| cost_amount_ngn | Missing and negative values |
| payment_method | Missing and unique categories |
| sales_channel | Missing and unique categories |
| order_status | Missing and inconsistent categories |
| campaign_id | Missing values |
| age | Missing and unreasonable ages |
| annual_income_ngn | Missing and negative values |
| region, city | Missing and inconsistent geographical values |
| signup_date | Missing dates |
| acquisition_channel | Missing and unique categories |
| membership_tier | Missing and unique categories |
| preferred_contact_channel | Missing and unique categories |
| marketing_consent | Missing and unique values |
| customer_satisfaction_score | Missing and out-of-range scores |

Conclusion: The assessment identified the variables requiring cleaning and established the basis for applying appropriate treatment methods according to their data types and distributions.

**Handling Missing and Inconsistent Values**

**Product Category**

Missing product categories were replaced with Unknown. Title casing and whitespace removal were also applied to standardise category labels.

Script:
```python
df['product_category'].fillna('Unknown', inplace=True)

df['product_category'] = (
    df['product_category']
    .apply(lambda x: x.title().strip())
)
```

**Units**

Negative unit values were treated as invalid and converted to missing values. A histogram was examined to understand the distribution, and the median was used for imputation because the distribution was skewed.

Script:
```python
df.loc[df['units'] < 0, 'units'] = np.nan

# Replace missing values with median
df['units'].fillna(
    df['units'].median(),
    inplace=True
)
```

Distribution of Units Before Median Imputation

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Units%20Distribution.png)

**Unit Price**

Unit prices were examined using descriptive statistics and a histogram. To identify unusual observations, category-specific Interquartile Range (IQR) boundaries were calculated, allowing prices to be assessed relative to their respective product categories.

Script:
```python
price_bounds = (
    df.groupby('product_category')['unit_price_ngn']
    .agg(
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    )
)

price_bounds['IQR'] = (
    price_bounds['Q3'] - price_bounds['Q1']
)

price_bounds['lower_bound'] = (
    price_bounds['Q1'] - 1.5 * price_bounds['IQR']
)

price_bounds['upper_bound'] = (
    price_bounds['Q3'] + 1.5 * price_bounds['IQR']
)
```

The calculated boundaries were mapped to the original dataset, and observations outside the category-specific boundaries were flagged for further investigation.
After examining the flagged transactions, 15 records containing an extreme unit price of ₦75,000,000 were identified as confirmed price errors. These values were replaced with missing values and subsequently imputed using the median unit price.

Script:
```python
# Identify confirmed price errors
extreme_price_mask = (
    df['unit_price_ngn'].eq(75_000_000)
)

# Preserve an audit flag
df['confirmed_price_error'] = extreme_price_mask

# Replace erroneous values
df.loc[
    extreme_price_mask,
    'unit_price_ngn'
] = np.nan

# Median imputation
df['unit_price_ngn'].fillna(
    df['unit_price_ngn'].median(),
    inplace=True
)
```

Distribution of Unit Prices Before Outlier Treatment

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Units%20Price%20Distribution%20Before%20Treatment.png)

Distribution of Unit Prices After Outlier Treatment and Imputation

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Units%20Price%20Distribution%20After%20Treatment.png)

Conclusion: Category-specific IQR boundaries helped identify unusual prices for investigation. Confirmed erroneous values were treated separately from other flagged observations to avoid automatically removing legitimate high-priced transactions.

**Discount Percentage**

Discount values greater than 1 were treated as invalid and converted to missing values. The median was subsequently used to replace missing observations.

Script:
```python
df.loc[
    df['discount_pct'] > 1,
    'discount_pct'
] = np.nan

df['discount_pct'].fillna(
    df['discount_pct'].median(),
    inplace=True
)
```

Distribution of Discount Percentage

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Discount%20Percentage.png)

**Payment Method**

Missing payment method entries were replaced with Unknown to preserve transaction records without assigning unsupported payment categories.

Script:
```python
df['payment_method'].fillna(
    'Unknown',
    inplace=True
)
```

**Order Status**

Inconsistent formatting in the order status column was standardised using title casing and whitespace removal.

Script:
```python
df['order_status'] = (
    df['order_status']
    .apply(lambda x: x.title().strip())
)
```

**Age**

Age values below 10 and above 100 were treated as invalid and converted to missing values. The resulting missing observations were replaced using the median age.

Script:
```python
df.loc[df['age'] < 10, 'age'] = np.nan
df.loc[df['age'] > 100, 'age'] = np.nan

df['age'].fillna(
    df['age'].median(),
    inplace=True
)
```

Distribution of Customer Age

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Age.png)

**Annual Income**

The annual income distribution was examined, and missing values were replaced with the median to reduce the influence of extreme observations on imputation.

Script:
```python
df['annual_income_ngn'].fillna(
    df['annual_income_ngn'].median(),
    inplace=True
)
```

Distribution of Annual Income

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Annual%20Income.png)

**Region and City**

Region values were standardised using title casing and whitespace removal, while missing city entries were replaced with Unknown.

Script:
```python
df['region'] = (
    df['region']
    .apply(lambda x: x.title().strip())
)

df['city'].fillna(
    'Unknown',
    inplace=True
)
```

**Customer Satisfaction Score**

Missing satisfaction scores were replaced using the median score after examining the distribution of the variable.

Script:
```
df['customer_satisfaction_score'].fillna(
    df['customer_satisfaction_score'].median(),
    inplace=True
)
```

Distribution of Customer Satisfaction Scores

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Satisfaction%20Scores.png)

**Validation of Calculated Columns**

The gross and net transaction amounts were recalculated to ensure consistency between the transaction quantities, unit prices, and discount percentages.
Gross amount was calculated as the product of units purchased and unit price, while net amount was obtained by deducting the applicable discount from the gross amount.

Python Implementation:
```python
Gross Amount
# Gross Amount = Units × Unit Price
df['gross_amount_ngn'] = (
    df['units'] * df['unit_price_ngn']
)

Net Amount
# Net Amount = Gross Amount - Discount Amount
df['net_amount_ngn'] = (
    df['gross_amount_ngn'] -
    (df['gross_amount_ngn'] * df['discount_pct'])
)
```

Conclusion: The calculated columns were regenerated using the cleaned transaction variables to ensure that gross and net amounts reflected the corrected units, unit prices, and discount percentages. These variables provide the foundation for subsequent customer spending, revenue, and profitability analysis.

**Final Missing Value Assessment**

After applying the cleaning and imputation procedures, the dataset was checked again to examine the remaining missing values.

Python Implementation:
```python
df.isna().sum()
```

## EXPLORATORY DATA ANALYSIS (EDA)

Exploratory Data Analysis was performed to understand customer demographics, purchasing patterns, spending behaviour, product preferences, sales channel usage, and marketing engagement. The analysis focused on identifying patterns and trends that could inform customer-level feature engineering and subsequent segmentation.

**Purchasing Behaviour Analysis**

Business Question: How frequently do customers make purchases, and how does purchasing activity vary across customers?

The order status distribution was examined to distinguish completed transactions from returned transactions. Only completed transactions were retained for the purchasing behaviour analysis to ensure that purchase frequency reflected successful purchases.

Python Implementation:
```python
# Examine transaction order statuses
df['order_status'].value_counts(dropna=False)

# Filter completed transactions
completed_df = df[
    df['order_status'] == 'Completed'
].copy()
```

Results:
| Order Status | Transaction Count |
|---|---:|
| Completed | 62,271 |
| Returned | 3,368 |
| **Total** | **65,639** |

The completed transactions were used to calculate the number of unique completed purchases made by each customer.

```python
# Calculate purchase frequency per customer
purchase_frequency = (
    completed_df.groupby('customer_id')['transaction_id']
    .nunique()
    .reset_index(name='purchase_frequency')
)

# Descriptive statistics
purchase_frequency['purchase_frequency'].describe()
```

Results:
| Statistic | Purchase Frequency |
|---|---:|
| Customers | 9,912 |
| Mean | 6.28 |
| Standard Deviation | 4.64 |
| Minimum | 1 |
| 25th Percentile | 2 |
| Median | 5 |
| 75th Percentile | 10 |
| Maximum | 25 |

Interpretation: The analysis shows that customers made an average of approximately 6 completed purchases, with a median of 5 purchases. Purchase frequency ranged from 1 to 25 transactions, indicating considerable variation in purchasing activity across the customer base.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Purchase%20Frequency.png)

**Customer Spending Behaviour**

Business Question: How much does each customer spend, and how does spending vary across the customer base?

Total customer spending was calculated by aggregating the net transaction amounts of completed purchases for each customer.

Python Implementation:
```python
# Calculate total spending per customer
customer_spending = (
    completed_df.groupby('customer_id')['net_amount_ngn']
    .sum()
    .reset_index(name='total_spending_ngn')
)

# Descriptive statistics
customer_spending['total_spending_ngn'].describe()
```

Results:
| Statistic | Total Spending (NGN) |
|---|---:|
| Customers | 9,912 |
| Mean | 471,104.70 |
| Standard Deviation | 500,038.30 |
| Minimum | 1,945.04 |
| 25th Percentile | 119,622.30 |
| Median | 329,314.10 |
| 75th Percentile | 630,604.30 |
| Maximum | 4,824,470.00 |

Interpretation: Customers spent an average of approximately ₦471,104.70, while the median spending was ₦329,314.10. The difference between the mean and median suggests a positively skewed spending distribution, indicating that some customers spent considerably more than others.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Total%20Customer%20Spending.png)

**Average Order Value (AOV)**

Business Question: How much does a customer spend, on average, per completed transaction?

Average Order Value was calculated as the mean net transaction amount for each customer.

Python Implementation:
```python
# Calculate average order value per customer
customer_aov = (
    completed_df.groupby('customer_id')['net_amount_ngn']
    .mean()
    .reset_index(name='average_order_value_ngn')
)

# Descriptive statistics
customer_aov['average_order_value_ngn'].describe()
```

Results:
| Statistic | Average Order Value (NGN) |
|---|---:|
| Customers | 9,912 |
| Mean | 81,538.92 |
| Standard Deviation | 92,131.28 |
| Minimum | 1,945.04 |
| 25th Percentile | 34,002.21 |
| Median | 53,481.84 |
| 75th Percentile | 98,915.99 |
| Maximum | 2,354,502.00 |

Interpretation: The average customer order value was approximately ₦81,538.92, while the median was ₦53,481.84. This indicates variation in transaction values, with some customers making considerably higher-value purchases than others.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Average%20Order%20Value.png)

**Purchasing Recency Analysis**

Business Question: How recently did each customer make a purchase?

Customer recency was calculated by determining the number of days between each customer's most recent completed transaction and the latest transaction date in the dataset.

Python Implementation:
```python
# Get the latest transaction date
snapshot_date = completed_df['transaction_date'].max()

# Calculate each customer's most recent purchase date
last_purchase = (
    completed_df.groupby('customer_id')['transaction_date']
    .max()
    .reset_index(name='last_purchase_date')
)

# Calculate recency in days
last_purchase['recency_days'] = (
    snapshot_date - last_purchase['last_purchase_date']
).dt.days
```

Results:
| Statistic | Recency (Days) |
|---|---:|
| Customers | 9,912 |
| Mean | 160.79 |
| Standard Deviation | 172.84 |
| Minimum | 0 |
| 25th Percentile | 46 |
| Median | 97 |
| 75th Percentile | 208 |
| Maximum | 1,173 |

Interpretation: The average purchase recency was approximately 161 days, while the median was 97 days. This indicates that customers differed considerably in how recently they had purchased. Some customers purchased on the latest transaction date, while others had much longer periods since their last recorded purchase.

Note: Recency is measured relative to the latest transaction date in the dataset, not the current date.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Purchase%20Recency.png)

**Product Preference Analysis**

Business Question: Which product categories do customers purchase most frequently?

The distribution of completed transactions across product categories was examined to identify the most frequently purchased categories and understand customer product preferences.

Python Implementation:
```python
# Calculate category purchase frequency
category_purchases = (
    completed_df.groupby('product_category')['transaction_id']
    .nunique()
    .reset_index(name='purchase_count')
    .sort_values('purchase_count', ascending=False)
)

# Calculate purchase percentages
category_purchases['purchase_percentage'] = (
    category_purchases['purchase_count']
    / category_purchases['purchase_count'].sum()
) * 100
```

Results:
| Product Category | Purchase Count | Percentage |
|---|---:|---:|
| Fashion | 13,713 | 22.02% |
| Groceries | 12,393 | 19.90% |
| Electronics | 11,203 | 17.99% |
| Home & Living | 10,034 | 16.11% |
| Beauty | 8,521 | 13.68% |
| Sports | 6,030 | 9.68% |
| Unknown | 377 | 0.61% |

Interpretation: Fashion recorded the highest number of completed purchases, accounting for approximately 22.02% of all purchases, followed by Groceries and Electronics. Sports recorded the lowest known product category purchase count. These findings indicate differences in customer demand across product categories.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Purchase%20Frequency%20by%20Product%20Category.png)

**Sales Channel Behaviour Analysis**

Business Question: Which sales channels do customers use most frequently to make purchases?

Completed transactions were grouped by sales channel to examine customer purchasing preferences across the available channels.

Python Implementation:
```python
# Examine sales channel distribution
channel_purchases = (
    completed_df['sales_channel']
    .value_counts()
    .reset_index()
)

channel_purchases.columns = [
    'sales_channel',
    'purchase_count'
]

# Calculate purchase percentages
channel_purchases['purchase_percentage'] = (
    channel_purchases['purchase_count']
    / channel_purchases['purchase_count'].sum()
) * 100
```

Results:
| Sales Channel | Purchase Count | Percentage |
|---|---:|---:|
| Mobile App | 21,158 | 33.98% |
| Website | 17,368 | 27.89% |
| In-store | 15,631 | 25.10% |
| Marketplace | 8,114 | 13.03% |

Interpretation: The Mobile App was the most frequently used sales channel, accounting for approximately 33.98% of completed purchases. The Website and In-store channels followed, while Marketplace recorded the lowest transaction share. This suggests that digital channels, particularly the Mobile App, play an important role in customer purchasing activities.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Purchase%20Distribution%20by%20Sales%20Channel.png)

**Marketing Behaviour Analysis**

Business Question: How do customers respond to the company's marketing campaigns?

The distribution of campaign responses among completed transactions was examined to understand the level of customer engagement with the company's marketing activities.

Python Implementation:
```python
# Examine campaign response distribution
campaign_response = (
    completed_df['campaign_response']
    .value_counts()
    .reset_index()
)

campaign_response.columns = [
    'campaign_response',
    'response_count'
]

# Calculate response percentages
campaign_response['response_percentage'] = (
    campaign_response['response_count']
    / campaign_response['response_count'].sum()
) * 100
```

Results:
| Campaign Response | Transaction Count | Percentage |
|---|---:|---:|
| Not Exposed | 29,920 | 48.05% |
| Ignored | 21,811 | 35.03% |
| Responded | 10,540 | 16.93% |

Interpretation: Approximately 16.93% of completed transactions were associated with customers who responded to marketing campaigns, while 35.03% were associated with ignored campaigns. The remaining 48.05% were classified as not exposed. This highlights differences in recorded campaign engagement across transactions.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Marketing%20Campaign%20Responses.png)

**Customer Demographics Analysis**

Business Question: What are the demographic characteristics of the company's customers?

The customer demographic dataset was constructed by retaining one record per unique customer. Age, annual income, membership tier, and geographical distribution were examined to understand the composition of the customer base.

Python Implementation:
```python
# Create customer-level demographic dataset
customer_demographics = (
    df.drop_duplicates(subset='customer_id').copy()
)

Age Distribution
The age distribution was examined using descriptive statistics, a histogram, and a boxplot\
# Descriptive statistics
customer_demographics['age'].describe()
```

Results:
| Statistic | Age |
|---|---:|
| Customers | 10,000 |
| Mean | 36.36 |
| Standard Deviation | 10.54 |
| Minimum | 11 |
| 25th Percentile | 29 |
| Median | 36 |
| 75th Percentile | 43 |
| Maximum | 78 |

Interpretation: The customer base had an average age of approximately 36 years, with a median of 36 years. Most customers were concentrated between the ages of 29 and 43, indicating a relatively young to middle-aged customer population.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Customer%20Age%20Distribution%20and%20Boxplot.png)

**Annual Income Distribution**

Annual income was examined to understand the income characteristics and variation across customers.

Python Implementation:
```python
# Descriptive statistics
customer_demographics['annual_income_ngn'].describe()
```

Results:
| Statistic | Annual Income (NGN) |
|---|---:|
| Customers | 10,000 |
| Mean | 3,947,983 |
| Standard Deviation | 34,564,000 |
| Minimum | 350,000 |
| 25th Percentile | 1,667,534 |
| Median | 2,378,022 |
| 75th Percentile | 3,393,782 |
| Maximum | 1,000,000,000 |

Interpretation: The median annual income was approximately ₦2.38 million, while the mean was approximately ₦3.95 million. The substantial difference between these values and the unusually high maximum indicate a highly skewed income distribution, suggesting the presence of extreme income observations requiring further investigation.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Customer%20Annual%20Income%20Distribution%20and%20Boxplot.png)

**Membership Tier Distribution**

Customer membership tiers were examined to understand how customers were distributed across the available membership categories.

Python Implementation:
```python
customer_demographics['membership_tier'].value_counts()
```

Results:
| Membership Tier | Customer Count |
|---|---:|
| Standard | 5,285 |
| Silver | 2,534 |
| Gold | 1,600 |
| Platinum | 556 |
| gold | 25 |

Interpretation: The Standard membership tier accounted for the largest proportion of customers, followed by Silver and Gold. Platinum had the lowest count. The separate gold category also revealed an inconsistency in categorical formatting that should be standardised before further categorical analysis.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Customer%20Distribution%20by%20Membership%20Tier.png)

**Regional Distribution**

The geographical distribution of customers was examined using the region variable.

Python Implementation:
```python
customer_demographics['region'].value_counts()
```

Results:
| Region | Customer Count |
|---|---:|
| South | 2,791 |
| West | 2,386 |
| North | 1,778 |
| East | 1,551 |
| Central | 1,494 |

Interpretation: The South had the highest customer representation, followed by the West and North. The Central region recorded the lowest customer count. This provides an overview of the geographical distribution of the customer base.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Customer%20Distribution%20by%20Region.png)

**Summary of Exploratory Data Analysis**

The exploratory analysis revealed several important patterns in customer demographics and purchasing behaviour:

1.	Purchasing Behaviour: Customers made an average of 6.28 completed purchases, with purchase frequency ranging from 1 to 25 transactions.
2.	Spending Behaviour: Average total customer spending was approximately ₦471,104.70, with considerable variation across customers.
3.	Average Order Value: The average order value was approximately ₦81,538.92.
4.	Purchase Recency: Customers had an average recency of approximately 161 days, indicating differences in how recently they made purchases.
5.	Product Preferences: Fashion was the most frequently purchased category, followed by Groceries and Electronics.
6.	Sales Channels: The Mobile App accounted for the highest transaction share at approximately 33.98%.
7.	Marketing Engagement: Only 16.93% of completed transactions were associated with a recorded campaign response.
8.	Customer Demographics: The average customer age was 36 years, while median annual income was approximately ₦2.38 million. The South had the highest customer representation.

**Overall Conclusion**

The analysis established that customers exhibit different purchasing frequencies, spending levels, product preferences, and engagement patterns. These differences provide useful behavioural indicators for developing customer-level features and identifying meaningful customer segments through machine learning.

The findings also highlighted data quality considerations, particularly the extreme annual income values and inconsistent membership tier labels, which require attention during subsequent feature preparation.

## CUSTOMER-LEVEL FEATURE ENGINEERING

This session documents the creation of customer-level variables from the cleaned transaction and customer datasets. The purpose was to transform individual transaction records into measurable customer characteristics that could be used for segmentation.

The features cover return behaviour, customer tenure, product and channel diversity, discount usage, marketing engagement, profitability, purchase frequency, spending consistency, recency, demographics, and historical customer value.

Important methodological distinction: completed_df was used for features based on successful purchases, while the original cleaned df was used for return-related features so that returned transactions were not excluded.

**Feature Engineering Overview
Feature engineering involved grouping records by customer_id, calculating customer-level metrics, and transforming selected numerical variables into behavioural or demographic categories.
The resulting feature tables were created separately and would subsequently be combined into a single customer-level dataset for modelling.

**Customer Return Behaviour**

Business question: What proportion of each customer's transactions are returned?

Return rate measures the proportion of a customer's total transactions that were returned. The original cleaned df was used because completed_df excludes returned transactions.

Python code:
```python
customer_returns = (
    df.groupby('customer_id')
    .agg(
        total_transactions=('transaction_id', 'nunique'),
        returned_transactions=(
            'order_status',
            lambda x: (x == 'Returned').sum()
        )
    )
    .reset_index()
)

customer_returns['return_rate'] = (
    customer_returns['returned_transactions']
    / customer_returns['total_transactions']
)

customer_returns['return_rate'].describe()
```

Results:
| Statistic | Return Rate |
|---|---:|
| Count | 10,000 |
| Mean | 0.0512 |
| Standard deviation | 0.1264 |
| Minimum | 0.0000 |
| 25th percentile | 0.0000 |
| Median | 0.0000 |
| 75th percentile | 0.0667 |
| Maximum | 1.0000 |

Interpretation: The average customer return rate was approximately 5.12%. At least half of the customers had a return rate of zero, while the maximum was 100%. This indicates that return behaviour varied across customers and could help distinguish customers with higher return tendencies.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Return%20Rates.png)

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Top%2010%20Customers%20by%20Return%20Rate.png)

Python code for visualisations:
```python
# Return rate distribution
plt.figure(figsize=(10, 5))

sns.histplot(
    data=customer_returns,
    x='return_rate',
    bins=30,
    kde=True
)

plt.title('Distribution of Customer Return Rates')
plt.xlabel('Return Rate')
plt.ylabel('Number of Customers')
plt.show()


# Customers with the highest return rates
top_return_customers = (
    customer_returns
    .sort_values('return_rate', ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=top_return_customers,
    x='customer_id',
    y='return_rate'
)

plt.title('Top 10 Customers by Return Rate')
plt.xlabel('Customer ID')
plt.ylabel('Return Rate')
plt.xticks(rotation=45)
plt.show()
```

**Customer Tenure**

Customer tenure measures how long a customer has maintained a relationship with the business, based on their signup date.

The latest transaction date in the completed transaction dataset was used as the reference date.

Python code:
```python
# Reference date
snapshot_date = completed_df['transaction_date'].max()

# Get one signup date per customer
customer_tenure = (
    df.groupby('customer_id')['signup_date']
    .min()
    .reset_index(name='signup_date')
)

# Calculate customer tenure
customer_tenure['tenure_days'] = (
    snapshot_date - customer_tenure['signup_date']
).dt.days

customer_tenure['tenure_days'].describe()
```

Results:
| Statistic | Tenure (days) |
|---|---:|
| Count | 10,000 |
| Mean | 711.23 |
| Standard deviation | 395.52 |
| Minimum | 29 |
| 25th percentile | 370 |
| Median | 714 |
| 75th percentile | 1,053 |
| Maximum | 1,398 |

Interpretation: Customers had an average tenure of approximately 711 days, or about 1.95 years. The wide range indicates differences in how long customers had maintained their relationship with the business.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Tenure.png)

```python
plt.figure(figsize=(10, 5))

sns.histplot(
    data=customer_tenure,
    x='tenure_days',
    bins=30,
    kde=True
)

plt.title('Distribution of Customer Tenure')
plt.xlabel('Customer Tenure (Days)')
plt.ylabel('Number of Customers')
plt.show()
```

**Customer Purchase Diversity**

This feature measures the number of unique product categories purchased by each customer. Only completed transactions were considered.

Python code:
```python
customer_diversity = (
    completed_df.groupby('customer_id')['product_category']
    .nunique()
    .reset_index(name='category_diversity')
)

customer_diversity['category_diversity'].describe()

customer_diversity['category_diversity'].value_counts().sort_index()
```

Results:
| Statistic | Category Diversity |
|---|---:|
| Count | 9,912 |
| Mean | 3.42 |
| Standard deviation | 1.72 |
| Minimum | 1 |
| 25th percentile | 2 |
| Median | 4 |
| 75th percentile | 5 |
| Maximum | 7 |

| Unique Categories Purchased | Customers |
|---|---:|
| 1 | 2,005 |
| 2 | 1,419 |
| 3 | 1,502 |
| 4 | 1,689 |
| 5 | 2,010 |
| 6 | 1,220 |
| 7 | 67 |

Interpretation: Customers purchased from an average of approximately 3.42 product categories. Some customers concentrated their purchases in one category, while others purchased across several categories. This feature can help distinguish specialised buyers from customers with broader product interests.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Purchase%20Diversity.png)

```python
plt.figure(figsize=(9, 5))

sns.countplot(
    data=customer_diversity,
    x='category_diversity'
)

plt.title('Distribution of Customer Purchase Diversity')
plt.xlabel('Number of Unique Product Categories Purchased')
plt.ylabel('Number of Customers')
plt.show()
```

**Customer Discount Behaviour**

This feature measures the average discount received by each customer across completed transactions. It helps identify customers who tend to purchase at full price and those who purchase with relatively higher discounts.

Python code:
```python
customer_discount = (
    completed_df.groupby('customer_id')['discount_pct']
    .mean()
    .reset_index(name='average_discount_pct')
)

customer_discount['average_discount_pct'].describe()
```

Results:
| Statistic | Average Discount |
|---|---:|
| Count | 9,912 |
| Mean | 0.1578 |
| Standard deviation | 0.0657 |
| Minimum | 0.0004 |
| 25th percentile | 0.1204 |
| Median | 0.1526 |
| 75th percentile | 0.1880 |
| Maximum | 0.5500 |

Interpretation: The average customer discount was approximately 15.78%, with a median of 15.26%. The variation suggests that customers differed in their exposure to or use of discounts.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Average%20Customer%20Discount.png)

```python
plt.figure(figsize=(10, 5))

sns.histplot(
    data=customer_discount,
    x='average_discount_pct',
    bins=30,
    kde=True
)

plt.title('Distribution of Average Customer Discount')
plt.xlabel('Average Discount Percentage')
plt.ylabel('Number of Customers')
plt.show()
```

**Customer Marketing Engagement**

This feature measures how actively customers respond to marketing campaigns.

The response rate was calculated as:
```
Campaign Response Rate} = Successful Responses/Total Exposed Campaigns
```
An important distinction was maintained: Not Exposed was not treated as Ignored. Customers who were not exposed to a campaign could not be classified as unresponsive to it.

Initial campaign response distribution:
```python
completed_df['campaign_response'].value_counts(dropna=False)
```

| Campaign Response | Transactions |
|---|---:|
| Not Exposed | 29,920 |
| Ignored | 21,811 |
| Responded | 10,540 |

Python code for customer response rate:
```python
customer_marketing = (
    completed_df.groupby('customer_id')
    .agg(
        total_campaigns=(
            'campaign_response',
            lambda x: x.isin(['Responded', 'Ignored']).sum()
        ),
        successful_responses=(
            'campaign_response',
            lambda x: (x == 'Responded').sum()
        )
    )
    .reset_index()
)

customer_marketing['campaign_response_rate'] = (
    customer_marketing['successful_responses']
    / customer_marketing['total_campaigns'].replace(0, np.nan)
)

customer_marketing['campaign_response_rate'].describe()
```

Results:
| Statistic | Campaign Response Rate |
|---|---:|
| Count | 8,686 |
| Mean | 0.3229 |
| Standard deviation | 0.3141 |
| Minimum | 0.0000 |
| 25th percentile | 0.0000 |
| Median | 0.2857 |
| 75th percentile | 0.5000 |
| Maximum | 1.0000 |

Customers without campaign exposure:
```python
print(
    'Customers without campaign exposure:',
    customer_marketing['total_campaigns'].eq(0).sum()
)
```

Output:

Customers without campaign exposure: 1226

Interpretation: Among customers with campaign exposure, the average response rate was approximately 32.29%. A total of 1,226 customers had no exposed campaign records and therefore had undefined response rates. These customers should not be interpreted as having a zero response rate.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Campaign%20Response%20Rates.png)

```python
plt.figure(figsize=(10, 5))

sns.histplot(
    data=customer_marketing,
    x='campaign_response_rate',
    bins=30,
    kde=True
)

plt.title('Distribution of Customer Campaign Response Rates')
plt.xlabel('Campaign Response Rate')
plt.ylabel('Number of Customers')
plt.show()
```

**Customer Profitability**

This feature measures the total profit generated by each customer from completed transactions

Customer profit was calculated as: 
```
Customer Profit} = Total Net Revenue - Total Cost
```

Python code:
```python
customer_profit = (
    completed_df.groupby('customer_id')
    .agg(
        total_revenue=('net_amount_ngn', 'sum'),
        total_cost=('cost_amount_ngn', 'sum')
    )
    .reset_index()
)

customer_profit['total_profit_ngn'] = (
    customer_profit['total_revenue'] -
    customer_profit['total_cost']
)

customer_profit['total_profit_ngn'].describe()
```

Results:
| Statistic | Total Profit (₦) |
|---|---:|
| Count | 9,912 |
| Mean | 120,102.90 |
| Standard deviation | 124,662.90 |
| Minimum | -273,461.70 |
| 25th percentile | 32,965.35 |
| Median | 86,267.17 |
| 75th percentile | 161,601.40 |
| Maximum | 1,140,839.00 |

Interpretation: The average customer generated approximately ₦120,102.90 in profit. However, the minimum value was negative, indicating that some customers' total costs exceeded their net revenue. Profitability therefore provides another useful dimension for distinguishing customer segments.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Profitability.png)

```python
plt.figure(figsize=(10, 5))

sns.histplot(
    data=customer_profit,
    x='total_profit_ngn',
    bins=40,
    kde=True
)

plt.title('Distribution of Customer Profitability')
plt.xlabel('Total Customer Profit (NGN)')
plt.ylabel('Number of Customers')
plt.show()
```

**Customer Purchase Channel Diversity**

This feature measures the number of distinct sales channels each customer uses to make purchases.

Python code:
```python
customer_channel = (
    completed_df.groupby('customer_id')['sales_channel']
    .nunique()
    .reset_index(name='channel_diversity')
)

customer_channel['channel_diversity'].describe()

customer_channel['channel_diversity'].value_counts().sort_index()
```

Results:
| Statistic | Channel Diversity |
|---|---:|
| Count | 9,912 |
| Mean | 2.69 |
| Standard deviation | 1.14 |
| Minimum | 1 |
| 25th percentile | 2 |
| Median | 3 |
| 75th percentile | 4 |
| Maximum | 4 |

| Number of Unique Channels | Customers |
|---|---:|
| 1 | 2,168 |
| 2 | 1,895 |
| 3 | 2,681 |
| 4 | 3,168 |

Interpretation: Customers used an average of approximately 2.69 different sales channels. A substantial number of customers used all four available channels, suggesting that many customers interacted with the business through multiple purchasing platforms.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Channel%20Diversity.png)

```python
plt.figure(figsize=(9, 5))

sns.countplot(
    data=customer_channel,
    x='channel_diversity'
)

plt.title('Distribution of Customer Channel Diversity')
plt.xlabel('Number of Unique Sales Channels Used')
plt.ylabel('Number of Customers')
plt.show()
```

**Monthly Customer Purchase Frequency**

This feature measures the average number of completed purchases a customer makes per active month.

Unlike total purchase frequency, it accounts for the period between a customer's first and last observed purchases. This helps distinguish customers who make purchases frequently within a shorter period from those whose purchases are spread over a longer period.

Python code:
```python
customer_monthly_frequency = (
    completed_df.groupby('customer_id')
    .agg(
        total_purchases=('transaction_id', 'nunique'),
        first_purchase=('transaction_date', 'min'),
        last_purchase=('transaction_date', 'max')
    )
    .reset_index()
)

customer_monthly_frequency['active_months'] = (
    (customer_monthly_frequency['last_purchase'].dt.year -
     customer_monthly_frequency['first_purchase'].dt.year) * 12
    +
    (customer_monthly_frequency['last_purchase'].dt.month -
     customer_monthly_frequency['first_purchase'].dt.month)
    + 1
)

customer_monthly_frequency['monthly_purchase_frequency'] = (
    customer_monthly_frequency['total_purchases'] /
    customer_monthly_frequency['active_months']
)

customer_monthly_frequency[
    'monthly_purchase_frequency'
].describe()
```

Results:
| Statistic | Monthly Purchase Frequency |
|---|---:|
| Count | 9,912 |
| Mean | 0.7069 |
| Standard deviation | 0.4593 |
| Minimum | 0.0690 |
| 25th percentile | 0.3571 |
| Median | 0.5833 |
| 75th percentile | 1.0000 |
| Maximum | 5.0000 |

Interpretation: Customers made an average of approximately 0.71 purchases per active month. The median was approximately 0.58 purchases per month, indicating differences in purchasing frequency across customers.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Monthly%20Customer%20Purchase%20Frequency.png)

```python
plt.figure(figsize=(10, 5))

sns.histplot(
    data=customer_monthly_frequency,
    x='monthly_purchase_frequency',
    bins=30,
    kde=True
)

plt.title('Distribution of Monthly Customer Purchase Frequency')
plt.xlabel('Average Purchases per Month')
plt.ylabel('Number of Customers')
plt.show()
```

**Customer Spending Consistency**

This feature measures how consistent each customer's spending is across completed transactions. The standard deviation of transaction-level net amounts was used.

A lower standard deviation indicates relatively consistent transaction values, while a higher value indicates greater variation in spending.

Customers with only one completed transaction have no measurable spending variation, so their standard deviation is missing.

Python code:
```python
customer_spending_consistency = (
    completed_df.groupby('customer_id')['net_amount_ngn']
    .std()
    .reset_index(name='spending_std_ngn')
)

customer_spending_consistency['spending_std_ngn'].describe()
```

Results:
| Statistic | Spending Standard Deviation (₦) |
|---|---:|
| Count | 8,130 |
| Mean | 78,397.32 |
| Standard deviation | 87,600.57 |
| Minimum | 94.88 |
| 25th percentile | 28,630.68 |
| Median | 50,584.19 |
| 75th percentile | 92,926.50 |
| Maximum | 1,173,783.00 |

Interpretation: The standard deviation varied considerably across customers, indicating differences in transaction spending consistency. The count of 8,130 also shows that this feature was unavailable for customers without sufficient transaction observations to calculate a standard deviation.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Spending%20Variability.png)

```python
plt.figure(figsize=(10, 5))

sns.histplot(
    data=customer_spending_consistency,
    x='spending_std_ngn',
    bins=30,
    kde=True
)

plt.title('Distribution of Customer Spending Variability')
plt.xlabel('Standard Deviation of Spending (NGN)')
plt.ylabel('Number of Customers')
plt.show()
```

**Average Profit per Transaction**

This feature measures the average profit generated per completed transaction for each customer.

It provides a transaction-level profitability measure rather than relying only on total customer profit.

Python code:
```python
customer_avg_profit = (
    completed_df.groupby('customer_id')
    .agg(
        total_profit_ngn=(
            'net_amount_ngn',
            lambda x: x.sum(min_count=1)
        ),
        total_cost_ngn=(
            'cost_amount_ngn',
            lambda x: x.sum(min_count=1)
        ),
        transaction_count=('transaction_id', 'nunique')
    )
    .reset_index()
)

customer_avg_profit['average_profit_per_transaction'] = (
    (customer_avg_profit['total_profit_ngn'] -
     customer_avg_profit['total_cost_ngn'])
    / customer_avg_profit['transaction_count']
)

customer_avg_profit[
    'average_profit_per_transaction'
].describe()
```

Results:
| Statistic | Average Profit per Transaction (₦) |
|---|---:|
| Count | 9,912 |
| Mean | 20,753.65 |
| Standard deviation | 22,434.90 |
| Minimum | -70,335.46 |
| 25th percentile | 9,315.84 |
| Median | 13,823.18 |
| 75th percentile | 25,680.09 |
| Maximum | 706,389.79 |

Interpretation: The average profit per completed transaction was approximately ₦20,753.65. Negative values indicate customers whose average transaction cost exceeded their net revenue. The difference between the mean and median also indicates variation in transaction-level profitability.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Average%20Profit%20per%20Transaction.png)

```python
plt.figure(figsize=(10, 5))

sns.histplot(
    data=customer_avg_profit,
    x='average_profit_per_transaction',
    bins=30,
    kde=True
)

plt.title('Distribution of Average Profit per Transaction')
plt.xlabel('Average Profit per Transaction (NGN)')
plt.ylabel('Number of Customers')
plt.show()
```

**Customer Return Frequency**

While return rate measures the proportion of transactions returned, return frequency measures the actual number of returned transactions for each customer.

The original cleaned df was used because returned transactions are excluded from completed_df.

Python code:
```python
customer_return_frequency = (
    df.groupby('customer_id')
    .agg(
        total_returned_transactions=(
            'order_status',
            lambda x: (x == 'Returned').sum()
        )
    )
    .reset_index()
)

customer_return_frequency[
    'total_returned_transactions'
].describe()

customer_return_frequency[
    'total_returned_transactions'
].value_counts().sort_index()
```

Results:
| Statistic | Returned Transactions |
|---|---:|
| Count | 10,000 |
| Mean | 0.3368 |
| Standard deviation | 0.6140 |
| Minimum | 0 |
| 25th percentile | 0 |
| Median | 0 |
| 75th percentile | 1 |
| Maximum | 5 |

| Returned Transactions | Customers |
|---|---:|
| 0 | 7,284 |
| 1 | 2,164 |
| 2 | 467 |
| 3 | 71 |
| 4 | 13 |
| 5 | 1 |

Interpretation: Customers had an average of approximately 0.34 returned transactions. Most customers (7,284) had no returned transactions, while only a small number recorded multiple returns.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Return%20Frequency.png)\

```python
plt.figure(figsize=(10, 5))

sns.countplot(
    data=customer_return_frequency,
    x='total_returned_transactions'
)

plt.title('Distribution of Customer Return Frequency')
plt.xlabel('Number of Returned Transactions')
plt.ylabel('Number of Customers')
plt.show()
```

**Customer Profit Margin**

This feature measures the percentage of a customer's net revenue retained as profit.

It was calculated using the previously created customer profitability table.
```
Profit Margin} = Total Profit/Total Revenue
```

Python code:
```python
customer_profit['profit_margin'] = (
    customer_profit['total_profit_ngn']
    / customer_profit['total_revenue']
)

customer_profit['profit_margin'].describe()
```

Results:
| Statistic | Profit Margin |
|---|---:|
| Count | 9,912 |
| Mean | 0.2715 |
| Standard deviation | 0.0934 |
| Minimum | -1.6401 |
| 25th percentile | 0.2123 |
| Median | 0.2720 |
| 75th percentile | 0.3249 |
| Maximum | 0.7658 |

Interpretation: The average customer profit margin was approximately 27.15%. The negative minimum indicates that some customers generated losses relative to their net revenue. This feature provides an additional measure of customer-level financial contribution.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Distribution%20of%20Customer%20Profit%20Margins.png)

```python
plt.figure(figsize=(10, 5))

sns.histplot(
    data=customer_profit,
    x='profit_margin',
    bins=30,
    kde=True
)

plt.title('Distribution of Customer Profit Margins')
plt.xlabel('Customer Profit Margin')
plt.ylabel('Number of Customers')
plt.show()
```

**Customer Purchase Recency Category**

Customer recency measures the number of days since a customer's last completed purchase. This feature was transformed from a continuous numerical variable into five behavioural categories using binning.

The categories were defined as follows:
| Recency Range | Category |
|---|---|
| 0–30 days | Very Recent |
| 31–90 days | Recent |
| 91–180 days | Moderate |
| 181–365 days | At Risk |
| More than 365 days | Inactive |

Python code:
```python
snapshot_date = completed_df['transaction_date'].max()

customer_recency = (
    completed_df.groupby('customer_id')['transaction_date']
    .max()
    .reset_index(name='last_purchase_date')
)

customer_recency['recency_days'] = (
    snapshot_date - customer_recency['last_purchase_date']
).dt.days

customer_recency['recency_category'] = pd.cut(
    customer_recency['recency_days'],
    bins=[-1, 30, 90, 180, 365, float('inf')],
    labels=[
        'Very Recent',
        'Recent',
        'Moderate',
        'At Risk',
        'Inactive'
    ]
)

customer_recency['recency_category'].value_counts().sort_index()
```

Results:
| Recency Category | Customers |
|---|---:|
| Very Recent | 1,395 |
| Recent | 3,324 |
| Moderate | 2,294 |
| At Risk | 1,766 |
| Inactive | 1,133 |
| Total | 9,912 |

Interpretation: The largest group consisted of customers classified as Recent, with 3,324 customers. A further 1,766 customers were classified as At Risk, while 1,133 were Inactive. These categories provide a practical way to identify customers who may require retention or re-engagement campaigns.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Customer%20Distribution%20by%20Purchase%20Recency.png)

```python
plt.figure(figsize=(10, 5))

sns.countplot(
    data=customer_recency,
    x='recency_category',
    order=[
        'Very Recent',
        'Recent',
        'Moderate',
        'At Risk',
        'Inactive'
    ]
)

plt.title('Customer Distribution by Purchase Recency')
plt.xlabel('Recency Category')
plt.ylabel('Number of Customers')
plt.xticks(rotation=30)
plt.show()
```

**Customer Age Group**

Customer age was transformed into demographic categories to make age-related customer patterns easier to interpret.

Python code:
```python
customer_demographics['age_group'] = pd.cut(
    customer_demographics['age'],
    bins=[-float('inf'), 24, 34, 44, 54, float('inf')],
    labels=[
        'Young',
        'Young Adult',
        'Adult',
        'Middle-aged',
        'Senior'
    ]
)

customer_demographics['age_group'].value_counts().sort_index()

customer_demographics['age_group'].value_counts(
    normalize=True
).mul(100).round(2)
```

Results:
| Age Group | Customers | Percentage |
|---|---:|---:|
| Young | 1,472 | 14.72% |
| Young Adult | 2,830 | 28.30% |
| Adult | 3,468 | 34.68% |
| Middle-aged | 1,747 | 17.47% |
| Senior | 483 | 4.83% |
| Total | 10,000 | 100% |

Interpretation: Adults aged 35–44 formed the largest demographic group, representing 34.68% of customers. Young Adults accounted for another 28.30%, while Seniors represented the smallest group at 4.83%. The age-group feature provides a categorical demographic variable for customer profiling.

![](https://github.com/Oluwaseun2024-ctrl/Customer-Segmentation-App/blob/main/Customer%20Distribution%20by%20Age%20Group.png)

```python
plt.figure(figsize=(10, 5))

sns.countplot(
    data=customer_demographics,
    x='age_group',
    order=[
        'Young',
        'Young Adult',
        'Adult',
        'Middle-aged',
        'Senior'
    ]
)

plt.title('Customer Distribution by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Customers')
plt.xticks(rotation=30)
plt.show()
```

**Customer Lifetime Value (CLV) Proxy**

This feature estimates the total revenue generated by each customer over their observed purchasing history. Since it uses historical transaction revenue rather than a predictive lifetime value model, it is treated as a historical customer value or CLV proxy.

Only completed transactions were included.

Python code:
```python
customer_lifetime_value = (
    completed_df.groupby('customer_id')['net_amount_ngn']
    .sum(min_count=1)
    .reset_index(name='historical_customer_value_ngn')
)

customer_lifetime_value[
    'historical_customer_value_ngn'
].describe()
```

Results:
