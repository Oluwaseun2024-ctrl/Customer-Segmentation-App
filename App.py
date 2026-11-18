
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import os

# =====================================================
# 1. STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# 2. LOAD SAVED MODEL ARTIFACTS
# =====================================================

ARTIFACT_PATH = "customer_segmentation_model.pkl"

if not os.path.exists(ARTIFACT_PATH):
    st.error("Model artifact not found. Check the file location.")
    st.stop()

artifacts = joblib.load(ARTIFACT_PATH)

scaler = artifacts["scaler"]
pca = artifacts["pca"]
kmeans = artifacts["kmeans"]

income_median = artifacts["income_median"]
selected_features = artifacts["selected_features"]
log_features = artifacts["log_features"]
continuous_features = artifacts["continuous_features"]
binary_features = artifacts["binary_features"]
feature_order = artifacts["feature_order"]


# =====================================================
# 3. CUSTOMER SEGMENT DESCRIPTIONS
# =====================================================

segment_details = {
    0: {
        "name": "Established, High-Order-Value Customers",
        "strategy": "Customer Reactivation",
        "recommendation": (
            "Use personalised reactivation campaigns, "
            "complementary product recommendations, "
            "and targeted repeat-purchase incentives."
        )
    },

    1: {
        "name": "High-Value, Loyal and Engaged Customers",
        "strategy": "Customer Retention",
        "recommendation": (
            "Prioritise loyalty rewards, personalised offers, "
            "cross-selling, and upselling opportunities."
        )
    },

    2: {
        "name": "Low-Frequency, Low-Spending Customers",
        "strategy": "Customer Development",
        "recommendation": (
            "Encourage repeat purchases through onboarding, "
            "product discovery, and first-repeat-purchase incentives."
        )
    }
}


# =====================================================
# 4. SINGLE CUSTOMER PREDICTION FUNCTION
# =====================================================

def predict_customer(customer_data):

    customer_df = pd.DataFrame([customer_data])

    # Derive campaign exposure
    customer_df["campaign_exposed"] = (
        customer_df["total_campaigns"] > 0
    ).astype(int)

    # No campaign exposure means zero response rate
    if customer_df["total_campaigns"].iloc[0] == 0:
        customer_df["campaign_response_rate"] = 0

    # Apply log transformations
    for col in log_features:

        if customer_df[col].isna().any():
            continue

        if (customer_df[col] < 0).any():
            raise ValueError(
                f"{col} cannot contain negative values."
            )

        customer_df[col] = np.log1p(customer_df[col])

    # Impute income using the saved training median
    customer_df["annual_income_ngn"] = (
        customer_df["annual_income_ngn"].fillna(income_median)
    )

    # Scale continuous features
    scaled_continuous = pd.DataFrame(
        scaler.transform(
            customer_df[continuous_features]
        ),
        columns=continuous_features
    )

    # Preserve binary features without scaling
    scaled_data = pd.concat(
        [
            scaled_continuous,
            customer_df[binary_features].reset_index(drop=True)
        ],
        axis=1
    )

    scaled_data = scaled_data[feature_order]

    # PCA transformation
    customer_pca = pca.transform(scaled_data)

    # K-Means prediction
    predicted_cluster = kmeans.predict(customer_pca)

    return int(predicted_cluster[0])


# =====================================================
# 5. BULK CUSTOMER PREDICTION FUNCTION
# =====================================================

def predict_bulk_customers(bulk_data):

    data = bulk_data.copy()

    # Identify customers without completed purchase history
    insufficient_history = data["purchase_frequency"] <= 0

    eligible_data = data.loc[
        ~insufficient_history
    ].copy()

    # Stop if no eligible customers exist
    if eligible_data.empty:
        raise ValueError(
            "No customers have sufficient purchase history."
        )

    # Derive campaign exposure
    eligible_data["campaign_exposed"] = (
        eligible_data["total_campaigns"] > 0
    ).astype(int)

    # Customers without campaign exposure have zero response rate
    eligible_data.loc[
        eligible_data["campaign_exposed"] == 0,
        "campaign_response_rate"
    ] = 0

    # Select the original model features
    model_data = eligible_data[selected_features].copy()

    # Apply the same log transformations used during training
    for col in log_features:
        model_data[col] = np.log1p(model_data[col])

    # Impute missing transformed income
    model_data["annual_income_ngn"] = (
        model_data["annual_income_ngn"].fillna(income_median)
    )

    # Apply the fitted scaler
    scaled_continuous = pd.DataFrame(
        scaler.transform(
            model_data[continuous_features]
        ),
        columns=continuous_features,
        index=model_data.index
    )

    # Preserve binary features
    scaled_data = pd.concat(
        [
            scaled_continuous,
            eligible_data[binary_features]
        ],
        axis=1
    )

    # Restore training feature order
    scaled_data = scaled_data[feature_order]

    # PCA transformation
    customer_pca = pca.transform(scaled_data)

    # Predict clusters
    predictions = kmeans.predict(customer_pca)

    # Store predictions
    data["predicted_cluster"] = pd.Series(
        pd.NA,
        index=data.index,
        dtype="Int64"
    )

    data.loc[
        eligible_data.index,
        "predicted_cluster"
    ] = predictions

    # Attach segment descriptions
    data["customer_segment"] = data["predicted_cluster"].map(
        {
            cluster: details["name"]
            for cluster, details in segment_details.items()
        }
    )

    data["marketing_strategy"] = data["predicted_cluster"].map(
        {
            cluster: details["strategy"]
            for cluster, details in segment_details.items()
        }
    )

    data["recommendation"] = data["predicted_cluster"].map(
        {
            cluster: details["recommendation"]
            for cluster, details in segment_details.items()
        }
    )

    # Prediction status
    data["prediction_status"] = np.where(
        insufficient_history,
        "Insufficient purchase history",
        "Successfully segmented"
    )

    return data


# =====================================================
# 6. APPLICATION HEADING
# =====================================================

st.title("Customer Segmentation and Behavioural Analytics")

st.write(
    "Identify customer behavioural segments and generate "
    "targeted marketing recommendations using Machine Learning."
)


# =====================================================
# 7. SINGLE CUSTOMER PREDICTION INTERFACE
# =====================================================

st.divider()

st.header("Single Customer Prediction")

st.write(
    "Enter the customer's demographic, purchasing, "
    "and marketing information below."
)

with st.form("customer_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35
        )

        annual_income_ngn = st.number_input(
            "Annual Income (₦)",
            min_value=0.0,
            value=4500000.0
        )

        purchase_frequency = st.number_input(
            "Purchase Frequency",
            min_value=1,
            value=8
        )

        total_spending_ngn = st.number_input(
            "Total Spending (₦)",
            min_value=0.0,
            value=600000.0
        )

        average_order_value_ngn = st.number_input(
            "Average Order Value (₦)",
            min_value=0.0,
            value=75000.0
        )

    with col2:

        recency_days = st.number_input(
            "Recency (Days)",
            min_value=0,
            value=60
        )

        return_rate = st.number_input(
            "Return Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.05
        )

        tenure_days = st.number_input(
            "Customer Tenure (Days)",
            min_value=0,
            value=700
        )

        category_diversity = st.number_input(
            "Category Diversity",
            min_value=1,
            value=4
        )

        average_discount_pct = st.number_input(
            "Average Discount",
            min_value=0.0,
            max_value=1.0,
            value=0.15
        )

    with col3:

        total_campaigns = st.number_input(
            "Total Campaigns",
            min_value=0,
            value=4
        )

        campaign_response_rate = st.number_input(
            "Campaign Response Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.35
        )

        profit_margin = st.number_input(
            "Profit Margin",
            min_value=0.0,
            max_value=1.0,
            value=0.26
        )

        channel_diversity = st.number_input(
            "Channel Diversity",
            min_value=1,
            value=3
        )

        monthly_purchase_frequency = st.number_input(
            "Monthly Purchase Frequency",
            min_value=0.0,
            value=0.7
        )

    submitted = st.form_submit_button(
        "Identify Customer Segment",
        use_container_width=True
    )


# =====================================================
# 8. SINGLE CUSTOMER PREDICTION RESULT
# =====================================================

if submitted:

    customer_data = {
        "age": age,
        "annual_income_ngn": annual_income_ngn,
        "purchase_frequency": purchase_frequency,
        "total_spending_ngn": total_spending_ngn,
        "average_order_value_ngn": average_order_value_ngn,
        "recency_days": recency_days,
        "return_rate": return_rate,
        "tenure_days": tenure_days,
        "category_diversity": category_diversity,
        "average_discount_pct": average_discount_pct,
        "total_campaigns": total_campaigns,
        "campaign_response_rate": campaign_response_rate,
        "profit_margin": profit_margin,
        "channel_diversity": channel_diversity,
        "monthly_purchase_frequency": monthly_purchase_frequency
    }

    try:

        predicted_cluster = predict_customer(customer_data)

        segment = segment_details[predicted_cluster]

        st.divider()

        st.subheader("Customer Segmentation Result")

        st.success(
            f"Identified Segment: {segment['name']}"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Predicted Cluster",
                f"Cluster {predicted_cluster}"
            )

        with result_col2:

            st.metric(
                "Recommended Strategy",
                segment["strategy"]
            )

        st.subheader("Business Recommendation")

        st.write(segment["recommendation"])

    except Exception as e:

        st.error(f"Prediction error: {e}")


# =====================================================
# 9. BULK CUSTOMER SEGMENTATION
# =====================================================

st.divider()

st.header("Bulk Customer Segmentation")

st.write(
    "Upload a CSV file containing multiple customer records "
    "to identify their behavioural segments simultaneously."
)


# =====================================================
# 10. DOWNLOAD CSV TEMPLATE
# =====================================================

st.subheader("Step 1: Download CSV Template")

bulk_columns = [
    "customer_id",
    "age",
    "annual_income_ngn",
    "purchase_frequency",
    "total_spending_ngn",
    "average_order_value_ngn",
    "recency_days",
    "return_rate",
    "tenure_days",
    "category_diversity",
    "average_discount_pct",
    "total_campaigns",
    "campaign_response_rate",
    "profit_margin",
    "channel_diversity",
    "monthly_purchase_frequency"
]

template_data = pd.DataFrame([
    {
        "customer_id": "C001",
        "age": 35,
        "annual_income_ngn": 4500000,
        "purchase_frequency": 8,
        "total_spending_ngn": 600000,
        "average_order_value_ngn": 75000,
        "recency_days": 60,
        "return_rate": 0.05,
        "tenure_days": 700,
        "category_diversity": 4,
        "average_discount_pct": 0.15,
        "total_campaigns": 4,
        "campaign_response_rate": 0.35,
        "profit_margin": 0.26,
        "channel_diversity": 3,
        "monthly_purchase_frequency": 0.7
    }
])

template_csv = template_data[bulk_columns].to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Customer CSV Template",
    data=template_csv,
    file_name="customer_segmentation_template.csv",
    mime="text/csv",
    use_container_width=True
)

st.info(
    "Download the template, replace the example record "
    "with your customer records, and upload the completed CSV."
)


# =====================================================
# 11. UPLOAD CUSTOMER CSV
# =====================================================

st.subheader("Step 2: Upload Customer Data")

uploaded_file = st.file_uploader(
    "Choose a customer CSV file",
    type=["csv"],
    help="Upload the completed customer segmentation template."
)


if uploaded_file is not None:

    try:

        bulk_df = pd.read_csv(uploaded_file)

        st.success("CSV file uploaded successfully!")

        if bulk_df.empty:

            st.error("The uploaded CSV file contains no customer records.")

        else:

            st.subheader("Uploaded Data Preview")

            st.dataframe(
                bulk_df.head(10),
                use_container_width=True
            )

            st.write(
                "Total customer records:",
                len(bulk_df)
            )

            # ==========================================
            # 12. REQUIRED COLUMN VALIDATION
            # ==========================================

            missing_columns = [
                col for col in bulk_columns
                if col not in bulk_df.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing required columns: {missing_columns}"
                )

            else:

                st.success("All required columns are present.")

                # ==========================================
                # 13. DATA TYPE VALIDATION
                # ==========================================

                st.subheader("Data Quality Checks")

                numeric_features = selected_features

                for col in numeric_features:

                    bulk_df[col] = pd.to_numeric(
                        bulk_df[col],
                        errors="coerce"
                    )

                # Allow missing income because the model
                # has a saved training-median imputation value.
                # Allow missing campaign response rate only
                # when the customer received zero campaigns.

                if "annual_income_ngn" in bulk_df.columns:

                    pass

                no_campaigns = bulk_df["total_campaigns"].eq(0)

                bulk_df.loc[
                    no_campaigns & bulk_df["campaign_response_rate"].isna(),
                    "campaign_response_rate"
                ] = 0

                # Identify missing required numerical values
                required_numeric = [
                    col for col in numeric_features
                    if col != "annual_income_ngn"
                ]

                invalid_values = (
                    bulk_df[required_numeric].isna().sum()
                )

                if invalid_values.sum() > 0:

                    st.error(
                        "Missing or invalid numerical values detected."
                    )

                    st.dataframe(
                        invalid_values[invalid_values > 0]
                        .rename("Invalid Count")
                    )

                else:

                    st.success(
                        "Required numerical values are valid."
                    )

                # ==========================================
                # 14. NON-NEGATIVE VALUE VALIDATION
                # ==========================================

                non_negative_features = [
                    "annual_income_ngn",
                    "purchase_frequency",
                    "total_spending_ngn",
                    "average_order_value_ngn",
                    "recency_days",
                    "tenure_days",
                    "category_diversity",
                    "average_discount_pct",
                    "total_campaigns",
                    "campaign_response_rate",
                    "profit_margin",
                    "channel_diversity",
                    "monthly_purchase_frequency"
                ]

                negative_counts = (
                    (bulk_df[non_negative_features] < 0).sum()
                )

                if negative_counts.sum() > 0:

                    st.error(
                        "Negative values detected in features "
                        "that must be non-negative."
                    )

                    st.write(
                        negative_counts[negative_counts > 0]
                    )

                else:

                    st.success(
                        "Non-negative feature checks passed."
                    )

                # ==========================================
                # 15. RATE VALIDATION
                # ==========================================

                rate_features = [
                    "return_rate",
                    "average_discount_pct",
                    "campaign_response_rate",
                    "profit_margin"
                ]

                invalid_rates = (
                    (bulk_df[rate_features] < 0) |
                    (bulk_df[rate_features] > 1)
                ).sum()

                if invalid_rates.sum() > 0:

                    st.error(
                        "Rate values must be between 0 and 1."
                    )

                    st.write(
                        invalid_rates[invalid_rates > 0]
                    )

                else:

                    st.success("Rate validation passed.")

                # ==========================================
                # 16. AGE VALIDATION
                # ==========================================

                invalid_age = (
                    (bulk_df["age"] < 18) |
                    (bulk_df["age"] > 100)
                ).sum()

                if invalid_age > 0:

                    st.error(
                        f"{invalid_age} customer records have "
                        "an invalid age. Age must be between 18 and 100."
                    )

                else:

                    st.success("Age validation passed.")

                # ==========================================
                # 17. DUPLICATE CUSTOMER ID CHECK
                # ==========================================

                duplicate_ids = (
                    bulk_df["customer_id"].duplicated().sum()
                )

                if duplicate_ids > 0:

                    st.warning(
                        f"{duplicate_ids} duplicate customer IDs detected."
                    )

                else:

                    st.success("No duplicate customer IDs detected.")

                # ==========================================
                # 18. RUN BULK PREDICTIONS
                # ==========================================

                validation_passed = (
                    invalid_values.sum() == 0
                    and negative_counts.sum() == 0
                    and invalid_rates.sum() == 0
                    and invalid_age == 0
                    and duplicate_ids == 0
                )

                if validation_passed:

                    st.divider()

                    st.subheader("Step 3: Identify Customer Segments")

                    if st.button(
                        "Run Bulk Customer Segmentation",
                        use_container_width=True
                    ):

                        try:

                            results_df = predict_bulk_customers(
                                bulk_df
                            )

                            st.success(
                                "Customer segmentation completed successfully!"
                            )

                            # ==========================================
                            # 19. DISPLAY SEGMENTATION RESULTS
                            # ==========================================

                            st.subheader("Segmentation Results")

                            st.dataframe(
                                results_df,
                                use_container_width=True
                            )

                            # ==========================================
                            # 20. BUSINESS SUMMARY
                            # ==========================================

                            st.subheader("Customer Segment Summary")

                            successful_results = results_df[
                                results_df["predicted_cluster"].notna()
                            ]

                            segment_counts = (
                                successful_results["customer_segment"]
                                .value_counts()
                            )

                            metric_col1, metric_col2, metric_col3 = (
                                st.columns(3)
                            )

                            metric_col1.metric(
                                "Total Customers",
                                len(results_df)
                            )

                            metric_col2.metric(
                                "Successfully Segmented",
                                len(successful_results)
                            )

                            metric_col3.metric(
                                "Insufficient History",
                                results_df["predicted_cluster"].isna().sum()
                            )

                            st.bar_chart(segment_counts)

                            # ==========================================
                            # 21. DOWNLOAD RESULTS
                            # ==========================================

                            st.subheader("Step 4: Download Results")

                            results_csv = results_df.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                label="Download Segmentation Results",
                                data=results_csv,
                                file_name="customer_segmentation_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        except Exception as e:

                            st.error(
                                f"Bulk prediction error: {e}"
                            )

                else:

                    st.warning(
                        "Please correct the data quality issues "
                        "before running customer segmentation."
                    )

    except Exception as e:

        st.error(
            f"Unable to process the uploaded CSV file: {e}"
        )


# =====================================================
# 22. APPLICATION FOOTER
# =====================================================

st.divider()

st.caption(
    "Customer Segmentation and Behavioural Analytics | "
    "Powered by PCA and K-Means Clustering"
)