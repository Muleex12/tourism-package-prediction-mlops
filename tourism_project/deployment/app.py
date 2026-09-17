
import streamlit as st
import pandas as pd

from huggingface_hub import hf_hub_download
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🏨",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND PREPROCESSOR FROM HUGGING FACE MODEL HUB
# ============================================================

MODEL_REPO = "Muleex12/tourism-package-prediction-model"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="final_random_forest_model.pkl"
)

preprocessor_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="preprocessor.pkl"
)

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🏨 Tourism Package Purchase Prediction")

st.write(
    "Enter the customer's information below to predict "
    "whether the customer is likely to purchase a tourism package."
)

st.divider()


# ============================================================
# CUSTOMER INPUTS
# ============================================================

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Column 1
# ------------------------------------------------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    type_of_contact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch",
        min_value=0,
        value=10
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )


# ------------------------------------------------------------
# Column 2
# ------------------------------------------------------------

with col2:

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        value=2
    )

    number_of_followups = st.number_input(
        "Number of Followups",
        min_value=0,
        value=2
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )

    preferred_property_star = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0,
        value=3
    )


# ------------------------------------------------------------
# Column 3
# ------------------------------------------------------------

with col3:

    passport = st.selectbox(
        "Passport",
        [0, 1]
    )

    pitch_satisfaction_score = st.number_input(
        "Pitch Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    own_car = st.selectbox(
        "Own Car",
        [0, 1]
    )

    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        value=0
    )

    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=25000
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Purchase",
    use_container_width=True
):

    # Create dataframe with EXACTLY the same
    # 18 feature names used during model training.

    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])


    try:

        # Apply the same preprocessing used during training
        processed_data = preprocessor.transform(
            input_data
        )

        # Generate prediction
        prediction = model.predict(
            processed_data
        )[0]


        # Display result
        if prediction == 1:

            st.success(
                "✅ The customer is likely to purchase "
                "the tourism package."
            )

        else:

            st.warning(
                "❌ The customer is unlikely to purchase "
                "the tourism package."
            )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )
