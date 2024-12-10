from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("D:\\PROJECT ML\\Ardent final project\\model\\mental_health_model.pkl")

@app.route("/")
def home():
    # Render the home page template
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the form
    Age = float(request.form["Age"])
    Gender = request.form["Gender"]
    self_employed = request.form["self_employed"]
    family_history = request.form["family_history"]
    work_interfere = request.form["work_interfere"]
    remote_work = request.form["remote_work"]
    tech_company = request.form["tech_company"]
    benefits = request.form["benefits"]
    care_options = request.form["care_options"]
    wellness_program = request.form["wellness_program"]
    seek_help = request.form["seek_help"]
    leave = request.form["leave"]
    mental_health_consequence = request.form["mental_health_consequence"]
    phys_health_consequence = request.form["phys_health_consequence"]
    coworkers = request.form["coworkers"]
    supervisor = request.form["supervisor"]
    mental_vs_physical = request.form["mental_vs_physical"]
    obs_consequence = request.form["obs_consequence"]
    
    # Encode Gender
    Gender_encoded = {
        "Male": 0,
        "Female": 1,
        "Others": 2
    }[Gender]
    
    # Encoded Self Employed
    self_employed_encoded={
        "No":0,
        "Yes":1
    }[self_employed]
    
    # Encode family_history
    family_history_encoded = 1 if family_history == "Yes" else 0

    # Encode work_interfere
    work_interfere_encoded = {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3
    }[work_interfere]

    # Encode remote_work
    remote_work_encoded = {
        "Yes": 1,
        "No": 0
    }[remote_work]

    # Encode tech_company
    tech_company_encoded = {
        "Yes": 1,
        "No": 0
    }[tech_company]

    # Encode benefits
    benefits_encoded = {
        "Yes": 1,
        "No": 0,
        "Don't know": 2
    }[benefits]

    # Encode care_options
    care_options_encoded = {
        "Yes": 1,
        "No": 0,
        "Not sure": 2
    }[care_options]

    # Encode wellness_program
    wellness_program_encoded = {
        "Yes": 1,
        "No": 0,
        "Don't know": 2
    }[wellness_program]

    # Encode seek_help
    seek_help_encoded = {
        "Yes": 1,
        "No": 0,
        "Don't know": 2
    }[seek_help]

    # Encode leave
    leave_encoded = {
        "Very easy": 0,
        "Somewhat easy": 1,
        "Somewhat difficult": 2,
        "Very difficult": 4,
        "Don't know": 5
    }[leave]

    # Encode mental_health_consequence
    mental_health_consequence_encoded = {
        "Yes": 1,
        "No": 0,
        "Maybe": 2
    }[mental_health_consequence]

    # Encode phys_health_consequence
    phys_health_consequence_encoded = {
        "Yes": 1,
        "No": 0,
        "Maybe": 2
    }[phys_health_consequence]

    # Encode coworkers
    coworkers_encoded = {
        "Yes": 1,
        "No": 0,
        "Some of them": 2
    }[coworkers]

    # Encode supervisor
    supervisor_encoded = {
        "Yes": 1,
        "No": 0,
        "Some of them": 2
    }[supervisor]

    # Encode mental_vs_physical
    mental_vs_physical_encoded = {
        "Yes": 1,
        "No": 0,
        "Don't know": 2
    }[mental_vs_physical]

    # Encode obs_consequence
    obs_consequence_encoded = {
        "Yes": 1,
        "No": 0
    }[obs_consequence]

    # Combine input features into a single array
    features = np.array([[Age, Gender_encoded, self_employed_encoded, family_history_encoded, work_interfere_encoded, remote_work_encoded, tech_company_encoded, benefits_encoded, care_options_encoded, wellness_program_encoded, seek_help_encoded, leave_encoded, mental_health_consequence_encoded, phys_health_consequence_encoded, coworkers_encoded, supervisor_encoded, mental_vs_physical_encoded, obs_consequence_encoded]])

    # Make prediction using the loaded model
    prediction = model.predict(features)

    # Return the prediction result to the user
    return render_template("result.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
