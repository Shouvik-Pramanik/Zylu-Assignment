from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open('churn_model.pkl', 'rb') as f:
    churn_model = pickle.load(f)

with open('recommendation_model.pkl', 'rb') as f:
    recommendation_model = pickle.load(f)

with open('repurchase_model.pkl', 'rb') as f:
    repurchase_model = pickle.load(f)
X_preprocessed = preprocessor.transform(df[features])

# API endpoint to predict churn
@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    data = request.json
    X_new = pd.DataFrame(data, index=[0])
    X_new_preprocessed = preprocessor.transform(X_new)
    proba = churn_model.predict_proba(X_new_preprocessed)[:, 1]
    return jsonify({'churn_probability': proba[0]})

# API endpoint to recommend similar customers
@app.route('/recommend_customers', methods=['POST'])
def recommend_customers():
    customer_id = int(request.json['Customer_ID'])
    customer_index = df[df['Customer_ID'] == customer_id].index[0]
    distances, indices = recommendation_model.kneighbors(X_preprocessed[customer_index].reshape(1, -1), n_neighbors=6)
    recommendations = df['Customer_ID'].iloc[indices.flatten()[1:]].tolist()
    return jsonify({'recommended_customers': recommendations})

# API endpoint to predict re-purchase likelihood
@app.route('/predict_repurchase', methods=['POST'])
def predict_repurchase():
    data = request.json
    X_new = pd.DataFrame(data, index=[0])
    X_new_preprocessed = preprocessor.transform(X_new)
    proba = repurchase_model.predict_proba(X_new_preprocessed)[:, 1]
    return jsonify({'repurchase_probability': proba[0]})

if __name__ == '__main__':
    app.run(debug=True)
