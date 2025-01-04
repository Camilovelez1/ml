from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import requests
import json
import os

load_dotenv()

# Inicializa la app de Flask
app = Flask(__name__)

endpoint_url = os.getenv("end_point")
api_token =os.getenv("api_token")

print(endpoint_url)

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para obtener la predicción
@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    # Obtener los datos del formulario
    try:
        monthly_income = float(request.form['monthly_income'])
        age = int(request.form['age'])
        employment_years = int(request.form['employment_years'])
        loan_amount = float(request.form['loan_amount'])
        credit_score = int(request.form['credit_score'])
        fecha_corte_year = int(request.form['fecha_corte_year'])
        fecha_corte_month = int(request.form['fecha_corte_month'])
        fecha_corte_day = int(request.form['fecha_corte_day'])
        fecha_pago_year = int(request.form['fecha_pago_year'])
        fecha_pago_month = int(request.form['fecha_pago_month'])
        fecha_pago_day = int(request.form['fecha_pago_day'])
    except ValueError:
        return jsonify({"error": "Por favor ingrese valores válidos para todos los campos."})

    # Datos para enviar al modelo
    data = {
        "instances": [
            {
                "monthly_income": float(monthly_income),
                "age": int(age),
                "employment_years": int(employment_years),
                "loan_amount": float(loan_amount),
                "credit_score": int(credit_score),
                "fecha_corte_year": int(fecha_corte_year),
                "fecha_corte_month": int(fecha_corte_month),
                "fecha_corte_day": int(fecha_corte_day),
                "fecha_pago_year": int(fecha_pago_year),
                "fecha_pago_month": int(fecha_pago_month),
                "fecha_pago_day": int(fecha_pago_day)
            }
        ]
    }


    # Encabezados para la autenticación y el tipo de contenido
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    #print("Datos a enviar:", json.dumps(data, indent=4))

    try:
        response = requests.post(endpoint_url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Lanza una excepción para respuestas con códigos 4xx/5xx
    except requests.exceptions.HTTPError as errh:
        return jsonify({"error": f"Error HTTP: {errh}"}), 400
    except requests.exceptions.RequestException as err:
        return jsonify({"error": f"Error en la solicitud: {err}"}), 500

    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        result = response.json()
        prediccion = result.get("predictions", [])
        if prediccion:
            # Interpretación basada en el valor 0 o 1
            prediction_value = prediccion[0]  # 0 o 1
            if prediction_value == 1:
                prediction_interpretation = "El pago se realizará a tiempo."
            else:
                prediction_interpretation = "El pago no se realizará a tiempo."
            
            # Devuelve tanto la predicción como su interpretación
            return jsonify({
                "prediction": prediction_value,
                "interpretation": prediction_interpretation
            })
        else:
            return jsonify({"error": "No se pudo obtener la predicción."}), 500
    else:
        return jsonify({"error": f"Error en la solicitud al modelo. Código: {response.status_code}"}), response.status_code

if __name__ == '__main__':
    app.run(debug=True)
