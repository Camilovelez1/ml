from flask import Flask, render_template, request, jsonify
import requests
import json

# Inicializar la app de Flask
app = Flask(__name__)

endpoint_url = dbutils.secrets.get(scope = "machine_learning_v2", key = "end_point")

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
        "columns": ["monthly_income", "age", "employment_years", "loan_amount", "credit_score", 
                    "fecha_corte_year", "fecha_corte_month", "fecha_corte_day", 
                    "fecha_pago_year", "fecha_pago_month", "fecha_pago_day"],
        "data": [
            [monthly_income, age, employment_years, loan_amount, credit_score, 
             fecha_corte_year, fecha_corte_month, fecha_corte_day, 
             fecha_pago_year, fecha_pago_month, fecha_pago_day]
        ]
    }

    # Encabezados para la autenticación y el tipo de contenido
    headers = {
        "Authorization": api_token,
        "Content-Type": "application/json"
    }

    # Realizar la solicitud POST al endpoint
    response = requests.post(endpoint_url, headers=headers, data=json.dumps(data))

    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        result = response.json()
        prediccion = result.get("predictions", [])
        if prediccion:
            return jsonify({"prediction": prediccion[0]})
        else:
            return jsonify({"error": "No se pudo obtener la predicción."})
    else:
        return jsonify({"error": "Error en la solicitud al modelo. Código: " + str(response.status_code)})

if __name__ == '__main__':
    app.run(debug=True)
