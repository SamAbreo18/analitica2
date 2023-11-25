from flask import Flask, request, jsonify
import joblib  # o la biblioteca que utilizaste para cargar el modelo

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('modelo_entrenado.pkl')

@app.route('/predict', methods=['POST'])
def predecir():
    try:
        # Obtener datos de entrada desde la solicitud POST en formato JSON
        data = request.json

        # Realizar la lógica de predicción utilizando el modelo
        # En este ejemplo, supongamos que el modelo toma 'year' y 'month' como entradas
        year = data['year']
        month = data['month']
        prediccion = modelo.predict([[year, month]])

        # Devolver la predicción como respuesta en formato JSON
        return jsonify({'prediccion': prediccion[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
