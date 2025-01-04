document.getElementById('creditForm').addEventListener('submit', function(event) {
  event.preventDefault();  // Evitar la recarga de la página

  // Recoger los datos del formulario
  const formData = new FormData(this);
  const formObject = {};
  formData.forEach((value, key) => formObject[key] = value);

  // Realizar la solicitud al servidor
  fetch('/get_prediction', {
      method: 'POST',
      body: new URLSearchParams(formObject)
  })
  .then(response => response.json())
  .then(data => {
      const resultDiv = document.getElementById('result');
      const prediction = document.getElementById('prediction');
      const errorMessage = document.getElementById('error_message');
      const interpretation = document.getElementById('interpretation');  // Elemento para mostrar la interpretación
      
      if (data.error) {
          prediction.textContent = "";
          errorMessage.textContent = data.error;
          interpretation.textContent = "";  // Limpiar la interpretación en caso de error
          resultDiv.style.display = 'block';
      } else {
          prediction.textContent = `Predicción: ${data.prediction}`;
          errorMessage.textContent = "";
          
          // Interpretación de la predicción
          if (data.prediction === 1) {
              interpretation.textContent = "El cliente es probable que pague a tiempo.";
          } else if (data.prediction === 0) {
              interpretation.textContent = "El cliente es probable que no pague a tiempo.";
          } else {
              interpretation.textContent = "Resultado de la predicción no reconocido.";
          }
          
          resultDiv.style.display = 'block';
      }
  })
  .catch(error => {
      console.error('Error:', error);
  });
});
