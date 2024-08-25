from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipelines.predict_pipeline import CustomData, PredictPipeline
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        film_code = int(request.form['film_code'])
        cinema_code = int(request.form['cinema_code'])
        show_time = int(request.form['show_time'])
        occu_perc = float(request.form['occu_perc'])
        ticket_price = float(request.form['ticket_price'])
        capacity = float(request.form['capacity'])
        date_str = request.form['date']
        day, month, year = map(int, date_str.split('-'))

        data = CustomData(film_code, cinema_code, show_time, occu_perc, ticket_price, capacity, day, month, year)
        df = data.get_data_as_df()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)
        return render_template('home.html', prediction_text='Predicted Ticket Sales: {}'.format(int(prediction[0])))
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)