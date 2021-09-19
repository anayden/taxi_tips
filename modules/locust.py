import json
import pandas as pd

from locust import HttpUser, task


data = pd.read_csv('/project/results/test.csv', usecols=["trip_distance", "payment_type",
                   "fare_amount", "tip_amount", "total_amount", "pickup_month", "pickup_hour", "pickup_day_of_week"])

# Predictions within EPSILON are deemed accurate for the test
EPSILON = 0.8


class LoadGenerator(HttpUser):
    @task
    def prediction(self) -> None:
        random_row = data.sample()
        expected_tip = float(random_row["tip_amount"])
        random_row = random_row.drop(columns=["tip_amount"])
        request_json = json.loads(random_row.to_json(
            index=False, orient='table'))['data'][0]

        with self.client.post("", json=request_json, catch_response=True) as response:
            predicted_tip = json.loads(response.text)["tip"]
            if abs(expected_tip - predicted_tip) > EPSILON:
                response.failure(
                    f"Incorrect prediction expected {expected_tip}, got {predicted_tip}."
                )