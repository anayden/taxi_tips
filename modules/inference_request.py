import numpy as np
from pydantic import BaseModel


class InferenceRequest(BaseModel):
    trip_distance: float
    payment_type: float
    fare_amount: float
    total_amount: float
    pickup_month: float
    pickup_hour: float
    pickup_day_of_week: float

    def to_nparray(self) -> np.array:
        return np.array([self.trip_distance, self.payment_type, self.fare_amount, self.total_amount,
                         self.pickup_month, self.pickup_hour, self.pickup_day_of_week])
