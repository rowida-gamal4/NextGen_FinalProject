import numpy as np

def create_features(data):

    data = data.copy()

    data["hour_of_day"] = data["step"] % 24

    data["day_number"] = data["step"] // 24

    data["log_amount"] = np.log1p(
        data["amount"].clip(lower=0)
    )

    data["amount_to_origin_balance"] = (
        data["amount"] / (data["oldbalanceOrg"] + 1)
    )

    data["amount_to_destination_balance"] = (
        data["amount"] / (data["oldbalanceDest"] + 1)
    )

    data["origin_balance_insufficient"] = (
        data["amount"] > data["oldbalanceOrg"]
    ).astype(int)

    return data