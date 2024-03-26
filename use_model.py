import numpy as np
import fickling

local_classifier = fickling.load(open("classifier.pickle", "rb"))
local_scaler = fickling.load(open("sc.pickle", "rb"))

new_pred = local_classifier.predict(local_scaler.transform(np.array([[40, 20000]])))

# age：40
# salary：20000
new_pred_proba = local_classifier.predict_proba(local_scaler.transform(np.array([[40, 20000]])))[:, 1]

print(new_pred_proba)
