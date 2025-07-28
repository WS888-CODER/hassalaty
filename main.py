from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. إنشاء التطبيق
app = FastAPI()

# 2. تحميل المودلز
model = joblib.load("child_behavior_model.pkl")  # ← اسم المودل الجديد
top_category_encoder = joblib.load("top_category_encoder.pkl")  # موجود حسب GitHub

# 3. نموذج البيانات من الواجهة الأمامية
class ChildInput(BaseModel):
    spend_rate: float
    top_spending_category: str
    save_rate: float
    tasks_failed: int
    tasks_completed: int
    impulsive_purchases: int
    max_purchase_amount: float
    saving_goal_success_rate: float
    refused_purchase_opportunities: int
    do_parent_challenges: int
    num_items_sold_in_marketplace: int

# 4. API Endpoint للتنبؤ
@app.post("/predict")
def predict(input: ChildInput):
    # تحويل JSON إلى DataFrame
    data = input.dict()
    df = pd.DataFrame([data])

    # تشفير الفئة النصية
    df["top_spending_category"] = top_category_encoder.transform(df["top_spending_category"])

    # تنبؤ
    prediction = model.predict(df)[0]

    return {"predicted_label": prediction}