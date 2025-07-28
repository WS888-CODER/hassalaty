from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. إنشاء التطبيق
app = FastAPI()

# 2. تحميل المودل
model = joblib.load("child_behavior_model.pkl")  # تأكد من رفع هذا الملف لـ Render
top_category_encoder = joblib.load("top_category_encoder.pkl")  # إذا عندك إيمبودنق لاسم الفئة

# 3. نموذج الطلب (اللي يجيك من الـ Frontend)
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

# 4. نقطة التنبؤ
@app.post("/predict")
def predict(input: ChildInput):
    # تحويل بيانات JSON إلى DataFrame
    data = input.dict()
    df = pd.DataFrame([data])

    # تحويل الفئة النصية إلى رقمية إذا لزم الأمر
    if "top_spending_category" in df.columns:
        df["top_spending_category"] = top_category_encoder.transform(df["top_spending_category"])

    # تنبؤ
    prediction = model.predict(df)[0]

    return {"predicted_label": prediction}
