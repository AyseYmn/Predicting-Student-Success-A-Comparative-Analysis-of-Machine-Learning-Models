
from fastapi import FastAPI
from pydantic import BaseModel
import joblib  # Modeli yüklemek için

class QueryData(BaseModel):
    # Model giriş şemasını tanımlayın
    features: list  # Modelin özellik listesi

app = FastAPI()

# trained model
model = joblib.load(r"student\models\linear_model_mat.pkl")

@app.post("/predict")
def predict(query: QueryData):
    try:
        data = query.dict().values()
        prediction = model.predict([list(data)])
        return {"prediction": prediction.tolist()}
    except Exception as e:
        print(f"Hata: {str(e)}")
        return {"error": str(e)}

# @app.post("/predict")
# def predict(query: QueryData):
#     data = [
#         query.sex, query.age, query.address, query.Pstatus, query.Fedu,
#         query.traveltime, query.studytime, query.failures, query.schoolsup,
#         query.famsup, query.paid, query.activities, query.nursery, query.higher,
#         query.internet, query.romantic, query.famrel, query.goout, query.Dalc,
#         query.Walc, query.health, query.absences, query.G1, query.G2,
#         query.school_MS, query.Mjob_health, query.Mjob_other, query.Mjob_services,
#         query.Mjob_teacher, query.Fjob_health, query.Fjob_other, query.Fjob_teacher,
#         query.reason_home, query.reason_other, query.reason_reputation,
#         query.guardian_mother, query.guardian_other, query.Avg_G, query.Change_G
#     ]
#     prediction = model.predict([data])
#     return {"prediction": prediction.tolist()}



# find model feature names
# if hasattr(model, 'feature_names_in_'):
#     print(model.feature_names_in_)
# else:
#     print("Model özellik isimlerini saklamıyor.")