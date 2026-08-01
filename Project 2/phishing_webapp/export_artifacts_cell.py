import os
import joblib

ARTIFACT_DIR = "model_artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "best_model.joblib"))
joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
joblib.dump(tfidf, os.path.join(ARTIFACT_DIR, "tfidf.joblib"))

with open(os.path.join(ARTIFACT_DIR, "model_info.txt"), "w") as f:
    f.write(f"Best model: {best_model_name}\n")
    f.write(f"Total feature count: {X_final.shape[1]}\n")
    f.write(f"Metadata features: 6\n")
    f.write(f"TF-IDF features: {len(tfidf.get_feature_names_out())}\n")
    f.write(f"Embedding dims: {X_embeddings.shape[1]}\n")

print(f"Artifacts saved to ./{ARTIFACT_DIR}/")
print("Copy this folder into the Flask app directory before running app.py")
