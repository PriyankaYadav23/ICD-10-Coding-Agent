# upload_model.py
# One-time script: upload the fine-tuned BioBERT classifier to the Hugging Face Hub,
# so the deployed app can download it (the model is too big for GitHub).
# Run:  python upload_model.py   (after logging in, see instructions)

from huggingface_hub import create_repo, upload_folder

REPO_ID = "PriyankaYadav777/biobert-icd10-classifier"
LOCAL_FOLDER = "saved_model/biobert_icd_classifier_weighted"

create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)

upload_folder(
    repo_id=REPO_ID,
    folder_path=LOCAL_FOLDER,
    repo_type="model",
    commit_message="Upload BioBERT ICD-10 classifier (class-weighted loss, 232 codes)",
)

print("Done! Model page: https://huggingface.co/" + REPO_ID)
