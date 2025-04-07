from huggingface_hub import HfApi
# upload a file to the hub
api = HfApi()
api.upload_file(
    path_or_fileobj="logs/summary.log",
    repo_id="Vivacem/results",
    path_in_repo="summary.log",
    token="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",
    repo_type="dataset",
)