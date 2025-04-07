from huggingface_hub import HfApi
# upload a file to the hub
def submit_log(token):
    api = HfApi()
    api.upload_file(
        path_or_fileobj="logs/summary.log",
        repo_id="Vivacem/results",
        path_in_repo="summary.log",
        token=token,
        repo_type="dataset",
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, help="Hugging Face token")
    args = parser.parse_args()
    submit_log(args.token)
