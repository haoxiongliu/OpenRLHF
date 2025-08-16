import fire

def push_local_model_to_hub(model_path: str, hub_path: str):
    """
    Push a local model to the hub.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(folder_path=model_path, repo_id=hub_path)


def lookup_dashboard():
    import ray
    ray.init(address="auto")  # 或者 head 节点地址

    # 集群资源
    print("Cluster resources:", ray.cluster_resources())

    # 所有 actor
    print("Actors:")
    for actor in ray.util.list_named_actors():
        print(actor)

if __name__ == "__main__":
    fire.Fire()