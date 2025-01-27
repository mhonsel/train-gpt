import io

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from huggingface_hub import snapshot_download

@contextmanager
def suppress_output(silent=True):
    """Context manager to suppress stdout and stderr if silent is True."""
    if silent:
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            yield
    else:
        yield

def download(dataset_repo, dataset_dir):
    snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        local_dir=dataset_dir,
        # allow_patterns=["*000000.npy", "*000001.npy", "*000002.npy"]  # DEBUG
    )

def upload(api, file_path, model_repo, silent=False):
    with suppress_output(silent=silent):
        future = api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path.name,
            repo_id=model_repo,
            repo_type="model",
            run_as_future=True,  # run in background
        )
    return future

if __name__ == "__main__":
    repo_id = "mhonsel/edu_fineweb10B_tokens"
    local_dir = "./edu_fineweb10B/"
    download(repo_id, local_dir)