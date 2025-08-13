"""
Utility functions for handling resume functionality in Krill.
"""
import os
import json
import tempfile
from typing import Optional, Union
from transformers.trainer_utils import get_last_checkpoint


def determine_resume_checkpoint(resume_option: str, output_dir: str, hub_model_id: str) -> Optional[Union[str, bool]]:
    """
    Determine the checkpoint to resume from based on the resume option.

    Args:
        resume_option: Resume option (auto, local, remote, false, true)
        output_dir: Local output directory where checkpoints are stored
        hub_model_id: Hugging Face Hub model ID for remote checkpoints

    Returns:
        None if no checkpoint should be used (start from scratch)
        Boolean True to let Hugging Face find the last local checkpoint automatically
        String path to local checkpoint directory (downloaded from remote if needed)
    """
    resume_option = resume_option.lower()

    if resume_option == "false":
        return None
    elif resume_option == "true":
        # Let Hugging Face automatically find the last checkpoint in output_dir
        return True
    elif resume_option == "auto":
        # Smart resume: check both local and remote, use the most recent
        return _handle_auto_resume(output_dir, hub_model_id)
    elif resume_option == "local":
        # Validate local checkpoint exists
        _validate_local_checkpoint(output_dir)
        return True  # Let Hugging Face handle the actual resuming
    elif resume_option == "remote":
        # Validate remote checkpoint exists and get step info
        _validate_remote_checkpoint(hub_model_id)
        remote_step = _get_remote_checkpoint_step(hub_model_id)
        if remote_step is not None:
            if remote_step > 0:
                print(f"🔄 Remote resume: Found remote checkpoint (step {remote_step})")
            else:
                print("🔄 Remote resume: Found remote checkpoint")
        # Download the remote checkpoint and return local path
        return _download_remote_checkpoint_to_temp(hub_model_id)
    else:
        raise ValueError(
            f"Invalid resume option: {resume_option}. "
            "Valid options are: auto, local, remote, false, true"
        )


def _get_local_checkpoint_step(output_dir: str) -> Optional[int]:
    """Get the global step of the last local checkpoint."""
    try:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            trainer_state_path = os.path.join(
                last_checkpoint, "trainer_state.json")
            if os.path.exists(trainer_state_path):
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
                    return trainer_state.get('global_step', 0)
        return None
    except Exception:
        return None


def _get_remote_checkpoint_step(hub_model_id: str) -> Optional[int]:
    """Get the global step of the remote last-checkpoint."""
    try:
        from huggingface_hub import hf_hub_download, HfApi
        api = HfApi()
        # Check if the last-checkpoint directory exists in the main branch
        repo_tree = list(api.list_repo_tree(
            repo_id=hub_model_id, revision="main"))
        # Check if any item in the tree is a directory named "last-checkpoint"
        # Folders have a 'tree_id' attribute, files have a 'blob_id' attribute
        last_checkpoint_exists = any(
            hasattr(item, 'tree_id') and item.path == "last-checkpoint"
            for item in repo_tree
        )
        if not last_checkpoint_exists:
            return None

        # Try to download the trainer_state.json to get the global step
        try:
            trainer_state_file = hf_hub_download(
                repo_id=hub_model_id,
                filename="last-checkpoint/trainer_state.json",
                revision="main"
            )
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
                return trainer_state.get('global_step', 0)
        except Exception:
            # If we can't get the trainer state, we still know the checkpoint exists
            return 0  # Return 0 to indicate it exists but we can't determine the step
    except Exception:
        return None


def _validate_local_checkpoint(output_dir: str) -> None:
    """Validate that a local checkpoint exists."""
    try:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None:
            print(f"🔄 Found local checkpoint: {last_checkpoint}")
        else:
            raise FileNotFoundError(
                f"No valid checkpoint found in output directory ({output_dir})."
            )
    except Exception as e:
        raise FileNotFoundError(
            f"Error accessing output directory ({output_dir}) for checkpoint detection: {str(e)}."
        )


def _validate_remote_checkpoint(hub_model_id: str) -> None:
    """Validate that a remote checkpoint exists."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Check if the last-checkpoint directory exists in the main branch
        # We list files in the main branch and look for the last-checkpoint directory
        repo_tree = list(api.list_repo_tree(
            repo_id=hub_model_id, revision="main"))
        # Check if any item in the tree is a directory named "last-checkpoint"
        # Folders have a 'tree_id' attribute, files have a 'blob_id' attribute
        last_checkpoint_exists = any(
            hasattr(item, 'tree_id') and item.path == "last-checkpoint"
            for item in repo_tree
        )
        if not last_checkpoint_exists:
            raise FileNotFoundError(
                f"'last-checkpoint' directory not found in remote model {hub_model_id}."
            )
        print(
            f"🔄 Found remote checkpoint: {hub_model_id} (last-checkpoint directory)")
    except Exception as e:
        # Handle various error cases:
        # - Repository doesn't exist
        # - last-checkpoint directory doesn't exist
        # - Network issues
        raise FileNotFoundError(
            f"Error accessing remote checkpoint 'last-checkpoint' for model {hub_model_id}: {str(e)}."
        )


def _download_remote_checkpoint_to_temp(hub_model_id: str) -> str:
    """Download the remote checkpoint to a temporary directory and return its path."""
    try:
        from huggingface_hub import snapshot_download
        import tempfile
        import os
        
        # Create a temporary directory to store the downloaded checkpoint
        temp_dir = tempfile.mkdtemp(prefix="krill_remote_checkpoint_")
        checkpoint_dir = os.path.join(temp_dir, "last-checkpoint")
        
        # Download the last-checkpoint directory from the hub
        snapshot_download(
            repo_id=hub_model_id,
            revision="main",
            allow_patterns="last-checkpoint/*",
            local_dir=temp_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"🔄 Downloaded remote checkpoint to: {checkpoint_dir}")
        return checkpoint_dir
    except Exception as e:
        raise FileNotFoundError(
            f"Error downloading remote checkpoint for model {hub_model_id}: {str(e)}."
        )


def _handle_auto_resume(output_dir: str, hub_model_id: str) -> Optional[Union[str, bool]]:
    """Handle auto resume option - smart detection with most recent checkpoint selection."""
    local_step = _get_local_checkpoint_step(output_dir)
    remote_step = _get_remote_checkpoint_step(hub_model_id)

    # Decision logic:
    # 1. If both exist, use the one with the higher step number
    # 2. If only one exists, use that one
    # 3. If neither exists, start from scratch

    if local_step is not None and remote_step is not None:
        # Both exist, compare steps
        if remote_step > local_step:
            print(
                f"🔄 Auto-resume: Found more recent remote checkpoint (step {remote_step} > {local_step})")
            # Download the remote checkpoint
            return _download_remote_checkpoint_to_temp(hub_model_id)
        else:
            print(
                f"🔄 Auto-resume: Found more recent local checkpoint (step {local_step} >= {remote_step})")
            return True  # Let Hugging Face handle the actual resuming
    elif local_step is not None:
        # Only local exists
        print(f"🔄 Auto-resume: Found local checkpoint (step {local_step})")
        return True  # Let Hugging Face handle the actual resuming
    elif remote_step is not None:
        # Only remote exists
        if remote_step > 0:
            print(
                f"🔄 Auto-resume: Found remote checkpoint (step {remote_step})")
        else:
            print(f"🔄 Auto-resume: Found remote checkpoint")
        # Download the remote checkpoint
        return _download_remote_checkpoint_to_temp(hub_model_id)
    else:
        # Neither exists
        print("⚠️  Auto-resume: No checkpoints found locally or remotely. Starting from scratch.")
        return None
