"""
Test suite for resume functionality in Krill.
"""
import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pytest

from krill.utils.resume import (
    determine_resume_checkpoint,
    _get_local_checkpoint_step,
    _get_remote_checkpoint_step,
    _validate_local_checkpoint,
    _validate_remote_checkpoint,
    _handle_auto_resume
)


@pytest.mark.unit
class TestResumeFunctionality(unittest.TestCase):
    """Test cases for resume functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = "./artifacts/models/test-model"
        self.hub_model_id = "test/model"
        
    def test_determine_resume_checkpoint_false(self):
        """Test resume option 'false' returns None."""
        result = determine_resume_checkpoint("false", self.output_dir, self.hub_model_id)
        self.assertIsNone(result)
        
    def test_determine_resume_checkpoint_false_case_insensitive(self):
        """Test resume option 'False' (case insensitive) returns None."""
        result = determine_resume_checkpoint("False", self.output_dir, self.hub_model_id)
        self.assertIsNone(result)
        
    def test_determine_resume_checkpoint_true(self):
        """Test resume option 'true' returns True."""
        result = determine_resume_checkpoint("true", self.output_dir, self.hub_model_id)
        self.assertTrue(result)
        
    def test_determine_resume_checkpoint_invalid_option(self):
        """Test invalid resume option raises ValueError."""
        with self.assertRaises(ValueError) as context:
            determine_resume_checkpoint("invalid", self.output_dir, self.hub_model_id)
        self.assertIn("Invalid resume option", str(context.exception))

    @patch('krill.utils.resume.get_last_checkpoint')
    def test_validate_local_checkpoint_exists(self, mock_get_last_checkpoint):
        """Test validation when local checkpoint exists."""
        mock_get_last_checkpoint.return_value = "/path/to/checkpoint-100"
        # Should not raise an exception
        _validate_local_checkpoint(self.output_dir)
        
    @patch('krill.utils.resume.get_last_checkpoint')
    def test_validate_local_checkpoint_not_found(self, mock_get_last_checkpoint):
        """Test validation when local checkpoint doesn't exist."""
        mock_get_last_checkpoint.return_value = None
        with self.assertRaises(FileNotFoundError) as context:
            _validate_local_checkpoint(self.output_dir)
        self.assertIn("No valid checkpoint found", str(context.exception))

    @patch('huggingface_hub.HfApi')
    def test_validate_remote_checkpoint_exists(self, mock_hf_api_class):
        """Test validation when remote checkpoint exists."""
        # Mock the API response to show last-checkpoint directory exists
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        # Create mock items - one folder named "last-checkpoint"
        mock_folder = MagicMock()
        mock_folder.tree_id = "some-tree-id"
        mock_folder.path = "last-checkpoint"
        
        mock_api.list_repo_tree.return_value = [mock_folder]
        
        # Should not raise an exception
        _validate_remote_checkpoint(self.hub_model_id)
        
    @patch('huggingface_hub.HfApi')
    def test_validate_remote_checkpoint_not_found(self, mock_hf_api_class):
        """Test validation when remote checkpoint doesn't exist."""
        # Mock the API response to show no last-checkpoint directory
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        # Create mock items - no folder named "last-checkpoint"
        mock_file = MagicMock()
        mock_file.blob_id = "some-blob-id"
        mock_file.path = "config.json"
        
        mock_api.list_repo_tree.return_value = [mock_file]
        
        with self.assertRaises(FileNotFoundError) as context:
            _validate_remote_checkpoint(self.hub_model_id)
        self.assertIn("'last-checkpoint' directory not found in remote model", str(context.exception))

    @patch('huggingface_hub.hf_hub_download')
    @patch('huggingface_hub.HfApi')
    def test_get_remote_checkpoint_step_exists(self, mock_hf_api_class, mock_hf_hub_download):
        """Test getting remote checkpoint step when it exists."""
        # Mock the API response to show last-checkpoint directory exists
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        # Create mock items - one folder named "last-checkpoint"
        mock_folder = MagicMock()
        mock_folder.tree_id = "some-tree-id"
        mock_folder.path = "last-checkpoint"
        
        mock_api.list_repo_tree.return_value = [mock_folder]
        
        # Mock the trainer state file download
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump({"global_step": 1234}, f)
            temp_file_path = f.name
            
        mock_hf_hub_download.return_value = temp_file_path
        
        try:
            step = _get_remote_checkpoint_step(self.hub_model_id)
            self.assertEqual(step, 1234)
        finally:
            os.unlink(temp_file_path)
            
    @patch('huggingface_hub.HfApi')
    def test_get_remote_checkpoint_step_no_directory(self, mock_hf_api_class):
        """Test getting remote checkpoint step when directory doesn't exist."""
        # Mock the API response to show no last-checkpoint directory
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        # Create mock items - no folder named "last-checkpoint"
        mock_file = MagicMock()
        mock_file.blob_id = "some-blob-id"
        mock_file.path = "config.json"
        
        mock_api.list_repo_tree.return_value = [mock_file]
        
        step = _get_remote_checkpoint_step(self.hub_model_id)
        self.assertIsNone(step)

    @patch('krill.utils.resume.get_last_checkpoint')
    def test_get_local_checkpoint_step_exists(self, mock_get_last_checkpoint):
        """Test getting local checkpoint step when it exists."""
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = os.path.join(temp_dir, "checkpoint-100")
            os.makedirs(checkpoint_dir)
            
            # Create trainer_state.json with a step
            trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
            with open(trainer_state_path, 'w') as f:
                json.dump({"global_step": 567}, f)
                
            mock_get_last_checkpoint.return_value = checkpoint_dir
            
            step = _get_local_checkpoint_step(temp_dir)
            self.assertEqual(step, 567)
            
    @patch('krill.utils.resume.get_last_checkpoint')
    def test_get_local_checkpoint_step_not_found(self, mock_get_last_checkpoint):
        """Test getting local checkpoint step when checkpoint doesn't exist."""
        mock_get_last_checkpoint.return_value = None
        step = _get_local_checkpoint_step("/nonexistent")
        self.assertIsNone(step)

    @patch('krill.utils.resume._get_cached_remote_checkpoint')
    @patch('krill.utils.resume._get_remote_checkpoint_step')
    @patch('krill.utils.resume._get_local_checkpoint_step')
    def test_handle_auto_resume_remote_newer(self, mock_get_local_step, mock_get_remote_step, mock_get_cached):
        """Test auto resume when remote checkpoint is newer."""
        mock_get_local_step.return_value = 1000
        mock_get_remote_step.return_value = 2000
        mock_get_cached.return_value = "/path/to/remote/checkpoint"
        
        result = _handle_auto_resume(self.output_dir, self.hub_model_id)
        self.assertEqual(result, "/path/to/remote/checkpoint")
        
    @patch('krill.utils.resume._get_cached_remote_checkpoint')
    @patch('krill.utils.resume._get_remote_checkpoint_step')
    @patch('krill.utils.resume._get_local_checkpoint_step')
    def test_handle_auto_resume_local_newer(self, mock_get_local_step, mock_get_remote_step, mock_get_cached):
        """Test auto resume when local checkpoint is newer."""
        mock_get_local_step.return_value = 2000
        mock_get_remote_step.return_value = 1000
        mock_get_cached.return_value = "/path/to/remote/checkpoint"
        
        result = _handle_auto_resume(self.output_dir, self.hub_model_id)
        self.assertTrue(result)  # Should return True to let HF handle it
        
    @patch('krill.utils.resume._get_cached_remote_checkpoint')
    @patch('krill.utils.resume._get_remote_checkpoint_step')
    @patch('krill.utils.resume._get_local_checkpoint_step')
    def test_handle_auto_resume_only_local(self, mock_get_local_step, mock_get_remote_step, mock_get_cached):
        """Test auto resume when only local checkpoint exists."""
        mock_get_local_step.return_value = 1500
        mock_get_remote_step.return_value = None
        mock_get_cached.return_value = "/path/to/remote/checkpoint"
        
        result = _handle_auto_resume(self.output_dir, self.hub_model_id)
        self.assertTrue(result)  # Should return True to let HF handle it
        
    @patch('krill.utils.resume._get_cached_remote_checkpoint')
    @patch('krill.utils.resume._get_remote_checkpoint_step')
    @patch('krill.utils.resume._get_local_checkpoint_step')
    def test_handle_auto_resume_only_remote(self, mock_get_local_step, mock_get_remote_step, mock_get_cached):
        """Test auto resume when only remote checkpoint exists."""
        mock_get_local_step.return_value = None
        mock_get_remote_step.return_value = 2500
        mock_get_cached.return_value = "/path/to/remote/checkpoint"
        
        result = _handle_auto_resume(self.output_dir, self.hub_model_id)
        self.assertEqual(result, "/path/to/remote/checkpoint")
        
    @patch('krill.utils.resume._get_remote_checkpoint_step')
    @patch('krill.utils.resume._get_local_checkpoint_step')
    def test_handle_auto_resume_none_exist(self, mock_get_local_step, mock_get_remote_step):
        """Test auto resume when no checkpoints exist."""
        mock_get_local_step.return_value = None
        mock_get_remote_step.return_value = None
        
        result = _handle_auto_resume(self.output_dir, self.hub_model_id)
        self.assertIsNone(result)

    @patch('krill.utils.resume._get_cached_remote_checkpoint')
    @patch('krill.utils.resume._validate_remote_checkpoint')
    @patch('krill.utils.resume._get_remote_checkpoint_step')
    def test_determine_resume_checkpoint_remote_with_step(self, mock_get_step, mock_validate, mock_get_cached):
        """Test remote resume option with step information."""
        mock_get_step.return_value = 3000
        mock_get_cached.return_value = "/path/to/remote/checkpoint"
        # validate should not raise exception
        
        result = determine_resume_checkpoint("remote", self.output_dir, self.hub_model_id)
        self.assertEqual(result, "/path/to/remote/checkpoint")
        
    @patch('krill.utils.resume._get_cached_remote_checkpoint')
    @patch('krill.utils.resume._validate_remote_checkpoint')
    @patch('krill.utils.resume._get_remote_checkpoint_step')
    def test_determine_resume_checkpoint_remote_without_step(self, mock_get_step, mock_validate, mock_get_cached):
        """Test remote resume option without step information."""
        mock_get_step.return_value = 0  # Exists but can't determine step
        mock_get_cached.return_value = "/path/to/remote/checkpoint"
        # validate should not raise exception
        
        result = determine_resume_checkpoint("remote", self.output_dir, self.hub_model_id)
        self.assertEqual(result, "/path/to/remote/checkpoint")


if __name__ == '__main__':
    unittest.main()