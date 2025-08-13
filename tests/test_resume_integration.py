"""
Integration tests for resume functionality that test actual parsing logic.
"""
import os
import json
import tempfile
import unittest
from unittest.mock import patch

import pytest

from krill.utils.resume import (
    _get_local_checkpoint_step,
    _get_remote_checkpoint_step,
    determine_resume_checkpoint,
    _handle_auto_resume
)


@pytest.mark.integration
class TestResumeIntegration(unittest.TestCase):
    """Integration tests for resume functionality."""

    def test_get_local_checkpoint_step_real_parsing(self):
        """Test actual parsing of local checkpoint step from real files."""
        # Create a temporary directory structure that mimics a real checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create output directory
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)
            
            # Create a checkpoint directory
            checkpoint_dir = os.path.join(output_dir, "checkpoint-500")
            os.makedirs(checkpoint_dir)
            
            # Create a trainer_state.json with a real step
            trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
            trainer_state = {
                "global_step": 500,
                "epoch": 2.5,
                "logging_steps": 100
            }
            with open(trainer_state_path, 'w') as f:
                json.dump(trainer_state, f)
            
            # Mock get_last_checkpoint to return our checkpoint directory
            with patch('krill.utils.resume.get_last_checkpoint') as mock_get_last:
                mock_get_last.return_value = checkpoint_dir
                
                # Test the actual parsing logic
                step = _get_local_checkpoint_step(output_dir)
                self.assertEqual(step, 500)

    def test_get_local_checkpoint_step_missing_file(self):
        """Test handling when trainer_state.json is missing."""
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)
            
            # Create a checkpoint directory but NO trainer_state.json
            checkpoint_dir = os.path.join(output_dir, "checkpoint-500")
            os.makedirs(checkpoint_dir)
            
            # Mock get_last_checkpoint to return our checkpoint directory
            with patch('krill.utils.resume.get_last_checkpoint') as mock_get_last:
                mock_get_last.return_value = checkpoint_dir
                
                # Should return None when file is missing
                step = _get_local_checkpoint_step(output_dir)
                self.assertIsNone(step)

    def test_get_remote_checkpoint_step_real_parsing(self):
        """Test actual parsing of remote checkpoint step from real repository."""
        # Test with the real remote repository
        # Note: This model (pretraining/krill-e2e-ci-pico) is just an arbitrary model training result
        # For reproduction, any model can be trained and uploaded to the hub
        hub_model_id = "pretraining/krill-e2e-ci-pico"
        step = _get_remote_checkpoint_step(hub_model_id)
        
        # Check that we got a valid step number (should be 44 for this model)
        self.assertIsNotNone(step)
        self.assertEqual(step, 44)

    def test_determine_resume_checkpoint_remote_real(self):
        """Test determining resume checkpoint with real remote repository."""
        from krill.utils.resume import determine_resume_checkpoint
        
        # Test with the real remote repository
        output_dir = "/tmp/test_output"
        hub_model_id = "pretraining/krill-e2e-ci-pico"
        
        # Mock get_last_checkpoint to return None (no local checkpoint)
        with patch('krill.utils.resume.get_last_checkpoint') as mock_get_last:
            mock_get_last.return_value = None
            
            # Test remote resume option
            result = determine_resume_checkpoint("remote", output_dir, hub_model_id)
            
            # Should return a path to the downloaded checkpoint
            self.assertIsNotNone(result)
            self.assertIsInstance(result, str)
            # Check that the path contains the expected structure
            self.assertIn("last-checkpoint", result)

    def test_handle_auto_resume_remote_real(self):
        """Test auto resume with real remote repository."""
        # Test with the real remote repository
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "output")
            hub_model_id = "pretraining/krill-e2e-ci-pico"
            
            # Mock get_last_checkpoint to return None (no local checkpoint)
            with patch('krill.utils.resume.get_last_checkpoint') as mock_get_last:
                mock_get_last.return_value = None
                
                # Test auto resume with only remote checkpoint available
                result = _handle_auto_resume(output_dir, hub_model_id)
                
                # Should return a path to the downloaded checkpoint
                self.assertIsNotNone(result)
                self.assertIsInstance(result, str)
                # Check that the path contains the expected structure
                self.assertIn("last-checkpoint", result)

    def test_end_to_end_resume_functionality(self):
        """End-to-end test of resume functionality with real remote repository."""
        # Note: This test uses pretraining/krill-e2e-ci-pico which is just an arbitrary model
        # training result. For reproduction, any model can be trained and uploaded to the hub.
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "output")
            hub_model_id = "pretraining/krill-e2e-ci-pico"
            
            # Test 1: Get remote checkpoint step
            step = _get_remote_checkpoint_step(hub_model_id)
            self.assertIsNotNone(step)
            self.assertEqual(step, 44)
            
            # Test 2: Determine resume checkpoint with remote option
            # Mock get_last_checkpoint to return None (no local checkpoint)
            with patch('krill.utils.resume.get_last_checkpoint') as mock_get_last:
                mock_get_last.return_value = None
                
                result = determine_resume_checkpoint("remote", output_dir, hub_model_id)
                self.assertIsNotNone(result)
                self.assertIsInstance(result, str)
                self.assertIn("last-checkpoint", result)
                
                # Test 3: Auto resume with only remote checkpoint
                auto_result = _handle_auto_resume(output_dir, hub_model_id)
                self.assertIsNotNone(auto_result)
                self.assertIsInstance(auto_result, str)
                self.assertIn("last-checkpoint", auto_result)

    # Note: Testing _get_remote_checkpoint_step would require more complex mocking
    # of the Hugging Face Hub API and network calls, or using a real repository


if __name__ == '__main__':
    unittest.main()