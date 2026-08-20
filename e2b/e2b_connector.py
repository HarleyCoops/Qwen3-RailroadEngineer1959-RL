"""
E2B Connector for Qwen3-VL Thinking model Deployment

This module provides a connector class for managing E2B sandbox environments
to run the Qwen3-VL Thinking model in a secure, isolated environment.
"""

import os
from typing import Optional, Dict, Any
from e2b import Sandbox


class E2BConnector:
    """
    Manages E2B sandbox lifecycle for running Qwen3-VL Thinking model.
    
    This connector handles:
    - Sandbox initialization and teardown
    - Model setup and loading
    - Inference execution
    - File management within the sandbox
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 300):
        """
        Initialize the E2B connector.
        
        Args:
            api_key: E2B API key. If None, reads from E2B_API_KEY environment variable.
            timeout: Timeout for sandbox operations in seconds (default: 300)
        """
        self.api_key = api_key or os.getenv('E2B_API_KEY')
        if not self.api_key:
            raise ValueError("E2B_API_KEY must be provided or set in environment variables")
        
        self.timeout = timeout
        self.sandbox: Optional[Sandbox] = None
        self.model_loaded = False
    
    def start_sandbox(self) -> None:
        """
        Start a new E2B sandbox session.
        
        Raises:
            Exception: If sandbox creation fails
        """
        try:
            print("Starting E2B sandbox...")
            self.sandbox = Sandbox(api_key=self.api_key, timeout=self.timeout)
            print(f"Sandbox started successfully. ID: {self.sandbox.id}")
        except Exception as e:
            raise Exception(f"Failed to start sandbox: {str(e)}")
    
    def upload_setup_files(self) -> None:
        """
        Upload sandbox setup files to the E2B environment.
        
        This includes requirements.txt and setup_model.py for model initialization.
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not started. Call start_sandbox() first.")
        
        try:
            print("Uploading setup files to sandbox...")
            
            # Upload requirements.txt
            with open('e2b/sandbox_setup/requirements.txt', 'r') as f:
                requirements_content = f.read()
            self.sandbox.filesystem.write('/root/requirements.txt', requirements_content)
            
            # Upload setup_model.py
            with open('e2b/sandbox_setup/setup_model.py', 'r') as f:
                setup_content = f.read()
            self.sandbox.filesystem.write('/root/setup_model.py', setup_content)
            
            print("Setup files uploaded successfully")
        except Exception as e:
            raise Exception(f"Failed to upload setup files: {str(e)}")
    
    def install_dependencies(self) -> None:
        """
        Install Python dependencies in the sandbox environment.
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not started. Call start_sandbox() first.")
        
        try:
            print("Installing dependencies in sandbox...")
            result = self.sandbox.process.start_and_wait(
                "pip install -r /root/requirements.txt"
            )
            
            if result.exit_code != 0:
                raise Exception(f"Dependency installation failed: {result.stderr}")
            
            print("Dependencies installed successfully")
        except Exception as e:
            raise Exception(f"Failed to install dependencies: {str(e)}")
    
    def setup_model(self) -> None:
        """
        Execute the model setup script within the sandbox.
        
        This downloads and initializes the Qwen3-VL Thinking model.
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not started. Call start_sandbox() first.")
        
        try:
            print("Setting up Qwen3-VL Thinking model in sandbox...")
            print("This may take several minutes depending on model size...")
            
            result = self.sandbox.process.start_and_wait(
                "python /root/setup_model.py"
            )
            
            if result.exit_code != 0:
                raise Exception(f"Model setup failed: {result.stderr}")
            
            self.model_loaded = True
            print("Model setup completed successfully")
        except Exception as e:
            raise Exception(f"Failed to setup model: {str(e)}")
    
    def run_inference(self, prompt: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run inference on the Qwen3-VL Thinking model.
        
        Args:
            prompt: Text prompt for the model
            image_path: Optional path to image file (local path, will be uploaded)
        
        Returns:
            Dict containing inference results
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not started. Call start_sandbox() first.")
        
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call setup_model() first.")
        
        try:
            # If image provided, upload it first
            if image_path:
                print(f"Uploading image: {image_path}")
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                remote_image_path = f"/root/input_image.{image_path.split('.')[-1]}"
                self.sandbox.filesystem.write(remote_image_path, image_data)
            else:
                remote_image_path = None
            
            # Create inference script
            inference_script = f"""
import json
import sys

# Your inference code here
# This is a placeholder - actual implementation depends on model requirements
result = {{
    "prompt": "{prompt}",
    "image_path": "{remote_image_path if remote_image_path else 'None'}",
    "response": "Model inference result placeholder",
    "status": "success"
}}

print(json.dumps(result))
"""
            
            self.sandbox.filesystem.write('/root/inference.py', inference_script)
            
            # Run inference
            print("Running inference...")
            result = self.sandbox.process.start_and_wait("python /root/inference.py")
            
            if result.exit_code != 0:
                raise Exception(f"Inference failed: {result.stderr}")
            
            # Parse and return results
            import json
            return json.loads(result.stdout)
            
        except Exception as e:
            raise Exception(f"Failed to run inference: {str(e)}")
    
    def close(self) -> None:
        """
        Close the sandbox session and clean up resources.
        """
        if self.sandbox:
            print("Closing sandbox...")
            try:
                self.sandbox.kill()
                print("Sandbox closed successfully")
            except Exception as e:
                print(f"Warning: Error closing sandbox: {str(e)}")
            finally:
                self.sandbox = None
                self.model_loaded = False
    
    def __enter__(self):
        """Context manager entry."""
        self.start_sandbox()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

