import json
import logging
from typing import Dict, Any, List, Optional
import subprocess
import os
import tempfile

logger = logging.getLogger(__name__)

class TypeScriptBridge:
    """
    Bridge to execute TypeScript code from Python.
    """
    
    def __init__(self, 
                ts_root_dir: str,
                node_path: Optional[str] = None,
                npm_path: Optional[str] = None):
        """
        Initialize TypeScript bridge.
        
        Args:
            ts_root_dir: Root directory for TypeScript code
            node_path: Path to Node.js executable (optional)
            npm_path: Path to npm executable (optional)
        """
        self.ts_root_dir = os.path.abspath(ts_root_dir)
        self.node_path = node_path or 'node'
        self.npm_path = npm_path or 'npm'
        
        # Ensure TypeScript is compiled
        self.ensure_compiled()
    
    def ensure_compiled(self) -> None:
        """Ensure TypeScript code is compiled."""
        logger.info(f"Compiling TypeScript code in {self.ts_root_dir}")
        
        try:
            # Run npm install if needed
            if not os.path.exists(os.path.join(self.ts_root_dir, 'node_modules')):
                subprocess.run(
                    [self.npm_path, 'install'],
                    cwd=self.ts_root_dir,
                    check=True,
                    capture_output=True,
                    text=True
                )
            
            # Compile TypeScript
            result = subprocess.run(
                [self.npm_path, 'run', 'build'],
                cwd=self.ts_root_dir,
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.info("TypeScript compilation successful")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile TypeScript: {e.stderr}")
            raise ValueError(f"TypeScript compilation failed: {e.stderr}")
    
    def execute_js(self, js_file: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute JavaScript code with input data.
        
        Args:
            js_file: Path to JavaScript file (relative to compiled output)
            input_data: Dictionary with input data
            
        Returns:
            Dictionary with execution results
        """
        # Full path to JS file
        dist_dir = os.path.join(self.ts_root_dir, 'dist')
        js_path = os.path.join(dist_dir, js_file)
        
        if not os.path.exists(js_path):
            raise FileNotFoundError(f"JavaScript file not found: {js_path}")
        
        # Create temporary file for input data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_input:
            json.dump(input_data or {}, temp_input)
            temp_input_path = temp_input.name
        
        # Create temporary file for output data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        try:
            # Execute JavaScript with Node.js
            result = subprocess.run(
                [
                    self.node_path,
                    js_path,
                    '--input', temp_input_path,
                    '--output', temp_output_path
                ],
                cwd=dist_dir,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Read output data
            with open(temp_output_path, 'r') as f:
                output_data = json.load(f)
            
            return output_data
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to execute JavaScript: {e.stderr}")
            raise ValueError(f"JavaScript execution failed: {e.stderr}")
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
    
    def process_addresses(self, addresses: List[str], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process addresses using TypeScript implementation.
        
        Args:
            addresses: List of addresses to process
            options: Processing options
            
        Returns:
            Processing results
        """
        input_data = {
            'addresses': addresses,
            'options': options or {}
        }
        
        return self.execute_js('services/address_processor.js', input_data)
    
    def match_addresses(self, 
                      query_address: str, 
                      target_addresses: List[str],
                      threshold: float = 0.8,
                      max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Match addresses using TypeScript implementation.
        
        Args:
            query_address: Address to match
            target_addresses: List of addresses to match against
            threshold: Minimum similarity threshold
            max_results: Maximum number of results
            
        Returns:
            List of matching results
        """
        input_data = {
            'query': query_address,
            'targets': target_addresses,
            'options': {
                'threshold': threshold,
                'maxResults': max_results
            }
        }
        
        result = self.execute_js('services/address_matcher.js', input_data)
        return result.get('matches', [])