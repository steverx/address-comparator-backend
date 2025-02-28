// TypeScript Definition File (types/address.ts)
export interface AddressComponent {
    label: string;
    value: string;
  }
  
  export interface AddressValidationResult {
    is_valid: boolean;
    confidence: number;
    components: Record<string, string>;
    flags: Record<string, boolean>;
  }
  
  export interface Address {
    original_address: string;
    spelling_corrected: string;
    parsed_components: Record<string, string>;
    similar_addresses: Array<[string, number]>;
    validation: {
      is_valid: boolean;
      confidence: number;
    };
  }
  
  export interface ComparisonResult {
    source_address: string;
    normalized_source: string;
    matched_address: string;
    normalized_match: string;
    match_score: number;
  }
  
  export interface ComparisonJob {
    job_id: string;
    task_id?: string;
    status: 'started' | 'processing' | 'completed' | 'error';
    progress?: number;
    results?: ComparisonResult[];
    error?: string;
  }
  
  // TypeScript Service for Address Comparison (services/address.service.ts)
  import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
  import { Address, AddressValidationResult, ComparisonJob, ComparisonResult } from '../types/address';
  
  export class AddressService {
    private api: AxiosInstance;
    private progressInterval: number | null = null;
  
    constructor(baseURL: string = '/api/v1') {
      this.api = axios.create({
        baseURL,
        timeout: 30000, // 30 seconds
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });
  
      // Add response interceptor for error handling
      this.api.interceptors.response.use(
        response => response,
        error => {
          console.error('API Error:', error.response?.data || error.message);
          return Promise.reject(error);
        }
      );
    }
  
    /**
     * Get columns from a file
     */
    async getColumns(file: File): Promise<{columns: string[], suggested_address_columns: string[]}> {
      const formData = new FormData();
      formData.append('file', file);
  
      const response = await this.api.post('/columns', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
  
      return response.data.data;
    }
  
    /**
     * Validate a single address
     */
    async validateAddress(address: string): Promise<AddressValidationResult> {
      const response = await this.api.post('/validate', {
        address
      });
  
      return response.data.data;
    }
  
    /**
     * Compare addresses between two files
     */
    async compareAddresses(
      file1: File,
      file2: File,
      columns1: string[],
      columns2: string[],
      threshold: number = 80,
      onProgress?: (progress: number) => void
    ): Promise<ComparisonJob> {
      const formData = new FormData();
      formData.append('file1', file1);
      formData.append('file2', file2);
      
      // Append column selections
      formData.append('columns1', JSON.stringify(columns1));
      formData.append('columns2', JSON.stringify(columns2));
      
      // Append threshold
      formData.append('threshold', threshold.toString());
  
      const response = await this.api.post('/compare', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
  
      const job: ComparisonJob = response.data.data;
  
      // Start progress tracking if callback provided
      if (onProgress && job.status === 'processing') {
        this.trackProgress(job.job_id, onProgress);
      }
  
      return job;
    }
  
    /**
     * Get progress for a job
     */
    async getProgress(jobId: string): Promise<ComparisonJob> {
      const response = await this.api.get(`/progress/${jobId}`);
      return response.data.data;
    }
  
    /**
     * Get results for a completed job
     */
    async getResults(jobId: string): Promise<ComparisonJob> {
      const response = await this.api.get(`/results/${jobId}`);
      return response.data.data;
    }
  
    /**
     * Get download URL for results
     */
    getDownloadUrl(jobId: string): string {
      return `${this.api.defaults.baseURL}/download/${jobId}`;
    }
  
    /**
     * Track progress of a job
     */
    private trackProgress(jobId: string, callback: (progress: number) => void): void {
      // Clear any existing interval
      if (this.progressInterval !== null) {
        window.clearInterval(this.progressInterval);
      }
  
      // Start new interval
      this.progressInterval = window.setInterval(async () => {
        try {
          const job = await this.getProgress(jobId);
          
          // Call progress callback
          if (job.progress !== undefined) {
            callback(job.progress);
          }
  
          // Stop tracking if job is completed or failed
          if (job.status === 'completed' || job.status === 'error') {
            this.stopProgressTracking();
          }
        } catch (error) {
          console.error('Error tracking progress:', error);
          this.stopProgressTracking();
        }
      }, 1000); // Check every second
    }
  
    /**
     * Stop progress tracking
     */
    private stopProgressTracking(): void {
      if (this.progressInterval !== null) {
        window.clearInterval(this.progressInterval);
        this.progressInterval = null;
      }
    }
  }
  
  // Python Bridge for TypeScript Integration (utils/ts_bridge.py)
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