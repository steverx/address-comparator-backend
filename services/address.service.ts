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