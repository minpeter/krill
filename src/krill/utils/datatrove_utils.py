"""
Datatrove integration utilities for enhanced preprocessing.
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from datasets import Dataset
from krill.utils.config import DatasetConfig

# Import datatrove components
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import ParquetWriter, HuggingFaceDatasetWriter
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.extractors import Trafilatura
from datatrove.data import Document


class DatatrovePreprocessor:
    """
    Datatrove-based preprocessor that provides enhanced text processing capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stats = {
            'total_processed': 0,
            'filtered_quality': 0,
            'filtered_duplicates': 0,
            'final_count': 0
        }

    def _is_local_path(self, path: str) -> bool:
        """
        Determine if a path is a local file path or a HuggingFace dataset ID.
        
        Returns True for local paths, False for HuggingFace dataset IDs.
        """
        if not path:
            return True
        
        # First check if it looks like a HuggingFace dataset ID (username/dataset-name)
        if '/' in path and len(path.split('/')) == 2:
            parts = path.split('/')
            # If both parts are valid identifiers and no other path indicators, it's likely HF ID
            if (parts[0] and parts[1] and  # Both parts must be non-empty
                self._is_valid_identifier(parts[0]) and 
                self._is_valid_identifier(parts[1])):
                # Check for local path indicators that would override HF detection
                local_override_indicators = [
                    './',  # Relative paths
                    '../',  # Parent directory
                    '~/',  # Home directory
                    '\\',  # Windows backslash
                ]
                
                for indicator in local_override_indicators:
                    if indicator in path:
                        return True
                
                # Check if path exists locally (strong indicator)
                try:
                    import os
                    if os.path.exists(path) or os.path.exists(os.path.dirname(path)):
                        return True
                except Exception:
                    pass
                
                # Additional check: if it contains common local path patterns, treat as local
                # Only check if the entire component matches, not partial matches
                local_path_patterns = [
                    'artifacts',  # Common build directory
                    'output',     # Common output directory
                    'tmp',        # Temporary directory
                    'build',      # Build directory
                ]
                
                path_parts = path.lower().split('/')
                for pattern in local_path_patterns:
                    if pattern in path_parts:
                        return True
                
                # If no local indicators found, treat as HF dataset ID
                return False
            
        # Check for obvious local path indicators
        local_indicators = [
            './',  # Relative paths
            '../',  # Parent directory
            '~/',  # Home directory
            '\\',  # Windows backslash
        ]
        
        for indicator in local_indicators:
            if indicator in path:
                return True
        
        # Check for absolute paths (starting with / on Unix or drive letter on Windows)
        if path.startswith('/') or (len(path) > 2 and path[1] == ':'):
            return True
        
        # Check if path exists locally or can be created
        try:
            import os
            if os.path.exists(path) or os.path.exists(os.path.dirname(path)):
                return True
        except Exception:
            pass
        
        # Default to local path if uncertain
        return True

    def _is_valid_identifier(self, identifier: str) -> bool:
        """Check if a string is a valid HuggingFace identifier (allows alphanumeric, hyphens, underscores)."""
        if not identifier:
            return False
        
        # Allow alphanumeric characters, hyphens, and underscores
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', identifier))

    def _parse_split_with_slice(self, split_str: str) -> tuple[str, int, int]:
        """
        Parse split string with slice notation (e.g., 'train[:10_000]') into split name, start, and limit.
        
        Returns:
            tuple of (split_name, skip, limit)
            - split_name: the base split (e.g., 'train')
            - skip: number of items to skip (for start index)
            - limit: maximum number of items to take (-1 for no limit)
        """
        import re
        
        # Handle slice notation like 'train[:10_000]' or 'train[100:1000]'
        slice_pattern = r'^([^[\]]+)\[([^:]*):([^]]*)\]$'
        match = re.match(slice_pattern, split_str)
        
        if match:
            split_name = match.group(1)
            start_str = match.group(2)
            end_str = match.group(3)
            
            # Parse start index (default to 0)
            skip = 0
            if start_str.strip():
                try:
                    skip = int(start_str.replace('_', ''))
                except ValueError:
                    print(f"Warning: Invalid start index '{start_str}', using 0")
            
            # Parse end index 
            limit = -1
            if end_str.strip():
                try:
                    end_idx = int(end_str.replace('_', ''))
                    limit = end_idx - skip  # Convert to limit (number of items to take)
                    if limit <= 0:
                        print(f"Warning: Invalid slice range, start={skip}, end={end_idx}")
                        limit = -1
                except ValueError:
                    print(f"Warning: Invalid end index '{end_str}', using no limit")
            
            return split_name, skip, limit
        
        # No slice notation, return as-is
        return split_str, 0, -1

    def create_pipeline_steps(self, dataset_configs: List[DatasetConfig]) -> List[Any]:
        """Create datatrove processing pipeline steps."""
        pipeline_steps = []

        # 1. Readers for each dataset
        for ds_config in dataset_configs:
            # Parse split with potential slice notation
            split_name, skip, limit = self._parse_split_with_slice(ds_config.split)
            
            if skip > 0 or limit > 0:
                print(f"Parsed split '{ds_config.split}' -> split='{split_name}', skip={skip}, limit={limit}")
            
            reader = HuggingFaceDatasetReader(
                dataset=ds_config.path,
                dataset_options={'split': split_name},  # Use parsed split name without slice notation
                text_key=ds_config.text_column,
                streaming=self.config.get('streaming', True),
                limit=limit if limit > 0 else -1,  # Use parsed limit
                skip=skip  # Use parsed skip
            )
            pipeline_steps.append(reader)

        # 2. Text extractor and cleaner (using Trafilatura for robust text extraction)
        if self.config.get('use_trafilatura', False):
            extractor = Trafilatura()
            pipeline_steps.append(extractor)

        # 3. Quality filters
        min_length = self.config.get('min_length', 100)
        max_length = self.config.get('max_length', None)

        # Custom quality filter
        class QualityFilter(BaseFilter):
            def __init__(self, min_len: int, max_len: Optional[int] = None):
                super().__init__()
                self.min_len = min_len
                self.max_len = max_len

            def filter(self, doc: Document) -> bool:
                text_len = len(doc.text)
                if text_len < self.min_len:
                    return False
                if self.max_len and text_len > self.max_len:
                    return False
                return True

        quality_filter = QualityFilter(min_length, max_length)
        pipeline_steps.append(quality_filter)

        # 4. Deduplication - simplified approach
        dedup_algorithm = self.config.get('deduplication_algorithm', 'minhash')
        if dedup_algorithm in ["minhash", "exact"]:
            print(f"Note: Advanced deduplication ({dedup_algorithm}) requires additional setup.")
            print("For now, using basic text quality filtering. Advanced deduplication coming soon.")
            # For basic deduplication, we could add a simple hash-based filter here
            # but for simplicity and to avoid dependency issues, we'll skip it for now

        # 5. Statistics collectors (optional)
        if self.config.get('collect_stats', False):
            try:
                # Check if stats classes are available with their dependencies
                stats_steps = []
                
                # Try TokenStats
                try:
                    from datatrove.pipeline.stats import TokenStats
                    stats_steps.append(TokenStats())
                except ImportError as e:
                    if 'tldextract' in str(e):
                        print("Info: TokenStats requires tldextract. Install with: pip install tldextract")
                    else:
                        print(f"Info: TokenStats not available: {e}")
                
                # Try WordStats
                try:
                    from datatrove.pipeline.stats import WordStats
                    stats_steps.append(WordStats())
                except ImportError as e:
                    print(f"Info: WordStats not available: {e}")
                
                if stats_steps:
                    pipeline_steps.extend(stats_steps)
                    print(f"Added {len(stats_steps)} statistics collectors to pipeline")
                else:
                    print("Info: No statistics collectors available (missing dependencies)")
                    
            except Exception as e:
                print(f"Warning: Could not add statistics collection: {e}")

        return pipeline_steps

    def process_datasets(self, dataset_configs: List[DatasetConfig], output_path: str) -> Dataset:
        """
        Process datasets using datatrove pipeline.

        Returns a HuggingFace Dataset ready for tokenization.
        """
        print("🚀 Starting datatrove preprocessing pipeline...")

        # Create output directory for local output
        os.makedirs(output_path, exist_ok=True)
        temp_output = os.path.join(output_path, "datatrove_temp")

        # Create pipeline steps
        pipeline_steps = self.create_pipeline_steps(dataset_configs)

        # Determine output strategy based on configuration
        hf_dataset_id = self.config.get('dataset_prepared_hf_id')
        local_path = output_path
        
        writers = []
        
        # Always save locally first
        local_writer = ParquetWriter(
            output_folder=temp_output,
            compression="gzip"
        )
        writers.append(local_writer)
        print(f"📁 Will save locally to: {temp_output}")
        
        # Optionally upload to HuggingFace Hub
        if hf_dataset_id:
            try:
                # Validate HF dataset ID format
                if not self._is_valid_hf_dataset_id(hf_dataset_id):
                    print(f"⚠️  Warning: Invalid HuggingFace dataset ID format: {hf_dataset_id}")
                    print("   Expected format: 'username/dataset-name'. Skipping HF upload.")
                else:
                    hf_writer = HuggingFaceDatasetWriter(
                        dataset=hf_dataset_id,
                        private=True,  # Default to private for safety
                        compression="snappy"
                    )
                    writers.append(hf_writer)
                    print(f"☁️  Will also upload to HuggingFace Hub: {hf_dataset_id}")
            except Exception as e:
                print(f"⚠️  Warning: Could not set up HuggingFace upload: {e}")
                print("   Continuing with local output only.")
        
        # Add writers to pipeline
        pipeline_steps.extend(writers)

        # Execute pipeline
        executor = LocalPipelineExecutor(
            pipeline=pipeline_steps,
            logging_dir=os.path.join(output_path, "logs"),
            workers=self.config.get('num_workers', 1) if self.config.get('num_workers', 1) > 1 else 1
        )

        print(f"📊 Running pipeline with {self.config.get('num_workers', 1)} workers...")
        try:
            executor.run()
            print("✅ Pipeline execution completed successfully!")
        except Exception as e:
            print(f"❌ Pipeline execution failed: {e}")
            raise

        # Load the processed dataset from local output
        print("📖 Loading processed dataset...")
        try:
            # Try to load from Parquet files generated by ParquetWriter
            processed_dataset = self._load_from_datatrove_output(temp_output)
        except Exception as e:
            print(f"Warning: Could not load from datatrove output: {e}")
            # Fallback: try HuggingFace dataset format
            try:
                from datasets import load_from_disk
                processed_dataset = load_from_disk(temp_output)
            except Exception as e2:
                print(f"Error: Could not load dataset in any format: {e2}")
                raise ValueError(f"Failed to load processed dataset: {e}")

        # Clean up temporary files if needed
        if self.config.get('cleanup_temp', True):
            import shutil
            try:
                shutil.rmtree(temp_output)
                print("🧹 Cleaned up temporary files")
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")

        print(f"✅ Datatrove preprocessing complete!")
        print(f"   Final dataset size: {len(processed_dataset)} samples")
        
        if hf_dataset_id:
            print(f"   Local output: {output_path}")
            print(f"   HuggingFace dataset: https://huggingface.co/datasets/{hf_dataset_id}")
        else:
            print(f"   Output saved to: {output_path}")

        return processed_dataset

    def _is_valid_hf_dataset_id(self, dataset_id: str) -> bool:
        """Validate HuggingFace dataset ID format."""
        if not dataset_id or '/' not in dataset_id:
            return False
        
        parts = dataset_id.split('/')
        if len(parts) != 2:
            return False
        
        username, dataset_name = parts
        
        # Basic validation: should be valid identifiers  
        if not (self._is_valid_identifier(username) and self._is_valid_identifier(dataset_name)):
            return False
        
        return True

    def _load_from_datatrove_output(self, output_path: str) -> Dataset:
        """Load dataset from datatrove output files (Parquet and JSONL formats)."""
        import json
        import gzip
        from pathlib import Path

        documents = []
        
        print(f"Looking for output files in: {output_path}")

        # First try to load Parquet files (preferred format from ParquetWriter)
        parquet_files = list(Path(output_path).glob("**/*.parquet"))
        if parquet_files:
            print(f"Found {len(parquet_files)} Parquet files")
            try:
                # Try to load with pandas first (if available)
                try:
                    import pandas as pd
                    all_data = []
                    for file_path in parquet_files:
                        print(f"Processing Parquet file: {file_path}")
                        df = pd.read_parquet(file_path)
                        if 'text' in df.columns:
                            all_data.extend(df[['text']].to_dict('records'))
                        else:
                            print(f"Warning: No 'text' column found in {file_path}")
                    documents = all_data
                except ImportError:
                    print("Pandas not available, trying pyarrow directly")
                    try:
                        import pyarrow.parquet as pq
                        all_data = []
                        for file_path in parquet_files:
                            print(f"Processing Parquet file: {file_path}")
                            table = pq.read_table(file_path)
                            if 'text' in table.column_names:
                                df = table.select(['text']).to_pandas()
                                all_data.extend(df.to_dict('records'))
                            else:
                                print(f"Warning: No 'text' column found in {file_path}")
                        documents = all_data
                    except ImportError:
                        print("PyArrow not available, falling back to JSONL loading")
            except Exception as e:
                print(f"Error loading Parquet files: {e}, falling back to JSONL")

        # Fallback to JSONL files if Parquet loading failed or no Parquet files found
        if not documents:
            # Look for .jsonl.gz files (JsonlWriter with gzip compression)
            jsonl_gz_files = list(Path(output_path).glob("**/*.jsonl.gz"))
            if jsonl_gz_files:
                print(f"Found {len(jsonl_gz_files)} gzipped JSONL files")
                for file_path in jsonl_gz_files:
                    print(f"Processing: {file_path}")
                    try:
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                try:
                                    doc = json.loads(line.strip())
                                    if 'text' in doc:
                                        documents.append({'text': doc['text']})
                                except json.JSONDecodeError as e:
                                    print(f"Warning: JSON decode error at line {line_num} in {file_path}: {e}")
                                    continue
                    except Exception as e:
                        print(f"Warning: Could not read {file_path}: {e}")
                        continue

        # Look for uncompressed .jsonl files
        if not documents:
            jsonl_files = list(Path(output_path).glob("**/*.jsonl"))
            if jsonl_files:
                print(f"Found {len(jsonl_files)} JSONL files")
                for file_path in jsonl_files:
                    print(f"Processing: {file_path}")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                try:
                                    doc = json.loads(line.strip())
                                    if 'text' in doc:
                                        documents.append({'text': doc['text']})
                                except json.JSONDecodeError as e:
                                    print(f"Warning: JSON decode error at line {line_num} in {file_path}: {e}")
                                    continue
                    except Exception as e:
                        print(f"Warning: Could not read {file_path}: {e}")
                        continue

        # Look for other JSON formats as fallback
        if not documents:
            print("No JSONL/Parquet files found, looking for other JSON formats...")
            for file_path in Path(output_path).glob("**/*.json"):
                print(f"Processing JSON file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'text' in item:
                                    documents.append({'text': item['text']})
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
                    continue

        if not documents:
            # List what files we actually found
            all_files = list(Path(output_path).glob("**/*"))
            print(f"No valid documents found. Files in output directory:")
            for f in all_files:
                print(f"  - {f}")
            raise ValueError(f"No valid documents found in {output_path}")

        print(f"Successfully loaded {len(documents)} documents")
        return Dataset.from_list(documents)


def create_datatrove_example_config() -> Dict[str, Any]:
    """Create an example datatrove configuration."""
    return {
        "deduplication_algorithm": "minhash",
        "min_length": 100,
        "max_length": 100000,
        "use_trafilatura": False,
        "collect_stats": True,
        "cleanup_temp": True,
        "streaming": True,
        "num_workers": max(1, os.cpu_count() // 2),
        "minhash_threshold": 0.8
    }


def enhanced_text_cleaning(text: str) -> str:
    """
    Enhanced text cleaning that mimics datatrove's text processing.
    Used as fallback when datatrove is not available.
    """
    if not isinstance(text, str):
        return ""

    # Basic cleaning similar to original implementation
    text = text.strip()

    # UTF-8 validation
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception:
        return ""

    # Remove surrogate characters
    import re
    text = re.sub(r'[\uD800-\uDFFF]', '', text)

    # Additional cleaning inspired by datatrove
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove very short lines (likely noise)
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip()) > 10]
    text = '\n'.join(cleaned_lines)

    return text
