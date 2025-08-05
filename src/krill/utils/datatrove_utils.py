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
from datatrove.pipeline.writers import JsonlWriter
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

    def create_pipeline_steps(self, dataset_configs: List[DatasetConfig]) -> List[Any]:
        """Create datatrove processing pipeline steps."""
        pipeline_steps = []

        # 1. Readers for each dataset
        for ds_config in dataset_configs:
            reader = HuggingFaceDatasetReader(
                dataset=ds_config.path,
                dataset_options={'split': ds_config.split},
                text_key=ds_config.text_column,
                streaming=self.config.get('streaming', True)
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

        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        temp_output = os.path.join(output_path, "datatrove_temp")

        # Create pipeline steps
        pipeline_steps = self.create_pipeline_steps(dataset_configs)

        # Add writer at the end
        writer = JsonlWriter(
            output_folder=temp_output,
            compression="gzip"
        )
        pipeline_steps.append(writer)

        # Execute pipeline
        executor = LocalPipelineExecutor(
            pipeline=pipeline_steps,
            logging_dir=os.path.join(output_path, "logs"),
            workers=self.config.get('num_workers', 1) if self.config.get('num_workers', 1) > 1 else 1
        )

        print(f"📊 Running pipeline with {self.config.get('num_workers', 1)} workers...")
        executor.run()

        # Load the processed dataset
        print("📖 Loading processed dataset...")
        try:
            # Try to load from JSONL files generated by JsonlWriter
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
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")

        print(f"✅ Datatrove preprocessing complete!")
        print(f"   Final dataset size: {len(processed_dataset)} samples")

        return processed_dataset

    def _load_from_datatrove_output(self, output_path: str) -> Dataset:
        """Load dataset from datatrove output files (JSONL format)."""
        import json
        import gzip
        from pathlib import Path

        documents = []
        
        print(f"Looking for output files in: {output_path}")

        # Look for .jsonl.gz files first (JsonlWriter with gzip compression)
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
            print("No JSONL files found, looking for other JSON formats...")
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
