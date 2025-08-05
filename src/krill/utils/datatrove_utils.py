"""
Datatrove integration utilities for enhanced preprocessing.
"""
import os
import warnings
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path

from datasets import Dataset
from krill.utils.config import DatatroveConfig, DatasetConfig

# Try to import datatrove, with graceful fallback
try:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import HuggingFaceDatasetReader
    from datatrove.pipeline.writers import HuggingFaceDatasetWriter
    # from datatrove.pipeline.filters import SampleFilter
    # from datatrove.pipeline.dedup import MinhashDedupFilter, ExactDedupFilter
    from datatrove.pipeline.dedup import MinhashDedupFilter
    from datatrove.pipeline.extractors import Trafilatura
    # from datatrove.pipeline.stats import WordsCounter, CharsCounter
    from datatrove.data import Document
    DATATROVE_AVAILABLE = True
except ImportError:
    DATATROVE_AVAILABLE = False
    warnings.warn(
        "Datatrove not available. Install with: pip install 'krill[datatrove]' "
        "or pip install datatrove>=0.2.0. Falling back to current preprocessing.",
        ImportWarning
    )


class DatatrovePreprocessor:
    """
    Datatrove-based preprocessor that provides enhanced text processing capabilities.
    """

    def __init__(self, config: DatatroveConfig):
        if not DATATROVE_AVAILABLE:
            raise ImportError(
                "Datatrove is not available. Install with: pip install 'krill[datatrove]' "
                "or pip install datatrove>=0.2.0"
            )

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
                streaming=self.config.streaming
            )
            pipeline_steps.append(reader)

        # 2. Text extractor and cleaner (using Trafilatura for robust text extraction)
        if self.config.quality_filters.get('use_trafilatura', False):
            extractor = Trafilatura()
            pipeline_steps.append(extractor)

        # 3. Quality filters
        min_length = self.config.quality_filters.get('min_length', 100)
        max_length = self.config.quality_filters.get('max_length', None)

        # Custom quality filter
        class QualityFilter(SampleFilter):
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

        # 4. Deduplication
        if self.config.deduplication_algorithm == "minhash":
            deduplicator = MinhashDedupFilter(
                threshold=self.config.minhash_threshold,
                store_fingerprints=True
            )
        elif self.config.deduplication_algorithm == "exact":
            deduplicator = ExactDedupFilter()
        else:
            raise ValueError(
                f"Unknown deduplication algorithm: {self.config.deduplication_algorithm}")

        pipeline_steps.append(deduplicator)

        # 5. Statistics collectors (optional)
        if self.config.quality_filters.get('collect_stats', False):
            pipeline_steps.extend([
                WordsCounter(),
                CharsCounter()
            ])

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
        writer = HuggingFaceDatasetWriter(
            output_folder=temp_output,
            compression=None
        )
        pipeline_steps.append(writer)

        # Execute pipeline
        executor = LocalPipelineExecutor(
            pipeline=pipeline_steps,
            logging_dir=os.path.join(output_path, "logs"),
            workers=self.config.num_workers if self.config.num_workers > 1 else 1
        )

        print(f"📊 Running pipeline with {self.config.num_workers} workers...")
        executor.run()

        # Load the processed dataset
        print("📖 Loading processed dataset...")
        try:
            from datasets import load_from_disk
            processed_dataset = load_from_disk(temp_output)
        except Exception as e:
            print(f"Warning: Could not load from disk cache: {e}")
            # Fallback: create dataset from files
            processed_dataset = self._load_from_datatrove_output(temp_output)

        # Clean up temporary files if needed
        if self.config.quality_filters.get('cleanup_temp', True):
            import shutil
            try:
                shutil.rmtree(temp_output)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")

        print(f"✅ Datatrove preprocessing complete!")
        print(f"   Final dataset size: {len(processed_dataset)} samples")

        return processed_dataset

    def _load_from_datatrove_output(self, output_path: str) -> Dataset:
        """Fallback method to load dataset from datatrove output files."""
        import json
        from pathlib import Path

        documents = []

        # Look for .jsonl files in the output directory
        for file_path in Path(output_path).glob("**/*.jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line.strip())
                        if 'text' in doc:
                            documents.append({'text': doc['text']})
                    except json.JSONDecodeError:
                        continue

        if not documents:
            # Look for other formats
            for file_path in Path(output_path).glob("**/*.json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'text' in item:
                                    documents.append({'text': item['text']})
                    except json.JSONDecodeError:
                        continue

        if not documents:
            raise ValueError(f"No valid documents found in {output_path}")

        return Dataset.from_list(documents)


def is_datatrove_available() -> bool:
    """Check if datatrove is available for use."""
    return DATATROVE_AVAILABLE


def create_datatrove_example_config() -> Dict[str, Any]:
    """Create an example datatrove configuration."""
    return {
        "datatrove": {
            "enabled": True,
            "deduplication_algorithm": "minhash",
            "quality_filters": {
                "min_length": 100,
                "max_length": 100000,
                "use_trafilatura": False,
                "collect_stats": True,
                "cleanup_temp": True
            },
            "distributed": False,
            "streaming": True,
            "num_workers": max(1, os.cpu_count() // 2),
            "minhash_threshold": 0.8
        }
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
