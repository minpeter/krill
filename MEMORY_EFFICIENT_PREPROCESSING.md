# Memory-Efficient Preprocessing

This document describes the memory-efficient preprocessing features added to Krill to handle large datasets that don't fit in memory.

## Problem

The original `preprocess` command had several memory inefficiencies:

1. **Full Dataset Loading**: Entire datasets were loaded into memory at once
2. **Global Deduplication Set**: A global `seen_texts` set stored all text strings in memory
3. **Batch Processing**: Tokenization processed the entire dataset in memory
4. **Pack Dataset**: The `pack_dataset()` function used `batch_size=len(tokenized)`, loading everything at once
5. **Length Filtering**: Created full lists of lengths in memory

These issues made preprocessing impossible on systems with limited memory when working with large datasets.

## Solution

The memory-efficient preprocessing mode addresses these issues through:

### 1. Chunked Dataset Processing

- Processes datasets in configurable chunks instead of loading everything at once
- Default chunk size: 10,000 samples (configurable via `preprocess_chunk_size`)
- Each chunk is processed independently and then concatenated

### 2. File-Based Deduplication

- Replaces the global in-memory `seen_texts` set with a file-based hash storage system
- Uses MD5 hashes of text content stored in organized directory structure
- Significantly reduces memory usage for deduplication on large datasets
- Cache can be persisted between runs using `preprocess_dedup_cache_dir`

### 3. Chunked Tokenization

- Tokenizes data in smaller batches to reduce peak memory usage
- Processes tokenization chunk by chunk, then concatenates results

### 4. Memory-Efficient Length Filtering

- Filters by length without creating full length lists in memory
- Processes length filtering in chunks

### 5. Smaller Pack Dataset Batch Size

- Uses configurable batch size for `pack_dataset()` instead of full dataset size
- Reduces memory pressure during sequence packing

## Configuration

Add these options to your YAML configuration to enable memory-efficient preprocessing:

```yaml
# Enable memory-efficient preprocessing
preprocess_memory_efficient: true

# Set chunk size (default: 10000)
preprocess_chunk_size: 5000

# Optional: Set persistent cache directory for deduplication
preprocess_dedup_cache_dir: ./cache/dedup

# Your existing config...
sequence_len: 2048
datasets:
  - path: your/large/dataset
    split: train
    text_column: text
```

## Memory Monitoring

The preprocessor includes built-in memory monitoring that reports memory usage at each stage:

```
📊 Memory monitoring started. Initial: 145.2 MB
📊 Memory before loading datasets: 145.8 MB (peak: 145.8 MB)
📊 Memory after chunked dataset processing: 167.3 MB (peak: 189.1 MB)
📊 Memory after chunked tokenization: 234.7 MB (peak: 267.4 MB)
📊 Memory Summary:
   - Start: 145.2 MB
   - Peak: 267.4 MB
   - Final: 198.6 MB
   - Peak increase: 122.2 MB
```

## Performance Comparison

| Mode | Memory Usage | Processing Speed | Scalability |
|------|-------------|------------------|-------------|
| Standard | High (full dataset in memory) | Faster | Limited by available RAM |
| Memory-Efficient | Low (chunk-sized) | Slightly slower | Scales to very large datasets |

## When to Use

### Use Memory-Efficient Mode When:
- Working with datasets larger than available RAM
- Running on memory-constrained environments
- Processing very large corpora (>10GB)
- Want to avoid out-of-memory errors

### Use Standard Mode When:
- Dataset fits comfortably in memory
- Maximum processing speed is required
- Working with smaller datasets (<1GB)

## Examples

### Basic Memory-Efficient Usage

```bash
# Create config with memory-efficient settings
cat > config.yaml << EOF
preprocess_memory_efficient: true
preprocess_chunk_size: 1000
sequence_len: 2048
datasets:
  - path: large/dataset
    split: train
    text_column: text
# ... other config
EOF

# Run preprocessing
krill preprocess config.yaml
```

### With Persistent Deduplication Cache

```bash
# Create config with cache directory
cat > config.yaml << EOF
preprocess_memory_efficient: true
preprocess_chunk_size: 5000
preprocess_dedup_cache_dir: /tmp/dedup_cache
# ... other config
EOF

# Run preprocessing (cache will be preserved)
krill preprocess config.yaml
```

## Technical Details

### File-Based Deduplication

The deduplication system uses a two-level directory structure:

```
dedup_cache/
├── a1/
│   ├── a1b2c3d4e5f6...
│   └── a1x9y8z7w6v5...
├── a2/
│   └── a2m3n4o5p6q7...
└── ...
```

- First 2 characters of MD5 hash form directory name
- Full hash becomes filename
- Empty files are used as existence markers
- In-memory set caches recently seen hashes for speed

### Memory Usage Patterns

Memory-efficient mode typically shows:
- Lower peak memory usage
- More consistent memory usage across processing stages
- Gradual memory increase rather than sudden spikes
- Better garbage collection opportunities between chunks

## Limitations

1. **Slightly Slower**: Processing in chunks adds some overhead
2. **Disk I/O**: File-based deduplication increases disk usage
3. **Chunk Boundaries**: Very small chunks may reduce efficiency
4. **Single-Process Deduplication**: File-based deduplication requires `num_proc=1` for consistency

## Future Improvements

Potential enhancements for even better memory efficiency:

1. **Streaming Datasets**: Use Hugging Face streaming datasets for even lower memory usage
2. **Bloom Filters**: Use probabilistic data structures for approximate deduplication
3. **Compression**: Compress intermediate results to reduce memory usage
4. **Database Backend**: Use SQLite or other databases for deduplication tracking
5. **Memory Limits**: Automatic chunk size adjustment based on available memory