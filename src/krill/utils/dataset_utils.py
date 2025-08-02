"""
Utility functions for dataset loading.
"""

import os
import re
from datasets import load_dataset, concatenate_datasets


def clean_text(text):
    """
    Clean and normalize text data.
    """
    if not isinstance(text, str):
        return ""  # 문자열이 아니면 빈 문자열 반환하거나 오류 처리

    # 문자열 양 끝의 공백 제거
    text = text.strip()

    # 0-1. UTF-8 유효성을 강제로 확인 및 유효하지 않은 문자 제거 (추가된 부분)
    # 이 과정에서 유효하지 않은 UTF-8 바이트 시퀀스가 제거됩니다.
    # 즉, 파이썬 문자열 내부에서 UTF-8로 다시 인코딩될 수 없는 문자를 제거합니다.
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception as e:
        # 이 예외는 이론적으로 발생하지 않아야 하지만, 만약을 위해 로깅합니다.
        print(
            f"Warning: Error during UTF-8 re-encoding/decoding: {e}. Original text: {text[:50]}...")
        text = ""  # 오류 발생 시 해당 텍스트를 비움

    # 1-6. fix UnicodeEncodeError: 'utf-8' codec can't encode character '\udd2b' in position 2095: surrogates not allowed
    text = re.sub(r'[\uD800-\uDFFF]', '', text)

    return text


# 중복 제거를 위한 전역 세트(set) 선언
seen_texts = set()


def is_high_quality_and_unique(example):
    """
    품질 필터링(길이)과 중복 제거를 동시에 수행하는 함수
    """
    text = example['text']

    # 2-1. 길이 필터링: 텍스트 길이가 100글자 미만이면 탈락
    if len(text) < 100:
        return False

    # 2-2. 중복 필터링: 이미 등장한 텍스트면 탈락
    if text in seen_texts:
        return False

    # 모든 필터를 통과한 경우, seen_texts에 추가하고 통과 처리
    seen_texts.add(text)
    return True


def load_and_prepare_raw_datasets(dataset_configs):
    """
    Load raw datasets, rename text column to 'text', drop other columns, concatenate,
    and apply text cleaning, quality filtering, and deduplication.
    Each config should have attributes 'path', 'split', and 'text_column'.
    """
    # 1. Load and concatenate raw datasets
    raw_datasets = []
    for ds_cfg in dataset_configs:
        print(
            f"Loading dataset {ds_cfg.path} columns={getattr(ds_cfg, 'text_column', 'text')} split={ds_cfg.split}...")
        ds = load_dataset(ds_cfg.path, split=ds_cfg.split)
        if getattr(ds_cfg, 'text_column', 'text') != 'text':
            ds = ds.rename_column(ds_cfg.text_column, 'text')
        # Drop all columns except 'text'
        ds = ds.remove_columns(
            [col for col in ds.column_names if col != 'text'])
        raw_datasets.append(ds)

    if not raw_datasets:
        raise ValueError("No datasets to load.")

    if len(raw_datasets) > 1:
        combined_dataset = concatenate_datasets(raw_datasets)
    else:
        combined_dataset = raw_datasets[0]

    print(
        f"Combined dataset total rows: {combined_dataset.num_rows / 1_000_000:.2f}M")

    # 2. Text cleaning and normalization
    num_processors = max(1, os.cpu_count() - 8)
    print(
        f"Total CPUs: {os.cpu_count()}, Using {num_processors} processes for mapping.")

    print("\n2. 텍스트 정제 및 정규화를 시작합니다... (.map)")
    cleaned_dataset = combined_dataset.map(
        lambda example: {'text': clean_text(example['text'])},
        num_proc=num_processors,
    )

    print(
        f"Cleaned dataset total rows: {cleaned_dataset.num_rows / 1_000_000:.2f}M")

    # 3. Quality filtering and deduplication
    # 전역 seen_texts 세트 초기화
    global seen_texts
    seen_texts = set()

    print("\n3. 품질 및 중복 필터링을 시작합니다... (.filter)")
    final_dataset = cleaned_dataset.filter(
        is_high_quality_and_unique,
        # 'seen_texts' 세트는 전역 변수이므로 다중 처리(num_proc > 1) 시 충돌할 수 있습니다.
        num_proc=1
        # 대용량 데이터 처리 시에는 다른 중복 제거 방식이 필요할 수 있습니다.
    )

    print(
        f"Final dataset total rows: {final_dataset.num_rows / 1_000_000:.4f}M")

    return final_dataset
