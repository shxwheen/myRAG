"""Utilities for extracting and using metadata from questions."""

import re
from typing import Dict, List, Optional


def extract_metadata_from_question(question: str) -> Dict[str, any]:
    """Extract company name, year, and document type from question.

    Args:
        question: Question text

    Returns:
        Dict with extracted metadata: company, years, doc_type
    """
    metadata = {
        'companies': [],
        'years': [],
        'doc_types': []
    }

    # Common company names in FinanceBench
    companies = [
        '3M', 'Adobe', 'Apple', 'Microsoft', 'Amazon', 'Netflix',
        'Oracle', 'Block', 'Square', 'Costco', 'CVS', 'AES',
        'Activision Blizzard', 'American Express', 'Best Buy',
        'Coca-Cola', 'Boeing', 'Pfizer', 'Walmart'
    ]

    # Extract companies (case insensitive)
    question_lower = question.lower()
    for company in companies:
        if company.lower() in question_lower:
            metadata['companies'].append(company)

    # Extract years - use comprehensive pattern that captures all formats
    # and prioritizes 4-digit years to avoid FY2022 being parsed as FY20 (year 2020)
    years_found = set()

    # First pass: Match FY + 4-digit year (FY2019, FY 2019)
    fy_4digit = re.findall(r'FY\s?(\d{4})', question, re.IGNORECASE)
    for year_str in fy_4digit:
        years_found.add(int(year_str))

    # Second pass: Match standalone 4-digit years (2019, 2020)
    standalone_4digit = re.findall(r'\b(20\d{2})\b', question, re.IGNORECASE)
    for year_str in standalone_4digit:
        years_found.add(int(year_str))

    # Third pass: Match FY + 2-digit year ONLY if no 4-digit FY year was found
    if not fy_4digit:
        fy_2digit = re.findall(r'FY\s?(\d{2})', question, re.IGNORECASE)
        for year_str in fy_2digit:
            year = int(year_str)
            full_year = 2000 + year
            years_found.add(full_year)

    # Fourth pass: Match abbreviated years ('19, '20)
    abbreviated = re.findall(r"'(\d{2})\b", question)
    for year_str in abbreviated:
        year = int(year_str)
        full_year = 2000 + year
        years_found.add(full_year)

    metadata['years'] = sorted(list(years_found))

    # Extract document types
    if any(term in question_lower for term in ['10-k', '10k', 'annual report']):
        metadata['doc_types'].append('10k')
    if any(term in question_lower for term in ['10-q', '10q', 'quarterly']):
        metadata['doc_types'].append('10q')
    if any(term in question_lower for term in ['8-k', '8k']):
        metadata['doc_types'].append('8k')

    return metadata


def filter_chunks_by_metadata(chunks: List, metadata: Dict) -> List:
    """Filter retrieved chunks based on extracted metadata.

    Args:
        chunks: List of Document objects from retriever
        metadata: Extracted metadata from question

    Returns:
        Filtered list of chunks
    """
    if not chunks:
        return chunks

    filtered = []

    for chunk in chunks:
        chunk_meta = chunk.metadata
        source = chunk_meta.get('source', '').lower()

        # Check if chunk matches company filter
        company_match = True
        if metadata['companies']:
            company_match = any(
                company.lower() in source
                for company in metadata['companies']
            )

        # Check if chunk matches year filter
        year_match = True
        if metadata['years']:
            year_match = any(
                str(year) in source
                for year in metadata['years']
            )

        # Check if chunk matches doc type filter
        doctype_match = True
        if metadata['doc_types']:
            doctype_match = any(
                dtype in source
                for dtype in metadata['doc_types']
            )

        # Include chunk if it matches all filters
        if company_match and year_match and doctype_match:
            filtered.append(chunk)

    # If filtering removed everything, return original chunks
    # (better to have wrong context than no context)
    if not filtered:
        return chunks

    return filtered
