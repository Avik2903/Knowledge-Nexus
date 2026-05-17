"""Document processing module for PDF, TXT, and CSV files."""

import os
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
import pandas as pd


class DocumentProcessor:
    """Processes various document formats into text chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_file(self, file_path: str, original_filename: str = None) -> List[Dict[str, Any]]:
        """Process a file and return chunks with metadata."""
        # Use original filename if provided, otherwise fall back to temp filename
        filename = original_filename if original_filename else Path(file_path).name
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            return self._process_pdf(file_path, filename)
        elif ext == '.txt':
            return self._process_txt(file_path, filename)
        elif ext == '.csv':
            return self._process_csv(file_path, filename)
        elif ext in ['.xlsx', '.xls']:
            return self._process_excel(file_path, filename)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _process_pdf(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract text from PDF and chunk it."""
        chunks = []

        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        full_text += f"\n--- Page {page_num + 1} ---\n{text}"

            text_chunks = self._chunk_text(full_text)
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'text': chunk,
                    'source': filename,
                    'metadata': {
                        'file_type': 'pdf',
                        'file_name': filename,
                        'chunk_index': i,
                        'source': filename,
                        'page': i + 1
                    }
                })
        except Exception as e:
            print(f"Error processing PDF {filename}: {e}")

        return chunks

    def _process_txt(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Read TXT file and chunk it."""
        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            text_chunks = self._chunk_text(text)
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'text': chunk,
                    'source': filename,
                    'metadata': {
                        'file_type': 'txt',
                        'file_name': filename,
                        'chunk_index': i,
                        'source': filename
                    }
                })
        except Exception as e:
            print(f"Error processing TXT {filename}: {e}")

        return chunks

    def _process_csv(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Parse CSV and convert rows to text chunks."""
        chunks = []

        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            total_rows = len(df)
            text_parts = []

            text_parts.append(f"CSV File: {filename}")
            text_parts.append(f"Columns: {', '.join(df.columns)}")
            text_parts.append(f"Total Rows: {total_rows}")
            text_parts.append("")

            for idx, row in df.iterrows():
                row_text = f"Row {idx + 1}: "
                for col in df.columns:
                    val = row[col]
                    if pd.isna(val):
                        val = ""
                    row_text += f"{col}={val}; "
                text_parts.append(row_text)

            full_text = "\n".join(text_parts)
            text_chunks = self._chunk_text(full_text)

            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'text': chunk,
                    'source': filename,
                    'metadata': {
                        'file_type': 'csv',
                        'file_name': filename,
                        'chunk_index': i,
                        'source': filename
                    }
                })
        except Exception as e:
            print(f"Error processing CSV {filename}: {e}")

        return chunks

    def _process_excel(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Parse Excel file and convert to text chunks."""
        chunks = []

        try:
            # Read all sheets from the Excel file
            excel_file = pd.ExcelFile(file_path)
            total_rows = 0
            text_parts = []

            text_parts.append(f"Excel File: {filename}")
            text_parts.append(f"Sheets: {', '.join(excel_file.sheet_names)}")
            text_parts.append("")

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheet_rows = len(df)
                total_rows += sheet_rows

                text_parts.append(f"=== Sheet: {sheet_name} ===")
                text_parts.append(f"Columns: {', '.join(df.columns)}")
                text_parts.append(f"Rows: {sheet_rows}")
                text_parts.append("")

                # Add column statistics for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    text_parts.append("Numeric Columns Summary:")
                    for col in numeric_cols:
                        text_parts.append(f"  {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")
                    text_parts.append("")

                # Add data rows
                for idx, row in df.iterrows():
                    row_text = f"Row {idx + 1}: "
                    for col in df.columns:
                        val = row[col]
                        if pd.isna(val):
                            val = ""
                        row_text += f"{col}={val}; "
                    text_parts.append(row_text)

                text_parts.append("")

            text_parts.append(f"Total Rows across all sheets: {total_rows}")

            full_text = "\n".join(text_parts)
            text_chunks = self._chunk_text(full_text)

            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'text': chunk,
                    'source': filename,
                    'metadata': {
                        'file_type': 'excel',
                        'file_name': filename,
                        'chunk_index': i,
                        'source': filename,
                        'total_rows': total_rows
                    }
                })
        except Exception as e:
            print(f"Error processing Excel {filename}: {e}")

        return chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            return [text] if text.strip() else []

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks if chunks else [text]