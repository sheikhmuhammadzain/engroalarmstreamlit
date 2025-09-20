import pandas as pd
import os
import glob
from datetime import datetime
import csv

class CSVPreprocessor:
    """Advanced CSV Preprocessor for event log data with error handling"""
    
    def __init__(self, skip_rows=8):
        self.skip_rows = skip_rows
        self.processed_files = []
        
    def find_csv_files(self, directory="."):
        """Find all CSV files in the specified directory"""
        pattern = os.path.join(directory, "*.csv")
        csv_files = glob.glob(pattern)
        # Filter out already cleaned files
        csv_files = [f for f in csv_files if '_cleaned' not in f]
        return csv_files
    
    def detect_delimiter(self, file_path):
        """Detect the delimiter used in the CSV file"""
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            # Skip metadata rows
            for _ in range(self.skip_rows):
                f.readline()
            # Read header and a few data rows
            sample = f.read(1024)
            
        sniffer = csv.Sniffer()
        try:
            delimiter = sniffer.sniff(sample).delimiter
        except:
            delimiter = '\t'  # Default to tab if detection fails
            
        return delimiter
    
    def read_csv_flexible(self, file_path):
        """Read CSV with flexible column handling"""
        
        # Method 1: Try with error handling
        try:
            # Detect delimiter
            delimiter = self.detect_delimiter(file_path)
            # Fix: Move the tab check outside the f-string
            tab_char = '\t'
            delimiter_name = 'TAB' if delimiter == tab_char else repr(delimiter)
            print(f"  Detected delimiter: {delimiter_name}")
            
            # Try reading with pandas, ignoring bad lines
            df = pd.read_csv(
                file_path, 
                skiprows=self.skip_rows,
                delimiter=delimiter,
                on_bad_lines='warn',  # or 'skip' to silently skip bad lines
                engine='python'
            )
            return df
            
        except Exception as e:
            print(f"  Method 1 failed: {str(e)}")
            print("  Trying alternative method...")
            
        # Method 2: Read line by line and handle inconsistencies
        try:
            return self.read_csv_manual(file_path)
        except Exception as e:
            print(f"  Method 2 failed: {str(e)}")
            raise
    
    def read_csv_manual(self, file_path):
        """Manually read CSV handling inconsistent columns"""
        
        rows = []
        headers = None
        max_cols = 0
        
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            # Skip metadata rows
            for _ in range(self.skip_rows):
                f.readline()
            
            # Read the rest of the file
            reader = csv.reader(f, delimiter='\t')  # Try tab delimiter first
            
            for i, row in enumerate(reader):
                if i == 0:
                    # First row after skip is header
                    headers = row
                    max_cols = len(headers)
                else:
                    # Ensure all rows have same number of columns as header
                    if len(row) > max_cols:
                        # Truncate extra columns
                        row = row[:max_cols]
                    elif len(row) < max_cols:
                        # Pad with empty strings
                        row.extend([''] * (max_cols - len(row)))
                    rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)
        return df
    
    def clean_dataframe(self, df):
        """Apply various cleaning operations to the dataframe"""
        
        # 1. Clean column names
        df.columns = df.columns.str.strip()
        
        # 2. Remove empty rows
        df = df.replace('', pd.NA)  # Replace empty strings with NA
        df = df.dropna(how='all')   # Drop rows where all values are NA
        
        # 3. Clean string values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', 'NaN', '<NA>'], '')
        
        # 4. Handle specific columns if they exist
        if 'Value' in df.columns:
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        if 'Priority' in df.columns:
            df['Priority'] = pd.to_numeric(df['Priority'], errors='coerce')
        
        return df
    
    def extract_metadata(self, file_path):
        """Extract metadata from the first 8 rows"""
        metadata = {}
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()[:8]
                
            for line in lines:
                # Clean the line first
                tab_char = '\t'
                clean_line = line.split(tab_char)[0] if tab_char in line else line
                clean_line = clean_line.split(',')[0] if ',' in clean_line else clean_line
                
                if 'Date/Time of Report:' in clean_line:
                    parts = clean_line.split(':', 1)
                    if len(parts) > 1:
                        metadata['report_datetime'] = parts[1].strip()
                elif 'Requester:' in clean_line:
                    parts = clean_line.split(':', 1)
                    if len(parts) > 1:
                        metadata['requester'] = parts[1].strip()
                elif 'Filter Applied:' in clean_line:
                    parts = clean_line.split(':', 1)
                    if len(parts) > 1:
                        metadata['filter'] = parts[1].strip()
                elif 'Server:' in clean_line:
                    parts = clean_line.split(':', 1)
                    if len(parts) > 1:
                        metadata['server'] = parts[1].strip()
        except Exception as e:
            print(f"  Warning: Could not extract metadata: {e}")
        
        return metadata
    
    def process_file(self, file_path):
        """Process a single CSV file"""
        
        print(f"\n📄 Processing: {file_path}")
        
        try:
            # Extract metadata
            metadata = self.extract_metadata(file_path)
            if metadata:
                print("  Metadata extracted:")
                for key, value in metadata.items():
                    print(f"    - {key}: {value}")
            
            # Read CSV with flexible method
            df = self.read_csv_flexible(file_path)
            
            print(f"  Successfully read {len(df)} rows, {len(df.columns)} columns")
            
            # Clean the dataframe
            df = self.clean_dataframe(df)
            
            # Generate output filename
            base_name = os.path.splitext(file_path)[0]
            output_file = f"{base_name}_cleaned.csv"
            
            # Save cleaned CSV
            df.to_csv(output_file, index=False)
            
            # Store processing info
            self.processed_files.append({
                'input': file_path,
                'output': output_file,
                'rows': len(df),
                'columns': len(df.columns),
                'metadata': metadata
            })
            
            print(f"  ✅ Success!")
            print(f"    - Output: {output_file}")
            print(f"    - Rows after cleaning: {len(df)}")
            
            # Fix: Handle column display
            if len(df.columns) > 5:
                cols_display = ', '.join(df.columns[:5]) + "..."
            else:
                cols_display = ', '.join(df.columns)
            print(f"    - Columns: {cols_display}")
            
            return df
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_report(self):
        """Generate a summary report of all processed files"""
        
        if not self.processed_files:
            print("\nNo files were processed successfully.")
            return
        
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        for i, info in enumerate(self.processed_files, 1):
            print(f"\n{i}. {info['input']}")
            print(f"   → {info['output']}")
            print(f"   Rows: {info['rows']}, Columns: {info['columns']}")
            
    def run(self, directory=".", show_preview=True):
        """Main execution method"""
        
        print("="*60)
        print("CSV PREPROCESSING TOOL - EVENT LOG CLEANER")
        print("="*60)
        
        # Find CSV files
        csv_files = self.find_csv_files(directory)
        
        if not csv_files:
            print(f"\n❌ No CSV files found in {directory}")
            return
        
        print(f"\n📁 Found {len(csv_files)} CSV file(s) to process")
        
        # Process each file
        last_df = None
        for csv_file in csv_files:
            df = self.process_file(csv_file)
            if df is not None:
                last_df = df
        
        # Generate summary report
        self.generate_report()
        
        # Show preview if requested
        if show_preview and last_df is not None:
            response = input("\n📊 Show preview of cleaned data? (y/n): ")
            if response.lower() == 'y':
                print("\nFirst 10 rows of the last processed file:")
                print(last_df.head(10).to_string())
                
                print("\nColumn names:")
                for i, col in enumerate(last_df.columns, 1):
                    print(f"  {i}. {col}")
                
                print("\nData info:")
                print(f"  Total rows: {len(last_df)}")
                print(f"  Total columns: {len(last_df.columns)}")
                
                # Show non-null counts
                print("\nNon-null counts per column:")
                print(last_df.count())

# Alternative simple version if the advanced one still has issues
def simple_clean_csv(file_path, skip_rows=8):
    """Simple version that's more forgiving with errors"""
    
    print(f"Processing {file_path} with simple method...")
    
    # Read all lines from file
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    # Skip first 8 rows
    lines = lines[skip_rows:]
    
    # Parse header
    tab_char = '\t'
    header = lines[0].strip().split(tab_char)
    
    # Parse data rows
    data_rows = []
    for line in lines[1:]:
        row = line.strip().split(tab_char)
        # Ensure row has same number of columns as header
        if len(row) > len(header):
            row = row[:len(header)]
        elif len(row) < len(header):
            row.extend([''] * (len(header) - len(row)))
        data_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=header)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Save cleaned file
    output_file = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(output_file, index=False)
    
    print(f"✅ Saved to {output_file}")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    
    return df

# Main execution
if __name__ == "__main__":
    # Try the advanced preprocessor first
    preprocessor = CSVPreprocessor(skip_rows=8)
    preprocessor.run(directory=".", show_preview=True)
    
    # If that fails, uncomment below to use simple method:
    # import glob
    # csv_files = glob.glob("*.csv")
    # csv_files = [f for f in csv_files if '_cleaned' not in f]
    # for csv_file in csv_files:
    #     try:
    #         simple_clean_csv(csv_file)
    #     except Exception as e:
    #         print(f"Error processing {csv_file}: {e}")