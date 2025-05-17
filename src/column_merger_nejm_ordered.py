import pandas as pd
import re
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Merge LLM response files and create long format data.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., cases_typical_concise_selected)')
    parser.add_argument('--stage', type=str, required=True,
                        help='Stage name (e.g., stage_1)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    dataset = args.dataset
    stage = args.stage
    stage_dataset = f"{dataset}_{stage}"
    
    print(f"Starting merge process for dataset: {dataset}, stage: {stage}...")

    # Read OpenAI responses
    df_openai = pd.read_csv(f'llm_openai_typical_responses_{stage_dataset}.csv')
    df_openai['case_number'] = df_openai['case_id'].astype(str).str.extract(r'(\d+)').astype(int)
    df_openai = df_openai.sort_values('case_number')

    # Read OpenAI O1 responses
    df_openai_o1 = pd.read_csv(f'llm_openai_o1_typical_responses_{stage_dataset}.csv')
    df_openai_o1['case_number'] = df_openai_o1['case_id'].astype(str).str.extract(r'(\d+)').astype(int)
    df_openai_o1 = df_openai_o1.sort_values('case_number')

    # Read Anthropic responses
    df_anthropic = pd.read_csv(f'llm_anthropic_typical_responses_{stage_dataset}.csv')
    df_anthropic['case_number'] = df_anthropic['case_id'].astype(str).str.extract(r'(\d+)').astype(int)
    df_anthropic = df_anthropic.sort_values('case_number')

    # Read Google responses
    df_google = pd.read_csv(f'llm_google_typical_responses_{stage_dataset}.csv')
    df_google['case_number'] = df_google['case_id'].astype(str).str.extract(r'(\d+)').astype(int)
    df_google = df_google.sort_values('case_number')

    # Initialize merged_df with OpenAI responses
    merged_df = df_openai.copy()

    # Add stage column after medical_case_text
    columns = merged_df.columns.tolist()
    medical_case_text_index = columns.index('medical_case_text')
    merged_df.insert(medical_case_text_index + 1, 'stage', f'stage-{stage.replace("stage_", "")}_{dataset}')

    # Read and merge true diagnosis
    try:
        print("\nAdding true diagnosis from case list...")
        # Generate case list file path based on dataset
        case_list_file = f"{dataset}_case_list.csv"
        print(f"Looking for case list file: {case_list_file}")
    
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
        case_list_df = None
        
        for encoding in encodings:
            try:
                case_list_df = pd.read_csv(case_list_file, encoding=encoding)
                print(f"Successfully read case list with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                print(f"File not found: {case_list_file}")
                # Try alternative file paths
                alternative_paths = [
                    f"{dataset}/cases_typical_case_list.csv", 
                    f"{dataset}/case_list.csv", 
                    f"{dataset}/cases_CPS_case_list.csv", 
                    f"{dataset}/{dataset}_case_list.csv",
                    f"{dataset}/cases_{dataset}_case_list.csv",
                    f"case_List_Ai_{dataset}.csv",
                    f"data/case_List_Ai_{dataset}.csv"
                ]
                for alt_path in alternative_paths:
                    print(f"Trying alternative path: {alt_path}")
                    try:
                        case_list_df = pd.read_csv(alt_path, encoding=encoding)
                        print(f"Successfully read case list from {alt_path} with encoding: {encoding}")
                        break
                    except (FileNotFoundError, UnicodeDecodeError):
                        continue
            if case_list_df is not None:
                break
    
        if case_list_df is None:
            raise Exception("Could not read case list file with any encoding")
            
        # Convert case_id to numeric for merging
        case_list_df['case_number'] = pd.to_numeric(case_list_df['case_id'])
        
        stage_index = merged_df.columns.get_loc('stage')
        merged_df = merged_df.merge(
            case_list_df[['case_number', 'true_dx']],
            on='case_number',
            how='left'
        )
        
        # Reorder columns to put true_dx after stage
        cols = merged_df.columns.tolist()
        true_dx_index = cols.index('true_dx')
        cols.pop(true_dx_index)
        cols.insert(stage_index + 1, 'true_dx')
        merged_df = merged_df[cols]
        
        # Verify true_dx column exists
        if 'true_dx' not in merged_df.columns:
            raise Exception("true_dx column not found after merge")
            
    except Exception as e:
        print(f"Warning: Could not add true diagnosis: {str(e)}")
        # Add a placeholder true_dx column if it doesn't exist
        if 'true_dx' not in merged_df.columns:
            print("Adding placeholder true_dx column")
            stage_index = merged_df.columns.get_loc('stage')
            merged_df.insert(stage_index + 1, 'true_dx', "Unknown")

    # Add Anthropic responses
    anthropic_response_cols = [col for col in df_anthropic.columns if 'llm_response' in col]
    merged_df = merged_df.merge(
        df_anthropic[['case_number'] + anthropic_response_cols],
        on='case_number',
        how='outer'
    )

    # Add OpenAI O1 responses
    openai_o1_response_cols = [col for col in df_openai_o1.columns if 'llm_response' in col]
    merged_df = merged_df.merge(
        df_openai_o1[['case_number'] + openai_o1_response_cols],
        on='case_number',
        how='outer'
    )

    # Add Google responses
    google_response_cols = [col for col in df_google.columns if 'llm_response' in col]
    merged_df = merged_df.merge(
        df_google[['case_number'] + google_response_cols],
        on='case_number',
        how='outer'
    )

    # Sort by case number
    merged_df = merged_df.sort_values('case_number')

    # Save the merged wide format
    merged_df.to_csv(f'merged_llm_responses_{stage_dataset}_ordered.csv', index=False)
    print(f"\nMerged results saved to merged_llm_responses_{stage_dataset}_ordered.csv")

    # Convert to long format
    print("\nConverting to long format...")
    response_columns = [col for col in merged_df.columns if 'llm_response' in col]

    # Create long format
    long_df = pd.melt(
        merged_df,
        id_vars=['case_number', 'case_id', 'medical_case_text', 'stage', 'true_dx'],
        value_vars=response_columns,
        var_name='model',
        value_name='response'
    )

    # Clean up model names by removing prefix
    long_df['model'] = long_df['model'].str.replace('llm_response_', '')

    # Sort by case_number
    long_df = long_df.sort_values('case_number')

    # Save long format
    long_df.to_csv(f'Results_typical_{stage_dataset}_long_format_ordered.csv', index=False)

    print(f"\nLong format results saved to Results_typical_{stage_dataset}_long_format_ordered.csv")
    print(f"Total rows in long format: {len(long_df)}")
    print(f"Total columns in long format: {len(long_df.columns)}")
    print("\nColumns in long format file:")
    for i, col in enumerate(long_df.columns):
        print(f"{i+1}. {col}")

if __name__ == "__main__":
    main()
