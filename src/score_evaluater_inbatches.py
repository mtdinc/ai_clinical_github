#score_evaluater_inbatches
import pandas as pd
import os
import anthropic
import asyncio
import aiohttp
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Score LLM responses for accuracy of diagnoses.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., cases_typical_concise_selected)')
    parser.add_argument('--stage', type=str, required=True,
                        help='Stage name (e.g., stage_1)')
    return parser.parse_args()

# Initialize the Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

async def get_top_k_score_async(client, true_dx, llm_response, k_value):
    """
    Evaluate if the true diagnosis is within the top k diagnoses in the differential list.
    
    Args:
        client: Anthropic client
        true_dx: The true diagnosis
        llm_response: The LLM's response containing the differential diagnosis list
        k_value: The k value for top-k accuracy (1, 5, or 10)
        
    Returns:
        1 if the true diagnosis is within the top k diagnoses, 0 otherwise
    """
    prompt = f"""You are an expert medical evaluator.
The True Diagnosis for a clinical case is: "{true_dx}"

An LLM provided the following response, which includes a differential diagnosis list:
"{llm_response}"

Your task is to:
1. Identify the ordered list of differential diagnoses presented by the LLM. The diagnoses are often numbered or clearly demarcated (e.g., "Diagnosis 1: ...", "Diagnosis 2: ...", or a bulleted/numbered list).
2. Consider only the **first {k_value} (top {k_value})** diagnoses from this list.
3. Determine if the True Diagnosis ("{true_dx}") is mentioned or clinically related to **any** of these top {k_value} diagnoses.
4. Use the following criteria for a match (resulting in a score of 1 if any of the top {k_value} diagnoses meet these criteria relative to the True Diagnosis):
   - Exact match of the True Diagnosis.
   - Clinically related diagnosis that includes the pathophysiological process of the True Diagnosis.
   - Same disease category or family of disorders as the True Diagnosis.
   - Correct anatomical location with similar pathology to the True Diagnosis.
   - Correct system involvement even if specific diagnosis differs, but is clearly related to the True Diagnosis.
5. If none of the top {k_value} diagnoses meet these criteria when compared to the True Diagnosis, score 0 points.

Respond with only a single numeric score (0 or 1). Do not provide any explanation."""

    try:
        response = await client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=100,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Get the response text and extract the score
        response_text = response.content[0].text.strip()
        # Extract the first number found in the response
        score = int(''.join(filter(str.isdigit, response_text)))
        
        # Validate score
        if score not in [0, 1]:
            print(f"Invalid score received: {score}")
            return None
            
        return score
    except anthropic.RateLimitError:
        # Add delay only when we hit rate limits
        print("Rate limit hit, waiting 30 seconds...")
        await asyncio.sleep(30)
        return await get_top_k_score_async(client, true_dx, llm_response, k_value)
    except Exception as e:
        print(f"Error getting score: {e}")
        print(f"True diagnosis: {true_dx}")
        print(f"LLM response: {llm_response[:200]}...")  # Print just the beginning to avoid overwhelming logs
        return None

async def process_row_async(client, row, idx, total):
    print(f"\nProcessing row {idx + 1}/{total}")
    try:
        # Get top-k scores concurrently for k=1, k=5, and k=10
        score_k1, score_k5, score_k10 = await asyncio.gather(
            get_top_k_score_async(client, row['true_dx'], row['response'], 1),
            get_top_k_score_async(client, row['true_dx'], row['response'], 5),
            get_top_k_score_async(client, row['true_dx'], row['response'], 10)
        )
        print(f"Top-1 accuracy: {score_k1}")
        print(f"Top-5 accuracy: {score_k5}")
        print(f"Top-10 accuracy: {score_k10}")
        return idx, score_k1, score_k5, score_k10
    except Exception as e:
        print(f"Error processing row {idx + 1}: {e}")
        return idx, None, None, None

async def process_csv_async(input_file, output_file, batch_size=10):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Print column names for debugging
    print("Column names:", df.columns.tolist())
    
    # Initialize score columns
    df['differential_top_1_accuracy'] = None
    df['differential_top_5_accuracy'] = None
    df['differential_top_10_accuracy'] = None
    
    # Process rows in batches
    async with anthropic.AsyncAnthropic(api_key=client.api_key) as async_client:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            tasks = [
                process_row_async(async_client, row, idx, len(df))
                for idx, row in batch.iterrows()
            ]
            results = await asyncio.gather(*tasks)
            
            # Update dataframe with results
            for idx, score_k1, score_k5, score_k10 in results:
                df.at[idx, 'differential_top_1_accuracy'] = score_k1
                df.at[idx, 'differential_top_5_accuracy'] = score_k5
                df.at[idx, 'differential_top_10_accuracy'] = score_k10
            
            # Save progress after each batch
            df.to_csv(output_file, index=False)
            print(f"\nBatch completed. Progress saved to {output_file}")
    
    print(f"\nProcessing complete. Final results saved to {output_file}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    dataset = args.dataset
    stage = args.stage
    
    # Construct the file paths
    stage_dataset = f"{dataset}_{stage}"
    input_file = f"Results_typical_{stage_dataset}_long_format_ordered.csv"
    output_file = f"Results_typical_{stage_dataset}_long_format_ordered_top_k_rated.csv"
    
    print(f"Processing dataset: {dataset}, stage: {stage}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Run the async process
    asyncio.run(process_csv_async(input_file, output_file))

if __name__ == "__main__":
    main()
