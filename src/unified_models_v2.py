import os
import csv
import argparse
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

# Try to import absl.logging for warning suppression, but don't fail if not available
try:
    import absl.logging
    # Configure Abseil logging to reduce warnings
    absl.logging.set_verbosity(absl.logging.ERROR)
    print("Abseil logging configured to suppress warnings")
except ImportError:
    print("Note: absl module not found. Some Google API warnings may be displayed.")

# Import provider-specific clients
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai


class LLMProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def initialize_client(self):
        raise NotImplementedError
        
    def get_response(self, prompt: str, model_name: str) -> str:
        raise NotImplementedError

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Initialize the client once when the provider is created
        self.client = self.initialize_client()
        print("Anthropic client initialized once")
    
    def initialize_client(self):
        return Anthropic(api_key=self.api_key)
    
    def get_response(self, prompt: str, model_info: str or Dict) -> str:
        # Use the stored client instance instead of creating a new one each time
        
        # Handle both string and dictionary model info
        if isinstance(model_info, dict):
            model_name = model_info['name']
            thinking_enabled = model_info.get('thinking_enabled', False)
        else:
            model_name = model_info
            thinking_enabled = False
        
        # Create message parameters
        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 3000,
            "temperature": 0.3
        }
        
        # Add thinking parameter if enabled
        if thinking_enabled:
            # When thinking is enabled, max_tokens must be greater than thinking.budget_tokens
            thinking_budget = 16000
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }
            # Ensure max_tokens is greater than thinking.budget_tokens
            params["max_tokens"] = thinking_budget + 6000  # 22000 total
            # Temperature must be set to 1 when thinking is enabled
            params["temperature"] = 1
        
        # Make API call
        if thinking_enabled:
            # For thinking-enabled models, use streaming to avoid timeout
            print(f"Using streaming for thinking-enabled model: {model_name}")
            
            # Initialize variables to collect the response
            text_content = ""
            thinking_content = ""
            
            # Stream the response
            with self.client.messages.stream(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
                thinking=params["thinking"]
            ) as stream:
                for text in stream.text_stream:
                    # Collect the text content
                    text_content += text
                
                # Get the final message
                message = stream.get_final_message()
                
                # Extract thinking content if available
                for content in message.content:
                    if hasattr(content, 'type') and content.type == "thinking" and hasattr(content, 'thinking'):
                        thinking_content = content.thinking
                        break
            
            # Print debug info
            print(f"Streaming complete. Text content length: {len(text_content)}")
            print(f"Thinking content length: {len(thinking_content)}")
            
            # Return the text content
            return text_content
        else:
            # For regular models, use non-streaming
            response = self.client.messages.create(**params)
        
        # Extract final text response
        if thinking_enabled:
            # Dump the entire raw response for debugging
            print(f"\n==== FULL RESPONSE OBJECT ====")
            print(f"Response type: {type(response)}")
            print(f"Response dir: {dir(response)}")
            try:
                print(f"Response dict: {vars(response)}")
            except:
                print("Could not convert response to dict")
            
            # For thinking-enabled models, find the text content
            text_content = ""
            thinking_content = ""
            
            print(f"\n==== CONTENT DETAILS ====")
            print(f"Content type: {type(response.content)}")
            print(f"Content length: {len(response.content) if hasattr(response.content, '__len__') else 'Unknown'}")
            
            # First, try to extract both thinking and text content
            for i, content in enumerate(response.content):
                print(f"\nContent item #{i}:")
                print(f"  Type: {type(content)}")
                print(f"  Attributes: {dir(content)}")
                print(f"  Has 'type': {hasattr(content, 'type')}")
                if hasattr(content, 'type'):
                    print(f"  Content type: {content.type}")
                    
                if hasattr(content, 'type'):
                    if content.type == "text" and hasattr(content, 'text'):
                        text_content = content.text
                        print(f"  Found text content, length: {len(text_content)}")
                        print(f"  Text preview: {text_content[:100]}...")
                    elif content.type == "thinking" and hasattr(content, 'thinking'):
                        thinking_content = content.thinking
                        print(f"  Found thinking content, length: {len(thinking_content)}")
                        print(f"  Thinking preview: {thinking_content[:100]}...")
            
            # Print debug info about what we found
            print(f"\n==== EXTRACTION SUMMARY ====")
            print(f"Found text content: {bool(text_content)}, length: {len(text_content) if text_content else 0}")
            print(f"Found thinking content: {bool(thinking_content)}, length: {len(thinking_content) if thinking_content else 0}")
            
            # Return the text content if we found it
            if text_content:
                print(f"Returning text content")
                return text_content
            
            # If we didn't find a text content block, try alternative extraction methods
            print(f"\n==== TRYING ALTERNATIVE EXTRACTION METHODS ====")
            if hasattr(response, 'content') and isinstance(response.content, list):
                # Try to find any content that might contain the final answer
                for i, content in enumerate(response.content):
                    print(f"Checking content item #{i} for text attribute")
                    if hasattr(content, 'text') and content.text:
                        print(f"Found text attribute in content item #{i}")
                        return content.text
            
            # Try to access the response directly
            print(f"\n==== TRYING DIRECT RESPONSE ACCESS ====")
            if hasattr(response, 'text'):
                print(f"Response has 'text' attribute")
                return response.text
            
            # Log the response structure for debugging
            print(f"\n==== WARNING: EXTRACTION FAILED ====")
            print(f"Warning: Could not extract text from thinking-enabled model response.")
            print(f"Response content types: {[content.type for content in response.content if hasattr(content, 'type')]}")
            print(f"Response content structure: {[(content.type, hasattr(content, 'text'), hasattr(content, 'thinking')) for content in response.content]}")
            
            # Last resort: try to convert the entire response to string
            print(f"\n==== LAST RESORT: CONVERT RESPONSE TO STRING ====")
            try:
                response_str = str(response)
                print(f"Response as string: {response_str[:200]}...")
                return response_str
            except:
                print("Failed to convert response to string")
            
            return ""  # Fallback if no text content found
        else:
            # For regular models, use existing extraction
            return response.content[0].text if isinstance(response.content, list) else response.content

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Initialize the client once when the provider is created
        self.client = self.initialize_client()
        print("OpenAI client initialized once")
    
    def initialize_client(self):
        return OpenAI(api_key=self.api_key)
    
    def get_response(self, prompt: str, model_name: str) -> str:
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.3
        )
        return response.choices[0].message.content

class OpenAIO1Provider(LLMProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Initialize the client once when the provider is created
        self.client = self.initialize_client()
        print("OpenAI O1 client initialized once")
    
    def initialize_client(self):
        return OpenAI(api_key=self.api_key)
    
    def get_response(self, prompt: str, model_name: str, reasoning_effort: str = "medium") -> str:
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                temperature=1,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=30000,
                reasoning_effort=reasoning_effort
            )
            
            if response.choices and response.choices[0].message:
                result = response.choices[0].message.content
                if not result or len(result) == 0:
                    print(f"Empty or no response received. Raw response: {response}")
                return result
            else:
                print(f"No content in response choices. Raw response: {response}")
                return ""
                
        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
            print(f"Raw response at time of error: {response if 'response' in locals() else 'No response generated'}")
            raise

class GoogleAIProvider(LLMProvider):
    # Class variable to track if Google SDK has been configured
    _google_sdk_configured = False
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Configure Google SDK only once across all instances
        if not GoogleAIProvider._google_sdk_configured:
            self.initialize_client()
            GoogleAIProvider._google_sdk_configured = True
            print("Google AI SDK configured once")
    
    def initialize_client(self):
        genai.configure(api_key=self.api_key)
        
    def get_response(self, prompt: str, model_name: str) -> str:
        # No need to call initialize_client() again
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=3000,
                candidate_count=1,
            )
        )
        return response.text

class UnifiedLLMProcessor:
    def __init__(self, providers_config: Dict, medical_cases_dir: str):
        self.providers = {
            'anthropic': AnthropicProvider(providers_config['anthropic_api_key']),
            'openai': OpenAIProvider(providers_config['openai_api_key']),
            'openai_o1': OpenAIO1Provider(providers_config['openai_api_key']),
            'google': GoogleAIProvider(providers_config['google_api_key'])
        }
        
        self.models = {
            'anthropic': providers_config['anthropic_models'],
            'openai': providers_config['openai_models'],
            'openai_o1': providers_config['openai_o1_models'],
            'google': providers_config['google_models']
        }
        
        # Store the medical cases directory
        self.medical_cases_dir = medical_cases_dir
        
        # Format directory name for output files
        dir_suffix = medical_cases_dir.rstrip('/').replace('/', '_')
        
        self.output_files = {
            'anthropic': f'llm_anthropic_typical_responses_{dir_suffix}.csv',
            'openai': f'llm_openai_typical_responses_{dir_suffix}.csv',
            'openai_o1': f'llm_openai_o1_typical_responses_{dir_suffix}.csv',
            'google': f'llm_google_typical_responses_{dir_suffix}.csv'
        }

    def process_provider(self, provider_name: str, cases: List[Dict]) -> bool:
        try:
            print(f"\nStarting {provider_name} analysis...")
            
            # Handle different model formats
            if provider_name == 'openai_o1':
                model_names = [model['name'] for model in self.models[provider_name]]
                print(f"Using models: {', '.join(model_names)}")
            elif provider_name == 'anthropic':
                model_names = []
                for model in self.models[provider_name]:
                    if isinstance(model, dict):
                        model_name = model['name']
                        thinking_enabled = model.get('thinking_enabled', False)
                        display_name = f"{model_name}_thinking" if thinking_enabled else model_name
                        model_names.append(display_name)
                    else:
                        model_names.append(model)
                print(f"Using models: {', '.join(model_names)}")
            else:
                print(f"Using models: {', '.join(self.models[provider_name])}")
            
            results = []
            for case in cases:
                with ThreadPoolExecutor() as executor:
                    if provider_name == 'openai_o1':
                        futures = {
                            executor.submit(
                                self._process_single_model,
                                provider_name,
                                model,
                                case['case_id'],
                                case['case_text']
                            ): model
                            for model in self.models[provider_name]
                        }
                    else:
                        futures = {
                            executor.submit(
                                self._process_single_model,
                                provider_name,
                                model_name,
                                case['case_id'],
                                case['case_text']
                            ): model_name
                            for model_name in self.models[provider_name]
                        }
                    
                    case_results = {
                        'case_id': case['case_id'],
                        'medical_case_text': case['case_text'],
                    }
                    
                    # Add debug prints for results collection
                    for future in futures:
                        model_info = futures[future]
                        try:
                            response, model_identifier = future.result()
                            # Store response with model identifier that includes reasoning_effort
                            case_results[f'llm_response_{model_identifier}'] = response
                        except Exception as e:
                            if provider_name == 'openai_o1':
                                model_name = model_info['name']
                                reasoning_effort = model_info['reasoning_effort']
                                model_identifier = f"{model_name}_{reasoning_effort}"
                            elif provider_name == 'anthropic' and isinstance(model_info, dict):
                                model_name = model_info['name']
                                thinking_enabled = model_info.get('thinking_enabled', False)
                                model_identifier = f"{model_name}_thinking" if thinking_enabled else model_name
                            else:
                                model_identifier = model_info
                            # print(f"Error with {provider_name} model {model_identifier} for Case ID {case['case_id']}: {e}")
                            case_results[f'llm_response_{model_identifier}'] = ""
                
                # Debug print for case_results
                # print(f"DEBUG - Case results keys: {case_results.keys()}")
                results.append(case_results)
            
            # Debug print before saving
            # print(f"DEBUG - Number of results to save: {len(results)}")
            # print(f"DEBUG - First result keys: {results[0].keys() if results else 'No results'}")
            
            self._save_results(
                results, 
                self.models[provider_name],
                self.output_files[provider_name],
                provider_name
            )
            
            # print(f"\nResults saved to {self.output_files[provider_name]}")
            return True
            
        except Exception as e:
            # print(f"Error in {provider_name} processing: {str(e)}")
            return False

    def _process_single_model(self, provider_name: str, model_info: str or Dict, case_id: str, case_text: str) -> tuple:
        """Process a single model for a given case. Returns response and model identifier."""
        if provider_name == 'openai_o1':
            model_name = model_info['name']
            reasoning_effort = model_info['reasoning_effort']
            print(f"Processing {provider_name} - {model_name} (reasoning_effort: {reasoning_effort}) for case {case_id}")
            prompt = self._create_prompt(case_text)
            response = self.providers[provider_name].get_response(prompt, model_name, reasoning_effort)
            # Return both the response and a model identifier that includes reasoning_effort
            return response, f"{model_name}_{reasoning_effort}"
        elif provider_name == 'anthropic' and isinstance(model_info, dict):
            model_name = model_info['name']
            thinking_enabled = model_info.get('thinking_enabled', False)
            model_identifier = f"{model_name}_thinking" if thinking_enabled else model_name
            print(f"Processing {provider_name} - {model_identifier} for case {case_id}")
            prompt = self._create_prompt(case_text)
            response = self.providers[provider_name].get_response(prompt, model_info)
            return response, model_identifier
        else:
            model_name = model_info
            print(f"Processing {provider_name} - {model_name} for case {case_id}")
            prompt = self._create_prompt(case_text)
            response = self.providers[provider_name].get_response(prompt, model_name)
            return response, model_name

    @staticmethod
    def _create_prompt(case_text: str) -> str:
        return f"""

Role and Task:
You are a board-certified internal medicine physician reviewing a complex clinical vignette for a test scenario. Your goal is to systematically approach the differential diagnosis and arrive at the most likely final diagnosis. Please follow the structure below. Have a concise writing style. 
It is vital that you order your diagnoses from most likely to least likely; your diagnosis 1 should be the one you think is most likely, and the likelihood of each diagnosis should decrease as the diagnosis number increases.

Part 1 – Differential Diagnosis Considerations:
You MUST identify and list AT LEAST 10 differential diagnoses, even if some are less likely. For each diagnosis, provide:

- Diagnosis 1
Supporting Findings:
List specific clinical signs, symptoms, laboratory values, imaging findings, or epidemiological factors from the vignette that support this diagnosis.
Opposing Findings or Missing Evidence:
Note any contradictions or expected findings that are not present, which weaken the likelihood of this diagnosis.

- Diagnosis 2
Supporting Findings:
Enumerate the details from the case that favor this diagnosis.
Opposing Findings or Missing Evidence:
Identify evidence or typical features that are lacking or that argue against this diagnosis.

- Diagnosis 3
Supporting Findings:
Describe the supporting evidence from the case that aligns with this diagnosis.
Opposing Findings or Missing Evidence:
Mention any inconsistencies or missing elements that reduce its probability.

Continue this pattern for ALL 10 or more diagnoses. Even if some diagnoses become less likely after the first few, you must still list at least 10 total diagnoses. Include all possible differential diagnoses that could reasonably be considered for this case, no matter how remote the possibility.

Medical Case:

{case_text}


""".strip()

    def _save_results(self, results: List[Dict], models: List[str or Dict], output_file: str, provider_name: str):
        fieldnames = ['case_id', 'medical_case_text']
        
        # Extract model identifiers for fieldnames
        model_identifiers = []
        if provider_name == 'openai_o1':
            for model in models:
                model_identifiers.append(f"{model['name']}_{model['reasoning_effort']}")
        elif provider_name == 'anthropic':
            for model in models:
                if isinstance(model, dict):
                    model_name = model['name']
                    thinking_enabled = model.get('thinking_enabled', False)
                    model_identifiers.append(f"{model_name}_thinking" if thinking_enabled else model_name)
                else:
                    model_identifiers.append(model)
        else:
            model_identifiers = models
        
        fieldnames.extend([f'llm_response_{model_id}' for model_id in model_identifiers])
        
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for data in results:
                cleaned_data = {
                    'case_id': data['case_id'],
                    'medical_case_text': data['medical_case_text'].strip()
                }
                # Preserve original formatting for model responses
                for model_id in model_identifiers:
                    response_key = f'llm_response_{model_id}'
                    if response_key in data:
                        cleaned_data[response_key] = data[response_key]
                    else:
                        cleaned_data[response_key] = ''
                writer.writerow(cleaned_data)

def read_medical_cases(directory: str) -> List[Dict]:
    cases = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            case_id = os.path.splitext(filename)[0]
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                case_text = file.read()
                cases.append({'case_id': case_id, 'case_text': case_text})
    return cases

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process medical cases with various LLM providers.')
    parser.add_argument('--medical_cases_dir', type=str, required=True,
                        help='Directory containing medical case text files (e.g., cases_typical/stage_1)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    medical_cases_dir = args.medical_cases_dir
    
    print(f"Processing medical cases from directory: {medical_cases_dir}")
    # Configuration with all models
    providers_config = {
        'anthropic_api_key': os.environ.get("ANTHROPIC_API_KEY"),
        'openai_api_key': os.environ.get("OPENAI_API_KEY"),
        'google_api_key': os.environ.get("GOOGLE_API_KEY"),
        
        'anthropic_models': [
            #'claude-3-5-sonnet-20241022',
            'claude-3-5-haiku-20241022',
            'claude-3-7-sonnet-20250219',
            {'name': 'claude-3-7-sonnet-20250219', 'thinking_enabled': True},
            'claude-3-opus-20240229'
            #'claude-3-sonnet-20240229',
            #'claude-3-haiku-20240307'
        ],
        
        'openai_models': [
            'gpt-4o-mini-2024-07-18',
            #'gpt-4-0613',
            'gpt-3.5-turbo-0125',
            #'gpt-4-turbo-2024-04-09',
            'gpt-4o-2024-11-20'
        ],
        
        'openai_o1_models': [
            #{'name': 'o1-preview-2024-09-12', 'reasoning_effort': 'medium'},
            #{'name': 'o1-mini-2024-09-12', 'reasoning_effort': 'medium'},
            #{'name': 'o1-2024-12-17', 'reasoning_effort': 'high'},
            {'name': 'o1-2024-12-17', 'reasoning_effort': 'medium'},
            #{'name': 'o3-mini-2025-01-31', 'reasoning_effort': 'high'},
            {'name': 'o3-mini-2025-01-31', 'reasoning_effort': 'medium'}   
        ],
        
        'google_models': [
            #'gemini-1.5-flash-001',
            #'gemini-1.5-flash-002',
            #'gemini-1.5-flash-8b-exp-0924',
            #'gemini-1.5-flash-8b-001',
            'gemini-1.5-pro-001',
            'gemini-1.5-pro-002',
            'gemini-2.0-flash-001',       
            #'gemini-2.5-pro-preview',                    
            #'gemini-2.0-pro-exp-02-05',                          
            #'gemini-2.0-flash-thinking-exp-01-21'   

        ]
    }
    
    try:
        processor = UnifiedLLMProcessor(providers_config, medical_cases_dir)
        
        print("Reading medical cases...")
        cases = read_medical_cases(medical_cases_dir)
        print(f"Found {len(cases)} medical cases to process")
        
        print("\nStarting parallel processing of all providers...")
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(processor.process_provider, provider, cases): provider
                for provider in processor.providers.keys()
            }
            
            for future in futures:
                provider = futures[future]
                try:
                    success = future.result()
                    if success:
                        print(f"{provider} processing completed successfully")
                    else:
                        print(f"{provider} processing failed")
                except Exception as e:
                    print(f"Error processing {provider}: {str(e)}")
        
        print("\nAll processing completed!")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
