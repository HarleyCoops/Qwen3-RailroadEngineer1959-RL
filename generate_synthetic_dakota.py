# This script creates a large set of practice questions and answers for Dakota language training.
# It uses Google's Gemini AI to help generate natural, meaningful questions that test
# different aspects of the Dakota language, from basic translations to complex cultural concepts.
# Adapted from the Stoney Nakoda bilingual_qa_generator.py pattern.

import json
import logging
from typing import Dict, List, Generator
from pathlib import Path
from dotenv import load_dotenv
import os
from tqdm import tqdm
import time
from datetime import datetime
import google.generativeai as genai

# Set up our logging system to track what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DakotaQAGenerator:
    def __init__(self, extracted_dict_dir: str):
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        # Use latest Gemini model (can be overridden via env var)
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
        self.model = genai.GenerativeModel(model_name)
        self.extracted_dict_dir = Path(extracted_dict_dir)
        
        if not self.extracted_dict_dir.exists():
            raise FileNotFoundError(f"Dictionary directory not found: {extracted_dict_dir}")

    def load_dakota_entries(self) -> Generator[Dict, None, None]:
        """Load Dakota dictionary entries from extracted JSON files."""
        json_files = sorted(self.extracted_dict_dir.glob("page_*.json"))
        logger.info(f"Found {len(json_files)} dictionary JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    entries = data.get('entries', [])
                    for entry in entries:
                        # Ensure entry has required fields
                        if entry.get('headword') and entry.get('definition_primary'):
                            yield entry
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON file {json_file}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
                continue

    def create_dakota_context_prompt(self, entries: List[Dict]) -> str:
        context = """You are an expert in the Dakota language. Using the following Dakota-English dictionary entries, 
        create diverse and natural question-answer pairs that test understanding of the language.
        
        Guidelines:
        1. Create questions that test translation from English to Dakota and vice versa
        2. Focus on how Dakota words express different aspects of English concepts
        3. Test understanding of grammatical classifications (verbs, nouns, adjectives) and subtle meaning differences
        4. Create scenarios that demonstrate when to use each Dakota variation
        5. Generate questions about word relationships and patterns
        6. Include cultural context where relevant
        7. Highlight Dakota special characters (ć, š, ŋ, ḣ, ṡ, á, é, í, ó, ú, etc.) in examples
        8. Test understanding of inflected forms and word derivations
        
        Dictionary entries:
        """
        
        for entry in entries:
            # Format entry for context
            entry_str = json.dumps(entry, ensure_ascii=False, indent=2)
            context += f"\n{entry_str}"
            
        return context

    def create_reverse_dakota_context_prompt(self, entries: List[Dict]) -> str:
        context = """You are an expert in the Dakota language. Using the following Dakota-English dictionary entries, 
        create diverse and natural question-answer pairs that test understanding of Dakota from the Dakota-speaker's perspective.
        
        Guidelines:
        1. Create questions that test translation from Dakota to English
        2. Focus on proper usage of Dakota words in different contexts
        3. Test understanding of parts of speech and grammatical rules
        4. Create scenarios for practical usage
        5. Generate questions about cultural significance where relevant
        6. Include questions about related words and concepts (see_also, compare_with)
        7. Test knowledge of inflected forms and derivations
        8. Emphasize preservation of Dakota orthography (special characters)
        
        Dictionary entries:
        """
        
        for entry in entries:
            entry_str = json.dumps(entry, ensure_ascii=False, indent=2)
            context += f"\n{entry_str}"
            
        return context

    def generate_qa_pairs(self, entries: List[Dict], is_english_perspective: bool, context_size: int = 5) -> Generator[Dict, None, None]:
        """Generate Q&A pairs from Dakota dictionary entries."""
        
        if not entries:
            return
        
        context = self.create_dakota_context_prompt(entries) if is_english_perspective else self.create_reverse_dakota_context_prompt(entries)
        
        prompt = """Based on these dictionary entries, generate 5 diverse 
        question-answer pairs. Format your response EXACTLY as shown below, maintaining
        valid JSON structure:

        [
            {
                "question": "What is the Dakota word for X?",
                "answer": "The Dakota word for X is Y."
            }
        ]

        Ensure your response is a valid JSON array containing exactly 5 question-answer pairs.
        Do not include any additional text or formatting.
        
        Important: Preserve Dakota orthography exactly (including special characters like ć, š, ŋ, etc.)"""
        
        try:
            logger.info(f"Sending request to Google API with {len(entries)} entries...")
            response = self.model.generate_content(
                contents=context + "\n" + prompt
            )
            logger.info("Received response from Google API.")
            response_text = response.text.strip()
            
            # Extract JSON array from response
            if not response_text.startswith('['):
                start_idx = response_text.find('[')
                if start_idx >= 0:
                    response_text = response_text[start_idx:]
                else:
                    logger.warning("No JSON array found in response")
                    return
            
            if not response_text.endswith(']'):
                end_idx = response_text.rfind(']')
                if end_idx >= 0:
                    response_text = response_text[:end_idx+1]
                else:
                    logger.warning("No closing bracket found in response")
                    return
            
            qa_pairs = json.loads(response_text)
            for qa_pair in qa_pairs:
                if isinstance(qa_pair, dict) and 'question' in qa_pair and 'answer' in qa_pair:
                    qa_pair['source_language'] = 'english' if is_english_perspective else 'dakota'
                    yield qa_pair
                else:
                    logger.warning("Skipping invalid QA pair format")
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing JSON response: {str(e)}")
            logger.debug(f"Response text: {response_text[:500]}")
        except Exception as e:
            logger.warning(f"Error generating Q&A pairs: {str(e)}")

    def generate_training_set(self, output_file: str, pairs_per_language: int = 75000, context_size: int = 5):
        """
        Generate Q&A pairs from Dakota dictionary entries.
        
        Args:
            output_file: Path to output JSONL file
            pairs_per_language: Target number of pairs per language direction
            context_size: Number of dictionary entries to include per API call
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_path.parent / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Load all entries into memory
        logger.info("Loading Dakota dictionary entries...")
        all_entries = list(self.load_dakota_entries())
        logger.info(f"Loaded {len(all_entries)} dictionary entries")
        
        if not all_entries:
            raise ValueError("No dictionary entries found. Check that data/extracted/*.json files exist.")
        
        total_pairs = pairs_per_language * 2
        pair_count = 0
        checkpoint_count = 0
        start_time = time.time()
        
        logger.info(f"Starting generation of {total_pairs} Q&A pairs ({pairs_per_language} per language)...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Generate English-perspective Q&A pairs
            logger.info("Generating English-perspective Q&A pairs...")
            english_count = 0
            entries_buffer = []
            
            for entry in tqdm(all_entries, desc="Processing entries for English perspective"):
                if english_count >= pairs_per_language:
                    break
                
                entries_buffer.append(entry)
                
                if len(entries_buffer) >= context_size:
                    for qa_pair in self.generate_qa_pairs(entries_buffer, is_english_perspective=True, context_size=context_size):
                        if english_count >= pairs_per_language:
                            break
                        qa_pair['generated_at'] = datetime.now().isoformat()
                        qa_pair['pair_id'] = pair_count + 1
                        f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                        pair_count += 1
                        english_count += 1
                        
                        if pair_count % 1000 == 0:
                            self._create_checkpoint(checkpoint_dir, checkpoint_count, pair_count, total_pairs)
                            checkpoint_count += 1
                    
                    # Keep last 2 entries for context continuity
                    entries_buffer = entries_buffer[-2:]
            
            # Generate Dakota-perspective Q&A pairs
            logger.info("Generating Dakota-perspective Q&A pairs...")
            dakota_count = 0
            entries_buffer = []
            
            for entry in tqdm(all_entries, desc="Processing entries for Dakota perspective"):
                if dakota_count >= pairs_per_language:
                    break
                
                entries_buffer.append(entry)
                
                if len(entries_buffer) >= context_size:
                    for qa_pair in self.generate_qa_pairs(entries_buffer, is_english_perspective=False, context_size=context_size):
                        if dakota_count >= pairs_per_language:
                            break
                        qa_pair['generated_at'] = datetime.now().isoformat()
                        qa_pair['pair_id'] = pair_count + 1
                        f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                        pair_count += 1
                        dakota_count += 1
                        
                        if pair_count % 1000 == 0:
                            self._create_checkpoint(checkpoint_dir, checkpoint_count, pair_count, total_pairs)
                            checkpoint_count += 1
                    
                    # Keep last 2 entries for context continuity
                    entries_buffer = entries_buffer[-2:]
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed. Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Generated {pair_count} Q&A pairs ({english_count} English-perspective, {dakota_count} Dakota-perspective)")

    def _create_checkpoint(self, checkpoint_dir: Path, checkpoint_count: int, pair_count: int, total_pairs: int):
        """Create a checkpoint file to track progress."""
        checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_count}.jsonl"
        with open(checkpoint_file, 'w', encoding='utf-8') as cf:
            cf.write(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'pairs_generated': pair_count,
                'target_pairs': total_pairs,
                'percent_complete': (pair_count / total_pairs) * 100 if total_pairs > 0 else 0
            }, ensure_ascii=False) + '\n')

def main():
    try:
        # Set up file paths
        extracted_dict_dir = "data/extracted"
        output_path = "data/bilingual_training_set.jsonl"
        
        # Create the generator
        generator = DakotaQAGenerator(extracted_dict_dir)
        
        # Generate all the questions and answers
        logger.info("Starting full training set generation...")
        generator.generate_training_set(
            output_path,
            pairs_per_language=75000,  # Match Stoney Nakoda default
            context_size=5  # Number of entries per API call
        )
        
        logger.info("Training set generation completed successfully")
                
    except Exception as e:
        logger.error(f"Error during training set generation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

