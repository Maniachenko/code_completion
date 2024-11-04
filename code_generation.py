import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

# Set device to GPU if available
device = 'cuda'

# Model configuration for Tiny Starcoder with quantization
model_name = 'bigcode/tiny_starcoder_py'
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

DATA_DIR = "data"

# Load dataset from CSV file
dataset_path = os.path.join(DATA_DIR, 'code_completion_examples.csv')
df = pd.read_csv(dataset_path)

# Define max allowable length for input tokens and set generation length for FIM
max_total_length = 4096
max_new_tokens = 128

# Load tokenizer and model with quantization configuration
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

# Add padding token to the tokenizer if it's missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})


def format_fim_input(prefix, suffix):
    """
    Formats the input for Fill-In-the-Middle (FIM) by combining prefix and suffix.
    FIM format uses specific tokens to mark prefix and suffix.
    """
    return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"


def truncate_middle_context(tokenizer, prompt, max_length):
    """
    Truncates the tokenized input prompt to fit within the specified max length.
    Retains an equal number of tokens around the FIM suffix marker.
    """
    full_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    total_tokens = full_tokens.shape[1]
    if total_tokens <= max_length:
        return full_tokens

    # Calculate half of the allowed length for balanced truncation
    half_length = max_length // 2
    fim_suffix_token_id = tokenizer.convert_tokens_to_ids("<fim_suffix>")
    fim_suffix_indices = (full_tokens == fim_suffix_token_id).nonzero(as_tuple=True)[1]

    if fim_suffix_indices.numel() == 0:
        raise ValueError("The token '<fim_suffix>' was not found in the prompt.")

    # Locate FIM suffix position and trim around it
    fim_suffix_index = fim_suffix_indices[0].item()
    start_index = max(0, fim_suffix_index - half_length)
    end_index = min(total_tokens, fim_suffix_index + half_length)
    return full_tokens[:, start_index:end_index]


def remove_subsequences_from_output(input_ids, generated_ids):
    """
    Removes subsequences of the input from the generated output to avoid repeating prompts.
    Helps ensure the generated output is novel content.
    """
    input_ids_list = input_ids.squeeze().tolist()
    generated_ids_list = generated_ids.squeeze().tolist()

    # Search for the input sequence in the generated output and remove if found
    for start in range(len(generated_ids_list)):
        if generated_ids_list[start:start + len(input_ids_list)] == input_ids_list:
            del generated_ids_list[start:start + len(input_ids_list)]

    return generated_ids_list


def generate_code(model, tokenizer, dataset, column_name="generated_code"):
    """
    Generates code completions for each row in the dataset and stores them in a specified column.
    Uses FIM format, quantized model, and removes input subsequences from the output.
    """
    for index, row in dataset.iterrows():
        print(f"\n--- Processing row {index + 1} ---")

        # Prepare prompt using FIM format and truncate to fit within max token limit
        prompt = format_fim_input(row['prefix'], row['suffix'])
        inputs = truncate_middle_context(tokenizer, prompt, max_total_length)
        input_ids = inputs
        attention_mask = torch.ones_like(input_ids, device=device)

        try:
            # Generate output with specified generation settings
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True
            )

            # Convert output IDs to text, remove input subsequences, and decode
            generated_ids = remove_subsequences_from_output(input_ids, outputs)
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            dataset.at[index, column_name] = output_text

            print(f"Generated output for row {index + 1}: {output_text}\n")

        except RuntimeError as e:
            print(f"Generation failed for row {index + 1}: {e}")
            dataset.at[index, column_name] = "Generation error"


# Run code generation on the dataset
generate_code(model, tokenizer, df)

# Save the output DataFrame to a CSV file in DATA_DIR
output_path = os.path.join(DATA_DIR, 'code_completion_output.csv')
df.to_csv(output_path, index=False)
print(f"\nAll generations completed. Results saved to '{output_path}'.")
