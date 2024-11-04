import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

device = 'cuda'

# Model configuration for Tiny Starcoder with quantization
model_name = 'bigcode/tiny_starcoder_py'
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load dataset
dataset_path = 'code_completion_examples.csv'
df = pd.read_csv(dataset_path)

# Define max allowable length for input tokens and set generation length for FIM
max_total_length = 4096
max_new_tokens = 128

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

# Add padding token if needed
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})


def format_fim_input(prefix, suffix):
    return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"


def truncate_middle_context(tokenizer, prompt, max_length):
    full_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    total_tokens = full_tokens.shape[1]
    if total_tokens <= max_length:
        return full_tokens

    half_length = max_length // 2
    fim_suffix_token_id = tokenizer.convert_tokens_to_ids("<fim_suffix>")
    fim_suffix_indices = (full_tokens == fim_suffix_token_id).nonzero(as_tuple=True)[1]

    if fim_suffix_indices.numel() == 0:
        raise ValueError("The token '<fim_suffix>' was not found in the prompt.")

    fim_suffix_index = fim_suffix_indices[0].item()
    start_index = max(0, fim_suffix_index - half_length)
    end_index = min(total_tokens, fim_suffix_index + half_length)
    return full_tokens[:, start_index:end_index]


def remove_subsequences_from_output(input_ids, generated_ids):
    input_ids_list = input_ids.squeeze().tolist()
    generated_ids_list = generated_ids.squeeze().tolist()

    # Find subsequences of input_ids in generated_ids and remove them
    for start in range(len(generated_ids_list)):
        if generated_ids_list[start:start + len(input_ids_list)] == input_ids_list:
            # Remove matched sequence from generated output
            del generated_ids_list[start:start + len(input_ids_list)]

    return generated_ids_list


def generate_code(model, tokenizer, dataset, column_name="generated_code"):
    for index, row in dataset.iterrows():
        print(f"\n--- Processing row {index + 1} ---")
        prompt = format_fim_input(row['prefix'], row['suffix'])
        inputs = truncate_middle_context(tokenizer, prompt, max_total_length)
        input_ids = inputs
        attention_mask = torch.ones_like(input_ids, device=device)

        try:
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

            # Convert outputs to IDs, remove input subsequences, and decode
            generated_ids = remove_subsequences_from_output(input_ids, outputs)
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            dataset.at[index, column_name] = output_text

            print(f"Generated output for row {index + 1}: {output_text}\n")

        except RuntimeError as e:
            print(f"Generation failed for row {index + 1}: {e}")
            dataset.at[index, column_name] = "Generation error"


generate_code(model, tokenizer, df)
output_path = 'code_completion_output.csv'
df.to_csv(output_path, index=False)
print("\nAll generations completed. Results saved to 'code_completion_output.csv'.")
