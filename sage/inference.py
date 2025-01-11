from model import *
from PIL import Image
import numpy as np
import torch

img = Image.open("kanye.jpg").convert('RGB')
img_array = np.array(img)
# Rearranging dimensions from (H, W, C) to (C, H, W)
img_tensor = torch.tensor(img_array).permute(2, 0, 1)
img_tensor = (img_tensor.float() / 255.0).unsqueeze(0)

tokenizer, model = Sage.from_pretrained()
model.eval()  # Set model to evaluation mode

# Initial input
input_text = "<|im_start|>user\n<image> What is he wearing and what color is it?<|im_end|>\n<|im_start|>assistant\n"
max_length = 20  # Maximum number of tokens to generate
stop_token = tokenizer.eos_token_id  # End-of-sequence token (if applicable)

# Tokenize initial input
input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
generated_text = input_text

# Generate tokens iteratively
for _ in range(max_length):
    # Forward pass
    with torch.no_grad():
        output = model.forward(input_ids, img_tensor)
        last_token_logits = output[:, -1, :]  # Get logits for the last token

    # Apply softmax to get probabilities
    probs = torch.softmax(last_token_logits, dim=-1)

    # Sample the next token (optional: use temperature for diversity)
    temperature = 0.7
    probs = torch.softmax(last_token_logits / temperature, dim=-1)
    predicted_token_index = torch.multinomial(probs, num_samples=1).item()

    if predicted_token_index == stop_token: break

    # Append the predicted token to the input IDs
    input_ids = torch.cat([input_ids, torch.tensor([[predicted_token_index]])], dim=-1)

    # Decode the new token and add it to the generated text
    new_token = tokenizer.decode([predicted_token_index])
    generated_text += new_token

    # Print intermediate results (optional)
    print(f"Generated so far: {generated_text}")

# Final generated text
print("\nFinal generated text:")
print(generated_text)