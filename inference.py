import torch
import time

class InferenceModule:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generator(self, inputs, max_new_tokens: int = 50, batch_size: int = 8, return_metadata: bool = False):

        """
    Generate responses from the model given single or multiple input prompts.

    Parameters:
    -----------
    inputs : str or List[str]
        A single input string or a list of input strings (e.g., questions, prompts).

    max_new_tokens : int, optional (default=50)
        The maximum number of new tokens the model is allowed to generate in response.

    batch_size : int, optional (default=8)
        Number of inputs to process in a single batch. Currently used for controlling loop externally
        (internal batching not implemented in this version).

    return_metadata : bool, optional (default=False)
        If True, returns detailed metadata for each input including token counts and generation time.
        If False, returns only the generated text.

    Returns:
    --------
    Union[str, List[str], Dict, List[Dict]]
        - If `inputs` is a single string and `return_metadata=False`: returns a single string.
        - If `inputs` is a list of strings and `return_metadata=False`: returns a list of generated strings.
        - If `inputs` is a single string and `return_metadata=True`: returns a metadata dictionary.
        - If `inputs` is a list of strings and `return_metadata=True`: returns a list of metadata dictionaries.

    Metadata Dictionary Structure (if return_metadata=True):
    --------------------------------------------------------
    {
        "input_text": str,
        "output_text": str,
        "input_tokens": int,
        "output_tokens": int,
        "total_tokens": int,
        "time_taken_per_sample_sec": float
    }
    """
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False


        encoded = self.tokenizer(
            list(inputs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)

        start_time = time.time()


        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False
            )


        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        end_time = time.time()
        elapsed = end_time - start_time
        time_per_sample = elapsed / len(inputs)

        if return_metadata:
            metadata = []
            for idx, (inp, out, inp_ids, out_ids) in enumerate(zip(inputs, decoded, encoded["input_ids"], outputs)):
                meta = {
                    "input_text": inp,
                    "output_text": out,
                    "input_tokens": len([t for t in inp_ids.tolist() if t != self.tokenizer.pad_token_id]),
                    "output_tokens": len(out_ids),
                    "total_tokens": len(inp_ids) + len(out_ids),
                    "time_taken_per_sample_sec": round(time_per_sample, 5)
                }
                metadata.append(meta)

            return metadata[0] if single_input else metadata

        return decoded[0] if single_input else decoded
