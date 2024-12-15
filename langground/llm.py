import re
import tenacity
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


class LLM:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct",):

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the model and tokenizer based on the model_id
        if "meta-llama" in self.model_id:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

        elif "InternVL" in self.model_id:
            self.model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto"
            ).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @torch.no_grad()
    def generate(self, query):
        if "meta-llama" in self.model_id:
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": f"{query}"}
                ]}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512)
            generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elif "InternVL" in self.model_id:
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            response = self.model.chat(self.tokenizer, None, query, generation_config, history=None, return_history=False)
        else:
            messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": query}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512)
            generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    @tenacity.retry(stop=tenacity.stop_after_attempt(5))
    def answer(self, query, objects):
        query = f"""
        Extract the object that satisfies the intent of the query or determine the tool that aligns with the purpose of {query}.
        pick the best option from the following: {', '.join(objects)},
        Please return a list of all suitable options as long as they make sense in the format of a Python list in the following format: ```python\n['option1', 'option2', ...]```"""
        res = self.generate(query)
        match = re.search(r"`{3}python\\n(.*)`{3}", res, re.DOTALL)
        if match:
            res = match.group(1)
            res = [r.translate(str.maketrans("", "", "_-")) for r in eval(res)]
            return res
        else:
            # Try to extract content directly from brackets []
            match_brackets = re.search(r"\[(.*?)\]", res, re.DOTALL)
            if match_brackets:
                res = match_brackets.group(0)  # Include brackets for eval
                res = [r.translate(str.maketrans("", "", "_-")) for r in eval(res)]
                return res
            else:
                raise ValueError(f"Failed to parse response: {res}")
