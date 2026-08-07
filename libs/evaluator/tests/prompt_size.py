from evaluator.evaluator_interface import count_prompt_tokens

if __name__ == "__main__":
    print(f"The prompt has size {count_prompt_tokens()} tok.")