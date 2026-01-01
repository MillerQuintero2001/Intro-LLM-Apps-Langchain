from langchain_huggingface import HuggingFacePipeline

# Define the LLM from the Hugging Face model ID
llm = HuggingFacePipeline.from_model_id(
    model_id="crumb/nano-mistral",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 40}
)

prompt = "The sky is blue, roses are red. I will be..."

# Invoke the model
response = llm.invoke(prompt)
print(response)