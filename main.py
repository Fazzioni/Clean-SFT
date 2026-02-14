import torch
import datasets
import dotenv
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, get_linear_schedule_with_warmup
from utils import Args, History, ConversationDataset
dotenv.load_dotenv()
torch.set_float32_matmul_precision('high')

def main(arguments :  Args):
    tokenizer = AutoTokenizer.from_pretrained(arguments.model_name)
    model = AutoModelForCausalLM.from_pretrained(arguments.model_name, dtype='bfloat16', device_map="cpu")
    if torch.cuda.is_available():
        model = model.to("cuda:0")

    if arguments.compile:
        model = torch.compile(model, mode="default")

    raw_dataset = datasets.load_dataset(arguments.dataset, split="train")

    dataset = ConversationDataset(raw_dataset, tokenizer, max_length=arguments.max_length, chat_template=arguments.path_chat_template)
    train_loader = DataLoader(dataset, batch_size=arguments.per_device_batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=arguments.learning_rate)
    history = History(total_steps=(len(train_loader) // arguments.accumulation_steps) * arguments.epochs)
    num_warmup_steps = int(history.total_steps * arguments.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup( optimizer,  num_warmup_steps=num_warmup_steps,  num_training_steps=history.total_steps)
    
    model.train()
    print(f"Iniciando treinamento...")
    print(f"Total de updates planejados: {history.total_steps} | Warmup steps: {num_warmup_steps}")

    for epoch in range(arguments.epochs):
        optimizer.zero_grad() 
        
        for step, batch in enumerate(train_loader):
            outputs = model(input_ids=batch["input_ids"].to(model.device),
                            attention_mask=batch["attention_mask"].to(model.device),
                            labels=batch["labels"].to(model.device),
                            num_items_in_batch=batch.pop("num_itens_sample").sum().item()
                            )
            
            loss = outputs.loss / arguments.accumulation_steps
            history.append_acc_loss(loss.item())
            loss.backward()

            if (step + 1) % arguments.accumulation_steps == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=arguments.max_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                history.do_step(learning_rate=scheduler.get_last_lr()[0], grad_norm=total_norm)
    
    model.save_pretrained(arguments.output_dir)
    tokenizer.save_pretrained(arguments.output_dir)
    history.save_pretrained(arguments.output_dir)
    arguments.save_pretrained(arguments.output_dir)
    print(f"Treinamento concluído. Modelo salvo em {arguments.output_dir}")


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)




'''
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "./output/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
prompt = "Qual a capital do Brasil?"
messages = [ {"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True, )
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate( **model_inputs, max_new_tokens=256)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
print(tokenizer.decode(output_ids))
'''