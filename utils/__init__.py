import dataclasses
import json
from tqdm import tqdm
import os
from .dataset import ConversationDataset
import wandb

class save_function:
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        obj = None
        try:
            obj  = dataclasses.asdict(self)
        except:
            obj = {k:v for k,v in self.__dict__.items() if type(v) in [str, int, float, bool, list, dict, type(None)]}
            
        with open(os.path.join(path, self.filename), 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            
@dataclasses.dataclass
class Args(save_function):
    filename: str = "arguments.json"
    model_name: str = None
    dataset: str = None
    max_length: int = 2048
    per_device_batch_size: int = 8
    learning_rate: float = 5e-6
    accumulation_steps: int = 64
    epochs: int = 2
    warmup_ratio: float = 0.1293
    max_grad: float = 1
    output_dir: str = "./output"
    path_chat_template: str = None
    compile : bool = True

    def __post_init__(self):
        assert self.model_name is not None, "model_name must be provided"
        assert self.dataset is not None, "dataset must be provided"
        if self.path_chat_template is not None:
            assert os.path.isfile(self.path_chat_template), f"Chat template file {self.path_chat_template} does not exist"

class History(save_function):
    filename: str = "training_log.json"
    
    def __init__(self, total_steps, arguments:Args=None):
        self.total_steps = total_steps
        self.global_step : int = 0
        self._loss : list[float] = []
        self._lr : list[float] = []
        self._grad_norm : list[float] = []
        self.tmp_step_loss : list[float] = []
        self.total_steps = total_steps
        
        wandb.init(config=arguments)
        self.desc = tqdm(total=self.total_steps, desc="Treinamento", unit="Steps")
    
    def do_step(self, learning_rate=None, grad_norm=None):
        self.global_step += 1
        self._loss.append(sum(self.tmp_step_loss))
        self.tmp_step_loss = []
        self.desc.update(1)
        self._lr.append(learning_rate)
        self._grad_norm.append(grad_norm)
        
        wandb.log({
            "train/loss": self.loss,
            "train/learning_rate": self.lr,
            "train/grad_norm": self.grad_norm,
            "global_step": self.global_step
        })
        print(f"Step {self.global_step} | LR: {self.lr:.2e} | Loss: {self.loss:.4f} | Grad Norm: {self.grad_norm:.4f}")
        

    def append_acc_loss(self, loss):
        self.tmp_step_loss.append(loss)
        
    @property
    def loss(self):
        return self._loss[-1] if self._loss else None
    @property
    def grad_norm(self):
        return self._grad_norm[-1] if self._grad_norm else None
    @property 
    def lr(self):
        return self._lr[-1] if self._lr else None