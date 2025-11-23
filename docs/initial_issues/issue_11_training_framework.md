# [Training] [M1] Set up baseline training framework

## Assigned Agent
distillation-training-engineer

## Milestone
M1: Setup & Baseline

## Task Description
Create the foundational training framework that will be used for model distillation. This includes the basic training loop, optimizer setup, learning rate scheduling, checkpointing, and basic logging infrastructure.

## Context & Background
A robust training framework is essential for all subsequent distillation work. This foundational framework should:
- Be modular and easy to extend
- Support standard PyTorch training patterns
- Handle checkpointing and resuming
- Provide clear logging and progress tracking
- Be ready for distillation-specific loss functions (to be added in M4)

## Acceptance Criteria
- [ ] Create `src/training/trainer.py` with base Trainer class
- [ ] Implement training loop with train/validation phases
- [ ] Support multiple optimizers (Adam, AdamW, SGD)
- [ ] Implement learning rate scheduling (cosine, step, plateau)
- [ ] Checkpoint saving and loading
- [ ] Basic logging (loss, learning rate, epoch time)
- [ ] Early stopping mechanism
- [ ] Gradient clipping support
- [ ] Mixed precision training support (optional but recommended)
- [ ] Comprehensive docstrings and type hints
- [ ] Unit tests for trainer components
- [ ] Integration test with dummy model and data

## Technical Notes

### Suggested API Design
```python
from mlff_distiller.training import Trainer, TrainingConfig

# Configure training
config = TrainingConfig(
    max_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    optimizer="adam",
    lr_scheduler="cosine",
    checkpoint_dir="checkpoints/",
    log_interval=10,
    val_interval=1
)

# Create trainer
trainer = Trainer(
    model=student_model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device="cuda"
)

# Train
trainer.fit()

# Load best checkpoint
trainer.load_checkpoint("checkpoints/best_model.pt")
```

### Key Components

1. **Training Loop**:
```python
def train_epoch(self):
    self.model.train()
    for batch in self.train_loader:
        # Forward pass
        outputs = self.model(batch)
        loss = self.compute_loss(outputs, batch)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

        self.optimizer.step()
```

2. **Checkpointing**:
```python
def save_checkpoint(self, path, is_best=False):
    checkpoint = {
        'epoch': self.current_epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
        'best_loss': self.best_loss,
        'config': self.config
    }
    torch.save(checkpoint, path)
```

3. **Validation**:
```python
def validate(self):
    self.model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in self.val_loader:
            outputs = self.model(batch)
            loss = self.compute_loss(outputs, batch)
            total_loss += loss.item()
    return total_loss / len(self.val_loader)
```

### Configuration System

Use Pydantic for configuration validation:
```python
from pydantic import BaseModel

class TrainingConfig(BaseModel):
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    grad_clip: Optional[float] = None
    checkpoint_dir: str = "checkpoints/"
    log_interval: int = 10
    val_interval: int = 1
    early_stopping_patience: int = 10
    mixed_precision: bool = False
```

## Related Issues
- Related to: #12 (training config system)
- Enables: #13 (distillation loss functions), #14 (monitoring), #15 (hyperparameter tuning)

## Dependencies
- torch
- pydantic (for config validation)
- tqdm (for progress bars)

## Estimated Complexity
Medium (4-5 days)

## Definition of Done
- [ ] Code implemented and follows style guide
- [ ] All acceptance criteria met
- [ ] Tests written and passing
- [ ] Documentation with complete example
- [ ] Example training script in `examples/train_example.py`
- [ ] PR created and reviewed
- [ ] PR merged to main

## Resources
- PyTorch Lightning: https://pytorch-lightning.readthedocs.io/ (for design inspiration)
- PyTorch training best practices: https://pytorch.org/tutorials/
- Hugging Face Trainer: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py

## Notes
- Keep the framework simple and focused for M1
- Distillation-specific loss functions will be added in M4
- Advanced features (distributed training, etc.) can be added later if needed
- Consider adding basic tensorboard logging support
