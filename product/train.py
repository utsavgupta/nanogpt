#!/usr/bin/env -tt python

from model import *

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) # lr = learning rate

# training loop
for iter in range(max_iters):
    
    # calculate loss after an interval
    if iter % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(m.state_dict(), './states/p3')