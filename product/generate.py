#!/usr/bin/env -tt python

from model import *

model = BigramLanguageModel()
m = model.to(device)

print("Please wait while we load the model ...", end = " ")
m.load_state_dict(torch.load('./states/zephyrus'))
print("Complete")
print("===")

# # generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, 1000)[0].tolist()))