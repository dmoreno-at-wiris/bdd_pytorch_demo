# features/steps/steps.py
from behave import given, when, then
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleClassifier, train_one_epoch

# --- Scenario 1 Steps ---

@given('a model with input size {input_size:d} and output size {output_size:d}')
def step_impl(context, input_size, output_size):
    context.input_size = input_size
    context.output_size = output_size
    context.model = SimpleClassifier(input_size, output_size)

@when('I perform inference with a batch size of {batch_size:d}')
def step_impl(context, batch_size):
    # Create random input tensor $X \in \mathbb{R}^{N \times D_{in}}$
    input_data = torch.randn(batch_size, context.input_size)
    context.output = context.model(input_data)

@then('the output tensor shape should be {batch_size:d}x{output_size:d}')
def step_impl(context, batch_size, output_size):
    expected_shape = torch.Size([batch_size, output_size])
    assert context.output.shape == expected_shape, \
        f"Expected shape {expected_shape}, but got {context.output.shape}"

# --- Scenario 2 Steps ---

@given('an initialized binary classifier')
def step_impl(context):
    context.model = SimpleClassifier(input_size=10, num_classes=1)
    context.criterion = nn.BCELoss()
    context.optimizer = optim.SGD(context.model.parameters(), lr=0.1)

@given('a synthetic dataset')
def step_impl(context):
    # Fixed seed for reproducibility
    torch.manual_seed(42)
    context.data = torch.randn(10, 10)
    context.target = torch.randint(0, 2, (10, 1)).float()

@when('I train the model for {steps:d} steps')
def step_impl(context, steps):
    # Record initial loss
    initial_output = context.model(context.data)
    context.initial_loss = context.criterion(initial_output, context.target).item()

    # Train loop
    for _ in range(steps):
        current_loss = train_one_epoch(
            context.model,
            context.optimizer,
            context.criterion,
            context.data,
            context.target
        )
    context.final_loss = current_loss

@then('the final loss should be strictly less than the initial loss')
def step_impl(context):
    assert context.final_loss < context.initial_loss, \
        f"Model did not learn! Initial: {context.initial_loss}, Final: {context.final_loss}"
