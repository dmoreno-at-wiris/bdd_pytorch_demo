Feature: Neural Network Training and Inference

  Scenario: Model preserves correct tensor dimensions
    Given a model with input size 10 and output size 2
    When I perform inference with a batch size of 5
    Then the output tensor shape should be 5x2

  Scenario: Model loss decreases after training
    Given an initialized binary classifier
    And a synthetic dataset
    When I train the model for 50 steps
    Then the final loss should be strictly less than the initial loss