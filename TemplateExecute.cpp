/// Step 4: Executing the Script Module in C++

// Having successfully loaded our serialized ResNet18 in C++, we are now just a couple lines of code away from executing it! Let’s add those lines to our C++ application’s main() function:

// Create a vector of inputs.
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::ones({1, 3, 224, 224}));

// Execute the model and turn its output into a tensor.
at::Tensor output = module.forward(inputs).toTensor();
std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';