# Quantum Circuit Cutting Experiment

This project demonstrates the use of circuit cutting techniques in quantum computing using Qiskit. It allows running experiments on both local simulators and IBM Quantum hardware.

## Dependencies

To run this project, you need to install the following Python packages:

```
numpy
qiskit
qiskit-ibm-runtime
qiskit-aer
circuit_knitting
```

You can install these dependencies using pip:

```
pip install numpy qiskit qiskit-ibm-runtime qiskit-aer circuit_knitting
```

## Configuration

Before running the experiment on IBM Quantum hardware, make sure to replace the token in the `QiskitRuntimeService` initialization with your own IBM Quantum account token:

```python
service = QiskitRuntimeService(channel="ibm_quantum", token="your_token_here")
```

## Usage

The main function in the script is `experiment_with_circuit(num_qubits, depth, use_ibmq=False)`. 

To run a local simulation:

```python
results, total_time, original_time = experiment_with_circuit(8, 4, use_ibmq=False)
```

To run on IBM Quantum hardware:

```python
results, total_time, original_time = experiment_with_circuit(8, 3, use_ibmq=True)
```

## Output

The script will print information about the circuit cutting process, execution times, and save the results in a `results.json` file.

## Note

Running on IBM Quantum hardware requires an IBM Quantum account and may incur usage costs. Make sure you understand the implications before running experiments on real quantum hardware.

