import numpy as np
import json
import time
from qiskit import transpile, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit
from circuit_knitting.cutting.automated_cut_finding import (
    find_cuts,
    OptimizationParameters,
    DeviceConstraints,
)
from circuit_knitting.cutting import (
    cut_wires,
    expand_observables,
    partition_problem,
    generate_cutting_experiments,
)
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler import PassManager
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import SetLayout

def experiment_with_circuit(num_qubits, depth, use_ibmq=False):
    circuit = random_circuit(num_qubits, depth, max_operands=2, seed=1242)
    
    # Adjust observable strings to match the number of qubits
    observable_strings = ["Z" + "I" * (num_qubits - 1),
                          "I" * (num_qubits // 2) + "Z" + "I" * (num_qubits // 2 - 1),
                          "I" * (num_qubits - 1) + "Z"]
    observable_strings = [obs if len(obs) == num_qubits else "I" * num_qubits for obs in observable_strings]
    observable = SparsePauliOp(observable_strings)

    optimization_settings = OptimizationParameters(seed=111)
    device_constraints = DeviceConstraints(qubits_per_subcircuit=7)

    cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
    print(f'Found solution using {len(metadata["cuts"])} cuts with a sampling overhead of {metadata["sampling_overhead"]}.')
    for cut in metadata["cuts"]:
        print(f"{cut[0]} at circuit instruction index {cut[1]}")

    qc_w_ancilla = cut_wires(cut_circuit)
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)

    partitioned_problem = partition_problem(circuit=qc_w_ancilla, observables=observables_expanded)
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables

    print(f"Sampling overhead: {np.prod([basis.overhead for basis in partitioned_problem.bases])}")

    subexperiments, coefficients = generate_cutting_experiments(circuits=subcircuits, observables=subobservables, num_samples=1000)
    print(f"{len(subexperiments[0]) + len(subexperiments[1])} total subexperiments to run on backend.")
    
    print("Subexperiments:", subexperiments)

    results = {}
    total_time = 0

    basis_gates = ['u1', 'u2', 'u3', 'cx']
    basis_translator = BasisTranslator(SessionEquivalenceLibrary, basis_gates)
    pass_manager = PassManager(basis_translator)

    if use_ibmq:
        # IBM Quantum setup
        service = QiskitRuntimeService(channel="ibm_quantum", token="your_ibm_token_here")
        backend = service.backend("ibm_brisbane")
        
        # Create a session with the ibm_brisbane backend
        session = Session(service=service, backend='ibm_brisbane')
        sampler = Sampler(session=session)
        
        for batch, subcircuits in subexperiments.items():
            transpiled_subcircuits = []
            for subcircuit in subcircuits:
                # Add classical register and measurement
                creg = ClassicalRegister(subcircuit.num_qubits, 'creg')
                subcircuit.add_register(creg)
                for q in range(subcircuit.num_qubits):
                    subcircuit.measure(q, creg[q])
                # Transpile the subcircuit
                subcircuit = pass_manager.run(subcircuit)
                transpiled_subcircuit = transpile(subcircuit, backend, optimization_level=1)
                transpiled_subcircuits.append(transpiled_subcircuit)
            
            subexperiment_start = time.time()
            job = sampler.run(transpiled_subcircuits)
            result = job.result()
            total_time += time.time() - subexperiment_start
            
            for i, counts in enumerate(result.quasi_dists):
                results[f"{batch}_{i}"] = counts
            
            print(f"Batch {batch} took {time.time() - subexperiment_start} seconds to run.")

        print("Running original circuit...")
        creg = ClassicalRegister(circuit.num_qubits, 'creg')
        circuit.add_register(creg)
        for q in range(circuit.num_qubits):
            circuit.measure(q, creg[q])
        circuit = pass_manager.run(circuit)
        transpiled_circuit = transpile(circuit, backend, optimization_level=1)
        
        original_start = time.time()
        job = sampler.run(transpiled_circuit, shots=1000)
        result = job.result()
        original_time = time.time() - original_start
        results["original"] = result.quasi_dists[0]

    else:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()

        for subexperiment in subexperiments.values():
            for i, circuit in enumerate(subexperiment):
                print(f"Running subcircuit {i}...")
                circuit = pass_manager.run(circuit)
                transpiled_circuit = transpile(circuit, backend)
                subexperiment_start = time.time()
                result = backend.run(transpiled_circuit).result()
                total_time += time.time() - subexperiment_start
                counts = result.get_counts()
                results[i] = counts
                print(f"Subcircuit {i} took {time.time() - subexperiment_start} seconds to run.")

        print("Running original circuit...")
        circuit = pass_manager.run(circuit)
        original_start = time.time()
        original_counts = backend.run(circuit).result().get_counts()
        original_time = time.time() - original_start
        results["original"] = original_counts

    print(f"Original circuit took {original_time} seconds to run.")
    print(f"Cuts took {total_time} seconds to run.")

    with open("results.json", "w") as f:
        json.dump(results, f)

    return results, total_time, original_time

# Example usage
# For local simulation:
results, total_time, original_time = experiment_with_circuit(8, 4, use_ibmq=False)

# For IBM Quantum hardware:
# results, total_time, original_time = experiment_with_circuit(8, 4, use_ibmq=True)

# Print results
# print(results, total_time, original_time)
