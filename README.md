# Group_Execution_Constraints_public
The response time analysis for DAG tasks with group execution constraints.
<br />
## Before starting
The Dirichlet-Rescale (DRS) algorithm[^1] is applied to generate utilizations of task sets randomly
```
pip3 install drs
```
## Generators
Inside the genrators folder, we provide tools to 1) DAG task sets generator with configurations; 2) add communication overheads to existed DAG task sets; 3) modify the existed DAG task sets' gourping configuration and store as a new file.
#### DAG Task generator
The arg vector contains the following parameters:
| Parameter        | Description                                                                                                  |
|------------------|--------------------------------------------------------------------------------------------------------------|
| -m               | The number of sets                                                                                           |
| -p               | The number of processors                                                                                     |
| -q               | The probability of edges between two vertices. 0:[10%, 30%], 1:[40%, 60], 2:[70%, 90%]                               |
| -s               | The number of tasks per set. 0:[0.25 * num_processors, num_processors], 1:[num_processors, 2 * num_processors], 2:[0.25 * num_processors, 2 * num_processors]   |
| -g               | Number of available grous. 0: [0.25 * num_processors, num_processors], 1: [0.25 * num_processors, 2 * num_processors], 2: [num_processors, 2 * num_processors]  |
| -r               | The probability that a vertex is groupped. 0: 90%, 1: 70%, 2: 50%.                                                                                       |
| -o               | The percentage of overheads.                                                                                     |
| -u               | The average utilization per core.                                                                                |

The original task set generator does not have the '-o' option, only overheads related scripts have the '-o' option. 


## Algorithms
Inside the `algorithms` folder, all these algorithms related files are included as follows:
- `Federated.py`: Federated scheduling from Li et al.[^1].
- `graph.py`: The response time analysis for DAG tasks with group execution constraints. 
- `HeECRTS_fixed.py`: Single path approach represented by He et al.[^2].
- `Optimal.py`: The lower bound of the makespan, which is defined as the maximum between the WCET of the task over the available processor and the length of its critical path.
- `UeterRTSS.py`: Parallel Path Progression DAG Scheduling approach proposed by Ueter et al.[^3].

## Experiments
Inside the `experiments` folder, all the scripts to deploy the task set to the specified response time analysis algorithm. 
- `makespan_fed.py`: calculate the makespan by applying federated scheduling.
- `makespan_group.py`: calculate the makespan by applying the new response time analysis for DAG tasks with group execution constraints. 
- `makespan_he.py`: calculate the makespan by applying He's approach.
- `makespan_lb.py`: calculate the lower bound of the makespan.
- `makespan_ueter.py`: calculate the makespan by applying Ueter's approach.

Please note, some scripts have the `_ove` end, that are used for calculating the makespan by considering the overheads.

In addition, each script calculate the makespan fore task sets with utilization levels from 5% to 50% per core (step 5%). However, the He's approach is relatively time consuming, we prepared scripts for calculating makespan for task sets at each utilization level as well, with `-u` argument, e.g., `-u 50` means 50% utilization per core. 

### Please Note: 



## References
[^1]: J. Li, J. Chen, K. Agrawal, C. Lu, C. D. Gill, and A. Saifullah. Analysis of federated and global scheduling
for parallel real-time tasks. In 26th Euromicro Conference on Real-Time Systems, ECRTS 2014, pages 85–96. IEEE Computer Society.
[^2]: Q. He, M. Lv, and N. Guan. Response time bounds for DAG tasks with arbitrary intra-task priority
assignment. In B. B. Brandenburg, editor, 33rd Euromicro Conference on Real-Time Systems, ECRTS 2021.
[^3]: N. Ueter, M. Günzel, G. von der Brüggen, and J. Chen. Parallel path progression DAG scheduling. CoRR,
abs/2208.11830, 2022
