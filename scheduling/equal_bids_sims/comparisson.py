import random
import time
import bisect

def generate_data(M, N, seed=0):
    """
    Generate random problem data for the assignment problem:
      - B[j]: budget of job j
      - L[j]: load of job j
      - K[i]: capacity of worker i
      - S[i]: score of worker i in [0,1]
    """
    random.seed(seed)
    B = [random.uniform(100, 500) for _ in range(N)]   # budgets
    L = [random.randint(5, 100) for _ in range(N)]     # loads
    K = [random.randint(50, 300) for _ in range(M)]    # capacities
    S = [random.random() for _ in range(M)]            # scores
    return B, L, K, S

def compute_objective(assignment, B, L, K, S, beta, alpha):
    """
    Given 'assignment': dict { job j -> worker i or -1 if unassigned },
    compute sum_{j assigned to i} [ beta*B[j] - alpha*(1 - S[i])^2 ].
    """
    total = 0.0
    for j, i in assignment.items():
        if i is not None and i >= 0:
            total += (beta * B[j] - alpha * (1 - S[i])**2)
    return total

# -----------------------------------------------------------------------------
# 1) Naive Greedy: O(M*N)
#    - Sort jobs by descending B_j
#    - For each job, pick the first feasible worker (in ascending ID order)
# -----------------------------------------------------------------------------
def solve_naive(M, N, B, L, K, S, beta, alpha):
    """
    Returns: (assignment_dict, elapsed_time)
      assignment_dict: job j -> worker i (or -1 if unassigned)
    """
    # Sort jobs by descending budget
    jobs_sorted = sorted(range(N), key=lambda j: B[j], reverse=True)

    # Copy capacities so we don't overwrite original
    capacities = K[:]
    assignment = {}

    start = time.time()
    for j in jobs_sorted:
        load_j = L[j]
        # Just pick the first feasible worker in ascending ID
        assigned_worker = None
        for i in range(M):
            if capacities[i] >= load_j:
                assigned_worker = i
                capacities[i] -= load_j
                break
        if assigned_worker is not None:
            assignment[j] = assigned_worker
        else:
            assignment[j] = -1
    end = time.time()

    return assignment, (end - start)

# -----------------------------------------------------------------------------
# 2) “Advanced” Greedy with Capacity-Degrading Score
#    - Sort jobs by descending B_j
#    - Sort workers by ascending capacity
#    - For each job, pick the feasible worker that maximizes an “effective score,”
#      where the effective score is updated as: S[i]*(remaining_cap / original_cap)^gamma
# -----------------------------------------------------------------------------

class SegmentTree:
    """
    Segment tree storing (score, index) pairs.
    Supports:
      - range_max(left, right) -> (best_score, best_idx) in that interval
      - update(idx, new_score)
    """
    def __init__(self, scores):
        self.n = len(scores)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        # We'll store (score, idx) in each node
        self.tree = [(-1.0, -1)]*(2*self.size)

        # Build leaves
        for i in range(self.n):
            self.tree[self.size + i] = (scores[i], i)
        # Build internal nodes
        for p in range(self.size - 1, 0, -1):
            self.tree[p] = self._merge(self.tree[2*p], self.tree[2*p + 1])

    def _merge(self, left_val, right_val):
        # each is (score, idx). Return the one with larger score
        if left_val[0] >= right_val[0]:
            return left_val
        else:
            return right_val

    def update(self, idx, new_score):
        """
        Update the leaf at 'idx' to (new_score, idx), recalc upward.
        """
        pos = self.size + idx
        self.tree[pos] = (new_score, idx)
        pos //= 2
        while pos > 0:
            self.tree[pos] = self._merge(self.tree[2*pos], self.tree[2*pos+1])
            pos //= 2

    def range_max(self, left, right):
        """
        Returns (best_score, best_idx) in the interval [left, right].
        """
        left += self.size
        right += self.size
        best = (-1.0, -1)
        while left <= right:
            if (left % 2) == 1:
                best = self._merge(best, self.tree[left])
                left += 1
            if (right % 2) == 0:
                best = self._merge(best, self.tree[right])
                right -= 1
            left //= 2
            right //= 2
        return best

def solve_capacity_degrading(M, N, B, L, K, S, beta, alpha, gamma=1.0):
    """
    "Advanced" approach with capacity-based score degradation.
    
    Steps:
      1) Sort jobs by descending B_j.
      2) Sort workers by ascending capacity => store (orig_capacity, score, worker_id).
      3) Build a segment tree over 'effective_scores[i]' = S[i]*(capacity_i/orig_capacity_i)^gamma
      4) For each job, do:
         - find the first worker index with capacity >= L[j] (via bisect).
         - do a range_max in [that..M-1] in the segment tree to find best worker
           by "effective score"
         - assign the job to that worker, update that worker's capacity,
           recalc new effective score in segment tree
    """
    # 1) Sort jobs by descending budget
    jobs_sorted = sorted(range(N), key=lambda j: B[j], reverse=True)

    # 2) Sort workers by ascending capacity
    #    store as (orig_cap, S_i, worker_id)
    workers_sorted = sorted([(K[i], S[i], i) for i in range(M)], key=lambda x: x[0])
    # separate them out for convenience
    orig_caps = [ws[0] for ws in workers_sorted]  # sorted ascending
    scores = [ws[1] for ws in workers_sorted]
    worker_ids = [ws[2] for ws in workers_sorted]

    # We'll track current capacities in a separate array
    curr_caps = orig_caps[:]  # copy

    # 3) Build a segment tree over the "effective scores"
    #    effective_score[i] = S[i] * (curr_caps[i]/orig_caps[i])^gamma
    def eff_score(i):
        if orig_caps[i] > 0:
            frac = curr_caps[i] / orig_caps[i]
            if frac < 0:
                frac = 0
            return scores[i] * (frac**gamma)
        else:
            # if orig_caps[i] == 0, or degenerate case
            return -1.0

    effective_scores = [eff_score(i) for i in range(M)]
    segtree = SegmentTree(effective_scores)

    assignment = {}
    start = time.time()
    for j in jobs_sorted:
        load_j = L[j]
        # find first feasible worker index
        idx = bisect.bisect_left(curr_caps, load_j)
        if idx == M:
            # no feasible worker
            assignment[j] = -1
            continue

        # among [idx..M-1], pick best effective score
        (best_score, best_idx) = segtree.range_max(idx, M-1)
        if best_score <= 0:
            # no good feasible worker
            assignment[j] = -1
            continue

        # we have a feasible worker
        w_cap = curr_caps[best_idx]
        w_id  = worker_ids[best_idx]
        if w_cap >= load_j:
            # assign
            assignment[j] = w_id
            # reduce capacity
            new_cap = w_cap - load_j
            curr_caps[best_idx] = new_cap
            # recompute effective score
            new_eff = eff_score(best_idx)
            segtree.update(best_idx, new_eff)
        else:
            assignment[j] = -1

    end = time.time()
    return assignment, (end - start)

# -----------------------------------------------------------------------------
#  Demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Problem size
    
    time_naive=[]
    time_advanced=[]
    ob_naive=[]
    ob_advanced=[]
    
    MM=[10,100,1000,10000]
    for N in MM:
        M =N
    
        # Coefficients
        beta = 1.0
        alpha = 0.5
        gamma = 1.0   # degrade score linearly with fraction of remaining capacity
    
        # Generate data
        B, L, K, S = generate_data(M, N, seed=0)
    
        # 1) Naive
        assign_naive, t_naive = solve_naive(M, N, B, L, K, S, beta, alpha)
        obj_naive = compute_objective(assign_naive, B, L, K, S, beta, alpha)
        print(f"[Naive] Time: {t_naive:.3f}s | Objective Value: {obj_naive:.2f}")
    
        # 2) Capacity-Degrading Advanced
        assign_degrade, t_degrade = solve_capacity_degrading(M, N, B, L, K, S, beta, alpha, gamma=gamma)
        obj_degrade = compute_objective(assign_degrade, B, L, K, S, beta, alpha)
        print(f"[Degrading] Time: {t_degrade:.3f}s | Objective Value: {obj_degrade:.2f}")
        time_naive.append(t_naive)
        time_advanced.append(t_degrade)
        ob_naive.append(obj_naive)
        ob_advanced.append(obj_degrade)
#%%
    import matplotlib.pyplot as plt
    import numpy as np
    x = 10**np.arange(1, 5)
    n_squared = x**2 / x[0]  # Scale O(n^2) to start at same point as empirical
    n_log_n = x * np.log(x) / (x[0] * np.log(x[0]))  # Scale O(n log n)
    
    # First plot: Time complexity
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.loglog(x, time_naive, 'o-', color='C0', label='Naive (Empirical)')
    plt.loglog(x, time_advanced, 'o-', color='C1', label='Advanced (Empirical)')
    plt.loglog(x, 0.0000001*n_squared, '--', color='C0', label='O(n²) Theoretical', alpha=0.7)
    plt.loglog(x, 0.00001*n_log_n, '--', color='C1', label='O(n log n) Theoretical', alpha=0.7)
    
    plt.title('Time Complexity Comparison')
    plt.xlabel('Input Size (n)')
    plt.ylabel('Time (seconds)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.show()

    plt.loglog(10**np.arange(1,5),ob_naive,label='Naive')
    plt.loglog(10**np.arange(1,5),ob_advanced,label='Advanced')
    plt.title('Objective')
    plt.legend()
