import numpy as np
import queue

def BFS(A, start_idx, end_idx):  #A = adjacency matrix
    q = queue.deque()
    q.append(start_idx)
    predecessors = np.zeros(A.shape[0], dtype=np.int32) - 1
    predecessors[start_idx] = start_idx
    while (len(q)):
        current_idx = q.popleft()
        for next_idx in np.argwhere(A[current_idx]):  #returns list of lists of single value
            if predecessors[next_idx[0]] == -1:
                q.append(next_idx[0])
                predecessors[next_idx[0]] = current_idx
    if predecessors[end_idx] == -1:
        return None
    
    ind = end_idx
    ans = [ind]
    while ind != start_idx:
        ind = predecessors[ind]
        ans.append(ind)
    
    
    return list(reversed (ans))


def BFSmulti(A, start_idx, end_idx):  #A = adjacency matrix
    q = queue.deque()
    q.append(start_idx)
    predecessors = [ [] for _ in range(A.shape[0]) ]
    distfromstart = np.zeros(A.shape[0], dtype=np.int32) + 99999
    predecessors[start_idx].append(start_idx)
    distfromstart[start_idx] = 0
    while (len(q)):
        current_idx = q.popleft()
        for next_idx in np.argwhere(A[current_idx]):  #returns list of lists of single value
            if len(predecessors[next_idx[0]]) == 0:
                q.append(next_idx[0])
                predecessors[next_idx[0]].append(current_idx)
                distfromstart[next_idx[0]] = distfromstart[current_idx] + 1
            elif (distfromstart[next_idx[0]] == distfromstart[current_idx] + 1):
                predecessors[next_idx[0]].append(current_idx)
                
    if len(predecessors[end_idx]) == 0:
        return None
    
    def give_pathes(ending, ind):
        if ind == start_idx:
            return [ending + [ind]]
        ret_list = []
        for pred_ind in predecessors[ind]:
            ret_list += give_pathes(ending + [ind], pred_ind)
        return ret_list
    
    return [list(reversed (l)) for l in give_pathes([], end_idx)]


